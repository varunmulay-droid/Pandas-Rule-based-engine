"""
FastAPI backend for the pipeline:

  Ingestion -> Preprocessing -> Rule Engine -> Embeddings -> Compression -> (Langflow: Prompt -> LLM)

The RAG "Prompt -> LLM" step itself is built and served via Langflow's
own UI (per the "strictly Langflow UI" requirement) — this FastAPI app
is the engine/backend that Langflow's flow calls into via HTTP, using
the custom component in `langflow_custom/rag_backend_component.py`.

API keys and model names are never hardcoded and never sit in Render's
dashboard or repo. The deployed app itself serves a small HTML form at
"/" where you type your OpenRouter API key + model names once the
service is live; that form POSTs to /config and the key is held only
in this process's memory for the life of the running instance — it is
never logged, written to disk, or included in any response.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .compression import compress_to_budget, select_top_k
from .loaders import load_file
from .openrouter_clients import OpenRouterChatClient, OpenRouterEmbeddingsClient
from .preprocessing import preprocess_pipeline
from .rule_engine import EngineBackend, MatchStrategy, RuleEngine, RuleEngineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="RAG Pipeline Backend", version="1.0.0")

# In-memory only — reset on every restart/redeploy. Swap for a real
# secrets manager + DB/vector store for anything beyond a demo.
_STATE: dict[str, Any] = {"records": [], "embeddings": [], "config": {}}


def _get_config() -> dict[str, str]:
    cfg = _STATE.get("config") or {}
    missing = [k for k in ("openrouter_api_key", "embedding_model", "llm_model") if not cfg.get(k)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing config: {missing}. Open the app's root URL in a browser and submit your API key/models first.",
        )
    return cfg


# ---------------------------------------------------------------- models --
class ModelConfig(BaseModel):
    openrouter_api_key: str = Field(..., description="User-supplied OpenRouter API key")
    embedding_model: str = Field(..., description="e.g. liquid/lfm-2.5-embedding-350m:free")
    llm_model: str = Field(..., description="e.g. nvidia/nemotron-3.5-content-safety:free")


class PreprocessRequest(BaseModel):
    filter_query: str | None = None
    drop_na_columns: list[str] | None = None
    dedup_subset: list[str] | None = None
    text_column: str | None = "text"
    near_dup_threshold: float | None = None
    normalize_columns: list[str] | None = None


class Rule(BaseModel):
    name: str
    condition: str
    action: str
    priority: int = 0


class RuleEngineRequest(BaseModel):
    rules: list[Rule]
    strategy: MatchStrategy = MatchStrategy.LAST_MATCH
    backend: EngineBackend = EngineBackend.QUERY
    default_result: str = "Unknown"


class EmbedRequest(BaseModel):
    text_column: str = "text"


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    max_context_chars: int = 4000
    min_score: float = 0.0


# ------------------------------------------------------------- endpoints --
CONFIG_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAG Pipeline — Configuration</title>
<style>
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 480px; margin: 60px auto; color: #1a1d23; }
  h1 { font-size: 20px; }
  label { display: block; margin-top: 16px; font-size: 13px; font-weight: 600; color: #4a505b; }
  input { width: 100%; padding: 10px; margin-top: 6px; box-sizing: border-box; border: 1px solid #d5d8dd; border-radius: 8px; font-size: 14px; }
  button { margin-top: 22px; padding: 10px 18px; border: none; border-radius: 8px; background: #2f6fe0; color: white; font-size: 14px; cursor: pointer; }
  button:hover { background: #2559b8; }
  #status { margin-top: 16px; font-size: 13px; }
  .note { font-size: 12px; color: #767c87; margin-top: 24px; line-height: 1.5; }
</style>
</head>
<body>
  <h1>RAG Pipeline Backend — Configuration</h1>
  <p style="font-size:13px;color:#4a505b;">Enter your OpenRouter API key and model names. This is sent directly to this
  server's in-memory config and is never written to disk, logged, or committed anywhere.</p>

  <form id="cfg-form">
    <label>OpenRouter API Key</label>
    <input type="password" id="openrouter_api_key" required autocomplete="off">

    <label>Embedding Model</label>
    <input type="text" id="embedding_model" placeholder="liquid/lfm-2.5-embedding-350m:free" required>

    <label>LLM Model</label>
    <input type="text" id="llm_model" placeholder="nvidia/nemotron-3.5-content-safety:free" required>

    <button type="submit">Save configuration</button>
  </form>
  <div id="status"></div>

  <p class="note">
    This config lives in server memory only for the lifetime of this running instance —
    it resets on every restart/redeploy, and no other user of this deployment can read it back out.
    Use <code>/docs</code> for the full API once configured.
  </p>

<script>
document.getElementById('cfg-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = {
    openrouter_api_key: document.getElementById('openrouter_api_key').value,
    embedding_model: document.getElementById('embedding_model').value,
    llm_model: document.getElementById('llm_model').value,
  };
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Saving...';
  try {
    const resp = await fetch('/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (resp.ok) {
      statusEl.style.color = '#159c81';
      statusEl.textContent = 'Saved. You can now use /ingest, /preprocess, /apply-rules, /embed, /query via /docs.';
    } else {
      const err = await resp.json();
      statusEl.style.color = '#d8306a';
      statusEl.textContent = 'Error: ' + JSON.stringify(err.detail);
    }
  } catch (err) {
    statusEl.style.color = '#d8306a';
    statusEl.textContent = 'Request failed: ' + err;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def config_page():
    """Serves the runtime configuration form — this is how the API key is supplied
    once the app is deployed, instead of via Render env vars or the repo."""
    return CONFIG_PAGE


@app.post("/config")
def set_config(cfg: ModelConfig):
    """Called by the form on '/' (or directly). Held in memory only for this process's lifetime."""
    _STATE["config"] = cfg.model_dump()
    return {"status": "ok"}


@app.get("/config/status")
def config_status():
    """Reports whether config is set, without ever revealing the key itself."""
    cfg = _STATE.get("config") or {}
    return {
        "configured": bool(cfg.get("openrouter_api_key")),
        "embedding_model": cfg.get("embedding_model"),
        "llm_model": cfg.get("llm_model"),
    }


@app.post("/ingest")
async def ingest(file: UploadFile):
    """Ingestion layer — accepts .csv/.xls/.xlsx/.json/.md/.pdf."""
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df = load_file(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    _STATE["records"] = df.to_dict(orient="records")
    return {"rows_ingested": len(df), "columns": list(df.columns)}


@app.post("/preprocess")
def preprocess(req: PreprocessRequest):
    """Preprocessing layer — filtering + dedup + misc normalization."""
    import pandas as pd

    if not _STATE["records"]:
        raise HTTPException(status_code=400, detail="No ingested data. Call /ingest first.")

    df = pd.DataFrame(_STATE["records"])
    out = preprocess_pipeline(
        df,
        filter_query=req.filter_query,
        drop_na_columns=req.drop_na_columns,
        dedup_subset=req.dedup_subset,
        text_column=req.text_column,
        near_dup_threshold=req.near_dup_threshold,
        normalize_columns=req.normalize_columns,
    )
    _STATE["records"] = out.to_dict(orient="records")
    return {"rows_after_preprocessing": len(out)}


@app.post("/apply-rules")
def apply_rules(req: RuleEngineRequest):
    """Pandas rule-based engine (vectorized query / np.select)."""
    import pandas as pd

    if not _STATE["records"]:
        raise HTTPException(status_code=400, detail="No preprocessed data. Call /ingest and /preprocess first.")

    df = pd.DataFrame(_STATE["records"])
    config = RuleEngineConfig(strategy=req.strategy, backend=req.backend, default_result=req.default_result)
    try:
        engine = RuleEngine([r.model_dump() for r in req.rules], config)
        out = engine.run(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _STATE["records"] = out.to_dict(orient="records")
    return {"rows": len(out), "result_counts": out["result"].value_counts().to_dict()}


@app.post("/embed")
async def embed(req: EmbedRequest):
    """Embeddings model step (Tokenization) via OpenRouter."""
    cfg = _get_config()
    if not _STATE["records"]:
        raise HTTPException(status_code=400, detail="No data to embed. Call /ingest first.")

    texts = [str(r.get(req.text_column, "")) for r in _STATE["records"]]
    client = OpenRouterEmbeddingsClient(cfg["openrouter_api_key"], cfg["embedding_model"])
    vectors = await client.embed(texts)
    _STATE["embeddings"] = vectors
    return {"embedded_chunks": len(vectors), "dimension": len(vectors[0]) if vectors else 0}


@app.post("/query")
async def query(req: QueryRequest):
    """
    Context compression + retrieval step. Returns the compressed context
    string that the Langflow flow's Prompt -> LLM component should use
    (Langflow calls this endpoint, then separately calls the LLM node).
    """
    cfg = _get_config()
    if not _STATE["embeddings"]:
        raise HTTPException(status_code=400, detail="No embeddings available. Call /embed first.")

    client = OpenRouterEmbeddingsClient(cfg["openrouter_api_key"], cfg["embedding_model"])
    query_vec = (await client.embed([req.query]))[0]

    texts = [str(r.get("text", "")) for r in _STATE["records"]]
    selected = select_top_k(texts, _STATE["embeddings"], query_vec, k=req.top_k, min_score=req.min_score)
    context = compress_to_budget(selected, req.max_context_chars)

    return {"context": context, "matched_chunks": len(selected), "scores": [s for _, s in selected]}


@app.post("/generate")
async def generate(req: QueryRequest):
    """
    Convenience end-to-end call (retrieve + compress + call base LLM).
    Langflow can call this directly, or call /query + its own LLM node
    if the flow needs to route the prompt through Langflow-native nodes.
    """
    retrieval = await query(req)
    cfg = _get_config()
    chat_client = OpenRouterChatClient(cfg["openrouter_api_key"], cfg["llm_model"])
    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{retrieval['context']}\n\nQuestion: {req.query}"
    )
    answer = await chat_client.chat([{"role": "user", "content": prompt}], stream=False)
    return {"answer": answer, "context_used": retrieval["context"]}


@app.get("/health")
def health():
    return {"status": "ok"}
