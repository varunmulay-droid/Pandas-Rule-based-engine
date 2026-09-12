"""
FastAPI backend for the pipeline:

  Ingestion -> Preprocessing -> Rule Engine -> Embeddings -> Compression -> (Langflow: Prompt -> LLM)

The RAG "Prompt -> LLM" step itself is built and served via Langflow's
own UI (per the "strictly Langflow UI" requirement) — this FastAPI app
is the engine/backend that Langflow's flow calls into via HTTP, using
the custom component in `langflow_custom/rag_backend_component.py`.

API keys and model names are never hardcoded. Precedence, highest first:
  1. Whatever the user supplies via POST /config (overrides everything, no server restart needed)
  2. Environment variables (OPENROUTER_API_KEY, EMBEDDING_MODEL, LLM_MODEL) —
     e.g. set as Render "envVars" with sync: false, which makes Render's
     dashboard prompt YOU (the deployer) to type them in as input at deploy
     time, rather than committing them to the repo.
If neither is set, calls that need credentials return a clear 400 asking
the caller to hit /config first.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .compression import compress_to_budget, select_top_k
from .loaders import load_file
from .openrouter_clients import OpenRouterChatClient, OpenRouterEmbeddingsClient
from .preprocessing import preprocess_pipeline
from .rule_engine import EngineBackend, MatchStrategy, RuleEngine, RuleEngineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="RAG Pipeline Backend", version="1.0.0")

# In-memory store for the demo/dev flow. Swap for a real DB/vector store in production.
# `config` seeds from environment variables (if present) so a Render deploy
# works immediately without a manual /config call; /config always overrides.
_STATE: dict[str, Any] = {
    "records": [],
    "embeddings": [],
    "config": {
        k: v
        for k, v in {
            "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY"),
            "embedding_model": os.environ.get("EMBEDDING_MODEL"),
            "llm_model": os.environ.get("LLM_MODEL"),
        }.items()
        if v
    },
}


def _get_config() -> dict[str, str]:
    cfg = _STATE.get("config") or {}
    missing = [k for k in ("openrouter_api_key", "embedding_model", "llm_model") if not cfg.get(k)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing config: {missing}. Call POST /config with your OpenRouter API key "
                "and model names, or set OPENROUTER_API_KEY / EMBEDDING_MODEL / LLM_MODEL "
                "as environment variables on the server."
            ),
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
@app.post("/config")
def set_config(cfg: ModelConfig):
    """User supplies API key + model names once; reused by later calls unless overridden."""
    _STATE["config"] = cfg.model_dump()
    return {"status": "ok"}


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
