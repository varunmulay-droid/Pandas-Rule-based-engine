# RAG Pipeline Backend (FastAPI) + Langflow UI

Implements the pipeline:

```
Ingestion -> Preprocessing (filter+dedup) -> Preprocessed File
   -> Pandas Rule-Based Engine -> Embeddings (OpenRouter) -> Context Compression
   -> RAG via Langflow (Prompt -> LLM, OpenRouter)
```

The **UI is strictly Langflow** — this repo is the backend engine that a
Langflow flow calls into. Nothing here builds a competing chat UI.

## Layout

```
app/
  loaders.py             # ingestion: .csv .xls .xlsx .json .md .pdf -> DataFrame
  preprocessing.py        # filtering + dedup (exact + near-dup) + normalization
  rule_engine.py           # vectorized rule engine (df.query() and np.select backends)
  openrouter_clients.py    # embeddings + chat clients (API key/model supplied by caller)
  compression.py           # cosine-similarity top-k retrieval + char-budget compression
  main.py                  # FastAPI app wiring it all together
langflow_custom/
  rag_backend_component.py # Langflow custom component that calls /query
requirements.txt
```

## Running the backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Flow

1. `POST /config` — supply your OpenRouter API key, embedding model
   name (e.g. `liquid/lfm-2.5-embedding-350m:free`), and LLM model name
   (e.g. `nvidia/nemotron-3.5-content-safety:free`). Nothing is
   hardcoded — you choose the models at runtime.
2. `POST /ingest` — upload a `.csv`, `.xls(x)`, `.json`, `.md`, or `.pdf` file.
3. `POST /preprocess` — filter rows, drop-dedup, normalize text columns.
4. `POST /apply-rules` — pass your rule list (see `Rule` schema below);
   choose `backend: "query"` (flexible, per-rule `.query()`) or
   `backend: "np_select"` (fastest, single-pass, best when rules are
   mutually exclusive).
5. `POST /embed` — embeds the preprocessed text via the OpenRouter
   embeddings model.
6. `POST /query` — given a natural-language query, retrieves and
   compresses the top-k most relevant chunks. **This is the endpoint
   the Langflow custom component calls.**
7. (optional) `POST /generate` — end-to-end retrieve + compress + call
   the base LLM, for testing outside Langflow.

### Rule schema

```json
{
  "rules": [
    {"name": "example_rule", "condition": "some_column > 10", "action": "Flag"}
  ],
  "strategy": "last_match",
  "backend": "query"
}
```

`condition` is any valid `DataFrame.query()` expression. Rules are
validated (schema + best-effort column-reference check) before
execution, and every match is logged with the row count it affected.

## Wiring the Langflow flow

1. Copy `langflow_custom/rag_backend_component.py` into your Langflow
   `custom_components` directory and restart Langflow.
2. In the Langflow UI, build:

   `Chat Input -> RAG Backend Bridge (backend_url = http://localhost:8000/query) -> Prompt -> OpenRouter LLM node -> Chat Output`

3. Set the OpenRouter LLM node's model to whatever `llm_model` you
   configured in step 1 above (e.g. `nvidia/nemotron-3.5-content-safety:free`).

All ingestion/preprocessing/rules/embeddings/compression happen in the
FastAPI backend ahead of time (steps 1-6); Langflow only owns the final
retrieval-augmented prompt -> LLM call and the chat UI around it.
