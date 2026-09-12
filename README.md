<div align="center">

# 🧩 Pandas Rule-Based Engine — RAG Pipeline Backend

**A vectorized, industry-grade rules engine feeding a Retrieval-Augmented Generation pipeline — served through Langflow.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Langflow](https://img.shields.io/badge/Langflow-UI%20Layer-8A2BE2)](https://www.langflow.org/)
[![OpenRouter](https://img.shields.io/badge/LLM%20%2F%20Embeddings-OpenRouter-orange)](https://openrouter.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

</div>

---

## 📐 Architecture

<div align="center">
  <img src="docs/architecture.svg" alt="RAG pipeline architecture diagram" width="620">
</div>

> **The UI is strictly Langflow.** This repository is the *engine room* — ingestion, cleaning, rule evaluation, embeddings, and context compression all happen here, behind a FastAPI backend. Langflow owns the retrieval-augmented prompt → LLM flow and the chat surface on top of it.

---

## ✨ Highlights

| | |
|---|---|
| ⚡ **Vectorized rules** | No `iterrows()`. Every rule runs via `DataFrame.query()` or `numpy.select` across the full dataset in one pass. |
| 🛡️ **Validated by design** | Rules are schema-checked, deduplicated by name, and cross-referenced against real DataFrame columns before execution — bad rules fail fast, not silently at row 40,000. |
| 🔀 **Two execution backends** | `query` for flexible, possibly-overlapping conditions. `np_select` for maximum throughput on mutually exclusive rules. |
| 🧠 **Bring your own models** | API key, embedding model, and LLM model are supplied at runtime — nothing is hardcoded. |
| 📄 **Multi-format ingestion** | CSV, Excel, JSON, Markdown, and PDF all normalize into the same tabular shape. |
| 🧵 **Context-aware compression** | Cosine-similarity top-k retrieval packed into a hard character budget before it ever reaches the LLM. |

---

## 🗂️ Project Layout

```
rag_pipeline/
├── app/
│   ├── loaders.py               # Ingestion — multi-format file readers
│   ├── preprocessing.py         # Filtering, exact + near-duplicate dedup, normalization
│   ├── rule_engine.py           # ⭐ Vectorized rule engine (query & np.select backends)
│   ├── openrouter_clients.py    # Embeddings + Chat clients for OpenRouter
│   ├── compression.py           # Top-k retrieval + context-budget compression
│   └── main.py                  # FastAPI app wiring the full pipeline
├── langflow_custom/
│   └── rag_backend_component.py # Langflow node bridging the flow to this backend
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/varunmulay-droid/Pandas-Rule-based-engine.git
cd Pandas-Rule-based-engine
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000` (interactive docs at `/docs`).

### Configuring credentials

Config is resolved with this precedence:

1. `POST /config` — overrides everything, no restart needed.
2. Environment variables — `OPENROUTER_API_KEY`, `EMBEDDING_MODEL`, `LLM_MODEL`.

If neither is set, credential-requiring endpoints return a `400` telling you what's missing.

---

## ☁️ Deploying on Render

This repo includes a `render.yaml` Blueprint at the root.

1. On [Render](https://dashboard.render.com) → **New** → **Blueprint** → select this repo.
2. Render detects `render.yaml` and prompts you to fill in the three environment variables it declares (`sync: false` means they're entered by you at deploy time, never committed to the repo):
   - `OPENROUTER_API_KEY`
   - `EMBEDDING_MODEL` (e.g. `liquid/lfm-2.5-embedding-350m:free`)
   - `LLM_MODEL` (e.g. `nvidia/nemotron-3.5-content-safety:free`)
3. Click **Apply** — Render runs `pip install -r requirements.txt` then `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. Once live, verify with `curl https://<your-service>.onrender.com/health`.

> Free-tier services spin down after ~15 min idle; the in-memory pipeline state (`_STATE`) resets on every restart, so re-run `/ingest` → `/preprocess` → `/apply-rules` → `/embed` after a cold start.

---

## 🔧 Pipeline API

| Step | Endpoint | Purpose |
|---|---|---|
| 0 | `POST /config` | Register your OpenRouter API key + embedding/LLM model names |
| 1 | `POST /ingest` | Upload a file (`.csv` / `.xls(x)` / `.json` / `.md` / `.pdf`) |
| 2 | `POST /preprocess` | Filter, dedup, normalize the ingested data |
| 3 | `POST /apply-rules` | Run the rule engine over the preprocessed data |
| 4 | `POST /embed` | Generate embeddings for downstream retrieval |
| 5 | `POST /query` | Retrieve + compress top-k context for a question |
| 6 | `POST /generate` | End-to-end retrieve → compress → LLM call (for testing outside Langflow) |

<details>
<summary><strong>Example: configuring your models</strong></summary>

```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
        "openrouter_api_key": "sk-or-...",
        "embedding_model": "liquid/lfm-2.5-embedding-350m:free",
        "llm_model": "nvidia/nemotron-3.5-content-safety:free"
      }'
```
</details>

<details>
<summary><strong>Example: defining rules</strong></summary>

```json
{
  "rules": [
    { "name": "example_rule", "condition": "some_column > 10", "action": "Flag" }
  ],
  "strategy": "last_match",
  "backend": "query"
}
```

- `condition` — any valid `DataFrame.query()` expression.
- `strategy` — `first_match` or `last_match` (which rule wins when several match the same row).
- `backend` — `query` (flexible) or `np_select` (fastest, mutually-exclusive rules).

Every rule is validated before it runs, and every match is logged with the row count it affected.
</details>

---

## 🔗 Wiring the Langflow Flow

1. Copy `langflow_custom/rag_backend_component.py` into your Langflow `custom_components/` directory and restart Langflow.
2. Build the flow:

   ```
   Chat Input → RAG Backend Bridge (→ http://localhost:8000/query) → Prompt → OpenRouter LLM → Chat Output
   ```

3. Point the OpenRouter LLM node at the same `llm_model` you registered in `/config`.

Everything upstream of the prompt — ingestion, preprocessing, rules, embeddings, compression — already happened in the FastAPI backend. Langflow's job is purely the final retrieval-augmented generation step and the UI around it.

---

## 🧱 Design Principles

- **Fail loud, not silent.** Malformed rules, unknown columns, and duplicate rule names are caught before execution.
- **No hidden state in models.** API keys and model identifiers travel with the request/config call — never baked into code.
- **Vectorization first.** Anything that touches every row is written to run in one pass over the DataFrame or NumPy array.
- **Separation of concerns.** The backend never renders UI; Langflow never touches raw data cleaning or rule logic.

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built by <a href="https://github.com/varunmulay-droid">@varunmulay-droid</a></sub>
</div>
