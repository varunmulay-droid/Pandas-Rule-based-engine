"""
Langflow custom component: RAG Backend Bridge
-----------------------------------------------
Drop this file into your Langflow `custom_components` directory. It
gives you a node with:

    Inputs:  query (str), backend_url (str), top_k (int), max_context_chars (int)
    Output:  context (Message) — the compressed retrieved context

Wire it as: [Chat Input] -> [RAG Backend Bridge] -> [Prompt] -> [LLM (OpenRouter)] -> [Chat Output]

This keeps the actual RAG orchestration UI/flow entirely inside Langflow
(per the "strictly Langflow UI" requirement), while ingestion,
preprocessing, the rule engine, embeddings, and compression all run in
the FastAPI backend this component calls.
"""

import httpx
from langflow.custom import Component
from langflow.io import IntInput, MessageTextInput, Output
from langflow.schema.message import Message


class RAGBackendBridge(Component):
    display_name = "RAG Backend Bridge"
    description = "Calls the FastAPI RAG backend's /query endpoint for compressed context retrieval."
    icon = "database"

    inputs = [
        MessageTextInput(name="query", display_name="Query", required=True),
        MessageTextInput(
            name="backend_url",
            display_name="Backend URL",
            value="http://localhost:8000/query",
            required=True,
        ),
        IntInput(name="top_k", display_name="Top K", value=5),
        IntInput(name="max_context_chars", display_name="Max Context Chars", value=4000),
    ]

    outputs = [Output(display_name="Context", name="context", method="fetch_context")]

    def fetch_context(self) -> Message:
        payload = {
            "query": self.query,
            "top_k": self.top_k,
            "max_context_chars": self.max_context_chars,
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(self.backend_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return Message(text=data["context"])
