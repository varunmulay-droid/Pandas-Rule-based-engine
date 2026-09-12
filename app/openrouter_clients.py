"""
OpenRouter Clients
------------------
Thin wrappers around OpenRouter's REST API for:
  1. Embeddings (Tokenization step in the pipeline)
  2. Base LLM chat (Prompt -> LLM step, called from the Langflow flow)

The API key and model name are ALWAYS supplied by the caller (per
request or via app.state at startup) — never hardcoded, per the
"take API keys and model names via user" requirement.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger("openrouter_clients")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    pass


class OpenRouterEmbeddingsClient:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        if not model:
            raise ValueError("Embeddings model name is required.")
        self.api_key = api_key
        self.model = model

    async def embed(self, texts: list[str], encoding_format: str = "float") -> list[list[float]]:
        url = f"{OPENROUTER_BASE_URL}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60) as client:
            # OpenRouter embeddings endpoint accepts a single input per call
            # for the widest model compatibility; batch sequentially.
            for text in texts:
                payload = {"model": self.model, "input": text, "encoding_format": encoding_format}
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.error("Embeddings call failed: %s", resp.text)
                    raise OpenRouterError(f"Embeddings request failed ({resp.status_code}): {resp.text}")
                data = resp.json()
                vectors.append(data["data"][0]["embedding"])
        return vectors


class OpenRouterChatClient:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        if not model:
            raise ValueError("LLM model name is required.")
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: list[dict[str, str]], stream: bool = False) -> Any:
        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "stream": stream}

        if not stream:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.error("Chat call failed: %s", resp.text)
                    raise OpenRouterError(f"Chat request failed ({resp.status_code}): {resp.text}")
                return resp.json()["choices"][0]["message"]["content"]

        return self._stream(url, headers, payload)

    async def _stream(self, url: str, headers: dict, payload: dict) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise OpenRouterError(f"Chat stream failed ({resp.status_code}): {body}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    yield data
