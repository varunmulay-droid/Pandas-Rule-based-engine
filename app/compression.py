"""
Context Compression Layer
--------------------------
Given a query embedding and a bank of chunk embeddings, selects and
compresses the most relevant chunks so the RAG prompt sent to the LLM
stays within a token budget.

Kept dependency-light (numpy only) since this sits in the hot path.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("compression")


def cosine_similarity(query_vec: list[float], chunk_vecs: list[list[float]]) -> np.ndarray:
    q = np.asarray(query_vec, dtype=float)
    m = np.asarray(chunk_vecs, dtype=float)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    m_norm = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)
    return m_norm @ q_norm


def select_top_k(chunks: list[str], chunk_vecs: list[list[float]], query_vec: list[float],
                  k: int = 5, min_score: float = 0.0) -> list[tuple[str, float]]:
    scores = cosine_similarity(query_vec, chunk_vecs)
    order = np.argsort(-scores)[:k]
    results = [(chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]
    logger.info("Selected %d/%d chunks above min_score=%s", len(results), len(chunks), min_score)
    return results


def compress_to_budget(selected: list[tuple[str, float]], max_chars: int) -> str:
    """Greedily packs highest-scoring chunks into the character budget."""
    out_parts: list[str] = []
    used = 0
    for text, _score in selected:
        remaining = max_chars - used
        if remaining <= 0:
            break
        snippet = text if len(text) <= remaining else text[:remaining]
        out_parts.append(snippet)
        used += len(snippet)
    compressed = "\n\n---\n\n".join(out_parts)
    logger.info("Compressed context to %d chars (budget %d)", len(compressed), max_chars)
    return compressed
