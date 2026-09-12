"""
Preprocessing Layer
--------------------
Data filtering + deduplication + misc normalization, applied after
ingestion and before the rule engine / embedding steps.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger("preprocessing")


def filter_rows(df: pd.DataFrame, query_expr: str | None = None, drop_na_columns: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if drop_na_columns:
        before = len(out)
        out = out.dropna(subset=[c for c in drop_na_columns if c in out.columns])
        logger.info("Dropped %d rows with NA in %s", before - len(out), drop_na_columns)
    if query_expr:
        before = len(out)
        out = out.query(query_expr, engine="python")
        logger.info("Filter '%s' kept %d/%d rows", query_expr, len(out), before)
    return out.reset_index(drop=True)


def deduplicate(df: pd.DataFrame, subset: list[str] | None = None, text_column: str | None = None,
                 near_dup_threshold: float | None = None) -> pd.DataFrame:
    """
    Exact dedup via subset columns. If `text_column` + `near_dup_threshold`
    are given, also drops near-duplicate text rows using a cheap Jaccard
    shingle similarity (no external deps) — good enough as a pre-filter
    before the (expensive) embeddings step.
    """
    before = len(df)
    out = df.drop_duplicates(subset=subset).reset_index(drop=True)
    logger.info("Exact dedup: %d -> %d rows", before, len(out))

    if text_column and near_dup_threshold and text_column in out.columns:
        keep_mask = _near_dup_filter(out[text_column].fillna("").tolist(), near_dup_threshold)
        before_nd = len(out)
        out = out.loc[keep_mask].reset_index(drop=True)
        logger.info("Near-dup dedup: %d -> %d rows", before_nd, len(out))

    return out


def _shingles(text: str, k: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    return {" ".join(tokens[i:i + k]) for i in range(max(0, len(tokens) - k + 1))} or {text.lower()}


def _near_dup_filter(texts: list[str], threshold: float) -> list[bool]:
    keep = [True] * len(texts)
    shingle_sets = [_shingles(t) for t in texts]
    for i in range(len(texts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(texts)):
            if not keep[j]:
                continue
            a, b = shingle_sets[i], shingle_sets[j]
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard >= threshold:
                keep[j] = False
    return keep


def normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Misc operations: strip whitespace, collapse internal whitespace, lowercase-safe."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
    return out


def preprocess_pipeline(
    df: pd.DataFrame,
    filter_query: str | None = None,
    drop_na_columns: list[str] | None = None,
    dedup_subset: list[str] | None = None,
    text_column: str | None = None,
    near_dup_threshold: float | None = None,
    normalize_columns: list[str] | None = None,
) -> pd.DataFrame:
    out = filter_rows(df, filter_query, drop_na_columns)
    out = deduplicate(out, dedup_subset, text_column, near_dup_threshold)
    if normalize_columns:
        out = normalize_text_columns(out, normalize_columns)
    return out
