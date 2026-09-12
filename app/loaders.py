"""
Ingestion Layer
---------------
Loads .csv, .xls/.xlsx, .json, .md, .pdf into a normalized pandas
DataFrame (tabular sources) or a list[dict] of {"source", "text"}
(document sources like .md/.pdf) that downstream preprocessing can
filter/dedupe uniformly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("ingestion")

TABULAR_EXTENSIONS = {".csv", ".xls", ".xlsx"}
TEXT_EXTENSIONS = {".md"}
PDF_EXTENSIONS = {".pdf"}
JSON_EXTENSIONS = {".json"}


def load_file(path: str | Path) -> pd.DataFrame:
    """
    Returns a DataFrame with at least a 'text' column for unstructured
    sources (md/pdf) and native columns for tabular sources (csv/xls/json
    records). A 'source_file' column is always attached for traceability.
    """
    path = Path(path)
    ext = path.suffix.lower()
    logger.info("Loading file: %s (type=%s)", path.name, ext)

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    elif ext in JSON_EXTENSIONS:
        raw = json.loads(path.read_text(encoding="utf-8"))
        df = pd.json_normalize(raw) if isinstance(raw, list) else pd.json_normalize([raw])
    elif ext in TEXT_EXTENSIONS:
        df = pd.DataFrame([{"text": path.read_text(encoding="utf-8")}])
    elif ext in PDF_EXTENSIONS:
        df = pd.DataFrame([{"text": t} for t in _extract_pdf_text(path)])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df["source_file"] = path.name
    return df


def _extract_pdf_text(path: Path) -> list[str]:
    """One list entry per page. Requires `pypdf` (see requirements.txt)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF ingestion: pip install pypdf") from exc

    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def load_many(paths: list[str | Path]) -> pd.DataFrame:
    frames = [load_file(p) for p in paths]
    return pd.concat(frames, ignore_index=True, sort=False)
