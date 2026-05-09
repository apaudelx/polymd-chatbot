"""
Offline lexical index snapshot for BM25-backed retrieval.

Written during `scripts/build_vectordb.py` after indexing; loaded by `Retriever`
to avoid a full-collection Chroma scan on every cold start when the file matches
the current collection size and name.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import Config


INDEX_FORMAT = 1


def _serialize_metadatas(metadatas: List[Dict]) -> List[Dict]:
    """Convert Chroma metadata values into JSON-safe primitive values."""
    out = []
    for m in metadatas:
        if not m:
            out.append({})
            continue
        clean: Dict[str, Any] = {}
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        out.append(clean)
    return out


def write_lexical_index_from_store(vector_store, path: Optional[Path] = None) -> Path:
    """Dump documents, ids, and metadatas from Chroma into a gzipped JSON file."""
    path = path or Config.LEXICAL_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Chroma 0.4.x: ids are always returned; "ids" is not a valid include item.
    batch = vector_store.collection.get(include=["documents", "metadatas"])
    docs = batch.get("documents") or []
    ids = batch.get("ids") or []
    metas = batch.get("metadatas") or []

    payload = {
        "format": INDEX_FORMAT,
        "collection_name": Config.COLLECTION_NAME,
        "n_docs": len(docs),
        "ids": ids,
        "documents": docs,
        "metadatas": _serialize_metadatas(metas),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)
    return path


def read_lexical_index(
    path: Path,
    expected_count: int,
    collection_name: str,
) -> Optional[Tuple[List[str], List[str], List[Dict]]]:
    """
    Load lexical snapshot if it matches the live collection identity and size.

    Returns (documents, ids, metadatas) or None to trigger in-memory rebuild from Chroma.
    """
    if not path.exists() or expected_count <= 0:
        return None

    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("format") != INDEX_FORMAT:
        return None
    if payload.get("collection_name") != collection_name:
        return None
    declared = int(payload.get("n_docs", -1))
    if declared != expected_count:
        return None

    docs = payload.get("documents") or []
    ids = payload.get("ids") or []
    metas = payload.get("metadatas") or []
    if len(docs) != len(ids) or len(docs) != len(metas):
        return None
    if len(docs) != declared:
        return None

    return docs, ids, metas
