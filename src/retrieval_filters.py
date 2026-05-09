"""
Pure dict transforms on retrieval result payloads (testable without Chroma).
"""
from __future__ import annotations

from typing import Dict, List


def drop_property_chunks(results: Dict) -> Dict:
    """Remove property chunks; keep paper or other types."""
    metas: List = results.get("metadatas", []) or []
    keep_idx = [i for i, m in enumerate(metas) if m.get("chunk_type") != "property"]
    if not keep_idx:
        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "scores": [],
        }
    out = dict(results)
    for key in ("documents", "metadatas", "ids", "scores"):
        if key in out and isinstance(out[key], list):
            out[key] = [out[key][i] for i in keep_idx if i < len(out[key])]
    return out
