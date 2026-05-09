"""Adapter that runs the benchmark against the local PolyMD service layer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_services import build_system, process_query  # noqa: E402


_SYSTEM: Dict[str, Any] = {}


def _system() -> Dict[str, Any]:
    """Build and cache the PolyMD system for benchmark module mode."""
    global _SYSTEM
    if not _SYSTEM:
        _SYSTEM = build_system(cache_buster="benchmark")
    return _SYSTEM


def _answer_text(response: Any) -> str:
    """Extract display answer text from a PolyMD response payload."""
    if isinstance(response, dict):
        return str(response.get("answer", ""))
    return str(response)


def answer_question(question: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Answer one benchmark question through the same service layer as the app."""
    response, stats = process_query(
        query=question,
        system=_system(),
        conversation_history=history,
    )
    answer = _answer_text(response)
    doi_links = []
    if isinstance(response, dict):
        doi_links = response.get("doi_links", []) or []

    return {
        "answer": answer,
        "retrieved_chunk_ids": stats.get("retrieved_chunk_ids", []),
        "retrieved_dois": stats.get("retrieved_dois", []) or doi_links,
        "abstained": bool(response.get("abstained", False))
        if isinstance(response, dict)
        else False,
        "history_content": response,
    }
