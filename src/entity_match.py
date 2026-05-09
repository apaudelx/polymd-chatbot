"""
Pure string helpers for entity mention checks (no heavy retrieval deps).
"""
from __future__ import annotations

import re
from typing import List


def tokenize_lower(text: str) -> List[str]:
    """Split text on whitespace and lowercase each token."""
    return text.lower().split()


def entity_match_in_text(text: str, entity: str) -> bool:
    """
    Match an entity against text using word-boundary checks on alphanumeric tokens.
    Avoids substring false positives (e.g. 'pet' inside unrelated words).
    """
    if not text or not entity:
        return False
    text_l = text.lower()
    ent = entity.strip().lower()
    if not ent:
        return False
    tokens = [t for t in re.findall(r"[a-z0-9]+", ent) if t]
    if not tokens:
        return False
    for tok in tokens:
        if not re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", text_l):
            return False
    return True
