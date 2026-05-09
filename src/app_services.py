"""Application service layer for chat processing and system construction."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.config import Config
from src.controller import ToolController
from src.embeddings import EmbeddingGenerator
from src.generator import Generator
from src.retriever import Retriever
from src.vector_store import VectorStore


PROPERTY_TERMS = {
    "density": "density",
    "glass transition": "glass transition temperature",
    "tg": "glass transition temperature",
    "young": "Young's modulus",
    "modulus": "Young's modulus",
    "diffusion": "diffusion coefficient",
    "radius of gyration": "radius of gyration",
    "viscosity": "viscosity",
    "thermal conductivity": "thermal conductivity",
    "property": "property",
}

PROPERTY_KEY_BY_PHRASE = {
    "density": "density",
    "glass transition temperature": "glass_transition_temp",
    "Young's modulus": "youngs_modulus",
    "diffusion coefficient": "diffusion_coefficient",
    "radius of gyration": "radius_gyration",
    "viscosity": "viscosity",
    "thermal conductivity": "thermal_conductivity",
}

CONTEXT_POINTERS = (
    "that",
    "this",
    "it",
    "they",
    "those",
    "these",
    "previous",
    "above",
    "same",
    "that one",
    "the first",
    "the second",
    "the value",
    "that value",
    "the property",
    "that property",
    "the result",
    "that result",
)

FOLLOW_UP_TOPICS = (
    "force field",
    "source",
    "doi",
    "citation",
    "paper",
    "reference",
    "condition",
    "temperature",
    "pressure",
    "method",
    "calculated",
    "computed",
    "obtained",
    "measured",
    "value",
    "property",
    "unit",
    "context",
)

GLOBAL_DATASET_TERMS = (
    "this dataset",
    "the dataset",
    "whole dataset",
    "entire dataset",
    "all dataset",
    "this database",
    "the database",
    "whole database",
    "entire database",
    "this corpus",
    "the corpus",
    "in here",
    "available in the app",
)

GLOBAL_ACTION_TERMS = (
    "what",
    "which",
    "how many",
    "count",
    "number",
    "total",
    "list",
    "show",
    "summarize",
    "study",
    "studies",
    "studied",
    "has",
    "have",
    "available",
    "included",
)

PAPER_COUNT_TERMS = (
    "paper",
    "papers",
    "doi",
    "dois",
    "article",
    "articles",
    "publication",
    "publications",
    "study",
    "studies",
)

COUNT_REQUEST_TERMS = (
    "how many",
    "count",
    "number",
    "total",
    "how much",
)


def build_system(cache_buster: str = "") -> Dict[str, Any]:
    """Build the RAG and tool components used by the application."""
    _ = cache_buster
    Config.validate()

    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore()
    retriever = Retriever(vector_store, embedding_gen)
    generator = Generator()
    controller = ToolController()

    count = vector_store.get_count()
    if count == 0:
        raise RuntimeError(
            "Vector database is empty. Run `python scripts/build_vectordb.py` first."
        )

    return {
        "retriever": retriever,
        "generator": generator,
        "controller": controller,
        "embedding_gen": embedding_gen,
        "vector_store": vector_store,
    }


def _retrieval_meta(results: Dict[str, Any]) -> Dict[str, Any]:
    """Return retrieval metadata passed into the generation prompt."""
    return {
        "confidence": results.get("confidence", 0.0),
        "abstained": results.get("abstained", False),
        "abstention_reason": results.get("abstention_reason", ""),
        "intent": results.get("intent"),
        "comparison_requires_property_alignment": results.get(
            "comparison_requires_property_alignment"
        ),
        "comparison_property_aligned": results.get("comparison_property_aligned"),
        "comparison_disclaimer": results.get("comparison_disclaimer"),
        "comparison_shared_properties": results.get("comparison_shared_properties"),
        "comparison_missing": results.get("comparison_missing"),
        "comparison_covered": results.get("comparison_covered"),
    }


def _last_user_query(conversation_history: List[Dict[str, Any]]) -> str:
    """Return the most recent prior user query from chat history."""
    for message in reversed(conversation_history or []):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _last_assistant_payload(
    conversation_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the most recent prior structured assistant payload."""
    for message in reversed(conversation_history or []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, dict):
            return content
    return {}


def _infer_property_phrase(text: str) -> str:
    """Infer a human-readable property phrase from text."""
    lowered = (text or "").lower()
    for trigger, phrase in PROPERTY_TERMS.items():
        if trigger in lowered:
            return phrase
    return ""


def _is_global_dataset_query(query: str) -> bool:
    """Return whether a query asks about the full dataset/corpus."""
    lowered = (query or "").lower()
    has_scope = any(term in lowered for term in GLOBAL_DATASET_TERMS)
    has_action = any(term in lowered for term in GLOBAL_ACTION_TERMS)
    return has_scope and has_action


def _is_global_force_field_query(query: str) -> bool:
    """Return whether a query asks for force fields across the full dataset."""
    lowered = (query or "").lower()
    return _is_global_dataset_query(query) and "force field" in lowered


def _is_global_paper_count_query(query: str) -> bool:
    """Return whether a query asks for the corpus-level paper count."""
    lowered = (query or "").lower()
    has_paper_target = any(term in lowered for term in PAPER_COUNT_TERMS)
    has_count_request = any(term in lowered for term in COUNT_REQUEST_TERMS)
    return _is_global_dataset_query(query) and has_paper_target and has_count_request


def _property_key_from_phrase(phrase: str) -> str:
    """Convert a human-readable property phrase to a metadata key."""
    return PROPERTY_KEY_BY_PHRASE.get(phrase, phrase)


def _extract_entity_from_property_query(text: str) -> str:
    """Extract a likely entity from a previous property lookup query."""
    if not text:
        return ""

    cleaned = text.strip().rstrip("?.")
    property_pattern = (
        r"density|tg|glass transition(?: temperature)?|young'?s? modulus|"
        r"diffusion coefficient|radius of gyration|viscosity|thermal conductivity|"
        r"property"
    )
    patterns = [
        rf"(?i)\b(?:{property_pattern})\s+(?:of|for)\s+(.+)$",
        rf"(?i)\bwhat\s+is\s+(?:the\s+)?(?:{property_pattern})\s+(?:of|for)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            entity = match.group(1).strip(" .,")
            return re.sub(r"(?i)\s+using\s+.*$", "", entity).strip()
    return ""


def _is_contextual_follow_up(query: str) -> bool:
    """Return whether a query appears to refer to previous chat evidence."""
    if _is_global_dataset_query(query):
        return False

    lowered = (query or "").lower()
    has_pointer = any(pointer in lowered for pointer in CONTEXT_POINTERS)
    has_topic = any(topic in lowered for topic in FOLLOW_UP_TOPICS)
    is_short_follow_up = len(lowered.split()) <= 8 and has_pointer
    return (has_pointer and has_topic) or is_short_follow_up


def _infer_follow_up_topic(query: str) -> str:
    """Infer the user-facing topic requested by a follow-up."""
    lowered = (query or "").lower()
    if "force field" in lowered:
        return "force field"
    if any(term in lowered for term in ("doi", "citation", "source", "paper")):
        return "source"
    if any(term in lowered for term in ("condition", "temperature", "pressure")):
        return "conditions"
    if any(term in lowered for term in ("method", "calculated", "computed", "obtained")):
        return "method"
    if any(term in lowered for term in ("unit", "value", "property")):
        return "property details"
    return "previous result"


def _extract_entity_from_evidence_payload(payload: Dict[str, Any]) -> str:
    """Extract a likely polymer entity from prior answer/evidence text."""
    snippets = payload.get("evidence_snippets", []) or []
    text = "\n".join(str(item) for item in snippets)
    match = re.search(r"(?im)^polymer:\s*([^\n]+?)(?:\s+property:|$)", text)
    if match:
        return match.group(1).strip()

    answer = str(payload.get("answer", ""))
    pmma_match = re.search(r"\bPMMA\b|poly\(methyl methacrylate\)", answer, re.I)
    if pmma_match:
        return "PMMA"
    return ""


def _infer_property_from_payload(payload: Dict[str, Any]) -> str:
    """Infer the property discussed in a prior assistant payload."""
    snippets = payload.get("evidence_snippets", []) or []
    text = "\n".join(str(item) for item in snippets)
    match = re.search(r"(?i)\bproperty:\s*([a-z_]+)", text)
    if match:
        raw = match.group(1).strip().lower()
        for phrase, key in PROPERTY_KEY_BY_PHRASE.items():
            if key == raw:
                return phrase
        return raw
    return _infer_property_phrase(str(payload.get("answer", "")))


def _build_decontextualized_query(
    query: str,
    entity: str,
    property_phrase: str,
    doi_links: List[str],
    prior_values: List[str],
) -> str:
    """Build a source-aware standalone query for retrieval and generation."""
    parts = [f"User follow-up: {query.strip()}"]
    if entity:
        parts.append(f"Referenced polymer/entity: {entity}")
    if property_phrase:
        parts.append(f"Referenced property: {property_phrase}")
    if prior_values:
        parts.append("Referenced value(s): " + ", ".join(prior_values[:3]))
    if doi_links:
        parts.append("Previous DOI(s): " + ", ".join(doi_links[:3]))
    return ". ".join(parts)


def _is_source_sensitive_follow_up(query: str) -> bool:
    """Return whether a follow-up should prefer previous DOI/value evidence."""
    return _is_contextual_follow_up(query)


def _extract_prior_doi_links(payload: Dict[str, Any]) -> List[str]:
    """Extract DOI links from a prior assistant payload."""
    links = payload.get("doi_links", []) or []
    return [str(link).strip() for link in links if str(link).strip()]


def _extract_prior_values(payload: Dict[str, Any]) -> List[str]:
    """Extract numeric property values from a prior assistant payload."""
    text_parts = [str(payload.get("answer", ""))]
    text_parts.extend(str(item) for item in payload.get("evidence_snippets", []) or [])
    text = "\n".join(text_parts)
    values = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])", text)
    return values[:8]


def _parse_numeric_value(value: Any) -> Optional[float]:
    """Parse the first numeric value from a metadata value field."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", str(value), flags=re.I)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_property_state_from_payload(
    payload: Dict[str, Any],
    property_key: str,
) -> Dict[str, Any]:
    """Extract the last cited entity/value/DOI for a property from an answer."""
    snippets = [str(item) for item in payload.get("evidence_snippets", []) or []]
    answer = str(payload.get("answer", ""))
    combined = snippets + [answer]

    property_pattern = re.escape(property_key).replace("_", r"[_\s]")
    for text in combined:
        pattern = (
            rf"(?is)polymer:\s*(?P<entity>.+?)\s+property:\s*{property_pattern}"
            rf"\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
        )
        match = re.search(pattern, text)
        if match:
            numeric = _parse_numeric_value(match.group("value"))
            if numeric is None:
                continue
            return {
                "entity": match.group("entity").strip(),
                "value": numeric,
                "value_text": match.group("value"),
                "doi_links": _extract_prior_doi_links(payload),
            }

    answer_pattern = (
        rf"(?is){property_pattern}.*?"
        rf"(?P<value>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
    )
    answer_match = re.search(answer_pattern, answer)
    if answer_match:
        numeric = _parse_numeric_value(answer_match.group("value"))
        if numeric is not None:
            return {
                "entity": _extract_entity_from_evidence_payload(payload),
                "value": numeric,
                "value_text": answer_match.group("value"),
                "doi_links": _extract_prior_doi_links(payload),
            }

    return {}


def _last_property_state(
    conversation_history: Optional[List[Dict[str, Any]]],
    property_key: str,
) -> Dict[str, Any]:
    """Return the most recent assistant-cited value for a property."""
    for message in reversed(conversation_history or []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        state = _extract_property_state_from_payload(content, property_key)
        if state:
            return state
    return {}


def _load_numeric_property_records(property_key: str) -> List[Dict[str, Any]]:
    """Load all numeric records for a property from the processed chunks file."""
    with Config.CHUNKS_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records: List[Dict[str, Any]] = []
    for chunk in payload.get("property_chunks", []):
        metadata = chunk.get("metadata", {}) or {}
        if metadata.get("property") != property_key:
            continue
        numeric_value = _parse_numeric_value(metadata.get("value"))
        if numeric_value is None:
            continue
        records.append(
            {
                "polymer_name": str(metadata.get("polymer_name", "")).strip(),
                "value": numeric_value,
                "value_text": str(metadata.get("value", "")).strip(),
                "force_field": str(metadata.get("force_field", "")).strip(),
                "doi": str(metadata.get("doi", "")).strip(),
                "extra_info": str(metadata.get("extra_info", "")).strip(),
            }
        )
    return records


def _is_density_comparison_query(query: str) -> bool:
    """Return whether a query asks for a structured density comparison."""
    lowered = (query or "").lower()
    mentions_density = "density" in lowered
    comparative = any(
        term in lowered
        for term in (
            "higher",
            "greater",
            "above",
            "more dense",
            "highest",
            "lower",
            "less",
            "below",
            "lowest",
        )
    )
    return mentions_density and comparative


def _comparison_direction(query: str) -> str:
    """Infer whether a structured comparison is higher, lower, max, or min."""
    lowered = (query or "").lower()
    if "highest" in lowered or "maximum" in lowered or "max " in lowered:
        return "max"
    if "lowest" in lowered or "minimum" in lowered or "min " in lowered:
        return "min"
    if any(term in lowered for term in ("lower", "less", "below")):
        return "lower"
    return "higher"


def _explicit_threshold(query: str) -> Optional[float]:
    """Extract an explicit numeric comparison threshold from the user query."""
    # Avoid treating DOI-like values as thresholds by only using numbers near
    # comparison language.
    match = re.search(
        r"(?i)(?:than|above|below|over|under)\s+"
        r"(?P<value>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
        query or "",
    )
    if not match:
        return None
    return _parse_numeric_value(match.group("value"))


def _format_density_records(records: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Format density records into answer text and evidence snippets."""
    snippets = []
    parts = []
    for record in records[:5]:
        force_field = f"; force field: {record['force_field']}" if record["force_field"] else ""
        condition = f"; condition: {record['extra_info']}" if record["extra_info"] else ""
        parts.append(
            f"{record['polymer_name']} = {record['value_text']} g/cm3{force_field}"
        )
        snippets.append(
            f"{record['polymer_name']}: density = {record['value_text']}"
            f"{force_field}{condition}"
        )
    return "; ".join(parts), snippets


def _unique_dois(records: List[Dict[str, Any]], limit: int = 6) -> List[str]:
    """Return unique DOI links from records."""
    out = []
    seen = set()
    for record in records:
        doi = record.get("doi", "")
        if doi and doi not in seen:
            seen.add(doi)
            out.append(doi)
        if len(out) >= limit:
            break
    return out


def _unique_values(values: List[str], limit: Optional[int] = None) -> List[str]:
    """Return unique non-empty strings in first-seen order."""
    out = []
    seen = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned.lower() in {"nan", "none", "null"}:
            continue
        if cleaned in seen:
            continue
        out.append(cleaned)
        seen.add(cleaned)
        if limit is not None and len(out) >= limit:
            break
    return out


def _extract_force_field_inventory_entity(query: str) -> str:
    """Extract the polymer target from a force-field inventory query."""
    text = (query or "").strip().rstrip("?.")
    if not re.search(r"(?i)\bforce fields?\b", text):
        return ""

    patterns = (
        r"(?i)\bforce fields?\s+(?:are\s+)?(?:reported|used|listed|available|studied)?\s*(?:for|of)\s+(.+?)(?:\s+in\s+(?:the|this)\s+(?:dataset|database|corpus))?$",
        r"(?i)\b(?:for|of)\s+(.+?)\s*,?\s+what\s+force fields?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        entity = match.group(1).strip(" .,'\"")
        if entity and entity.lower() not in {"this dataset", "the dataset"}:
            return entity
    return ""


def _force_field_inventory_records(entity: str) -> List[Dict[str, Any]]:
    """Return property records whose polymer name matches a target entity."""
    with Config.CHUNKS_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records: List[Dict[str, Any]] = []
    for chunk in payload.get("property_chunks", []):
        metadata = chunk.get("metadata", {}) or {}
        polymer_name = str(metadata.get("polymer_name", "")).strip()
        force_field = str(metadata.get("force_field", "")).strip()
        if not polymer_name or not force_field:
            continue
        if force_field.lower() in {"nan", "none", "null"}:
            continue
        if not _entity_matches_polymer_name(entity, polymer_name):
            continue
        records.append(
            {
                "polymer_name": polymer_name,
                "force_field": force_field,
                "property": str(metadata.get("property", "")).strip(),
                "value": str(metadata.get("value", "")).strip(),
                "doi": str(metadata.get("doi", "")).strip(),
                "paper_title": str(metadata.get("paper_title", "")).strip(),
                "chunk_id": str(metadata.get("chunk_id", "")).strip(),
            }
        )
    return records


def _try_force_field_inventory_response(
    query: str,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Answer polymer-specific force-field inventory questions from metadata."""
    entity = _extract_force_field_inventory_entity(query)
    if not entity:
        return None

    records = _force_field_inventory_records(entity)
    if not records:
        return {
            "answer": (
                f"I don't have force-field records for {entity} in the dataset."
            ),
            "evidence_snippets": [],
            "doi_links": [],
            "confidence": 0.95,
            "abstained": True,
            "abstention_reason": "No force-field records found for requested polymer.",
        }, {
            "total_retrieved": 0,
            "chunk_types": {"force_field_inventory": 1},
            "score_stats": {},
            "retrieved_chunk_ids": [],
            "retrieved_dois": [],
        }

    force_fields = _unique_values([record["force_field"] for record in records])
    dois = _unique_dois(records, limit=20)
    chunk_ids = _unique_values([record["chunk_id"] for record in records], limit=50)
    polymer_names = _unique_values([record["polymer_name"] for record in records], limit=5)
    snippets = [
        f"{record['polymer_name']}: force field = {record['force_field']}; "
        f"property = {record['property']}; DOI = {record['doi']}"
        for record in records[:10]
    ]
    answer = (
        f"For {entity}, the dataset reports {len(force_fields)} force field(s): "
        f"{'; '.join(force_fields)}. "
        f"Matched dataset polymer name(s): {', '.join(polymer_names)}."
    )
    if dois:
        answer += " Source DOI(s): " + "; ".join(dois) + "."

    return {
        "answer": answer,
        "evidence_snippets": snippets,
        "doi_links": dois,
        "confidence": 0.98,
        "abstained": False,
        "abstention_reason": "",
    }, {
        "total_retrieved": len(records),
        "chunk_types": {"force_field_inventory": len(records)},
        "score_stats": {},
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_dois": dois,
    }


def _try_structured_density_response(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Answer dataset-wide density comparisons with structured records."""
    if not _is_density_comparison_query(query):
        return None

    direction = _comparison_direction(query)
    records = _load_numeric_property_records("density")
    if not records:
        return None

    if direction in {"max", "min"}:
        reverse = direction == "max"
        ranked = sorted(records, key=lambda item: item["value"], reverse=reverse)
        formatted, snippets = _format_density_records(ranked)
        label = "highest" if direction == "max" else "lowest"
        return {
            "answer": f"The {label} density records in the dataset are: {formatted}.",
            "evidence_snippets": snippets,
            "doi_links": _unique_dois(ranked),
            "confidence": 0.98,
            "abstained": False,
            "abstention_reason": "",
        }

    threshold = _explicit_threshold(query)
    state = _last_property_state(conversation_history, "density")
    if threshold is None and state:
        threshold = float(state["value"])

    if threshold is None:
        return {
            "answer": (
                "I need a referenced density value to compare against. Please ask "
                "with a number, such as 'higher density than 0.96 g/cm3', or ask "
                "the density of a specific polymer first."
            ),
            "evidence_snippets": [],
            "doi_links": [],
            "confidence": 0.0,
            "abstained": True,
            "abstention_reason": "No density threshold was available for comparison.",
        }

    source_entity = str(state.get("entity", ""))
    if direction == "higher":
        filtered = [record for record in records if record["value"] > threshold]
        filtered.sort(key=lambda item: item["value"], reverse=True)
    else:
        filtered = [record for record in records if record["value"] < threshold]
        filtered.sort(key=lambda item: item["value"])

    if source_entity and "other polymer" in (query or "").lower():
        filtered = [
            record
            for record in filtered
            if not _entity_matches_polymer_name(source_entity, record["polymer_name"])
        ]

    if not filtered:
        comparator = "above" if direction == "higher" else "below"
        return {
            "answer": (
                f"No density records {comparator} {threshold:g} g/cm3 were found "
                "in the dataset."
            ),
            "evidence_snippets": [],
            "doi_links": [],
            "confidence": 0.98,
            "abstained": False,
            "abstention_reason": "",
        }

    formatted, snippets = _format_density_records(filtered)
    comparator = "higher than" if direction == "higher" else "lower than"
    answer = (
        f"Yes. I found {len(filtered)} density records {comparator} "
        f"{threshold:g} g/cm3. Top examples: {formatted}."
    )
    if source_entity:
        answer += f" The comparison threshold came from the prior {source_entity} density."

    return {
        "answer": answer,
        "evidence_snippets": snippets,
        "doi_links": _unique_dois(filtered),
        "confidence": 0.98,
        "abstained": False,
        "abstention_reason": "",
    }


def _values_match(candidate_value: Any, prior_values: List[str]) -> bool:
    """Return whether a candidate metadata value matches prior cited values."""
    if not prior_values:
        return True

    candidate = str(candidate_value).strip()
    if not candidate:
        return False

    for value in prior_values:
        if candidate == value:
            return True
        try:
            if abs(float(candidate) - float(value)) < 1e-9:
                return True
        except ValueError:
            continue
    return False


def _entity_matches_polymer_name(entity: str, polymer_name: str) -> bool:
    """Return whether a metadata polymer name matches the requested entity."""
    entity_lower = (entity or "").lower()
    polymer_lower = (polymer_name or "").lower()
    if not entity_lower or not polymer_lower:
        return False
    if entity_lower in polymer_lower:
        return True

    aliases = {
        "pmma": ("pmma", "methyl methacrylate", "polymethyl methacrylate"),
        "pet": ("pet", "polyethylene terephthalate"),
        "ps": ("ps", "polystyrene"),
    }
    return any(alias in polymer_lower for alias in aliases.get(entity_lower, ()))


def _source_lock_context(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build source-lock constraints from previous chat context."""
    if (
        not conversation_history
        or _is_global_dataset_query(query)
        or not _is_source_sensitive_follow_up(query)
    ):
        return {}

    previous_query = _last_user_query(conversation_history)
    assistant_payload = _last_assistant_payload(conversation_history)
    entity = (
        _extract_entity_from_property_query(previous_query)
        or _extract_entity_from_evidence_payload(assistant_payload)
    )
    query_property = _infer_property_phrase(query)
    property_phrase = (
        query_property
        if query_property and query_property != "property"
        else _infer_property_phrase(previous_query)
        or _infer_property_from_payload(assistant_payload)
    )
    doi_links = _extract_prior_doi_links(assistant_payload)
    prior_values = _extract_prior_values(assistant_payload)

    if not doi_links:
        return {}

    return {
        "entity": entity,
        "property_phrase": property_phrase,
        "property_key": _property_key_from_phrase(property_phrase),
        "doi_links": doi_links,
        "prior_values": prior_values,
        "topic": _infer_follow_up_topic(query),
    }


def contextualize_query(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Rewrite simple follow-up questions using prior user context."""
    if (
        not conversation_history
        or _is_global_dataset_query(query)
        or not _is_contextual_follow_up(query)
    ):
        return query

    previous_query = _last_user_query(conversation_history)
    assistant_payload = _last_assistant_payload(conversation_history)
    entity = (
        _extract_entity_from_property_query(previous_query)
        or _extract_entity_from_evidence_payload(assistant_payload)
    )
    query_property = _infer_property_phrase(query)
    property_phrase = (
        query_property
        if query_property and query_property != "property"
        else _infer_property_phrase(previous_query)
        or _infer_property_from_payload(assistant_payload)
    )
    doi_links = _extract_prior_doi_links(assistant_payload)
    prior_values = _extract_prior_values(assistant_payload)

    if not entity and not doi_links:
        return query

    return _build_decontextualized_query(
        query=query,
        entity=entity,
        property_phrase=property_phrase,
        doi_links=doi_links,
        prior_values=prior_values,
    )


def _try_global_dataset_response(query: str) -> Optional[Dict[str, Any]]:
    """Return deterministic full-corpus answers for dataset-level questions."""
    if not (_is_global_force_field_query(query) or _is_global_paper_count_query(query)):
        return None

    with Config.CHUNKS_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if _is_global_paper_count_query(query):
        metadata = payload.get("metadata", {}) or {}
        paper_chunks = payload.get("paper_chunks", [])
        property_chunks = payload.get("property_chunks", [])
        dois = {
            str((chunk.get("metadata", {}) or {}).get("doi", "")).strip()
            for chunk in paper_chunks + property_chunks
        }
        dois.discard("")

        paper_count = int(metadata.get("num_paper_chunks") or len(paper_chunks))
        property_count = int(metadata.get("num_property_chunks") or len(property_chunks))
        total_chunks = int(
            metadata.get("total_chunks") or paper_count + property_count
        )
        unique_dois = int(metadata.get("unique_dois") or len(dois))

        answer = (
            f"The dataset contains {paper_count} paper-level records representing "
            f"{unique_dois} unique DOI/paper records. It also contains "
            f"{property_count:,} property records, for {total_chunks:,} total "
            "processed chunks."
        )
        return {
            "answer": answer,
            "evidence_snippets": [
                "data/polymer_chunks.json metadata: "
                f"num_paper_chunks={paper_count}, unique_dois={unique_dois}, "
                f"num_property_chunks={property_count}, total_chunks={total_chunks}"
            ],
            "doi_links": [],
            "confidence": 1.0,
            "abstained": False,
            "abstention_reason": "",
        }

    force_fields: Counter[str] = Counter()
    for chunk in payload.get("property_chunks", []):
        metadata = chunk.get("metadata", {}) or {}
        force_field = str(metadata.get("force_field", "")).strip()
        if force_field and force_field.lower() not in {"nan", "none", "null"}:
            force_fields[force_field] += 1

    if not force_fields:
        return {
            "answer": "No force-field metadata was found in the local dataset.",
            "evidence_snippets": [],
            "doi_links": [],
            "confidence": 1.0,
            "abstained": True,
            "abstention_reason": "No force-field metadata found.",
        }

    top_items = force_fields.most_common(15)
    top_text = "; ".join(f"{name} ({count} records)" for name, count in top_items)
    answer = (
        f"The dataset contains {len(force_fields)} unique force-field entries. "
        f"The most common are: {top_text}. Use a more specific polymer or paper "
        "query if you want the force field associated with a particular result."
    )
    return {
        "answer": answer,
        "evidence_snippets": [
            f"Computed from {sum(force_fields.values())} property chunks in "
            "`data/polymer_chunks.json`."
        ],
        "doi_links": [],
        "confidence": 1.0,
        "abstained": False,
        "abstention_reason": "",
    }


def _retrieval_abstention_response(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a user-facing response when retrieval confidence is too low."""
    return {
        "answer": (
            "I don't have sufficient evidence in the current database to answer "
            "this confidently. Please rephrase with a specific polymer, property, "
            "force field, or DOI."
        ),
        "evidence_snippets": [],
        "doi_links": [],
        "confidence": float(results.get("confidence", 0.0)),
        "abstained": True,
        "abstention_reason": results.get("abstention_reason", "Insufficient evidence."),
    }


def _empty_retrieval_response() -> Dict[str, Any]:
    """Create a response for queries that retrieve no chunks."""
    return {
        "answer": (
            "I couldn't find relevant information in the database. Please try "
            "rephrasing your question or ask about specific polymers, properties, "
            "or force fields."
        ),
        "evidence_snippets": [],
        "doi_links": [],
        "confidence": 0.0,
        "abstained": True,
        "abstention_reason": "No chunks were retrieved.",
    }


def _extract_doi(metadata: Dict[str, Any]) -> str:
    """Return a normalized DOI value from chunk metadata."""
    doi = str(metadata.get("doi", "")).strip()
    return doi.strip()


def _append_parent_paper_chunks(
    vector_store: VectorStore,
    results: Dict[str, Any],
    max_papers: int = 2,
) -> None:
    """Append parent paper chunks for retrieved property chunks when available."""
    metadatas = results.get("metadatas", []) or []
    existing_ids = set(results.get("ids", []) or [])
    added = 0

    for metadata in metadatas:
        if metadata.get("chunk_type") != "property":
            continue
        doi = _extract_doi(metadata)
        if not doi:
            continue

        # Chroma metadata filtering supports exact matching on one field. Filter
        # paper chunks in Python so this remains compatible with Chroma 0.4.x.
        candidates = vector_store.collection.get(
            where={"doi": doi},
            include=["documents", "metadatas"],
        )
        candidate_ids = candidates.get("ids", []) or []
        candidate_docs = candidates.get("documents", []) or []
        candidate_metas = candidates.get("metadatas", []) or []

        for index, doc_id in enumerate(candidate_ids):
            if index >= len(candidate_docs) or index >= len(candidate_metas):
                continue
            candidate_meta = candidate_metas[index]
            if candidate_meta.get("chunk_type") != "paper" or doc_id in existing_ids:
                continue
            results.setdefault("ids", []).append(doc_id)
            results.setdefault("documents", []).append(candidate_docs[index])
            results.setdefault("metadatas", []).append(candidate_meta)
            results.setdefault("scores", []).append(0.0)
            existing_ids.add(doc_id)
            added += 1
            break

        if added >= max_papers:
            return


def _retrieve_source_locked_follow_up(
    vector_store: VectorStore,
    lock: Dict[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Retrieve chunks constrained to previous DOI/value context when possible."""
    if not lock:
        return {"documents": [], "metadatas": [], "ids": [], "scores": []}

    locked_docs: List[str] = []
    locked_metas: List[Dict[str, Any]] = []
    locked_ids: List[str] = []
    locked_scores: List[float] = []
    topic = str(lock.get("topic", ""))
    property_key = str(lock.get("property_key", ""))
    prior_values = lock.get("prior_values", [])

    for doi in lock.get("doi_links", []):
        candidates = vector_store.collection.get(
            where={"doi": doi},
            include=["documents", "metadatas"],
        )
        candidate_ids = candidates.get("ids", []) or []
        candidate_docs = candidates.get("documents", []) or []
        candidate_metas = candidates.get("metadatas", []) or []

        for index, doc_id in enumerate(candidate_ids):
            if index >= len(candidate_docs) or index >= len(candidate_metas):
                continue
            metadata = candidate_metas[index]
            chunk_type = metadata.get("chunk_type")
            if chunk_type == "paper":
                # Paper chunks often contain methodology and source context that
                # property chunks intentionally keep compact.
                locked_ids.append(doc_id)
                locked_docs.append(candidate_docs[index])
                locked_metas.append(metadata)
                locked_scores.append(0.8)
                continue

            if chunk_type != "property":
                continue
            if property_key and metadata.get("property") != property_key:
                continue
            entity = str(lock.get("entity", ""))
            if entity and not _entity_matches_polymer_name(
                entity,
                str(metadata.get("polymer_name", "")),
            ):
                continue
            if topic in {"force field", "property details", "previous result"}:
                if not _values_match(metadata.get("value"), prior_values):
                    continue
            elif prior_values and _values_match(metadata.get("value"), prior_values):
                locked_scores.append(1.0)
                locked_ids.append(doc_id)
                locked_docs.append(candidate_docs[index])
                locked_metas.append(metadata)
                continue

            locked_ids.append(doc_id)
            locked_docs.append(candidate_docs[index])
            locked_metas.append(metadata)
            locked_scores.append(1.0)
            if len(locked_docs) >= k:
                break
        if len(locked_docs) >= k:
            break

    if len(locked_docs) > k:
        locked_ids = locked_ids[:k]
        locked_docs = locked_docs[:k]
        locked_metas = locked_metas[:k]
        locked_scores = locked_scores[:k]

    return {
        "ids": locked_ids,
        "documents": locked_docs,
        "metadatas": locked_metas,
        "scores": locked_scores,
        "source_locked": bool(locked_docs),
        "source_lock_dois": lock.get("doi_links", []),
    }


def _add_comparison_notes(response: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Attach comparison caveats to generated responses when applicable."""
    if results.get("intent") != "comparison":
        return

    notes = []
    disclaimer = str(results.get("comparison_disclaimer") or "").strip()
    if disclaimer:
        notes.append(disclaimer)

    needs_alignment = results.get("comparison_requires_property_alignment")
    is_aligned = results.get("comparison_property_aligned")
    if needs_alignment and not is_aligned:
        notes.append(
            "Retrieval could not lock a single shared property across materials."
        )

    missing = results.get("comparison_missing") or []
    if missing:
        notes.append(
            "Limited evidence for: " + ", ".join(str(item) for item in missing)
        )

    if notes:
        response["retrieval_notes"] = " ".join(notes)


def process_query(
    query: str,
    system: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Process a user query through RDKit tools or the RAG pipeline."""
    retriever = system["retriever"]
    generator = system["generator"]
    controller = system.get("controller")
    retrieval_query = contextualize_query(query, conversation_history)
    source_lock = _source_lock_context(query, conversation_history)

    force_field_inventory_response = _try_force_field_inventory_response(query)
    if force_field_inventory_response is not None:
        return force_field_inventory_response

    global_response = _try_global_dataset_response(query)
    if global_response is not None:
        return global_response, {
            "total_retrieved": 0,
            "chunk_types": {"dataset_summary": 1},
            "score_stats": {},
            "retrieved_chunk_ids": [],
            "retrieved_dois": global_response.get("doi_links", []),
        }

    structured_response = _try_structured_density_response(
        query,
        conversation_history,
    )
    if structured_response is not None:
        return structured_response, {
            "total_retrieved": 0,
            "chunk_types": {"structured_density": 1},
            "score_stats": {},
            "retrieved_chunk_ids": [],
            "retrieved_dois": structured_response.get("doi_links", []),
        }

    if controller is not None:
        handled, tool_response = controller.try_handle(query)
        if handled:
            return tool_response, {
                "total_retrieved": 0,
                "chunk_types": {"tool": 1},
                "score_stats": {},
                "retrieved_chunk_ids": [],
                "retrieved_dois": [],
            }

    results = _retrieve_source_locked_follow_up(
        system["vector_store"],
        source_lock,
        Config.TOP_K,
    )
    if not results.get("documents"):
        results = retriever.retrieve(retrieval_query, k=Config.TOP_K)
    _append_parent_paper_chunks(system["vector_store"], results)
    stats = retriever.get_retrieval_stats(results)

    if results.get("abstained"):
        return _retrieval_abstention_response(results), stats

    if stats["total_retrieved"] == 0:
        return _empty_retrieval_response(), stats

    context = retriever.format_retrieved_context(results)
    response = generator.generate_structured_response(
        query=retrieval_query,
        retrieved_context=context,
        retrieval_meta=_retrieval_meta(results),
    )
    if retrieval_query != query:
        response["resolved_query"] = retrieval_query
    if results.get("source_locked"):
        response["retrieval_notes"] = (
            "Source-locked to the DOI(s) cited in the previous answer."
        )

    # Retrieval confidence is the lower bound displayed in the UI.
    retrieval_confidence = float(results.get("confidence", 0.0))
    response["confidence"] = max(
        float(response.get("confidence", 0.0)),
        retrieval_confidence,
    )
    _add_comparison_notes(response, results)

    return response, stats
