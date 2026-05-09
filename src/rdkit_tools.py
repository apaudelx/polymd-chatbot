"""
RDKit tool layer for local molecular operations.

This module is designed as a local-first extension for agent/tool-calling flows.
It exposes:
- tool schema definitions
- argument validation
- deterministic tool executors

All functions fail safely when RDKit is unavailable.
"""

from __future__ import annotations

import json
import threading
import time
from urllib.parse import quote
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Any, Dict, List
import re

from src.config import Config


PUBCHEM_MIN_INTERVAL_SECONDS = 0.22
_PUBCHEM_RATE_LOCK = threading.Lock()
_LAST_PUBCHEM_REQUEST_AT = 0.0

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    AllChem = None
    Descriptors = None
    Draw = None
    DataStructs = None
    RDKIT_AVAILABLE = False


TOOL_CANONICALIZE_SMILES = {
    "name": "canonicalize_smiles",
    "description": "Validate and canonicalize a SMILES string.",
    "parameters": {
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "Input SMILES string"
            }
        },
        "required": ["smiles"],
        "additionalProperties": False
    }
}

TOOL_MOLECULAR_DESCRIPTORS = {
    "name": "molecular_descriptors",
    "description": "Compute core molecular descriptors from SMILES.",
    "parameters": {
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "Input SMILES string"
            }
        },
        "required": ["smiles"],
        "additionalProperties": False
    }
}

TOOL_TANIMOTO_SIMILARITY = {
    "name": "tanimoto_similarity",
    "description": "Compute Tanimoto similarity between two molecules.",
    "parameters": {
        "type": "object",
        "properties": {
            "smiles_a": {
                "type": "string",
                "description": "First SMILES string"
            },
            "smiles_b": {
                "type": "string",
                "description": "Second SMILES string"
            }
        },
        "required": ["smiles_a", "smiles_b"],
        "additionalProperties": False
    }
}

TOOL_RENDER_MOLECULE_2D = {
    "name": "render_molecule_2d",
    "description": "Render a 2D PNG image from a SMILES string.",
    "parameters": {
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "Input SMILES string"
            },
            "output_path": {
                "type": "string",
                "description": "Optional output PNG filename (stored in render sandbox)"
            }
        },
        "required": ["smiles"],
        "additionalProperties": False
    }
}

TOOL_RESOLVE_NAME_TO_SMILES = {
    "name": "resolve_name_to_smiles",
    "description": "Resolve a compound or polymer name to canonical SMILES using PubChem.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Compound or polymer/common name"
            },
            "allow_alias_fallback": {
                "type": "boolean",
                "description": "Deprecated; name resolution uses PubChem only"
            }
        },
        "required": ["name"],
        "additionalProperties": False
    }
}


TOOL_SCHEMAS = {
    "canonicalize_smiles": TOOL_CANONICALIZE_SMILES,
    "molecular_descriptors": TOOL_MOLECULAR_DESCRIPTORS,
    "tanimoto_similarity": TOOL_TANIMOTO_SIMILARITY,
    "render_molecule_2d": TOOL_RENDER_MOLECULE_2D,
    "resolve_name_to_smiles": TOOL_RESOLVE_NAME_TO_SMILES,
}


def _wait_for_pubchem_slot() -> None:
    """Throttle PubChem calls to stay below the public 5 requests/second limit."""
    global _LAST_PUBCHEM_REQUEST_AT

    with _PUBCHEM_RATE_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_PUBCHEM_REQUEST_AT
        wait_time = max(0.0, PUBCHEM_MIN_INTERVAL_SECONDS - elapsed)
        if wait_time:
            time.sleep(wait_time)
            now = time.monotonic()
        _LAST_PUBCHEM_REQUEST_AT = now


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return all RDKit tool schemas."""
    return list(TOOL_SCHEMAS.values())


def _validate_tool_args(tool_name: str, args: Dict[str, Any]) -> None:
    """Minimal JSON-schema-style validation without external dependency."""
    if tool_name not in TOOL_SCHEMAS:
        raise KeyError(f"Unknown tool: {tool_name}")

    schema = TOOL_SCHEMAS[tool_name]["parameters"]
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    allow_extra = not schema.get("additionalProperties") is False

    for key in required:
        if key not in args:
            raise ValueError(f"Missing required argument: {key}")

    if not allow_extra:
        extra_keys = [k for k in args.keys() if k not in properties]
        if extra_keys:
            raise ValueError(f"Unexpected arguments: {extra_keys}")

    for key, value in args.items():
        if key not in properties:
            continue
        expected_type = properties[key].get("type")
        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"Argument '{key}' must be a string")
        if expected_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Argument '{key}' must be a boolean")


def _require_rdkit() -> None:
    """Raise a clear runtime error when RDKit is unavailable."""
    if not RDKIT_AVAILABLE:
        raise RuntimeError(
            "RDKit is not installed. Install it in your environment before using RDKit tools."
        )


def _smiles_to_mol(smiles: str):
    """Parse a SMILES string into an RDKit molecule or raise validation errors."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def exec_canonicalize_smiles(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and canonicalize an input SMILES string."""
    mol = _smiles_to_mol(args["smiles"])
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return {
        "valid": True,
        "input_smiles": args["smiles"],
        "canonical_smiles": canonical,
    }


def exec_molecular_descriptors(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compute core molecular descriptors for an input SMILES string."""
    mol = _smiles_to_mol(args["smiles"])
    return {
        "input_smiles": args["smiles"],
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 4),
        "logp": round(float(Descriptors.MolLogP(mol)), 4),
        "h_donors": int(Descriptors.NumHDonors(mol)),
        "h_acceptors": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": round(float(Descriptors.TPSA(mol)), 4),
    }


def exec_tanimoto_similarity(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Morgan-fingerprint Tanimoto similarity for two molecules."""
    mol_a = _smiles_to_mol(args["smiles_a"])
    mol_b = _smiles_to_mol(args["smiles_b"])

    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
    similarity = float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

    return {
        "smiles_a": args["smiles_a"],
        "smiles_b": args["smiles_b"],
        "similarity": round(similarity, 6),
    }


def exec_render_molecule_2d(args: Dict[str, Any]) -> Dict[str, Any]:
    """Render a validated molecule to a sandboxed PNG file."""
    mol = _smiles_to_mol(args["smiles"])
    output_path = args.get("output_path") or "molecule.png"

    # Multi-user safety: confine writes to render sandbox and sanitize names.
    filename = Path(output_path).name
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    filename = filename or "molecule.png"

    sandbox_root = Config.RENDER_OUTPUT_DIR.resolve()
    output = (sandbox_root / filename).resolve()
    try:
        output.relative_to(sandbox_root)
    except ValueError as exc:
        raise ValueError("Resolved output path escapes render sandbox") from exc

    output.parent.mkdir(parents=True, exist_ok=True)

    Draw.MolToFile(mol, str(output), size=(500, 500))

    return {
        "input_smiles": args["smiles"],
        "output_path": str(output),
        "format": "png",
    }


def exec_resolve_name_to_smiles(args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a compound name to SMILES using PubChem."""
    name = args["name"].strip()
    if not name:
        raise ValueError("Name cannot be empty")

    endpoint = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name)}/property/SMILES,ConnectivitySMILES,CanonicalSMILES,IsomericSMILES/JSON"
    )

    request = Request(endpoint, headers={"User-Agent": "PolyMD-RDKit-Resolver/1.0"})
    payload = None
    pubchem_error = None
    pubchem_not_found = False
    pubchem_temporarily_unavailable = False
    backoff = 0.4
    for attempt in range(3):
        try:
            _wait_for_pubchem_slot()
            with urlopen(request, timeout=12) as response:
                raw = response.read().decode("utf-8")
            if not raw.lstrip().startswith(("{", "[")):
                pubchem_temporarily_unavailable = True
                pubchem_error = None
                break
            payload = json.loads(raw)
            pubchem_error = None
            break
        except json.JSONDecodeError as exc:
            pubchem_error = f"PubChem returned an unreadable response: {exc}"
            break
        except HTTPError as exc:
            if exc.code == 404:
                pubchem_not_found = True
                pubchem_error = None
                break
            if exc.code == 503:
                pubchem_temporarily_unavailable = True
                pubchem_error = None
            pubchem_error = str(exc)
            retryable = attempt < 2 and (
                exc.code in (429, 500, 502, 503, 504)
            )
            if retryable:
                time.sleep(backoff)
                backoff *= 2
            else:
                break
        except (URLError, OSError) as exc:
            pubchem_error = str(exc)
            retryable = attempt < 2
            if retryable:
                time.sleep(backoff)
                backoff *= 2
            else:
                break

    props = (payload or {}).get("PropertyTable", {}).get("Properties", [])
    if props:
        record = props[0]
        smiles = (
            record.get("CanonicalSMILES")
            or record.get("IsomericSMILES")
            or record.get("SMILES")
            or record.get("ConnectivitySMILES")
        )
        if smiles:
            return {
                "name": name,
                "canonical_smiles": smiles,
                "source": "PubChem",
            }

    if pubchem_not_found:
        raise ValueError(f"PubChem did not find a SMILES record for '{name}'.")
    if pubchem_temporarily_unavailable:
        raise RuntimeError(
            "PubChem is temporarily unavailable or returned a non-JSON response. "
            "Please try again later."
        )
    if pubchem_error:
        raise RuntimeError(f"Failed to resolve name via PubChem: {pubchem_error}")
    raise ValueError(f"No SMILES found for '{name}' via PubChem.")


TOOL_EXECUTORS = {
    "canonicalize_smiles": exec_canonicalize_smiles,
    "molecular_descriptors": exec_molecular_descriptors,
    "tanimoto_similarity": exec_tanimoto_similarity,
    "render_molecule_2d": exec_render_molecule_2d,
    "resolve_name_to_smiles": exec_resolve_name_to_smiles,
}


def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and execute one RDKit tool.

    Returns a structured result object with `ok`, `tool`, and `result` or `error`.
    """
    try:
        _validate_tool_args(tool_name, args)
        if tool_name not in TOOL_EXECUTORS:
            raise KeyError(f"No executor for tool: {tool_name}")
        result = TOOL_EXECUTORS[tool_name](args)
        return {"ok": True, "tool": tool_name, "result": result}
    except Exception as exc:
        return {"ok": False, "tool": tool_name, "error": str(exc)}
