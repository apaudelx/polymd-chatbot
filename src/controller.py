"""
Tool-calling controller for local-first RDKit extensions.

Handles explicit `rdkit:` commands, light natural-language routing to resolve names,
SMILES-based tool runs, and UI-shaped payloads (for example 2D render paths).

Supported command format:
  rdkit: canonicalize <SMILES>
  rdkit: descriptors <SMILES>
  rdkit: similarity <SMILES_A> <SMILES_B>
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from src.rdkit_tools import execute_tool


class ToolController:
    """Routes tool-eligible queries and executes supported tools."""

    def try_handle(self, query: str) -> Tuple[bool, Optional[Any]]:
        """
        Try to handle a query via a local tool.

        Returns:
            (handled, response_text)
        """
        if not query:
            return False, None

        stripped = query.strip()
        lowered = stripped.lower()
        if not lowered.startswith("rdkit:"):
            if self._is_rdkit_like_query(lowered):
                return True, self._handle_natural_language_rdkit(stripped)
            return False, None

        command = stripped[len("rdkit:") :].strip()
        if not command:
            return True, self._help_text()

        parts = command.split()
        action = parts[0].lower()

        if action == "canonicalize":
            if len(parts) < 2:
                return True, "Usage: rdkit: canonicalize <SMILES>"
            smiles = " ".join(parts[1:]).strip()
            return True, self._run("canonicalize_smiles", {"smiles": smiles})

        if action == "descriptors":
            if len(parts) < 2:
                return True, "Usage: rdkit: descriptors <SMILES>"
            smiles = " ".join(parts[1:]).strip()
            return True, self._run("molecular_descriptors", {"smiles": smiles})

        if action == "similarity":
            if len(parts) < 3:
                return True, "Usage: rdkit: similarity <SMILES_A> <SMILES_B>"
            smiles_a = parts[1].strip()
            smiles_b = parts[2].strip()
            return True, self._run(
                "tanimoto_similarity",
                {"smiles_a": smiles_a, "smiles_b": smiles_b},
            )

        if action == "render":
            if len(parts) < 2:
                return True, "Usage: rdkit: render <SMILES> [output_path]"
            smiles = parts[1].strip()
            output_path = parts[2].strip() if len(parts) > 2 else "molecule.png"
            return True, self._run(
                "render_molecule_2d",
                {"smiles": smiles, "output_path": output_path},
            )

        if action == "resolve":
            if len(parts) < 2:
                return True, "Usage: rdkit: resolve <compound_or_polymer_name> [--alias]"
            allow_alias = "--alias" in parts or "--allow-alias" in parts
            name_parts = [p for p in parts[1:] if p not in {"--alias", "--allow-alias"}]
            name = " ".join(name_parts).strip()
            return True, self._run(
                "resolve_name_to_smiles",
                {"name": name, "allow_alias_fallback": allow_alias},
            )

        return True, (
            "Unknown RDKit action. Supported actions: canonicalize, descriptors, similarity, render, resolve.\n"
            + self._help_text()
        )

    @staticmethod
    def _is_rdkit_like_query(lowered_query: str) -> bool:
        """Return whether a query appears to request molecular tooling."""
        if "rdkit" in lowered_query or "smiles" in lowered_query:
            return True

        structure_patterns = (
            r"\b(?:draw|render|visualize)\b",
            r"\bshow\s+me\s+(?:a|the)?\s*structure\b",
            r"\bwhat\s+does\s+\S+\s+look\s+like\b",
        )
        descriptor_patterns = (
            r"\b(?:calculate|compute|get|show)\s+(?:the\s+)?(?:descriptors?|molecular\s+weight|logp)\b",
            r"\bsimilarity\s+(?:between|of|for)\b",
        )
        return any(
            re.search(pattern, lowered_query)
            for pattern in structure_patterns + descriptor_patterns
        )

    @staticmethod
    def _looks_like_smiles(text: str) -> bool:
        """Return whether text has enough SMILES syntax to treat as a structure."""
        candidate = text.strip().strip(" .?\"'")
        if not candidate or " " in candidate:
            return False
        has_atom = bool(re.search(r"[BCNOFPSIbcclnops]", candidate))
        has_structure_syntax = bool(re.search(r"[=#@\[\]\(\)\\/0-9]", candidate))
        return has_atom and has_structure_syntax

    def _extract_smiles_from_query(self, query: str) -> Optional[str]:
        """Extract a likely SMILES string from a natural-language query."""
        for quoted in re.findall(r'"([^"]+)"', query):
            if self._looks_like_smiles(quoted):
                return quoted.strip()

        tokens = re.findall(r"[A-Za-z0-9@\+\-\[\]\(\)=#$\\/%.]+", query)
        for token in sorted(tokens, key=len, reverse=True):
            if self._looks_like_smiles(token):
                return token.strip()
        return None

    def _extract_name_from_query(self, query: str) -> Optional[str]:
        """Extract a candidate molecule or polymer name from natural language."""
        q = query.strip()

        patterns = [
            r"^(?:draw|render|visualize)\s+([a-zA-Z0-9\-\s\(\)]+)[\?\.]?$",
            r"(?:show|give)\s+me\s+(?:(?:a|the)\s+)?structure\s+(?:for|of)\s+([a-zA-Z0-9\-\s\(\)]+)[\?\.]?$",
            r"(?:calculate|compute|get|show)\s+(?:the\s+)?(?:descriptors?|molecular\s+weight|logp)\s+(?:for|of)\s+([a-zA-Z0-9\-\s\(\)]+)[\?\.]?$",
            r"(?:smiles|structure)\s+(?:for|of)\s+([a-zA-Z0-9\-\s\(\)]+)[\?\.]?$",
            r"for\s+([a-zA-Z0-9\-\s\(\)]+)[\?\.]?$",
            r"of\s+([a-zA-Z0-9\-\s\(\)]+)[\?\.]?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, q, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .?\"")
                if candidate.lower() not in {"this polymer", "this molecule", "this compound", "polymer"}:
                    return candidate

        quoted = re.findall(r'"([^"]+)"', q)
        if quoted:
            return quoted[-1].strip()

        return None

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Convert a user-provided name into a safe PNG filename."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
        return f"{slug or 'molecule'}.png"

    def _handle_natural_language_rdkit(self, query: str) -> Any:
        """Resolve and execute a simple natural-language RDKit request."""
        lowered = query.lower()
        smiles = self._extract_smiles_from_query(query)
        wants_structure = any(
            token in lowered
            for token in ("draw", "render", "structure", "look like", "visualize")
        )
        if smiles and wants_structure:
            rendered = self._run(
                "render_molecule_2d",
                {"smiles": smiles, "output_path": self._safe_filename("smiles")},
            )
            if isinstance(rendered, dict):
                rendered["answer"] = (
                    "I interpreted your query as a request to render the SMILES "
                    f"`{smiles}`.\n\n"
                    + rendered.get("answer", "")
                )
            return rendered

        if smiles and "descriptor" in lowered:
            return self._run("molecular_descriptors", {"smiles": smiles})

        name = self._extract_name_from_query(query)

        if not name:
            return (
                "I can run RDKit for you, but I need a specific molecule or polymer identifier. "
                "Please provide a chemical name or a SMILES string."
            )

        resolve_result = execute_tool(
            "resolve_name_to_smiles",
            {"name": name, "allow_alias_fallback": False},
        )
        if not resolve_result.get("ok"):
            resolver_error = resolve_result.get("error", "unknown error")
            return (
                f"I could not resolve '{name}' to a SMILES string automatically. "
                "Please check the spelling, try a more specific chemical name, or provide a SMILES string directly. "
                "Polymer names may not resolve cleanly in PubChem; in those cases, provide a "
                "monomer, oligomer, repeat-unit, or other explicit SMILES representation. "
                f"Resolution detail: {resolver_error}"
            )

        resolved_smiles = resolve_result["result"].get("canonical_smiles", "")
        source = resolve_result["result"].get("source", "unknown")
        note = resolve_result["result"].get("note", "")
        monomer_name = resolve_result["result"].get("monomer_name", "")
        repeat_unit = resolve_result["result"].get("repeat_unit_text", "")
        if not resolved_smiles:
            return (
                f"No canonical SMILES was returned for '{name}'. "
                "Please provide a SMILES string directly."
            )

        if "descriptor" in lowered:
            details = self._run("molecular_descriptors", {"smiles": resolved_smiles})
            return (
                f"Resolved '{name}' to SMILES: {resolved_smiles} (Source: {source})\n"
                + (f"Note: {note}\n" if note else "")
                + "\n"
                f"{details}"
            )

        if wants_structure:
            output_path = self._safe_filename(name)
            rendered = self._run("render_molecule_2d", {"smiles": resolved_smiles, "output_path": output_path})
            prefix = (
                f"Resolved '{name}' to SMILES: `{resolved_smiles}` (Source: {source})\n"
                + (f"Note: {note}\n" if note else "")
            )
            if isinstance(rendered, dict):
                rendered["answer"] = prefix + "\n" + rendered.get("answer", "")
                return rendered
            return (
                prefix
                + "\n"
                f"{rendered}"
            )

        if "smiles" in lowered or "rdkit" in lowered:
            return (
                f"Resolved '{name}' to canonical SMILES: {resolved_smiles} (Source: {source})\n"
                + (f"Note: {note}\n" if note else "")
                + "You can now run, for example:\n"
                f"- rdkit: render {resolved_smiles}\n"
                f"- rdkit: descriptors {resolved_smiles}"
            )

        return (
            f"Resolved '{name}' to canonical SMILES: {resolved_smiles} (Source: {source}). "
            + (f"Note: {note} " if note else "")
            + "Tell me if you want descriptors, similarity, or a rendered 2D structure."
        )

    def _run(self, tool_name: str, args: dict) -> Any:
        """Execute an RDKit tool and format its result for the UI."""
        result = execute_tool(tool_name, args)
        if not result.get("ok"):
            return f"RDKit tool error: {result.get('error', 'Unknown error')}"

        payload = result.get("result", {})

        if tool_name == "canonicalize_smiles":
            return (
                "RDKit Result\n"
                f"- Input: {payload.get('input_smiles')}\n"
                f"- Canonical: {payload.get('canonical_smiles')}"
            )

        if tool_name == "molecular_descriptors":
            return (
                "RDKit Descriptors\n"
                f"- Input: {payload.get('input_smiles')}\n"
                f"- Canonical: {payload.get('canonical_smiles')}\n"
                f"- MolWt: {payload.get('molecular_weight')}\n"
                f"- LogP: {payload.get('logp')}\n"
                f"- H Donors: {payload.get('h_donors')}\n"
                f"- H Acceptors: {payload.get('h_acceptors')}\n"
                f"- TPSA: {payload.get('tpsa')}"
            )

        if tool_name == "tanimoto_similarity":
            return (
                "RDKit Similarity\n"
                f"- A: {payload.get('smiles_a')}\n"
                f"- B: {payload.get('smiles_b')}\n"
                f"- Tanimoto: {payload.get('similarity')}"
            )

        if tool_name == "render_molecule_2d":
            return {
                "tool": "rdkit_render",
                "answer": (
                    "RDKit Render\n"
                    f"- Input: {payload.get('input_smiles')}"
                ),
                "image_path": payload.get("output_path"),
                "image_width": 180,
            }

        if tool_name == "resolve_name_to_smiles":
            return (
                "RDKit Name Resolution\n"
                f"- Name: {payload.get('name')}\n"
                f"- Canonical SMILES: {payload.get('canonical_smiles')}\n"
                f"- Source: {payload.get('source')}"
            )

        return f"RDKit tool executed successfully: {payload}"

    @staticmethod
    def _help_text() -> str:
        """Return usage examples for explicit RDKit commands."""
        return (
            "RDKit command examples:\n"
            "- rdkit: canonicalize CCO\n"
            "- rdkit: descriptors CCO\n"
            "- rdkit: similarity CCO CCN\n"
            "- rdkit: render CCO molecule.png\n"
            "- rdkit: resolve polystyrene\n"
            "- rdkit: resolve polystyrene --alias"
        )
