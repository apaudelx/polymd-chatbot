"""
Response generation module using LLM to create answers from retrieved context
"""
import json
import re
from typing import List, Dict, Optional, Any

from openai import OpenAI
from src.config import Config


class Generator:
    """Handles LLM-based response generation"""

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the generator

        Args:
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
            model: LLM model name (defaults to Config.LLM_MODEL)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.LLM_MODEL
        self.client = OpenAI(api_key=self.api_key)

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    def generate_response(
        self,
        query: str,
        retrieved_context: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Generate a response using the LLM

        Args:
            query: User query
            retrieved_context: Context string from retrieved chunks
            temperature: LLM temperature (defaults to Config.TEMPERATURE)
            max_tokens: Maximum response tokens (defaults to Config.MAX_TOKENS)

        Returns:
            Generated response string
        """
        structured = self.generate_structured_response(
            query=query,
            retrieved_context=retrieved_context,
            retrieval_meta=None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return structured.get("answer", "")

    def _response_schema(self) -> Dict[str, Any]:
        """Return the strict JSON schema expected from the LLM."""
        return {
            "name": "polymd_structured_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Final user-facing answer"
                    },
                    "evidence_snippets": {
                        "type": "array",
                        "description": "Direct evidence snippets from retrieved context",
                        "items": {"type": "string"}
                    },
                    "doi_links": {
                        "type": "array",
                        "description": "DOI links supporting the answer",
                        "items": {"type": "string"}
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score from 0 to 1"
                    },
                    "abstained": {
                        "type": "boolean",
                        "description": "Whether the assistant abstained"
                    },
                    "abstention_reason": {
                        "type": "string",
                        "description": "Reason for abstention if abstained is true"
                    }
                },
                "required": [
                    "answer",
                    "evidence_snippets",
                    "doi_links",
                    "confidence",
                    "abstained",
                    "abstention_reason"
                ],
                "additionalProperties": False
            }
        }

    def _extract_evidence_snippets(self, retrieved_context: str, limit: int = 3) -> List[str]:
        """Extract compact fallback evidence snippets from retrieved context."""
        snippets: List[str] = []
        chunks = [c.strip() for c in retrieved_context.split("--- Chunk") if c.strip()]
        for chunk in chunks:
            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            cleaned = []
            for ln in lines:
                if ln.startswith("Type:") or ln.startswith("Relevance:"):
                    continue
                if ln[0].isdigit() and "]" in ln[:6]:
                    continue
                cleaned.append(ln)
            text = " ".join(cleaned)
            if text:
                snippets.append(text[:280])
            if len(snippets) >= limit:
                break
        return snippets

    def _extract_doi_links(self, retrieved_context: str) -> List[str]:
        """Extract DOI links from retrieved context for citation fallback."""
        doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
        matches = re.findall(doi_pattern, retrieved_context, flags=re.IGNORECASE)
        normalized = []
        seen = set()
        for doi in matches:
            link = f"https://doi.org/{doi.strip().rstrip('.,;') }"
            if link not in seen:
                seen.add(link)
                normalized.append(link)
        return normalized[:6]

    def _coerce_structured_response(self, payload: Dict[str, Any], fallback_confidence: float) -> Dict[str, Any]:
        """Coerce an LLM payload into the UI's structured response contract."""
        response = {
            "answer": str(payload.get("answer", "")).strip(),
            "evidence_snippets": payload.get("evidence_snippets", []),
            "doi_links": payload.get("doi_links", []),
            "confidence": payload.get("confidence", fallback_confidence),
            "abstained": bool(payload.get("abstained", False)),
            "abstention_reason": str(payload.get("abstention_reason", "")),
        }

        if not isinstance(response["evidence_snippets"], list):
            response["evidence_snippets"] = []
        response["evidence_snippets"] = [str(s) for s in response["evidence_snippets"] if str(s).strip()]

        if not isinstance(response["doi_links"], list):
            response["doi_links"] = []
        response["doi_links"] = [str(s) for s in response["doi_links"] if str(s).strip()]

        try:
            response["confidence"] = float(response["confidence"])
        except Exception:
            response["confidence"] = fallback_confidence

        response["confidence"] = max(0.0, min(1.0, response["confidence"]))

        if response["abstained"] and not response["abstention_reason"]:
            response["abstention_reason"] = "Insufficient evidence in retrieved context."

        # Corpus-only contract: abstained must not show parametric LLM chemistry in answer.
        # Keep evidence_snippets / doi_links so users see what retrieval returned.
        if response["abstained"]:
            response["answer"] = Config.ABSTAIN_ANSWER_TEXT
        elif not response["answer"]:
            response["answer"] = "I couldn't generate a reliable answer from the available context."

        return response

    @staticmethod
    def _is_likely_truncated_json(error: Exception, content: str) -> bool:
        """Return whether a JSON parse error likely came from truncation."""
        msg = str(error).lower()
        return (
            isinstance(error, json.JSONDecodeError)
            and (
                "unterminated string" in msg
                or "expecting value" in msg
                or "expecting ',' delimiter" in msg
            )
            and len(content.strip()) > 0
        )

    @staticmethod
    def _is_comparison_query(query: str) -> bool:
        """Return whether a query has comparison-oriented wording."""
        q = (query or "").lower()
        return (
            "compare" in q
            or " vs " in q
            or " versus " in q
            or "difference" in q
            or ("higher" in q and " or " in q)
            or ("lower" in q and " or " in q)
        )

    def generate_structured_response(
        self,
        query: str,
        retrieved_context: str,
        retrieval_meta: Optional[Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        """
        Generate a schema-valid structured response.

        Returns fields:
        - answer
        - evidence_snippets
        - doi_links
        - confidence
        - abstained
        - abstention_reason
        """
        temperature = temperature if temperature is not None else Config.TEMPERATURE
        max_tokens = max_tokens or Config.MAX_TOKENS
        fallback_confidence = float((retrieval_meta or {}).get("confidence", 0.0))

        prompt_template = Config.get_prompt_template()
        grounded_prompt = prompt_template.format(
            retrieved_chunks=retrieved_context, user_query=query
        )

        meta = retrieval_meta or {}
        comparison_guidance = ""
        if meta.get("intent") == "comparison":
            lines = [
                "\nComparison rules (retrieval-aligned):",
            ]
            disc = str(meta.get("comparison_disclaimer") or "").strip()
            if disc:
                lines.append(f"- {disc}")
            if meta.get("comparison_requires_property_alignment"):
                if meta.get("comparison_property_aligned"):
                    props = meta.get("comparison_shared_properties") or []
                    if props:
                        lines.append(
                            "- Only compare on this shared property axis: "
                            + ", ".join(str(p) for p in props)
                            + ". Do not mix other property keys in the same ranking."
                        )
                    lines.append(
                        "- Lead with aligned numbers for each material; cite units; avoid cross-property superlatives."
                    )
                else:
                    lines.append(
                        "- CRITICAL: Evidence is NOT property-aligned for this question. "
                        "Do not rank materials on one numerical axis. State the limitation first, "
                        "then summarize only what the context supports (e.g. paper-level or qualitative notes)."
                    )
            else:
                lines.append(
                    "- This is a general comparison: include an explicit disclaimer that properties or "
                    "conditions may differ and you are not implying a single-axis scientific ranking."
                )
            missing = meta.get("comparison_missing") or []
            if missing:
                lines.append(
                    "- Evidence may be incomplete for: "
                    + ", ".join(str(m) for m in missing)
                    + ". Acknowledge gaps rather than inferring."
                )
            comparison_guidance = "\n".join(lines) + "\n"
        elif self._is_comparison_query(query):
            comparison_guidance = (
                "\nComparison rules for scientific validity:\n"
                "- Only make direct comparisons when the same property is available for both materials.\n"
                "- Do not compare values across different properties (for example density vs diffusion vs Tg).\n"
                "- If no shared property exists in context, explicitly say a direct comparison is not possible from current evidence.\n"
                "- If no shared property exists, provide a brief 'available data by material' summary without claiming one is higher/better overall.\n"
            )

        base_prompt = (
            f"{grounded_prompt}\n\n"
            "Return a JSON object with exactly these fields: "
            "answer, evidence_snippets, doi_links, confidence, abstained, abstention_reason. "
            "Use only the provided context. "
            "If evidence is weak, set abstained=true and provide a clear abstention_reason."
            f"{comparison_guidance}"
        )

        compact_prompt = (
            f"{base_prompt}\n"
            "Keep output compact to ensure valid JSON: "
            "answer under 120 words, at most 3 evidence_snippets, each under 180 characters, "
            "and at most 4 doi_links."
        )

        attempt_settings = [
            {"prompt": base_prompt, "max_tokens": max_tokens},
            {"prompt": compact_prompt, "max_tokens": max(max_tokens * 2, 900)},
        ]

        last_error: Optional[Exception] = None
        last_content: str = ""

        for i, attempt in enumerate(attempt_settings):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": attempt["prompt"]}],
                    temperature=temperature,
                    max_tokens=attempt["max_tokens"],
                    response_format={"type": "json_schema", "json_schema": self._response_schema()},
                )
                content = response.choices[0].message.content or "{}"
                last_content = content
                payload = json.loads(content)
                structured = self._coerce_structured_response(payload, fallback_confidence)
                if not structured["doi_links"]:
                    structured["doi_links"] = self._extract_doi_links(retrieved_context)
                if not structured["evidence_snippets"]:
                    structured["evidence_snippets"] = self._extract_evidence_snippets(retrieved_context)
                return structured
            except Exception as e:
                last_error = e
                if i == len(attempt_settings) - 1:
                    break
                if not self._is_likely_truncated_json(e, last_content):
                    break

        print(f"Error generating structured response: {last_error}")
        return {
            "answer": "I apologize, but I encountered an error generating a structured response.",
            "evidence_snippets": self._extract_evidence_snippets(retrieved_context),
            "doi_links": self._extract_doi_links(retrieved_context),
            "confidence": fallback_confidence,
            "abstained": True,
            "abstention_reason": str(last_error),
        }

    def generate_response_from_results(
        self, query: str, retrieval_results: Dict
    ) -> str:
        """
        Generate response directly from retrieval results

        Args:
            query: User query
            retrieval_results: Results dictionary from Retriever

        Returns:
            Generated response string
        """
        # Format the context from retrieval results
        context = self._format_context(retrieval_results)

        # Generate response
        return self.generate_response(query, context)

    def _format_context(self, results: Dict) -> str:
        """
        Format retrieval results into context string

        Args:
            results: Results dictionary with 'documents', 'metadatas', etc.

        Returns:
            Formatted context string
        """
        context_parts = []

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for i, doc in enumerate(documents):
            chunk_text = f"\n--- Chunk {i + 1} ---\n"

            # Add metadata info
            if metadatas and i < len(metadatas):
                metadata = metadatas[i]
                if "chunk_type" in metadata:
                    chunk_text += f"[Type: {metadata['chunk_type']}]\n"

            # Add document content
            chunk_text += f"{doc}\n"

            context_parts.append(chunk_text)

        return "\n".join(context_parts)

    def generate_streaming_response(
        self, query: str, retrieved_context: str
    ):
        """
        Generate a streaming response (for real-time display)

        Args:
            query: User query
            retrieved_context: Context string from retrieved chunks

        Yields:
            Response chunks as they're generated
        """
        prompt_template = Config.get_prompt_template()
        prompt = prompt_template.format(
            retrieved_chunks=retrieved_context, user_query=query
        )

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"
