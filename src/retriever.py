"""
Retrieval module for finding relevant chunks based on user queries
"""
import re
from typing import List, Dict, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.config import Config
from src.entity_match import entity_match_in_text
from src.retrieval_filters import drop_property_chunks
from src.lexical_index import read_lexical_index


class Retriever:
    """Handles retrieval of relevant chunks from the vector store"""

    def __init__(
        self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize the retriever

        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self._lexical_ready = False
        self._bm25_index = None
        self._docs: List[str] = []
        self._ids: List[str] = []
        self._metadatas: List[Dict] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25 using lowercase alphanumeric terms."""
        # Word tokens only so "polymer?" / "polymers?" match in-doc "polymer"
        return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t]

    @staticmethod
    def _normalize_retrieval_query(query: str) -> str:
        """
        Normalize phrasing for retrieval only (BM25 + embeddings). Reduces brittleness
        between singular/plural and "what is" vs "what are" definitional variants.

        The user's exact question string is still used for intent, comparisons, and generation.
        """
        q = (query or "").strip()
        if not q:
            return q
        # Common plural → singular for sparse retrieval (no heavy stemmer dependency)
        for plural, singular in (
            (r"\bpolymers\b", "polymer"),
            (r"\bmonomers\b", "monomer"),
            (r"\bcopolymers\b", "copolymer"),
            (r"\belastomers\b", "elastomer"),
            (r"\bplastics\b", "plastic"),
        ):
            q = re.sub(plural, singular, q, flags=re.IGNORECASE)
        # "What are polymer …" → "What is polymer …" (after plural fold) aligns with "what is polymer?"
        q = re.sub(r"(?i)^what\s+are\s+(?=polymer\b)", "what is ", q)
        return q

    @staticmethod
    def _merge_retrieval_results(a: Dict, b: Dict, k: int) -> Dict:
        """Union two ranked lists by chunk id, keeping the higher score for each id."""
        best: Dict[str, Dict] = {}
        for src in (a, b):
            ids = src.get("ids") or []
            docs = src.get("documents") or []
            metas = src.get("metadatas") or []
            scores = src.get("scores") or []
            for i, doc_id in enumerate(ids):
                if i >= len(docs) or i >= len(metas):
                    continue
                sc = float(scores[i]) if i < len(scores) else 0.0
                prev = best.get(doc_id)
                if prev is None or sc > prev["score"]:
                    best[doc_id] = {"score": sc, "doc": docs[i], "meta": metas[i]}
        items = sorted(best.items(), key=lambda x: x[1]["score"], reverse=True)[:k]
        return {
            "ids": [i for i, _ in items],
            "documents": [v["doc"] for _, v in items],
            "metadatas": [v["meta"] for _, v in items],
            "scores": [v["score"] for _, v in items],
        }

    @staticmethod
    def _doc_mentions_entity(doc: str, entity: str) -> bool:
        """Return whether document text mentions an entity as whole tokens."""
        return entity_match_in_text(doc, entity)

    @staticmethod
    def _metadata_mentions_entity(metadata: Dict, entity: str) -> bool:
        """Return whether metadata names a requested entity."""
        polymer_name = str(metadata.get("polymer_name", ""))
        return entity_match_in_text(polymer_name, entity)

    @staticmethod
    def _hybrid_overfetch_k(k: int, intent: str) -> int:
        """Cap hybrid candidate pool without always using a large multiplier."""
        base = max(k, min(k * 2, 18))
        return base + 6 if intent == "comparison" else base

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Normalize a score list to the 0-to-1 range."""
        if not scores:
            return []
        lo = min(scores)
        hi = max(scores)
        if hi == lo:
            if hi <= 0:
                return [0.0 for _ in scores]
            return [1.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def _ensure_lexical_index(self) -> None:
        """Load or build the BM25 corpus cache for lexical retrieval."""
        if self._lexical_ready:
            return

        n_live = self.vector_store.get_count()
        loaded = read_lexical_index(
            Config.LEXICAL_INDEX_PATH,
            n_live,
            Config.COLLECTION_NAME,
        )
        if loaded:
            self._docs, self._ids, self._metadatas = loaded
        else:
            all_docs = self.vector_store.collection.get(
                include=["documents", "metadatas"]
            )
            self._docs = all_docs.get("documents", []) or []
            self._ids = all_docs.get("ids", []) or []
            self._metadatas = all_docs.get("metadatas", []) or []

        if BM25Okapi and self._docs:
            tokenized = [self._tokenize(doc) for doc in self._docs]
            self._bm25_index = BM25Okapi(tokenized)

        self._lexical_ready = True

    def _infer_intent(self, query: str) -> str:
        """Infer a coarse retrieval intent from the user query."""
        q = query.lower()
        if ("higher" in q or "lower" in q) and (" or " in q or " vs " in q or " versus " in q):
            return "comparison"
        if "compare" in q or "vs" in q or "difference" in q:
            return "comparison"
        if "force field" in q or "opls" in q or "compass" in q or "martini" in q:
            return "force_field"
        if "doi" in q or "paper" in q or "reference" in q or "citation" in q:
            return "paper_lookup"
        if (
            "density" in q
            or "glass transition" in q
            or re.search(r"\btg\b", q)
            or "young" in q
            or "diffusion" in q
        ):
            return "property_lookup"
        return "unknown"

    @staticmethod
    def _intent_chunk_filter(intent: str) -> Optional[Dict]:
        """Return the metadata filter associated with an inferred intent."""
        if intent in {"property_lookup", "force_field", "comparison"}:
            return {"chunk_type": "property"}
        if intent == "paper_lookup":
            return {"chunk_type": "paper"}
        return None

    @staticmethod
    def _extract_comparison_entities(query: str) -> List[str]:
        """Extract up to two compared material names from a query."""
        q = query.lower().strip()
        if ":" in q:
            q = q.split(":", 1)[1].strip()
        q = q.replace("?", " ")

        # Normalize common comparison wrappers.
        q = re.sub(r"^compare\s+", "", q).strip()
        q = re.sub(r"^(the\s+)?(properties?|data|values?|characteristics?)\s+of\s+", "", q).strip()
        q = re.sub(r"^(which\s+has\s+(higher|lower|more|less)\s+[^:]+:\s*)", "", q).strip()

        for sep in (" versus ", " vs ", " or "):
            if sep in q:
                parts = [p.strip(" .,") for p in q.split(sep) if p.strip()]
                cleaned = []
                for p in parts:
                    p = re.sub(r"^(which|what|has|higher|lower|more|less|the)\s+", "", p).strip()
                    if p and p not in cleaned:
                        cleaned.append(p)
                return cleaned[:2]

        # Handle natural comparison phrasing: "A and B".
        if " and " in q:
            left, right = q.split(" and ", 1)
            left = re.sub(r"^(between|of)\s+", "", left).strip(" .,")
            right = right.strip(" .,")
            if left and right:
                return [left, right]

        return []

    def _is_property_comparison_query(self, query: str) -> bool:
        """True when the user is comparing along a measurable property axis (must stay aligned)."""
        if self._infer_property_key(query):
            return True
        q = query.lower()
        ordinal = any(
            w in q
            for w in (
                "higher",
                "lower",
                "greater",
                "less",
                "more",
                "stronger",
                "weaker",
                "larger",
                "smaller",
                "faster",
                "slower",
            )
        )
        comparative = "compare" in q or " vs " in q or " versus " in q or " or " in q
        if ordinal and comparative:
            return True
        if comparative and re.search(r"\b(propert(y|ies))\b", q):
            return True
        return False

    def _shared_properties_for_entities(self, metadatas: List[Dict], entities: List[str]) -> List[str]:
        """Return property keys that are present for every compared entity."""
        if len(entities) < 2:
            return []

        per_entity_props: List[set] = []
        for entity in entities:
            props = {
                str(meta.get("property", "")).strip()
                for meta in metadatas
                if self._metadata_mentions_entity(meta, entity) and meta.get("property")
            }
            per_entity_props.append(props)

        if not per_entity_props:
            return []

        shared = set.intersection(*per_entity_props) if len(per_entity_props) > 1 else per_entity_props[0]
        return [p for p in shared if p]

    @staticmethod
    def _infer_property_key(query: str) -> Optional[str]:
        """Infer the canonical property key requested by a query."""
        q = query.lower()
        if "glass transition" in q or re.search(r"\btg\b", q):
            return "glass_transition_temp"
        if "density" in q:
            return "density"
        if "young" in q:
            return "youngs_modulus"
        if "diffusion" in q:
            return "diffusion_coefficient"
        if "radius of gyration" in q or "radius_gyration" in q:
            return "radius_gyration"
        return None

    def _ensure_comparison_coverage(self, query: str, results: Dict, k: int) -> Dict:
        """Inject evidence for missing comparison entities when possible."""
        entities = self._extract_comparison_entities(query)
        target_property = self._infer_property_key(query)
        if len(entities) < 2:
            return results

        existing_docs = results.get("documents", [])
        existing_ids = set(results.get("ids", []))
        missing_entities = [
            entity
            for entity in entities
            if not any(self._doc_mentions_entity(doc, entity) for doc in existing_docs)
        ]

        if not missing_entities:
            return results

        for entity in missing_entities:
            if target_property == "glass_transition_temp":
                targeted_query = f"{entity} glass transition temperature property"
            else:
                targeted_query = f"{entity} polymer property"
            if Config.RETRIEVER_MODE == "bm25":
                candidate = self._retrieve_lexical(
                    targeted_query,
                    k=max(k * 2, 6),
                    filter_metadata={"chunk_type": "property"},
                )
            elif Config.RETRIEVER_MODE == "vec":
                candidate = self._retrieve_vector(
                    targeted_query,
                    k=max(k * 2, 6),
                    filter_metadata={"chunk_type": "property"},
                    return_scores=True,
                )
            else:
                candidate = self._retrieve_hybrid(
                    targeted_query,
                    k=max(k * 2, 6),
                    filter_metadata={"chunk_type": "property"},
                )

            injected = False
            for i, doc_id in enumerate(candidate.get("ids", [])):
                doc = candidate["documents"][i]
                meta = candidate["metadatas"][i]
                if doc_id in existing_ids:
                    continue
                if not self._doc_mentions_entity(doc, entity):
                    continue
                if target_property and meta.get("property") != target_property:
                    continue

                results["ids"].append(doc_id)
                results["documents"].append(doc)
                results["metadatas"].append(meta)
                results.setdefault("scores", []).append(candidate.get("scores", [0.0])[i])
                existing_ids.add(doc_id)
                injected = True
                break

            if not injected:
                continue

        # Keep highest-score chunks after augmentation.
        if results.get("scores"):
            combined = list(
                zip(
                    results["ids"],
                    results["documents"],
                    results["metadatas"],
                    results["scores"],
                )
            )
            combined.sort(key=lambda x: x[3], reverse=True)
            trimmed = combined[:k]
            results["ids"] = [x[0] for x in trimmed]
            results["documents"] = [x[1] for x in trimmed]
            results["metadatas"] = [x[2] for x in trimmed]
            results["scores"] = [x[3] for x in trimmed]

        return results

    def _confidence_and_abstain(self, results: Dict) -> Tuple[float, bool, str]:
        """Calculate retrieval confidence and determine abstention."""
        scores = results.get("scores", [])
        if not scores:
            return 0.0, True, "No relevant evidence found."

        top = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        gap = max(0.0, top - second)

        metadatas = results.get("metadatas", [])
        consistency = 0.5
        if metadatas:
            types = [m.get("chunk_type", "unknown") for m in metadatas]
            dominant = max(types, key=types.count)
            consistency = types.count(dominant) / len(types)

        confidence = (0.6 * top) + (0.3 * gap) + (0.1 * consistency)

        if top < Config.ABSTAIN_MIN_TOP_SCORE:
            return confidence, True, "Top evidence score is below reliability threshold."
        if confidence < Config.ABSTAIN_MIN_CONFIDENCE:
            return confidence, True, "Overall evidence confidence is too low."

        return confidence, False, ""

    def _retrieve_vector(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict],
        return_scores: bool,
    ) -> Dict:
        """Retrieve candidate chunks using vector similarity."""
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        results = self.vector_store.query(
            query_embeddings=[query_embedding], n_results=k, where=filter_metadata
        )

        formatted = {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "ids": results["ids"][0] if results["ids"] else [],
        }

        if return_scores and "distances" in results:
            distances = results["distances"][0] if results["distances"] else []
            formatted["distances"] = distances
            formatted["scores"] = [max(0.0, 1 - d) for d in distances] if distances else []

        return formatted

    def _retrieve_lexical(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict],
    ) -> Dict:
        """Retrieve candidate chunks using BM25 or token overlap fallback."""
        self._ensure_lexical_index()
        if not self._docs:
            return {"documents": [], "metadatas": [], "ids": [], "scores": []}

        candidate_indices = list(range(len(self._docs)))
        if filter_metadata:
            key, value = next(iter(filter_metadata.items()))
            candidate_indices = [
                i for i in candidate_indices if self._metadatas[i].get(key) == value
            ]

        if not candidate_indices:
            return {"documents": [], "metadatas": [], "ids": [], "scores": []}

        if self._bm25_index:
            raw_scores = self._bm25_index.get_scores(self._tokenize(query))
            scored = [(i, float(raw_scores[i])) for i in candidate_indices]
        else:
            query_tokens = set(self._tokenize(query))
            scored = []
            for i in candidate_indices:
                doc_tokens = set(self._tokenize(self._docs[i]))
                overlap = len(query_tokens.intersection(doc_tokens))
                scored.append((i, float(overlap)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        norm = self._normalize_scores([s for _, s in top])

        return {
            "documents": [self._docs[i] for i, _ in top],
            "metadatas": [self._metadatas[i] for i, _ in top],
            "ids": [self._ids[i] for i, _ in top],
            "scores": norm,
        }

    def _retrieve_hybrid(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict],
    ) -> Dict:
        """Retrieve candidates by fusing vector and lexical rankings."""
        fetch_k = self._hybrid_overfetch_k(k, self._infer_intent(query))
        vec = self._retrieve_vector(query, k=fetch_k, filter_metadata=filter_metadata, return_scores=True)
        lex = self._retrieve_lexical(query, k=fetch_k, filter_metadata=filter_metadata)

        vec_scores = self._normalize_scores(vec.get("scores", []))
        lex_scores = self._normalize_scores(lex.get("scores", []))
        alpha = Config.HYBRID_ALPHA
        target_property = self._infer_property_key(query)

        fused: Dict[str, Dict] = {}

        for i, doc_id in enumerate(vec.get("ids", [])):
            fused[doc_id] = {
                "id": doc_id,
                "doc": vec["documents"][i],
                "meta": vec["metadatas"][i],
                "vec": vec_scores[i] if i < len(vec_scores) else 0.0,
                "lex": 0.0,
            }

        for i, doc_id in enumerate(lex.get("ids", [])):
            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "doc": lex["documents"][i],
                    "meta": lex["metadatas"][i],
                    "vec": 0.0,
                    "lex": lex_scores[i] if i < len(lex_scores) else 0.0,
                }
            else:
                fused[doc_id]["lex"] = lex_scores[i] if i < len(lex_scores) else 0.0

        ranked = []
        for item in fused.values():
            score = (alpha * item["vec"]) + ((1 - alpha) * item["lex"])
            if target_property and item.get("meta", {}).get("property") == target_property:
                score += 0.2
            ranked.append((item, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:k]

        return {
            "documents": [item["doc"] for item, _ in ranked],
            "metadatas": [item["meta"] for item, _ in ranked],
            "ids": [item["id"] for item, _ in ranked],
            "scores": [score for _, score in ranked],
        }

    def retrieve(
        self,
        query: str,
        k: int = None,
        filter_metadata: Optional[Dict] = None,
        return_scores: bool = True,
        mode: Optional[str] = None,
    ) -> Dict:
        """
        Retrieve top-k most relevant chunks for a query

        Args:
            query: User query string
            k: Number of chunks to retrieve (defaults to Config.TOP_K)
            filter_metadata: Optional metadata filter (e.g., {'chunk_type': 'property'})
            return_scores: Whether to include similarity scores

        Returns:
            Dictionary with 'documents', 'metadatas', 'ids', and optionally 'scores'
        """
        k = k or Config.TOP_K
        active_mode = mode or Config.RETRIEVER_MODE
        intent = self._infer_intent(query)
        active_filter = filter_metadata or self._intent_chunk_filter(intent)
        comparison_entities = self._extract_comparison_entities(query) if intent == "comparison" else []

        effective_k = max(k * 2, 8) if intent == "comparison" else k
        rq = self._normalize_retrieval_query(query)

        if active_mode == "bm25":
            results = self._retrieve_lexical(rq, effective_k, active_filter)
        elif active_mode == "vec":
            results = self._retrieve_vector(
                rq,
                effective_k,
                active_filter,
                return_scores=return_scores,
            )
        else:
            results = self._retrieve_hybrid(rq, effective_k, active_filter)

        if intent == "comparison":
            results = self._ensure_comparison_coverage(query, results, effective_k)
            target_property = self._infer_property_key(query)
            requires_alignment = self._is_property_comparison_query(query)
            comparison_disclaimer = ""
            comparison_property_aligned = True
            shared_properties: List[str] = []

            if requires_alignment:
                if target_property:
                    keep = [
                        i
                        for i, meta in enumerate(results.get("metadatas", []))
                        if meta.get("property") == target_property
                    ]
                    if len(keep) >= 2:
                        for key in ("documents", "metadatas", "ids", "scores"):
                            if key in results and isinstance(results[key], list):
                                results[key] = [results[key][i] for i in keep]
                        shared_properties = [target_property]
                    else:
                        comparison_property_aligned = False
                        comparison_disclaimer = (
                            "Direct property-aligned comparison is not possible from the retrieved evidence. "
                            "Only per-material available values should be summarized."
                        )
                else:
                    shared_properties = self._shared_properties_for_entities(
                        results.get("metadatas", []), comparison_entities
                    )
                    if shared_properties:
                        keep = [
                            i
                            for i, meta in enumerate(results.get("metadatas", []))
                            if meta.get("property") in shared_properties
                        ]
                        if keep:
                            for key in ("documents", "metadatas", "ids", "scores"):
                                if key in results and isinstance(results[key], list):
                                    results[key] = [results[key][i] for i in keep]
                        else:
                            comparison_property_aligned = False
                            comparison_disclaimer = (
                                "Direct property-aligned comparison is not possible from the retrieved evidence. "
                                "Only per-material available values should be summarized."
                            )
                    else:
                        comparison_property_aligned = False
                        comparison_disclaimer = (
                            "Direct property-aligned comparison is not possible from the retrieved evidence. "
                            "Only per-material available values should be summarized."
                        )
            else:
                comparison_disclaimer = (
                    "This comparison may span different properties or conditions; treat it as contextual, not a single-axis ranking."
                )

            results["comparison_requires_property_alignment"] = requires_alignment
            results["comparison_property_aligned"] = comparison_property_aligned
            results["comparison_shared_properties"] = shared_properties
            results["comparison_disclaimer"] = comparison_disclaimer

            if requires_alignment and not comparison_property_aligned:
                results = drop_property_chunks(results)

        # Return caller-requested k after retrieval-time augmentation.
        if len(results.get("documents", [])) > k:
            for key in ("documents", "metadatas", "ids", "scores"):
                if key in results and isinstance(results[key], list):
                    results[key] = results[key][:k]

        confidence, abstained, reason = self._confidence_and_abstain(results)
        comparison_covered = []
        comparison_missing = []
        if comparison_entities:
            metas = results.get("metadatas", [])
            comparison_covered = [
                e for e in comparison_entities if any(self._metadata_mentions_entity(meta, e) for meta in metas)
            ]
            comparison_missing = [e for e in comparison_entities if e not in comparison_covered]

        results["intent"] = intent
        results["mode"] = active_mode
        results["confidence"] = confidence
        results["abstained"] = abstained
        results["abstention_reason"] = reason
        if comparison_entities:
            results["comparison_entities"] = comparison_entities
            results["comparison_covered"] = comparison_covered
            results["comparison_missing"] = comparison_missing

        return results

    def retrieve_with_threshold(
        self, query: str, k: int = None, threshold: float = None
    ) -> Dict:
        """
        Retrieve chunks above a similarity threshold

        Args:
            query: User query string
            k: Maximum number of chunks to retrieve
            threshold: Minimum similarity score (defaults to Config.SIMILARITY_THRESHOLD)

        Returns:
            Dictionary with filtered results
        """
        threshold = threshold or Config.SIMILARITY_THRESHOLD
        k = k or Config.TOP_K * 2  # Retrieve more to filter

        # Get results
        results = self.retrieve(query, k=k, return_scores=True)

        # Filter by threshold if scores are available
        if "scores" in results and results["scores"]:
            filtered_indices = [
                i for i, score in enumerate(results["scores"]) if score >= threshold
            ]

            # Apply filter to all fields
            for key in results:
                if isinstance(results[key], list):
                    results[key] = [results[key][i] for i in filtered_indices]

        return results

    def retrieve_by_type(
        self, query: str, chunk_type: str, k: int = None
    ) -> Dict:
        """
        Retrieve chunks of a specific type

        Args:
            query: User query string
            chunk_type: Type of chunk ('property' or 'paper')
            k: Number of chunks to retrieve

        Returns:
            Dictionary with filtered results
        """
        return self.retrieve(
            query, k=k, filter_metadata={"chunk_type": chunk_type}
        )

    def format_retrieved_context(self, results: Dict) -> str:
        """
        Format retrieved chunks into a context string for the LLM

        Args:
            results: Results dictionary from retrieve()

        Returns:
            Formatted context string
        """
        context_parts = []

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        scores = results.get("scores", [])

        for i, doc in enumerate(documents):
            # Add chunk number
            chunk_context = f"\n--- Chunk {i + 1} ---\n"

            # Add metadata if available
            if metadatas and i < len(metadatas):
                metadata = metadatas[i]
                if "chunk_type" in metadata:
                    chunk_context += f"Type: {metadata['chunk_type']}\n"
                if metadata.get("polymer_name"):
                    chunk_context += f"Polymer: {metadata['polymer_name']}\n"
                if metadata.get("property"):
                    chunk_context += f"Property key: {metadata['property']}\n"

            # Add similarity score if available
            if scores and i < len(scores):
                chunk_context += f"Relevance: {scores[i]:.2f}\n"

            # Add document content
            chunk_context += f"\n{doc}\n"

            context_parts.append(chunk_context)

        return "\n".join(context_parts)

    def get_retrieval_stats(self, results: Dict) -> Dict:
        """
        Get statistics about retrieval results

        Args:
            results: Results dictionary from retrieve()

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_retrieved": len(results.get("documents", [])),
            "chunk_types": {},
            "retrieved_chunk_ids": [],
            "retrieved_dois": [],
        }

        # Count chunk types
        metadatas = results.get("metadatas", [])
        seen_chunk_ids = set()
        seen_dois = set()
        for metadata in metadatas:
            chunk_type = metadata.get("chunk_type", "unknown")
            stats["chunk_types"][chunk_type] = (
                stats["chunk_types"].get(chunk_type, 0) + 1
            )
            chunk_id = str(metadata.get("chunk_id", "")).strip()
            if chunk_id and chunk_id not in seen_chunk_ids:
                stats["retrieved_chunk_ids"].append(chunk_id)
                seen_chunk_ids.add(chunk_id)
            doi = str(metadata.get("doi", "")).strip()
            if doi:
                if not doi.startswith("http"):
                    doi = f"https://doi.org/{doi}"
                if doi not in seen_dois:
                    stats["retrieved_dois"].append(doi)
                    seen_dois.add(doi)

        # Add score statistics if available
        if "scores" in results and results["scores"]:
            scores = results["scores"]
            stats["score_stats"] = {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores),
            }

        if "mode" in results:
            stats["mode"] = results.get("mode")
        if "intent" in results:
            stats["intent"] = results.get("intent")
        if "confidence" in results:
            stats["confidence"] = results.get("confidence")
        if "abstained" in results:
            stats["abstained"] = results.get("abstained")
        if "abstention_reason" in results:
            stats["abstention_reason"] = results.get("abstention_reason")

        return stats
