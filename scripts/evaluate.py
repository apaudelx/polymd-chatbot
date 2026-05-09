#!/usr/bin/env python3
"""
Baseline evaluation harness for PolyMD.

Computes:
- Retrieval Accuracy@k (term-based proxy)
- Abstention precision/recall
- Median retrieval latency
- Optional median end-to-end latency with generation

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --with-generation --limit 10
"""

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator


def load_golden_set(path: Path) -> List[Dict[str, Any]]:
    """Load newline-delimited golden-set evaluation items."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def term_hit(retrieved_docs: List[str], expected_terms: List[str]) -> bool:
    """Return whether any expected term appears in retrieved documents."""
    if not expected_terms:
        return False
    text = "\n".join(retrieved_docs).lower()
    return any(term.lower() in text for term in expected_terms)


def dominant_chunk_type(metadatas: List[Dict[str, Any]]) -> str:
    """Return the most common chunk type in retrieved metadata."""
    if not metadatas:
        return "none"
    counts: Dict[str, int] = {}
    for item in metadatas:
        ctype = str(item.get("chunk_type", "unknown"))
        counts[ctype] = counts.get(ctype, 0) + 1
    return max(counts, key=counts.get)


def ensure_reports_dir(path: Path) -> None:
    """Create the reports directory if it is missing."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a dictionary payload to a formatted JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write per-query evaluation rows to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    """Write a compact Markdown summary of evaluation metrics."""
    lines = [
        "# PolyMD Evaluation Report",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Run Settings",
        f"- Top K: {summary['settings']['top_k']}",
        f"- Retriever Mode: {summary['settings']['retriever_mode']}",
        f"- Golden Set Size: {summary['settings']['golden_size']}",
        f"- With Generation: {summary['settings']['with_generation']}",
        "",
        "## Metrics",
        f"- Retrieval Accuracy@k: {summary['metrics']['retrieval_accuracy_at_k']:.3f}",
        f"- Chunk-Type Match Rate: {summary['metrics']['chunk_type_match_rate']:.3f}",
        f"- Abstention Precision: {summary['metrics']['abstention_precision']:.3f}",
        f"- Abstention Recall: {summary['metrics']['abstention_recall']:.3f}",
        f"- Median Retrieval Latency (ms): {summary['metrics']['median_retrieval_latency_ms']:.2f}",
        f"- Median End-to-End Latency (ms): {summary['metrics']['median_end_to_end_latency_ms']:.2f}",
        "",
        "## Notes",
        "- Retrieval Accuracy@k is a term-based proxy against expected terms in golden set entries.",
        "- Use this baseline with manual factuality review for final course reporting.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def safe_median(values: List[float]) -> float:
    """Return the median for a list, or zero for empty input."""
    return float(statistics.median(values)) if values else 0.0


def main() -> None:
    """Run retrieval and optional generation evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate PolyMD baseline metrics")
    parser.add_argument("--golden", type=str, default="data/golden_set.jsonl")
    parser.add_argument("--top-k", type=int, default=Config.TOP_K)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    golden_path = root / args.golden
    reports_dir = root / "reports"
    ensure_reports_dir(reports_dir)

    Config.validate()

    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore()
    indexed_chunks = vector_store.get_count()
    if indexed_chunks == 0:
        raise SystemExit(
            "Vector store is empty. Build the local Chroma index before "
            "running evaluation:\n\n"
            "  python scripts/build_vectordb.py --reset\n"
        )

    retriever = Retriever(vector_store, embedding_gen)
    generator = Generator() if args.with_generation else None

    golden = load_golden_set(golden_path)
    if args.limit > 0:
        golden = golden[: args.limit]

    per_query_rows: List[Dict[str, Any]] = []
    retrieval_latencies: List[float] = []
    end_to_end_latencies: List[float] = []

    retrieval_hits = 0
    chunk_type_hits = 0

    predicted_abstain_total = 0
    predicted_abstain_correct = 0
    expected_abstain_total = 0
    expected_abstain_hit = 0

    for item in golden:
        qid = item["id"]
        query = item["query"]
        expected_abstain = bool(item.get("expected_abstain", False))
        expected_terms = item.get("expected_terms", [])
        expected_chunk_type = item.get("expected_chunk_type", "unknown")

        t0 = time.perf_counter()
        results = retriever.retrieve(query, k=args.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        retrieval_latencies.append(retrieval_ms)

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        abstained = bool(results.get("abstained", False))
        confidence = float(results.get("confidence", 0.0))
        intent = str(results.get("intent", "unknown"))
        mode = str(results.get("mode", "unknown"))

        retrieved_chunk_type = dominant_chunk_type(metas)
        hit = term_hit(docs, expected_terms)
        ctype_match = retrieved_chunk_type == expected_chunk_type

        retrieval_hits += 1 if hit else 0
        chunk_type_hits += 1 if ctype_match else 0

        if abstained:
            predicted_abstain_total += 1
            if expected_abstain:
                predicted_abstain_correct += 1

        if expected_abstain:
            expected_abstain_total += 1
            if abstained:
                expected_abstain_hit += 1

        e2e_ms = retrieval_ms
        if generator is not None:
            t1 = time.perf_counter()
            context = retriever.format_retrieved_context(results)
            _ = generator.generate_structured_response(
                query=query,
                retrieved_context=context,
                retrieval_meta={
                    "confidence": confidence,
                    "abstained": abstained,
                    "abstention_reason": results.get("abstention_reason", ""),
                },
            )
            e2e_ms = (time.perf_counter() - t1) * 1000.0 + retrieval_ms

        end_to_end_latencies.append(e2e_ms)

        per_query_rows.append(
            {
                "id": qid,
                "query": query,
                "expected_abstain": expected_abstain,
                "predicted_abstain": abstained,
                "expected_chunk_type": expected_chunk_type,
                "retrieved_chunk_type": retrieved_chunk_type,
                "retrieval_hit_at_k": hit,
                "chunk_type_match": ctype_match,
                "confidence": round(confidence, 4),
                "intent": intent,
                "mode": mode,
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "end_to_end_latency_ms": round(e2e_ms, 2),
            }
        )

    n = len(golden)
    retrieval_acc = retrieval_hits / n if n else 0.0
    chunk_type_rate = chunk_type_hits / n if n else 0.0

    abstention_precision = (
        predicted_abstain_correct / predicted_abstain_total
        if predicted_abstain_total
        else 0.0
    )
    abstention_recall = (
        expected_abstain_hit / expected_abstain_total
        if expected_abstain_total
        else 0.0
    )

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "settings": {
            "top_k": args.top_k,
            "retriever_mode": Config.RETRIEVER_MODE,
            "golden_size": n,
            "with_generation": bool(args.with_generation),
        },
        "metrics": {
            "retrieval_accuracy_at_k": retrieval_acc,
            "chunk_type_match_rate": chunk_type_rate,
            "abstention_precision": abstention_precision,
            "abstention_recall": abstention_recall,
            "median_retrieval_latency_ms": safe_median(retrieval_latencies),
            "median_end_to_end_latency_ms": safe_median(end_to_end_latencies),
        },
    }

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = reports_dir / f"evaluation_summary_{stamp}.json"
    details_path = reports_dir / f"evaluation_details_{stamp}.csv"
    report_md_path = reports_dir / f"evaluation_report_{stamp}.md"

    write_json(summary_path, summary)
    write_csv(details_path, per_query_rows)
    write_markdown(report_md_path, summary)

    print("=" * 60)
    print("PolyMD Evaluation Complete")
    print("=" * 60)
    print(f"Golden set size: {n}")
    print(f"Retrieval Accuracy@k: {retrieval_acc:.3f}")
    print(f"Chunk-type match rate: {chunk_type_rate:.3f}")
    print(f"Abstention precision: {abstention_precision:.3f}")
    print(f"Abstention recall: {abstention_recall:.3f}")
    print(f"Median retrieval latency (ms): {safe_median(retrieval_latencies):.2f}")
    print(f"Median end-to-end latency (ms): {safe_median(end_to_end_latencies):.2f}")
    print("-")
    print(f"Summary JSON: {summary_path}")
    print(f"Details CSV:  {details_path}")
    print(f"Report MD:    {report_md_path}")


if __name__ == "__main__":
    main()
