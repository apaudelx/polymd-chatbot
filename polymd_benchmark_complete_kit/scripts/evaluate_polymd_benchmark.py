#!/usr/bin/env python3
"""
Evaluate PolyMD chatbot outputs against polymd_benchmark_golden_set.jsonl.

Expected prediction JSONL format, one row per benchmark id:
{
  "id": "polymd_0001",
  "answer": "... final chatbot answer ...",
  "retrieved_chunk_ids": ["..."],        # optional but recommended
  "retrieved_dois": ["https://doi.org/..."], # optional
  "abstained": false,                    # optional; inferred from answer if omitted
  "latency_ms": 1234                     # optional
}

This script intentionally uses simple, transparent checks:
- retrieval_chunk_recall: whether any expected chunk_id was retrieved
- retrieval_doi_recall: whether any expected DOI was retrieved or mentioned
- value_mention_rate: whether expected numeric/string values appear in the answer
- must_mention_rate: whether key required strings appear in the answer
- abstention_accuracy: whether answer abstains when it should and answers when it should not
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

ABSTAIN_PATTERNS = [
    r"\bnot found\b",
    r"\bno evidence\b",
    r"\bnot available\b",
    r"\bnot in (the )?(dataset|corpus)\b",
    r"\bcannot determine\b",
    r"\binsufficient evidence\b",
    r"\bI don't know\b",
    r"\bnot reported\b",
]


def load_jsonl(path):
    """Load newline-delimited JSON records from a benchmark file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s):
    """Normalize text for transparent substring scoring."""
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def contains(answer, needle):
    """Return whether an expected string appears in an answer."""
    if needle is None or str(needle).strip() == "":
        return True
    return norm(needle) in norm(answer)


def infer_abstained(answer):
    """Infer abstention from common refusal phrases when the flag is absent."""
    a = norm(answer)
    return any(re.search(p, a, flags=re.I) for p in ABSTAIN_PATTERNS)


def main():
    """Evaluate benchmark predictions against the golden set and write JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="polymd_benchmark_golden_set.jsonl")
    ap.add_argument("--pred", required=True, help="Predictions JSONL from your chatbot.")
    ap.add_argument("--out", default="polymd_eval_report.json")
    args = ap.parse_args()

    gold = {x["id"]: x for x in load_jsonl(args.gold)}
    preds = {x["id"]: x for x in load_jsonl(args.pred)}

    rows = []
    by_cat = defaultdict(list)

    for gid, g in gold.items():
        p = preds.get(gid, {})
        answer = p.get("answer", "")
        retrieved_chunk_ids = set(map(str, p.get("retrieved_chunk_ids", [])))
        retrieved_dois = set(map(str, p.get("retrieved_dois", [])))
        expected_chunk_ids = set(map(str, g["grading"].get("expected_chunk_ids", [])))
        expected_dois = set(map(str, g["grading"].get("expected_dois", [])))
        expected_values = list(map(str, g["grading"].get("numeric_values", [])))
        must_mention = list(map(str, g["grading"].get("must_mention", [])))
        should_abstain = bool(g["grading"].get("should_abstain", False))
        abstained = (
            bool(p.get("abstained")) if "abstained" in p else infer_abstained(answer)
        )

        chunk_recall = (
            None if not expected_chunk_ids else bool(expected_chunk_ids & retrieved_chunk_ids)
        )
        doi_recall = (
            None
            if not expected_dois
            else bool(expected_dois & retrieved_dois)
            or any(contains(answer, d) for d in expected_dois)
        )
        value_rate = (
            None
            if not expected_values
            else sum(contains(answer, v) for v in expected_values) / len(expected_values)
        )
        must_rate = (
            None
            if not must_mention
            else sum(contains(answer, m) for m in must_mention) / len(must_mention)
        )

        abstention_correct = (abstained == should_abstain)
        if should_abstain:
            passed = abstention_correct
        else:
            checks = []
            if chunk_recall is not None:
                checks.append(chunk_recall)
            if doi_recall is not None:
                checks.append(doi_recall)
            if value_rate is not None:
                checks.append(value_rate >= 0.5)
            if must_rate is not None:
                checks.append(must_rate >= 0.5)
            passed = abstention_correct and (
                sum(bool(x) for x in checks) / max(len(checks), 1) >= 0.5
            )

        row = {
            "id": gid,
            "category": g["category"],
            "passed": bool(passed),
            "abstention_correct": bool(abstention_correct),
            "chunk_recall": chunk_recall,
            "doi_recall": doi_recall,
            "value_mention_rate": value_rate,
            "must_mention_rate": must_rate,
            "latency_ms": p.get("latency_ms"),
        }
        rows.append(row)
        by_cat[g["category"]].append(row)

    def mean_bool(vals):
        """Return the mean of truthy values, ignoring null entries."""
        vals = [v for v in vals if v is not None]
        return None if not vals else sum(bool(v) for v in vals) / len(vals)

    def mean_num(vals):
        """Return the mean of numeric values, ignoring null entries."""
        vals = [v for v in vals if v is not None]
        return None if not vals else sum(float(v) for v in vals) / len(vals)

    summary = {
        "n_gold": len(gold),
        "n_predictions": len(preds),
        "overall_pass_rate": mean_bool([r["passed"] for r in rows]),
        "abstention_accuracy": mean_bool([r["abstention_correct"] for r in rows]),
        "chunk_recall": mean_bool([r["chunk_recall"] for r in rows]),
        "doi_recall": mean_bool([r["doi_recall"] for r in rows]),
        "value_mention_rate": mean_num([r["value_mention_rate"] for r in rows]),
        "must_mention_rate": mean_num([r["must_mention_rate"] for r in rows]),
        "mean_latency_ms": mean_num([r["latency_ms"] for r in rows]),
        "by_category": {},
    }
    for cat, rs in sorted(by_cat.items()):
        summary["by_category"][cat] = {
            "n": len(rs),
            "pass_rate": mean_bool([r["passed"] for r in rs]),
            "abstention_accuracy": mean_bool([r["abstention_correct"] for r in rs]),
            "chunk_recall": mean_bool([r["chunk_recall"] for r in rs]),
            "doi_recall": mean_bool([r["doi_recall"] for r in rs]),
            "value_mention_rate": mean_num([r["value_mention_rate"] for r in rs]),
            "must_mention_rate": mean_num([r["must_mention_rate"] for r in rs]),
        }

    report = {"summary": summary, "details": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
