#!/usr/bin/env python3
"""
Create a Markdown summary report from polymd_eval_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    """Format a metric fraction as a percentage string."""
    if x is None:
        return "n/a"
    return f"{100*x:.1f}%"


def main():
    """Create a Markdown benchmark report from an evaluation JSON file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default="polymd_eval_report.json")
    ap.add_argument("--out", default="reports/polymd_eval_report.md")
    args = ap.parse_args()

    with open(args.eval_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    s = report["summary"]

    lines = []
    lines.append("# PolyMD Chatbot Benchmark Report\n")
    lines.append("## Overall\n")
    lines.append(f"- Gold items: **{s['n_gold']}**")
    lines.append(f"- Predictions: **{s['n_predictions']}**")
    lines.append(f"- Overall pass rate: **{pct(s['overall_pass_rate'])}**")
    lines.append(f"- Abstention accuracy: **{pct(s['abstention_accuracy'])}**")
    lines.append(f"- Chunk recall: **{pct(s['chunk_recall'])}**")
    lines.append(f"- DOI recall: **{pct(s['doi_recall'])}**")
    lines.append(f"- Value mention rate: **{pct(s['value_mention_rate'])}**")
    lines.append(f"- Must-mention rate: **{pct(s['must_mention_rate'])}**")
    if s.get("mean_latency_ms") is not None:
        lines.append(f"- Mean latency: **{s['mean_latency_ms']:.1f} ms**")
    lines.append("\n## By category\n")
    lines.append(
        "| Category | n | Pass | Abstention | Chunk recall | DOI recall | "
        "Value mention | Must mention |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cat, row in sorted(s["by_category"].items()):
        lines.append(
            f"| {cat} | {row['n']} | {pct(row['pass_rate'])} | "
            f"{pct(row['abstention_accuracy'])} | "
            f"{pct(row['chunk_recall'])} | {pct(row['doi_recall'])} | "
            f"{pct(row['value_mention_rate'])} | {pct(row['must_mention_rate'])} |"
        )

    lines.append("\n## Interpretation guide\n")
    lines.append("- Low chunk recall means the retriever is not finding the expected evidence.")
    lines.append(
        "- High chunk recall but low value/must-mention rates means generation "
        "is not using evidence faithfully."
    )
    lines.append("- Low DOI recall means answers are not properly grounded to source papers.")
    lines.append(
        "- Low abstention accuracy means the chatbot is either hallucinating "
        "unsupported values or refusing answerable questions."
    )
    lines.append(
        "- The `comparison` category should be manually inspected for "
        "same-property comparisons and no cross-property mixing."
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
