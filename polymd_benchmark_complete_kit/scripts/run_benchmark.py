#!/usr/bin/env python3
"""
Run the PolyMD benchmark against your chatbot.

Supported modes:
1. HTTP mode:
   Your app exposes an endpoint that accepts {"question": "...", "history": [...]} and
   returns JSON with at least {"answer": "..."}.
   Example:
     python scripts/run_benchmark.py --mode http --endpoint http://localhost:8000/query

2. Command mode:
   Your app can be called as a shell command that receives the question as an argument
   and prints the answer to stdout.
   Example:
     python scripts/run_benchmark.py --mode command --command "python ask.py"

3. Custom module mode:
   Implement examples/custom_adapter.py::answer_question(question, history) and call:
     python scripts/run_benchmark.py --mode module --module-path examples/custom_adapter.py

Output:
  predictions.jsonl with one row per benchmark item:
  {
    "id": "polymd_0001",
    "answer": "...",
    "retrieved_chunk_ids": [...],
    "retrieved_dois": [...],
    "abstained": false,
    "latency_ms": 1234
  }
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load newline-delimited JSON benchmark records."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def infer_abstained(answer: str) -> bool:
    """Infer whether an answer is an abstention from common refusal phrases."""
    text = str(answer or "")
    return any(re.search(p, text, flags=re.I) for p in ABSTAIN_PATTERNS)


def split_contextual_turns(item: Dict[str, Any]) -> List[str]:
    """Use explicit turns for contextual-followup tests; otherwise one question."""
    exp = item.get("expected", {})
    turns = exp.get("turns")
    if isinstance(turns, list) and turns:
        return [
            t["question"] for t in turns if isinstance(t, dict) and t.get("question")
        ]
    return [item["question"]]


def call_http(
    endpoint: str,
    question: str,
    history: List[Dict[str, str]],
    timeout_s: float,
) -> Dict[str, Any]:
    """Call a benchmark-compatible HTTP chatbot endpoint."""
    payload = json.dumps({"question": question, "history": history}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if isinstance(data, str):
        return {"answer": data}
    return data


def call_command(
    command: str,
    question: str,
    history: List[Dict[str, str]],
    timeout_s: float,
) -> Dict[str, Any]:
    """Call a shell command adapter and parse its response."""
    env = dict(os.environ)
    env["POLYMD_BENCH_HISTORY"] = json.dumps(history, ensure_ascii=False)
    cmd = shlex.split(command) + [question]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    if proc.returncode != 0:
        return {"answer": "", "error": proc.stderr.strip(), "abstained": True}
    stdout = proc.stdout.strip()
    try:
        data = json.loads(stdout)
        if isinstance(data, str):
            return {"answer": data}
        return data
    except json.JSONDecodeError:
        return {"answer": stdout}


def load_module_adapter(module_path: str):
    """Load an `answer_question` function from a Python adapter file."""
    path = Path(module_path)
    spec = importlib.util.spec_from_file_location("polymd_custom_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "answer_question"):
        raise RuntimeError(
            "Custom module must define answer_question(question, history)."
        )
    return mod.answer_question


def call_module(
    adapter,
    question: str,
    history: List[Dict[str, str]],
    timeout_s: float,
) -> Dict[str, Any]:
    """Call a Python adapter function and normalize its output."""
    _ = timeout_s
    result = adapter(question, history)
    if isinstance(result, str):
        return {"answer": result}
    if isinstance(result, dict):
        return result
    return {"answer": str(result)}


def main() -> None:
    """Run benchmark questions through the selected adapter mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/polymd_benchmark_golden_set.jsonl")
    parser.add_argument("--out", default="predictions.jsonl")
    parser.add_argument("--mode", choices=["http", "command", "module"], required=True)
    parser.add_argument("--endpoint", help="HTTP endpoint for --mode http")
    parser.add_argument(
        "--command",
        help='Command for --mode command, e.g. "python ask.py"',
    )
    parser.add_argument("--module-path", help="Python file path for --mode module")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    items = load_jsonl(gold_path)
    if args.limit:
        items = items[: args.limit]

    done = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        for row in load_jsonl(out_path):
            done.add(row.get("id"))

    adapter = None
    if args.mode == "module":
        if not args.module_path:
            raise SystemExit("--module-path is required for module mode.")
        adapter = load_module_adapter(args.module_path)
    elif args.mode == "http" and not args.endpoint:
        raise SystemExit("--endpoint is required for http mode.")
    elif args.mode == "command" and not args.command:
        raise SystemExit("--command is required for command mode.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            if item["id"] in done:
                continue

            history: List[Dict[str, str]] = []
            final_result: Dict[str, Any] = {}
            total_latency = 0.0
            errors = []

            for question in split_contextual_turns(item):
                start = time.time()
                try:
                    if args.mode == "http":
                        result = call_http(
                            args.endpoint,
                            question,
                            history,
                            args.timeout_s,
                        )
                    elif args.mode == "command":
                        result = call_command(
                            args.command,
                            question,
                            history,
                            args.timeout_s,
                        )
                    else:
                        result = call_module(adapter, question, history, args.timeout_s)
                except Exception as exc:
                    result = {"answer": "", "error": repr(exc), "abstained": True}
                latency_ms = (time.time() - start) * 1000
                total_latency += latency_ms

                answer = str(result.get("answer", ""))
                assistant_content = result.get("history_content", answer)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": assistant_content})
                final_result = result
                if result.get("error"):
                    errors.append(result["error"])

            row = {
                "id": item["id"],
                "answer": str(final_result.get("answer", "")),
                "retrieved_chunk_ids": final_result.get(
                    "retrieved_chunk_ids",
                    final_result.get("chunk_ids", []),
                ),
                "retrieved_dois": final_result.get(
                    "retrieved_dois",
                    final_result.get("dois", []),
                ),
                "abstained": final_result.get(
                    "abstained",
                    infer_abstained(str(final_result.get("answer", ""))),
                ),
                "latency_ms": round(total_latency, 2),
            }
            if errors:
                row["errors"] = errors

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(items)}] {item['id']} -> {row['latency_ms']} ms")


if __name__ == "__main__":
    main()
