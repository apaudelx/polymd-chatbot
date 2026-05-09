# PolyMD Chatbot Benchmark Complete Kit

This is a complete evaluation harness for the PolyMD chatbot.

It includes:

- a PolyMD-specific golden Q&A benchmark,
- a runner that can call chatbot through HTTP, shell command, or a Python adapter,
- an evaluator for retrieval, grounding, value correctness, and abstention,
- a Markdown report generator,
- integration examples and a grading rubric.

## Contents

```text
polymd_benchmark_complete_kit/
├── data/
│   ├── polymd_benchmark_golden_set.jsonl
│   └── polymd_benchmark_manifest.csv
├── scripts/
│   ├── run_benchmark.py
│   ├── evaluate_polymd_benchmark.py
│   └── make_report.py
├── examples/
│   ├── custom_adapter.py
│   └── http_contract.md
├── reports/
├── RUBRIC.md
├── requirements-benchmark.txt
└── README.md
```

## Benchmark composition

| Category | Count |
|---|---:|
| abstention | 20 |
| comparison | 20 |
| contextual_followup | 10 |
| dataset_summary | 10 |
| force_field_inventory | 20 |
| multi_value_lookup | 25 |
| paper_level_summary | 15 |
| single_property_lookup | 60 |
| source_lookup | 15 |

Total benchmark items: **195**

## Quick start

From inside this folder:

```bash
python scripts/run_benchmark.py \
  --mode module \
  --module-path examples/polymd_adapter.py \
  --gold data/polymd_benchmark_golden_set.jsonl \
  --out predictions.jsonl \
  --limit 5
```

`examples/polymd_adapter.py` calls this repository's `src.app_services.process_query()`
directly and preserves structured chat history for contextual follow-up tests.
`examples/custom_adapter.py` remains as a template for other chatbot projects.

Then evaluate:

```bash
python scripts/evaluate_polymd_benchmark.py \
  --gold data/polymd_benchmark_golden_set.jsonl \
  --pred predictions.jsonl \
  --out polymd_eval_report.json
```

Generate a Markdown report:

```bash
python scripts/make_report.py \
  --eval-json polymd_eval_report.json \
  --out reports/polymd_eval_report.md
```

## Integration options

### Option A: Python module adapter

Edit `examples/custom_adapter.py` and return:

```python
{
    "answer": "...",
    "retrieved_chunk_ids": ["..."],
    "retrieved_dois": ["https://doi.org/..."],
    "abstained": False
}
```

Run:

```bash
python scripts/run_benchmark.py \
  --mode module \
  --module-path examples/polymd_adapter.py \
  --out predictions.jsonl
```

### Option B: HTTP endpoint

Expose a POST endpoint matching `examples/http_contract.md`.

Run:

```bash
python scripts/run_benchmark.py \
  --mode http \
  --endpoint http://localhost:8000/query \
  --out predictions.jsonl
```

### Option C: Command-line wrapper

Create a script that accepts the question as the final argument and prints either plain text or JSON.

Run:

```bash
python scripts/run_benchmark.py \
  --mode command \
  --command "python ask.py" \
  --out predictions.jsonl
```

## Prediction format

Each prediction row should look like:

```json
{
  "id": "polymd_0001",
  "answer": "The dataset reports ...",
  "retrieved_chunk_ids": ["546c153ec4ca"],
  "retrieved_dois": ["https://doi.org/10.1016/j.polymer.2019.121570"],
  "abstained": false,
  "latency_ms": 1200
}
```

`retrieved_chunk_ids` are important because they allow you to separate retrieval failures from answer-generation failures.

## Metrics

The evaluator reports:

- overall pass rate,
- abstention accuracy,
- chunk recall,
- DOI recall,
- value mention rate,
- must-mention rate,
- mean latency,
- all metrics by benchmark category.
