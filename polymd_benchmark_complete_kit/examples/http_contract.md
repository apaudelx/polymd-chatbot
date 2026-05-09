# HTTP Adapter Contract

If you run your chatbot as an HTTP service, expose a POST endpoint like:

```http
POST /query
Content-Type: application/json
```

Request:

```json
{
  "question": "What is the density of PMMA?",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Response:

```json
{
  "answer": "The dataset reports ...",
  "retrieved_chunk_ids": ["546c153ec4ca"],
  "retrieved_dois": ["https://doi.org/10.1016/j.polymer.2019.121570"],
  "abstained": false,
  "history_content": {
    "answer": "The dataset reports ...",
    "doi_links": ["https://doi.org/10.1016/j.polymer.2019.121570"],
    "abstained": false
  }
}
```

`history_content` is optional, but recommended for contextual follow-up
benchmarks. When present, the benchmark runner stores it in the next turn's chat
history instead of storing only the plain answer string.

Then run:

```bash
python scripts/run_benchmark.py \
  --mode http \
  --endpoint http://localhost:8000/query \
  --gold data/polymd_benchmark_golden_set.jsonl \
  --out predictions.jsonl
```
