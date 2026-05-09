import importlib.util
from pathlib import Path

from src.retriever import Retriever


def test_retrieval_stats_include_benchmark_ids_and_dois():
    retriever = Retriever.__new__(Retriever)
    stats = retriever.get_retrieval_stats(
        {
            "documents": ["doc"],
            "metadatas": [
                {
                    "chunk_type": "property",
                    "chunk_id": "abc123",
                    "doi": "10.1234/example",
                }
            ],
        }
    )

    assert stats["retrieved_chunk_ids"] == ["abc123"]
    assert stats["retrieved_dois"] == ["https://doi.org/10.1234/example"]


def test_polymd_benchmark_adapter_formats_response(monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "polymd_benchmark_complete_kit"
        / "examples"
        / "polymd_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("polymd_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    response = {
        "answer": "Density is 1.1 g/cm3.",
        "doi_links": ["https://doi.org/10.1000/example"],
        "abstained": False,
    }
    stats = {
        "retrieved_chunk_ids": ["chunk-1"],
        "retrieved_dois": ["https://doi.org/10.1000/example"],
    }

    monkeypatch.setattr(module, "_system", lambda: {"fake": True})
    monkeypatch.setattr(module, "process_query", lambda **kwargs: (response, stats))

    out = module.answer_question("What is density of PMMA?", [])

    assert out["answer"] == "Density is 1.1 g/cm3."
    assert out["retrieved_chunk_ids"] == ["chunk-1"]
    assert out["retrieved_dois"] == ["https://doi.org/10.1000/example"]
    assert out["abstained"] is False
    assert out["history_content"] == response
