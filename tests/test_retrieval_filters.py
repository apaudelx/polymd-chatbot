from src.retrieval_filters import drop_property_chunks


def test_drop_property_chunks_keeps_papers():
    results = {
        "documents": ["p1", "d1"],
        "metadatas": [{"chunk_type": "property"}, {"chunk_type": "paper"}],
        "ids": ["a", "b"],
        "scores": [0.9, 0.5],
    }
    out = drop_property_chunks(results)
    assert out["documents"] == ["d1"]
    assert out["metadatas"] == [{"chunk_type": "paper"}]
    assert out["ids"] == ["b"]
    assert out["scores"] == [0.5]


def test_drop_property_chunks_all_property_yields_empty():
    results = {
        "documents": ["p1"],
        "metadatas": [{"chunk_type": "property"}],
        "ids": ["a"],
        "scores": [0.9],
    }
    out = drop_property_chunks(results)
    assert out["documents"] == []
    assert out["metadatas"] == []
    assert out["ids"] == []
    assert out["scores"] == []
