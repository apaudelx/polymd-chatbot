import gzip
import json
from pathlib import Path

from src.lexical_index import INDEX_FORMAT, read_lexical_index


def test_read_lexical_index_roundtrip(tmp_path: Path, monkeypatch):
    from src import config

    monkeypatch.setattr(config.Config, "COLLECTION_NAME", "test_coll")

    path = tmp_path / "lexical_index.json.gz"
    payload = {
        "format": INDEX_FORMAT,
        "collection_name": "test_coll",
        "n_docs": 2,
        "ids": ["1", "2"],
        "documents": ["hello world", "pmma density"],
        "metadatas": [{"chunk_type": "paper"}, {"chunk_type": "property", "polymer_name": "PMMA"}],
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded = read_lexical_index(path, 2, "test_coll")
    assert loaded is not None
    docs, ids, metas = loaded
    assert docs == payload["documents"]
    assert ids == payload["ids"]
    assert metas == payload["metadatas"]


def test_read_lexical_index_rejects_count_mismatch(tmp_path: Path, monkeypatch):
    from src import config

    monkeypatch.setattr(config.Config, "COLLECTION_NAME", "test_coll")

    path = tmp_path / "lexical_index.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(
            {
                "format": INDEX_FORMAT,
                "collection_name": "test_coll",
                "n_docs": 2,
                "ids": ["1"],
                "documents": ["a"],
                "metadatas": [{}],
            },
            f,
        )
    assert read_lexical_index(path, 2, "test_coll") is None


def test_read_lexical_index_rejects_wrong_collection(tmp_path: Path):
    path = tmp_path / "lexical_index.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(
            {
                "format": INDEX_FORMAT,
                "collection_name": "other",
                "n_docs": 1,
                "ids": ["1"],
                "documents": ["a"],
                "metadatas": [{}],
            },
            f,
        )
    assert read_lexical_index(path, 1, "expected") is None
