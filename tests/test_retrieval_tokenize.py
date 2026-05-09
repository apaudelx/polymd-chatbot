from src.retriever import Retriever


def test_tokenize_strips_trailing_punctuation_from_words():
    toks = Retriever._tokenize("what is polymer?")
    assert "polymer" in toks
    assert "polymer?" not in " ".join(toks)


def test_tokenize_polymers_plural():
    toks = Retriever._tokenize("what are polymers?")
    assert "polymers" in toks


def test_normalize_retrieval_query_merges_polymers_and_what_are():
    assert Retriever._normalize_retrieval_query("What are polymers?") == "what is polymer?"


def test_normalize_retrieval_query_empty():
    assert Retriever._normalize_retrieval_query("") == ""
    assert Retriever._normalize_retrieval_query("   ") == ""


def test_merge_retrieval_results_prefers_higher_score():
    a = {
        "ids": ["1", "2"],
        "documents": ["d1", "d2"],
        "metadatas": [{"k": 1}, {"k": 2}],
        "scores": [0.5, 0.4],
    }
    b = {
        "ids": ["2", "3"],
        "documents": ["d2b", "d3"],
        "metadatas": [{"k": 22}, {"k": 3}],
        "scores": [0.9, 0.1],
    }
    out = Retriever._merge_retrieval_results(a, b, k=2)
    assert out["ids"] == ["2", "1"]
    assert out["scores"][0] == 0.9
    assert out["documents"][0] == "d2b"


def test_normalize_scores_keeps_all_zero_scores_zero():
    assert Retriever._normalize_scores([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
