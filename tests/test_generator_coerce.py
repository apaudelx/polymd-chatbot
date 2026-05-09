"""Generator coercion rules (no live OpenAI calls)."""

from src.generator import Generator

TEST_API_KEY = "test-dummy-key-for-coerce-tests"


def test_coerce_abstained_strips_parametric_answer():
    g = Generator(api_key=TEST_API_KEY)
    out = g._coerce_structured_response(
        {
            "answer": "PLGA is a copolymer because lactic and glycolic acids.",
            "evidence_snippets": ["snippet"],
            "doi_links": ["https://doi.org/10.0000/example"],
            "confidence": 0.63,
            "abstained": True,
            "abstention_reason": "Context lacks PLGA homo/copolymer wording.",
        },
        fallback_confidence=0.5,
    )
    assert out["abstained"] is True
    assert "PLGA" not in out["answer"]
    assert out["evidence_snippets"] == ["snippet"]
    assert out["doi_links"] == ["https://doi.org/10.0000/example"]


def test_coerce_not_abstained_keeps_answer():
    g = Generator(api_key=TEST_API_KEY)
    out = g._coerce_structured_response(
        {
            "answer": "Density is 1.0 g/cm³ per context.",
            "evidence_snippets": ["x"],
            "doi_links": [],
            "confidence": 0.8,
            "abstained": False,
            "abstention_reason": "",
        },
        fallback_confidence=0.5,
    )
    assert out["abstained"] is False
    assert "Density" in out["answer"]
    assert out["evidence_snippets"] == ["x"]
