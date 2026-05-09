from src.entity_match import entity_match_in_text, tokenize_lower


def test_entity_match_rejects_substring_false_positive():
    assert not entity_match_in_text("competent polymer study", "pet")


def test_entity_match_accepts_whole_word():
    assert entity_match_in_text("PET is a polyester", "pet")
    assert entity_match_in_text("polyethylene terephthalate bottle", "polyethylene terephthalate")


def test_tokenize_lower_splits():
    assert tokenize_lower("A B") == ["a", "b"]
