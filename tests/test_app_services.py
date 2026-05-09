from src.app_services import (
    _is_contextual_follow_up,
    _is_global_dataset_query,
    _is_global_force_field_query,
    _is_global_paper_count_query,
    _retrieve_source_locked_follow_up,
    _source_lock_context,
    _try_force_field_inventory_response,
    _try_global_dataset_response,
    _try_structured_density_response,
    contextualize_query,
)


def test_contextualize_force_field_follow_up_from_density_query():
    history = [
        {"role": "user", "content": "What is density of PMMA?"},
        {
            "role": "assistant",
            "content": {
                "answer": "The density of PMMA is 1.1 g/cm3.",
                "evidence_snippets": [
                    "Polymer: Poly(methyl methacrylate) Property: density = 1.1"
                ],
                "doi_links": ["https://doi.org/10.1016/j.polymer.2019.121570"],
            },
        },
    ]

    out = contextualize_query(
        "using which force field did they get that density?",
        history,
    )

    assert "Referenced polymer/entity: PMMA" in out
    assert "Referenced property: density" in out
    assert "Referenced value(s): 1.1" in out
    assert "Previous DOI(s): https://doi.org/10.1016/j.polymer.2019.121570" in out


def test_contextualize_leaves_standalone_query_unchanged():
    query = "What force fields were used for PET?"

    assert contextualize_query(query, []) == query


def test_global_dataset_query_does_not_trigger_contextual_follow_up():
    query = "what force fields are studied in this dataset?"

    assert _is_global_dataset_query(query)
    assert _is_global_force_field_query(query)
    assert not _is_contextual_follow_up(query)


def test_global_paper_count_query_does_not_trigger_contextual_follow_up():
    query = "how many papers does this dataset study?"

    assert _is_global_dataset_query(query)
    assert _is_global_paper_count_query(query)
    assert not _is_contextual_follow_up(query)


def test_global_force_field_response_uses_full_dataset():
    out = _try_global_dataset_response("what force fields are studied in this dataset?")

    assert out is not None
    assert out["abstained"] is False
    assert "unique force-field entries" in out["answer"]
    assert "COMPASS" in out["answer"]


def test_force_field_inventory_response_uses_requested_polymer_scope():
    response, stats = _try_force_field_inventory_response(
        "What force fields are reported for Poly(ethylene terephthalate) in the dataset?"
    )

    assert response["abstained"] is False
    assert "Poly(ethylene terephthalate)" in response["answer"]
    assert "CG1 model" in response["answer"]
    assert "CG2 model" in response["answer"]
    assert "CG3 model" in response["answer"]
    assert "https://doi.org/10.1063/1.5145142" in response["doi_links"]
    assert "7e71ac21a746" in stats["retrieved_chunk_ids"]
    assert "https://doi.org/10.3390/polym14061161" in stats["retrieved_dois"]


def test_force_field_inventory_response_does_not_catch_global_query():
    out = _try_force_field_inventory_response(
        "what force fields are studied in this dataset?"
    )

    assert out is None


def test_global_paper_count_response_uses_dataset_metadata():
    out = _try_global_dataset_response("how many papers does this dataset has?")

    assert out is not None
    assert out["abstained"] is False
    assert "198 paper-level records" in out["answer"]
    assert "198 unique DOI/paper records" in out["answer"]
    assert "1,125 property records" in out["answer"]


def test_structured_density_response_uses_prior_threshold():
    history = [
        {"role": "user", "content": "what is the density of polystyrene?"},
        {
            "role": "assistant",
            "content": {
                "answer": "The density of polystyrene is 0.96 g/cm3.",
                "evidence_snippets": ["Polymer: Polystyrene Property: density = 0.96"],
                "doi_links": ["https://doi.org/10.1021/acs.jctc.0c00954"],
            },
        },
    ]

    out = _try_structured_density_response(
        "Are there any other polymers in this dataset with a higher density than that?",
        history,
    )

    assert out is not None
    assert out["abstained"] is False
    assert "higher than 0.96" in out["answer"]
    assert "0.7587" not in out["answer"]


def test_structured_density_response_abstains_without_threshold():
    out = _try_structured_density_response(
        "Are there any other polymers in this dataset with a higher density than that?",
        [],
    )

    assert out is not None
    assert out["abstained"] is True
    assert "referenced density value" in out["answer"]


def test_source_lock_context_uses_previous_dois_and_values():
    history = [
        {"role": "user", "content": "What is the density of PMMA?"},
        {
            "role": "assistant",
            "content": {
                "answer": "The density of PMMA is 1.1 g/cm3.",
                "evidence_snippets": [
                    "Polymer: Poly(methyl methacrylate) Property: density = 1.1"
                ],
                "doi_links": ["https://doi.org/10.1016/j.polymer.2019.121570"],
            },
        },
    ]

    lock = _source_lock_context(
        "using which force field did they get that density for the PMMA?",
        history,
    )

    assert lock["entity"] == "PMMA"
    assert lock["property_key"] == "density"
    assert lock["doi_links"] == ["https://doi.org/10.1016/j.polymer.2019.121570"]
    assert "1.1" in lock["prior_values"]


def test_source_lock_context_ignores_global_dataset_query():
    history = [
        {"role": "user", "content": "What is the density of PMMA?"},
        {
            "role": "assistant",
            "content": {
                "answer": "The density of PMMA is 1.1 g/cm3.",
                "evidence_snippets": [
                    "Polymer: Poly(methyl methacrylate) Property: density = 1.1"
                ],
                "doi_links": ["https://doi.org/10.1016/j.polymer.2019.121570"],
            },
        },
    ]

    assert _source_lock_context("what force fields are studied in this dataset?", history) == {}
    assert contextualize_query("what force fields are studied in this dataset?", history) == (
        "what force fields are studied in this dataset?"
    )
    assert _source_lock_context("how many papers does this dataset study?", history) == {}
    assert contextualize_query("how many papers does this dataset has?", history) == (
        "how many papers does this dataset has?"
    )


def test_source_lock_context_handles_broad_follow_up_wording():
    history = [
        {"role": "user", "content": "What is the glass transition temperature of PMMA?"},
        {
            "role": "assistant",
            "content": {
                "answer": "The glass transition temperature is 459 K.",
                "evidence_snippets": [
                    "Polymer: Poly(methyl methacrylate) "
                    "Property: glass_transition_temp = 459"
                ],
                "doi_links": ["https://doi.org/10.1016/j.polymer.2019.121570"],
            },
        },
    ]

    lock = _source_lock_context("what force field was used for that property?", history)

    assert lock["property_key"] == "glass_transition_temp"
    assert lock["entity"] == "PMMA"
    assert "459" in lock["prior_values"]


def test_contextualize_handles_short_anaphora():
    history = [
        {"role": "user", "content": "What is viscosity of polymer X?"},
        {
            "role": "assistant",
            "content": {
                "answer": "The viscosity is 12.",
                "evidence_snippets": ["Polymer: polymer X Property: viscosity = 12"],
                "doi_links": ["https://doi.org/10.example/test"],
            },
        },
    ]

    out = contextualize_query("what about that one?", history)

    assert "Referenced polymer/entity: polymer X" in out
    assert "Referenced property: viscosity" in out


class FakeCollection:
    def get(self, where, include):
        assert where == {"doi": "https://doi.org/10.1016/j.polymer.2019.121570"}
        assert include == ["documents", "metadatas"]
        return {
            "ids": ["locked", "wrong_value"],
            "documents": [
                "Polymer: Poly(methyl methacrylate)\n"
                "Property: density = 1.1\n"
                "Force Field: OPLS-AA",
                "Polymer: Polymethyl Methacrylate\n"
                "Property: density = 1.156\n"
                "Force Field: Combined PCFF-TraPPE (UA)",
            ],
            "metadatas": [
                {
                    "chunk_type": "property",
                    "polymer_name": "Poly(methyl methacrylate)",
                    "property": "density",
                    "value": "1.1",
                    "doi": "https://doi.org/10.1016/j.polymer.2019.121570",
                },
                {
                    "chunk_type": "property",
                    "polymer_name": "Polymethyl Methacrylate",
                    "property": "density",
                    "value": "1.156",
                    "doi": "https://doi.org/10.1016/j.polymer.2019.121570",
                },
            ],
        }


class FakeVectorStore:
    collection = FakeCollection()


def test_source_locked_retrieval_keeps_previous_value():
    lock = {
        "entity": "PMMA",
        "property_key": "density",
        "doi_links": ["https://doi.org/10.1016/j.polymer.2019.121570"],
        "prior_values": ["1.1"],
        "topic": "force field",
    }

    results = _retrieve_source_locked_follow_up(FakeVectorStore(), lock, k=5)

    assert results["source_locked"] is True
    assert results["ids"] == ["locked"]
    assert "OPLS-AA" in results["documents"][0]
