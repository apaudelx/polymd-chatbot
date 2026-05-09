from src import controller as controller_module
from src.controller import ToolController


def test_natural_language_structure_uses_pubchem_resolution(monkeypatch):
    calls = []

    def fake_execute_tool(tool_name, args):
        calls.append((tool_name, args))
        if tool_name == "resolve_name_to_smiles":
            return {
                "ok": True,
                "result": {
                    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "source": "PubChem",
                },
            }
        if tool_name == "render_molecule_2d":
            return {
                "ok": True,
                "result": {
                    "input_smiles": args["smiles"],
                    "output_path": "/tmp/aspirin.png",
                },
            }
        raise AssertionError(f"Unexpected tool: {tool_name}")

    monkeypatch.setattr(controller_module, "execute_tool", fake_execute_tool)

    handled, response = ToolController().try_handle("Show me the structure of aspirin")

    assert handled is True
    assert calls[0] == (
        "resolve_name_to_smiles",
        {"name": "aspirin", "allow_alias_fallback": False},
    )
    assert calls[1][0] == "render_molecule_2d"
    assert calls[1][1]["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert "Source: PubChem" in response["answer"]


def test_natural_language_draws_smiles_directly(monkeypatch):
    def fake_execute_tool(tool_name, args):
        assert tool_name == "render_molecule_2d"
        assert args["smiles"] == "C=CC1=CC=CC=C1"
        return {
            "ok": True,
            "result": {
                "input_smiles": args["smiles"],
                "output_path": "/tmp/styrene.png",
            },
        }

    monkeypatch.setattr(controller_module, "execute_tool", fake_execute_tool)

    handled, response = ToolController().try_handle("What does C=CC1=CC=CC=C1 look like?")

    assert handled is True
    assert response["tool"] == "rdkit_render"
    assert "render the SMILES" in response["answer"]


def test_natural_language_draw_trigger_routes_without_rdkit_prefix(monkeypatch):
    def fake_execute_tool(tool_name, args):
        if tool_name == "resolve_name_to_smiles":
            return {
                "ok": True,
                "result": {
                    "canonical_smiles": "C=CC1=CC=CC=C1",
                    "source": "PubChem",
                },
            }
        if tool_name == "render_molecule_2d":
            return {
                "ok": True,
                "result": {
                    "input_smiles": args["smiles"],
                    "output_path": "/tmp/styrene.png",
                },
            }
        raise AssertionError(f"Unexpected tool: {tool_name}")

    monkeypatch.setattr(controller_module, "execute_tool", fake_execute_tool)

    handled, response = ToolController().try_handle("Draw styrene")

    assert handled is True
    assert response["tool"] == "rdkit_render"
    assert "Source: PubChem" in response["answer"]


def test_natural_language_descriptors_use_name_resolution(monkeypatch):
    calls = []

    def fake_execute_tool(tool_name, args):
        calls.append((tool_name, args))
        if tool_name == "resolve_name_to_smiles":
            return {
                "ok": True,
                "result": {
                    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "source": "PubChem",
                },
            }
        if tool_name == "molecular_descriptors":
            return {
                "ok": True,
                "result": {
                    "input_smiles": args["smiles"],
                    "canonical_smiles": args["smiles"],
                    "molecular_weight": 180.159,
                    "logp": 1.3101,
                    "h_donors": 1,
                    "h_acceptors": 4,
                    "tpsa": 63.6,
                },
            }
        raise AssertionError(f"Unexpected tool: {tool_name}")

    monkeypatch.setattr(controller_module, "execute_tool", fake_execute_tool)

    handled, response = ToolController().try_handle("Calculate descriptors for aspirin")

    assert handled is True
    assert calls[0][0] == "resolve_name_to_smiles"
    assert calls[1][0] == "molecular_descriptors"
    assert "MolWt" in response


def test_natural_language_pubchem_not_found_message_is_user_facing(monkeypatch):
    def fake_execute_tool(tool_name, args):
        assert tool_name == "resolve_name_to_smiles"
        return {
            "ok": False,
            "error": "PubChem did not find a SMILES record for 'polybutadine'.",
        }

    monkeypatch.setattr(controller_module, "execute_tool", fake_execute_tool)

    handled, response = ToolController().try_handle("Show me the structure of polybutadine")

    assert handled is True
    assert "Please check the spelling" in response
    assert "monomer, oligomer, repeat-unit" in response
    assert "HTTP Error" not in response
    assert "PUGREST.NotFound" not in response
    assert "polymer_structures.json" not in response


def test_property_question_with_structure_word_stays_in_rag_route():
    query = (
        "In the paper 'Research on structures_ mechanical properties_ and "
        "mechanical responses of TKX-50 and TKX-50 based PBX with molecular "
        "dynamics', what density is reported for Poly(vinylidene fluoride) "
        "using the DREIDING force field?"
    )

    handled, response = ToolController().try_handle(query)

    assert handled is False
    assert response is None
