import json
from urllib.error import HTTPError

from src import rdkit_tools


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if isinstance(self.payload, str):
            return self.payload.encode("utf-8")
        return json.dumps(self.payload).encode("utf-8")


def test_name_resolution_uses_pubchem(monkeypatch):
    payload = {
        "PropertyTable": {
            "Properties": [
                {
                    "CanonicalSMILES": "PUBCHEM_SMILES",
                }
            ]
        }
    }

    monkeypatch.setattr(rdkit_tools, "urlopen", lambda request, timeout: FakeResponse(payload))
    monkeypatch.setattr(rdkit_tools, "_wait_for_pubchem_slot", lambda: None)

    out = rdkit_tools.exec_resolve_name_to_smiles(
        {"name": "aspirin", "allow_alias_fallback": False}
    )

    assert out["canonical_smiles"] == "PUBCHEM_SMILES"
    assert out["source"] == "PubChem"


def test_name_resolution_fails_when_pubchem_has_no_smiles(monkeypatch):
    monkeypatch.setattr(
        rdkit_tools,
        "urlopen",
        lambda request, timeout: FakeResponse({"PropertyTable": {"Properties": []}}),
    )
    monkeypatch.setattr(rdkit_tools, "_wait_for_pubchem_slot", lambda: None)

    try:
        rdkit_tools.exec_resolve_name_to_smiles(
            {"name": "unknown polymer", "allow_alias_fallback": True}
        )
    except ValueError as exc:
        assert "via PubChem" in str(exc)
    else:
        raise AssertionError("Expected PubChem-only resolution to fail")


def test_name_resolution_sanitizes_pubchem_not_found(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            url="https://pubchem.ncbi.nlm.nih.gov/",
            code=404,
            msg="PUGREST.NotFound",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(rdkit_tools, "urlopen", fake_urlopen)
    monkeypatch.setattr(rdkit_tools, "_wait_for_pubchem_slot", lambda: None)

    try:
        rdkit_tools.exec_resolve_name_to_smiles(
            {"name": "polybutadine", "allow_alias_fallback": False}
        )
    except ValueError as exc:
        message = str(exc)
        assert "PubChem did not find a SMILES record" in message
        assert "HTTP Error" not in message
        assert "PUGREST.NotFound" not in message
    else:
        raise AssertionError("Expected PubChem 404 to become a clean ValueError")


def test_name_resolution_reports_pubchem_503_as_temporary(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            url="https://pubchem.ncbi.nlm.nih.gov/",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(rdkit_tools, "urlopen", fake_urlopen)
    monkeypatch.setattr(rdkit_tools, "_wait_for_pubchem_slot", lambda: None)
    monkeypatch.setattr(rdkit_tools.time, "sleep", lambda seconds: None)

    try:
        rdkit_tools.exec_resolve_name_to_smiles(
            {"name": "aspirin", "allow_alias_fallback": False}
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "temporarily unavailable" in message
        assert "HTTP Error" not in message
    else:
        raise AssertionError("Expected PubChem 503 to become a clean RuntimeError")


def test_name_resolution_reports_non_json_response_cleanly(monkeypatch):
    monkeypatch.setattr(
        rdkit_tools,
        "urlopen",
        lambda request, timeout: FakeResponse("<html>maintenance</html>"),
    )
    monkeypatch.setattr(rdkit_tools, "_wait_for_pubchem_slot", lambda: None)

    try:
        rdkit_tools.exec_resolve_name_to_smiles(
            {"name": "aspirin", "allow_alias_fallback": False}
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "non-JSON response" in message
        assert "<html>" not in message
    else:
        raise AssertionError("Expected non-JSON PubChem response to fail cleanly")


def test_pubchem_rate_limit_waits_between_requests(monkeypatch):
    sleeps = []
    times = iter([10.0, 10.1, 10.23])

    monkeypatch.setattr(rdkit_tools.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(rdkit_tools.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(rdkit_tools, "_LAST_PUBCHEM_REQUEST_AT", 0.0)

    rdkit_tools._wait_for_pubchem_slot()
    rdkit_tools._wait_for_pubchem_slot()

    assert [round(value, 2) for value in sleeps] == [0.12]
