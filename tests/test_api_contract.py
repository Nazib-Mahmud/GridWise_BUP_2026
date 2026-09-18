from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_contract():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_invalid_request_is_controlled_400():
    r = client.post("/optimize-energy", json={"scenario_id": "bad"})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_request"


def test_optimize_response_contract_with_validated_interpretation(monkeypatch):
    import json
    from pathlib import Path
    from app.guardrails import validate_and_normalize_interpretations
    from app.main import interpreter
    from app.models import OptimizeRequest

    root = Path(__file__).resolve().parents[1]
    pack = json.loads((root / "samples" / "public_sample_cases.json").read_text())
    case = pack["cases"][0]
    req = OptimizeRequest.model_validate(case["input"])
    expected_directives = validate_and_normalize_interpretations(
        case["expected_output"]["directive_interpretation"],
        len(req.operator_notes),
        req.battery,
    )

    async def fake_interpret(notes, battery):
        return expected_directives

    monkeypatch.setattr(interpreter, "interpret", fake_interpret)
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200
    data = r.json()
    assert data["scenario_id"] == case["id"]
    assert len(data["hourly_plan"]) == 24
    assert abs(data["total_cost_bdt"] - case["expected_output"]["total_cost_bdt"]) <= 0.01
    assert set(data) == {
        "scenario_id",
        "directive_interpretation",
        "hourly_plan",
        "total_grid_kwh",
        "total_cost_bdt",
        "peak_grid_kwh",
        "plan_summary",
    }
