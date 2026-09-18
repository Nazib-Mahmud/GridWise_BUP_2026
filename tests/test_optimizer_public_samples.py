import json
from pathlib import Path

from app.guardrails import validate_and_normalize_interpretations
from app.models import OptimizeRequest
from app.optimizer import optimize_energy

ROOT = Path(__file__).resolve().parents[1]


def test_all_public_reference_costs():
    data = json.loads((ROOT / "samples" / "public_sample_cases.json").read_text())
    for case in data["cases"]:
        request = OptimizeRequest.model_validate(case["input"])
        directives = validate_and_normalize_interpretations(
            case["expected_output"]["directive_interpretation"],
            len(request.operator_notes),
            request.battery,
        )
        result = optimize_energy(request, directives)
        expected_cost = float(case["expected_output"]["total_cost_bdt"])
        assert abs(result["total_cost_bdt"] - expected_cost) <= 0.01, (
            case["id"], result["total_cost_bdt"], expected_cost
        )
