from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.guardrails import validate_and_normalize_interpretations
from app.models import OptimizeRequest
from app.optimizer import optimize_energy

DATA = ROOT / "samples" / "public_sample_cases.json"


def main() -> None:
    pack = json.loads(DATA.read_text(encoding="utf-8"))
    passed = 0
    for case in pack["cases"]:
        request = OptimizeRequest.model_validate(case["input"])
        directives = validate_and_normalize_interpretations(
            case["expected_output"]["directive_interpretation"],
            len(request.operator_notes),
            request.battery,
        )
        result = optimize_energy(request, directives)
        expected = float(case["expected_output"]["total_cost_bdt"])
        diff = abs(result["total_cost_bdt"] - expected)
        ok = diff <= 0.01
        passed += int(ok)
        print(
            f"{case['id']}: {'PASS' if ok else 'FAIL'} | "
            f"cost={result['total_cost_bdt']} expected={expected} diff={diff:.6f}"
        )
    print(f"\nOptimizer public samples: {passed}/{len(pack['cases'])} passed")
    raise SystemExit(0 if passed == len(pack["cases"]) else 1)


if __name__ == "__main__":
    main()
