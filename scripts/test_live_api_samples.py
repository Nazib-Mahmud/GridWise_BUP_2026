from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

DATA = ROOT / "samples" / "public_sample_cases.json"


def normalize_interpretations(items):
    return [
        {
            "note_index": x["note_index"],
            "applies": x["applies"],
            "directive_type": x["directive_type"],
            "structured_adjustment": x["structured_adjustment"],
        }
        for x in items
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between cases. Useful when validating on a provider free tier with low RPM quota.",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    pack = json.loads(DATA.read_text(encoding="utf-8"))
    with httpx.Client(timeout=35) as client:
        health = client.get(f"{base}/health")
        print("/health:", health.status_code, health.text)
        health.raise_for_status()

        passed = 0
        for i, case in enumerate(pack["cases"]):
            if i and args.delay > 0:
                time.sleep(args.delay)
            response = client.post(f"{base}/optimize-energy", json=case["input"])
            if response.status_code != 200:
                print(case["id"], "FAIL", response.status_code, response.text)
                continue
            got = response.json()
            expected = case["expected_output"]
            semantic_ok = normalize_interpretations(got["directive_interpretation"]) == normalize_interpretations(
                expected["directive_interpretation"]
            )
            cost_ok = abs(float(got["total_cost_bdt"]) - float(expected["total_cost_bdt"])) <= 0.01
            ok = semantic_ok and cost_ok
            passed += int(ok)
            print(
                f"{case['id']}: {'PASS' if ok else 'FAIL'} | "
                f"interpretation={semantic_ok} cost={got['total_cost_bdt']}"
            )

    print(f"\nLive API public samples: {passed}/{len(pack['cases'])} passed")
    if passed != len(pack["cases"]):
        print("Tip: inspect Render logs for '[GridWise][LLM]' lines. HTTP 429 means provider quota/rate-limit, not optimizer failure.")
    raise SystemExit(0 if passed == len(pack["cases"]) else 1)


if __name__ == "__main__":
    main()
