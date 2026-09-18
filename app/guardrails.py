from __future__ import annotations

import math
from typing import Any

from .models import BatteryInput, DirectiveInterpretation

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


class GuardrailError(ValueError):
    pass


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardrailError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise GuardrailError(f"{name} must be finite")
    return value


def _hours(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) == 0:
        raise GuardrailError("hours must be a non-empty array")
    parsed: list[int] = []
    for h in value:
        if isinstance(h, bool) or not isinstance(h, int) or not 0 <= h <= 23:
            raise GuardrailError("each hour must be an integer from 0 through 23")
        parsed.append(h)
    if len(set(parsed)) != len(parsed):
        raise GuardrailError("hours must not contain duplicates")
    return sorted(parsed)


def validate_and_normalize_interpretations(
    raw: Any,
    note_count: int,
    battery: BatteryInput,
) -> list[DirectiveInterpretation]:
    if isinstance(raw, dict):
        raw = raw.get("directive_interpretation")
    if not isinstance(raw, list):
        raise GuardrailError("LLM output must contain a directive_interpretation array")
    if len(raw) != note_count:
        raise GuardrailError("LLM must return exactly one interpretation per operator note")

    normalized: list[DirectiveInterpretation] = []
    seen_indices: set[int] = set()

    for item in raw:
        if not isinstance(item, dict):
            raise GuardrailError("each directive_interpretation entry must be an object")

        required = {
            "note_index",
            "applies",
            "directive_type",
            "structured_adjustment",
            "explanation",
        }
        if set(item.keys()) != required:
            raise GuardrailError("directive_interpretation entry has wrong fields")

        idx = item["note_index"]
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < note_count:
            raise GuardrailError("note_index is invalid")
        if idx in seen_indices:
            raise GuardrailError("duplicate note_index")
        seen_indices.add(idx)

        applies = item["applies"]
        if not isinstance(applies, bool):
            raise GuardrailError("applies must be boolean")

        dtype = item["directive_type"]
        if dtype not in ALLOWED_TYPES:
            raise GuardrailError("unsupported directive_type")

        explanation = item["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise GuardrailError("explanation must be a non-empty string")
        explanation = explanation.strip()

        adj = item["structured_adjustment"]

        if dtype == "no_op":
            if applies is not False or adj is not None:
                raise GuardrailError("no_op requires applies=false and structured_adjustment=null")
            normalized_adj = None
        else:
            if applies is not True or not isinstance(adj, dict):
                raise GuardrailError("non-no_op directives require applies=true and an adjustment object")

            if dtype == "solar_reduction":
                if set(adj.keys()) != {"hours", "factor"}:
                    raise GuardrailError("solar_reduction adjustment must contain hours and factor")
                factor = _number(adj["factor"], "factor")
                if not 0 <= factor <= 1:
                    raise GuardrailError("solar factor must be between 0 and 1")
                normalized_adj = {"hours": _hours(adj["hours"]), "factor": factor}

            elif dtype == "minimum_battery_reserve":
                if set(adj.keys()) != {"hours", "minimum_energy_kwh"}:
                    raise GuardrailError(
                        "minimum_battery_reserve adjustment must contain hours and minimum_energy_kwh"
                    )
                reserve = _number(adj["minimum_energy_kwh"], "minimum_energy_kwh")
                if reserve < 0 or reserve > battery.capacity_kwh:
                    raise GuardrailError("battery reserve must be within battery capacity")
                normalized_adj = {
                    "hours": _hours(adj["hours"]),
                    "minimum_energy_kwh": reserve,
                }

            elif dtype in {"no_charge_window", "no_discharge_window"}:
                if set(adj.keys()) != {"hours"}:
                    raise GuardrailError(f"{dtype} adjustment must contain only hours")
                normalized_adj = {"hours": _hours(adj["hours"])}

            elif dtype == "max_grid_window":
                if set(adj.keys()) != {"hours", "max_grid_kwh"}:
                    raise GuardrailError("max_grid_window adjustment must contain hours and max_grid_kwh")
                cap = _number(adj["max_grid_kwh"], "max_grid_kwh")
                if cap < 0:
                    raise GuardrailError("max_grid_kwh must be non-negative")
                normalized_adj = {"hours": _hours(adj["hours"]), "max_grid_kwh": cap}

            else:  # pragma: no cover
                raise GuardrailError("unsupported directive_type")

        normalized.append(
            DirectiveInterpretation(
                note_index=idx,
                applies=applies,
                directive_type=dtype,
                structured_adjustment=normalized_adj,
                explanation=explanation,
            )
        )

    normalized.sort(key=lambda x: x.note_index)
    if [x.note_index for x in normalized] != list(range(note_count)):
        raise GuardrailError("note_index entries must cover 0..N-1 exactly once")
    return normalized
