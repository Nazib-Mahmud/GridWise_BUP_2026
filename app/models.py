from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _finite_nonnegative(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return value


class HourInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float

    @field_validator("hour")
    @classmethod
    def valid_hour(cls, v: int) -> int:
        if isinstance(v, bool) or not 0 <= v <= 23:
            raise ValueError("hour must be an integer from 0 through 23")
        return v

    @field_validator("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")
    @classmethod
    def valid_hour_numbers(cls, v: float, info):
        return _finite_nonnegative(v, info.field_name)


class BatteryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float

    @field_validator(
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
    )
    @classmethod
    def valid_battery_numbers(cls, v: float, info):
        return _finite_nonnegative(v, info.field_name)

    @model_validator(mode="after")
    def valid_state(self):
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if not self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh:
            raise ValueError(
                "initial_energy_kwh must be between minimum_energy_kwh and capacity_kwh"
            )
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    operator_notes: list[str]
    hours: list[HourInput]
    battery: BatteryInput

    @field_validator("scenario_id")
    @classmethod
    def strip_scenario_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("scenario_id cannot be empty")
        return v

    @field_validator("operator_notes")
    @classmethod
    def valid_notes(cls, notes: list[str]) -> list[str]:
        if not 1 <= len(notes) <= 3:
            raise ValueError("operator_notes must contain 1 to 3 notes")
        cleaned = []
        for note in notes:
            if not isinstance(note, str) or not note.strip():
                raise ValueError("every operator note must be a non-empty string")
            cleaned.append(note.strip())
        return cleaned

    @field_validator("hours")
    @classmethod
    def valid_hours(cls, hours: list[HourInput]) -> list[HourInput]:
        if len(hours) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        ids = [h.hour for h in hours]
        if sorted(ids) != list(range(24)):
            raise ValueError("hours must contain each hour 0 through 23 exactly once")
        return hours


DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None
    explanation: str


class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
