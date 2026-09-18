from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .models import DirectiveInterpretation, OptimizeRequest


class OptimizationError(RuntimeError):
    pass


@dataclass
class AppliedConstraints:
    effective_solar: list[float]
    reserve_floor: list[float]
    no_charge: set[int]
    no_discharge: set[int]
    grid_caps: dict[int, float]


def compile_directives(
    request: OptimizeRequest,
    directives: list[DirectiveInterpretation],
) -> AppliedConstraints:
    by_hour = {h.hour: h for h in request.hours}
    effective_solar = [float(by_hour[h].solar_kwh) for h in range(24)]
    reserve_floor = [float(request.battery.minimum_energy_kwh)] * 24
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    grid_caps: dict[int, float] = {}

    # For overlapping absolute caps/floors, satisfy every directive by applying the strictest value.
    solar_factor = [1.0] * 24

    for directive in directives:
        if not directive.applies or directive.directive_type == "no_op":
            continue
        adj = directive.structured_adjustment or {}
        hours = adj["hours"]

        if directive.directive_type == "solar_reduction":
            factor = float(adj["factor"])
            for h in hours:
                solar_factor[h] = min(solar_factor[h], factor)
        elif directive.directive_type == "minimum_battery_reserve":
            reserve = float(adj["minimum_energy_kwh"])
            for h in hours:
                reserve_floor[h] = max(reserve_floor[h], reserve)
        elif directive.directive_type == "no_charge_window":
            no_charge.update(hours)
        elif directive.directive_type == "no_discharge_window":
            no_discharge.update(hours)
        elif directive.directive_type == "max_grid_window":
            cap = float(adj["max_grid_kwh"])
            for h in hours:
                grid_caps[h] = min(grid_caps.get(h, math.inf), cap)

    for h in range(24):
        effective_solar[h] *= solar_factor[h]

    return AppliedConstraints(
        effective_solar=effective_solar,
        reserve_floor=reserve_floor,
        no_charge=no_charge,
        no_discharge=no_discharge,
        grid_caps=grid_caps,
    )


def optimize_energy(
    request: OptimizeRequest,
    directives: list[DirectiveInterpretation],
) -> dict:
    hours = {h.hour: h for h in request.hours}
    battery = request.battery
    applied = compile_directives(request, directives)

    # Variable block per hour: grid, solar_used, charge, discharge, battery_energy_after
    n = 24 * 5

    def idx(h: int, offset: int) -> int:
        return h * 5 + offset

    GRID, SOLAR, CHARGE, DISCHARGE, ENERGY = range(5)

    objective = np.zeros(n, dtype=float)
    bounds: list[tuple[float | None, float | None]] = [(0, None)] * n

    for h in range(24):
        objective[idx(h, GRID)] = float(hours[h].tariff_bdt_per_kwh)
        bounds[idx(h, GRID)] = (0.0, applied.grid_caps.get(h))
        bounds[idx(h, SOLAR)] = (0.0, applied.effective_solar[h])
        bounds[idx(h, CHARGE)] = (
            0.0,
            0.0 if h in applied.no_charge else float(battery.max_charge_kwh_per_hour),
        )
        bounds[idx(h, DISCHARGE)] = (
            0.0,
            0.0 if h in applied.no_discharge else float(battery.max_discharge_kwh_per_hour),
        )
        bounds[idx(h, ENERGY)] = (applied.reserve_floor[h], float(battery.capacity_kwh))

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []

    for h in range(24):
        # Energy balance: grid + solar + discharge = demand + charge
        row = np.zeros(n, dtype=float)
        row[idx(h, GRID)] = 1.0
        row[idx(h, SOLAR)] = 1.0
        row[idx(h, CHARGE)] = -1.0
        row[idx(h, DISCHARGE)] = 1.0
        a_eq.append(row)
        b_eq.append(float(hours[h].demand_kwh))

        # Battery transition: E_after = E_before + charge - discharge
        row = np.zeros(n, dtype=float)
        row[idx(h, ENERGY)] = 1.0
        row[idx(h, CHARGE)] = -1.0
        row[idx(h, DISCHARGE)] = 1.0
        if h == 0:
            rhs = float(battery.initial_energy_kwh)
        else:
            row[idx(h - 1, ENERGY)] = -1.0
            rhs = 0.0
        a_eq.append(row)
        b_eq.append(rhs)

    # End-of-day battery neutrality.
    row = np.zeros(n, dtype=float)
    row[idx(23, ENERGY)] = 1.0
    a_eq.append(row)
    b_eq.append(float(battery.initial_energy_kwh))

    result = linprog(
        c=objective,
        A_eq=np.array(a_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise OptimizationError("No feasible GridWise schedule could be found")

    x = result.x
    plan: list[dict] = []
    previous_energy = float(battery.initial_energy_kwh)

    for h in range(24):
        solar = _clean(x[idx(h, SOLAR)])
        charge = _clean(x[idx(h, CHARGE)])
        discharge = _clean(x[idx(h, DISCHARGE)])

        # LPs can return a mathematically redundant simultaneous charge/discharge pair.
        # Net it out without changing battery state, grid energy, cost, or feasibility.
        if charge > 1e-8 and discharge > 1e-8:
            cancel = min(charge, discharge)
            charge -= cancel
            discharge -= cancel
            charge = _clean(charge)
            discharge = _clean(discharge)

        energy = previous_energy + charge - discharge
        grid = float(hours[h].demand_kwh) + charge - solar - discharge
        grid = _clean(grid)

        if charge > 1e-7:
            action = "charge"
            battery_kwh = charge
        elif discharge > 1e-7:
            action = "discharge"
            battery_kwh = discharge
        else:
            action = "idle"
            battery_kwh = 0.0

        entry = {
            "hour": h,
            "grid_kwh": _round(grid),
            "solar_used_kwh": _round(solar),
            "battery_action": action,
            "battery_kwh": _round(battery_kwh),
            "battery_energy_after_kwh": _round(energy),
        }
        plan.append(entry)
        previous_energy = energy

    # Recalculate aggregate values only from the returned hourly_plan.
    total_grid = sum(p["grid_kwh"] for p in plan)
    total_cost = sum(
        p["grid_kwh"] * float(hours[p["hour"]].tariff_bdt_per_kwh) for p in plan
    )
    peak_grid = max(p["grid_kwh"] for p in plan)

    _validate_returned_plan(request, directives, plan)

    return {
        "hourly_plan": plan,
        "total_grid_kwh": _round(total_grid),
        "total_cost_bdt": _round(total_cost),
        "peak_grid_kwh": _round(peak_grid),
    }


def _validate_returned_plan(
    request: OptimizeRequest,
    directives: list[DirectiveInterpretation],
    plan: list[dict],
) -> None:
    tol = 0.01
    hours = {h.hour: h for h in request.hours}
    applied = compile_directives(request, directives)
    previous = float(request.battery.initial_energy_kwh)

    if len(plan) != 24 or [p["hour"] for p in plan] != list(range(24)):
        raise OptimizationError("Internal replay failed: hourly_plan shape")

    for p in plan:
        h = p["hour"]
        grid = p["grid_kwh"]
        solar = p["solar_used_kwh"]
        amount = p["battery_kwh"]
        action = p["battery_action"]
        energy = p["battery_energy_after_kwh"]

        if min(grid, solar, amount, energy) < -tol:
            raise OptimizationError("Internal replay failed: negative energy value")
        if solar > applied.effective_solar[h] + tol:
            raise OptimizationError("Internal replay failed: solar overuse")
        if action == "charge":
            charge, discharge = amount, 0.0
            if h in applied.no_charge:
                raise OptimizationError("Internal replay failed: no-charge directive")
            if amount > request.battery.max_charge_kwh_per_hour + tol:
                raise OptimizationError("Internal replay failed: charge rate")
        elif action == "discharge":
            charge, discharge = 0.0, amount
            if h in applied.no_discharge:
                raise OptimizationError("Internal replay failed: no-discharge directive")
            if amount > request.battery.max_discharge_kwh_per_hour + tol:
                raise OptimizationError("Internal replay failed: discharge rate")
        else:
            charge = discharge = 0.0
            if amount > tol:
                raise OptimizationError("Internal replay failed: idle battery amount")

        expected_energy = previous + charge - discharge
        if abs(expected_energy - energy) > tol:
            raise OptimizationError("Internal replay failed: battery transition")
        if energy < applied.reserve_floor[h] - tol or energy > request.battery.capacity_kwh + tol:
            raise OptimizationError("Internal replay failed: battery bounds")
        if h in applied.grid_caps and grid > applied.grid_caps[h] + tol:
            raise OptimizationError("Internal replay failed: grid cap")

        lhs = grid + solar + discharge
        rhs = float(hours[h].demand_kwh) + charge
        if abs(lhs - rhs) > tol:
            raise OptimizationError("Internal replay failed: hourly energy balance")
        previous = energy

    if abs(previous - request.battery.initial_energy_kwh) > tol:
        raise OptimizationError("Internal replay failed: end-of-day battery neutrality")


def _clean(value: float) -> float:
    value = float(value)
    if abs(value) < 1e-7:
        return 0.0
    return value


def _round(value: float) -> float:
    value = round(float(value), 6)
    if abs(value) < 5e-7:
        return 0.0
    return value
