import pytest

from app.guardrails import GuardrailError, validate_and_normalize_interpretations
from app.models import BatteryInput

BATTERY = BatteryInput(
    capacity_kwh=200,
    initial_energy_kwh=100,
    minimum_energy_kwh=40,
    max_charge_kwh_per_hour=50,
    max_discharge_kwh_per_hour=50,
)


def test_no_op_semantics():
    result = validate_and_normalize_interpretations(
        [
            {
                "note_index": 0,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "Unrelated to energy scheduling.",
            }
        ],
        1,
        BATTERY,
    )
    assert result[0].directive_type == "no_op"


def test_hours_are_sorted_deterministically():
    result = validate_and_normalize_interpretations(
        [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": [5, 3, 4]},
                "explanation": "Charging unavailable.",
            }
        ],
        1,
        BATTERY,
    )
    assert result[0].structured_adjustment["hours"] == [3, 4, 5]


def test_rejects_invalid_solar_factor():
    with pytest.raises(GuardrailError):
        validate_and_normalize_interpretations(
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": [12], "factor": 1.5},
                    "explanation": "Bad factor.",
                }
            ],
            1,
            BATTERY,
        )
