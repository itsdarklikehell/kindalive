"""Tests for ImpulseValidator."""

import pytest

from kindalive.engine.chemicals import Chemical
from kindalive.interpreter.validator import (
    MAX_DELTA,
    MAX_DURATION,
    MIN_DELTA,
    ValidationError,
    validate_raw_impulses,
)


def test_valid_impulse():
    raw = [{"chemical": "dopamine", "delta": 0.2, "duration_seconds": 0}]
    result = validate_raw_impulses(raw)
    assert len(result) == 1
    assert result[0].chemical == Chemical.DOPAMINE
    assert result[0].delta == 0.2


def test_delta_clamped_high():
    raw = [{"chemical": "dopamine", "delta": 1.5}]
    result = validate_raw_impulses(raw)
    assert result[0].delta == MAX_DELTA


def test_delta_clamped_low():
    raw = [{"chemical": "dopamine", "delta": -1.5}]
    result = validate_raw_impulses(raw)
    assert result[0].delta == MIN_DELTA


def test_duration_clamped():
    raw = [{"chemical": "dopamine", "delta": 0.1, "duration_seconds": 999}]
    result = validate_raw_impulses(raw)
    assert result[0].duration_seconds == MAX_DURATION


def test_unknown_chemical_skipped():
    raw = [
        {"chemical": "unobtanium", "delta": 0.1},
        {"chemical": "dopamine", "delta": 0.2},
    ]
    result = validate_raw_impulses(raw)
    assert len(result) == 1
    assert result[0].chemical == Chemical.DOPAMINE


def test_missing_delta_raises():
    raw = [{"chemical": "dopamine"}]
    with pytest.raises(ValidationError, match="missing 'delta'"):
        validate_raw_impulses(raw)


def test_missing_chemical_raises():
    raw = [{"delta": 0.1}]
    with pytest.raises(ValidationError, match="missing or invalid 'chemical'"):
        validate_raw_impulses(raw)


def test_not_a_list_raises():
    with pytest.raises(ValidationError, match="Expected list"):
        validate_raw_impulses({"chemical": "dopamine", "delta": 0.1})  # type: ignore[arg-type]


def test_too_many_impulses_raises():
    raw = [{"chemical": "dopamine", "delta": 0.1}] * 9
    with pytest.raises(ValidationError, match="Too many impulses"):
        validate_raw_impulses(raw)


def test_source_fields_preserved():
    raw = [{
        "chemical": "dopamine",
        "delta": 0.2,
        "source_id": "sports:goal:BOS",
        "source_label": "Bruins goal",
    }]
    result = validate_raw_impulses(raw)
    assert result[0].source_id == "sports:goal:BOS"
    assert result[0].source_label == "Bruins goal"


def test_case_insensitive_chemical():
    raw = [{"chemical": "DOPAMINE", "delta": 0.1}]
    result = validate_raw_impulses(raw)
    assert result[0].chemical == Chemical.DOPAMINE


def test_defaults_for_optional_fields():
    raw = [{"chemical": "dopamine", "delta": 0.1}]
    result = validate_raw_impulses(raw)
    assert result[0].duration_seconds == 0.0
    assert result[0].source_id == ""
    assert result[0].source_label == ""


def test_all_8_chemicals_at_once():
    """Validator should accept a response that touches all 8 chemicals."""
    raw = [
        {"chemical": "dopamine", "delta": 0.1},
        {"chemical": "serotonin", "delta": 0.05},
        {"chemical": "oxytocin", "delta": -0.1},
        {"chemical": "testosterone", "delta": 0.15},
        {"chemical": "cortisol", "delta": -0.2},
        {"chemical": "adrenaline", "delta": 0.3},
        {"chemical": "endorphins", "delta": 0.2},
        {"chemical": "gaba", "delta": -0.05},
    ]
    result = validate_raw_impulses(raw)
    assert len(result) == 8
    chemicals = {imp.chemical for imp in result}
    assert chemicals == set(Chemical)
