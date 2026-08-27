"""Tests for the 12-muscle FaceProjection layer."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from kindalive.engine.chemicals import Chemical, ChemicalState
from kindalive.expression.face import (
    FACE_WEIGHTS,
    FaceState,
    project_face,
)


def _state_with(**levels: float) -> ChemicalState:
    """Build a ChemicalState and overwrite specific levels."""
    state = ChemicalState()
    for chem in Chemical:
        if chem.value in levels:
            state.set(chem, levels[chem.value])
    return state


# ── Bounds ────────────────────────────────────────────────────────


def test_face_state_dataclass_has_12_muscles():
    state = ChemicalState()
    face = project_face(state)
    assert isinstance(face, FaceState)
    assert len(face.as_dict()) == 12


@settings(max_examples=200, deadline=None)
@given(
    dopamine=st.floats(min_value=0.0, max_value=1.0),
    serotonin=st.floats(min_value=0.0, max_value=1.0),
    oxytocin=st.floats(min_value=0.0, max_value=1.0),
    testosterone=st.floats(min_value=0.0, max_value=1.0),
    cortisol=st.floats(min_value=0.0, max_value=1.0),
    adrenaline=st.floats(min_value=0.0, max_value=1.0),
    endorphins=st.floats(min_value=0.0, max_value=1.0),
    gaba=st.floats(min_value=0.0, max_value=1.0),
)
def test_face_values_in_unit_interval(
    dopamine, serotonin, oxytocin, testosterone,
    cortisol, adrenaline, endorphins, gaba,
):
    """Property: any chemical state in [0,1]^8 → all 12 muscles in [0,1]."""
    state = _state_with(
        dopamine=dopamine, serotonin=serotonin, oxytocin=oxytocin,
        testosterone=testosterone, cortisol=cortisol, adrenaline=adrenaline,
        endorphins=endorphins, gaba=gaba,
    )
    face = project_face(state)
    for name, value in face.as_dict().items():
        assert 0.0 <= value <= 1.0, f"{name}={value}"


# ── Baseline neutrality ───────────────────────────────────────────


def test_baseline_state_is_neutral():
    """At species-default baselines, no muscle dominates."""
    face = project_face(ChemicalState())
    values = face.as_dict()
    # No muscle should be strongly activated at rest.
    assert all(v < 0.40 for v in values.values()), values
    # Cheek raise should be modest at baseline.
    assert face.cheek_raise < 0.30
    # Brow_lower should not register frustration at rest.
    assert face.brow_lower < 0.30


def test_calm_baseline_does_not_grin_or_scowl():
    """The smile-minus-frown swing at rest stays near zero."""
    face = project_face(ChemicalState())
    smile = face.lip_corner_pull - face.lip_corner_depress
    assert -0.10 <= smile <= 0.25, f"smile factor at baseline = {smile:.3f}"


# ── Targeted activations ──────────────────────────────────────────


def test_intense_joy_activates_smile_muscles():
    state = _state_with(
        dopamine=1.0, endorphins=1.0, oxytocin=0.7,
        serotonin=0.7, cortisol=0.05,
    )
    face = project_face(state)
    assert face.cheek_raise > 0.7
    assert face.lip_corner_pull > 0.7
    assert face.lip_corner_depress < 0.10


def test_acute_stress_activates_anger_muscles():
    state = _state_with(
        cortisol=1.0, adrenaline=0.9, testosterone=0.7,
        gaba=0.05, dopamine=0.1,
    )
    face = project_face(state)
    assert face.brow_lower > 0.6
    assert face.eyelid_upper_raise > 0.7
    assert face.lip_press > 0.4


def test_sadness_via_deficit_activates_inner_brow_and_frown():
    """Dopamine/serotonin well below baseline (no cortisol spike) — the
    AU1 / AU15 pair (sadness) should fire even without stress."""
    state = _state_with(
        dopamine=0.0, serotonin=0.0, oxytocin=0.0,
        cortisol=0.05,
    )
    face = project_face(state)
    assert face.brow_inner_raise > 0.20, face.brow_inner_raise
    assert face.lip_corner_depress > 0.15, face.lip_corner_depress


def test_high_oxytocin_activates_lip_pucker():
    state = _state_with(oxytocin=1.0, endorphins=0.6)
    face = project_face(state)
    assert face.lip_pucker > 0.5


def test_acute_excitement_drops_jaw():
    state = _state_with(
        adrenaline=1.0, dopamine=0.8, endorphins=0.5,
        gaba=0.05, cortisol=0.05,
    )
    face = project_face(state)
    assert face.jaw_open > 0.5


# ── Weights table integrity ───────────────────────────────────────


def test_face_weights_table_covers_all_muscles():
    """Every FaceState field must appear in FACE_WEIGHTS."""
    face = project_face(ChemicalState())
    assert set(FACE_WEIGHTS.keys()) == set(face.as_dict().keys())


def test_face_weights_only_reference_known_chemicals():
    valid = set(Chemical)
    for muscle, terms in FACE_WEIGHTS.items():
        for term in terms:
            assert term.chemical in valid, (
                f"unknown chemical {term.chemical} in {muscle}"
            )


def test_top_n_returns_sorted_descending():
    face = project_face(_state_with(
        dopamine=1.0, endorphins=1.0, adrenaline=0.9, gaba=0.05,
    ))
    top = face.top_n(3)
    assert len(top) == 3
    values = [v for _, v in top]
    assert values == sorted(values, reverse=True)
