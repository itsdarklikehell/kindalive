"""Tests for emotion projection from chemical state."""

from kindalive.emotions.projection import EmotionProjection
from kindalive.engine.chemicals import Chemical, ChemicalState


def test_high_dopamine_serotonin_means_happiness():
    state = ChemicalState()
    state.set(Chemical.DOPAMINE, 0.9)
    state.set(Chemical.SEROTONIN, 0.9)
    state.set(Chemical.CORTISOL, 0.0)
    emotions = EmotionProjection.compute(state)
    assert emotions.happiness > 0.6


def test_high_cortisol_low_gaba_means_anxiety():
    state = ChemicalState()
    state.set(Chemical.CORTISOL, 0.9)
    state.set(Chemical.ADRENALINE, 0.7)
    state.set(Chemical.GABA, 0.0)
    state.set(Chemical.SEROTONIN, 0.1)
    emotions = EmotionProjection.compute(state)
    assert emotions.anxiety > emotions.calm
    assert emotions.anxiety > emotions.happiness


def test_default_state_is_neutral():
    state = ChemicalState()
    emotions = EmotionProjection.compute(state)
    assert 0.15 < emotions.happiness < 0.5
    assert 0.15 < emotions.calm < 0.5
    assert 0.15 < emotions.bonding < 0.5
    assert emotions.anxiety < 0.15
    assert emotions.anger < 0.15
    assert emotions.excitement < 0.3
    for value in emotions.as_dict().values():
        assert value < 0.6

    # Sadness must not dominate at baseline — the robot should be quietly
    # calm/neutral when nothing is happening, not perpetually sad.
    dominant_name, _ = emotions.dominant()
    assert dominant_name != "sadness", (
        f"sadness should not dominate at baseline but was {dominant_name}"
    )
    assert emotions.sadness < 0.3


def test_all_high_means_euphoria():
    state = ChemicalState()
    state.set(Chemical.DOPAMINE, 1.0)
    state.set(Chemical.ENDORPHINS, 1.0)
    state.set(Chemical.ADRENALINE, 1.0)
    state.set(Chemical.OXYTOCIN, 1.0)
    emotions = EmotionProjection.compute(state)
    assert emotions.euphoria > 0.8


def test_low_everything_positive_means_sadness():
    state = ChemicalState()
    state.set(Chemical.DOPAMINE, 0.0)
    state.set(Chemical.SEROTONIN, 0.0)
    state.set(Chemical.OXYTOCIN, 0.0)
    state.set(Chemical.CORTISOL, 0.5)
    emotions = EmotionProjection.compute(state)
    assert emotions.sadness > 0.7


def test_dominant_emotion():
    state = ChemicalState()
    state.set(Chemical.CORTISOL, 0.9)
    state.set(Chemical.ADRENALINE, 0.8)
    state.set(Chemical.GABA, 0.0)
    state.set(Chemical.SEROTONIN, 0.0)
    emotions = EmotionProjection.compute(state)
    name, _ = emotions.dominant()
    assert name in ("anxiety", "anger", "excitement", "sadness")


def test_emotions_always_in_0_1():
    # Extreme state: all at 0
    state = ChemicalState()
    for chem in Chemical:
        state.set(chem, 0.0)
    emotions = EmotionProjection.compute(state)
    for val in emotions.as_dict().values():
        assert 0.0 <= val <= 1.0

    # Extreme state: all at 1
    for chem in Chemical:
        state.set(chem, 1.0)
    emotions = EmotionProjection.compute(state)
    for val in emotions.as_dict().values():
        assert 0.0 <= val <= 1.0
