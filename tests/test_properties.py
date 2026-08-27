"""Property-based tests using Hypothesis."""

from hypothesis import given, settings
from hypothesis import strategies as st

from kindalive.emotions.projection import EmotionProjection
from kindalive.engine.chemicals import Chemical, ChemicalState
from kindalive.engine.clock import ManualClock
from kindalive.engine.neurochemical_engine import NeurochemicalEngine


@given(
    levels=st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=8, max_size=8,
    ),
    dt=st.floats(min_value=0.01, max_value=3600.0),
)
@settings(max_examples=200)
def test_chemicals_always_stay_in_range(levels, dt):
    engine = NeurochemicalEngine(clock=ManualClock())
    for chem, level in zip(Chemical, levels):
        engine.state.set(chem, level)
    engine.advance(dt=dt)
    for chem in Chemical:
        val = engine.state.get(chem)
        assert 0.0 <= val <= 1.0, f"{chem}: {val} after dt={dt}"


@given(
    levels=st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=8, max_size=8,
    ),
)
@settings(max_examples=200)
def test_emotions_always_in_range(levels):
    state = ChemicalState()
    for chem, level in zip(Chemical, levels):
        state.set(chem, level)
    emotions = EmotionProjection.compute(state)
    for name, value in emotions.as_dict().items():
        assert 0.0 <= value <= 1.0, f"{name}: {value}"
