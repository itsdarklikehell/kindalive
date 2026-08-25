"""Tests for state persistence — serialization round-trip."""

import json
import tempfile
from pathlib import Path

from kindalive.engine.chemicals import Chemical
from kindalive.engine.clock import ManualClock
from kindalive.engine.impulse import ChemicalImpulse
from kindalive.engine.neurochemical_engine import NeurochemicalEngine
from kindalive.engine.seed_chemistry import SeedChemistry
from kindalive.persistence.state_store import (
    deserialize_engine,
    load_state,
    save_state,
    serialize_engine,
)


def test_round_trip_levels():
    """Chemical levels survive serialization round-trip."""
    clock = ManualClock()
    engine = NeurochemicalEngine(clock=clock)

    # Set non-default levels
    engine.state.set(Chemical.DOPAMINE, 0.75)
    engine.state.set(Chemical.CORTISOL, 0.6)
    engine.state.set(Chemical.GABA, 0.1)

    data = serialize_engine(engine)

    # Restore into a fresh engine
    engine2 = NeurochemicalEngine(clock=ManualClock())
    deserialize_engine(data, engine2)

    assert abs(engine2.state.get(Chemical.DOPAMINE) - 0.75) < 1e-9
    assert abs(engine2.state.get(Chemical.CORTISOL) - 0.6) < 1e-9
    assert abs(engine2.state.get(Chemical.GABA) - 0.1) < 1e-9


def test_round_trip_baselines():
    """Drifted baselines persist."""
    clock = ManualClock()
    engine = NeurochemicalEngine(clock=clock)

    # Simulate baseline drift
    engine.state.set_baseline(Chemical.CORTISOL, 0.35)

    data = serialize_engine(engine)
    engine2 = NeurochemicalEngine(clock=ManualClock())
    deserialize_engine(data, engine2)

    assert abs(engine2.state.baseline(Chemical.CORTISOL) - 0.35) < 1e-9


def test_round_trip_sustained_impulses():
    """Active sustained impulses survive round-trip."""
    clock = ManualClock()
    engine = NeurochemicalEngine(clock=clock)

    imp = ChemicalImpulse(
        Chemical.OXYTOCIN, delta=0.2, duration_seconds=300,
        source_id="presence:nearby", source_label="Owner home",
    )
    engine.apply_impulse(imp)

    data = serialize_engine(engine)
    assert len(data["sustained_impulses"]) == 1

    engine2 = NeurochemicalEngine(clock=ManualClock())
    deserialize_engine(data, engine2)
    assert len(engine2._sustained) == 1
    assert engine2._sustained[0].impulse.chemical == Chemical.OXYTOCIN
    assert engine2._sustained[0].remaining_seconds == 300.0


def test_save_and_load_file():
    """Save to file and load back."""
    clock = ManualClock()
    engine = NeurochemicalEngine(clock=clock)
    engine.state.set(Chemical.ADRENALINE, 0.8)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        save_state(engine, path)

        # Verify file content
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert abs(data["levels"]["adrenaline"] - 0.8) < 1e-9

        # Load into fresh engine
        engine2 = NeurochemicalEngine(clock=ManualClock())
        load_state(engine2, path)
        assert abs(engine2.state.get(Chemical.ADRENALINE) - 0.8) < 1e-9
    finally:
        path.unlink(missing_ok=True)


def test_serialize_includes_seed():
    """Seed chemistry is included in serialized data."""
    seed = SeedChemistry.from_dict({
        "baselines": {"dopamine": 0.5},
        "half_life_multipliers": {"adrenaline": 0.7},
        "interaction_scale": 1.3,
    })
    engine = NeurochemicalEngine(clock=ManualClock(), seed=seed)
    data = serialize_engine(engine)

    assert "seed" in data
    assert data["seed"]["interaction_scale"] == 1.3


def test_stressed_robot_stays_stressed():
    """A robot that was stressed before save should still be stressed after load."""
    clock = ManualClock()
    engine = NeurochemicalEngine(clock=clock)

    # Inject stress
    engine.apply_impulse(ChemicalImpulse(Chemical.CORTISOL, delta=0.4))
    engine.apply_impulse(ChemicalImpulse(Chemical.ADRENALINE, delta=0.3))
    stressed_cortisol = engine.state.get(Chemical.CORTISOL)

    data = serialize_engine(engine)

    engine2 = NeurochemicalEngine(clock=ManualClock())
    deserialize_engine(data, engine2)

    assert abs(engine2.state.get(Chemical.CORTISOL) - stressed_cortisol) < 1e-9
