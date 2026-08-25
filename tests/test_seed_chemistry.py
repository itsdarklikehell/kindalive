"""Tests for SeedChemistry — baseline configuration."""

from kindalive.engine.chemicals import Chemical
from kindalive.engine.clock import ManualClock
from kindalive.engine.neurochemical_engine import NeurochemicalEngine
from kindalive.engine.seed_chemistry import SeedChemistry


def test_species_defaults_produce_expected_baselines():
    seed = SeedChemistry.from_species_defaults()
    assert seed.baselines[Chemical.DOPAMINE] == 0.3
    assert seed.baselines[Chemical.SEROTONIN] == 0.5
    assert seed.baselines[Chemical.GABA] == 0.4
    assert len(seed.baselines) == 8


def test_partial_override_keeps_other_defaults():
    seed = SeedChemistry.from_dict({"baselines": {"dopamine": 0.5, "cortisol": 0.1}})
    assert seed.baselines[Chemical.DOPAMINE] == 0.5
    assert seed.baselines[Chemical.CORTISOL] == 0.1
    assert seed.baselines[Chemical.SEROTONIN] == 0.5  # untouched


def test_seed_determines_resting_emotions():
    from kindalive.emotions.projection import EmotionProjection

    clock = ManualClock()
    cheerful_seed = SeedChemistry.from_dict({
        "baselines": {"serotonin": 0.6, "dopamine": 0.4},
    })
    default_seed = SeedChemistry.from_species_defaults()

    cheerful = NeurochemicalEngine(clock=clock, seed=cheerful_seed)
    default = NeurochemicalEngine(clock=clock, seed=default_seed)

    ce = EmotionProjection.compute(cheerful.state)
    de = EmotionProjection.compute(default.state)
    assert ce.happiness > de.happiness


def test_half_life_multiplier_changes_decay_speed():
    clock = ManualClock()
    fast_seed = SeedChemistry.from_dict({
        "half_life_multipliers": {"adrenaline": 0.5},
    })
    normal_seed = SeedChemistry.from_species_defaults()

    fast = NeurochemicalEngine(clock=clock, seed=fast_seed)
    norm = NeurochemicalEngine(clock=clock, seed=normal_seed)

    fast.state.set(Chemical.ADRENALINE, 0.8)
    norm.state.set(Chemical.ADRENALINE, 0.8)

    fast.advance(dt=180.0)
    norm.advance(dt=180.0)

    assert fast.state.get(Chemical.ADRENALINE) < norm.state.get(Chemical.ADRENALINE)


def test_interaction_scale_amplifies_effects():
    amplified = SeedChemistry.from_dict({"interaction_scale": 2.0})
    normal = SeedChemistry.from_species_defaults()

    amp = NeurochemicalEngine(clock=ManualClock(), seed=amplified)
    nor = NeurochemicalEngine(clock=ManualClock(), seed=normal)

    # Use adrenaline + GABA interaction (GABA dampens adrenaline)
    # Set adrenaline high, let GABA dampen it — amplified should dampen faster
    amp.state.set(Chemical.ADRENALINE, 0.8)
    nor.state.set(Chemical.ADRENALINE, 0.8)
    amp.state.set(Chemical.GABA, 0.8)
    nor.state.set(Chemical.GABA, 0.8)

    amp.advance(dt=2.0)
    nor.advance(dt=2.0)

    # Amplified interactions: GABA dampens adrenaline harder
    assert amp.state.get(Chemical.ADRENALINE) < nor.state.get(Chemical.ADRENALINE)


def test_seed_persists_through_serialization():
    seed = SeedChemistry.from_dict({
        "baselines": {"dopamine": 0.5, "cortisol": 0.1},
        "half_life_multipliers": {"adrenaline": 0.7},
        "interaction_scale": 1.2,
    })
    data = seed.to_dict()
    restored = SeedChemistry.from_dict(data)
    assert restored.baselines[Chemical.DOPAMINE] == 0.5
    assert restored.half_life_multipliers[Chemical.ADRENALINE] == 0.7
    assert restored.interaction_scale == 1.2


def test_random_seed_stays_valid():
    import random

    for _ in range(50):
        seed = SeedChemistry.from_dict({
            "baselines": {
                chem.value: round(random.uniform(0.1, 0.7), 2)
                for chem in Chemical
            },
            "interaction_scale": round(random.uniform(0.7, 1.3), 2),
        })
        engine = NeurochemicalEngine(clock=ManualClock(), seed=seed)
        engine.advance(dt=3600.0)
        for chem in Chemical:
            level = engine.state.get(chem)
            assert 0.0 <= level <= 1.0, f"{chem}: {level}"
