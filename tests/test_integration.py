"""Full pipeline scenario tests."""

import pytest

from kindalive.engine.chemicals import Chemical
from kindalive.engine.clock import ManualClock
from kindalive.engine.neurochemical_engine import NeurochemicalEngine
from kindalive.engine.seed_chemistry import SeedChemistry
from kindalive.personality.presets import PERSONALITY_PRESETS, get_seed
from kindalive.robot import Robot
from tests.conftest import ImpulseFactory


def test_couch_scenario():
    clock = ManualClock()
    seed = SeedChemistry.from_dict(PERSONALITY_PRESETS["cheerful"])
    robot = Robot(engine=NeurochemicalEngine(clock=clock, seed=seed), personality="cheerful")

    robot.receive_impulses(ImpulseFactory.presence_nearby())
    clock.advance(minutes=5)
    robot.receive_impulses(ImpulseFactory.weather_sunny())
    clock.advance(minutes=10)
    robot.receive_impulses(ImpulseFactory.market_up(percent=1.2))
    clock.advance(minutes=2)
    robot.receive_impulses(ImpulseFactory.sports_goal(team="home"))

    # Advance a tiny bit for sustained impulses to drip
    robot.advance(dt=1.0)

    emotions = robot.current_emotions()
    assert emotions.happiness > emotions.anxiety
    assert emotions.bonding > 0.2
    assert emotions.anxiety < 0.2


def test_bad_day_scenario():
    clock = ManualClock()
    robot = Robot(engine=NeurochemicalEngine(clock=clock))

    robot.receive_impulses(ImpulseFactory.market_crash(percent=5.0))
    robot.receive_impulses(ImpulseFactory.weather_storm())
    robot.advance(dt=1.0)

    emotions = robot.current_emotions()
    assert emotions.anxiety > emotions.happiness or emotions.sadness > emotions.happiness


def test_mood_recovery():
    clock = ManualClock()
    robot = Robot(engine=NeurochemicalEngine(clock=clock))

    robot.receive_impulses(ImpulseFactory.market_crash(percent=5.0))
    robot.advance(dt=1.0)
    stressed_cortisol = robot.current_chemicals().get(Chemical.CORTISOL)

    clock.advance(hours=4)
    robot.advance(dt=1.0)

    recovered_cortisol = robot.current_chemicals().get(Chemical.CORTISOL)
    assert recovered_cortisol < stressed_cortisol


def test_personality_affects_reaction():
    stoic_seed = get_seed("stoic")
    anxious_seed = get_seed("anxious")

    stoic = Robot(engine=NeurochemicalEngine(clock=ManualClock(), seed=stoic_seed), personality="stoic")
    anxious = Robot(engine=NeurochemicalEngine(clock=ManualClock(), seed=anxious_seed), personality="anxious")

    impulses = ImpulseFactory.market_crash(percent=3.0)
    stoic.receive_impulses(impulses)
    anxious.receive_impulses(impulses)

    stoic.advance(dt=1.0)
    anxious.advance(dt=1.0)

    assert anxious.current_emotions().anxiety > stoic.current_emotions().anxiety


@pytest.mark.asyncio
async def test_couch_expression():
    clock = ManualClock()
    seed = SeedChemistry.from_dict(PERSONALITY_PRESETS["cheerful"])
    robot = Robot(engine=NeurochemicalEngine(clock=clock, seed=seed), personality="cheerful")

    robot.receive_impulses(ImpulseFactory.presence_nearby())
    clock.advance(minutes=5)
    robot.receive_impulses(ImpulseFactory.sports_goal())
    robot.advance(dt=15.0)  # let adrenaline drip

    text = await robot.express()
    # Should be some positive expression
    lower = text.lower()
    assert any(w in lower for w in ["happy", "excited", "euphoric", "calm"])
