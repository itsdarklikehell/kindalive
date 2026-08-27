"""Shared test fixtures and helpers."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

from kindalive.engine.chemicals import Chemical

# Coverage instrumentation makes individual hypothesis examples slow enough
# to trip the default 200ms deadline; the properties themselves are fast.
settings.register_profile(
    "ci", deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("ci")
from kindalive.engine.clock import ManualClock
from kindalive.engine.impulse import ChemicalImpulse
from kindalive.engine.neurochemical_engine import NeurochemicalEngine


@pytest.fixture
def manual_clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def engine(manual_clock: ManualClock) -> NeurochemicalEngine:
    return NeurochemicalEngine(clock=manual_clock)


class ImpulseFactory:
    """Convenience factory for common impulse patterns in engine-level tests."""

    @staticmethod
    def presence_nearby() -> list[ChemicalImpulse]:
        return [ChemicalImpulse(Chemical.OXYTOCIN, delta=0.3, duration_seconds=300)]

    @staticmethod
    def weather_sunny() -> list[ChemicalImpulse]:
        return [ChemicalImpulse(Chemical.SEROTONIN, delta=0.15)]

    @staticmethod
    def market_up(percent: float = 1.0) -> list[ChemicalImpulse]:
        return [ChemicalImpulse(Chemical.DOPAMINE, delta=0.1 + percent * 0.05)]

    @staticmethod
    def sports_goal(team: str = "home") -> list[ChemicalImpulse]:
        return [
            ChemicalImpulse(Chemical.DOPAMINE, delta=0.25),
            ChemicalImpulse(Chemical.ADRENALINE, delta=0.35),
        ]

    @staticmethod
    def market_crash(percent: float = 5.0) -> list[ChemicalImpulse]:
        return [
            ChemicalImpulse(Chemical.CORTISOL, delta=min(0.1 + percent * 0.05, 0.5)),
            ChemicalImpulse(Chemical.ADRENALINE, delta=0.25),
            ChemicalImpulse(Chemical.DOPAMINE, delta=-0.2),
        ]

    @staticmethod
    def weather_storm() -> list[ChemicalImpulse]:
        return [
            ChemicalImpulse(Chemical.CORTISOL, delta=0.15),
            ChemicalImpulse(Chemical.ADRENALINE, delta=0.1),
        ]
