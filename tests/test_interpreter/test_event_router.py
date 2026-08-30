"""Tests for RealtimeRouter (the collapsed event_router)."""

from __future__ import annotations

import pytest

from kindalive.engine.chemicals import Chemical
from kindalive.engine.impulse import ChemicalImpulse
from kindalive.interpreter.event_router import RealtimeRouter
from kindalive.interpreter.llm_interpreter import (
    LLMInterpreter,
    MockLLMBackend,
)
from kindalive.interpreter.prompt_builder import RobotContext
from kindalive.interpreter.text_input import UserText


def _context() -> RobotContext:
    return RobotContext(
        personality_name="default",
        affinity=1.0,
        dominant_emotion="calm",
        chemical_summary="gaba=0.40",
    )


@pytest.mark.asyncio
async def test_route_calls_llm_and_applies_impulses():
    backend = MockLLMBackend()
    backend.enqueue_impulses([
        {"chemical": "dopamine", "delta": 0.3},
    ])
    interp = LLMInterpreter(backend=backend, use_cache=False)
    applied: list[list[ChemicalImpulse]] = []

    router = RealtimeRouter(
        interpreter=interp,
        context_fn=_context,
        apply_fn=applied.append,
    )

    await router.route(UserText(summary="you won the lottery"))

    assert len(applied) == 1
    assert applied[0][0].chemical == Chemical.DOPAMINE
    assert applied[0][0].delta == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_route_with_no_impulses_does_not_apply():
    backend = MockLLMBackend()
    backend.enqueue_impulses([])  # empty array — neutral event
    interp = LLMInterpreter(backend=backend, use_cache=False)
    applied: list[list[ChemicalImpulse]] = []

    router = RealtimeRouter(
        interpreter=interp,
        context_fn=_context,
        apply_fn=applied.append,
    )

    await router.route(UserText(summary="nothing happened"))

    assert applied == []
