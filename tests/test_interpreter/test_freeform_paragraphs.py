"""Live-LLM calibration tests for the freeform-paragraph pipeline.

These hit the real Anthropic API and are gated on the ``llm`` marker
plus an ``ANTHROPIC_API_KEY`` environment variable. They protect
against regressions in the multi-fact preamble and the calibration
examples — the single most important output-quality surface in the
post-strip system.

Run with::

    ANTHROPIC_API_KEY=sk-ant-... pytest tests/ -m llm
"""

from __future__ import annotations

import os

import pytest

from kindalive.engine.chemicals import Chemical
from kindalive.engine.clock import ManualClock
from kindalive.robot import Robot

# `anthropic` is an optional dependency — skip the whole module if the
# SDK isn't installed, before importing AnthropicBackend at use time.
anthropic = pytest.importorskip("anthropic")

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    ),
]


def _robot() -> Robot:
    from kindalive.interpreter.anthropic_backend import AnthropicBackend
    return Robot(
        clock=ManualClock(),
        llm_backend=AnthropicBackend(api_key=os.environ["ANTHROPIC_API_KEY"]),
    )


def _chemicals(impulses) -> set[Chemical]:
    return {imp.chemical for imp in impulses}


@pytest.mark.asyncio
async def test_long_paragraph_winning_lottery():
    impulses = await _robot().interpret_text(
        "You won the lottery, your team won the championship, "
        "and your best friend is coming over to celebrate."
    )
    chems = _chemicals(impulses)
    assert Chemical.DOPAMINE in chems
    # Excitement chemicals should rise; cortisol should not.
    assert Chemical.CORTISOL not in chems or all(
        i.delta <= 0.05 for i in impulses if i.chemical == Chemical.CORTISOL
    )


@pytest.mark.asyncio
async def test_multi_fact_mixed_paragraph():
    """Four small positive facts should yield modest broad-positive lift."""
    impulses = await _robot().interpret_text(
        "It's Friday, finances are up, and tomorrow I have off."
    )
    chems = _chemicals(impulses)
    # At least one of the well-being chemicals should rise.
    assert chems & {
        Chemical.DOPAMINE,
        Chemical.SEROTONIN,
        Chemical.ENDORPHINS,
        Chemical.GABA,
    }
    # No emergency chemicals should fire on this paragraph.
    for imp in impulses:
        if imp.chemical in (Chemical.CORTISOL, Chemical.ADRENALINE):
            assert imp.delta <= 0.10


@pytest.mark.asyncio
async def test_negative_paragraph():
    impulses = await _robot().interpret_text(
        "Power is out, the cat is missing, and we just had a fight."
    )
    chems = _chemicals(impulses)
    assert Chemical.CORTISOL in chems
    cort = next(i for i in impulses if i.chemical == Chemical.CORTISOL)
    assert cort.delta >= 0.20


@pytest.mark.asyncio
async def test_ambient_paragraph_uses_duration():
    impulses = await _robot().interpret_text(
        "Sitting quietly in the sun on a long peaceful afternoon."
    )
    # An ambient paragraph should use a non-zero duration on at least
    # one impulse — that's what distinguishes ambience from spikes.
    assert any(i.duration_seconds > 0 for i in impulses)


@pytest.mark.asyncio
async def test_short_paragraph_still_works():
    impulses = await _robot().interpret_text("you won the lottery")
    chems = _chemicals(impulses)
    assert Chemical.DOPAMINE in chems
