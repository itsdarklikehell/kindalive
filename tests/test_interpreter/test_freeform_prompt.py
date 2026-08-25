"""Mock-LLM tests for the multi-fact freeform prompt rework.

These run without an API key — they exercise the prompt construction
and the surrounding pipeline. Live LLM calibration runs lives in
``test_freeform_paragraphs.py`` under the ``llm`` marker.
"""

from __future__ import annotations

from kindalive.emotions.projection import EmotionProjection
from kindalive.engine.chemicals import ChemicalState
from kindalive.interpreter.prompt_builder import (
    INTERPRETER_SYSTEM_PROMPT,
    PromptBuilder,
)
from kindalive.interpreter.text_input import UserText


def _context() -> object:
    state = ChemicalState()
    emotions = EmotionProjection.compute(state)
    return PromptBuilder.build_context(
        personality="default",
        affinity=1.0,
        emotions=emotions,
        chemicals=state,
    )


def test_freeform_preamble_handles_multi_fact_paragraphs():
    prompt = PromptBuilder.build_user_prompt(
        UserText(summary="Friday, finances up, and tomorrow I have off"),
        _context(),
    )
    assert "OWNER is speaking" in prompt
    assert "ONE consolidated impulse array" in prompt
    assert "do not return" in prompt
    assert "per-fact" in prompt


def test_freeform_preamble_describes_intensity_strategy():
    prompt = PromptBuilder.build_user_prompt(
        UserText(summary="anything"),
        _context(),
    )
    assert "Match intensity to the strongest fact" in prompt


def test_system_prompt_contains_multi_fact_calibration_examples():
    sysprompt = INTERPRETER_SYSTEM_PROMPT
    # Each new multi-fact example must appear verbatim.
    assert "Friday, finances are up" in sysprompt
    assert "Power's out, the cat is missing" in sysprompt
    assert "Quiet rainy afternoon, finished a book" in sysprompt
    # Existing single-event examples are still there.
    assert "I won the lottery" in sysprompt
    assert "war broke out in my country" in sysprompt


def test_system_prompt_distinguishes_single_vs_multi_fact_sections():
    sysprompt = INTERPRETER_SYSTEM_PROMPT
    assert "Single events:" in sysprompt
    assert "Multi-fact paragraphs" in sysprompt


def test_freeform_preamble_absent_for_legacy_non_user_events():
    """Defensive — synthetic events with non-user source skip the preamble."""
    prompt = PromptBuilder.build_user_prompt(
        UserText(summary="hello", source="legacy", event_type="other"),
        _context(),
    )
    assert "OWNER is speaking" not in prompt
