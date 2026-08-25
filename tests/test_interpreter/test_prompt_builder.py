"""Tests for PromptBuilder."""

from kindalive.emotions.projection import EmotionProjection
from kindalive.engine.chemicals import ChemicalState
from kindalive.interpreter.prompt_builder import PromptBuilder, RobotContext
from kindalive.interpreter.text_input import UserText


def test_system_prompt_contains_chemicals():
    prompt = PromptBuilder.build_system_prompt()
    assert "dopamine" in prompt
    assert "cortisol" in prompt
    assert "JSON array" in prompt


def test_user_prompt_contains_event():
    ev = UserText(
        source="sports",
        event_type="goal_scored",
        summary="BOS scored, leads 3-2",
        urgency="realtime",
    )
    context = RobotContext(
        personality_name="cheerful",
        affinity=1.2,
        dominant_emotion="happiness",
        chemical_summary="dopamine=0.30, serotonin=0.50",
    )
    prompt = PromptBuilder.build_user_prompt(ev, context)
    assert "cheerful" in prompt
    assert "goal_scored" in prompt
    assert "BOS scored" in prompt
    assert "1.2" in prompt


def test_build_context_from_state():
    state = ChemicalState()
    emotions = EmotionProjection.compute(state)
    context = PromptBuilder.build_context(
        personality="default",
        affinity=1.0,
        emotions=emotions,
        chemicals=state,
    )
    assert context.personality_name == "default"
    assert context.affinity == 1.0
    assert isinstance(context.dominant_emotion, str)
    assert len(context.chemical_summary) > 0


def test_freeform_preamble_present_for_user_freeform():
    """Owner-speaking preamble should be added for user freeform events."""
    ev = UserText(
        source="user",
        event_type="freeform",
        summary="I won the lottery",
        urgency="realtime",
    )
    context = RobotContext(
        personality_name="default",
        affinity=1.0,
        dominant_emotion="calm",
        chemical_summary="dopamine=0.30",
    )
    prompt = PromptBuilder.build_user_prompt(ev, context)
    assert "OWNER" in prompt
    assert "speaking to the robot directly" in prompt


def test_freeform_preamble_absent_for_non_freeform():
    """Regular events should NOT get the owner-speaking preamble."""
    ev = UserText(
        source="sports",
        event_type="goal_scored",
        summary="BOS scored!",
        urgency="realtime",
    )
    context = RobotContext(
        personality_name="default",
        affinity=1.0,
        dominant_emotion="calm",
        chemical_summary="dopamine=0.30",
    )
    prompt = PromptBuilder.build_user_prompt(ev, context)
    assert "OWNER" not in prompt


def test_system_prompt_no_plus_prefix_on_numbers():
    """Prompt examples must not use +0.xx notation — it breaks JSON output."""
    prompt = PromptBuilder.build_system_prompt()
    # The intensity scale and examples should use plain decimals
    assert "CORRECT:   \"delta\": 0.35" in prompt or "delta\": 0.35" in prompt
    # Should NOT contain +0. followed by a digit as a standalone example value
    import re
    # Find lines that look like example notation: "dopamine +0.35"
    plus_examples = re.findall(r"→.*?\+0\.\d", prompt)
    assert len(plus_examples) == 0, f"Found +0.xx in examples: {plus_examples}"
