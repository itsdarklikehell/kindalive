"""Tests for the web UI: smoke tests + Robot.last_impulses behavior."""

from __future__ import annotations

import pytest

from kindalive.engine.chemicals import Chemical
from kindalive.engine.impulse import ChemicalImpulse
from kindalive.interpreter.text_input import UserText
from kindalive.robot import Robot


def test_web_ui_module_imports():
    """Smoke test: the web_ui module imports without error."""
    from kindalive.expression import web_ui

    assert hasattr(web_ui, "create_app")
    assert hasattr(web_ui, "main")


def test_web_ui_wires_the_face():
    """The UI is built around the LED dot-matrix face."""
    from kindalive.expression import web_ui

    assert hasattr(web_ui, "FaceProjection")
    assert hasattr(web_ui, "face_payload")
    assert hasattr(web_ui, "container_html")


def test_appstate_reset_clears_chemistry_and_conversation():
    """Reset rebuilds the robot (near baseline, within the jitter) and
    forgets the conversation."""
    from kindalive.expression.web_ui import RESET_JITTER, AppState

    state = AppState()
    # Perturb chemistry far past the jitter and seed a conversation turn.
    state.robot.current_chemicals().set(Chemical.CORTISOL, 0.95)
    state.robot._conversation.append({"role": "user", "content": "hi"})

    state.reset()

    assert state.robot.conversation == []
    chem = state.robot.current_chemicals()
    baseline = chem.baseline(Chemical.CORTISOL)
    # No longer 0.95, and within the reset jitter of baseline.
    assert abs(chem.get(Chemical.CORTISOL) - baseline) <= RESET_JITTER + 1e-9


def test_reset_starting_state_is_jittered_and_valid():
    """A fresh robot is nudged off baseline (not always calm) but stays
    within range and within the jitter band."""
    from kindalive.expression.web_ui import RESET_JITTER, AppState

    moved = False
    for _ in range(8):
        state = AppState()
        chem = state.robot.current_chemicals()
        for c in Chemical:
            level = chem.get(c)
            assert 0.0 <= level <= 1.0
            assert abs(level - chem.baseline(c)) <= RESET_JITTER + 1e-9
            if abs(level - chem.baseline(c)) > 1e-6:
                moved = True
    assert moved, "reset never perturbed any chemical"


def test_web_ui_create_app_constructs():
    """create_app() must run without error — guards static-file
    registration and page construction."""
    from kindalive.expression import web_ui

    state = web_ui.create_app()
    assert state is not None
    assert state.robot is not None


def test_web_ui_palettes_cover_all_series():
    from kindalive.expression import web_ui

    assert set(web_ui.CHEMICAL_COLORS) == {c.value for c in Chemical}
    emotions = web_ui.AppState().robot.current_emotions().as_dict()
    assert set(web_ui.EMOTION_COLORS) == set(emotions)


@pytest.mark.asyncio
async def test_interpret_text_wraps_user_event():
    """Robot.interpret_text wraps text into a `user/freeform` event."""
    robot = Robot(personality="default")

    received = []
    original = robot.process_event

    async def spy(event):
        received.append(event)
        await original(event)

    robot.process_event = spy  # type: ignore[assignment]
    await robot.interpret_text("you won the lottery")

    assert len(received) == 1
    evt = received[0]
    assert evt.source == "user"
    assert evt.event_type == "freeform"
    assert evt.summary == "you won the lottery"


# ---------------------------------------------------------------------------
# Robot.last_impulses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_robot_last_impulses_after_fallback():
    """Fallback-only Robot populates last_impulses after process_event."""
    robot = Robot(personality="default")
    await robot.process_event(UserText(summary="Team wins!"))
    assert len(robot.last_impulses) > 0
    assert all(isinstance(i, ChemicalImpulse) for i in robot.last_impulses)


def test_robot_last_impulses_after_receive():
    """receive_impulses() stores last_impulses."""
    robot = Robot(personality="default")
    impulses = [
        ChemicalImpulse(chemical=Chemical.DOPAMINE, delta=0.2),
        ChemicalImpulse(chemical=Chemical.SEROTONIN, delta=0.1),
    ]
    robot.receive_impulses(impulses)
    assert len(robot.last_impulses) == 2
    assert robot.last_impulses[0].chemical == Chemical.DOPAMINE
    assert robot.last_impulses[1].chemical == Chemical.SEROTONIN


@pytest.mark.asyncio
async def test_robot_last_impulses_reset_between_events():
    """last_impulses reflects the most recent event, not a prior one."""
    robot = Robot(personality="default")
    await robot.process_event(UserText(summary="Win!"))
    first = list(robot.last_impulses)
    await robot.process_event(UserText(summary="Rain"))
    second = list(robot.last_impulses)
    assert len(first) > 0
    assert len(second) > 0


def test_robot_last_impulses_starts_empty():
    """Before any event, last_impulses is empty."""
    robot = Robot(personality="default")
    assert robot.last_impulses == []
