"""Full pipeline integration tests: text → MockLLM → Router → Engine → Expression.

The last test locks the entire chain together: a paragraph of text in,
the 3D-face renderer payload out.
"""

from __future__ import annotations

import pytest
from kindalive.engine.chemicals import Chemical
from kindalive.engine.clock import ManualClock
from kindalive.engine.neurochemical_engine import NeurochemicalEngine
from kindalive.expression.face import FACE_WEIGHTS, FaceProjection
from kindalive.expression.face_3d import face_payload
from kindalive.interpreter.llm_interpreter import MockLLMBackend
from kindalive.interpreter.text_input import UserText
from kindalive.robot import Robot


@pytest.mark.asyncio
async def test_robot_process_event_with_llm():
    """Text flows through the LLM interpreter to the engine."""
    backend = MockLLMBackend()
    backend.enqueue_impulses([
        {"chemical": "dopamine", "delta": 0.25},
        {"chemical": "adrenaline", "delta": 0.3},
    ])

    clock = ManualClock()
    robot = Robot(
        engine=NeurochemicalEngine(clock=clock),
        llm_backend=backend,
    )
    baseline_dop = robot.current_chemicals().get(Chemical.DOPAMINE)

    await robot.process_event(UserText(summary="you won the lottery"))

    assert robot.current_chemicals().get(Chemical.DOPAMINE) > baseline_dop
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_robot_process_event_fallback_only():
    """Without an LLM backend the robot still nudges cortisol."""
    clock = ManualClock()
    robot = Robot(engine=NeurochemicalEngine(clock=clock))

    baseline_cort = robot.current_chemicals().get(Chemical.CORTISOL)

    await robot.process_event(UserText(summary="something happened"))

    # Default fallback rule is a small cortisol nudge.
    assert robot.current_chemicals().get(Chemical.CORTISOL) > baseline_cort


@pytest.mark.asyncio
async def test_pipeline_with_advance():
    """Full cycle: text → process → advance → emotions decay."""
    backend = MockLLMBackend()
    backend.enqueue_impulses([
        {"chemical": "cortisol", "delta": 0.3},
        {"chemical": "adrenaline", "delta": 0.25},
    ])

    clock = ManualClock()
    robot = Robot(
        engine=NeurochemicalEngine(clock=clock),
        llm_backend=backend,
    )

    emotions_before = robot.current_emotions()

    await robot.process_event(
        UserText(summary="the market just crashed five percent")
    )

    emotions_after = robot.current_emotions()
    assert emotions_after.anxiety > emotions_before.anxiety

    robot.advance(dt=600.0)
    emotions_recovered = robot.current_emotions()
    assert emotions_recovered.anxiety < emotions_after.anxiety


@pytest.mark.asyncio
async def test_pipeline_expression_output():
    """Process a paragraph, then express the resulting state."""
    backend = MockLLMBackend()
    backend.enqueue_impulses([
        {"chemical": "dopamine", "delta": 0.4},
        {"chemical": "adrenaline", "delta": 0.35},
    ])

    clock = ManualClock()
    robot = Robot(
        engine=NeurochemicalEngine(clock=clock),
        llm_backend=backend,
    )

    await robot.process_event(
        UserText(summary="overtime winner — your team takes the cup")
    )

    text = await robot.express()
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_robot_interpret_text_convenience():
    """`Robot.interpret_text` accepts a bare string and applies impulses."""
    backend = MockLLMBackend()
    backend.enqueue_impulses([{"chemical": "serotonin", "delta": 0.15}])

    clock = ManualClock()
    robot = Robot(
        engine=NeurochemicalEngine(clock=clock),
        llm_backend=backend,
    )

    impulses = await robot.interpret_text(
        "Friday, finances are up, and tomorrow I have off"
    )

    assert len(impulses) == 1
    assert impulses[0].chemical == Chemical.SEROTONIN


@pytest.mark.asyncio
async def test_robot_exposes_spoken_reply():
    """The LLM's spoken reply is surfaced on Robot.last_reply."""
    backend = MockLLMBackend()
    backend.enqueue_response(
        '{"reply": "Heh, that\'s great news!", '
        '"impulses": [{"chemical": "dopamine", "delta": 0.3}]}'
    )
    robot = Robot(
        engine=NeurochemicalEngine(clock=ManualClock()),
        llm_backend=backend,
    )
    assert robot.last_reply == ""
    await robot.interpret_text("I got the job")
    assert robot.last_reply == "Heh, that's great news!"


@pytest.mark.asyncio
async def test_robot_reply_empty_without_llm():
    """No interpreter → no reply (fallback path stays silent)."""
    robot = Robot(engine=NeurochemicalEngine(clock=ManualClock()))
    await robot.interpret_text("anything")
    assert robot.last_reply == ""


@pytest.mark.asyncio
async def test_robot_remembers_conversation():
    """Each new message carries the prior exchanges to the LLM."""
    backend = MockLLMBackend()
    backend.enqueue_response(
        '{"reply": "Congrats!", "impulses": [{"chemical": "dopamine", "delta": 0.3}]}'
    )
    backend.enqueue_response(
        '{"reply": "Still nice.", "impulses": [{"chemical": "serotonin", "delta": 0.1}]}'
    )
    robot = Robot(
        engine=NeurochemicalEngine(clock=ManualClock()), llm_backend=backend
    )
    await robot.interpret_text("I won the lottery")
    await robot.interpret_text("well, it was only five dollars")

    # First call had no history; second call received the first exchange.
    assert backend.histories[0] == []
    h = backend.histories[1]
    assert len(h) == 2
    assert h[0] == {"role": "user", "content": "I won the lottery"}
    assert h[1]["role"] == "assistant"
    assert "Congrats!" in h[1]["content"]
    assert len(robot.conversation) == 4


@pytest.mark.asyncio
async def test_conversation_bypasses_cache_midthread():
    """Repeating a paragraph mid-conversation must re-ask the LLM (the
    same words mean something different in context), not hit the cache."""
    backend = MockLLMBackend()
    for _ in range(3):
        backend.enqueue_response('{"reply": "ok", "impulses": []}')
    robot = Robot(
        engine=NeurochemicalEngine(clock=ManualClock()), llm_backend=backend
    )
    await robot.interpret_text("the same thing")
    await robot.interpret_text("the same thing")
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_conversation_history_is_bounded():
    from kindalive.robot import MAX_CONVERSATION_MESSAGES

    backend = MockLLMBackend()
    for _ in range(40):
        backend.enqueue_response('{"reply": "ok", "impulses": []}')
    robot = Robot(
        engine=NeurochemicalEngine(clock=ManualClock()), llm_backend=backend
    )
    for i in range(40):
        await robot.interpret_text(f"message {i}")
    assert len(robot.conversation) == MAX_CONVERSATION_MESSAGES
    # Trimmed history still starts on a user turn (valid for the API).
    assert robot.conversation[0]["role"] == "user"


@pytest.mark.asyncio
async def test_reset_conversation_clears_history():
    backend = MockLLMBackend()
    backend.enqueue_response('{"reply": "hi", "impulses": []}')
    robot = Robot(
        engine=NeurochemicalEngine(clock=ManualClock()), llm_backend=backend
    )
    await robot.interpret_text("hello")
    assert len(robot.conversation) == 2
    robot.reset_conversation()
    assert robot.conversation == []


@pytest.mark.asyncio
async def test_text_to_face_payload_chain():
    """The whole chain in one test: text → LLM → impulses →
    ChemicalState → FaceState → renderer payload.

    Joyful text must move the smile muscles, and the payload handed to
    the 3D face must carry exactly the 12 muscles the JS consumes.
    """
    backend = MockLLMBackend()
    backend.enqueue_impulses([
        {"chemical": "dopamine", "delta": 0.45},
        {"chemical": "endorphins", "delta": 0.35},
        {"chemical": "adrenaline", "delta": 0.30},
    ])
    robot = Robot(
        engine=NeurochemicalEngine(clock=ManualClock()),
        llm_backend=backend,
    )

    face_before = FaceProjection.compute(robot.current_chemicals())
    await robot.interpret_text("you won the lottery")
    face_after = FaceProjection.compute(robot.current_chemicals())

    # State moved the face muscles in the joyful direction
    assert face_after.lip_corner_pull > face_before.lip_corner_pull
    assert face_after.cheek_raise > face_before.cheek_raise
    assert face_after.jaw_open > face_before.jaw_open

    # The renderer payload carries the muscles + the dominant mood,
    # exactly as the web UI builds it each tick.
    dominant_val = robot.current_emotions().dominant()[1]
    payload = face_payload(face_after, mood_intensity=dominant_val)
    assert set(payload["muscles"]) == set(FACE_WEIGHTS)
    assert payload["muscles"]["lip_corner_pull"] == round(
        face_after.lip_corner_pull, 4)
    assert 0.0 <= payload["mood"]["intensity"] <= 1.0
