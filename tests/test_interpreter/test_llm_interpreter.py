"""Tests for LLMInterpreter with MockLLMBackend."""

import json

import pytest
from kindalive.engine.chemicals import Chemical
from kindalive.interpreter.llm_interpreter import LLMInterpreter, MockLLMBackend
from kindalive.interpreter.prompt_builder import RobotContext
from kindalive.interpreter.text_input import UserText


def _context() -> RobotContext:
    return RobotContext(
        personality_name="default",
        affinity=1.0,
        dominant_emotion="calm",
        chemical_summary="dopamine=0.30",
    )


def _event(urgency: str = "realtime") -> UserText:
    return UserText(summary="you won the lottery", urgency=urgency)


@pytest.mark.asyncio
async def test_interpret_valid_response():
    backend = MockLLMBackend()
    backend.enqueue_impulses([
        {"chemical": "dopamine", "delta": 0.25},
        {"chemical": "adrenaline", "delta": 0.3, "duration_seconds": 10},
    ])
    interp = LLMInterpreter(backend=backend, use_cache=False)
    impulses = await interp.interpret(_event(), _context())
    assert len(impulses) == 2
    assert impulses[0].chemical == Chemical.DOPAMINE
    assert impulses[1].chemical == Chemical.ADRENALINE
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_interpret_object_with_reply():
    """New format: {"reply": ..., "impulses": [...]} — reply captured,
    impulses validated."""
    backend = MockLLMBackend()
    backend.enqueue_response(json.dumps({
        "reply": "Oh wow, congratulations!",
        "impulses": [{"chemical": "dopamine", "delta": 0.5}],
    }))
    interp = LLMInterpreter(backend=backend, use_cache=False)
    impulses = await interp.interpret(_event(), _context())
    assert len(impulses) == 1
    assert impulses[0].chemical == Chemical.DOPAMINE
    assert interp.last_reply == "Oh wow, congratulations!"


@pytest.mark.asyncio
async def test_interpret_bare_array_has_no_reply():
    """Legacy/mock bare-array responses parse with an empty reply."""
    backend = MockLLMBackend()
    backend.enqueue_impulses([{"chemical": "dopamine", "delta": 0.2}])
    interp = LLMInterpreter(backend=backend, use_cache=False)
    await interp.interpret(_event(), _context())
    assert interp.last_reply == ""


@pytest.mark.asyncio
async def test_reply_resets_between_calls():
    """A second call that yields no reply must not keep the old one."""
    backend = MockLLMBackend()
    backend.enqueue_response(json.dumps({
        "reply": "Nice!", "impulses": [{"chemical": "dopamine", "delta": 0.2}],
    }))
    backend.enqueue_impulses([{"chemical": "cortisol", "delta": 0.1}])
    interp = LLMInterpreter(backend=backend, use_cache=False)
    await interp.interpret(_event(), _context())
    assert interp.last_reply == "Nice!"
    await interp.interpret(UserText(summary="something else"), _context())
    assert interp.last_reply == ""


@pytest.mark.asyncio
async def test_interpret_cache_hit_skips_llm():
    backend = MockLLMBackend()
    backend.enqueue_impulses([{"chemical": "dopamine", "delta": 0.2}])
    interp = LLMInterpreter(backend=backend)
    ev = _event()
    ctx = _context()

    # First call — cache miss, calls LLM
    result1 = await interp.interpret(ev, ctx)
    assert backend.call_count == 1
    assert len(result1) == 1

    # Second call — cache hit, no LLM call
    result2 = await interp.interpret(ev, ctx)
    assert backend.call_count == 1  # still 1
    assert len(result2) == 1


@pytest.mark.asyncio
async def test_interpret_invalid_json_falls_back():
    backend = MockLLMBackend()
    backend.enqueue_response("not valid json!!!")
    interp = LLMInterpreter(backend=backend, use_cache=False)
    impulses = await interp.interpret(_event(), _context())
    # Default fallback nudges cortisol up by a small amount.
    assert len(impulses) == 1
    assert impulses[0].chemical == Chemical.CORTISOL
    assert interp.last_path == "fallback"


@pytest.mark.asyncio
async def test_interpret_fallback_disabled_returns_empty():
    backend = MockLLMBackend()
    backend.enqueue_response("garbage")
    interp = LLMInterpreter(backend=backend, use_cache=False, use_fallback=False)
    impulses = await interp.interpret(_event(), _context())
    assert impulses == []


@pytest.mark.asyncio
async def test_interpret_empty_response():
    backend = MockLLMBackend()
    backend.enqueue_impulses([])
    interp = LLMInterpreter(backend=backend, use_cache=False)
    impulses = await interp.interpret(_event(), _context())
    assert impulses == []


@pytest.mark.asyncio
async def test_mock_backend_default_empty():
    """MockLLMBackend returns [] when no responses enqueued."""
    backend = MockLLMBackend()
    result = await backend.call("sys", "user")
    assert result == "[]"
    assert backend.call_count == 1


# ── JSON payload extraction (fence / prose stripping) ─────────────


def test_extract_json_bare():
    from kindalive.interpreter.llm_interpreter import _extract_json_payload
    out = _extract_json_payload('[{"chemical": "dopamine", "delta": 0.3}]')
    assert out.startswith("[")
    assert json.loads(out) == [{"chemical": "dopamine", "delta": 0.3}]


def test_extract_json_with_markdown_fence():
    from kindalive.interpreter.llm_interpreter import _extract_json_payload
    raw = '```json\n[{"chemical": "dopamine", "delta": 0.3}]\n```'
    out = _extract_json_payload(raw)
    assert json.loads(out) == [{"chemical": "dopamine", "delta": 0.3}]


def test_extract_json_with_unlabeled_fence():
    from kindalive.interpreter.llm_interpreter import _extract_json_payload
    raw = '```\n[{"chemical": "serotonin", "delta": 0.15}]\n```'
    out = _extract_json_payload(raw)
    assert json.loads(out) == [{"chemical": "serotonin", "delta": 0.15}]


def test_extract_json_with_prose_preamble():
    from kindalive.interpreter.llm_interpreter import _extract_json_payload
    raw = 'Here is the response:\n[{"chemical": "dopamine", "delta": 0.4}]'
    out = _extract_json_payload(raw)
    assert json.loads(out) == [{"chemical": "dopamine", "delta": 0.4}]


@pytest.mark.asyncio
async def test_interpret_survives_fenced_response():
    """LLM wrapping its response in ```json fences``` should still work."""
    backend = MockLLMBackend()
    backend.enqueue_response(
        '```json\n'
        '[{"chemical": "dopamine", "delta": 0.5},'
        ' {"chemical": "adrenaline", "delta": 0.4}]\n'
        '```'
    )
    interp = LLMInterpreter(backend=backend, use_cache=False)
    impulses = await interp.interpret(_event(), _context())
    assert len(impulses) == 2
    assert interp.last_path == "llm"
    assert impulses[0].chemical == Chemical.DOPAMINE
    assert impulses[0].delta == 0.5


@pytest.mark.asyncio
async def test_last_path_reports_fallback_on_bad_json():
    backend = MockLLMBackend()
    backend.enqueue_response("this is not json at all")
    interp = LLMInterpreter(backend=backend, use_cache=False)
    await interp.interpret(_event(), _context())
    assert interp.last_path == "fallback"
    assert interp.last_error  # non-empty error string


# ── JSON quirk sanitization (leading +, trailing commas) ──────────
# These are the narrow model quirks we see when asking Haiku for JSON
# with a prompt that contains "+0.35" style intensity notation. The
# interpreter strips them before json.loads rather than silently falling
# back to the default cortisol drizzle.


def test_extract_json_strips_leading_plus_on_numbers():
    from kindalive.interpreter.llm_interpreter import _extract_json_payload
    raw = '[{"chemical": "cortisol", "delta": +0.30}]'
    out = _extract_json_payload(raw)
    assert json.loads(out) == [{"chemical": "cortisol", "delta": 0.30}]


def test_extract_json_strips_plus_across_multiline_response():
    from kindalive.interpreter.llm_interpreter import _extract_json_payload
    raw = (
        '[\n'
        '  {"chemical": "cortisol",  "delta": +0.30},\n'
        '  {"chemical": "dopamine",  "delta": -0.25},\n'
        '  {"chemical": "serotonin", "delta": -0.20}\n'
        ']'
    )
    parsed = json.loads(_extract_json_payload(raw))
    assert len(parsed) == 3
    assert parsed[0]["delta"] == 0.30
    assert parsed[1]["delta"] == -0.25  # negative preserved
    assert parsed[2]["delta"] == -0.20


def test_extract_json_strips_trailing_commas():
    from kindalive.interpreter.llm_interpreter import _extract_json_payload
    raw = '[{"chemical": "dopamine", "delta": 0.5,}, {"chemical": "adrenaline", "delta": 0.4,}]'
    parsed = json.loads(_extract_json_payload(raw))
    assert len(parsed) == 2


def test_extract_json_preserves_plus_inside_strings():
    """A literal `+` inside a string value must survive the sanitizer."""
    from kindalive.interpreter.llm_interpreter import _extract_json_payload
    raw = '[{"chemical": "dopamine", "delta": 0.3, "source_label": "A+ exam"}]'
    parsed = json.loads(_extract_json_payload(raw))
    assert parsed[0]["source_label"] == "A+ exam"


@pytest.mark.asyncio
async def test_interpret_recovers_from_plus_prefix():
    """The grandma bug: Haiku emits +0.30 and we used to silently fall back."""
    backend = MockLLMBackend()
    backend.enqueue_response(
        '[\n'
        '  {"chemical": "cortisol",  "delta": +0.30},\n'
        '  {"chemical": "dopamine",  "delta": -0.25},\n'
        '  {"chemical": "serotonin", "delta": -0.20}\n'
        ']'
    )
    interp = LLMInterpreter(backend=backend, use_cache=False)
    impulses = await interp.interpret(_event(), _context())
    assert interp.last_path == "llm"  # NOT fallback
    assert len(impulses) == 3
    assert impulses[0].chemical == Chemical.CORTISOL
    assert impulses[0].delta == 0.30
    assert impulses[1].delta == -0.25
