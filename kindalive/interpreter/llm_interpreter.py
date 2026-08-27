"""LLMInterpreter — event-to-impulse translation via LLM or mock.

The interpreter pipeline:
  UserText → cache check → (LLM call or fallback) → validate → cache store → impulses
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from typing import Any, Protocol

from kindalive.engine.impulse import ChemicalImpulse
from kindalive.interpreter.fallback_rules import lookup_fallback
from kindalive.interpreter.impulse_cache import ImpulseCache
from kindalive.interpreter.prompt_builder import PromptBuilder, RobotContext
from kindalive.interpreter.text_input import UserText
from kindalive.interpreter.validator import ValidationError, validate_raw_impulses

# Strip ```json ... ``` or ``` ... ``` wrappers that some models add even
# when told not to. Matches fences at the start/end of the response with
# optional whitespace and an optional language tag.
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE
)

# Claude Haiku sometimes emits "delta": +0.35 — valid Python, INVALID JSON.
# JSON number grammar forbids a leading `+` on numerics. We strip them
# wherever a `+` sits directly after a JSON token boundary (`:`, `,`, `[`,
# `(`, or whitespace) and directly before a digit or decimal point.
# This is the single most common LLM-to-JSON quirk we see with this prompt
# (which contains "+0.35" style intensity notation in its examples).
_LEADING_PLUS_RE = re.compile(r"(?<=[:,\[(\s])\+(?=\d|\.\d)")

# Trailing commas before `]` or `}` — another common LLM JSON slip.
_TRAILING_COMMA_RE = re.compile(r",\s*(?=[\]}])")


def _sanitize_json_quirks(text: str) -> str:
    """Fix the narrow set of JSON syntax errors we see from LLMs.

    We deliberately keep this to a very small, well-understood set:
      - leading `+` on positive numbers
      - trailing commas inside arrays/objects
    We do NOT try to fix unquoted keys, single quotes, or comments —
    those are rarer and carry higher risk of corrupting string content.
    """
    text = _LEADING_PLUS_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub("", text)
    return text


def _extract_json_payload(raw: str) -> str:
    """Best-effort extraction of a JSON array from an LLM text response.

    Handles four common deviations from "bare JSON only":
      1. Leading/trailing whitespace (trivial)
      2. Markdown code fences (```json ... ```)
      3. Prose preamble followed by a JSON array — we locate the first
         top-level `[` or `{`.
      4. Narrow JSON syntax quirks — leading `+` on numbers, trailing commas.
    """
    text = raw.strip()
    # Strip code fences
    text = _FENCE_RE.sub("", text).strip()
    # If it already starts with [ or {, keep as-is; otherwise locate the
    # first array/object in the text.
    if not text.startswith(("[", "{")):
        first_bracket = text.find("[")
        first_brace = text.find("{")
        candidates = [i for i in (first_bracket, first_brace) if i != -1]
        if not candidates:
            return text  # nothing we can do, let json.loads complain
        text = text[min(candidates):]
    # Repair the narrow quirks the JSON parser would otherwise reject.
    return _sanitize_json_quirks(text)


def _split_reply_and_impulses(data: object) -> tuple[str, Any]:
    """Separate the spoken reply from the impulse array.

    The LLM returns ``{"reply": "...", "impulses": [...]}``. Older
    prompts (and the mock backend in tests) return a bare impulse
    array; treat that as no reply for backward compatibility.
    """
    if isinstance(data, dict):
        reply = data.get("reply", "")
        impulses = data.get("impulses", [])
        if not isinstance(reply, str):
            reply = ""
        return reply.strip(), impulses
    # Bare array (legacy / fallback shape)
    return "", data


def _log_fallback(where: str, event: UserText, reason: str, raw: str = "") -> None:
    """Print a visible warning to stderr when the LLM path bails out."""
    print(
        f"[kindalive.llm] {where} FELL BACK for "
        f"{event.source}:{event.event_type} — {reason}",
        file=sys.stderr,
    )
    if raw:
        snippet = raw if len(raw) < 400 else raw[:380] + "...[truncated]"
        print(f"[kindalive.llm] raw response: {snippet!r}", file=sys.stderr)


class LLMBackend(Protocol):
    """Protocol for the actual LLM call."""

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Send prompts to the LLM, return the raw response string.

        ``history`` is prior conversation turns ({"role", "content"}
        dicts, oldest first) prepended before ``user_prompt`` so the
        model keeps the thread of the discussion.
        """
        ...


class MockLLMBackend:
    """Test backend that returns pre-configured responses."""

    def __init__(self) -> None:
        self._responses: list[str] = []
        self._calls: list[tuple[str, str]] = []
        self.histories: list[list[dict[str, str]]] = []  # history passed to each call

    def enqueue_response(self, response: str) -> None:
        self._responses.append(response)

    def enqueue_impulses(self, impulses: list[dict[str, Any]]) -> None:
        self._responses.append(json.dumps(impulses))

    @property
    def calls(self) -> list[tuple[str, str]]:
        return list(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        self._calls.append((system_prompt, user_prompt))
        self.histories.append(list(history or []))
        if not self._responses:
            return "[]"
        return self._responses.pop(0)


class LLMInterpreter:
    """Interprets events into chemical impulses using an LLM backend.

    Pipeline: cache check → LLM call → validate → cache store.
    Falls back to heuristic rules on LLM failure.

    Fallback paths are LOUD — every fallback logs to stderr with the
    raw response and the reason, so misbehaving prompts/models are
    visible instead of silently producing cortisol drizzle.
    """

    def __init__(
        self,
        backend: LLMBackend,
        cache: ImpulseCache | None = None,
        use_cache: bool = True,
        use_fallback: bool = True,
    ) -> None:
        self._backend = backend
        self._cache = cache or ImpulseCache()
        self._use_cache = use_cache
        self._use_fallback = use_fallback
        self._prompt_builder = PromptBuilder()
        # Last-call telemetry (for UI surfacing)
        self.last_path: str = "none"  # "cache" | "llm" | "fallback"
        self.last_error: str = ""
        self.last_reply: str = ""     # spoken reply from the last LLM call

    @property
    def cache(self) -> ImpulseCache:
        return self._cache

    async def interpret(
        self,
        event: UserText,
        context: RobotContext,
    ) -> list[ChemicalImpulse]:
        """Interpret a single event into impulses."""
        self.last_error = ""
        self.last_reply = ""
        # Mid-conversation, the same words mean different things, so the
        # paragraph-keyed cache would return stale answers — bypass it
        # once there is any prior dialogue.
        use_cache_now = self._use_cache and not context.history
        # 1. Cache check
        if use_cache_now:
            cached = self._cache.get(
                event, context.personality_name, context.affinity
            )
            if cached is not None:
                self.last_path = "cache"
                return cached

        # 2. LLM call (with prior conversation turns as context)
        raw_json = ""
        try:
            system = self._prompt_builder.build_system_prompt()
            user = self._prompt_builder.build_user_prompt(event, context)
            raw_json = await self._backend.call(
                system, user, history=context.history
            )
            payload = _extract_json_payload(raw_json)
            raw_data = json.loads(payload)
            reply, raw_impulses = _split_reply_and_impulses(raw_data)
            impulses = validate_raw_impulses(raw_impulses)
            self.last_reply = reply
        except (json.JSONDecodeError, ValidationError) as e:
            self.last_error = f"{type(e).__name__}: {e}"
            _log_fallback("interpret", event, self.last_error, raw_json)
            if self._use_fallback:
                self.last_path = "fallback"
                return lookup_fallback(event)
            self.last_path = "fallback"
            return []
        except Exception as e:  # noqa: BLE001  (offline-fallback: any backend init failure → run without LLM)
            # Network errors, timeouts, etc.
            self.last_error = f"{type(e).__name__}: {e}"
            _log_fallback("interpret", event, self.last_error, raw_json)
            traceback.print_exc(file=sys.stderr)
            if self._use_fallback:
                self.last_path = "fallback"
                return lookup_fallback(event)
            self.last_path = "fallback"
            return []

        # 3. Cache store
        if use_cache_now:
            self._cache.put(
                event, context.personality_name, context.affinity, impulses
            )

        self.last_path = "llm"
        return impulses

