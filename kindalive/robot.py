"""Robot — top-level integration class.

Wires together the neurochemical engine, emotion projection,
personality, expression layer, and interpreter pipeline.
"""

from __future__ import annotations

import json

from kindalive.emotions.emotion_vector import EmotionVector
from kindalive.emotions.projection import EmotionProjection
from kindalive.engine.chemicals import ChemicalState
from kindalive.engine.clock import (
    Clock,
    ManualClock,
)
from kindalive.engine.impulse import ChemicalImpulse
from kindalive.engine.neurochemical_engine import NeurochemicalEngine
from kindalive.engine.seed_chemistry import SeedChemistry
from kindalive.expression.text_output import TextExpressionOutput
from kindalive.interpreter.event_router import RealtimeRouter
from kindalive.interpreter.llm_interpreter import LLMBackend, LLMInterpreter
from kindalive.interpreter.prompt_builder import PromptBuilder, RobotContext
from kindalive.interpreter.text_input import UserText
from kindalive.personality.presets import get_default_affinity, get_seed

# How many chat messages of history to keep (each exchange is 2: the
# owner's line + the robot's reply). Bounds the prompt size over a long
# session. Kept even so the trimmed history always starts on a user turn.
MAX_CONVERSATION_MESSAGES = 40


class Robot:
    """A robot with neurochemical emotions.

    Args:
        engine: Pre-configured NeurochemicalEngine (takes priority).
        personality: Name of personality preset (used if engine not provided).
        seed: Explicit SeedChemistry (used if engine not provided, overrides personality seed).
        clock: Clock instance (used if engine not provided).
        expression: Expression output layer.
        llm_backend: LLM backend for interpreting events (None = fallback-only mode).
    """

    def __init__(
        self,
        engine: NeurochemicalEngine | None = None,
        personality: str = "default",
        seed: SeedChemistry | None = None,
        clock: Clock | None = None,
        expression: TextExpressionOutput | None = None,
        llm_backend: LLMBackend | None = None,
    ) -> None:
        self._personality = personality

        if engine is not None:
            self._engine = engine
        else:
            effective_seed = seed or get_seed(personality)
            self._engine = NeurochemicalEngine(
                clock=clock or ManualClock(),
                seed=effective_seed,
            )

        self._expression = expression or TextExpressionOutput()
        self._affinity = get_default_affinity(personality)
        self._last_impulses: list[ChemicalImpulse] = []
        # Session conversation memory — chat turns ({"role", "content"})
        # fed back to the LLM so the robot remembers the discussion.
        self._conversation: list[dict[str, str]] = []

        # Interpreter pipeline (optional — None means fallback-only)
        self._interpreter: LLMInterpreter | None = None
        self._router: RealtimeRouter | None = None
        if llm_backend is not None:
            self._interpreter = LLMInterpreter(backend=llm_backend)
            self._router = RealtimeRouter(
                interpreter=self._interpreter,
                context_fn=self._build_context,
                apply_fn=self._apply_impulses_sync,
            )

    @property
    def personality(self) -> str:
        return self._personality

    @property
    def affinity(self) -> float:
        return self._affinity

    @property
    def interpreter(self) -> LLMInterpreter | None:
        return self._interpreter

    @property
    def router(self) -> RealtimeRouter | None:
        return self._router

    @property
    def last_impulses(self) -> list[ChemicalImpulse]:
        """Impulses applied by the most recent process_event/receive_impulses call."""
        return self._last_impulses

    @property
    def last_reply(self) -> str:
        """The robot's spoken reply from the most recent LLM interpretation.

        Empty when there is no interpreter, on a cache hit, on fallback,
        or when the model chose to stay silent.
        """
        return self._interpreter.last_reply if self._interpreter else ""

    def receive_impulses(self, impulses: list[ChemicalImpulse]) -> None:
        """Inject impulses directly into the engine (bypasses interpreter)."""
        self._last_impulses = list(impulses)
        self._engine.apply_impulses(impulses)

    async def process_event(self, event: UserText) -> None:
        """Full pipeline: event → interpreter → impulses → engine."""
        self._last_impulses = []
        if self._router is not None:
            await self._router.route(event)
            # Remember the exchange only when the LLM actually answered,
            # so a fallback/outage doesn't poison the thread.
            if (
                self._interpreter is not None
                and self._interpreter.last_path == "llm"
            ):
                self._remember_turn(event.summary, self._interpreter.last_reply)
        else:
            # No interpreter — use the default fallback nudge
            from kindalive.interpreter.fallback_rules import lookup_fallback
            impulses = lookup_fallback(event)
            self._last_impulses = list(impulses)
            self._engine.apply_impulses(impulses)

    def _remember_turn(self, user_text: str, reply: str) -> None:
        """Append one exchange to the session conversation history.

        The assistant turn is stored as a compact JSON object matching
        the model's own output schema, so the dialogue the LLM sees is
        format-consistent (owner speaks plainly → robot answers in JSON).
        """
        self._conversation.append({"role": "user", "content": user_text})
        assistant = json.dumps({
            "reply": reply,
            "impulses": [
                {"chemical": imp.chemical.value, "delta": round(imp.delta, 3)}
                for imp in self._last_impulses
            ],
        })
        self._conversation.append({"role": "assistant", "content": assistant})
        if len(self._conversation) > MAX_CONVERSATION_MESSAGES:
            self._conversation = self._conversation[-MAX_CONVERSATION_MESSAGES:]

    @property
    def conversation(self) -> list[dict[str, str]]:
        """The session conversation history (chat-message dicts)."""
        return list(self._conversation)

    def reset_conversation(self) -> None:
        """Forget the discussion thread (keeps chemical state)."""
        self._conversation = []

    async def interpret_text(self, text: str) -> list[ChemicalImpulse]:
        """Run a freeform paragraph through the pipeline and return the
        impulses that were applied. Convenience wrapper around
        ``process_event(UserText(summary=text))``."""
        await self.process_event(UserText(summary=text))
        return list(self._last_impulses)

    def current_emotions(self) -> EmotionVector:
        """Compute and return current emotion vector."""
        return EmotionProjection.compute(self._engine.state)

    def current_chemicals(self) -> ChemicalState:
        """Return the current chemical state."""
        return self._engine.state

    async def express(self) -> str:
        """Get a text description of the current emotional state."""
        emotions = self.current_emotions()
        return await self._expression.express(emotions, self._engine.state)

    def advance(self, dt: float) -> None:
        """Advance the simulation by dt seconds.

        Also advances the clock so time-dependent features (saturation
        windows, sustained impulses) work correctly.
        """
        clock = self._engine.clock
        if isinstance(clock, ManualClock):
            clock.advance(seconds=dt)
        self._engine.advance(dt)

    def _build_context(self) -> RobotContext:
        """Build a RobotContext snapshot for the interpreter."""
        emotions = self.current_emotions()
        return PromptBuilder.build_context(
            personality=self._personality,
            affinity=self._affinity,
            emotions=emotions,
            chemicals=self._engine.state,
            history=self._conversation,
        )

    def _apply_impulses_sync(self, impulses: list[ChemicalImpulse]) -> None:
        """Synchronous impulse application (called by RealtimeRouter)."""
        self._last_impulses = list(impulses)
        self._engine.apply_impulses(impulses)
