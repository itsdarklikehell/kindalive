"""RealtimeRouter — interpret a single freeform UserText event end-to-end.

Every input is a paragraph the owner typed: there is no batching, no
urgency distinction, no buffering. The router exists only to compose
the cache + LLM + apply pipeline into one call site that
`Robot.process_event` can hold onto.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Callable

from kindalive.engine.impulse import ChemicalImpulse
from kindalive.interpreter.llm_interpreter import LLMInterpreter
from kindalive.interpreter.prompt_builder import RobotContext
from kindalive.interpreter.text_input import UserText


class RealtimeRouter:
    """Run a single UserText through the interpreter and apply impulses."""

    def __init__(
        self,
        interpreter: LLMInterpreter,
        context_fn: Callable[[], RobotContext],
        apply_fn: Callable[[list[ChemicalImpulse]], Awaitable[None] | None],
    ) -> None:
        self._interpreter = interpreter
        self._context_fn = context_fn
        self._apply_fn = apply_fn

    async def route(self, event: UserText) -> None:
        """Interpret the event and apply the resulting impulses."""
        context = self._context_fn()
        impulses = await self._interpreter.interpret(event, context)
        if impulses:
            result = self._apply_fn(impulses)
            if asyncio.iscoroutine(result):
                await result
