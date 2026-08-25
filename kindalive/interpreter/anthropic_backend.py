"""AnthropicBackend — real LLM backend using Claude via the Anthropic SDK.

API key is read from the ANTHROPIC_API_KEY environment variable.
Never hardcode keys in source code.
"""

from __future__ import annotations

import os

try:
    import anthropic
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The Anthropic backend needs the 'anthropic' package. "
        'Install it with: pip install "kindalive[anthropic]"'
    ) from exc



class AnthropicBackend:
    """Calls Claude (Haiku by default) via the Anthropic SDK.

    Args:
        model: Model ID. Defaults to claude-haiku-4-5-20251001.
        api_key: Explicit key. If None, reads ANTHROPIC_API_KEY from env.
        max_tokens: Max response tokens.
        temperature: Sampling temperature (0 = deterministic).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY env var "
                "or pass api_key= to AnthropicBackend."
            )
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def model(self) -> str:
        return self._model

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send prompts to Claude, return the text response.

        Args:
            history: Prior conversation turns ({"role", "content"} dicts,
                oldest first) prepended before the new user message so the
                robot keeps the thread of the discussion.
            max_tokens: Override the default max_tokens for this call.
        """
        messages: list[anthropic.types.MessageParam] = [
            {"role": "user" if turn.get("role") != "assistant" else "assistant",
             "content": turn.get("content", "")}
            for turn in (history or [])
        ]
        messages.append({"role": "user", "content": user_prompt})
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=messages,
        )
        self._call_count += 1
        # Extract text from the response
        for block in message.content:
            if isinstance(block, anthropic.types.TextBlock):
                return block.text
        return ""
