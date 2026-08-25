"""ExpressionOutput protocol — abstract interface for emotion expression."""

from __future__ import annotations

from typing import Protocol

from kindalive.emotions.emotion_vector import EmotionVector
from kindalive.engine.chemicals import ChemicalState


class ExpressionOutput(Protocol):
    async def express(self, emotions: EmotionVector, chemicals: ChemicalState) -> str:
        ...
