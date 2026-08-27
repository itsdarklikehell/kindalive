"""TextExpressionOutput — Phase 1 human-readable emotion descriptions.

Produces text like:
  "Happy and excited, leaning forward, highly engaged"
  "Anxious but alert, tense posture, scanning"
  "Calm and content, relaxed posture, passively content"
"""

from __future__ import annotations

from typing import Callable

from kindalive.emotions.emotion_vector import EmotionVector
from kindalive.engine.chemicals import Chemical, ChemicalState

# Type alias for rule functions
_RuleFn = Callable[[dict[Chemical, float]], bool]

# Emotion labels used in text output
_EMOTION_WORDS: dict[str, str] = {
    "happiness": "happy",
    "excitement": "excited",
    "anger": "angry",
    "calm": "calm",
    "bonding": "bonded",
    "anxiety": "anxious",
    "sadness": "sad",
    "euphoria": "euphoric",
}

# Posture rules: keyed on description, lambda takes chemical dict
_POSTURE_RULES: list[tuple[str, _RuleFn]] = [
    ("leaning forward", lambda c: c[Chemical.ADRENALINE] > 0.5 and c[Chemical.DOPAMINE] > 0.4),
    ("tense posture", lambda c: c[Chemical.CORTISOL] > 0.5 and c[Chemical.GABA] < 0.3),
    ("restless", lambda c: c[Chemical.ADRENALINE] > 0.4 and c[Chemical.CORTISOL] > 0.4),
    ("relaxed posture", lambda c: c[Chemical.GABA] > 0.5 and c[Chemical.ADRENALINE] < 0.3),
    ("still and steady", lambda c: c[Chemical.GABA] > 0.4 and c[Chemical.SEROTONIN] > 0.4),
]

_ENGAGEMENT_RULES: list[tuple[str, _RuleFn]] = [
    ("highly engaged", lambda c: c[Chemical.ADRENALINE] > 0.4 or c[Chemical.DOPAMINE] > 0.6),
    ("withdrawn", lambda c: c[Chemical.CORTISOL] > 0.5 and c[Chemical.DOPAMINE] < 0.2),
    ("scanning", lambda c: c[Chemical.CORTISOL] > 0.4 and c[Chemical.ADRENALINE] > 0.3),
    ("passively content", lambda c: c[Chemical.SEROTONIN] > 0.5 and c[Chemical.ADRENALINE] < 0.3),
]


class TextExpressionOutput:
    """Produces human-readable text descriptions of emotional state."""

    async def express(self, emotions: EmotionVector, chemicals: ChemicalState) -> str:
        parts: list[str] = []

        # 1. Dominant emotions (top 2-3 above threshold)
        emotion_parts = self._describe_emotions(emotions)
        if emotion_parts:
            parts.append(emotion_parts)

        # 2. Physical posture
        chem_dict = {chem: chemicals.get(chem) for chem in Chemical}
        posture = self._pick_posture(chem_dict)
        if posture:
            parts.append(posture)

        # 3. Engagement level
        engagement = self._pick_engagement(chem_dict)
        if engagement:
            parts.append(engagement)

        return ", ".join(parts) if parts else "neutral"

    def _describe_emotions(self, emotions: EmotionVector) -> str:
        top = emotions.top_n(3)
        # Filter to emotions above a threshold
        significant = [(name, val) for name, val in top if val > 0.15]
        if not significant:
            return "neutral"

        words = [_EMOTION_WORDS[name] for name, _ in significant[:2]]
        if len(words) == 2:
            return f"{words[0].capitalize()} and {words[1]}"
        return words[0].capitalize()

    def _pick_posture(self, chem_dict: dict[Chemical, float]) -> str | None:
        for label, rule in _POSTURE_RULES:
            if rule(chem_dict):
                return label
        return None

    def _pick_engagement(self, chem_dict: dict[Chemical, float]) -> str | None:
        for label, rule in _ENGAGEMENT_RULES:
            if rule(chem_dict):
                return label
        return None
