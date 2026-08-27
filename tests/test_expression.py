"""Tests for TextExpressionOutput."""

import pytest

from kindalive.emotions.emotion_vector import EmotionVector
from kindalive.engine.chemicals import Chemical, ChemicalState
from kindalive.expression.text_output import TextExpressionOutput


@pytest.fixture
def expression():
    return TextExpressionOutput()


def _make_chemicals(**overrides: float) -> ChemicalState:
    state = ChemicalState()
    for name, val in overrides.items():
        state.set(Chemical[name.upper()], val)
    return state


@pytest.mark.asyncio
async def test_happy_expression(expression):
    emotions = EmotionVector(happiness=0.8, excitement=0.6, calm=0.3,
                             bonding=0.4, sadness=0.0, anxiety=0.0,
                             anger=0.0, euphoria=0.5)
    chemicals = _make_chemicals(dopamine=0.9, serotonin=0.8, adrenaline=0.6)
    text = await expression.express(emotions, chemicals)
    assert "happy" in text.lower() or "excited" in text.lower()


@pytest.mark.asyncio
async def test_anxious_expression(expression):
    emotions = EmotionVector(happiness=0.1, excitement=0.2, calm=0.05,
                             bonding=0.1, sadness=0.2, anxiety=0.8,
                             anger=0.3, euphoria=0.0)
    chemicals = _make_chemicals(cortisol=0.8, adrenaline=0.5, gaba=0.1)
    text = await expression.express(emotions, chemicals)
    assert "anxious" in text.lower() or "angry" in text.lower()


@pytest.mark.asyncio
async def test_neutral_expression(expression):
    emotions = EmotionVector(happiness=0.3, excitement=0.1, calm=0.4,
                             bonding=0.3, sadness=0.05, anxiety=0.05,
                             anger=0.05, euphoria=0.1)
    chemicals = _make_chemicals()
    text = await expression.express(emotions, chemicals)
    assert "calm" in text.lower() or "content" in text.lower() or "happy" in text.lower()


@pytest.mark.asyncio
async def test_leaning_forward_posture(expression):
    emotions = EmotionVector(happiness=0.7, excitement=0.8, calm=0.1,
                             bonding=0.2, sadness=0.0, anxiety=0.0,
                             anger=0.0, euphoria=0.6)
    chemicals = _make_chemicals(adrenaline=0.6, dopamine=0.8)
    text = await expression.express(emotions, chemicals)
    assert "leaning forward" in text.lower()


@pytest.mark.asyncio
async def test_highly_engaged(expression):
    emotions = EmotionVector(happiness=0.5, excitement=0.7, calm=0.2,
                             bonding=0.2, sadness=0.0, anxiety=0.0,
                             anger=0.0, euphoria=0.4)
    chemicals = _make_chemicals(dopamine=0.8)
    text = await expression.express(emotions, chemicals)
    assert "highly engaged" in text.lower()
