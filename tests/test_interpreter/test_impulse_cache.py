"""Tests for ImpulseCache."""

from kindalive.engine.chemicals import Chemical
from kindalive.engine.impulse import ChemicalImpulse
from kindalive.interpreter.impulse_cache import (
    BACKGROUND_TTL,
    REALTIME_TTL,
    ImpulseCache,
)
from kindalive.interpreter.text_input import UserText


def _event(urgency: str = "background") -> UserText:
    return UserText(
        source="sports",
        event_type="goal_scored",
        summary="Goal!",
        urgency=urgency,
    )


def _impulses() -> list[ChemicalImpulse]:
    return [ChemicalImpulse(Chemical.DOPAMINE, delta=0.2)]


def test_cache_miss():
    cache = ImpulseCache()
    result = cache.get(_event(), "default", 1.0, now=0.0)
    assert result is None
    assert cache.misses == 1


def test_cache_hit():
    cache = ImpulseCache()
    ev = _event()
    cache.put(ev, "default", 1.0, _impulses(), now=0.0)
    result = cache.get(ev, "default", 1.0, now=1.0)
    assert result is not None
    assert len(result) == 1
    assert cache.hits == 1


def test_cache_ttl_expiry():
    cache = ImpulseCache()
    ev = _event()
    cache.put(ev, "default", 1.0, _impulses(), now=0.0)
    # Background TTL is 4 hours
    result = cache.get(ev, "default", 1.0, now=BACKGROUND_TTL + 1)
    assert result is None
    assert cache.misses == 1


def test_realtime_ttl():
    cache = ImpulseCache()
    ev = _event(urgency="realtime")
    cache.put(ev, "default", 1.0, _impulses(), now=0.0)
    # Still valid within 1 hour
    result = cache.get(ev, "default", 1.0, now=REALTIME_TTL - 1)
    assert result is not None
    # Expired after 1 hour
    result = cache.get(ev, "default", 1.0, now=REALTIME_TTL + 1)
    assert result is None


def test_lru_eviction():
    cache = ImpulseCache(max_size=2)
    ev1 = UserText(source="a", event_type="1", summary="one")
    ev2 = UserText(source="b", event_type="2", summary="two")
    ev3 = UserText(source="c", event_type="3", summary="three")

    cache.put(ev1, "default", 1.0, _impulses(), now=0.0)
    cache.put(ev2, "default", 1.0, _impulses(), now=0.0)
    cache.put(ev3, "default", 1.0, _impulses(), now=0.0)

    assert cache.size == 2
    # ev1 (LRU) should be evicted
    assert cache.get(ev1, "default", 1.0, now=1.0) is None
    assert cache.get(ev3, "default", 1.0, now=1.0) is not None


def test_different_personality_different_key():
    cache = ImpulseCache()
    ev = _event()
    cache.put(ev, "cheerful", 1.0, _impulses(), now=0.0)
    result = cache.get(ev, "stoic", 1.0, now=1.0)
    assert result is None  # Different personality = cache miss


def test_affinity_bucketing():
    cache = ImpulseCache()
    ev = _event()
    # Affinity 0.3 = "low" bucket, 1.5 = "high" bucket
    cache.put(ev, "default", 0.3, _impulses(), now=0.0)
    # Same bucket (low) should hit
    result = cache.get(ev, "default", 0.4, now=1.0)
    assert result is not None
    # Different bucket (high) should miss
    result = cache.get(ev, "default", 1.5, now=1.0)
    assert result is None


def test_clear():
    cache = ImpulseCache()
    cache.put(_event(), "default", 1.0, _impulses(), now=0.0)
    assert cache.size == 1
    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0
    assert cache.misses == 0
