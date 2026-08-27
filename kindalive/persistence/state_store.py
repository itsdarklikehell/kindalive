"""StateStore — save/load robot state for persistence across restarts.

Serializes:
- Chemical levels (current concentrations)
- Baselines (including any runtime drift)
- Seed chemistry config
- Active sustained impulses
- Saturation tracking data

Format: JSON for simplicity and human readability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kindalive.engine.chemicals import Chemical
from kindalive.engine.impulse import ActiveSustainedImpulse, ChemicalImpulse
from kindalive.engine.neurochemical_engine import NeurochemicalEngine


def serialize_engine(engine: NeurochemicalEngine) -> dict[str, Any]:
    """Serialize engine state to a JSON-compatible dict."""
    state = engine.state

    # Chemical levels
    levels = {chem.value: state.get(chem) for chem in Chemical}

    # Current baselines (may have drifted from seed)
    baselines = {chem.value: state.baseline(chem) for chem in Chemical}

    # Half-lives
    half_lives = {chem.value: state.half_life(chem) for chem in Chemical}

    # Seed chemistry
    seed = engine.seed
    seed_data = seed.to_dict()

    # Active sustained impulses
    sustained = []
    for active in engine._sustained:
        sustained.append({
            "chemical": active.impulse.chemical.value,
            "delta": active.impulse.delta,
            "duration_seconds": active.impulse.duration_seconds,
            "source_id": active.impulse.source_id,
            "source_label": active.impulse.source_label,
            "remaining_seconds": active.remaining_seconds,
            "rate_per_second": active.rate_per_second,
        })

    return {
        "version": 1,
        "levels": levels,
        "baselines": baselines,
        "half_lives": half_lives,
        "seed": seed_data,
        "sustained_impulses": sustained,
    }


def deserialize_engine(
    data: dict[str, Any],
    engine: NeurochemicalEngine,
) -> None:
    """Restore engine state from a serialized dict.

    Mutates the provided engine in place.
    """
    version = data.get("version", 1)
    if version != 1:
        raise ValueError(f"Unknown state version: {version}")

    # Restore levels
    for chem_name, level in data.get("levels", {}).items():
        chem = Chemical.from_string(chem_name)
        engine.state.set(chem, float(level))

    # Restore baselines (may have drifted)
    for chem_name, baseline in data.get("baselines", {}).items():
        chem = Chemical.from_string(chem_name)
        engine.state.set_baseline(chem, float(baseline))

    # Restore sustained impulses
    engine._sustained.clear()
    for imp_data in data.get("sustained_impulses", []):
        chem = Chemical.from_string(imp_data["chemical"])
        impulse = ChemicalImpulse(
            chemical=chem,
            delta=float(imp_data["delta"]),
            duration_seconds=float(imp_data["duration_seconds"]),
            source_id=str(imp_data.get("source_id", "")),
            source_label=str(imp_data.get("source_label", "")),
        )
        active = ActiveSustainedImpulse(
            impulse=impulse,
            remaining_seconds=float(imp_data["remaining_seconds"]),
            rate_per_second=float(imp_data["rate_per_second"]),
        )
        engine._sustained.append(active)


def save_state(engine: NeurochemicalEngine, path: str | Path) -> None:
    """Save engine state to a JSON file."""
    data = serialize_engine(engine)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_state(engine: NeurochemicalEngine, path: str | Path) -> None:
    """Load engine state from a JSON file. Mutates engine in place."""
    path = Path(path)
    data = json.loads(path.read_text())
    deserialize_engine(data, engine)
