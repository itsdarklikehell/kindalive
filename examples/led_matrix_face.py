"""Drive a physical 8x8 LED matrix (MAX7219) from Kindalive's FaceState.

The 12-muscle ``FaceState`` is renderer-agnostic — the web UI draws it as
a big dot-matrix canvas, and this script draws it on real hardware: a
MAX7219 8x8 LED matrix wired to a Raspberry Pi's SPI pins.

No hardware? It still runs: without ``luma.led_matrix`` installed the
frames render in your terminal, so you can develop the mapping on a
laptop and deploy the same file to the Pi.

Hardware setup (Raspberry Pi):
    VCC -> 5V, GND -> GND, DIN -> MOSI (pin 19),
    CS -> CE0 (pin 24), CLK -> SCLK (pin 23)
    Enable SPI:  sudo raspi-config  ->  Interface Options -> SPI
    Install:     pip install luma.led_matrix

Run:
    python3 examples/led_matrix_face.py            # joy -> decay demo
    python3 examples/led_matrix_face.py --mood fear
"""

from __future__ import annotations

import argparse
import time

from kindalive.engine.chemicals import Chemical
from kindalive.engine.clock import ManualClock
from kindalive.engine.impulse import ChemicalImpulse
from kindalive.engine.neurochemical_engine import NeurochemicalEngine
from kindalive.expression.face import FaceProjection, FaceState

# Impulse recipes for the demo moods. In a real robot these come from
# the LLM interpreter; here we inject them directly.
MOODS: dict[str, list[ChemicalImpulse]] = {
    "joy": [
        ChemicalImpulse(Chemical.DOPAMINE, delta=0.4),
        ChemicalImpulse(Chemical.ENDORPHINS, delta=0.3),
    ],
    "fear": [
        ChemicalImpulse(Chemical.CORTISOL, delta=0.45),
        ChemicalImpulse(Chemical.ADRENALINE, delta=0.4),
    ],
    "anger": [
        ChemicalImpulse(Chemical.TESTOSTERONE, delta=0.4),
        ChemicalImpulse(Chemical.CORTISOL, delta=0.3),
        ChemicalImpulse(Chemical.DOPAMINE, delta=-0.2),
    ],
    "sadness": [
        ChemicalImpulse(Chemical.DOPAMINE, delta=-0.35),
        ChemicalImpulse(Chemical.SEROTONIN, delta=-0.3),
    ],
}


def face_to_rows(face: FaceState) -> list[int]:
    """Map the 12-muscle FaceState onto 8 row-bitmasks (bit 7 = left).

    Layout on the 8x8 grid:
        row 0-1: brows   (raise -> row 0, lower -> row 1, inner tilt)
        row 2-3: eyes    (wide open -> both rows, tightened -> row 3 only)
        row 5-7: mouth   (corners curl up/down, jaw opens downward)
    """
    rows = [0] * 8

    # --- Brows: two 3-dot dashes. AU1/AU2 lift them, AU4 pulls them
    # down and inward (the angry knit).
    raise_amt = max(face.brow_inner_raise, face.brow_outer_raise)
    brow_row = 0 if raise_amt > face.brow_lower else 1
    if face.brow_lower > 0.25:
        # Knitted: shift each brow one column toward the nose.
        rows[brow_row] |= 0b01100110
    else:
        rows[brow_row] |= 0b11100111

    # --- Eyes: 2x2 blocks at columns 1-2 and 5-6. AU5 widens (two rows),
    # AU7 tightens to a squint (bottom row only).
    open_amt = face.eyelid_upper_raise - face.eyelid_lower_tighten
    if open_amt > -0.15:
        rows[2] |= 0b01100110
    rows[3] |= 0b01100110

    # --- Mouth: corners at columns 0/7, center span in between.
    smile = face.lip_corner_pull
    frown = face.lip_corner_depress
    jaw = face.jaw_open

    if jaw > 0.35:
        # Open mouth: a hollow box, taller with wider jaw.
        rows[5] |= 0b00111100
        rows[6] |= 0b00100100 if jaw < 0.7 else 0b01000010
        rows[7] |= 0b00111100
    elif smile > frown and smile > 0.2:
        # Smile: corners up (row 5), center dips to row 6.
        rows[5] |= 0b10000001
        rows[6] |= 0b01111110
    elif frown > 0.2:
        # Frown: corners down (row 7), center arcs at row 6.
        rows[6] |= 0b01111110
        rows[7] |= 0b10000001
    else:
        # Neutral / pressed lips: a straight line, thinner when AU24 high.
        rows[6] |= 0b00111100 if face.lip_press > 0.5 else 0b01111110

    return rows


class TerminalMatrix:
    """Fallback renderer: prints the 8x8 frame with block characters."""

    def draw(self, rows: list[int]) -> None:
        print("\033[H\033[2J", end="")  # clear screen, cursor home
        for row in rows:
            print("".join("██" if row & (1 << (7 - c)) else "··" for c in range(8)))
        print("(install luma.led_matrix + wire a MAX7219 for the real thing)")


class Max7219Matrix:
    """Real renderer: pushes frames to a MAX7219 over SPI."""

    def __init__(self) -> None:
        from luma.core.interface.serial import noop, spi
        from luma.led_matrix.device import max7219

        self._device = max7219(spi(port=0, device=0, gpio=noop()))

    def draw(self, rows: list[int]) -> None:
        from luma.core.render import canvas

        with canvas(self._device) as draw:
            for y, row in enumerate(rows):
                for x in range(8):
                    if row & (1 << (7 - x)):
                        draw.point((x, y), fill="white")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mood", choices=sorted(MOODS), default="joy")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to run the decay animation")
    args = parser.parse_args()

    try:
        matrix: TerminalMatrix | Max7219Matrix = Max7219Matrix()
    except ImportError:
        matrix = TerminalMatrix()

    # Deterministic simulation clock; wall time only paces the frames.
    clock = ManualClock()
    engine = NeurochemicalEngine(clock=clock)
    engine.apply_impulses(MOODS[args.mood])

    fps = 8.0
    steps = int(args.seconds * fps)
    for _ in range(steps):
        clock.advance(1.0 / fps)
        engine.advance(dt=1.0 / fps)
        face = FaceProjection.compute(engine.state)
        matrix.draw(face_to_rows(face))
        time.sleep(1.0 / fps)


if __name__ == "__main__":
    main()
