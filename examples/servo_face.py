"""Map Kindalive's FaceState onto hobby-servo angles for an animatronic face.

Each of the 12 FACS-named muscles becomes one servo channel. The mapping
is a plain linear interpolation between a per-channel ``rest`` and
``full`` angle, so tuning an animatronic head is just editing the table
below — no engine knowledge required.

With ``adafruit-servokit`` installed (PCA9685 16-channel board on I2C)
the angles are written to real servos; otherwise they print to the
terminal so you can sanity-check the mapping anywhere.

Hardware setup (Raspberry Pi + PCA9685):
    Enable I2C:  sudo raspi-config -> Interface Options -> I2C
    Install:     pip install adafruit-circuitpython-servokit

Run:
    python3 examples/servo_face.py                 # joy -> decay demo
    python3 examples/servo_face.py --mood anger
"""

from __future__ import annotations

import argparse
import time

from led_matrix_face import MOODS  # same demo impulse recipes

from kindalive.engine.clock import ManualClock
from kindalive.engine.neurochemical_engine import NeurochemicalEngine
from kindalive.expression.face import FaceProjection, FaceState

# channel -> (muscle attribute, rest angle, full-contraction angle).
# Angles can run "backwards" (rest > full) for mirrored linkages.
SERVO_MAP: dict[int, tuple[str, float, float]] = {
    0: ("brow_inner_raise", 90.0, 130.0),
    1: ("brow_outer_raise", 90.0, 130.0),
    2: ("brow_lower", 90.0, 50.0),
    3: ("eyelid_upper_raise", 80.0, 140.0),
    4: ("eyelid_lower_tighten", 90.0, 60.0),
    5: ("cheek_raise", 90.0, 120.0),
    6: ("nose_wrinkle", 90.0, 110.0),
    7: ("lip_corner_pull", 90.0, 150.0),
    8: ("lip_corner_depress", 90.0, 30.0),
    9: ("jaw_open", 70.0, 160.0),
    10: ("lip_pucker", 90.0, 115.0),
    11: ("lip_press", 90.0, 70.0),
}


def face_to_angles(face: FaceState) -> dict[int, float]:
    """Linear-map each muscle's [0,1] activation to its servo angle."""
    angles = {}
    for channel, (muscle, rest, full) in SERVO_MAP.items():
        activation = getattr(face, muscle)
        angles[channel] = rest + (full - rest) * activation
    return angles


class TerminalServos:
    """Fallback: prints a bar per channel instead of moving hardware."""

    def apply(self, angles: dict[int, float]) -> None:
        print("\033[H\033[2J", end="")
        for channel, angle in sorted(angles.items()):
            muscle = SERVO_MAP[channel][0]
            bar = "#" * int(angle / 6)
            print(f"ch{channel:2d} {muscle:<22s} {angle:6.1f}°  {bar}")
        print("(install adafruit-circuitpython-servokit + a PCA9685 to drive servos)")


class Pca9685Servos:
    """Real driver: writes angles to a PCA9685 via adafruit-servokit."""

    def __init__(self) -> None:
        from adafruit_servokit import ServoKit

        self._kit = ServoKit(channels=16)

    def apply(self, angles: dict[int, float]) -> None:
        for channel, angle in angles.items():
            self._kit.servo[channel].angle = max(0.0, min(180.0, angle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mood", choices=sorted(MOODS), default="joy")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    try:
        servos: TerminalServos | Pca9685Servos = Pca9685Servos()
    except (ImportError, ValueError, OSError):
        servos = TerminalServos()

    clock = ManualClock()
    engine = NeurochemicalEngine(clock=clock)
    engine.apply_impulses(MOODS[args.mood])

    fps = 10.0
    for _ in range(int(args.seconds * fps)):
        clock.advance(1.0 / fps)
        engine.advance(dt=1.0 / fps)
        face = FaceProjection.compute(engine.state)
        servos.apply(face_to_angles(face))
        time.sleep(1.0 / fps)


if __name__ == "__main__":
    main()
