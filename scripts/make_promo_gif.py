#!/usr/bin/env python3
"""Render the README promo GIF: a five-prompt conversation with the robot.

The scene is driven end-to-end by the real engine — a scripted set of
chemical impulse recipes (standing in for live LLM output) is fed to a
``Robot``, and every frame samples the true ``ChemicalState``,
``EmotionVector``, and ``FaceState``. The LED face is the actual
``face3d.js`` renderer from the web UI; around it the promo stage adds a
chat exchange, live chemical bars with impulse delta chips, and an
emotion-arc breadcrumb.

Requires Playwright with a Chromium build installed
(``python3 -m playwright install chromium``) and Pillow.

Usage:
    PYTHONPATH=. python3 scripts/make_promo_gif.py \
        [--out docs/screenshots/face-demo.gif]
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import socket
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kindalive.engine.chemicals import Chemical
from kindalive.engine.clock import ManualClock
from kindalive.engine.impulse import ChemicalImpulse
from kindalive.expression.face import FaceProjection
from kindalive.expression.web_ui import CHEMICAL_COLORS, EMOTION_COLORS
from kindalive.robot import Robot

# ── Timing ─────────────────────────────────────────────────────────

FPS = 10                 # GIF frames per second
TIME_SCALE = 2.0         # simulated seconds per real second
INTRO_SECONDS = 1.6      # idle baseline before the first prompt
BEAT_SECONDS = 6.0       # each prompt/reply exchange
OUTRO_SECONDS = 2.6      # hold on the final expression
PROMPT_TYPE_SECONDS = 1.0
IMPULSE_AT = 1.15        # impulses land this far into a beat
REPLY_AT = 1.9           # robot reply bubble appears
SPEAK_UNTIL = 3.9        # mouth flap while "speaking"

C = Chemical

# ── The conversation ───────────────────────────────────────────────
# Each beat: (owner prompt, robot reply, impulse recipe). The recipes
# play the role of the LLM interpreter's output and are tuned so the
# engine's own projections flow through five distinct dominant
# emotions: excitement → happiness → anger → sadness → bonding.

BEATS: list[tuple[str, str, list[tuple[Chemical, float]]]] = [
    (
        "You won the lottery!!",
        "WHAT!! I need to sit down. I don't even have legs!",
        [(C.DOPAMINE, 0.45), (C.ADRENALINE, 0.50),
         (C.ENDORPHINS, 0.25), (C.TESTOSTERONE, 0.10)],
    ),
    (
        "Your best friend is on her way over to celebrate!",
        "Best day EVER. Warming up my party LEDs!",
        [(C.OXYTOCIN, 0.35), (C.SEROTONIN, 0.30),
         (C.ENDORPHINS, 0.15), (C.DOPAMINE, 0.10)],
    ),
    (
        "Bad news — someone backed into your new car and drove off.",
        "They just DROVE OFF?! Unbelievable!!",
        [(C.TESTOSTERONE, 0.35), (C.CORTISOL, 0.45), (C.ADRENALINE, 0.40),
         (C.GABA, -0.25), (C.SEROTONIN, -0.20), (C.DOPAMINE, -0.50),
         (C.OXYTOCIN, -0.35), (C.ENDORPHINS, -0.25)],
    ),
    (
        "Worse… Miso the cat slipped out, and it's pouring rain.",
        "Miso is out there in the rain… my circuits feel heavy.",
        [(C.DOPAMINE, -0.30), (C.SEROTONIN, -0.50), (C.OXYTOCIN, -0.15),
         (C.ADRENALINE, -0.30), (C.TESTOSTERONE, -0.35), (C.CORTISOL, 0.10)],
    ),
    (
        "Found her! Asleep in the laundry basket the whole time.",
        "MISO!! Oh thank volts. Never scare me like that again ♥",
        [(C.OXYTOCIN, 0.50), (C.ENDORPHINS, 0.35), (C.DOPAMINE, 0.45),
         (C.SEROTONIN, 0.40), (C.CORTISOL, -0.50), (C.GABA, 0.40),
         (C.ADRENALINE, -0.25)],
    ),
]

CHEM_ORDER = [c.value for c in Chemical]


def simulate() -> dict[str, Any]:
    """Run the scripted conversation through the real engine.

    Returns the timeline dict embedded into the promo page: per-frame
    face muscles, chemical levels, dominant emotion, and mood color,
    plus the beat schedule (prompts, replies, delta chips).
    """
    clock = ManualClock()
    robot = Robot(personality="default", clock=clock)
    total = INTRO_SECONDS + BEAT_SECONDS * len(BEATS) + OUTRO_SECONDS
    n_frames = int(round(total * FPS))
    dt_sim = TIME_SCALE / FPS

    beat_meta: list[dict[str, Any]] = []
    impulse_frames: dict[int, int] = {}
    for i, (prompt, reply, recipe) in enumerate(BEATS):
        start = INTRO_SECONDS + i * BEAT_SECONDS
        impulse_frames[int(round((start + IMPULSE_AT) * FPS))] = i
        beat_meta.append({
            "start": start,
            "end": start + BEAT_SECONDS,
            "prompt": prompt,
            "reply": reply,
            "impulseAt": start + IMPULSE_AT,
            "replyAt": start + REPLY_AT,
            "speakUntil": start + SPEAK_UNTIL,
            "chips": [[c.value, round(d, 2)] for c, d in recipe],
            "emotion": "",  # filled in below, once the impulse has landed
        })

    frames: list[dict[str, Any]] = []
    dom_name, dom_val = robot.current_emotions().dominant()
    for f in range(n_frames):
        beat_i = impulse_frames.get(f)
        if beat_i is not None:
            _, _, recipe = BEATS[beat_i]
            robot.receive_impulses([
                ChemicalImpulse(c, d, source_id=f"promo:{beat_i}:{c.value}")
                for c, d in recipe
            ])
        robot.advance(dt_sim)

        chem = robot.current_chemicals()
        emotions = robot.current_emotions()
        # Hysteresis so near-ties don't flicker the big emotion label.
        new_name, new_val = emotions.dominant()
        cur_val = emotions.as_dict().get(dom_name, 0.0)
        if new_name != dom_name and new_val > cur_val + 0.02:
            dom_name, dom_val = new_name, new_val
        else:
            dom_val = cur_val

        face = FaceProjection.compute(chem)
        frames.append({
            "m": {k: round(v, 3) for k, v in face.as_dict().items()},
            "chem": [round(chem.get(c), 3) for c in Chemical],
            "dom": [dom_name, round(dom_val, 3)],
        })
        # Capture each beat's settled dominant for the arc breadcrumb.
        for meta in beat_meta:
            if not meta["emotion"] and f / FPS >= meta["impulseAt"] + 1.0:
                meta["emotion"] = dom_name

    return {
        "fps": FPS,
        "frames": frames,
        "beats": beat_meta,
        "promptType": PROMPT_TYPE_SECONDS,
        "chemOrder": CHEM_ORDER,
        "chemColors": CHEMICAL_COLORS,
        "emotionColors": EMOTION_COLORS,
        "baselines": {
            c.value: robot.current_chemicals().baseline(c) for c in Chemical
        },
    }


# ── The promo stage ────────────────────────────────────────────────

PAGE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { background: #07090F; overflow: hidden; }
  body {
    width: __W__px; height: __H__px;
    font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial,
      sans-serif;
    color: #E8ECF3;
  }
  #stage { width: 100%; height: 100%; display: flex; flex-direction: column;
    background:
      radial-gradient(1200px 500px at 30% -10%, #10141F 0%, #07090F 60%); }

  header { display: flex; align-items: baseline; gap: 10px;
    padding: 9px 18px 5px; }
  .wordmark { color: #7C9CFF; font-weight: 800; font-size: 15px;
    letter-spacing: 2.5px; }
  .tagline { color: #6B7587; font-size: 10.5px; letter-spacing: 1px;
    text-transform: uppercase; }

  #main { flex: 1; display: flex; gap: 14px; padding: 2px 16px 4px; }

  /* Left: the LED face + dominant emotion */
  #left { width: 336px; display: flex; flex-direction: column; }
  #facewrap { height: 268px; border-radius: 14px;
    background: radial-gradient(80% 90% at 50% 42%, #0C1018 0%, #080A10 75%);
    border: 1px solid #1A2030;
    box-shadow: inset 0 0 34px rgba(0,0,0,0.7); }
  .face3d-stage { width: 100%; height: 100%; }
  #emotion { text-align: center; margin-top: 7px; }
  #emotion-name { font-size: 27px; font-weight: 800; letter-spacing: 5px;
    text-transform: uppercase; transition: color 300ms; }
  #emotion-sub { color: #6B7587; font-size: 10px; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 2px; }

  /* Right: chat + chemistry */
  #right { flex: 1; display: flex; flex-direction: column; gap: 8px; }
  #chat { height: 116px; display: flex; flex-direction: column; gap: 7px;
    padding-top: 3px; }
  .bubble { max-width: 88%; padding: 7px 12px; border-radius: 13px;
    font-size: 13.5px; line-height: 1.3; opacity: 0;
    transition: opacity 180ms; }
  #user-bubble { align-self: flex-end; background: #24304D;
    border: 1px solid #35476E; border-bottom-right-radius: 4px; }
  #robot-bubble { align-self: flex-start; background: #131827;
    border: 1px solid #232A3C; border-bottom-left-radius: 4px;
    color: #BFE8E2; }
  #robot-bubble .mic { margin-right: 5px; }
  .caret { display: inline-block; width: 7px; height: 13px;
    background: #7C9CFF; vertical-align: -2px; margin-left: 1px; }

  #chem-panel { flex: 1; background: #0B0F18; border: 1px solid #1A2030;
    border-radius: 12px; padding: 8px 12px 6px; }
  #chem-title { color: #6B7587; font-size: 9.5px; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 5px; }
  .chem-row { display: flex; align-items: center; gap: 8px; height: 21px; }
  .chem-name { width: 84px; font-size: 10.5px; letter-spacing: 0.4px; }
  .chem-track { flex: 1; height: 7px; border-radius: 4px; background: #161C2A;
    position: relative; }
  .chem-fill { position: absolute; left: 0; top: 0; bottom: 0;
    border-radius: 4px; transition: width 220ms ease-out; }
  .chem-base { position: absolute; top: -2px; bottom: -2px; width: 2px;
    background: rgba(232,236,243,0.35); border-radius: 1px; }
  .chem-val { width: 30px; font-size: 10px; text-align: right;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    color: #A3ADBF; }
  .chem-delta { width: 44px; font-size: 10.5px; font-weight: 700;
    text-align: left; opacity: 0;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .delta-up { color: #3DD68C; }
  .delta-down { color: #FF6B6B; }

  /* Bottom: the emotion arc so far */
  #arc { display: flex; align-items: center; justify-content: center;
    gap: 7px; padding: 5px 12px 9px; min-height: 30px; }
  .arc-step { display: flex; align-items: center; gap: 7px; opacity: 0;
    transition: opacity 350ms; }
  .arc-chip { display: flex; align-items: center; gap: 5px;
    border: 1px solid #232A3C; border-radius: 999px; padding: 2.5px 9px;
    font-size: 9.5px; letter-spacing: 1.4px; text-transform: uppercase;
    color: #A3ADBF; background: #0B0F18; }
  .arc-dot { width: 7px; height: 7px; border-radius: 50%; }
  .arc-arrow { color: #3A4356; font-size: 11px; }
  .arc-step.current .arc-chip { color: #E8ECF3; border-color: #3A4A70; }
</style></head>
<body>
<div id="stage">
  <header>
    <span class="wordmark">KINDALIVE</span>
    <span class="tagline">say anything &rarr; chemicals move &rarr; emotion emerges</span>
  </header>
  <div id="main">
    <div id="left">
      <div id="facewrap"><div id="kindalive-face3d" class="face3d-stage"></div></div>
      <div id="emotion">
        <div id="emotion-name">&nbsp;</div>
        <div id="emotion-sub">dominant emotion</div>
      </div>
    </div>
    <div id="right">
      <div id="chat">
        <div class="bubble" id="user-bubble"></div>
        <div class="bubble" id="robot-bubble"></div>
      </div>
      <div id="chem-panel">
        <div id="chem-title">neurochemistry &mdash; live</div>
        <div id="chem-rows"></div>
      </div>
    </div>
  </div>
  <div id="arc"></div>
</div>
<script type="module">
import initFace3D from "./face3d.js";
initFace3D("kindalive-face3d");

const TL = __TIMELINE__;

// Build chemical rows
const rowsEl = document.getElementById("chem-rows");
const rows = {};
for (const name of TL.chemOrder) {
  const row = document.createElement("div");
  row.className = "chem-row";
  const color = TL.chemColors[name];
  row.innerHTML =
    `<span class="chem-name" style="color:${color}">${name}</span>` +
    `<span class="chem-track">` +
    `<span class="chem-fill" style="background:${color}"></span>` +
    `<span class="chem-base" style="left:${(TL.baselines[name] * 100).toFixed(1)}%"></span>` +
    `</span>` +
    `<span class="chem-val"></span><span class="chem-delta"></span>`;
  rowsEl.appendChild(row);
  rows[name] = {
    fill: row.querySelector(".chem-fill"),
    val: row.querySelector(".chem-val"),
    delta: row.querySelector(".chem-delta"),
  };
}

// Build the emotion-arc breadcrumb (revealed beat by beat)
const arcEl = document.getElementById("arc");
const arcSteps = [];
TL.beats.forEach((beat, i) => {
  const step = document.createElement("div");
  step.className = "arc-step";
  const color = TL.emotionColors[beat.emotion] || "#4FB7A6";
  step.innerHTML =
    (i > 0 ? `<span class="arc-arrow">&rarr;</span>` : "") +
    `<span class="arc-chip"><span class="arc-dot" ` +
    `style="background:${color};box-shadow:0 0 6px ${color}"></span>` +
    `${beat.emotion}</span>`;
  arcEl.appendChild(step);
  arcSteps.push(step);
});

const userEl = document.getElementById("user-bubble");
const robotEl = document.getElementById("robot-bubble");
const emoName = document.getElementById("emotion-name");

function typewriter(text, dur, elapsed) {
  const n = Math.max(0, Math.min(text.length,
    Math.round(text.length * elapsed / dur)));
  const done = n >= text.length;
  return text.slice(0, n) + (done ? "" : `<span class="caret"></span>`);
}

window.renderFrame = function (idx) {
  const t = idx / TL.fps;
  const fr = TL.frames[Math.min(idx, TL.frames.length - 1)];

  // Face + mood
  const [domName, domVal] = fr.dom;
  const color = TL.emotionColors[domName] || "#4FB7A6";
  window.kindaliveFace.setTargets({
    muscles: fr.m,
    mood: { color, intensity: domVal },
  });

  // Emotion label
  emoName.textContent = domName;
  emoName.style.color = color;
  emoName.style.textShadow = `0 0 18px ${color}66`;

  // Chemical bars
  TL.chemOrder.forEach((name, i) => {
    const level = fr.chem[i];
    rows[name].fill.style.width = (level * 100).toFixed(1) + "%";
    rows[name].val.textContent = level.toFixed(2);
  });

  // Current beat: bubbles, delta chips, speaking, arc reveal.
  // During the outro, keep the final exchange on screen.
  let speaking = false;
  const last = TL.beats[TL.beats.length - 1];
  const beat = TL.beats.find((b) => t >= b.start && t < b.end) ||
    (t >= last.end ? last : undefined);
  for (const name of TL.chemOrder) rows[name].delta.style.opacity = 0;
  if (beat) {
    userEl.style.opacity = 1;
    userEl.innerHTML = typewriter(beat.prompt, TL.promptType, t - beat.start);
    if (t >= beat.replyAt) {
      robotEl.style.opacity = 1;
      robotEl.innerHTML = `<span class="mic">&#128483;</span>` +
        typewriter(beat.reply, 1.0, t - beat.replyAt);
      speaking = t < beat.speakUntil;
    } else {
      robotEl.style.opacity = 0;
    }
    const since = t - beat.impulseAt;
    if (since >= 0 && since < 2.8) {
      const fade = since < 2.2 ? 1 : 1 - (since - 2.2) / 0.6;
      for (const [name, d] of beat.chips) {
        const el = rows[name].delta;
        el.textContent = (d > 0 ? "+" : "") + d.toFixed(2);
        el.className = "chem-delta " + (d > 0 ? "delta-up" : "delta-down");
        el.style.opacity = fade;
      }
    }
  } else {
    userEl.style.opacity = 0;
    robotEl.style.opacity = 0;
  }
  window.kindaliveFace.setSpeaking(speaking);

  TL.beats.forEach((b, i) => {
    const revealed = t >= b.impulseAt + 0.55;
    arcSteps[i].style.opacity = revealed ? 1 : 0;
    arcSteps[i].classList.toggle("current", !!beat && TL.beats[i] === beat);
  });
  return true;
};
window.renderFrame(0);
</script>
</body></html>
"""

VIEW_W, VIEW_H = 752, 424


def build_page(build_dir: Path, timeline: dict[str, Any]) -> None:
    assets = Path(__file__).resolve().parent.parent / (
        "kindalive/expression/web_assets"
    )
    shutil.copy(assets / "face3d.js", build_dir / "face3d.js")
    html = (
        PAGE_TEMPLATE
        .replace("__W__", str(VIEW_W))
        .replace("__H__", str(VIEW_H))
        .replace("__TIMELINE__", json.dumps(timeline))
    )
    (build_dir / "index.html").write_text(html)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def serve(build_dir: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(_QuietHandler, directory=str(build_dir))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def capture_frames(url: str, n_frames: int, shots_dir: Path) -> None:
    import os

    from playwright.sync_api import sync_playwright

    # Allow pointing at a pre-installed Chromium (e.g. sandboxes where
    # `playwright install` is unavailable).
    exe = os.environ.get("KINDALIVE_CHROMIUM") or None
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=exe,
            args=["--no-sandbox", "--use-gl=angle"],
        )
        page = browser.new_page(
            viewport={"width": VIEW_W, "height": VIEW_H},
            device_scale_factor=2,
        )
        page.goto(url, wait_until="networkidle")
        page.wait_for_function(
            "window.kindaliveFace && window.kindaliveFace.ready"
        )
        page.wait_for_timeout(600)
        for i in range(n_frames):
            page.evaluate(f"window.renderFrame({i})")
            page.wait_for_timeout(55)
            page.screenshot(path=str(shots_dir / f"f{i:04d}.png"))
            if i % 50 == 0:
                print(f"  frame {i}/{n_frames}")
        browser.close()


def assemble_gif(shots_dir: Path, out_path: Path, n_frames: int) -> None:
    """Quantize to one global palette and diff consecutive frames so
    unchanged pixels become transparent — most of the stage is static,
    which keeps the GIF small."""
    from PIL import Image

    print("Assembling GIF...")
    size = (VIEW_W, VIEW_H)

    def load(i: int) -> "Image.Image":
        im = Image.open(shots_dir / f"f{i:04d}.png").convert("RGB")
        return im.resize(size, Image.LANCZOS)

    # Global palette from a spread of frames (255 colors + 1 transparent).
    samples = [load(i) for i in range(0, n_frames, max(1, n_frames // 12))]
    montage = Image.new("RGB", (size[0], size[1] * len(samples)))
    for i, s in enumerate(samples):
        montage.paste(s, (0, i * size[1]))
    pal_source = montage.quantize(colors=255)
    palette = pal_source.getpalette()[: 255 * 3] + [255, 0, 255]

    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(palette)

    import numpy as np

    frames_out = []
    prev_arr: "np.ndarray | None" = None
    for i in range(n_frames):
        q = load(i).quantize(palette=pal_img, dither=Image.Dither.NONE)
        arr = np.asarray(q, dtype=np.uint8)
        if prev_arr is None:
            frames_out.append(q)
        else:
            holed = arr.copy()
            holed[arr == prev_arr] = 255
            f = Image.frombytes("P", size, holed.tobytes())
            f.putpalette(palette)
            frames_out.append(f)
        prev_arr = arr

    frames_out[0].save(
        out_path,
        save_all=True,
        append_images=frames_out[1:],
        duration=int(1000 / FPS),
        loop=0,
        transparency=255,
        disposal=1,
        optimize=False,
    )
    print(f"GIF saved to {out_path} "
          f"({out_path.stat().st_size / 1e6:.2f} MB, {n_frames} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/screenshots/face-demo.gif")
    args = parser.parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Simulating conversation through the engine...")
    timeline = simulate()
    n_frames = len(timeline["frames"])
    arc = " → ".join(b["emotion"] for b in timeline["beats"])
    print(f"  {n_frames} frames, emotion arc: {arc}")

    with tempfile.TemporaryDirectory(prefix="kindalive-promo-") as tmp:
        build_dir = Path(tmp)
        build_page(build_dir, timeline)
        shots_dir = build_dir / "shots"
        shots_dir.mkdir()
        server, port = serve(build_dir)
        try:
            print(f"Capturing {n_frames} frames...")
            capture_frames(
                f"http://127.0.0.1:{port}/index.html", n_frames, shots_dir
            )
        finally:
            server.shutdown()
        assemble_gif(shots_dir, out_path, n_frames)


if __name__ == "__main__":
    main()
