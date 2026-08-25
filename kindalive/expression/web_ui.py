"""NiceGUI web dashboard for Kindalive — the only UI.

A deliberately small page built around the LED dot-matrix robot face:
type a paragraph, the LLM translates it to chemical impulses, the
chemistry shifts, and the face contorts. Chemical levels and the mix are
shown beside the face; simulation time always runs in real time (with
an optional speed multiplier to watch decay).

Usage:
    python3 -m kindalive.expression.web_ui [--personality PRESET]
        [--llm auto|anthropic|openai|off] [--port PORT]
"""

from __future__ import annotations

import inspect
import json
import random
import time
from typing import Any

try:
    from nicegui import app, ui
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The web dashboard needs the 'nicegui' package. "
        'Install it with: pip install "kindalive[web]"'
    ) from exc

from kindalive.engine.chemicals import Chemical
from kindalive.engine.clock import ManualClock
from kindalive.expression.face import FaceProjection
from kindalive.expression.face_3d import (
    DEFAULT_MOOD_COLOR,
    WEB_ASSETS_DIR,
    boot_js,
    container_html,
    face_payload,
    payload_js,
)
from kindalive.interpreter.llm_interpreter import LLMBackend
from kindalive.robot import Robot

# Newer NiceGUI makes ``sanitize`` a required keyword-only argument on
# ``ui.html`` (and would strip our container's id/class if it ran);
# older releases don't accept the argument at all. Detect support once
# so we can pass ``sanitize=False`` for our own trusted static markup.
_UI_HTML_HAS_SANITIZE = "sanitize" in inspect.signature(ui.html).parameters


def _raw_html(content: str) -> Any:
    """``ui.html`` for trusted static markup, across NiceGUI versions."""
    if _UI_HTML_HAS_SANITIZE:
        return ui.html(content, sanitize=False)
    return ui.html(content)

# ── Config ────────────────────────────────────────────────────────

PERSONALITIES = ["default", "cheerful", "stoic", "anxious"]
SPEED_OPTIONS = {"1×": 1.0, "10×": 10.0, "60×": 60.0}
REFRESH_HZ = 10.0

# Valence-family palette (docs/design/color-palette-review.md)
CHEMICAL_COLORS: dict[str, str] = {
    "dopamine":     "#E6A23C",  # warm amber — reward
    "serotonin":    "#4FB7A6",  # muted teal — wellbeing
    "oxytocin":     "#7FB3D9",  # dusty sky — bonding
    "testosterone": "#D96B2C",  # burnt orange — drive
    "cortisol":     "#C4513F",  # rust red — stress
    "adrenaline":   "#E8526B",  # hot coral — arousal
    "endorphins":   "#E8C36B",  # pale straw — pleasure
    "gaba":         "#5682B5",  # slate blue — calm
}

EMOTION_COLORS: dict[str, str] = {
    "happiness":  "#E6A23C",
    "excitement": "#D96B2C",
    "anger":      "#C4513F",
    "calm":       "#4FB7A6",
    "bonding":    "#7FB3D9",
    "anxiety":    "#B892D1",
    "sadness":    "#6B87A8",
    "euphoria":   "#C678DD",
}

CHEMICAL_NAMES = [c.value for c in Chemical]
EMOTION_NAMES = list(EMOTION_COLORS.keys())

# Max per-chemical offset from baseline applied to a fresh robot, so you
# never meet a guaranteed-calm robot — it might be cheerful, restless,
# wistful, or a touch on edge. Modest, so the state stays plausible.
RESET_JITTER = 0.22


def _jostle_chemistry(robot: Robot) -> None:
    """Nudge each chemical a little off its baseline for a fresh start."""
    chem = robot.current_chemicals()
    for c in Chemical:
        base = chem.baseline(c)
        chem.set(c, max(0.0, min(1.0,
                                 base + random.uniform(-RESET_JITTER, RESET_JITTER))))


# ── App state (single-client assumption) ──────────────────────────


class AppState:
    """Module-level state for the single-client dashboard."""

    def __init__(
        self,
        personality: str = "default",
        llm_backend: LLMBackend | None = None,
        llm_label: str = "",
    ) -> None:
        self.personality = personality
        self.llm_backend = llm_backend
        self.llm_label = llm_label
        self.speed = 1.0
        self.voice_on = True   # speak the robot's reply via the browser
        self._build_robot()

    def _build_robot(self) -> None:
        """(Re)create the robot with a slightly random starting mood."""
        self.clock = ManualClock()
        self.robot = Robot(
            personality=self.personality,
            clock=self.clock,
            llm_backend=self.llm_backend,
        )
        _jostle_chemistry(self.robot)
        self.last_tick = time.monotonic()

    def reset(self) -> None:
        """Start over: rebuild the robot (fresh, slightly random chemistry)
        and clear the conversation thread. Keeps speed and voice settings."""
        self._build_robot()


def _interpreter_status(state: AppState) -> str:
    """Summarize which path the last interpretation took."""
    interp = state.robot.interpreter
    if interp is None:
        return "no LLM — fallback nudge applied"
    path = getattr(interp, "last_path", "unknown")
    err = getattr(interp, "last_error", "")
    if path == "llm":
        return "LLM OK"
    if path == "cache":
        return "cache hit"
    if path == "fallback":
        return f"FALLBACK — {err}" if err else "FALLBACK"
    return path


def _impulse_summary(state: AppState) -> str:
    parts = []
    for imp in state.robot.last_impulses:
        sign = "+" if imp.delta >= 0 else ""
        parts.append(f"{imp.chemical.value} {sign}{imp.delta:.2f}")
    return ", ".join(parts)


# ── Text-to-speech (browser Web Speech API, client-side) ──────────
# Defines window.kindaliveSpeak / kindaliveStopSpeak and primes the
# speech engine on the first user gesture (iOS/Safari only allow
# programmatic speech after a real gesture). Injected once at boot.
SPEECH_SETUP_JS = """
(function(){
  if (window.kindaliveSpeak) return;
  var primed = false;
  function prime(){
    if (primed || !('speechSynthesis' in window)) return;
    primed = true;
    try { var u = new SpeechSynthesisUtterance(' '); u.volume = 0;
          window.speechSynthesis.speak(u); } catch (e) {}
  }
  window.addEventListener('pointerdown', prime, true);
  function mouth(on){
    try { window.kindaliveFace && window.kindaliveFace.setSpeaking(on); } catch(e){}
  }
  window.kindaliveStopSpeak = function(){
    try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch(e){}
    mouth(false);
  };
  window.kindaliveSpeak = function(text, rate, pitch){
    if (!('speechSynthesis' in window) || !text) return;
    try {
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(text);
      u.rate = rate; u.pitch = pitch;
      // Drive the LED face's mouth from the speech timeline.
      u.onstart = function(){ mouth(true); };
      u.onend = function(){ mouth(false); };
      u.onerror = function(){ mouth(false); };
      u.onboundary = function(){
        try { window.kindaliveFace && window.kindaliveFace.mouthPulse(); } catch(e){}
      };
      // Safety net: if onstart/onend don't fire (some mobile voices),
      // flap for an estimated duration based on text length.
      mouth(true);
      var est = Math.min(12000, 350 + text.length * 55 / (rate || 1));
      setTimeout(function(){ if(!window.speechSynthesis.speaking) mouth(false); }, est);
      window.speechSynthesis.speak(u);
    } catch (e) { console.error('tts failed', e); mouth(false); }
  };
})();
"""


def _voice_params(emotions: Any) -> tuple[float, float]:
    """Map the current mood to TTS rate/pitch.

    More arousal (excitement/anxiety/anger) → faster; more positive
    valence (happiness/euphoria over sadness/anxiety) → higher pitch.
    """
    d = emotions.as_dict()
    arousal = d["excitement"] + d["anxiety"] + d["anger"]
    valence = d["happiness"] + d["euphoria"] - d["sadness"] - d["anxiety"]
    rate = max(0.8, min(1.3, 1.0 + 0.35 * min(1.0, arousal)))
    pitch = max(0.7, min(1.4, 1.0 + 0.30 * max(-1.0, min(1.0, valence))))
    return round(rate, 2), round(pitch, 2)


def _speak_js(text: str, rate: float, pitch: float) -> str:
    return (
        "window.kindaliveSpeak && "
        f"window.kindaliveSpeak({json.dumps(text)}, {rate}, {pitch});"
    )


# ── UI construction ───────────────────────────────────────────────


def create_app(
    personality: str = "default",
    llm_backend: LLMBackend | None = None,
    llm_label: str = "",
) -> AppState:
    """Build the NiceGUI app and return its state.

    Registers the page and static files. Call ``ui.run()`` afterwards
    to start the server.
    """
    state = AppState(
        personality=personality,
        llm_backend=llm_backend,
        llm_label=llm_label,
    )

    app.add_static_files("/webassets", str(WEB_ASSETS_DIR))

    @ui.page("/")
    def main_page() -> None:
        _build_page(state)

    return state


def _bar_panel(
    title: str, names: list[str], colors: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """One side panel of labeled progress bars; returns name → widgets."""
    rows: dict[str, dict[str, Any]] = {}
    with ui.card().classes("w-full").style(
        "background: #141820; color: #E8ECF3; padding: 14px;"
    ):
        ui.label(title).classes("section-title").style(
            "margin-bottom: 6px;"
        )
        for name in names:
            with ui.row().classes("items-center gap-2 w-full no-wrap").style(
                "min-height: 26px;"
            ):
                ui.label(name).style(
                    f"color: {colors[name]}; width: 86px; font-size: 11px; "
                    "font-weight: 600; text-transform: uppercase; "
                    "letter-spacing: 0.04em;"
                )
                bar = ui.linear_progress(
                    value=0.0, show_value=False,
                ).style(f"flex: 1; --q-primary: {colors[name]};")
                val = ui.label("0.00").style(
                    "color: #A3ADBF; font-size: 11px; width: 36px; "
                    "text-align: right; font-variant-numeric: tabular-nums;"
                )
                rows[name] = {"bar": bar, "label": val}
    return rows


def _build_page(state: AppState) -> None:
    ui.add_head_html(
        # Responsive + installable-as-an-app on phones. The
        # apple-mobile-web-app metas make "Add to Home Screen" launch
        # fullscreen, so it feels like a native app on iOS.
        '<meta name="viewport" content="width=device-width, initial-scale=1, '
        'viewport-fit=cover">'
        '<meta name="theme-color" content="#0B0D12">'
        '<meta name="mobile-web-app-capable" content="yes">'
        '<meta name="apple-mobile-web-app-capable" content="yes">'
        '<meta name="apple-mobile-web-app-status-bar-style" '
        'content="black-translucent">'
        '<meta name="apple-mobile-web-app-title" content="Kindalive">'
        '<link rel="manifest" href="/webassets/manifest.json">'
        '<link rel="apple-touch-icon" href="/webassets/icon.svg">'
        '<link rel="stylesheet" href="/webassets/style.css">'
    )
    ui.query("body").style(
        "background: #0B0D12; color: #E8ECF3; "
        "font-family: 'Inter', -apple-system, sans-serif;"
    )

    # ── Header ────────────────────────────────────────────────
    with ui.row().classes("w-full items-center").style(
        "background: #0F1218; border-bottom: 1px solid #232834; "
        "padding: 10px 18px; gap: 14px; flex-wrap: wrap;"
    ):
        ui.label("KINDALIVE").style(
            "color: #7C9CFF; font-size: 18px; font-weight: 700; "
            "letter-spacing: 0.18em;"
        )
        ui.label(state.personality).style(
            "color: #6B7587; font-size: 11px; text-transform: uppercase; "
            "letter-spacing: 0.1em; border: 1px solid #232834; "
            "border-radius: 10px; padding: 1px 10px;"
        )
        ui.space()
        ui.label("time").style("color: #6B7587; font-size: 11px;")
        ui.toggle(
            list(SPEED_OPTIONS.keys()),
            value="1×",
            on_change=lambda e: setattr(
                state, "speed", SPEED_OPTIONS.get(e.value or "1×", 1.0),
            ),
        ).props("dense no-caps toggle-color=primary")
        # Voice (TTS) toggle
        def _toggle_voice(e: Any) -> None:
            state.voice_on = bool(e.value)
            if not state.voice_on:
                ui.run_javascript(
                    "window.kindaliveStopSpeak && window.kindaliveStopSpeak();"
                )

        with ui.row().classes("items-center").style("gap: 4px;"):
            ui.icon("volume_up").style("color: #6B7587; font-size: 18px;")
            ui.switch(value=state.voice_on, on_change=_toggle_voice).props(
                "dense color=primary"
            )

        llm_on = state.llm_backend is not None
        ui.label(
            f"LLM: {state.llm_label}" if llm_on else "LLM: OFFLINE"
        ).style(
            f"color: {'#3DD68C' if llm_on else '#6B7587'}; "
            "font-weight: 600; font-size: 12px;"
        )
        # Reset button — wired below once the labels it clears exist.
        reset_btn = ui.button("Reset").props("flat dense no-caps").style(
            "color: #A3ADBF; border: 1px solid #232834; "
            "border-radius: 8px; padding: 0 12px;"
        )

    # ── Main layout ───────────────────────────────────────────
    # Sizing/order live in style.css (.kl-*) so the media query can
    # collapse the three columns to one on phones. The face column is
    # DOM-first, so on a phone it stacks on top with the chemical and
    # emotion bars below it.
    with ui.element("div").classes("kl-main"):
        with ui.element("div").classes("kl-col kl-col-face"):
            # The LED face
            with ui.card().classes("w-full items-center").style(
                "background: #05070d; padding: 10px;"
            ):
                _raw_html(container_html()).classes("kl-face")
                dominant_label = ui.label("CALM").classes(
                    "dominant-emotion"
                ).style(
                    "color: #7C9CFF; font-size: 15px; font-weight: 700; "
                    "text-transform: uppercase; margin-top: 8px;"
                )
                mix_label = ui.label("").style(
                    "color: #A3ADBF; font-size: 12px; margin-bottom: 2px;"
                )

            # The input
            with ui.card().classes("w-full").style(
                "background: #141820; padding: 14px;"
            ):
                input_row = ui.row().classes(
                    "w-full items-stretch no-wrap"
                ).style("gap: 8px;")
                with input_row:
                    text_input = ui.textarea(
                        placeholder=(
                            "Tell the robot something…"
                            if state.llm_backend
                            else "No LLM configured — set ANTHROPIC_API_KEY "
                                 "or KINDALIVE_LLM_BASE_URL and restart."
                        ),
                    ).props(
                        "dark outlined dense autogrow rows=2"
                        + ("" if state.llm_backend else " disable")
                    ).classes("kl-input").style("flex: 1; min-width: 0;")
                # The robot's spoken reply (also read aloud via TTS).
                reply_label = ui.label("").style(
                    "color: #E8ECF3; font-size: 14px; margin-top: 8px; "
                    "font-style: italic; word-break: break-word;"
                )
                status_label = ui.label("").style(
                    "color: #6B7587; font-size: 11px; min-height: 14px; "
                    "margin-top: 4px; word-break: break-word;"
                )

                async def submit() -> None:
                    text = (text_input.value or "").strip()
                    if not text or not state.llm_backend:
                        return
                    text_input.value = ""
                    reply_label.text = ""
                    status_label.text = "thinking…"
                    await state.robot.interpret_text(text)
                    status = _interpreter_status(state)
                    impulses = _impulse_summary(state)
                    # Show the full phrase (no truncation) so it's clear
                    # exactly what was sent to the LLM.
                    status_label.text = (
                        f'"{text}" — {status}'
                        + (f"  →  {impulses}" if impulses else "")
                    )
                    reply = state.robot.last_reply
                    if reply:
                        reply_label.text = f"🗣  {reply}"
                        if state.voice_on:
                            rate, pitch = _voice_params(
                                state.robot.current_emotions()
                            )
                            ui.run_javascript(_speak_js(reply, rate, pitch))

                # Tap-target Send button — soft keyboards on phones vary
                # in how (or whether) Enter submits, so a visible button
                # is the reliable path. Enter-to-send still works for
                # desktop keyboards.
                with input_row:
                    ui.button("Send", on_click=submit).props(
                        "unelevated color=primary no-caps"
                        + ("" if state.llm_backend else " disable")
                    ).classes("kl-send")

                text_input.on(
                    "keydown.enter.exact",
                    lambda e: submit(),
                    ["prevent", "stop"],
                )
                text_input.on("keydown.ctrl-enter", submit)

        with ui.element("div").classes("kl-col kl-col-chem"):
            chem_rows = _bar_panel(
                "Neurochemistry", CHEMICAL_NAMES, CHEMICAL_COLORS,
            )

        with ui.element("div").classes("kl-col kl-col-emo"):
            emo_rows = _bar_panel(
                "Emotion mix", EMOTION_NAMES, EMOTION_COLORS,
            )

    # Reset — fresh chemistry (back to the personality's baseline) and a
    # new conversation thread. The next refresh tick repaints the bars and
    # relaxes the face to neutral.
    def do_reset() -> None:
        state.reset()
        text_input.value = ""
        reply_label.text = ""
        status_label.text = "Reset — the robot's in a fresh mood; new conversation."
        ui.run_javascript(
            "window.kindaliveStopSpeak && window.kindaliveStopSpeak();"
        )

    reset_btn.on("click", do_reset)

    # Boot the LED face once the client is connected. A one-shot
    # run_javascript dynamic import avoids relying on an inline <script>
    # tag surviving NiceGUI's HTML sanitization.
    ui.timer(0.1, lambda: ui.run_javascript(boot_js()), once=True)
    # Install the speech helpers (and gesture-prime TTS) once.
    ui.timer(0.1, lambda: ui.run_javascript(SPEECH_SETUP_JS), once=True)

    # ── Refresh loop: advance time, repaint bars, drive the face ──
    def tick() -> None:
        now = time.monotonic()
        dt = (now - state.last_tick) * state.speed
        state.last_tick = now
        if dt > 0:
            state.robot.advance(dt=dt)

        chem = state.robot.current_chemicals()
        emotions = state.robot.current_emotions()
        emo_dict = emotions.as_dict()
        dominant_name, dominant_val = emotions.dominant()

        for name, widgets in chem_rows.items():
            level = chem.get(Chemical.from_string(name))
            widgets["bar"].value = level
            widgets["label"].text = f"{level:.2f}"
        for name, widgets in emo_rows.items():
            level = emo_dict[name]
            widgets["bar"].value = level
            widgets["label"].text = f"{level:.2f}"

        color = EMOTION_COLORS.get(dominant_name, DEFAULT_MOOD_COLOR)
        dominant_label.text = f"{dominant_name} {dominant_val:.2f}"
        dominant_label.style(f"color: {color};")
        top3 = [
            f"{n} {v:.2f}" for n, v in emotions.top_n(3) if v > 0.05
        ]
        mix_label.text = "  ·  ".join(top3) if top3 else "—"

        face = FaceProjection.compute(chem)
        payload = face_payload(
            face, mood_color=color, mood_intensity=dominant_val,
        )
        ui.run_javascript(payload_js(payload))

    ui.timer(1.0 / REFRESH_HZ, tick)


# ── Entry point ───────────────────────────────────────────────────


def _resolve_backend(args: Any) -> tuple[LLMBackend | None, str]:
    """Pick an LLM backend from CLI flags + environment.

    Returns (backend, short label for the header). Resolution order in
    ``auto`` mode: Anthropic if a key is present, then any
    OpenAI-compatible server if KINDALIVE_LLM_BASE_URL is set, else
    offline.
    """
    import os

    choice = args.llm
    anthropic_key = args.anthropic_key or os.environ.get(
        "ANTHROPIC_API_KEY", "",
    )
    openai_base = args.llm_base_url or os.environ.get(
        "KINDALIVE_LLM_BASE_URL", "",
    )

    if choice == "off":
        return None, ""

    if choice in ("auto", "anthropic") and anthropic_key:
        from kindalive.interpreter.anthropic_backend import AnthropicBackend

        kwargs: dict[str, Any] = {"api_key": anthropic_key}
        if args.llm_model:
            kwargs["model"] = args.llm_model
        backend = AnthropicBackend(**kwargs)
        return backend, backend.model

    if choice == "anthropic":
        raise SystemExit(
            "--llm anthropic requires ANTHROPIC_API_KEY (env, .env, or "
            "--anthropic-key)."
        )

    if choice == "openai" or (choice == "auto" and openai_base):
        from kindalive.interpreter.openai_backend import OpenAICompatBackend

        oai_backend = OpenAICompatBackend(
            base_url=openai_base or None,
            model=args.llm_model or None,
            api_key=args.llm_key or None,
        )
        return oai_backend, f"{oai_backend.model} @ {oai_backend.base_url}"

    return None, ""


def main() -> None:
    import argparse
    import os

    from kindalive.env_loader import load_dotenv

    # Host/port default from the environment so a container or PaaS can
    # bind without CLI flags. Hosts commonly inject $PORT; the Docker
    # image sets KINDALIVE_HOST=0.0.0.0. See docs/web-ui.md.
    default_host = os.environ.get("KINDALIVE_HOST", "127.0.0.1")
    default_port = int(
        os.environ.get("PORT") or os.environ.get("KINDALIVE_PORT") or 8080
    )

    parser = argparse.ArgumentParser(description="Kindalive dashboard")
    parser.add_argument(
        "--personality", default="default", choices=PERSONALITIES,
        help="Robot personality preset",
    )
    parser.add_argument(
        "--host", default=default_host,
        help="Bind host (env: KINDALIVE_HOST). Use 0.0.0.0 to reach it "
             "from your phone on the same network.",
    )
    parser.add_argument(
        "--port", type=int, default=default_port,
        help="Bind port (env: PORT or KINDALIVE_PORT).",
    )
    parser.add_argument(
        "--llm", default="auto",
        choices=["auto", "anthropic", "openai", "off"],
        help="LLM backend: Claude, any OpenAI-compatible server "
             "(Ollama/LM Studio/vLLM/OpenAI), or off. Default: auto-detect.",
    )
    parser.add_argument(
        "--llm-model", default=None,
        help="Model name/ID override for the chosen backend.",
    )
    parser.add_argument(
        "--llm-base-url", default=None,
        help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1 "
             "or https://openrouter.ai/api/v1 (or set KINDALIVE_LLM_BASE_URL).",
    )
    parser.add_argument(
        "--llm-key", default=None,
        help="API key for the OpenAI-compatible backend, e.g. an OpenRouter "
             "key (or set KINDALIVE_LLM_KEY / OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--anthropic-key", default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY / .env).",
    )
    parser.add_argument(
        "--no-dotenv", action="store_true",
        help="Skip auto-loading .env files from the project root.",
    )
    args = parser.parse_args()

    if not args.no_dotenv:
        loaded = load_dotenv()
        if loaded is not None:
            print(f"Loaded environment from {loaded}")

    try:
        backend, label = _resolve_backend(args)
    except SystemExit:
        raise
    except Exception as e:  # backend init failure → run offline  # noqa: BLE001 (intentional catch-all in runtime)
        print(f"Warning: could not initialize LLM backend: {e}")
        backend, label = None, ""

    if backend is not None:
        print(f"LLM backend: {label}")
    else:
        print(
            "LLM backend: OFFLINE — set ANTHROPIC_API_KEY (cloud) or "
            "KINDALIVE_LLM_BASE_URL (local OpenAI-compatible server) "
            "to enable text input."
        )

    create_app(
        personality=args.personality, llm_backend=backend, llm_label=label,
    )
    ui.run(
        host=args.host,
        port=args.port,
        title="Kindalive",
        dark=True,
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
