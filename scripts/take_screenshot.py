#!/usr/bin/env python3
"""Start the web UI, nudge the chemistry, and take a screenshot.

Requires Playwright with a Chromium build installed
(``python3 -m playwright install chromium``).

Usage:
    PYTHONPATH=. python3 scripts/take_screenshot.py \
        [--out docs/screenshots/web_ui.png]
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="docs/screenshots/web_ui.png",
        help="Output path for the screenshot",
    )
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from nicegui import ui

    from kindalive.engine.chemicals import Chemical
    from kindalive.engine.impulse import ChemicalImpulse
    from kindalive.expression.web_ui import create_app

    state = create_app(personality="default", llm_backend=None)

    # Give the face something to express: a joyful chemical signature.
    state.robot.receive_impulses([
        ChemicalImpulse(Chemical.DOPAMINE, 0.35, source_id="shot"),
        ChemicalImpulse(Chemical.ENDORPHINS, 0.25, source_id="shot"),
        ChemicalImpulse(Chemical.ADRENALINE, 0.20, source_id="shot"),
    ])

    def run_server() -> None:
        ui.run(
            host="127.0.0.1",
            port=args.port,
            title="Kindalive",
            dark=True,
            reload=False,
            show=False,
        )

    threading.Thread(target=run_server, daemon=True).start()

    import urllib.request

    url = f"http://127.0.0.1:{args.port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:  # noqa: BLE001  (transient: wait then retry screenshot)
            time.sleep(0.5)
    else:
        print("Server did not start in time")
        sys.exit(1)

    print(f"Taking screenshot at {args.width}x{args.height}...")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--use-gl=angle"],
        )
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,
        )
        page.goto(url, wait_until="networkidle")
        # Let the WebGL face settle into its expression
        page.wait_for_timeout(4000)
        page.screenshot(path=str(out_path), full_page=False)
        browser.close()

    print(f"Screenshot saved to {out_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
