"""Remake public/og.png (the 1200x630 share card) from a real screenshot of the site, not a drawing.

Serves public/ locally, runs the recorded job (?mock=1, no server) to the end, closes the revision notes, then
screenshots the demo's viewer row (the photo with the numbered gaze path, the camera crop and the trace) at 2x and
scales it to 1200x630 with OpenCV. Re-run it after a UI change:

    uv run --with playwright==1.63.0 python scripts/og_image.py
"""
import functools
import http.server
import threading
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
W, H = 1200, 630


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main(out: Path = ROOT / "public/og.png") -> None:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "public")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, reduced_motion="reduce")
            page.goto(f"http://127.0.0.1:{srv.server_address[1]}/?mock=1#demo")
            expect(page.locator("#look")).to_be_enabled(timeout=30_000)
            page.click("#look")
            page.locator("#skip").click(timeout=30_000)
            expect(page.locator("#job-pill")).to_have_text("needs sign-off", timeout=30_000)
            page.wait_for_timeout(1500)
            for x in page.locator("#toasts .win__x").all():
                x.click()
            page.evaluate("document.querySelector('#step-watch').scrollIntoView({block: 'start', behavior: 'instant'})")
            page.wait_for_timeout(800)
            left, right = page.locator("#step-watch").bounding_box(), page.locator(".col--side").bounding_box()
            x0, y0 = left["x"] - 4, left["y"] - 4
            w = right["x"] + right["width"] + 4 - x0
            png = page.screenshot(clip={"x": x0, "y": y0, "width": w, "height": w * H / W})
            browser.close()
    finally:
        srv.shutdown()
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(out), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"{out}: {W}x{H} from a {img.shape[1]}x{img.shape[0]} capture")


if __name__ == "__main__":
    main()
