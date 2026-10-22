"""Re-render the two docs figures with the site's own fonts and code (Playwright and its Chromium, as for the e2e tests):

    uv run --no-project --with playwright==1.63.0 --with opencv-python-headless==5.0.0.93 python scripts/render_figures.py

docs/architecture.png: docs/architecture.svg at 2x. docs/recall_vs_pixels.png: the site's FIG. 4.1 (dataset label,
chart, legend, marks note), drawn by public/js/results.js from public/results/summary.json, at 2x, scaled to 1200 px
wide. CHROMIUM=<path> runs a Chromium other than Playwright's own (e.g. an older cached build).
"""

import functools
import http.server
import os
import threading
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FONTS = [("IBM Plex Mono", "IBMPlexMono", w) for w in (400, 500, 700)] + [("Archivo Narrow", "ArchivoNarrow", w) for w in (500, 700)]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def architecture(browser) -> None:
    faces = "".join(f"@font-face{{font-family:'{fam}';font-weight:{w};src:url('{(ROOT / f'public/fonts/{f}-{w}.woff2').as_uri()}')}}"
                    for fam, f, w in FONTS)
    page_file = ROOT / "build/architecture.html"
    page_file.parent.mkdir(exist_ok=True)
    page_file.write_text(f"<!doctype html><style>{faces}html,body{{margin:0}}svg{{display:block}}</style>"
                         + (ROOT / "docs/architecture.svg").read_text())
    pg = browser.new_page(viewport={"width": 1340, "height": 1720}, device_scale_factor=2)
    pg.goto(page_file.as_uri())
    pg.evaluate("document.fonts.ready")
    pg.locator("svg").first.screenshot(path=str(ROOT / "docs/architecture.png"))


def fig41(browser) -> None:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "public")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pg = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, reduced_motion="reduce")
    pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html")
    pg.locator("#bench svg.chart").wait_for()
    pg.add_style_tag(content="#bench .dataset { justify-self: start; }")  # the label box as wide as its text
    pg.evaluate("document.fonts.ready")
    pg.locator("#bench").scroll_into_view_if_needed()
    pg.wait_for_timeout(800)
    label, chart, note = (pg.locator(s).first.bounding_box() for s in ("#bench .dataset", "#bench svg.chart", "#bench .bench .note"))
    pad = 12
    x0, y0 = chart["x"] - pad, label["y"] - pad
    out = ROOT / "docs/recall_vs_pixels.png"
    pg.screenshot(path=str(out), clip={"x": x0, "y": y0, "width": chart["width"] + 2 * pad,
                                       "height": note["y"] + note["height"] - y0})  # stops above the folded data table
    srv.shutdown()
    img = cv2.imread(str(out))
    cv2.imwrite(str(out), cv2.resize(img, (1200, round(img.shape[0] * 1200 / img.shape[1])), interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
        architecture(b)
        fig41(b)
        b.close()
