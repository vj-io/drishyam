"""Browser test for the Drishyam UI under the production CSP.

Run:  make e2e   (or: .venv/bin/python tests/e2e/test_ui_mock.py; needs playwright + chromium)

Every expectation about the job (step count, findings, critical ids, planner events, causes, done status,
unvisited regions, work-order id, budget numbers) is DERIVED from public/fixtures/job.ndjson at start-up,
so regenerating the fixture from the real API never needs a test edit. The fixture is never hand-written here.

Scenarios (a real Chromium, the document served with the production CSP header):
  1. desktop, ?mock=1: hero, gallery thumbnails, KILL mid-run, Resume, findings (one card per id), human
     review of every critical finding, replay slider, heatmap toggle, work-order download, screenshots.
  2. live-API path (no ?mock) against a local stand-in for /api/*: query string, X-Drishyam-Replay header,
     /api/review tokens, /api/kill {job, kill_token}, privacy.boxes blur, halt step.
  3. mobile 375 px, 4. prefers-reduced-motion, 5. heading names, 6. pending states, 7. a COOL file with missing fields.
Any CSP violation, console error or page error fails the run. wait_for_function is avoided on purpose
(it is blocked by a strict CSP): every wait uses locators or polls page.evaluate.
"""
import base64
import functools
import http.server
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public"
SHOTS = Path(os.environ.get("DRISHYAM_SHOTS", "/tmp/drishyam-e2e"))
SHOTS.mkdir(parents=True, exist_ok=True)

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'")
T = 30_000  # ms, generous: the mock job streams one event per 250 ms

# ---- everything the tests know about the job comes from the fixture ---------------------------------
JOB_LINES = [ln for ln in (PUBLIC / "fixtures/job.ndjson").read_text().splitlines() if ln.strip()]
EVENTS = [json.loads(ln) for ln in JOB_LINES]
JOB = EVENTS[0]
DONE = EVENTS[-1]
assert JOB["type"] == "job" and DONE["type"] == "done"
STEPS = [e for e in EVENTS if e["type"] == "step"]
N_STEPS = len(STEPS)
N_PLANNER = sum(e["type"] == "planner" for e in EVENTS)
N_TRACE = N_STEPS + N_PLANNER
FINDINGS = {}                                        # id -> the latest finding event for it
for _e in EVENTS:
    if _e["type"] == "finding":
        FINDINGS[_e["finding"]["id"]] = _e["finding"]
CRIT = [fid for fid, f in FINDINGS.items() if f["critical"]]
UNVISITED = DONE["unvisited"]
DONE_LABEL = DONE["status"].replace("_", " ")
WO_ID = DONE["work_order"]["id"]
GOVERNOR = {"p95>SLA", "warp-spike", "usd≥80%", "cap-hit", "kill-switch", "planner-fallback"}
MOVING = ("gaze", "escalate", "verify")
HALTS = [s for s in STEPS if s["decision"]["action"] == "halt"]
assert N_STEPS >= 4 and CRIT, "the fixture should hold several steps and a critical finding"


def move(step):
    a = step["decision"]["args"]
    return [step["decision"]["action"], a["x"], a["y"], a["level"], *([a["size"]] if "size" in a else [])]


MOVES = [move(s) for s in STEPS if s["decision"]["action"] in MOVING]

MOCK_JPG = (PUBLIC / "fixtures/mock.jpg").read_bytes()
THUMB = next(iter(sorted((PUBLIC / "samples/thumbs").glob("*.jpg"))), None)
THUMB_BYTES = THUMB.read_bytes() if THUMB else MOCK_JPG[:1000]
SUMMARY = {
    "dataset": "synthetic-200 (labelled synthetic)",
    "policies": {
        "downscale": [{"pixels_frac": 0.06, "recall": 0.41, "usd": 0.0001}],
        "tiling": [{"pixels_frac": 0.25, "recall": 0.55, "usd": 0.0009}, {"pixels_frac": 0.5, "recall": 0.74, "usd": 0.0016},
                   {"pixels_frac": 1.0, "recall": 0.93, "usd": 0.0031}],
        "drishyam": [{"pixels_frac": 0.05, "recall": 0.52, "usd": 0.0002}, {"pixels_frac": 0.1, "recall": 0.7, "usd": 0.0003},
                     {"pixels_frac": 0.2, "recall": 0.8, "usd": 0.0005}, {"pixels_frac": 0.29, "recall": 0.88, "usd": 0.0008},
                     {"pixels_frac": 0.5, "recall": 0.92, "usd": 0.0014}],
    },
    "micro": [{"op": "pyrDown", "size": "4000x3000", "p50_ms": 3.1, "p95_ms": 3.4},
              {"op": "warpPerspective", "size": "4000x3000", "p50_ms": 9.2, "p95_ms": 11.8}],
    "provenance": {"cv2": "5.0.0", "machine": "test rig", "build": "stock"},
}
# a model summary (evals.model_eval --publish): FIG. 4.1/4.2 add its Drishyam curve as a fourth series
MODEL = {"dataset": "fixture", "segmenter": "onnx:drishyam_dacl10k.onnx",
         "policies": {"drishyam": [{"pixels_frac": 0.1, "recall": 0.6}, {"pixels_frac": 0.29, "recall": 0.9}]}}
REAL = {**SUMMARY, "dataset": "CODEBRIM (test fixture)",
        "policies": {k: [{**pt, "recall_iou_only": round(pt["recall"] / 2, 2)} for pt in v] for k, v in SUMMARY["policies"].items()}}
# the shape bench.compare_arms --summary writes; the numbers are made up for the test
COOL = {"expected": {"op_level": [1.2, 1.8], "end_to_end": [1.2, 1.3]},
        "end_to_end": {"cool_gain": 1.04, "hardware_gain": 1.1, "a_over_c": 1.144},
        "op_level_geomean": {"cool_gain": 1.5, "hardware_gain": 1.2},
        "parity_a_vs_b": {"crops": 24, "pixel_agreement": 0.9991, "worst_crop_agreement": 0.98},
        "arms": {r: {"what": what, "instance_type": it, "cv2": "5.0.0", "cv2_from_opt_cool": r == "A", "images_per_s": ips,
                     "usd_per_hour": usd, "usd_per_1k_images": round(usd / 3.6 / ips, 6), "covered_fraction": 0.093}
                 for r, what, it, ips, usd in (("A", "COOL build, Graviton", "c8g.2xlarge", 2.6, 0.339),
                                               ("B", "stock wheel, Graviton", "c8g.2xlarge", 2.5, 0.319),
                                               ("C", "stock wheel, x86", "c7i.2xlarge", 2.27, 0.357))}}
FAILURES = [{"image": "failures/f01.jpg", "category": "graffiti read as damage",
             "what_happened": "A spray-painted line was drafted as a crack.",
             "rule_that_would_catch_it": "reject masks with saturated paint hues"}]
DATA = {
    "/samples/index.json": ("application/json", json.dumps([{"file": "bridge_01.jpg", "thumb": "thumbs/bridge_01.jpg", "title": "Pier 3, north face",
                                                              "note": "rust bleed and a hairline crack", "synthetic": True}]).encode()),
    "/samples/bridge_01.jpg": ("image/jpeg", MOCK_JPG),
    "/samples/thumbs/bridge_01.jpg": ("image/jpeg", THUMB_BYTES),
    "/results/summary.json": ("application/json", json.dumps(SUMMARY).encode()),
    "/results/codebrim/summary.json": ("application/json", json.dumps(REAL).encode()),
    "/results/synth_dacl10k/summary.json": ("application/json", json.dumps(MODEL).encode()),
    "/results/codebrim_dacl10k/summary.json": ("application/json", json.dumps(MODEL).encode()),
    "/results/cool.json": ("application/json", json.dumps(COOL).encode()),
    "/results/failures.json": ("application/json", json.dumps(FAILURES).encode()),
    "/results/failures/f01.jpg": ("image/jpeg", MOCK_JPG),
}

# ---- a local stand-in for /api/* (streams NDJSON like the real server) --------------------------------
LIVE = {"plans": [], "jobs": [], "kills": [], "reviews": [], "order": None, "tokens": 0}


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/api/health":
            return self.reply(200, {"ok": True, "opencv": "5.0.0", "build": "stock (not COOL)", "segmenter": "classical-v1",
                                    "max_upload_bytes": 4400000, "max_pixels": 25000000})
        super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        if url.path == "/api/jobs":
            LIVE["jobs"].append({"query": parse_qs(url.query), "headers": {k.lower(): v for k, v in self.headers.items()}, "size": len(raw)})
            plan = LIVE["plans"].pop(0)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for line in plan["lines"]:
                    self.wfile.write((line + "\n").encode())
                    self.wfile.flush()
                    time.sleep(plan["delay"])
            except (BrokenPipeError, ConnectionResetError):
                pass                                   # the browser aborted (KILL)
        elif url.path == "/api/kill":
            LIVE["kills"].append(json.loads(raw))
            self.reply(200, {"ok": True})
        elif url.path == "/api/review":
            body = json.loads(raw)
            LIVE["reviews"].append(body)
            order = LIVE["order"]
            for f in order["findings"]:
                if f["id"] == body["finding_id"]:
                    f["state"] = "approved" if body["decision"] == "approve" else "rejected"
            order["approvals"].append({"finding_id": body["finding_id"], "decision": body["decision"], "reviewer": body["reviewer"], "at": "2026-10-24T17:05:00Z"})
            order["status"] = "blocked" if any(f["critical"] and f["state"] == "pending_review" for f in order["findings"]) else "ready"
            LIVE["tokens"] += 1
            self.reply(200, {"token": f"tok-{LIVE['tokens']}", "work_order": order})
        else:
            self.reply(404, {"error": "not found"})


def serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(PUBLIC)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class Session:
    """A browser context with the production CSP, console/CSP collectors and optional fake JSON data."""

    def __init__(self, browser, name, with_data=True, **ctx):
        self.name = name
        self.errors = []
        self.api_calls = []
        self.with_data = with_data
        self.context = browser.new_context(**ctx)
        self.context.add_init_script(
            "window.__csp = [];"
            "document.addEventListener('securitypolicyviolation', e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));")
        self.context.route("**/*", self.route)
        self.page = self.context.new_page()
        self.page.on("console", self.on_console)
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.on("request", lambda r: "/api/" in r.url and self.api_calls.append(urlparse(r.url).path))

    def on_console(self, msg):
        if msg.type != "error":
            return
        if not self.with_data and "Failed to load resource" in msg.text:
            return  # the pending-state scenario expects 404s on the optional JSON files
        self.errors.append(f"console.error: {msg.text}")

    def route(self, route):
        req = route.request
        path = urlparse(req.url).path
        if path in DATA:  # deterministic whatever public/samples and public/results hold
            if not self.with_data:
                return route.fulfill(status=404, content_type="text/plain", body="not found")
            mime, body = DATA[path]
            return route.fulfill(status=200, content_type=mime, body=body)
        if req.resource_type == "document":
            resp = route.fetch()
            return route.fulfill(response=resp, headers={**resp.headers, "content-security-policy": CSP})
        return route.continue_()

    def check_clean(self):
        csp = self.page.evaluate("window.__csp")
        assert not csp, f"[{self.name}] CSP violations: {csp}"
        assert not self.errors, f"[{self.name}] errors: {self.errors}"

    def scroll_through(self):
        h = self.page.evaluate("document.documentElement.scrollHeight")
        step = max(300, self.page.viewport_size["height"] // 2)
        for y in range(0, h, step):
            self.page.evaluate(f"window.scrollTo({{top: {y}, behavior: 'instant'}})")
            self.page.wait_for_timeout(120)
        self.page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
        self.page.wait_for_timeout(300)

    def shot(self, name, **kw):
        self.page.screenshot(path=str(SHOTS / name), **kw)

    def to(self, selector, block="start"):
        self.page.evaluate(f"document.querySelector('{selector}').scrollIntoView({{block: '{block}', behavior: 'instant'}})")
        self.page.wait_for_timeout(700)


def until(fn, what, timeout=T / 1000):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def hscroll_offenders(page):
    return page.evaluate("""() => ({
      sw: document.documentElement.scrollWidth, iw: window.innerWidth,
      off: [...document.querySelectorAll('body *')].filter(e => {
        if (e.closest('.bg, .marquee, .fx, .table-wrap, .toasts')) return false;
        const r = e.getBoundingClientRect();
        return r.width > 0 && r.right > window.innerWidth + 1;
      }).map(e => e.tagName.toLowerCase() + '.' + String(e.className).split(' ')[0]).slice(0, 8)})""")


def blur_energy(page):
    """Mean gradient energy of a canvas region inside the privacy box (low = blurred)."""
    return page.evaluate("""([W, H, bx, by, bw, bh]) => {
      const c = document.getElementById('stage'), g = c.getContext('2d'), d = devicePixelRatio, cw = c.width / d, ch = c.height / d;
      const s = Math.min((cw - 24) / W, (ch - 24) / H), ox = (cw - W * s) / 2, oy = (ch - H * s) / 2;
      const x0 = Math.round((ox + (bx + bw * 0.1) * s) * d), y0 = Math.round((oy + (by + bh * 0.1) * s) * d);
      const w = Math.round(bw * 0.8 * s * d), h = Math.round(bh * 0.8 * s * d), px = g.getImageData(x0, y0, w, h).data;
      let e = 0;
      for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) { const i = (y * w + x) * 4; e += Math.abs(px[i] - px[i + 4]) + Math.abs(px[i] - px[i + w * 4]); }
      return e / (w * h);
    }""", [JOB["width"], JOB["height"], *BOX])


BOX = [round(JOB["width"] * 0.4), round(JOB["height"] * 0.4), round(JOB["width"] * 0.2), round(JOB["height"] * 0.2)]


def card(page, fid):
    return page.locator(f".finding[data-id='{fid}']")


def pending_tile(page):
    return page.locator("#done .tile__v").nth(5)


def approve_all(page, ids, expect_tokens=None):
    """Approve every critical finding in turn; the pending tile and pill must follow each decision."""
    for i, fid in enumerate(ids):
        c = card(page, fid)
        if i == 0:
            c.get_by_role("button", name="Approve").click()
            expect(c.locator(".finding__err")).to_contain_text("reviewer name")  # name is required
        c.get_by_label("Reviewer").fill("A. Engineer")
        c.get_by_role("button", name="Approve").click()
        expect(c.locator(".chip--state")).to_have_text("approved", timeout=T)
        expect(c.locator(".review")).to_contain_text("approved by A. Engineer")
        left = len(ids) - i - 1
        assert pending_tile(page).inner_text() == str(left), "pending tile must change with the card state, not animate down later"
        time.sleep(0.9)                                  # a stale count-up must not overwrite the latest value
        assert pending_tile(page).inner_text() == str(left), "pending tile was overwritten by a stale animation"
        expect(page.locator("#done .stack > .note").first).to_contain_text(f"{left} critical" if left else "is ready")
        expect(page.locator("#job-pill")).to_have_text("completed" if not left else DONE_LABEL)


# ---------------------------------------------------------------------------------------------
def test_static_csp_hygiene():
    html = (PUBLIC / "index.html").read_text()
    assert not re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline <script> found"
    assert not re.findall(r"\sstyle\s*=", html), "inline style attribute in HTML"
    assert not re.findall(r"\son[a-z]+\s*=", html), "inline event handler in HTML"
    assert not re.findall(r'(?:src|href)="https?://(?!github\.com/vj-io/drishyam)', html), "external resource in HTML"
    banned = re.compile(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function|setAttribute\(\s*['\"]style|cssText|https?://(?!www\.w3\.org/2000/svg)")
    for f in [PUBLIC / "app.js", *sorted((PUBLIC / "js").glob("*.js"))]:
        bad = banned.findall(f.read_text())
        assert not bad, f"{f.name}: {bad}"
    css = (PUBLIC / "styles.css").read_text()
    assert not re.findall(r"url\(\s*['\"]?(?:https?:)?//", css), "external url() in CSS"
    print("ok  static CSP hygiene")


def test_desktop_mock(browser, base):
    s = Session(browser, "desktop", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("h1.wordmark")).to_be_visible()
    expect(p.locator("#look")).to_be_enabled(timeout=T)          # mock.jpg is preloaded
    expect(p.locator(".thumb")).to_have_count(1)                  # routed sample manifest
    expect(p.locator(".thumb img")).to_have_attribute("src", re.compile(r"samples/thumbs/bridge_01\.jpg$"))  # manifest `thumb` is used
    expect(p.locator("#sound")).to_have_attribute("aria-pressed", "false")  # sound defaults to off
    expect(p.locator("#stat-recall .stat__v")).to_contain_text("tiling", timeout=T)  # benchmarks replaced the chip
    expect(p.locator("#bench svg.chart")).to_be_visible()
    expect(p.locator("#bench .legend li")).to_have_count(4)                       # three policies + the trained model
    expect(p.locator("#bench-real .chart__line--model")).to_have_count(1)
    n_iou = sum(len(v) for v in REAL["policies"].values())
    expect(p.locator("#bench-real .chart__plot path[d*='l10 10']")).to_have_count(n_iou)  # IoU-only crosses on FIG. 4.2 only
    expect(p.locator("#bench .chart__plot path[d*='l10 10']")).to_have_count(0)
    expect(p.locator("#fail .fail")).to_have_count(1)
    expect(p.locator("#bench-cool tbody tr")).to_have_count(3)                   # FIG. 4.3 from results/cool.json
    expect(p.locator("#bench-cool .kv")).to_contain_text("PRD expects 1.2–1.3×")  # the expected range sits by the measured gain
    expect(p.locator("#cool-status")).to_contain_text(f"ran at {COOL['end_to_end']['cool_gain']}×")  # the banner says what was measured
    p.keyboard.press("Tab")
    outline = p.evaluate("(() => { const s = getComputedStyle(document.activeElement); return s.outlineStyle + ' ' + s.outlineWidth; })()")
    assert outline.startswith("solid"), f"no visible keyboard focus: {outline}"
    s.scroll_through()
    s.shot("desktop-hero.png")

    # --- KILL mid-run ---------------------------------------------------------------------
    s.to(".win--view")
    p.click("#look")
    expect(p.locator("#kill")).to_be_visible()
    expect(p.locator("#look")).to_be_disabled()
    expect(p.locator(".trace__line").nth(2)).to_be_visible(timeout=T)
    p.click("#kill")
    expect(p.locator("#job-pill")).to_have_text("killed")
    expect(p.locator("#job-text")).to_contain_text("Killed at step")
    expect(p.locator("#kill")).to_be_hidden()
    expect(p.locator("#resume")).to_be_visible()
    n = p.locator(".trace__line").count()
    seen = p.locator(".trace__line[data-step]").count()
    assert 3 <= n < N_TRACE, f"kill should stop the stream mid-run, trace lines={n} of {N_TRACE}"
    time.sleep(1.2)
    assert p.locator(".trace__line").count() == n, "stream kept going after KILL"
    assert p.locator("#replay").evaluate("e => Number(e.max)") == seen
    assert not s.api_calls, f"mock mode must not touch the network API: {s.api_calls}"

    # --- Resume: replays the same moves, then finishes ---------------------------------------
    replayed = len(MOVES[:seen])
    p.click("#resume")
    shot_taken = False
    end = time.time() + T / 1000
    while time.time() < end and p.locator("#job-pill").inner_text().strip().lower() != DONE_LABEL:
        if not shot_taken and p.locator(".win--toast").count() > 0:
            s.shot("desktop-running.png")
            shot_taken = True
        time.sleep(0.15)
    expect(p.locator("#job-pill")).to_have_text(re.compile(DONE_LABEL, re.IGNORECASE), timeout=T)
    live_causes = [c for st in STEPS if st["step"] > replayed       # a halt record may also carry a governor note
                   for c in (st["cause"], st.get("governor", {}).get("cause")) if c]
    if any(c in GOVERNOR for c in live_causes):
        assert shot_taken, "a governor pop-up window should have appeared during the run"
    expect(p.locator(".trace__line")).to_have_count(N_TRACE)
    expect(p.locator(".trace__line--planner")).to_have_count(N_PLANNER)
    expect(p.locator(".trace__line[data-step]")).to_have_count(N_STEPS)
    expect(p.locator(".t-act--halt")).to_have_count(len(HALTS))
    shown = Counter(["replay"] * replayed + live_causes)       # replayed steps are tagged `replay`
    for cause, count in shown.items():
        chips = p.locator(".trace__line .chip", has_text=re.compile(f"^{re.escape(cause)}$"))
        assert chips.count() == count, f"cause chip {cause!r}: {chips.count()} shown, {count} expected"

    # --- findings: one card per id, latest values --------------------------------------------
    expect(p.locator(".finding")).to_have_count(len(FINDINGS))
    expect(p.locator(".finding--critical")).to_have_count(len(CRIT))
    for fid, f in FINDINGS.items():
        c = card(p, fid)
        expect(c.locator(".chip--cls")).to_have_text(f["cls"])
        later = f.get("steps", [])[1:]
        refined = f", refined at {', '.join(map(str, later))}" if later else ""
        expect(c.locator(".finding__where")).to_have_text(f"found at step {f['step']}{refined}, L{f['level']}")
        expect(c.locator(".chip--state")).to_have_text("needs review" if f["state"] == "pending_review" else f["state"])
        expect(c.locator("dl")).to_contain_text(f"{f['measure']['area_px']:,}")
        if f["critical"]:
            expect(c.locator(".badge")).to_have_text("NEEDS A HUMAN")
            expect(c.get_by_role("button", name="Approve")).to_be_enabled()
            expect(c.get_by_role("button", name="Reject")).to_be_enabled()
    assert "mm" not in p.locator(".finding").first.inner_text().replace("millimetres", "")
    sm = DONE["summary"]
    expect(p.locator("#done .tile")).to_have_count(6)
    expect(p.locator("#done .tile__v").nth(0)).to_have_text(str(sm["steps"]))
    expect(pending_tile(p)).to_have_text(str(sm["critical_pending"]))
    expect(p.locator("#done .chips .chip")).to_have_count(len(UNVISITED))
    if not UNVISITED:
        expect(p.locator("#done")).to_contain_text("none: every candidate was visited")
    expect(p.locator("#done")).to_contain_text(f"stopped by {DONE['cause']}")
    s.to(".win--findings")
    s.shot("desktop-findings.png")
    approve_all(p, CRIT)
    expect(p.locator("#done .pill")).to_have_text("completed")

    # --- replay slider, trace click, heatmap, play (positions come from the fixture) ------------
    s.to(".win--view")
    value = lambda meter: p.locator(f"[data-meter={meter}] .meter__val").evaluate("e => Number(e.dataset.value)")
    k = 2
    p.locator("#replay").fill(str(k))
    expect(p.locator("#replay-out")).to_have_text(f"step {k} / {N_STEPS}")
    until(lambda: abs(value("pixels") - STEPS[k - 1]["pixels_frac"] * 100) < 1e-6, "pixels meter to follow the slider")
    until(lambda: abs(value("usd") - STEPS[k - 1]["usd_used"]) < 1e-9, "$ meter to follow the slider")
    expect(p.locator(".trace__line.is-current")).to_have_count(1)
    expect(p.locator("#step-detail")).to_contain_text(f"step {STEPS[k - 1]['step']}")
    p.locator("#replay").fill(str(N_STEPS))
    until(lambda: abs(value("pixels") - STEPS[-1]["pixels_frac"] * 100) < 1e-6, "pixels meter at the last step")
    mid = STEPS[N_STEPS // 2]
    p.locator(f".trace__line[data-step='{mid['step']}'] button").click()
    expect(p.locator("#replay-out")).to_have_text(f"step {N_STEPS // 2 + 1} / {N_STEPS}")
    with_heat = p.evaluate("document.getElementById('stage').toDataURL().length")
    p.locator("label.switch").click()
    time.sleep(0.2)
    assert p.evaluate("document.getElementById('stage').toDataURL().length") != with_heat, "heatmap toggle changed nothing"
    p.locator("label.switch").click()
    p.locator("#replay").fill("1")
    p.click("#play")
    expect(p.locator("#play")).to_have_text("Pause")
    until(lambda: p.locator("#replay").evaluate("e => Number(e.value)") > 1, "play to advance")
    p.click("#play")
    expect(p.locator("#play")).to_have_text("Play")
    p.locator("#replay").fill(str(min(4, N_STEPS)))
    s.shot("desktop-viewport.png")

    # --- work order download -----------------------------------------------------------------
    s.to("#done")
    with p.expect_download() as dl:
        p.get_by_role("button", name="Download work_order.json").click()
    order = json.loads(Path(dl.value.path()).read_text())
    assert order["id"] == WO_ID and order["status"] == "ready" and len(order["approvals"]) == len(CRIT), order["approvals"]
    s.scroll_through()
    s.to(".win--view", "start")
    s.shot("desktop-app-done.png")
    s.to("#how")
    s.shot("desktop-how.png")
    s.to("#benchmarks")
    s.shot("desktop-bench.png")
    s.to("#failures")
    s.shot("desktop-failures.png")
    off = hscroll_offenders(p)
    assert off["sw"] <= off["iw"] and not off["off"], off
    s.check_clean()
    s.context.close()
    print(f"ok  desktop mock: kill, resume, {len(FINDINGS)} findings ({len(CRIT)} critical), review, replay, download")


def test_live_api_contract(browser, base):
    """No ?mock: the real fetch path against a local /api/* stand-in that streams the fixture."""
    third = [i for i, e in enumerate(EVENTS) if e["type"] == "step"][2]
    first3 = [move(s) for s in STEPS[:3]]
    truncated = JOB_LINES[: third + 1]                      # the stream dies after step 3, before `done`
    patched_job = json.loads(JOB_LINES[0])
    patched_job["privacy"] = {**patched_job.get("privacy", {}), "faces_blurred": 1, "boxes": [BOX]}
    first_step = json.loads(JOB_LINES[next(i for i, e in enumerate(EVENTS) if e["type"] == "step")])
    first_step["runtime"]["p50_ms"].update({"segment:accurate": 30, "pyrDown": 2, "saliency": 4, "quality": 1})   # extra op keys
    full = [json.dumps(patched_job), *JOB_LINES[1:]]
    full[next(i for i, e in enumerate(EVENTS) if e["type"] == "step")] = json.dumps(first_step)
    LIVE["plans"] = [{"lines": truncated, "delay": 0}, {"lines": full, "delay": 0}, {"lines": JOB_LINES, "delay": 0.25}]
    LIVE["jobs"].clear(); LIVE["kills"].clear(); LIVE["reviews"].clear(); LIVE["tokens"] = 0
    LIVE["order"] = json.loads(json.dumps(DONE["work_order"]))

    s = Session(browser, "live-api", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/")
    expect(p.locator("#api-status")).to_contain_text("stock (not COOL)", timeout=T)
    expect(p.locator("#look")).to_be_disabled()
    p.locator(".thumb").click()
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    expect(p.locator("#prep")).to_contain_text("MB")
    p.click("#sound")                                     # WebAudio blips and chime must not throw
    expect(p.locator("#sound")).to_have_attribute("aria-pressed", "true")
    p.locator("label.toggle", has_text="p95_spike").click()
    p.locator("#cap-px").fill("45")
    p.click("#look")
    expect(p.locator("#job-pill")).to_have_text("killed", timeout=T)
    expect(p.locator("#job-text")).to_contain_text("stream ended")
    p.locator("label.switch").click()                      # heatmap off: photo, candidate boxes and nothing else
    p.locator("#replay").fill("0")
    time.sleep(0.4)
    plain = blur_energy(p)
    assert plain > 1, f"probe region has no texture to blur: {plain}"
    p.click("#resume")
    expect(p.locator("#job-pill")).to_have_text(re.compile(DONE_LABEL, re.IGNORECASE), timeout=T)
    first, second = LIVE["jobs"]
    q = first["query"]
    assert first["headers"]["content-type"] == "image/jpeg" and first["size"] == len(MOCK_JPG), "the full file is uploaded, not the thumbnail"
    assert q == {"pixel_cap": ["0.45"], "usd_cap": ["0.05"], "seconds_cap": ["20"], "chaos": ["p95_spike"]}, q
    assert "x-drishyam-replay" not in first["headers"]
    raw = second["headers"]["x-drishyam-replay"]
    assert "=" not in raw and "+" not in raw and "/" not in raw, "header must be base64url without padding"
    assert json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))) == first3
    expect(p.locator(".t-act--halt")).to_have_count(len(HALTS))             # halt step: no window, observation or crop
    assert p.locator("#replay").evaluate("e => Number(e.max)") == N_STEPS
    p.locator("#replay").fill(str(N_STEPS))
    last = STEPS[-1].get("rationale")
    if last:
        expect(p.locator("#step-detail")).to_contain_text(last)
    p.locator("#replay").fill("0")
    time.sleep(0.4)
    blurred = blur_energy(p)
    assert blurred < plain * 0.6, f"privacy.boxes region should be blurred: {plain:.2f} -> {blurred:.2f}"
    p.locator("#replay").fill(str(N_STEPS))
    p.locator("label.switch").click()                      # heatmap back on
    expect(p.locator(".finding")).to_have_count(len(FINDINGS))
    approve_all(p, CRIT)
    expect_tokens = [DONE["token"], *[f"tok-{i}" for i in range(1, len(CRIT))]]
    assert LIVE["reviews"] == [{"token": t, "finding_id": fid, "decision": "approve", "reviewer": "A. Engineer"}
                               for t, fid in zip(expect_tokens, CRIT)], LIVE["reviews"]   # each review spends the latest token
    s.to(".win--view")
    s.shot("desktop-privacy-blur.png")

    # --- KILL mid-run reaches the server with the job id and kill_token ---------------------------
    p.click("#look")
    expect(p.locator(".trace__line[data-step='2']")).to_be_visible(timeout=T)
    p.click("#kill")
    expect(p.locator("#job-pill")).to_have_text("killed")
    until(lambda: LIVE["kills"], "POST /api/kill")
    assert LIVE["kills"] == [{"job": JOB["job"], "kill_token": JOB["kill_token"]}], LIVE["kills"]
    s.check_clean()
    s.context.close()
    print("ok  live API path: query, replay header, review tokens, /api/kill, privacy.boxes, halt step")


def test_mobile(browser, base):
    s = Session(browser, "mobile", viewport={"width": 375, "height": 812}, device_scale_factor=2, is_mobile=True, has_touch=True)
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    s.scroll_through()
    off = hscroll_offenders(p)
    assert off["sw"] <= 375 and not off["off"], f"horizontal overflow before the run: {off}"
    s.shot("mobile-hero.png")
    p.click("#look")
    expect(p.locator("#kill")).to_be_visible()
    box = p.locator("#kill").bounding_box()
    assert 0 <= box["y"] and box["y"] + box["height"] <= 812, f"KILL not on screen: {box}"
    expect(p.locator("#job-pill")).to_have_text(re.compile(DONE_LABEL, re.IGNORECASE), timeout=T)
    expect(p.locator(".finding")).to_have_count(len(FINDINGS))
    s.scroll_through()
    off = hscroll_offenders(p)
    assert off["sw"] <= 375 and not off["off"], f"horizontal overflow after the run: {off}"
    s.to(".win--view")
    s.shot("mobile-app.png")
    s.to(".win--findings")
    s.shot("mobile-findings.png")
    s.to("#benchmarks")
    s.shot("mobile-bench.png")
    s.page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
    s.shot("mobile-full.png", full_page=True)
    s.check_clean()
    s.context.close()
    print("ok  mobile 375 px: no horizontal scroll, KILL on screen")


def test_reduced_motion(browser, base):
    s = Session(browser, "reduced-motion", viewport={"width": 1280, "height": 800}, reduced_motion="reduce")
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    assert p.evaluate("getComputedStyle(document.querySelector('.art .d')).animationName") == "none"      # hero lines do not draw themselves
    assert p.evaluate("getComputedStyle(document.querySelector('.art .co')).animationName") == "none"     # callouts do not snap in
    assert p.evaluate("getComputedStyle(document.querySelector('.pop')).opacity") == "1"   # no hidden-until-scrolled state
    p.click("#look")
    expect(p.locator("#job-pill")).to_have_text(re.compile(DONE_LABEL, re.IGNORECASE), timeout=T)
    expect(p.locator(".finding")).to_have_count(len(FINDINGS))
    assert p.evaluate("getComputedStyle(document.querySelector('.finding')).animationName") == "none", "finding cards appear without drawing in"
    expect(p.locator("[data-meter=pixels] .meter__val")).to_have_text(f"{STEPS[-1]['pixels_frac'] * 100:.1f} %")   # counts land without animating
    s.check_clean()
    s.context.close()
    print("ok  prefers-reduced-motion: nothing draws in, no hidden windows, numbers still land")


def test_heading_names(browser, base):
    s = Session(browser, "a11y", viewport={"width": 1280, "height": 800})
    p = s.page
    p.goto(f"{base}/")
    names = p.evaluate("[...document.querySelectorAll('h1, h2')].map(h => h.textContent.trim())")
    assert names == ["DRISHYAM", "INSPECT", "HOW IT WORKS", "BENCHMARKS", "FAILURES"], names
    expect(p.get_by_role("heading", name="DRISHYAM", exact=True)).to_have_count(1)  # generated outline text is not read twice
    s.context.close()
    print("ok  headings: one accessible name each")


def test_pending_states(browser, base):
    s = Session(browser, "pending", with_data=False, viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/")
    expect(p.locator("#bench")).to_contain_text("benchmarks pending", timeout=T)
    expect(p.locator("#fail")).to_contain_text("failure gallery pending")
    expect(p.locator("#bench-cool")).to_contain_text("COOL benchmark pending")
    expect(p.locator("#cool-status")).to_contain_text("has not been done yet")
    expect(p.locator("#gallery-empty")).to_be_visible()
    expect(p.locator("#look")).to_be_disabled()
    expect(p.locator("#stat-recall .stat__v")).to_have_text("see benchmarks")
    s.to("#benchmarks")
    s.shot("desktop-pending.png")
    s.check_clean()
    s.context.close()
    print("ok  pending states: friendly empty states, no errors")


def test_cool_missing_fields(browser, base):
    """A cool.json holding little more than the headline gain still draws FIG. 4.3 and the banner, with dashes."""
    s = Session(browser, "cool-partial", viewport={"width": 375, "height": 812})
    partial = {"end_to_end": {"cool_gain": 0.97}, "arms": {"A": {}, "B": None}}
    s.context.route("**/results/cool.json", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(partial)))
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("#bench-cool tbody th")).to_have_text(["A"], timeout=T)  # the null arm is skipped
    expect(p.locator("#bench-cool tbody td").first).to_have_text("–")
    expect(p.locator("#bench-cool .kv__sub")).to_have_count(0)                # no expected range in the file, none shown
    expect(p.locator("#cool-status")).to_have_text("The COOL/Graviton run has been measured (FIG. 4.3): a whole job ran at "
                                                   "0.97× the speed of the stock wheel on the same instance type.")
    s.check_clean()
    s.context.close()
    print("ok  COOL file with missing fields: dashes, no errors")


def test_halt_governor_note_and_critical_trigger(browser, base):
    """A halt record carrying a governor override shows its chip and detail row; every critical card shows
    the look that made it critical."""
    events = [json.loads(ln) for ln in JOB_LINES]
    halt = max(i for i, e in enumerate(events) if e["type"] == "step")
    events[halt]["governor"] = {"cause": "p95>SLA", "causes": ["p95>SLA"],
                                "rationale": "p95(segment) 900 ms > SLA 500 ms; model -> fast"}
    critical = {e["finding"]["id"] for e in events if e["type"] == "finding" and e["finding"].get("critical")}
    LIVE["plans"] = [{"lines": [json.dumps(e) for e in events], "delay": 0}]
    LIVE["jobs"].clear(); LIVE["kills"].clear(); LIVE["reviews"].clear(); LIVE["tokens"] = 0
    LIVE["order"] = json.loads(json.dumps(DONE["work_order"]))
    s = Session(browser, "halt-governor", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/")
    p.locator(".thumb").click()
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    p.click("#look")
    expect(p.locator("#job-pill")).to_have_text(re.compile(DONE_LABEL, re.IGNORECASE), timeout=T)
    last = p.locator("#trace .trace__line[data-step]").last
    expect(last.locator(".chip", has_text="p95>SLA")).to_have_count(1)
    last.locator("button").click()
    expect(p.locator("#step-detail")).to_contain_text("governor")
    expect(p.locator("#step-detail")).to_contain_text("model -> fast")
    expect(p.locator(".finding__trigger")).to_have_count(len(critical))
    s.check_clean()
    s.context.close()
    print("ok  halt governor note and critical trigger rendered")


if __name__ == "__main__":
    test_static_csp_hygiene()
    srv, base = serve()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            test_desktop_mock(browser, base)
            test_live_api_contract(browser, base)
            test_mobile(browser, base)
            test_reduced_motion(browser, base)
            test_heading_names(browser, base)
            test_pending_states(browser, base)
            test_cool_missing_fields(browser, base)
            test_halt_governor_note_and_critical_trigger(browser, base)
            browser.close()
    finally:
        srv.shutdown()
    print(f"ALL OK. screenshots in {SHOTS}")
