"""Browser test for the Drishyam UI under the production CSP.

Run:  make e2e   (or: python tests/e2e/test_ui_mock.py; needs playwright + chromium)
DRISHYAM_E2E_PORT pins the local test server's port (default: any free port); DRISHYAM_SHOTS is the screenshot folder.

Every expectation about the job (step count, findings, critical ids, planner events, causes, done status,
unvisited regions, work-order id, budget numbers) is DERIVED from public/fixtures/job.ndjson at start-up,
so regenerating the fixture from the real API never needs a test edit. The fixture is never hand-written here.

The page shows a run at reading speed (one decision every few seconds), so most scenarios press "Skip to end"
once the run has started; the guided and keyboard scenarios check the pacing itself.

Scenarios (a real Chromium, the document served with the production CSP header):
  1. desktop, ?mock=1: nav, hero, gallery, KILL mid-run, Resume, the trace's because/saw rows and its finding and
     review lines, narrator, camera panel, key, findings grouped by region, human review of every critical finding,
     replay slider, heatmap and tiling toggles, Show on photo, work order and trace downloads, screenshots.
  2. live-API path (no ?mock) against a local stand-in for /api/*: query string, X-Drishyam-Replay header,
     x-amz-content-sha256 on every POST, /api/review tokens, /api/kill {job, kill_token} answered by a kill-switch
     halt, privacy.boxes blur, halt step.
  3. guided run: pace_ms on the request, the paced-for-viewing label, the viewer in view, the critical NOTE held,
     the recap in view, hand-off to the reviewer, and Try KILL (planner_timeout, shown live) halted by KILL.
  4. photo, NOTE and newest trace line on one screen (1440x900; 375x812 with the crop and the replay label),
  5. keyboard only, 6. KILL after the job had finished, 7. mobile 375 px, 8. prefers-reduced-motion,
  9. heading names, 10. pending states, 11. halt governor note.
Any CSP violation, console error or page error fails the run. wait_for_function is avoided on purpose
(it is blocked by a strict CSP): every wait uses locators or polls page.evaluate.
"""
import base64
import functools
import hashlib
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
PORT = int(os.environ.get("DRISHYAM_E2E_PORT", "0"))

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'")
T = 30_000  # ms

# ---- everything the tests know about the job comes from the fixture ---------------------------------
JOB_LINES = [ln for ln in (PUBLIC / "fixtures/job.ndjson").read_text().splitlines() if ln.strip()]
EVENTS = [json.loads(ln) for ln in JOB_LINES]
JOB = EVENTS[0]
DONE = EVENTS[-1]
assert JOB["type"] == "job" and DONE["type"] == "done"
STEPS = [e for e in EVENTS if e["type"] == "step"]
N_STEPS = len(STEPS)
N_PLANNER = sum(e["type"] == "planner" for e in EVENTS)
FINDINGS = {}                                        # id -> the latest finding event for it
for _e in EVENTS:
    if _e["type"] == "finding":
        FINDINGS[_e["finding"]["id"]] = _e["finding"]
CRIT = [fid for fid, f in FINDINGS.items() if f["critical"]]
REGIONS = {f["region"] for f in FINDINGS.values()}
# a review request other than "<cls> is critical" (said by the finding's own line) gets a trace line of its own
REVIEWS = [e for e in EVENTS if e["type"] == "review_request" and not e["reason"].endswith(" is critical")]
N_TRACE = N_STEPS + N_PLANNER + len(FINDINGS) + len(REVIEWS)
UNVISITED = DONE["unvisited"]
PILL = {"awaiting_review": "needs sign-off", "completed": "done", "halted": "halted"}
DONE_LABEL = PILL[DONE["status"]]
WO_ID = DONE["work_order"]["id"]
GOVERNOR = {"p95>SLA", "warp-spike", "usd≥80%", "cap-hit", "kill-switch", "planner-fallback"}
MOVING = ("gaze", "escalate", "verify")
HALTS = [s for s in STEPS if s["decision"]["action"] == "halt"]
assert N_STEPS >= 4 and CRIT, "the fixture should hold several steps and a critical finding"
assert REVIEWS and N_PLANNER >= 2, "the fixture should hold a review request and both planner drills (a recording is one sample)"


def move(step):
    a = step["decision"]["args"]
    return [step["decision"]["action"], a["x"], a["y"], a["level"], *([a["size"]] if "size" in a else [])]


MOVES = [move(s) for s in STEPS if s["decision"]["action"] in MOVING]

MOCK_JPG = (PUBLIC / "fixtures/mock.jpg").read_bytes()
THUMB = next(iter(sorted((PUBLIC / "samples/thumbs").glob("*.jpg"))), None)
THUMB_BYTES = THUMB.read_bytes() if THUMB else MOCK_JPG[:1000]
SUMMARY = {
    "dataset": "synthetic-200 (labelled synthetic)",
    "n_images": 200,
    # the fields the hero cards read; the numbers are made up for the test
    "headline_stats": {"budget": 0.3, "target": {"recall_ratio": 0.9}, "recall_ratio_vs_tiling": 0.946,
                       "recall_ratio_vs_downscale": 1.6, "recall_ratio_vs_downscale_ci": [1.4, 1.8]},
    "agent_metrics": {"by_budget": {"0.3": {"halt_causes": {"cap-hit": 150, "stop-rule": 50}},
                                    "1.0": {"halt_causes": {"cap-hit": 10, "stop-rule": 190}}},
                      "planner_recovery": {"timeout": {"fired": 20}, "malformed": {"fired": 20}, "recovered_total": 40},
                      "governor_response": {"fired": 10, "recovered": 9}},
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
# real photos: every policy within 0.05 of the others, as on CODEBRIM, so the hero card says "about 1 in N"
REAL = {**SUMMARY, "dataset": "CODEBRIM (test fixture)", "policies": {
    "downscale": [{"pixels_frac": 0.07, "recall": 0.2, "recall_iou_only": 0.08}],
    "tiling": [{"pixels_frac": 0.1, "recall": 0.21, "recall_iou_only": 0.08}, {"pixels_frac": 0.42, "recall": 0.22, "recall_iou_only": 0.08},
               {"pixels_frac": 1.62, "recall": 0.21, "recall_iou_only": 0.06}],
    "drishyam": [{"pixels_frac": 0.09, "recall": 0.13, "recall_iou_only": 0.05},
                 {"pixels_frac": 0.29, "recall": 0.2, "recall_iou_only": 0.09, "chance_recall": 0.14, "chance_recall_iou_only": 0.02},
                 {"pixels_frac": 0.44, "recall": 0.22, "recall_iou_only": 0.1}]}}
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
    "/results/failures.json": ("application/json", json.dumps(FAILURES).encode()),
    "/results/failures/f01.jpg": ("image/jpeg", MOCK_JPG),
}


def killed_tail(step_no):
    """What the real server sends once /api/kill lands: a kill-switch halt record and a halted done record."""
    last = STEPS[-1]
    halt = {**last, "step": step_no, "cause": "kill-switch", "rationale": "kill switch set; state preserved for resume"}
    halt.pop("governor", None)
    return [json.dumps(halt), json.dumps({**DONE, "status": "halted", "cause": "kill-switch",
                                          "summary": {**DONE["summary"], "steps": max(0, step_no - 1)}})]


# ---- a local stand-in for /api/* (streams NDJSON like the real server) --------------------------------
LIVE = {"plans": [], "jobs": [], "kills": [], "reviews": [], "order": None, "tokens": 0}


def reset_live(plans):
    LIVE.update(plans=plans, jobs=[], kills=[], reviews=[], tokens=0, order=json.loads(json.dumps(DONE["work_order"])))


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
                                    "max_upload_bytes": 4400000, "max_pixels": 25000000,
                                    "host": "local", "arch": "x86_64", "commit": "0123456789abcdef"})
        super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        headers = {k.lower(): v for k, v in self.headers.items()}
        signed = headers.get("x-amz-content-sha256") == hashlib.sha256(raw).hexdigest()
        if url.path == "/api/jobs":
            LIVE["jobs"].append({"query": parse_qs(url.query), "headers": headers, "size": len(raw), "signed": signed})
            plan = LIVE["plans"].pop(0)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                step = 0
                for line in plan["lines"]:
                    is_step = json.loads(line)["type"] == "step"
                    if plan.get("killable") and LIVE["kills"] and is_step:  # like the server: halt before the next look
                        self.wfile.write("".join(f"{out}\n" for out in killed_tail(step + 1)).encode())
                        break
                    step += is_step
                    self.wfile.write((line + "\n").encode())
                    self.wfile.flush()
                    time.sleep(plan["delay"])
            except (BrokenPipeError, ConnectionResetError):
                pass                                   # the browser aborted
        elif url.path == "/api/kill":
            LIVE["kills"].append({**json.loads(raw), "signed": signed})
            self.reply(200, {"ok": True})
        elif url.path == "/api/review":
            body = json.loads(raw)
            LIVE["reviews"].append({**body, "signed": signed})
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
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=str(PUBLIC)))
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
        if "Failed to load resource" in msg.text and not self.with_data:
            return  # 404s on the optional data files (the pending-state scenario)
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

    def fast(self, start="#look"):
        """Start a run and show it at once: the run itself is unchanged, only the reading-speed replay is skipped."""
        self.page.click(start)
        self.page.locator("#skip").click(timeout=T)


def until(fn, what, timeout=T / 1000, page=None):
    """Poll fn. Pass `page` when fn waits on a request: the sync API runs the route handlers (which forward
    requests to the stand-in server) only while the script is inside a Playwright call, so plain sleeps starve them."""
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        if page:
            page.wait_for_timeout(100)
        else:
            time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def hscroll_offenders(page):
    return page.evaluate("""() => ({
      sw: document.documentElement.scrollWidth, iw: window.innerWidth,
      off: [...document.querySelectorAll('body *')].filter(e => {
        if (e.closest('.table-wrap, .toasts, .gallery')) return false;  // scroll on their own, never widen the page
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


def viewer_in_view(page):
    """The photo canvas sits on screen, below the sticky nav and job bar."""
    return page.evaluate("""() => { const c = document.getElementById('stage').getBoundingClientRect();
      const bar = document.getElementById('taskbar').getBoundingClientRect();
      return c.top >= bar.bottom - 1 && c.bottom <= innerHeight + 1; }""")


BOX = [round(JOB["width"] * 0.4), round(JOB["height"] * 0.4), round(JOB["width"] * 0.2), round(JOB["height"] * 0.2)]


def recap_in_view(page):
    """The recap's top sits below the sticky nav and job bar, and every one of its buttons is on screen."""
    return page.evaluate("""() => { const top = Math.max(document.getElementById('nav').getBoundingClientRect().bottom,
        document.getElementById('taskbar').getBoundingClientRect().bottom);
      return document.getElementById('recap').getBoundingClientRect().top >= top - 1
        && [...document.querySelectorAll('#recap button')].every((b) => b.getBoundingClientRect().bottom <= innerHeight + 1); }""")


def card(page, fid):
    return page.locator(f".finding[data-id='{fid}']")


def pending_tile(page):
    return page.locator("#done .tile__v").nth(5)


def approve_all(page, ids):
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
        expect(page.locator("#done .stack > .note").first).to_contain_text(f"sign-off ({left})" if left else "Signed: ready to send")
        expect(page.locator("#job-pill")).to_have_text("signed" if not left else DONE_LABEL)
        expect(page.locator("#state-4")).to_have_text("Done" if not left else "Waiting")


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
    assert 'id="stat-real"' in html, "WS3 fills the #stat-real hero slot"
    assert html.count("og:image") == 1 and '<meta name="twitter:card" content="summary_large_image">' in html, "one share card"
    print("ok  static CSP hygiene")


def test_desktop_mock(browser, base):
    s = Session(browser, "desktop", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("h1.wordmark")).to_be_visible()
    cta = p.locator("#guided").bounding_box()
    assert cta["y"] + cta["height"] <= 900, f"the primary CTA must sit above the fold: {cta}"
    expect(p.locator(".nav__links a")).to_have_text(["Demo", "How it works", "Results", "Docs"])
    expect(p.locator(".nav__menu")).to_be_hidden()                # the phone menu only
    expect(p.locator("#look")).to_be_enabled(timeout=T)          # mock.jpg is preloaded
    expect(p.locator("#state-1")).to_have_text("Done")
    expect(p.locator("#say-1")).to_contain_text("Ready:")
    expect(p.locator(".thumb")).to_have_count(1)                  # routed sample manifest
    expect(p.locator(".thumb img")).to_have_attribute("src", re.compile(r"samples/thumbs/bridge_01\.jpg$"))  # manifest `thumb` is used
    expect(p.locator("#sound")).to_have_attribute("aria-pressed", "false")  # sound defaults to off
    expect(p.locator("#advanced")).not_to_have_attribute("open", "")  # budget and drills start folded away
    # the hero cards: one unit (a share of the image's own pixels), every value from the summaries
    hs = SUMMARY["headline_stats"]
    head = max((pt for pt in SUMMARY["policies"]["drishyam"] if pt["pixels_frac"] <= hs["budget"]), key=lambda pt: pt["pixels_frac"])
    expect(p.locator("#stat-recall .stat__v")).to_have_text(f"{round(100 * hs['recall_ratio_vs_tiling'])} %", timeout=T)
    expect(p.locator("#stat-recall .stat__k")).to_have_text(f"Synthetic test set ({SUMMARY['n_images']} photos)")
    expect(p.locator("#stat-recall .stat__s")).to_contain_text(f"reading {round(100 * head['pixels_frac'])} % of the image's pixels (tiling reads 100 %).")
    expect(p.locator("#stat-recall .stat__s")).to_contain_text("Target was 90 %: met. It beats one downscaled pass (1.60×, 95 % CI 1.40–1.80).")
    halts = SUMMARY["agent_metrics"]["by_budget"]
    expect(p.locator("#stat-cost .stat__k")).to_have_text("Stops on its own")
    expect(p.locator("#stat-cost .stat__v")).to_have_text(f"{halts['1.0']['halt_causes']['stop-rule']} of 200 jobs")
    expect(p.locator("#stat-cost .stat__s")).to_contain_text(f"under the 30 % cap the cap binds first ({halts['0.3']['halt_causes']['cap-hit']} of 200)")
    expect(p.locator("#stat-ops .stat__v")).to_have_text("49 / 50 recovered")
    expect(p.locator("#stat-real .stat__k")).to_have_text("Real bridge photos (CODEBRIM, 200)")
    expect(p.locator("#stat-real .stat__v")).to_have_text("about 1 in 5")           # every policy within 0.05: recall about 0.21
    expect(p.locator("#stat-real .stat__s")).to_contain_text("Its boxes at random places score 0.14. With strict IoU matching it is 0.09 against 0.02 by chance.")
    expect(p.locator("#bench svg.chart")).to_be_visible()
    expect(p.locator("#bench .legend li")).to_have_count(3)                       # three policies; no trained-model summary is fetched
    n_iou = sum(len(v) for v in REAL["policies"].values())
    expect(p.locator("#bench-real .chart__plot path[d*='l10 10']")).to_have_count(n_iou)  # IoU-only crosses on FIG. 4.2 only
    expect(p.locator("#bench .chart__plot path[d*='l10 10']")).to_have_count(0)
    expect(p.locator("#bench details.data summary").last).to_have_text("Micro-benchmarks and provenance: test rig")  # folded away
    expect(p.locator("#bench .tbl caption")).to_be_hidden()
    expect(p.locator("#fail .fail")).to_have_count(1)
    expect(p.locator("#bench-cool")).to_contain_text("COOL on Graviton: not run.")  # FIG. 4.3 and the banner say why, nothing pending
    expect(p.locator("#cool-status")).to_contain_text("COOL on Graviton: not run.")
    p.keyboard.press("Tab")
    outline = p.evaluate("(() => { const s = getComputedStyle(document.activeElement); return s.outlineStyle + ' ' + s.outlineWidth; })()")
    assert outline.startswith("solid"), f"no visible keyboard focus: {outline}"
    s.scroll_through()
    s.shot("desktop-hero.png")
    s.to("#loop")                                                 # FIG. 3.0, served from public/docs/
    until(lambda: p.locator("#loop img").evaluate("i => i.complete && i.naturalWidth"), "FIG. 3.0 to load")
    expect(p.locator("#loop img")).to_be_visible()

    # --- the key explains the drawing --------------------------------------------------------
    s.to(".win--view")
    p.locator("#key summary").click()
    expect(p.locator(".key__list")).to_be_visible()
    expect(p.locator(".key__list")).to_contain_text("Pyramid level")
    expect(p.locator(".key__list")).to_contain_text("amber = inspected")
    p.locator("#key summary").click()

    # --- KILL mid-run -------------------------------------------------------------------------
    p.click("#look")
    expect(p.locator("#kill")).to_be_visible()
    expect(p.locator("#look")).to_be_disabled()
    expect(p.locator("#state-2")).to_have_text("Running")
    expect(p.locator("#narrator")).to_contain_text("Photo received", timeout=T)   # stage 1, before the first look
    expect(p.locator("#stages li.is-on")).to_have_count(5, timeout=T)            # job and overview stages
    expect(p.locator(".trace__line[data-step='1']")).to_be_visible(timeout=T)
    assert viewer_in_view(p), "the viewer must be on screen while the run plays"
    p.click("#kill")
    expect(p.locator("#job-pill")).to_have_text("killed")
    expect(p.locator("#job-text")).to_contain_text("Killed at step")
    expect(p.locator("#kill")).to_be_hidden()
    expect(p.locator("#resume")).to_be_visible()
    n = p.locator(".trace__line").count()
    seen = p.locator(".trace__line[data-step]").count()
    assert 1 <= seen < N_STEPS and n < N_TRACE, f"kill should stop the stream mid-run, trace lines={n} of {N_TRACE}"
    time.sleep(1.2)
    assert p.locator(".trace__line").count() == n, "stream kept going after KILL"
    assert p.locator("#replay").evaluate("e => Number(e.max)") == seen
    assert not s.api_calls, f"mock mode must not touch the network API: {s.api_calls}"

    # --- Resume: replays the same moves, then finishes ---------------------------------------
    replayed = len(MOVES[:seen])
    s.fast("#resume")
    shot_taken = False
    end = time.time() + T / 1000
    while time.time() < end and p.locator("#job-pill").inner_text().strip().lower() != DONE_LABEL:
        if not shot_taken and p.locator(".win--toast").count() > 0:
            s.shot("desktop-running.png")
            shot_taken = True
        time.sleep(0.15)
    expect(p.locator("#job-pill")).to_have_text(DONE_LABEL, timeout=T)
    live_causes = [c for st in STEPS if st["step"] > replayed       # a halt record may also carry a governor note
                   for c in (st["cause"], st.get("governor", {}).get("cause")) if c]
    if any(c in GOVERNOR for c in live_causes):
        expect(p.locator(".win--toast")).to_have_count(1)            # one revision note at a time
        toast, trace = p.locator(".win--toast").bounding_box(), p.locator("#step-why").bounding_box()
        assert toast["y"] + toast["height"] <= trace["y"] or toast["x"] >= trace["x"] + trace["width"], "the note covers the trace"
    expect(p.locator(".trace__line")).to_have_count(N_TRACE)
    expect(p.locator(".trace__line--planner")).to_have_count(N_PLANNER)
    expect(p.locator(".trace__line[data-step]")).to_have_count(N_STEPS)
    expect(p.locator(".trace__line--finding")).to_have_count(len(FINDINGS))
    expect(p.locator(".trace__line--review")).to_have_count(len(REVIEWS))
    expect(p.locator(".trace__line--review").first).to_contain_text(REVIEWS[0]["reason"])
    expect(p.locator(".trace__line--review").first).to_contain_text("→ human")
    expect(p.locator(".t-act--halt")).to_have_count(len(HALTS))
    shown = Counter(["replay"] * replayed + live_causes)       # replayed steps are tagged `replay`
    for cause, count in shown.items():
        chips = p.locator(".trace__line .chip", has_text=re.compile(f"^{re.escape(cause)}$"))
        assert chips.count() == count, f"cause chip {cause!r}: {chips.count()} shown, {count} expected"
    # every step line says why (visibly, not in a tooltip) and, for a look, what it saw
    expect(p.locator(".trace__line[data-step] .trace__why")).to_have_count(sum(bool(st.get("rationale")) for st in STEPS))
    expect(p.locator(".trace__line[data-step] .trace__saw")).to_have_count(sum("observation" in st for st in STEPS))
    assert p.locator(".trace__btn[title]").count() == 0, "the reason must be visible text, not a title tooltip"
    follow = next(st for st in STEPS if st["step"] > replayed and st["decision"]["action"] in ("escalate", "verify"))
    expect(p.locator(f".trace__line[data-step='{follow['step']}'] .trace__why")).to_contain_text("(from #")
    crit_line = p.locator(".trace__line--finding", has_text="human sign-off requested").first
    expect(crit_line).to_contain_text("because")

    # --- the run in plain words: narrator, step states, recap ----------------------------------
    pending = DONE["summary"]["critical_pending"]
    expect(p.locator("#narrator")).to_contain_text(f"{pending} critical finding")
    expect(p.locator("#state-2")).to_have_text("Done")
    expect(p.locator("#state-4")).to_have_text("Waiting")
    expect(p.locator("#say-4")).to_contain_text(f"Waiting for your sign-off ({pending})")
    expect(p.locator("#recap")).to_contain_text("What you just saw")
    expect(p.locator("#recap")).to_contain_text(f"{N_STEPS} decisions, every one with a recorded cause")
    expect(p.locator("#cam-img")).to_be_visible()
    expect(p.locator("#cam-img")).to_have_attribute("src", re.compile(r"^data:image/jpeg;base64,"))
    expect(p.locator("#cam-cap")).to_contain_text(re.compile(r"L\d · 1:\d+"))

    # --- findings: one card per id, grouped by region ------------------------------------------
    expect(p.locator(".finding")).to_have_count(len(FINDINGS))
    expect(p.locator(".finding--critical")).to_have_count(len(CRIT))
    expect(p.locator(".fgroup")).to_have_count(len(REGIONS))
    for fid, f in FINDINGS.items():
        c = card(p, fid)
        expect(c.locator(".chip--cls")).to_have_text(f["cls"])
        later = f.get("steps", [])[1:]
        refined = f", refined at {', '.join(map(str, later))}" if later else ""
        expect(c.locator(".finding__where")).to_have_text(f"found at step {f['step']}{refined}, L{f['level']}")
        expect(c.locator(".chip--state")).to_have_text("needs review" if f["state"] == "pending_review" else f["state"])
        expect(c.locator("dl")).to_contain_text(f"{f['measure']['area_px']:,}")
        expect(p.locator(f".fgroup[data-region='{f['region']}'] .finding[data-id='{fid}']")).to_have_count(1)
        if f["critical"]:
            expect(c.locator(".badge")).to_have_text("NEEDS A HUMAN")
            expect(c.locator(".review__only")).to_have_text("The agent cannot approve this. Only you can.")
            expect(c.locator(".review__why")).to_contain_text("Why you:")
            expect(c.get_by_role("button", name="Approve")).to_be_enabled()
            expect(c.get_by_role("button", name="Reject")).to_be_enabled()
    assert "mm" not in p.locator(".finding").first.inner_text().replace("millimetres", "")
    sm = DONE["summary"]
    expect(p.locator("#done .tile")).to_have_count(6)
    expect(p.locator("#done .tile__v").nth(0)).to_have_text(str(sm["steps"]))
    expect(p.locator("#done .tile__v").nth(2)).to_have_text(f"${sm['usd']:.6f}")
    expect(pending_tile(p)).to_have_text(str(sm["critical_pending"]))
    expect(p.locator("#done .chips .chip")).to_have_count(len(UNVISITED))
    if not UNVISITED:
        expect(p.locator("#done")).to_contain_text("none: every candidate region was looked at")
    expect(p.locator("#done")).to_contain_text(f"({DONE['cause']})")
    expect(p.locator("#done")).to_contain_text("salient area seen at any level (incl. survey)")
    s.to(".win--findings")
    s.shot("desktop-findings.png")

    # a trace line opens its card; a card points the photo at its finding
    p.locator(".trace__line--finding").first.locator("button").click()
    until(lambda: p.evaluate("document.activeElement.dataset.id") == next(iter(FINDINGS)), "the trace line to focus its card")
    first = FINDINGS[CRIT[0]]
    card(p, CRIT[0]).get_by_role("button", name="Show on photo").click()
    expect(p.locator("#replay-out")).to_have_text(f"step {[st['step'] for st in STEPS].index(first['step']) + 1} / {N_STEPS}")
    approve_all(p, CRIT)
    expect(p.locator("#done .pill")).to_have_text("signed")

    # --- replay slider, trace click, heatmap and tiling toggles, play --------------------------
    s.to(".win--view")
    value = lambda meter: p.locator(f"[data-meter={meter}] .meter__val").evaluate("e => Number(e.dataset.value)")
    k = 2
    p.locator("#replay").fill(str(k))
    expect(p.locator("#replay-out")).to_have_text(f"step {k} / {N_STEPS}")
    until(lambda: abs(value("pixels") - STEPS[k - 1]["pixels_frac"] * 100) < 1e-6, "pixels meter to follow the slider")
    until(lambda: abs(value("usd") - STEPS[k - 1]["usd_used"]) < 1e-9, "$ meter to follow the slider")
    expect(p.locator(".trace__line.is-current")).to_have_count(1)
    expect(p.locator("#step-detail")).to_contain_text(f"step {STEPS[k - 1]['step']}")
    expect(p.locator("#step-detail")).to_contain_text("Why this look")
    expect(p.locator("#step-detail")).to_contain_text("What it saw")
    expect(p.locator("#narrator")).not_to_contain_text("critical finding")   # the NOTE follows the picked step
    p.locator("#replay").fill(str(N_STEPS))
    until(lambda: abs(value("pixels") - STEPS[-1]["pixels_frac"] * 100) < 1e-6, "pixels meter at the last step")
    mid = STEPS[N_STEPS // 2]
    p.locator(f".trace__line[data-step='{mid['step']}'] button").click()
    expect(p.locator("#replay-out")).to_have_text(f"step {N_STEPS // 2 + 1} / {N_STEPS}")
    canvas = lambda: p.evaluate("document.getElementById('stage').toDataURL().length")
    with_heat = canvas()
    p.locator("label.switch:has(#heat)").click()
    time.sleep(0.2)
    assert canvas() != with_heat, "heatmap toggle changed nothing"
    p.locator("label.switch:has(#heat)").click()
    time.sleep(0.2)
    before = canvas()
    p.locator("label.switch:has(#tiles)").click()
    expect(p.locator("#tiles-note")).to_contain_text(f"vs {sm['steps']} agent looks")
    time.sleep(0.2)
    assert canvas() != before, "tiling toggle drew nothing"
    s.shot("desktop-tiling.png")
    p.locator("label.switch:has(#tiles)").click()
    p.locator("#replay").fill("1")
    p.click("#play")
    expect(p.locator("#play")).to_have_text("Pause")
    until(lambda: p.locator("#replay").evaluate("e => Number(e.value)") > 1, "play to advance")
    p.click("#play")
    expect(p.locator("#play")).to_have_text("Play")
    p.locator("#replay").fill(str(min(4, N_STEPS)))
    s.shot("desktop-viewport.png")

    # --- downloads: the signed work order and the trace -----------------------------------------
    s.to("#done")
    with p.expect_download() as dl:
        p.get_by_role("button", name="Download work_order.json").click()
    saved = json.loads(Path(dl.value.path()).read_text())
    order = saved["work_order"]
    assert order["id"] == WO_ID and order["status"] == "ready" and len(order["approvals"]) == len(CRIT), order["approvals"]
    assert saved["alg"] == "HMAC-SHA256" and saved["token"] and "exp" in saved, saved.keys()
    with p.expect_download() as dl:
        p.get_by_role("button", name="Download trace.ndjson").click()
    lines = [json.loads(ln) for ln in Path(dl.value.path()).read_text().splitlines()]
    assert [e["type"] for e in lines] == [e["type"] for e in EVENTS], "trace.ndjson holds every event of the run, in order"
    s.scroll_through()
    s.to(".win--view", "start")
    s.shot("desktop-app-done.png")
    for sel, name in (("#how", "how"), ("#benchmarks", "bench"), ("#failures", "failures"), ("#safety", "safety"), ("#faq", "faq")):
        s.to(sel)
        s.shot(f"desktop-{name}.png")
    expect(p.locator(".nav__links a[aria-current]")).to_have_count(0)     # FAQ is not in the nav
    s.to("#how")
    expect(p.locator(".nav__links a[aria-current]")).to_have_text("How it works")
    off = hscroll_offenders(p)
    assert off["sw"] <= off["iw"] and not off["off"], off
    s.check_clean()
    s.context.close()
    print(f"ok  desktop mock: kill, resume, trace rows, {len(FINDINGS)} findings ({len(CRIT)} critical), review, replay, downloads")


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
    reset_live([{"lines": truncated, "delay": 0}, {"lines": full, "delay": 0}, {"lines": JOB_LINES, "delay": 0.25, "killable": True}])

    s = Session(browser, "live-api", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/")
    expect(p.locator("#api-status")).to_contain_text("stock (not COOL)", timeout=T)
    expect(p.locator("#api-status")).to_contain_text("host local · x86_64 · commit 0123456")   # /api/health host, arch, commit
    expect(p.locator("#live-where")).to_contain_text("on local (x86_64)")
    expect(p.locator("#recorded")).to_be_hidden()
    expect(p.locator("#look")).to_be_enabled(timeout=T)        # the first sample is preselected
    expect(p.locator(".thumb[aria-pressed='true']")).to_have_count(1)
    expect(p.locator("#prep")).to_contain_text("MB")
    p.click("#sound")                                     # WebAudio blips and chime must not throw
    expect(p.locator("#sound")).to_have_attribute("aria-pressed", "true")
    p.locator("#advanced summary").click()
    p.locator("label.toggle", has_text="p95_spike").click()
    p.locator("#cap-px").fill("45")
    s.fast("#run-bar")                                    # the Run button in the job bar runs the same job
    expect(p.locator("#job-pill")).to_have_text("killed", timeout=T)
    expect(p.locator("#job-text")).to_contain_text("stream ended")
    p.locator("label.switch:has(#heat)").click()           # heatmap off: photo, candidate boxes and nothing else
    p.locator("#replay").fill("0")
    time.sleep(0.4)
    plain = blur_energy(p)
    assert plain > 1, f"probe region has no texture to blur: {plain}"
    s.fast("#resume")
    expect(p.locator("#job-pill")).to_have_text(DONE_LABEL, timeout=T)
    first, second = LIVE["jobs"]
    q = first["query"]
    assert first["headers"]["content-type"] == "image/jpeg" and first["size"] == len(MOCK_JPG), "the full file is uploaded, not the thumbnail"
    assert q == {"pixel_cap": ["0.45"], "usd_cap": ["0.05"], "seconds_cap": ["20"], "chaos": ["p95_spike"]}, q
    assert first["signed"] and second["signed"], "x-amz-content-sha256 must be the hex SHA-256 of the body"
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
    p.locator("label.switch:has(#heat)").click()           # heatmap back on
    expect(p.locator(".finding")).to_have_count(len(FINDINGS))
    approve_all(p, CRIT)
    expect_tokens = [DONE["token"], *[f"tok-{i}" for i in range(1, len(CRIT))]]
    assert [{k: v for k, v in r.items() if k != "signed"} for r in LIVE["reviews"]] == [
        {"token": t, "finding_id": fid, "decision": "approve", "reviewer": "A. Engineer"}
        for t, fid in zip(expect_tokens, CRIT)], LIVE["reviews"]   # each review spends the latest token
    assert all(r["signed"] for r in LIVE["reviews"]), "every review POST carries x-amz-content-sha256"
    s.to(".win--view")
    s.shot("desktop-privacy-blur.png")

    # --- KILL mid-run reaches the server, which halts with kill-switch ----------------------------
    p.click("#look")
    until(lambda: p.locator("#trace .trace__line").count() == 0, "the previous job's trace to clear")  # never wait on stale lines
    expect(p.locator(".trace__line[data-step='1']")).to_be_visible(timeout=T)
    p.click("#kill")
    until(lambda: LIVE["kills"], "POST /api/kill", page=p)
    assert [{k: v for k, v in r.items() if k != "signed"} for r in LIVE["kills"]] == [{"job": JOB["job"], "kill_token": JOB["kill_token"]}], LIVE["kills"]
    assert LIVE["kills"][0]["signed"]
    expect(p.locator("#job-pill")).to_have_text("halted", timeout=T)
    expect(p.locator("#job-text")).to_contain_text("Halted by KILL")
    expect(p.locator(".trace__line .chip", has_text=re.compile("^kill-switch$"))).to_have_count(1)
    expect(p.locator("#resume")).to_be_visible()
    s.check_clean()
    s.context.close()
    print("ok  live API path: query, replay header, sha256 headers, review tokens, /api/kill -> kill-switch, privacy.boxes, halt step")


def test_guided(browser, base):
    """The hero CTA: paced live run (pace_ms), labelled as paced, viewer in view, the critical finding held on
    screen, then the recap in view and the hand-off to a person; its Try KILL starts a live run KILL can stop."""
    job = json.loads(JOB_LINES[0])
    job["pace_ms"] = 1000                                  # the server echoes the pacing it applied
    reset_live([{"lines": [json.dumps(job), *JOB_LINES[1:]], "delay": 0.05},
                {"lines": JOB_LINES, "delay": 0.25, "killable": True}])
    s = Session(browser, "guided", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/")
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    expect(p.locator("#guided")).to_contain_text(re.compile(r"about \d+ s"))
    p.click("#guided")
    until(lambda: LIVE["jobs"], "the guided job request", page=p)
    assert LIVE["jobs"][0]["query"].get("pace_ms") == ["1000"], LIVE["jobs"][0]["query"]
    expect(p.locator("#job-text")).to_contain_text("Paced for viewing: the agent waits 1 s between looks; timings exclude the wait.", timeout=T)
    expect(p.locator(".trace__line[data-step='1']")).to_be_visible(timeout=T)
    for _ in range(3):                                     # it stays on screen while the run plays
        assert viewer_in_view(p), "the viewer left the screen during the guided run"
        time.sleep(0.5)
    assert p.locator(".trace__line[data-step]").count() < N_STEPS, "a guided run is shown at reading speed, not in a burst"
    expect(p.locator("#narrator")).to_contain_text("whole photo")       # step 1 in plain words
    s.shot("desktop-guided-mid.png")
    # the first critical finding is the hand-off moment: its NOTE outlasts an ordinary decision (5 s)
    expect(p.locator("#narrator")).to_contain_text("Critical:", timeout=T)
    t0 = time.time()
    until(lambda: "Critical:" not in p.locator("#narrator").inner_text(), "the critical NOTE to give way")
    held = time.time() - t0
    assert held >= 7, f"the critical NOTE stayed {held:.1f} s, no longer than an ordinary decision"
    p.click("#skip")
    expect(p.locator("#job-pill")).to_have_text(DONE_LABEL, timeout=T)
    # it ends on the recap, on screen below the bars, and Sign them off hands over to the reviewer field
    until(lambda: p.evaluate("document.activeElement.id") == "sign-off", "focus on Sign them off")
    until(lambda: recap_in_view(p), "the recap and its buttons on screen")
    expect(p.locator("#recap")).to_contain_text("What you just saw")
    s.shot("desktop-guided-done.png")
    p.click("#sign-off")
    until(lambda: p.evaluate("document.activeElement.id") == f"rev-{CRIT[0]}", "focus on the first critical card's reviewer field")
    expect(card(p, CRIT[0]).locator(".review__only")).to_be_visible()
    # Try KILL: planner_timeout parks the job on the server, the page shows it live, and KILL halts it
    p.get_by_role("button", name="Try KILL").click()
    until(lambda: len(LIVE["jobs"]) == 2, "the Try KILL job request", page=p)
    q = LIVE["jobs"][1]["query"]
    assert q.get("chaos") == ["planner_timeout"] and q.get("pace_ms") == ["1000"], q
    expect(p.locator("#job-text")).to_contain_text("shown the moment it arrives", timeout=T)   # the job event cleared the old trace
    expect(p.locator("#skip")).to_be_hidden()              # live: nothing to skip
    expect(p.locator(".trace__line[data-step='1']")).to_be_visible(timeout=T)
    p.click("#kill")
    until(lambda: LIVE["kills"], "POST /api/kill", page=p)
    expect(p.locator("#job-pill")).to_have_text("halted", timeout=T)
    expect(p.locator(".trace__line .chip", has_text=re.compile("^kill-switch$"))).to_have_count(1)
    s.check_clean()
    s.context.close()
    print("ok  guided run: pace_ms sent, labelled paced, viewer in view, critical NOTE held, recap in view, hand-off, Try KILL halts")


def test_fits_on_screen(browser, base):
    """While a run plays, the photo and its NOTE share one screen: at 1440x900 with the newest trace line too, for the
    overview and every decision of the fixture (it holds the p95>SLA override and both planner drills that "Now break
    it" shows); at 375x812 with the camera crop over the photo and a visible replay label in the job bar."""
    for size, phone in (({"width": 1440, "height": 900}, False), ({"width": 375, "height": 812}, True)):
        s = Session(browser, f"fit-{size['width']}", viewport=size, is_mobile=phone, has_touch=phone, reduced_motion="reduce")
        p = s.page
        p.goto(f"{base}/?mock=1")
        expect(p.locator("#look")).to_be_enabled(timeout=T)
        p.click("#look")
        expect(p.locator(".trace__line[data-step='1']")).to_be_visible(timeout=T)
        top = p.evaluate("document.getElementById('stage').getBoundingClientRect().top")   # where run() put the photo
        if phone:
            expect(p.locator("#pip")).to_be_visible()
            assert p.locator("#job-text").evaluate("e => e.getBoundingClientRect().height") > 10, "the replay label is hidden"
        p.click("#skip")
        expect(p.locator("#job-pill")).to_have_text(DONE_LABEL, timeout=T)
        for k in range(N_STEPS + 1):                       # step 0 is the overview's NOTE
            p.locator("#replay").fill(str(k))
            ends = p.evaluate("""() => { const c = document.getElementById('stage').getBoundingClientRect().top;
              const end = (sel) => document.querySelector(sel).getBoundingClientRect().bottom - c;
              return {NOTE: end('.narrator'), trace: end('#trace')}; }""")
            if phone:
                del ends["trace"]                          # one column: the trace sits further down on a phone
            for what, end in ends.items():
                assert top + end <= size["height"] + 1, f"{size['width']} px, step {k}: the {what} ends {top + end - size['height']:.0f} px below the fold"
        s.check_clean()
        s.context.close()
    print("ok  fits: photo, NOTE and newest trace line on one 1440x900 screen; photo, crop, NOTE and replay label at 375x812")


def test_keyboard(browser, base):
    """Tab to the CTA, run it, skip with Enter, then sign off a critical finding without a mouse."""
    s = Session(browser, "keyboard", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    for _ in range(30):
        p.keyboard.press("Tab")
        if p.evaluate("document.activeElement.id") == "guided":
            break
    assert p.evaluate("document.activeElement.id") == "guided", "the guided CTA is reachable by Tab"
    p.keyboard.press("Enter")
    expect(p.locator("#job-pill")).to_have_text("running")
    until(lambda: p.evaluate("document.activeElement.id") == "skip", "focus to move to Skip to end")
    p.keyboard.press("Enter")
    expect(p.locator("#job-pill")).to_have_text(DONE_LABEL, timeout=T)
    until(lambda: p.evaluate("document.activeElement.id") == "sign-off", "focus on Sign them off")
    p.keyboard.press("Enter")
    until(lambda: p.evaluate("document.activeElement.id") == f"rev-{CRIT[0]}", "focus on the reviewer field")
    p.keyboard.type("K. Board")
    p.keyboard.press("Tab")
    p.keyboard.press("Enter")                               # Approve
    expect(card(p, CRIT[0]).locator(".chip--state")).to_have_text("approved", timeout=T)
    s.check_clean()
    s.context.close()
    print("ok  keyboard: CTA, Skip to end and a sign-off without a mouse")


def test_kill_after_finish(browser, base):
    """KILL pressed while the page is still replaying a job the server already finished says so."""
    s = Session(browser, "kill-late", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    p.click("#look")
    expect(p.locator("#job-text")).to_contain_text("Replay: the recorded agent took", timeout=T)   # done arrived; the replay is behind
    p.click("#kill")
    expect(p.locator("#job-text")).to_contain_text("had already finished", timeout=T)
    expect(p.locator("#job-pill")).to_have_text(DONE_LABEL)
    s.check_clean()
    s.context.close()
    print("ok  KILL after the job finished: says so instead of claiming a kill")


def test_mobile(browser, base):
    s = Session(browser, "mobile", viewport={"width": 375, "height": 812}, device_scale_factor=2, is_mobile=True, has_touch=True)
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    cta = p.locator("#guided").bounding_box()
    assert cta["y"] + cta["height"] <= 812, f"the primary CTA must sit above the fold on a phone: {cta}"
    expect(p.locator(".nav__links")).to_be_hidden()
    p.locator("#nav-menu summary").click()
    expect(p.locator(".nav__drop a", has_text="How it works")).to_be_visible()
    p.locator(".nav__drop a", has_text="Demo").click()
    expect(p.locator("#nav-menu")).not_to_have_attribute("open", "")  # the menu closes once a link is picked
    s.scroll_through()
    off = hscroll_offenders(p)
    assert off["sw"] <= 375 and not off["off"], f"horizontal overflow before the run: {off}"
    s.shot("mobile-hero.png")
    p.click("#look")
    expect(p.locator("#kill")).to_be_visible()
    box = p.locator("#kill").bounding_box()
    assert 0 <= box["y"] and box["y"] + box["height"] <= 812, f"KILL not on screen: {box}"
    p.click("#skip")
    expect(p.locator("#job-pill")).to_have_text(DONE_LABEL, timeout=T)
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
    print("ok  mobile 375 px: CTA above the fold, menu, no horizontal scroll, KILL on screen")


def test_reduced_motion(browser, base):
    s = Session(browser, "reduced-motion", viewport={"width": 1280, "height": 800}, reduced_motion="reduce")
    p = s.page
    p.goto(f"{base}/?mock=1")
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    assert p.evaluate("getComputedStyle(document.querySelector('.art .d')).animationName") == "none"      # hero lines do not draw themselves
    assert p.evaluate("getComputedStyle(document.querySelector('.art .co')).animationName") == "none"     # callouts do not snap in
    assert p.evaluate("getComputedStyle(document.querySelector('.pop')).opacity") == "1"   # no hidden-until-scrolled state
    p.click("#look")
    expect(p.locator(".trace__line[data-step='1']")).to_be_visible(timeout=T)
    assert p.locator(".trace__line[data-step]").count() < N_STEPS, "reduced motion keeps the reading-speed pacing"
    p.click("#skip")
    expect(p.locator("#job-pill")).to_have_text(DONE_LABEL, timeout=T)
    expect(p.locator(".finding")).to_have_count(len(FINDINGS))
    assert p.evaluate("getComputedStyle(document.querySelector('.finding')).animationName") == "none", "finding cards appear without drawing in"
    expect(p.locator("[data-meter=pixels] .meter__val")).to_have_text(f"{STEPS[-1]['pixels_frac'] * 100:.1f} %")   # counts land without animating
    s.check_clean()
    s.context.close()
    print("ok  prefers-reduced-motion: nothing draws in, pacing kept, no hidden windows, numbers still land")


def test_heading_names(browser, base):
    s = Session(browser, "a11y", viewport={"width": 1280, "height": 800})
    p = s.page
    p.goto(f"{base}/")
    names = p.evaluate("[...document.querySelectorAll('h1, h2')].map(h => h.textContent.trim())")
    assert names == ["DRISHYAM", "DEMO", "HOW IT WORKS", "BENCHMARKS", "FAILURES", "SAFETY AND HUMAN CONTROL", "QUESTIONS", "Docs and links"], names
    expect(p.get_by_role("heading", name="DRISHYAM", exact=True)).to_have_count(1)  # generated outline text is not read twice
    expect(p.get_by_role("navigation", name="Sections", exact=True)).to_have_count(1)
    s.context.close()
    print("ok  headings: one accessible name each")


def test_pending_states(browser, base):
    s = Session(browser, "pending", with_data=False, viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/")
    expect(p.locator("#bench")).to_contain_text("benchmarks pending", timeout=T)
    expect(p.locator("#fail")).to_contain_text("failure gallery pending")
    expect(p.locator("#bench-cool")).to_contain_text("COOL on Graviton: not run.")   # static: it does not wait for a file
    expect(p.locator("#cool-status")).to_contain_text("COOL on Graviton: not run.")
    expect(p.locator("#gallery-empty")).to_be_visible()
    expect(p.locator("#say-1")).to_contain_text("Drop your own JPEG or PNG")
    expect(p.locator("#state-1")).to_have_text("Waiting")
    expect(p.locator("#look")).to_be_disabled()
    expect(p.locator("#run-bar")).to_be_disabled()
    expect(p.locator("#stat-recall .stat__v")).to_have_text("see benchmarks")
    expect(p.locator("#stat-real")).to_be_hidden()                # an empty slot until WS3 fills it
    s.to("#benchmarks")
    s.shot("desktop-pending.png")
    s.check_clean()
    s.context.close()
    print("ok  pending states: friendly empty states, no errors")


def test_halt_governor_note_and_critical_trigger(browser, base):
    """A halt record carrying a governor override shows its chip and detail row; every critical card shows
    the look that made it critical."""
    events = [json.loads(ln) for ln in JOB_LINES]
    halt = max(i for i, e in enumerate(events) if e["type"] == "step")
    events[halt]["governor"] = {"cause": "p95>SLA", "causes": ["p95>SLA"],
                                "rationale": "p95(segment) 900 ms > SLA 500 ms; model -> fast"}
    critical = {e["finding"]["id"] for e in events if e["type"] == "finding" and e["finding"].get("critical")}
    reset_live([{"lines": [json.dumps(e) for e in events], "delay": 0}])
    s = Session(browser, "halt-governor", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/")
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    s.fast()
    expect(p.locator("#job-pill")).to_have_text(DONE_LABEL, timeout=T)
    last = p.locator("#trace .trace__line[data-step]").last
    expect(last.locator(".chip", has_text="p95>SLA")).to_have_count(1)
    last.locator("button").click()
    expect(p.locator("#step-detail")).to_contain_text("governor")
    expect(p.locator("#step-detail")).to_contain_text("model -> fast")
    expect(p.locator("#narrator")).to_contain_text("The segmenter got slow (p95 900 ms, over the 500 ms limit)")
    expect(p.locator(".finding__trigger")).to_have_count(len(critical))
    s.check_clean()
    s.context.close()
    print("ok  halt governor note and critical trigger rendered")


def test_planner_status_shows_the_fallback_chip(browser, base):
    """The API keeps a look's own cause and puts a failed planner consultation in `planner_status`: the trace
    still flags that look `planner-fallback`."""
    events = [json.loads(ln) for ln in JOB_LINES]
    fell = {e["step"]: e for e in events if e["type"] == "planner" and e["status"] in ("rejected", "fallback")}
    assert fell, "the fixture should hold a failed planner consultation"
    for e in events:
        if e["type"] == "step" and e["step"] in fell:
            e["cause"] = "best-score"
            e["planner_status"] = {"status": fell[e["step"]]["status"], "reason": fell[e["step"]]["reason"]}
    LIVE["plans"] = [{"lines": [json.dumps(e) for e in events], "delay": 0}]
    LIVE["jobs"].clear(); LIVE["kills"].clear(); LIVE["reviews"].clear(); LIVE["tokens"] = 0
    LIVE["order"] = json.loads(json.dumps(DONE["work_order"]))
    s = Session(browser, "planner-status", viewport={"width": 1440, "height": 900})
    p = s.page
    p.goto(f"{base}/")
    p.locator(".thumb").click()
    expect(p.locator("#look")).to_be_enabled(timeout=T)
    p.click("#look")
    p.click("#skip")
    expect(p.locator("#job-pill")).to_have_text(re.compile(DONE_LABEL, re.IGNORECASE), timeout=T)
    for n in fell:
        expect(p.locator(f"#trace .trace__line[data-step='{n}'] .chip--gov", has_text="planner-fallback")).to_have_count(1)
    s.check_clean()
    s.context.close()
    print("ok  planner_status rejected/fallback shows the planner-fallback chip")


if __name__ == "__main__":
    test_static_csp_hygiene()
    srv, base = serve()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            test_desktop_mock(browser, base)
            test_live_api_contract(browser, base)
            test_guided(browser, base)
            test_fits_on_screen(browser, base)
            test_keyboard(browser, base)
            test_kill_after_finish(browser, base)
            test_mobile(browser, base)
            test_reduced_motion(browser, base)
            test_heading_names(browser, base)
            test_pending_states(browser, base)
            test_halt_governor_note_and_critical_trigger(browser, base)
            test_planner_status_shows_the_fallback_chip(browser, base)
            browser.close()
    finally:
        srv.shutdown()
    print(f"ALL OK. screenshots in {SHOTS}")
