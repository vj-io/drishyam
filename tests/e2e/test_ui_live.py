"""End-to-end check of the real stack: the page, the FastAPI app and the agent, nothing mocked.

Run: make e2e   (or: python tests/e2e/test_ui_live.py [BASE_URL]; needs playwright + chromium)
Without BASE_URL it starts the app locally (uvicorn, production headers from infra/headers.json) on DRISHYAM_E2E_PORT
(default: any free port). With BASE_URL (e.g. the CloudFront URL) it runs the same flows against it.
Waits use locators and polling, never wait_for_function (blocked under the strict CSP).
"""

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SHOTS = Path(os.environ.get("DRISHYAM_SHOTS", "/tmp/drishyam-e2e"))
DONE = ("needs sign-off", "signed", "done", "halted", "budget spent")
OPTIONAL = ("/results/",)  # the page shows a pending state without these
JARGON = re.compile(r"\b\d+(?:\.\d+)?e[-+]?\d+\b|\b\w+_\w+\b")  # scientific notation, snake_case (p95_spike too)
RATE_LIMITED = "Too many jobs from this address: 10 starts per 2 minutes. Wait a minute and retry."
BUSY = "The demo is busy: no server is free right now. Wait a few seconds and retry."


def start_local():
    port = int(os.environ.get("DRISHYAM_E2E_PORT", "0"))
    if not port:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
    env = {**os.environ, "DRISHYAM_SERVE_STATIC": "1", "PYTHONUNBUFFERED": "1"}
    log = open(SHOTS / "server.log", "w")  # noqa: SIM115
    proc = subprocess.Popen([str(ROOT / ".venv/bin/uvicorn"), "drishyam.web:app", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return proc, base
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


class Page:
    def __init__(self, browser, base, **ctx):
        self.errors = []
        self.requests = []
        self.context = browser.new_context(**ctx)
        self.context.add_init_script(
            "window.__csp = []; document.addEventListener('securitypolicyviolation',"
            " e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));")
        self.page = self.context.new_page()
        self.page.on("console", lambda m: m.type == "error" and self.errors.append(m.text))
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("request", lambda r: "/api/" in r.url and self.requests.append(r.url))
        self.missing = []
        self.page.on("response", lambda r: r.status == 404 and self.missing.append(r.url))
        self.page.goto(base + "/")

    def clean(self):
        csp = self.page.evaluate("window.__csp")
        assert not csp, f"CSP violations: {csp}"
        unexpected = [u for u in self.missing if not any(o in u for o in OPTIONAL)]
        assert not unexpected, f"unexpected 404s: {unexpected}"
        errors = [e for e in self.errors if not e.startswith("Failed to load resource") or unexpected]
        assert not errors, f"console errors: {errors}"

    def pick(self, file):
        self.page.locator(f"#gallery .thumb:has(img[src$='{file}'])").first.click()
        expect(self.page.locator(f"#gallery .thumb[data-file='{file}']")).to_have_attribute("aria-pressed", "true", timeout=15000)

    def status(self):
        return self.page.locator("#job-pill").inner_text().strip().lower()

    def wait_done(self, timeout=60, notes=None):
        """Poll the job pill until the run is over; `notes` (a set) collects the NOTE's secondary lines meanwhile."""
        end = time.time() + timeout
        while time.time() < end:
            if notes is not None:
                notes.add(self.page.locator("#narrator-sub").inner_text())
            if self.status() in DONE:
                return self.status()
            self.page.wait_for_timeout(200)
        raise AssertionError(f"job did not finish; pill = {self.status()!r}")

    def recap_in_view(self):
        return self.page.evaluate("""() => { const top = Math.max(document.getElementById('nav').getBoundingClientRect().bottom,
            document.getElementById('taskbar').getBoundingClientRect().bottom);
          return document.getElementById('recap').getBoundingClientRect().top >= top - 1
            && [...document.querySelectorAll('#recap button')].every((b) => b.getBoundingClientRect().bottom <= innerHeight + 1); }""")

    def plain(self):
        """The visible trace, NOTE and finding cards carry no scientific notation and no snake_case: the raw
        record sits in closed 'All numbers' folds, which innerText leaves out."""
        text = self.page.evaluate("[...document.querySelectorAll('#trace, #narrator, #findings')].map((e) => e.innerText).join(' ')")
        assert not JARGON.findall(text), f"jargon on screen: {sorted(set(JARGON.findall(text)))}"
        raw = self.page.locator("#trace details").evaluate_all("(ds) => ds.map((d) => d.textContent).join(' ')")
        assert "recorded: " in raw, "the raw record must stay in the DOM behind 'All numbers'"

    def viewer_in_view(self):
        return self.page.evaluate("""() => { const c = document.getElementById('stage').getBoundingClientRect();
          const bar = document.getElementById('taskbar').getBoundingClientRect();
          return c.top >= bar.bottom - 1 && c.bottom <= innerHeight + 1; }""")


def run(base, server_log=None):
    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        p = Page(browser, base, viewport={"width": 1440, "height": 900})
        page = p.page
        csp = page.request.get(base + "/").headers.get("content-security-policy", "")
        assert "script-src 'self'" in csp, "production CSP missing from the document"
        expect(page.locator("#gallery .thumb")).to_have_count(8, timeout=15000)

        # 1. The guided run: bridge_03 (exposed rebar) is preselected, the run is paced and stays on screen,
        #    then the page hands the critical findings to a person, whose approval the server signs.
        expect(page.locator("#gallery .thumb[data-file='bridge_03.jpg']")).to_have_attribute("aria-pressed", "true", timeout=15000)
        expect(page.locator("#look")).to_be_enabled(timeout=15000)
        page.locator("#guided").click()
        t0 = time.time()
        expect(page.locator("#narrator")).to_contain_text("Sending the photo", timeout=2000)  # the page answers the click at once
        expect(page.locator("#trace .trace__line[data-step='1']")).to_be_visible(timeout=30000)
        assert any("pace_ms=1000" in u for u in p.requests if "/api/jobs" in u), p.requests
        assert p.viewer_in_view(), "the viewer must stay on screen during the guided run"
        expect(page.locator("#narrator")).not_to_be_empty()
        expect(page.locator("#trace .trace__line[data-step='1'] .trace__why")).to_be_visible()
        page.screenshot(path=str(SHOTS / "live-desktop-guided.png"))
        t1 = time.time()
        expect(page.locator("#trace .trace__line[data-step='2']")).to_be_visible(timeout=15000)
        assert time.time() - t1 < 5, f"step 1 stayed {time.time() - t1:.1f} s: the camera should move every 3-4 s"
        notes = set()
        status = p.wait_done(timeout=90, notes=notes)     # the whole guided run at reading speed, no Skip
        took = time.time() - t0
        assert took < 40, f"the guided run took {took:.1f} s"
        expect(page.locator("#job-text")).to_contain_text("we replayed it slowly so you can read it")
        p.plain()
        expect(page.locator("#findings .finding").first).to_be_visible()
        assert page.locator("#trace .trace__line").count() >= 3
        critical = page.locator("#findings .finding--critical").first
        expect(critical).to_be_visible()
        assert status in ("needs sign-off", "halted", "budget spent"), status
        expect(page.locator("#recap")).to_contain_text("What you just saw")
        page.wait_for_timeout(1500)                       # the smooth scroll to the recap
        assert page.evaluate("document.activeElement.id") == "sign-off", "the guided run ends on Sign them off"
        assert p.recap_in_view(), "the recap and its buttons must be on screen at the end of the guided run"
        page.screenshot(path=str(SHOTS / "live-desktop-done.png"))
        page.locator("#sign-off").click()
        assert page.evaluate("document.activeElement.id").startswith("rev-"), "Sign them off goes to a reviewer field"
        critical.locator("input").fill("E2E Engineer")
        critical.get_by_role("button", name="Approve").click()
        expect(critical.locator(".chip--state")).to_have_text("approved", timeout=10000)
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download trace.ndjson").click()
        lines = Path(dl.value.path()).read_text().splitlines()
        job = json.loads(lines[0])
        handoffs = [e for e in map(json.loads, lines) if e["type"] == "review_request" and not e["reason"].endswith(" is critical")]
        assert not handoffs or any("goes to a person" in n for n in notes), f"a review hand-off never reached the NOTE: {handoffs}"
        if "prep_ms" in job:  # the server's read, decode and privacy times: the status line adds them to the agent's
            expect(page.locator("#job-text")).to_contain_text("server total")
            expect(page.locator("#stages [data-stage='photo']")).to_contain_text("decoded in")
        else:
            expect(page.locator("#job-text")).not_to_contain_text("server total")
        print(f"ok  guided live run ({took:.1f} s): paced, plain words, viewer on screen, findings, recap in view, critical card, "
              f"human approval signed by the server; server total {'shown' if 'prep_ms' in job else 'not sent'}")

        # Now break it, from the recap: the injected faults read in plain words too.
        page.get_by_role("button", name="Now break it").click()
        page.locator("#skip").click()
        p.wait_done()
        expect(page.locator("#trace .chip--inject")).to_have_text("injected: slow segmenter")
        p.plain()
        print("ok  Now break it: injected faults and the governor's reaction in plain words")

        # Try KILL from the recap: a run shown live whose planner stalls 2 s at step 3, so KILL lands mid-job.
        page.get_by_role("button", name="Try KILL").click()
        expect(page.locator("#job-text")).to_contain_text("shown the moment it arrives", timeout=15000)
        expect(page.locator("#trace .trace__line[data-step='1']")).to_be_visible(timeout=30000)
        page.locator("#kill").click()
        expect(page.locator("#job-pill")).to_have_text("halted", ignore_case=True, timeout=15000)
        expect(page.locator("#trace .trace__line .chip", has_text="kill-switch")).to_have_count(1)
        print("ok  Try KILL: the live run was halted by the server with kill-switch")

        # 2. KILL mid-run reaches the server, which halts before its next look and sends a kill-switch record.
        # planner_timeout parks the job for 2 s at step 3, so the click lands while it is still running.
        # A fresh page first: otherwise the previous job's trace lines would satisfy the waits at once.
        page.reload()
        expect(page.locator("#gallery .thumb")).to_have_count(8, timeout=15000)
        p.pick("bridge_05.jpg")
        expect(page.locator("#look")).to_be_enabled(timeout=15000)
        page.locator("#advanced summary").click()
        page.locator("label.toggle:has(input[value='planner_timeout'])").click()
        page.locator("#look").click()
        expect(page.locator("#stages li.is-on").first).to_be_visible(timeout=30000)  # the job has started on the server
        page.locator("#kill").click()
        expect(page.locator("#job-pill")).to_have_text("halted", ignore_case=True, timeout=15000)
        expect(page.locator("#trace .trace__line .chip", has_text="kill-switch")).to_have_count(1)
        expect(page.locator("#resume")).to_be_visible()
        if server_log:
            time.sleep(1)
            text = server_log.read_text()
            assert "POST /api/kill" in text, "the KILL button never called /api/kill"
            assert "cause=kill-switch" in text, "the server did not halt the job with kill-switch"
            print("ok  KILL: /api/kill called, the server halted with kill-switch and the trace shows it")
        else:
            print("ok  KILL: the trace shows the server's kill-switch halt (server log not available for a remote base)")

        # 3. 429s, stubbed: a test must not trip the real limit. Lambda's own 429 (every environment busy, no `error`
        #    field) refuses the Resume: the killed run, its trace and Resume stay.
        steps = page.locator("#trace .trace__line[data-step]").count()
        page.route("**/api/jobs*", lambda r: r.fulfill(status=429, content_type="application/json", body=json.dumps({"Message": "Rate Exceeded."})))
        page.locator("#resume").click()
        expect(page.locator("#job-text")).to_have_text(BUSY, timeout=15000)
        expect(page.locator("#job-pill")).to_have_text("error")
        expect(page.locator("#resume")).to_be_visible()
        expect(page.locator("#trace .trace__line[data-step]")).to_have_count(steps)
        expect(page.locator("#narrator")).to_have_text(BUSY)
        page.unroute("**/api/jobs*")
        #    The WAF's rate rule answers 429 with its own body: the per-address limit.
        page.route("**/api/jobs*", lambda r: r.fulfill(status=429, content_type="application/json",
                                                        body=json.dumps({"error": "rate limited: 10 job starts per 2 min per IP; wait and retry"})))
        page.locator("#look").click()
        expect(page.locator("#job-text")).to_have_text(RATE_LIMITED, timeout=15000)
        expect(page.locator("#job-pill")).to_have_text("error")
        expect(page.locator("#narrator")).not_to_contain_text("Sending the photo")
        page.unroute("**/api/jobs*")
        print("ok  429: a refused Resume keeps the killed run; busy and per-address messages in the bar and the NOTE")
        #    The overview (heatmap, pyramid, regions) is shown the moment it arrives, not after the job record's pause.
        page.route("**/api/jobs*", lambda r: r.fulfill(status=200, content_type="application/x-ndjson", body="".join(f"{ln}\n" for ln in lines if json.loads(ln)["type"] in ("stage", "job", "overview"))))
        page.locator("#look").click()
        expect(page.locator("#stages li[data-stage='photo']")).to_have_class(re.compile(r"\bis-on\b"), timeout=15000)  # the job record
        expect(page.locator("#stages li[data-stage='saliency']")).to_have_class(re.compile(r"\bis-on\b"), timeout=500)  # not 900 ms later
        page.unroute("**/api/jobs*")
        print("ok  the overview is lit at once")
        p.clean()
        print("ok  no CSP violations, no console errors")

        # 4. Phone width: no horizontal scroll after a run.
        m = Page(browser, base, viewport={"width": 375, "height": 812}, is_mobile=True, has_touch=True)
        expect(m.page.locator("#gallery .thumb")).to_have_count(8, timeout=15000)
        cta = m.page.locator("#guided").bounding_box()
        assert cta["y"] + cta["height"] <= 812, f"CTA below the fold at 375 px: {cta}"
        m.pick("bridge_01.jpg")
        m.page.locator("#look").click()
        m.page.locator("#skip").click()
        m.wait_done()
        over = m.page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert over <= 0, f"horizontal scroll of {over}px at 375 px"
        m.plain()
        m.page.screenshot(path=str(SHOTS / "live-mobile-done.png"))
        m.clean()
        print("ok  375 px: CTA above the fold, run completes, plain words, no horizontal scroll")
        browser.close()


if __name__ == "__main__":
    SHOTS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        run(sys.argv[1].rstrip("/"))
    else:
        proc, base = start_local()
        try:
            run(base, SHOTS / "server.log")
        finally:
            proc.terminate()
    print("ALL OK")
