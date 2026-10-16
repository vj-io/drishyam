"""End-to-end check of the real stack: the page, the FastAPI app and the agent, nothing mocked.

Run: ~/.claude/brag-scaled-venv/bin/python tests/e2e/test_ui_live.py [BASE_URL]
Without BASE_URL it starts the app locally (uvicorn, production headers from vercel.json).
With BASE_URL (e.g. the Vercel deployment) it runs the same read-only flows against it.
Waits use locators and polling, never wait_for_function (blocked under the strict CSP).
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SHOTS = Path(os.environ.get("DRISHYAM_SHOTS", "/tmp/drishyam-e2e"))
DONE = ("awaiting review", "completed", "halted")


def start_local():
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
        self.context = browser.new_context(**ctx)
        self.context.add_init_script(
            "window.__csp = []; document.addEventListener('securitypolicyviolation',"
            " e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));")
        self.page = self.context.new_page()
        self.page.on("console", lambda m: m.type == "error" and self.errors.append(m.text))
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.missing = []
        self.page.on("response", lambda r: r.status == 404 and self.missing.append(r.url))
        self.page.goto(base + "/")

    def clean(self):
        csp = self.page.evaluate("window.__csp")
        assert not csp, f"CSP violations: {csp}"
        # Benchmark files are optional (the page shows a pending state); any other 404 is a bug.
        unexpected = [u for u in self.missing if "/results/" not in u]
        assert not unexpected, f"unexpected 404s: {unexpected}"
        errors = [e for e in self.errors if not e.startswith("Failed to load resource") or unexpected]
        assert not errors, f"console errors: {errors}"

    def pick(self, file):
        self.page.locator(f"#gallery .thumb:has(img[src$='{file}'])").first.click()

    def status(self):
        return self.page.locator("#job-pill").inner_text().strip().lower()

    def wait_done(self, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if self.status() in DONE:
                return self.status()
            time.sleep(0.2)
        raise AssertionError(f"job did not finish; pill = {self.status()!r}")


def run(base, server_log=None):
    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        p = Page(browser, base, viewport={"width": 1440, "height": 900})
        page = p.page
        csp = page.request.get(base + "/").headers.get("content-security-policy", "")
        assert "script-src 'self'" in csp, "production CSP missing from the document"
        expect(page.locator("#gallery .thumb")).to_have_count(8, timeout=15000)

        # 1. A full run on the exposed-rebar sample: findings, a critical card, a human approval.
        p.pick("bridge_03.jpg")
        page.locator("#look").click()
        status = p.wait_done()
        expect(page.locator("#findings .finding").first).to_be_visible()
        assert page.locator("#trace .trace__line").count() >= 3
        critical = page.locator("#findings .finding--critical").first
        expect(critical).to_be_visible()
        assert status in ("awaiting review", "halted")
        critical.locator("input").fill("E2E Engineer")
        critical.get_by_role("button", name="Approve").click()
        expect(critical.locator(".chip--state")).to_have_text("approved", timeout=10000)
        page.screenshot(path=str(SHOTS / "live-desktop-done.png"))
        print("ok  live run: findings, critical card, human approval signed by the server")

        # 2. KILL mid-run reaches the server: its log records the halt with cause kill-switch.
        # planner_timeout parks the job for 2 s at step 3, so the click lands while it is still running.
        p.pick("bridge_05.jpg")
        page.locator("label.toggle:has(input[value='planner_timeout'])").click()
        page.locator("#look").click()
        expect(page.locator("#trace .trace__line[data-step='2']")).to_be_visible(timeout=30000)
        page.locator("#kill").click()
        expect(page.locator("#job-pill")).to_have_text("killed", ignore_case=True, timeout=5000)
        if server_log:
            # Locally uvicorn drops the stream on disconnect, so the job is abandoned mid-step and
            # never completes; on Vercel (which hides the disconnect) the /api/kill flag halts it.
            time.sleep(6)
            text = server_log.read_text()
            assert "POST /api/kill" in text, "the KILL button never called /api/kill"
            completed = [ln for ln in text.splitlines() if ln.startswith("INFO drishyam job ")]
            assert len(completed) == 1 or "cause=kill-switch" in text, f"killed job ran on: {completed}"
            print("ok  KILL: UI stopped, /api/kill called, the server did not finish the job")
        else:
            print("ok  KILL: UI stopped (server log not available for a remote base)")
        p.clean()
        print("ok  no CSP violations, no console errors")

        # 3. Phone width: no horizontal scroll after a run.
        m = Page(browser, base, viewport={"width": 375, "height": 812}, is_mobile=True, has_touch=True)
        expect(m.page.locator("#gallery .thumb")).to_have_count(8, timeout=15000)
        m.pick("bridge_01.jpg")
        m.page.locator("#look").click()
        m.wait_done()
        over = m.page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert over <= 0, f"horizontal scroll of {over}px at 375 px"
        m.page.screenshot(path=str(SHOTS / "live-mobile-done.png"))
        m.clean()
        print("ok  375 px: run completes, no horizontal scroll")
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
