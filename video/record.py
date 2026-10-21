"""Record the live site for the submission video: one continuous screencast of real jobs on
https://d1xg58guluup8c.cloudfront.net, plus a log of where the mouse went and when the page changed.

    ~/.claude/brag-scaled-venv/bin/python video/record.py            # -> brag-output/work/rec/

Flow (4 job starts, under the WAF's 10 per IP per 120 s): the guided run on bridge_03 ("Watch it inspect a
bridge"), the human sign-off of each critical finding, "Now break it", "Try KILL" + KILL + Resume, then a
tampered work-order token sent to /api/review. Each run's trace is saved through the page's own "Download
trace.ndjson" button, with the token fields and embedded images removed, to video/live/<run>.ndjson (the source of every live-run
number the video quotes).

Headless Chromium with --force-device-scale-factor=2: the screencast then delivers device pixels (2560 px wide
for a 1280 px layout; with only the context's scale factor it delivers CSS pixels). Frames arrive only when the
page changes; each is saved with its timestamp and scroll offset.
"""
import base64, hashlib, json, pathlib, sys, time

from playwright.sync_api import sync_playwright

BASE = "https://d1xg58guluup8c.cloudfront.net"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "brag-output" / "work" / "rec"
LIVE = ROOT / "video" / "live"
W, H = 1280, 1200
REVIEWER = "Demo reviewer"   # not a person's name: the stamp shows it
DONE = ("needs sign-off", "signed", "done", "halted", "budget spent")
SECRET = ("token", "kill_token")
IMAGES = ("crop_jpeg_b64", "heatmap_png_b64")   # embedded pictures: not needed for any number, and most of the file

T0 = time.time()
EV = []


def now():
    return round(time.time() - T0, 3)


def log(kind, **kw):
    EV.append({"t": now(), "kind": kind, **kw})
    print(f"{EV[-1]['t']:7.2f} {kind} {json.dumps(kw)[:110]}", flush=True)


def strip(obj):
    if isinstance(obj, dict):
        return {k: ("<removed>" if k in SECRET else "<removed: image>" if k in IMAGES else strip(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "frames").mkdir(exist_ok=True); LIVE.mkdir(parents=True, exist_ok=True)
    frames = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--force-device-scale-factor=2", "--hide-scrollbars"])
        ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=2, accept_downloads=True)
        pg = ctx.new_page()
        cdp = ctx.new_cdp_session(pg)

        def on_frame(f):
            m = f["metadata"]
            name = f"frames/{len(frames):05d}.jpg"
            (OUT / name).write_bytes(base64.b64decode(f["data"]))
            frames.append({"t": round(m["timestamp"] - T0, 3), "f": name, "sy": round(m.get("scrollOffsetY", 0), 1)})
            cdp.send("Page.screencastFrameAck", {"sessionId": f["sessionId"]})

        cdp.on("Page.screencastFrame", on_frame)
        mouse = [W * 0.62, H * 0.55]

        def snap():
            return pg.evaluate("""() => ({ pill: (document.querySelector('#job-pill')?.textContent || '').trim().toLowerCase(),
              text: document.querySelector('#job-text')?.textContent || '', note: document.querySelector('#narrator')?.textContent || '',
              sub: document.querySelector('#narrator-sub')?.textContent || '',
              steps: [...document.querySelectorAll('#trace .trace__line[data-step]')].map(e => +e.dataset.step), sy: scrollY })""")

        last = {}

        def watch(seconds, until=None):
            """Poll the page, log every change of the pill, status line, NOTE and trace; stop when until(state) is true."""
            nonlocal last
            end = time.time() + seconds
            while time.time() < end:
                s = snap()
                for k in ("pill", "text", "note", "sub"):
                    if s[k] != last.get(k):
                        log(k, v=s[k])
                new = [n for n in s["steps"] if n not in (last.get("steps") or [])]
                for n in new:
                    log("step", n=n)
                last = s
                if until and until(s):
                    return s
                pg.wait_for_timeout(100)
            if until:
                sys.exit(f"timed out; last state {last}")
            return last

        def box(sel):
            loc = pg.locator(sel).first
            loc.wait_for(state="visible", timeout=30000)
            loc.scroll_into_view_if_needed(timeout=5000)
            pg.wait_for_timeout(250)
            r = loc.bounding_box()
            return loc, r

        def move(sel, label):
            _, r = box(sel)
            x, y = r["x"] + r["width"] / 2, r["y"] + r["height"] / 2
            t = now(); pg.mouse.move(x, y, steps=18)
            log("move", label=label, x0=round(mouse[0]), y0=round(mouse[1]), x=round(x), y=round(y), t0=t,
                sy=pg.evaluate("scrollY"), box=[round(r["x"]), round(r["y"]), round(r["width"]), round(r["height"])])
            mouse[:] = [x, y]

        def click(sel, label):
            move(sel, label)
            pg.wait_for_timeout(250)
            pg.mouse.down(); pg.wait_for_timeout(90); pg.mouse.up()
            log("click", label=label, x=round(mouse[0]), y=round(mouse[1]), sy=pg.evaluate("scrollY"))

        def download(name, button):
            """The page's own download button, clicked from script: no scroll, no mouse, nothing on screen moves."""
            with pg.expect_download() as dl:
                pg.evaluate("t => [...document.querySelectorAll('#done button')].find(b => b.textContent.trim() === t).click()", button)
            return pathlib.Path(dl.value.path()).read_text()

        def save_trace(name):
            lines = [json.dumps(strip(json.loads(ln))) for ln in download(name, "Download trace.ndjson").splitlines() if ln.strip()]
            (LIVE / f"{name}.ndjson").write_text("\n".join(lines) + "\n")
            log("saved", file=f"video/live/{name}.ndjson", lines=len(lines))

        def running_then_done(timeout=150):
            watch(30, lambda s: s["pill"] == "running")
            return watch(timeout, lambda s: s["pill"] in DONE)

        pg.goto(BASE + "/", wait_until="load")
        pg.locator("#gallery .thumb[data-file='bridge_03.jpg'][aria-pressed='true']").wait_for(timeout=30000)
        pg.locator("#look:not([disabled])").wait_for(timeout=30000)
        pg.wait_for_timeout(1200)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 86})
        log("start")
        pg.mouse.move(*mouse)
        watch(2.0)

        # 1. the guided run
        click("#guided", "Watch it inspect a bridge")
        s = running_then_done()
        log("done", run="guided", pill=s["pill"])
        watch(2.5)                                   # the page scrolls the recap into view
        save_trace("guided")
        watch(1.0)

        # 2. the human sign-off: every critical finding still waiting
        click("#sign-off", "Sign them off")
        watch(1.2)
        crit = pg.locator("#findings .finding--critical")
        for n in range(crit.count()):
            card = crit.nth(n)
            if card.locator(".review input").count() == 0:
                continue
            inp = card.locator(".review input").first
            inp_id = inp.get_attribute("id")
            click(f"#{inp_id}", "Reviewer")
            t = now(); pg.keyboard.type(REVIEWER, delay=65); log("type", text=REVIEWER, t0=t)
            approve = card.get_by_role("button", name="Approve")
            approve.scroll_into_view_if_needed(); r = approve.bounding_box()
            x, y = r["x"] + r["width"] / 2, r["y"] + r["height"] / 2
            t = now(); pg.mouse.move(x, y, steps=14)
            log("move", label="Approve", x0=round(mouse[0]), y0=round(mouse[1]), x=round(x), y=round(y), t0=t, sy=pg.evaluate("scrollY"),
                box=[round(r["x"]), round(r["y"]), round(r["width"]), round(r["height"])])
            mouse[:] = [x, y]
            pg.wait_for_timeout(250); pg.mouse.down(); pg.wait_for_timeout(90); pg.mouse.up()
            log("click", label="Approve", x=round(x), y=round(y), sy=pg.evaluate("scrollY"))
            card.locator(".chip--state", has_text="approved").wait_for(timeout=15000)
            log("approved", n=n)
            watch(1.5)
        watch(2.0)
        order = json.loads(download("order", "Download work_order.json"))
        (OUT / "order_status.json").write_text(json.dumps({"status": order["work_order"]["status"],
                                                           "approvals": strip(order["work_order"].get("approvals", []))}, indent=1))
        token = order["token"]

        # 3. Now break it (from the recap)
        click("#recap button:has-text('Now break it')", "Now break it")
        s = running_then_done()
        log("done", run="break", pill=s["pill"])
        watch(2.5)
        save_trace("break")
        watch(1.0)

        # 4. Try KILL: KILL while the planner stalls at step 3, then Resume
        click("#recap button:has-text('Try KILL')", "Try KILL")
        watch(40, lambda s: s["pill"] == "running" and len(s["steps"]) >= 2)
        watch(0.4)
        click("#kill", "KILL")
        s = watch(30, lambda s: s["pill"] in DONE + ("killed",))
        log("done", run="kill", pill=s["pill"])
        watch(2.5)
        save_trace("kill")
        click("#resume", "Resume")
        s = running_then_done()
        log("done", run="resume", pill=s["pill"])
        watch(3.0)
        save_trace("resume")
        cdp.send("Page.stopScreencast")
        log("stop")

        # 5. A work-order token with one character of its signature changed, sent to /api/review
        head, sig = token.rsplit(".", 1)
        bad = head + "." + (("A" if sig[0] != "A" else "B") + sig[1:])
        body = json.dumps({"token": bad, "finding_id": order["work_order"]["findings"][0]["id"], "decision": "approve", "reviewer": REVIEWER})
        res = pg.evaluate("""async ([body, sha]) => { const r = await fetch('/api/review', { method: 'POST', body,
            headers: { 'Content-Type': 'application/json', 'x-amz-content-sha256': sha } });
            return { status: r.status, body: await r.text() }; }""", [body, hashlib.sha256(body.encode()).hexdigest()])
        (LIVE / "tamper.json").write_text(json.dumps({"request": "POST /api/review with the work order's token, one character of its HMAC signature changed",
                                                      "status": res["status"], "response": json.loads(res["body"]) if res["body"].startswith("{") else res["body"]}, indent=1) + "\n")
        log("tamper", status=res["status"], body=res["body"][:120])
        b.close()
    (OUT / "frames.json").write_text(json.dumps(frames))
    (OUT / "events.json").write_text(json.dumps({"viewport": [W, H], "dpr": 2, "events": EV}, indent=0))
    print(f"{len(frames)} frames, {len(EV)} events -> {OUT}")


def health():
    """/api/health as the browser shows it (walk.py records the picture): the response, and where its key values sit."""
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": 720}, device_scale_factor=2)
        pg.goto(BASE + "/api/health", wait_until="load"); pg.wait_for_timeout(1500)
        text = pg.evaluate("document.querySelector('pre')?.textContent || document.body.innerText")
        boxes = pg.evaluate("""keys => { const pre = document.querySelector('pre'), node = pre.firstChild, out = {};
          for (const k of keys) { const i = node.textContent.indexOf(k); if (i < 0) continue; const r = document.createRange();
            r.setStart(node, i); r.setEnd(node, i + k.length); const rs = [...r.getClientRects()].map(q => [q.x, q.y, q.width, q.height].map(Math.round));
            out[k] = rs; } return out; }""", ['"opencv":"5.0.0"', '"arch":"aarch64"', '"host":"aws-lambda"', '"build":"stock (not COOL)"'])
        b.close()
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "health.json").write_text(text.strip() + "\n")
    (OUT / "health_boxes.json").write_text(json.dumps(boxes, indent=1))
    print(text, boxes)


if __name__ == "__main__":
    health() if "--health" in sys.argv else main()
