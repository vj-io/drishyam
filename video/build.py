"""Build the narrated submission video from the result files: one command, numbers read at build time.

    ~/.claude/brag-scaled-venv/bin/python video/build.py            # data -> voice -> layout -> audit -> render
    ~/.claude/brag-scaled-venv/bin/python video/build.py --stills   # data, voice and layout, then stills only
    ~/.claude/brag-scaled-venv/bin/python video/build.py --audit    # ... then fastrender --audit only

Reads every number from the files named in SOURCES below (and records which field it came from), writes the
narration (video/narration.py) through the brag-scaled narrate skill (Kokoro, offline), fits the chapters to
the voice, writes brag-output/work/film-data.js for video/film.html, the captions (public/video/drishyam.vtt),
the generated half of video/plan.md, a code-made score, and renders with the fast-render skill into
brag-output/work/, then moves the MP4, the captions and the poster into public/video/ together, so the three
always come from one build (a stopped render leaves public/video/ as it was), plus the 9:16 cut
brag-output/drishyam-short.mp4.

The live footage is recorded once by video/record.py (brag-output/work/rec/, git-ignored) and walk.py
(brag-output/work/walkrec/); re-recording is a separate step because the live site's WAF allows 10 job starts
per IP per 120 s. The trained segmenter is "results pending" until public/results/codebrim_dacl10k/summary.json
exists; then the line states its recall against the classical engine charged the same cost table, with the
governor off, where tiling still finds more, its chance recall and the leakage count, and the build stops if
any of those checks fails. `--check-template` tries that line on a stand-in summary without touching the film.
"""
import argparse, bisect, glob, gzip, json, math, os, pathlib, re, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIDEO = ROOT / "video"
WORK = ROOT / "brag-output" / "work"
REC = WORK / "rec"
PUB = ROOT / "public" / "video"
SKILLS = pathlib.Path(sorted(glob.glob(str(pathlib.Path.home() / ".claude/plugins/cache/brag-scaled/brag-scaled/*/skills")))[-1])
FAST = SKILLS / "fast-render/scripts/fastrender.py"
NARRATE = SKILLS / "narrate/scripts/narrate.py"
SOUNDKIT = SKILLS / "motion-kit/scripts"
PY = sys.executable
VOICE = "af_heart"
LIVE_URL = "https://d1xg58guluup8c.cloudfront.net"
REPO_URL = "https://github.com/vj-io/drishyam"
MODEL = ROOT / "public/results/codebrim_dacl10k/summary.json"   # the trained segmenter; the film quotes it once it exists

SRC = []   # (what, value, file, field) for every number and URL the video shows or says


def src(what, value, file, field):
    SRC.append((what, value, file, field))
    return value


def load(rel):
    return json.loads((ROOT / rel).read_text())


def main_checkout():
    """runs/ is git-ignored, so the smoke log lives in the main checkout; find it through git's common dir."""
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (ROOT / common).resolve().parent if common else ROOT


def pts(lst):
    return sorted([p for p in lst if p.get("recall") is not None], key=lambda p: p["pixels_frac"])


def head(policies):
    """The site's headline point (public/js/results.js headline()): Drishyam's last point within 30 % of the pixels."""
    return [p for p in pts(policies["drishyam"]) if p["pixels_frac"] <= 0.3][-1]


def ndjson(rel):
    return [json.loads(l) for l in (ROOT / rel).read_text().splitlines() if l.strip()]


def dims(dataset):
    """'synthetic-200 at 2048x1536 (labelled synthetic)' -> [2048, 1536]"""
    m = re.search(r"at (\d+)x(\d+)", dataset)
    return [int(m[1]), int(m[2])]


def inside(a, b):
    """box a [x, y, w, h] lies inside box b"""
    return a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


# ---------------------------------------------------------------------------------------------- data
def arm_of(summary):
    """'results/linux-aarch64/micro.json job_ops ...' -> 'linux-aarch64' (an ONNX run's '-dacl10k' table is the same machine)"""
    return re.match(r"results/([^/ ]+?)(?:-dacl10k)?/", summary["cost_model"]["source"])[1]


def half(n):
    """cv2.pyrDown's output side: half, rounded up (data() checks it against the guided job's recorded levels)"""
    return (n + 1) // 2


def in_text(rel, *phrases):
    """Each phrase is in the file (case and whitespace folded): scene text typed in film.js is checked against its source."""
    t = re.sub(r"[\s*]+", " ", (ROOT / rel).read_text()).lower()
    for p in phrases:
        assert re.sub(r"\s+", " ", p).lower() in t, f"{rel} no longer says {p!r}"
    return phrases[0] if len(phrases) == 1 else list(phrases)


def data(model=MODEL):
    D = {}
    f = "public/results/summary.json"; s = load(f); hs = s["headline_stats"]; h = head(s["policies"])
    assert hs["budget"] == 0.3, "head() and the narration say a 30 % cap"
    arms = {r: arm_of(load(r)) for r in ("public/results/summary.json", "public/results/large/summary.json", "public/results/codebrim/summary.json", "public/results/clean/summary.json")}
    assert set(arms.values()) == {"linux-aarch64"}, f"the results rail says every set is charged the arm64 costs: {arms}"
    assert all(not load(r)["agent_config"]["governor"] for r in arms), "the results compare policies with the governor off"
    assert len(pts(s["policies"]["tiling"])) == 3, "the chart legend says tiling at L2, L1, L0"
    D["syn"] = {
        "n": src("synthetic photos", s["n_images"], f, "n_images"),
        "size": src("synthetic photo size (px)", dims(s["dataset"]), f, "dataset"),
        "cap": src("pixel cap of the headline point", hs["budget"], f, "headline_stats.budget"),
        "head": h, "tiling": pts(s["policies"]["tiling"])[-1], "ds": pts(s["policies"]["downscale"])[0],
        "ratio": src("recall vs exhaustive tiling", hs["recall_ratio_vs_tiling"], f, "headline_stats.recall_ratio_vs_tiling"),
        "ratio_ci": src("its 95 % CI", hs["recall_ratio_vs_tiling_ci"], f, "headline_stats.recall_ratio_vs_tiling_ci"),
        "target": src("target recall ratio", hs["target"]["recall_ratio"], f, "headline_stats.target.recall_ratio"),
        "meets_target": src("target met?", hs["meets_target"], f, "headline_stats.meets_target"),
        "ds_ratio": src("recall vs one downscale", hs["recall_ratio_vs_downscale"], f, "headline_stats.recall_ratio_vs_downscale"),
        "ds_ci": src("its 95 % CI", hs["recall_ratio_vs_downscale_ci"], f, "headline_stats.recall_ratio_vs_downscale_ci"),
        "policies": {k: pts(v) for k, v in s["policies"].items()},
    }
    src("headline point: pixels, recall", [h["pixels_frac"], h["recall"]], f, "policies.drishyam[pixels_frac <= 0.3][-1] (as public/js/results.js headline())")
    src("exhaustive tiling: pixels, recall", [D["syn"]["tiling"]["pixels_frac"], D["syn"]["tiling"]["recall"]], f, "policies.tiling[max pixels_frac]")
    src("one downscale: pixels, recall", [D["syn"]["ds"]["pixels_frac"], D["syn"]["ds"]["recall"]], f, "policies.downscale[0]")
    src("chart: every point and its chance tick", "all rows", f, "policies.*[].pixels_frac, recall, recall_ci, chance_recall")
    assert max(p["recall_ci"][1] if p.get("recall_ci") else p["recall"] for v in D["syn"]["policies"].values() for p in v) < 0.7, "the chart's y axis stops at 0.7"
    src("cost table every result set is charged", "arm64 (linux-aarch64)", "public/results/{,large/,codebrim/,clean/}summary.json", "cost_model.source")

    f = "public/results/large/summary.json"; s = load(f); hs = s["headline_stats"]
    D["large"] = {"n": src("4608 px photos", s["n_images"], f, "n_images"), "head": head(s["policies"]),
                  "size": src("4608 px photo size (px)", dims(s["dataset"]), f, "dataset"),
                  "l2": pts(s["policies"]["tiling"])[0], "ds": pts(s["policies"]["downscale"])[0],
                  "ds_ratio": src("4608 px: recall vs one downscale", hs["recall_ratio_vs_downscale"], f, "headline_stats.recall_ratio_vs_downscale"),
                  "ds_ci": src("its 95 % CI", hs["recall_ratio_vs_downscale_ci"], f, "headline_stats.recall_ratio_vs_downscale_ci")}
    w = D["large"]["size"][0]; assert w % half(half(w)) == 0
    D["large"]["l2_div"] = src("4608 px: full width / L2 width (the tiling level called quarter-size)", w // half(half(w)), f, "dataset size, halved twice by pyrDown's rule (job.levels in video/live/guided.ndjson)")
    src("4608 px headline point: pixels, recall", [D["large"]["head"]["pixels_frac"], D["large"]["head"]["recall"]], f, "policies.drishyam[pixels_frac <= 0.3][-1]")
    src("4608 px L2 tiling: pixels, recall", [D["large"]["l2"]["pixels_frac"], D["large"]["l2"]["recall"]], f, "policies.tiling[min pixels_frac] (L2)")
    src("4608 px one downscale: pixels, recall", [D["large"]["ds"]["pixels_frac"], D["large"]["ds"]["recall"]], f, "policies.downscale[0]")

    f = "public/results/codebrim/summary.json"; s = load(f); rd = head(s["policies"])
    others = [p["recall"] for k in ("downscale", "tiling") for p in pts(s["policies"][k])]
    lo, hi = min([rd["recall"], *others]), max([rd["recall"], *others])
    D["codebrim"] = {"n": src("CODEBRIM photos", s["n_images"], f, "n_images"),
                     "recall": src("CODEBRIM recall (ours)", rd["recall"], f, "policies.drishyam[pixels_frac <= 0.3][-1].recall"),
                     "range": src("CODEBRIM recall, every method (min, max)", [lo, hi], f, "policies.{downscale,tiling,drishyam head}.recall"),
                     "plain": hi - lo <= 0.05, "one_in": round(1 / ((lo + hi) / 2)),
                     "chance": src("CODEBRIM: its boxes at random", rd["chance_recall"], f, "policies.drishyam[head].chance_recall")}
    src("'about 1 in N' (N = round(1 / mid of the range), as the site's hero card)", D["codebrim"]["one_in"], f, "derived from the range above")

    f = "public/results/clean/summary.json"; s = load(f); fa = s["false_alarms"]
    assert "synthetic" in s["dataset"], "the narration calls the clean set synthetic"
    D["clean"] = {"n": src("synthetic photos with no damage", s["n_images"], f, "n_images (dataset: 'synthetic clean set')"),
                  "any": src("share of them with any report (ours, 30 % cap)", fa["drishyam@0.3"]["images_with_any"], f, "false_alarms['drishyam@0.3'].images_with_any"),
                  "boxes": src("boxes per image (ours)", fa["drishyam@0.3"]["boxes_per_image"], f, "false_alarms['drishyam@0.3'].boxes_per_image"),
                  "crit": src("critical per image (ours)", fa["drishyam@0.3"]["critical_per_image"], f, "false_alarms['drishyam@0.3'].critical_per_image"),
                  "ds_boxes": src("boxes per image (one downscale)", fa["downscale"]["boxes_per_image"], f, "false_alarms.downscale.boxes_per_image"),
                  "tl_boxes": src("boxes per image (L0 tiling)", fa["tiling@L0"]["boxes_per_image"], f, "false_alarms['tiling@L0'].boxes_per_image"),
                  "cap": src("pixel cap of those numbers", float("drishyam@0.3".split("@")[1]), f, "false_alarms key 'drishyam@0.3'")}

    f = "public/results/ablations.json"; a = load(f); D["abl"] = {}
    names = {"synth200": "2048 px synthetic", "synth_large": "4608 px synthetic", "codebrim": "CODEBRIM"}
    variants = ["no-survey", "no-verify", "no-uncertainty", "governor-on", "random-order-s0", "random-order-s1", "random-order-s2"]
    for ds, v in a["datasets"].items():
        row = v["by_cap"]["0.3"]
        D["abl"][ds] = {k: src(f"ablation {names.get(ds, ds)}: {k} / full", row[k]["ratio_vs_full"], f, f"datasets.{ds}.by_cap['0.3']['{k}'].ratio_vs_full") for k in variants if k in row}
        D["abl"][ds]["vs"] = {k: src(f"ablation {names.get(ds, ds)}: {k} verdict", row[k]["vs_full"], f, f"datasets.{ds}.by_cap['0.3']['{k}'].vs_full")
                              for k in ("no-survey", "no-verify", "no-uncertainty", "governor-on")}
        m = re.search(r"seeds pooled per image: ([\d.]+)-([\d.]+)", v["random_order"][0])
        D["abl"][ds]["random_lo"], D["abl"][ds]["random_hi"] = float(m[1]), float(m[2])
        src(f"ablation {names.get(ds, ds)}: random order / full, pooled 95 % CI", [float(m[1]), float(m[2])], f, f"datasets.{ds}.random_order[0]")
        D["abl"][ds]["label"] = names.get(ds, ds)

    f = "public/results/failures.json"; fails = load(f)
    want = {"Graffiti read as damage": None, "Formwork line read as crack": None, "Glare caught by the quality gate": None, "Crack hidden in blur": None}
    for x in fails:
        if x["category"] in want and want[x["category"]] is None:
            want[x["category"]] = x
    assert all(want.values()), f"failures.json no longer has every shown category: {want}"
    assert all(x["rule_that_would_catch_it"].rstrip().endswith("Not implemented.") for x in want.values()), "a shown rule is now implemented"
    def short_rule(r):   # the rule's name: the text before the first colon, in the first sentence that has one
        sent = next((x for x in re.split(r"(?<=\.)\s+", r) if ":" in x), r)
        name, rest = sent.split(":", 1)
        if name.startswith("The missing rule"):   # 'The missing rule is bookkeeping: a refused window is written ... with a re-shoot suggestion, ...'
            return rest.split(",")[0].split(" with ")[0].strip()
        return name.strip()
    D["fails"] = [{"img": "../public/results/" + x["image"], "cat": x["category"], "id": x["id"], "ruleShort": short_rule(x["rule_that_would_catch_it"])} for x in want.values()]
    for x in want.values():
        src(f"failure plate {x['id']}", x["category"], f, f"[image={x['image']}].category, rule_that_would_catch_it")
    f = "public/results/codebrim/failures_text.json"; ft = load(f)
    rf = next(x for x in ft if x["category"] == "Rebar, spall and stain confusion" and "reported corrosion" in x["what_happened"])
    m = re.search(r"ground truth is (\w+), the job reported (\w+)", rf["what_happened"])
    D["real_fail"] = {"id": rf["id"], "gt": m[1].replace("_", " "), "got": m[2].replace("_", " ")}
    src("real-photo failure (text only)", f"{rf['id']}: {m[1]} reported as {m[2]}", f, "[category='Rebar, spall and stain confusion'].what_happened")

    # the live runs recorded by video/record.py (token fields removed)
    f = "video/live/guided.ndjson"; ev = ndjson(f)
    done = next(e for e in ev if e["type"] == "done"); steps = [e for e in ev if e["type"] == "step"]
    job = next(e for e in ev if e["type"] == "job")
    rr = next(e for e in ev if e["type"] == "review_request" and e.get("iou") is not None and e.get("level_b") == 0)
    zoom = next(e for e in steps if e["cause"] == "entropy>tau")
    tau = float(re.search(r"> ([\d.]+) at L", zoom["rationale"])[1])
    iou_min = float(re.search(r"< ([\d.]+) between L1 and L0", rr["reason"])[1])
    # the finding the verify left unconfirmed: exposed rebar, handed to a person by rule at its first look, before any verify
    fin = {e["finding"]["id"]: e["finding"] for e in ev if e["type"] == "finding"}   # last state of each finding
    f1 = fin[rr["finding_id"]]
    first_rr = next(i for i, e in enumerate(ev) if e["type"] == "review_request" and e["finding_id"] == f1["id"])
    assert f1["cls"] == "exposed_rebar" and f1["critical_rule"].startswith("exposed_rebar: always critical") and f1["verified"] is False, f1
    assert first_rr < ev.index(rr), "the rebar's review request came before the verify"
    # what the stop left out: the unvisited candidates, and any ground-truth box inside them that no finding touches
    cand = {c["id"]: c["box"] for e in ev if e["type"] == "overview" for c in e["candidates"]}
    gt = next(x for x in load("public/samples/index.json") if x["file"] == "bridge_03.jpg")
    assert gt["synthetic"] and gt["width"] == job["width"], gt
    missed = [b for b in gt["boxes"] if any(inside(b["box"], cand[r]) for r in done["unvisited"])
              and not any(overlaps(b["box"], x["box"]) for x in fin.values())]
    assert [b["cls"] for b in missed] == ["crack"], missed
    lv = job["levels"]
    assert all(lv[k + 1] == [half(x) for x in lv[k]] for k in range(len(lv) - 1)), f"each pyramid level halves the last, rounding up: {lv}"
    assert lv[0][0] % lv[3][0] == 0 and steps[0]["cause"] == "survey" and steps[0]["rationale"].startswith("one L3 look"), steps[0]
    D["guided"] = {"pixels_frac": src("guided run: share of pixels read", done["summary"]["pixels_frac"], f, "done.summary.pixels_frac"),
                   "critical": src("guided run: critical findings", done["summary"]["critical_pending"], f, "done.summary.critical_pending"),
                   "seconds": src("guided run: agent seconds", done["summary"]["seconds"], f, "done.summary.seconds"),
                   "survey_div": src("survey look: full width / L3 width", lv[0][0] // lv[3][0], f, "job.levels[0][0] / job.levels[3][0]"),
                   "privacy": src("privacy pass: faces, people, vehicles blurred", [job["privacy"][k] for k in ("faces_blurred", "people_blurred", "vehicles_blurred")], f, "job.privacy"),
                   "entropy1": src("step 2 entropy (the zoom decision)", steps[0]["observation"]["entropy"], f, f"step[1].observation.entropy; step[2] rationale '{zoom['rationale']}'"),
                   "tau": src("entropy threshold", tau, f, "step[cause=entropy>tau].rationale"),
                   "iou": src("step 4 verify: mask IoU L1 vs L0", rr["iou"], f, "review_request[level_a=1, level_b=0].iou"),
                   "iou_min": src("verify threshold", iou_min, f, f"review_request reason '{rr['reason']}'"),
                   "rebar_rule": src("why the rebar goes to a person", f1["critical_rule"], f, f"finding {f1['id']}.critical_rule; its first review_request precedes the verify"),
                   "unvisited": src("unvisited candidates at the stop", done["unvisited"], f, "done.unvisited"),
                   "missed": src("ground truth inside them that no finding touches", [b["cls"] for b in missed], "public/samples/index.json", "bridge_03 boxes inside overview.candidates[unvisited]"),
                   "job": done.get("job") or ev[0]["job"]}
    f = "video/live/break.ndjson"; ev = ndjson(f)
    pl = next(e for e in ev if e["type"] == "planner" and e["status"] == "rejected")
    gov = next(e for e in ev if e["type"] == "step" and e.get("model") == "fast")
    D["break"] = {"planner_bad": src("planner drill: rejected strategy", re.search(r"strategy '([^']+)'", pl["reason"])[1], f, "planner[status=rejected].reason"),
                  "p95": src("governor drill: segmenter p95 (ms)", gov["runtime"]["p95_ms"]["segment:accurate"], f, f"step[{gov['step']}].runtime.p95_ms['segment:accurate']"),
                  "model": src("governor drill: model switched to", gov["model"], f, f"step[{gov['step']}].model"), "step": gov["step"]}
    f = "video/live/kill.ndjson"; ev = ndjson(f); done = next(e for e in ev if e["type"] == "done")
    D["kill"] = {"cause": src("KILL drill: halt cause", done["cause"], f, "done.cause"), "steps": done["summary"]["steps"]}
    f = "video/live/resume.ndjson"; ev = ndjson(f)
    D["resume"] = {"replayed": src("Resume: steps marked replay", sum(1 for e in ev if e["type"] == "step" and e["cause"] == "replay"), f, "step[cause=replay] count")}
    f = "video/live/tamper.json"; t = load(f)
    D["tamper"] = {"status": src("tampered token: HTTP status", t["status"], f, "status"), "error": src("tampered token: error", t["response"]["error"], f, "response.error")}
    f = "video/live/health.json"; h = load(f)
    D["health"] = {k: src(f"/api/health {k}", h[k], f, k) for k in ("opencv", "arch", "host", "build", "commit", "segmenter")}

    # MCP session
    f = "docs/mcp-session/exchange.jsonl"; ex = ndjson(f)
    ver = [e for e in ex if e["tool"] == "verify"]
    rs = next(e for e in ex if e["tool"] == "runtime_stats")
    D["mcp"] = {"call_a": src("MCP: first verify call", ver[0]["n"], f, "[tool=verify][0].n"),
                "iou_a": src("MCP: its IoU", ver[0]["content"][0]["value"]["iou"], f, "[tool=verify][0].content[0].value.iou"),
                "call_b": src("MCP: second verify call", ver[1]["n"], f, "[tool=verify][1].n"),
                "iou_b": src("MCP: its IoU", ver[1]["content"][0]["value"]["iou"], f, "[tool=verify][1].content[0].value.iou"),
                "call_c": src("MCP: the budget check", rs["n"], f, "[tool=runtime_stats].n"),
                "job": src("MCP: job id", ex[0]["content"][0]["value"]["job"], f, "[tool=open_image].content[0].value.job")}
    hr = next(e for e in ex if e["tool"] == "request_human_review" and e["n"] > ver[0]["n"])
    assert "always critical" in hr["args"]["reason"] and ver[0]["content"][0]["value"]["agrees"] is False, hr
    D["mcp"]["rows"] = mcp_rows()
    D["mcp"]["crops"] = {}
    for crop, lab in (("crops/img-004-1.jpg", "three rust bars"), ("crops/img-016-1.jpg", "a spall")):   # the two gaze crops the chapter shows, by the call that returned them
        e = next(e for e in ex if any(c.get("file") == crop for c in e["content"]))
        why = next(r["why"] for r in D["mcp"]["rows"] if r["n"] == e["n"])
        assert lab in why.lower(), f"transcript.md call {e['n']} no longer says {lab!r}: {why}"
        D["mcp"]["crops"][crop] = src(f"MCP: call that returned {crop}, its level, its label", [e["n"], e["args"]["level"], lab], f"{f}; docs/mcp-session/transcript.md",
                                      "[content.file=crop].n, .args.level; the label is in that call's 'why' cell")
    tr = (ROOT / "docs/mcp-session/transcript.md").read_text()
    D["mcp"]["missed_box"] = src("MCP: the large spall it missed (box)", re.search(r"\(box (\d+,\d+,\d+,\d+)\)", tr)[1], "docs/mcp-session/transcript.md", "'(box x,y,w,h)' after the call table")
    src("MCP call table: call, what came back, why", f"{len(D['mcp']['rows'])} rows", "docs/mcp-session/transcript.md", "'How each measurement chose the next call' table")

    # AWS facts
    f = "public/docs/aws-simple.svg"; svg = (ROOT / f).read_text()
    m = re.search(r"WAF rate limit \((\d+) job starts per (\d+) s per IP\)", svg)
    D["aws"] = {"waf_n": src("WAF: job starts", int(m[1]), f, "<desc> 'WAF rate limit (10 job starts per 120 s per IP)'"), "waf_s": int(m[2])}
    smoke = main_checkout() / "runs/aws-smoke.log"
    lat = [re.search(r"p95 ([\d.]+) ms per step over (\d+) steps", l) for l in smoke.read_text().splitlines() if l.startswith("INFO latency: p95")]
    runs = [l.split()[1] for l in smoke.read_text().splitlines() if l.startswith("smoke ")]
    p95 = sorted(float(m[1]) for m in lat)
    D["aws"]["p95"] = src("Lambda step p95 (ms), every smoke run (min, max)", [p95[0], p95[-1]], "runs/aws-smoke.log (main checkout)", f"'INFO latency' lines, runs {', '.join(runs)}")
    D["aws"]["smoke_runs"] = src("smoke runs", len(lat), "runs/aws-smoke.log (main checkout)", "count of 'INFO latency' lines")
    D["aws"]["smoke_steps"] = src("steps per smoke run", sorted({int(m[2]) for m in lat}), "runs/aws-smoke.log (main checkout)", "'over N steps'")
    assert "HMAC-SHA256 signed" in (ROOT / "docs/security.md").read_text(), "the tamper inset says HMAC-SHA256"
    D["aws"]["hmac"] = src("work-order signature", "HMAC-SHA256", "docs/security.md", "'Tampered or replayed work order' row")
    f = "results/linux-aarch64/micro.json"; mi = load(f)
    D["aws"]["hal"] = src("stock aarch64 wheel custom HAL (GitHub arm64 runner, not Lambda)", mi["provenance"]["cv2_custom_hal"], f, "provenance.cv2_custom_hal")

    # problem figures
    f = "docs/impact-sources.md"; md = (ROOT / f).read_text()
    m = re.search(r"All bridges ([\d,]+); good [\d,]+; fair [\d,]+; poor ([\d,]+)", md)
    D["impact"] = {"bridges": src("US bridges (FHWA NBI 2026)", m[1], f, "section 1 Figure"), "poor": src("in poor condition", m[2], f, "section 1 Figure"),
                   "months": src("routine inspection interval, months (23 CFR 650.311)", re.search(r"not to exceed (\d+) months", md)[1], f, "section 2 Quote"),
                   "photos": src("photos from the largest drone inspection in the study (ODOT SPR 787)", re.search(r"([\d,]+) high-resolution still photos", md)[1], f, "section 5 Figure")}
    m = re.search(r"Other bridges in the same table: (\d+) to (\d+) photos.*?average over the (\d+) bridges flown", md, re.S)
    D["impact"]["others"] = src("photos from the other bridges in that table (min, max)", [int(m[1]), int(m[2])], f, "section 5 Figure")
    D["impact"]["n_bridges"] = src("bridges flown in the study", int(m[3]), f, "section 5 Figure")
    assert int(D["impact"]["photos"].replace(",", "")) > int(m[2]), "the narration calls this inspection the largest"

    # the trained segmenter: quoted once evals/model_eval.py --publish has written its summary, and only against the
    # classical engine charged the same machine's cost table (the published classical CODEBRIM summary, or the classical
    # re-run model_eval.py makes when that one was charged another machine's costs), with the governor off on both
    f = str(model.relative_to(ROOT)) if model.is_relative_to(ROOT) else str(model)
    if model.exists():
        s = json.loads(model.read_text()); p = head(s["policies"]); lk = s["leakage"]   # model_eval.py writes the leakage count beside the number
        assert s["segmenter"].startswith("onnx:") and not s["agent_config"]["governor"], (s["segmenter"], s["agent_config"])
        am = s.get("agent_metrics", {})
        assert not am.get("governor_activity_in_eval", {}).get("p95>SLA") and all(set(u) <= {"accurate"} for u in am.get("model_usage", {}).values()), "a confounded run"
        same = lambda c: (c["segmenter"] == "classical-v1" and not c["agent_config"]["governor"] and arm_of(c) == arm_of(s) and c["n_images"] == s["n_images"]
                          and all(c["provenance"].get(k) == s["provenance"].get(k) is not None for k in ("dataset_gt_sha256", "dataset_images_sha256")))
        cands = [ROOT / "public/results/codebrim/summary.json", main_checkout() / "runs/model-eval/classical-codebrim/summary.json"]
        base = next((c for c in cands if c.exists() and same(json.loads(c.read_text()))), None)
        if base is None:
            sys.exit(f"{f}: no classical CODEBRIM summary charged the same cost table ({arm_of(s)}) on the same data, governor off; looked at "
                     + ", ".join(map(str, cands)) + ". The film compares the two only like for like.")
        cp = head(json.loads(base.read_text())["policies"])
        wins = [f"L{lvl}" for lvl, t in zip((2, 1, 0), pts(s["policies"]["tiling"])) if t["recall"] > p["recall"]]   # policies.tiling is L2, L1, L0 by pixels
        D["model"] = {"recall": src("trained segmenter: CODEBRIM recall (30 % cap)", p["recall"], f, "policies.drishyam[pixels_frac <= 0.3][-1].recall"),
                      "ci": src("its 95 % CI", p["recall_ci"], f, "policies.drishyam[head].recall_ci"),
                      "chance": src("its boxes at random", p["chance_recall"], f, "policies.drishyam[head].chance_recall"),
                      "classical": src("classical engine, same cost table and data", cp["recall"], str(base), f"policies.drishyam[head].recall (cost table {arm_of(s)})"),
                      "arm": arm_of(s),
                      "tiling_wins": src("tiling levels that still find more than the trained segmenter's agent", wins, f, "policies.tiling[].recall > policies.drishyam[head].recall"),
                      "near_dups": src("CODEBRIM photos with a near-duplicate in dacl10k", lk["near_duplicates"], f, "leakage.near_duplicates")}
    else:
        src("trained segmenter", "results pending", f, "absent at build time")

    # words and figures typed into the scene text, each checked against the file it comes from
    D["lit"] = {
        "competition": src("competition", in_text("README.md", "OpenCV AI Competition 2026, powered by AWS"), "README.md", "line 3"),
        "award": src("entry", in_text("docs/PRD.md", "Agentic Vision"), "docs/PRD.md", "awards line"),
        "sanskrit": src("name", in_text("README.md", "Sanskrit: that which is seen"), "README.md", "line 5"),
        "team": src("team and roles", in_text("README.md", "Vaibhav Jangid, lead engineer (agent policy, OpenCV pipeline, evaluation, AWS deployment)",
                                              "Bhavya Dhoot, design, demo and presentation (site design, video)"), "README.md", "Team"),
        "cites": src("problem-sheet citations", in_text("docs/impact-sources.md", "FHWA, Bridge Condition by Highway System 2026", "23 CFR 650.311", "SPR 787", "Table 4.1"), "docs/impact-sources.md", "sections 1, 2, 5"),
        "client": src("MCP client", in_text("docs/mcp-session/transcript.md", "Claude (Opus 5.5) inside Claude Code"), "docs/mcp-session/transcript.md", "line 4"),
        "graviton": src("Lambda CPU", in_text("docs/aws.md", "arm64 (Graviton)"), "docs/aws.md", "line 4"),
        "behind": src("health card: what the Lambda sits behind", in_text("README.md", "behind CloudFront"), "README.md", "Live demo line"),
        "stack": src("reproduce sheet: what the CDK stack is", in_text("infra/drishyam_stack.py", "Lambda arm64 behind CloudFront and WAF"), "infra/drishyam_stack.py", "stack description"),
        "title": src("cold-open title", in_text("public/index.html", "ONE PHOTO, A FEW LOOKS, ONE HUMAN"), "public/index.html", "the site's figure note"),
        "limits": src("limits, in the site's words", in_text("public/index.html", "Pixels, not millimetres", "AI-assisted, not an inspection", "Only a person can approve",
                                                             "Its licence does not allow sharing the photos"), "public/index.html", "footer note, Human sign-off, the CODEBRIM citation"),
        "test_note": src("reproduce: what make test runs", re.search(r"^test:\s*## (.+)$", (ROOT / "Makefile").read_text(), re.M)[1], "Makefile", "'test:' comment"),
        "recorded": src("footage recorded on", __import__("datetime").date.fromtimestamp((REC / "events.json").stat().st_mtime).isoformat(), "brag-output/work/rec/events.json", "file date"),
    }
    in_text("README.md", "uv sync --all-extras"); in_text("scripts/aws.sh", "deploy)", "smoke)", "destroy)")

    D["urls"] = {"live": src("live site", LIVE_URL, "README.md", "Live demo"), "repo": src("repository", REPO_URL, "README.md / public/index.html", "GitHub link"),
                 "health": src("health check", LIVE_URL + "/api/health", "video/live/health.json", "recorded response")}
    return D


def mcp_rows():
    """The call table for the MCP chapter, as docs/mcp-session/transcript.md prints it (call, what came back, why)."""
    rows = []
    for line in (ROOT / "docs/mcp-session/transcript.md").read_text().splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|(.*)\|(.*)\|(.*)\|\s*$", line)
        if not m:
            continue
        clean = lambda x: re.sub(r"\s*\(\[img\]\([^)]*\)\)|\[img\]\([^)]*\)|`|\*\*", "", x).strip()
        rows.append({"n": int(m[1]), "call": clean(m[2]), "got": clean(m[3]), "why": clean(m[4])})
    return rows


# ---------------------------------------------------------------------------------------------- voice and layout
def narrate(script, out, timings):
    subprocess.run([PY, str(NARRATE), str(script), "--voice", VOICE, "--out", str(out), "--timings", str(timings)], check=True, cwd=WORK)
    return json.loads(pathlib.Path(timings).read_text())


# scene specs: lead before the first line, gap between lines, tail after the last, min seconds
SCENES = [
    ("open", "Cold open", dict(lead=1.9, gap=0.35, tail=0.5)),
    ("team", "Team", dict(lead=0.5, gap=0.35, tail=0.6)),
    ("problem", "Problem and user", dict(lead=0.5, gap=0.35, tail=0.6)),
    ("live", "Live demo", dict(lead=0.5, gap=0.35, tail=0.5)),
    ("break", "Failure drills", dict(lead=0.4, gap=0.35, tail=0.6)),
    ("mcp", "An LLM over MCP", dict(lead=0.5, gap=0.35, tail=0.7)),
    ("arch", "Architecture", dict(lead=0.5, gap=0.35, tail=0.6)),
    ("results", "Results", dict(lead=0.5, gap=0.35, tail=0.6)),
    ("failures", "Failure cases", dict(lead=0.5, gap=0.35, tail=0.6)),
    ("limits", "Limits", dict(lead=0.5, gap=0.35, tail=1.6)),   # the fourth limit is read, not said
    ("repro", "Reproduce", dict(lead=0.5, gap=0.35, tail=0.7)),
    ("end", "End", dict(lead=0.4, gap=0.4, tail=0.2, hold=3.0)),
]


def events():
    E = json.loads((REC / "events.json").read_text())["events"]
    out, approve = {}, 0
    run = None
    for e in E:
        if e["kind"] == "click":
            key = f"click:{e['label']}"
            if e["label"] == "Approve":
                key = f"click:Approve#{approve}"; approve += 1
            out.setdefault(key, e["t"])
            if e["label"] in ("Watch it inspect a bridge", "Now break it", "Try KILL", "Resume"):
                run = {"Watch it inspect a bridge": "guided", "Now break it": "break", "Try KILL": "kill", "Resume": "resume"}[e["label"]]
        elif e["kind"] == "step" and run:
            out.setdefault(f"step:{e['n']}" if run == "guided" else f"step-{run}:{e['n']}", e["t"])
        elif e["kind"] == "done":
            out.setdefault(f"done:{e['run']}", e["t"])
        elif e["kind"] == "approved":
            out.setdefault(f"approved:{e['n']}", e["t"])
    return out, E


def layout(lines, dur):
    """Place every line, scene after scene; a line with `on` waits for nothing (the footage is retimed to it)."""
    t, scenes, placed = 0.0, {}, []
    for sid, title, spec in SCENES:
        mine = [l for l in lines if l["scene"] == sid]
        t0 = t; u = t0 + spec["lead"]
        for k, l in enumerate(mine):
            placed.append({**l, "t": round(u, 3), "dur": dur[l["id"]]})
            u += dur[l["id"]] + (spec["gap"] if k < len(mine) - 1 else 0)
        t1 = u + spec["tail"] + spec.get("hold", 0)
        scenes[sid] = {"t0": round(t0, 3), "t1": round(t1, 3), "title": title}
        t = t1
    return scenes, placed, round(t, 3)


def footage_warp(placed, scenes, ev):
    """For each footage scene: breakpoints [film t, footage t] so each recorded event lands just before its line,
    plus fixed in and out points. Between breakpoints the footage plays at a constant speed (shown on screen)."""
    at = {l["id"]: l["t"] for l in placed}
    du = {l["id"]: l["dur"] for l in placed}
    W = {}
    sc = scenes["live"]
    hs = at["l1"] + du["l1"] + 0.25      # the health still first, then the site
    cw = ev["click:Watch it inspect a bridge"]
    W["live"] = {"t0": round(hs, 3), "t1": sc["t1"], "pts": [
        [hs, cw - 1.9],
        [at["l2"] + 0.5, cw + 0.15],
        ["cut", cw + 0.2, cw + 2.9],
        [at["l3"] - 0.25, ev["step:1"]],
        [at["l4"] - 0.25, ev["step:2"]],
        [at["l5"] - 0.25, ev["step:3"]],
        [at["l6"] - 0.25, ev["step:4"]],
        [at["l7"] - 0.25, ev["step:5"]],
        [at["l8"] - 0.4, ev["done:guided"] + 0.3],
        ["cut", ev["done:guided"] + 2.6, ev["click:Sign them off"] - 1.4],
        [at["l9"] - 0.2, ev["click:Sign them off"]],
        [at["l11"] - 0.3, ev["approved:2"] + 0.9],
        [sc["t1"], ev["approved:2"] + 2.5],
    ]}
    sc = scenes["break"]
    cb = ev["click:Now break it"]
    W["break"] = {"t0": sc["t0"], "t1": sc["t1"], "pts": [
        [sc["t0"], cb - 1.2],
        [at["b0"] + 0.25, cb + 0.1],
        ["cut", cb + 0.4, ev["step-break:5"] - 1.2],
        [at["b1"] - 0.2, ev["step-break:5"]],
        [at["b2"] - 0.2, ev["step-break:6"]],
        [at["b2"] + du["b2"] + 0.1, ev["step-break:6"] + 3.4],
        ["cut", ev["step-break:6"] + 3.5, ev["click:Try KILL"] - 0.9],
        [at["b3"] - 0.2, ev["click:Try KILL"] + 0.1],
        [at["b3"] + 0.25 * du["b3"], ev["click:KILL"] - 0.15],
        [at["b3"] + 0.45 * du["b3"], ev["done:kill"] + 0.6],
        [at["b3"] + 0.62 * du["b3"], ev["click:Resume"]],
        [at["b4"], ev["step-resume:2"]],
        [sc["t1"], ev["step-resume:3"] + 1.5],
    ]}
    sc = scenes["open"]
    W["open"] = {"t0": sc["t0"] + 1.6, "t1": sc["t1"], "pts": [
        [sc["t0"] + 1.6, ev["step:1"] - 0.2],
        [sc["t1"] - 2.8, ev["step:4"] + 1.5],
        ["cut", ev["step:4"] + 1.6, ev["click:Sign them off"] + 0.9],
        [sc["t1"], ev["click:Sign them off"] + 2.6],
    ]}
    for k, w in W.items():   # resolve: a "cut" holds the film time of the previous point; check monotonic
        out, prev = [], None
        for p in w["pts"]:
            if p[0] == "cut":
                out.append({"cut": True, "a": round(p[1], 3), "b": round(p[2], 3)})
            else:
                if prev is not None and p[0] <= prev:
                    sys.exit(f"warp {k}: film time {p[0]:.2f} not after {prev:.2f}; lengthen the lines before it")
                out.append({"t": round(p[0], 3), "f": round(p[1], 3)}); prev = p[0]
        w["pts"] = out
    return W


def cues(placed):
    """Burned-in captions and WebVTT: one cue per line, split at sentence or clause breaks to at most 2 x 44 chars."""
    out = []
    for l in placed:
        text = re.sub(r"(\d) (s|ms|px|%|months)(?!\w)", "\\1\u00a0\\2", l["cap"])   # a number never wraps away from its unit
        parts = [text]
        for pat in (r"(?<=[.!?]) +", r"(?<=[;:]) +", r"(?<=,) +"):   # sentences, then clauses, then commas; plain spaces only
            if all(len(x) <= 88 for x in parts):
                break
            parts = [y.strip() for x in parts for y in (re.split(pat, x) if len(x) > 88 else [x]) if y.strip()]
        merged = []
        for p in parts:
            if merged and len(merged[-1]) + 1 + len(p) <= 88:
                merged[-1] += " " + p
            else:
                merged.append(p)
        total = sum(len(p) for p in merged)
        t = l["t"]
        for p in merged:
            d = l["dur"] * len(p) / total
            out.append({"t": round(t, 3), "end": round(t + d, 3), "text": p, "id": l["id"]})
            t += d
    for k, c in enumerate(out):   # each cue lingers 0.25 s, never into the next one; film.js and the VTT both stop at `stop`
        c["stop"] = round(min(c["end"] + 0.25, out[k + 1]["t"] - 0.01) if k + 1 < len(out) else c["end"] + 0.25, 3)
    return out


def vtt(cs, path):
    def ts(x):
        h, r = divmod(x, 3600); m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    body = ["WEBVTT", ""]
    for k, c in enumerate(cs, 1):
        body += [str(k), f"{ts(c['t'])} --> {ts(c['stop'])}", c["text"], ""]
    path.write_text("\n".join(body))


# ---------------------------------------------------------------------------------------------- sound
def score(sfx_json, dur, out):
    """A soft analogue pad in C under the whole film (the patent-office set: pencil, ruler click, soft synth),
    with the page's cues placed by soundkit. narrate.py ducks it under the voice."""
    sys.path.insert(0, str(SOUNDKIT))
    from soundkit import Mix, pad, play_cues, pluck, hz
    cues_ = json.loads(pathlib.Path(sfx_json).read_text())["events"]
    mix = Mix(dur + 0.5)
    prog = [[48, 55, 60, 64], [45, 52, 57, 60], [41, 48, 53, 57], [43, 50, 55, 59]]
    bar = 8.0
    k, t = 0, 0.0
    while t < dur - 2:
        d = min(bar + 1.0, dur - t)
        mix.add(pad(prog[k % 4], d, 900), t, 0.05, -0.15 if k % 2 else 0.15)
        for j, m in enumerate(prog[k % 4][1:]):
            mix.add(pluck(hz(m + 12), 1.2), t + j * 0.5, 0.012, (j - 1) * 0.3)
        k += 1; t += bar
    mix.add(pad([48, 55, 60, 64, 67], 4.0, 1100), max(0, dur - 4.2), 0.06)
    play_cues(mix, cues_, gain=0.7)
    mix.save(out, peak_db=-6, fade=2.0)


# ---------------------------------------------------------------------------------------------- plan.md
REQS = [   # competition requirement -> where the film covers it (scene ids or narration line ids)
    ("At most 5 minutes, 1920x1080", ["open", "end"]),
    ("Narrated (offline Kokoro voice af_heart) with burned-in captions; the same cues in public/video/drishyam.vtt", ["open", "end"]),
    ("The team (name cards, no footage): Vaibhav Jangid, Bhavya Dhoot", ["team"]),
    ("Problem and user, with sources on screen", ["problem"]),
    ("/api/health on the live site: opencv 5.0.0, aarch64, aws-lambda", ["l1"]),
    ("The application working: the guided run on bridge_03 ('Watch it inspect a bridge'), recorded live, with what it missed", ["l2", "l3", "l4", "l5", "l6", "l7", "l8"]),
    ("Plain-words trace (the NOTE and 'What it decided and why' panels)", ["l3", "l4", "l5", "l6", "l7"]),
    ("Human sign-off: reviewer name, Approve, APPROVED stamp, work order signed", ["l9"]),
    ("Human control: a tampered work-order token refused (403)", ["l11"]),
    ("Failure handling: 'Now break it' (slowed segmenter -> governor; malformed planner reply -> allow-list)", ["b0", "b1", "b2"]),
    ("Human control: 'Try KILL' (KILL halts with kill-switch, Resume replays)", ["b3"]),
    ("Honesty: no LLM runs in the live demo; the drill planners are scripted", ["b4"]),
    ("An LLM driving the OpenCV tools over MCP: IoU 0.00 at call 6 -> human; IoU 0.72 at call 18 -> confirmed; call 20 budget", ["mcp"]),
    ("Architecture: docs/agent-loop.svg, public/docs/aws-simple.svg, docs/architecture.png (AWS path and OpenCV layer)", ["arch"]),
    ("Principal results: recall vs pixels, with one downscale, the tiling family and chance", ["r1", "r3"]),
    ("Larger photos (4608 px): beats one downscale (which reads far fewer pixels), L2 tiling matches it with fewer pixels", ["r4"]),
    ("Ablations (public/results/ablations.json)", ["r5"]),
    ("Real photos (CODEBRIM), said plainly weak; no CODEBRIM image shown", ["r6"]),
    ("False alarms on synthetic photos with no damage (public/results/clean/summary.json)", ["r7"]),
    ("Trained segmenter: 'results pending' until public/results/codebrim_dacl10k/summary.json exists", ["r8"]),
    ("Failure cases: failures.json plates and their rules; one CODEBRIM case in words only", ["failures"]),
    ("Limits and responsible use", ["limits"]),
    ("Reproduce: make test, make eval, scripts/aws.sh", ["repro"]),
    ("End card: live URL, repository, team", ["end"]),
]


def plan_tables(scenes, placed, D):
    def mmss(x):
        x = round(x, 1)
        return f"{int(x // 60)}:{x % 60:04.1f}"
    L = {l["id"]: l for l in placed}
    req = ["| Requirement | Covered by | From | To |", "| --- | --- | --- | --- |"]
    for what, ids in REQS:
        span = [(scenes[i]["t0"], scenes[i]["t1"]) if i in scenes else (L[i]["t"], L[i]["t"] + L[i]["dur"]) for i in ids]
        lo, hi = min(a for a, _ in span), max(b for _, b in span)
        if ids == ["open", "end"]:
            lo, hi = 0.0, scenes["end"]["t1"]
        req.append(f"| {what} | {', '.join(ids)} | {mmss(lo)} | {mmss(hi)} |")
    rows = ["| Chapter | Starts | Ends | Lines |", "| --- | --- | --- | --- |"]
    for sid, title, _ in SCENES:
        s = scenes[sid]
        rows.append(f"| {title} | {mmss(s['t0'])} | {mmss(s['t1'])} | {', '.join(l['id'] for l in placed if l['scene'] == sid)} |")
    nums = ["| What | Value | Source file | Field |", "| --- | --- | --- | --- |"]
    for what, value, f, field in SRC:
        v = json.dumps(value) if not isinstance(value, str) else value
        nums.append(f"| {what} | {v} | `{f}` | {field} |")
    script = ["| Id | Starts | Spoken | Caption |", "| --- | --- | --- | --- |"]
    for l in placed:
        script.append(f"| {l['id']} | {mmss(l['t'])} | {l['say']} | {l['cap']} |")
    return "\n".join(["### Requirement checklist (timestamps from this build)", "", *req, "", "### Chapters (from this build)", "", *rows, "", "### Every number and URL, with its source", "", *nums,
                      "", "### Narration and captions (from video/narration.py)", "", *script])


def write_plan(text):
    p = VIDEO / "plan.md"
    s = p.read_text()
    a, b = "<!-- generated:start -->", "<!-- generated:end -->"
    s = s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):]
    p.write_text(s)


# ---------------------------------------------------------------------------------------------- main
def fast(*args):
    subprocess.run([PY, str(FAST), *map(str, args)], check=True, cwd=VIDEO)


def measure(L):
    """Each line's spoken length (sequential start times; lines are cached by text), then the layout fitted to it."""
    probe = [{"t": round(i * 20.0, 1), "text": l["say"]} for i, l in enumerate(L)]
    (WORK / "probe.json").write_text(json.dumps(probe, indent=1))
    tm = narrate(WORK / "probe.json", WORK / "probe.wav", WORK / "probe_timings.json")
    by_text = {x["text"]: x["dur"] for x in tm}
    return layout(L, {l["id"]: by_text[l["say"]] for l in L})


def check_template(narration_lines):
    """The trained-segmenter line on a stand-in summary (the classical CODEBRIM numbers relabelled as an ONNX run):
    it must switch on, pass every check, fit the 5-minute cap, and refuse a summary charged another machine's costs."""
    c = load("public/results/codebrim/summary.json")
    fake = {**c, "segmenter": "onnx:stand-in.onnx", "leakage": {"near_duplicates": 0},
            "cost_model": {**c["cost_model"], "source": c["cost_model"]["source"].replace("/micro.json", "-dacl10k/micro.json", 1)}}
    tmp = WORK / "template-check"; tmp.mkdir(exist_ok=True)
    (tmp / "summary.json").write_text(json.dumps(fake))
    L = narration_lines(data(tmp / "summary.json"))
    r8 = next(l for l in L if l["id"] == "r8")
    assert "pending" not in r8["say"] and "governor off" in r8["say"].lower() and "same cost table" in r8["say"], r8
    print("r8 say:", r8["say"], "\nr8 cap:", r8["cap"])
    _, _, total = measure(L)
    print(f"with the trained-segmenter line the film runs {total:.1f} s (cap 300)")
    (tmp / "summary.json").write_text(json.dumps({**fake, "cost_model": {**fake["cost_model"], "source": "results/some-other-machine/micro.json"}}))
    try:
        data(tmp / "summary.json")
    except SystemExit as e:
        print("another machine's cost table is refused:", str(e)[:90], "...")
    else:
        sys.exit("a summary charged another machine's cost table was accepted")
    shutil.rmtree(tmp)
    if total > 300:
        sys.exit("the trained-segmenter line would take the film over 5 minutes: shorten the narration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--layout", action="store_true", help="stop after the voice, the layout and plan.md (public/video/ is left alone)")
    ap.add_argument("--short-only", action="store_true", help="after the layout, render only the 9:16 cut")
    ap.add_argument("--check-template", action="store_true", help="try the trained-segmenter line on a stand-in summary, then stop")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True); PUB.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(VIDEO))
    from narration import lines as narration_lines
    if a.check_template:
        return check_template(narration_lines)
    D = data()
    L = narration_lines(D)
    # 1. measure every line, 2. fit the chapters to the voice, then synthesize the real track
    scenes, placed, total = measure(L)
    if total > 300:
        sys.exit(f"the film would run {total:.1f} s, over the 5-minute limit: shorten the narration")
    (WORK / "script.json").write_text(json.dumps([{"t": l["t"], "text": l["say"]} for l in placed], indent=1))
    narrate(WORK / "script.json", WORK / "narration.wav", WORK / "timings.json")
    ev, E = events()
    warp = footage_warp(placed, scenes, ev)
    cs = cues(placed)
    frames = json.loads((REC / "frames.json").read_text())
    walk = json.loads((WORK / "walkrec/demo.js").read_text().split("=", 1)[1].rstrip(";\n "))
    film = {"D": D, "lines": placed, "cues": cs, "scenes": scenes, "order": [s[0] for s in SCENES], "dur": total, "warp": warp,
            "frames": frames, "events": E, "ev": ev, "walk": walk, "health_boxes": json.loads((REC / "health_boxes.json").read_text()),
            "urls": D["urls"]}
    (WORK / "film-data.js").write_text("window.FILM = " + json.dumps(film) + ";")
    vtt(cs, WORK / "drishyam.vtt")   # public/video/ gets it only with the MP4 rendered from this same layout
    write_plan(plan_tables(scenes, placed, D))
    print(f"film {total:.1f} s, {len(placed)} lines, {len(cs)} cues; chapters:")
    for sid, s in scenes.items():
        print(f"  {sid:9s} {s['t0']:7.2f} -> {s['t1']:7.2f}")
    page = VIDEO / "film.html"
    if a.layout:
        return
    if a.short_only:
        return short(D, scenes, placed)
    if a.stills is not None:
        fast(page, "--stills", *(a.stills or [s["t0"] + 2 for s in scenes.values()]), "--still-dir", WORK / "stills")
        return
    if a.audit:
        fast(page, "--audit")
        return
    fast(page, "--dump-sfx", WORK / "sfx.json")
    score(WORK / "sfx.json", total, WORK / "score.wav")
    subprocess.run([PY, str(NARRATE), str(WORK / "script.json"), "--voice", VOICE, "--out", str(WORK / "narration.wav"),
                    "--timings", str(WORK / "timings.json"), "--music", str(WORK / "score.wav"), "--mix", str(WORK / "mix.wav")], check=True, cwd=WORK)
    # rendered into WORK (fast-render keeps its segment folder beside --out), then moved into public/video/ together
    fast(page, "--out", WORK / "drishyam.mp4", "--fps", 30, "--crf", 20, "--audio", WORK / "mix.wav", "--poster", film_poster(scenes, placed), "--poster-jpg", WORK / "poster.jpg")
    from PIL import Image   # the poster goes in git: a q100 capture is ~1 MB, q85 is a fifth of that and looks the same
    Image.open(WORK / "poster.jpg").convert("RGB").save(PUB / "poster.jpg", quality=85, optimize=True, progressive=True)
    shutil.move(WORK / "drishyam.mp4", PUB / "drishyam.mp4"); shutil.copy(WORK / "drishyam.vtt", PUB / "drishyam.vtt")
    if not a.no_short:
        short(D, scenes, placed)


def film_poster(scenes, placed):
    """The poster: the end card, fully settled (URL, tagline, team)."""
    return round(scenes["end"]["t1"] - 0.5, 2)


def short(D, scenes, placed):
    """The 15-20 s 9:16 cut: the cold open and the sign-off stamp, its own two lines of voice."""
    s = [{"t": 0.8, "text": next(l["say"] for l in placed if l["id"] == "o1")},
         {"t": 0.0, "text": "Only a person can approve a critical finding."}]
    tm = narrate_short(s)
    fast(VIDEO / "short.html", "--width", 1080, "--height", 1920, "--dump-sfx", WORK / "short-sfx.json")
    dur = json.loads((WORK / "short-sfx.json").read_text())["dur"]
    score(WORK / "short-sfx.json", dur, WORK / "short-score.wav")
    subprocess.run([PY, str(NARRATE), str(WORK / "short-script.json"), "--voice", VOICE, "--out", str(WORK / "short-narration.wav"),
                    "--timings", str(WORK / "short-timings.json"), "--music", str(WORK / "short-score.wav"), "--mix", str(WORK / "short-mix.wav")], check=True, cwd=WORK)
    fast(VIDEO / "short.html", "--width", 1080, "--height", 1920, "--out", ROOT / "brag-output/drishyam-short.mp4", "--fps", 30,
         "--audio", WORK / "short-mix.wav", "--poster", 1.2)


def narrate_short(s):
    """Line 1 at 0.8 s; line 2 placed after the stamp, which the page puts at SHORT_STAMP (film.html)."""
    (WORK / "short-probe.json").write_text(json.dumps([{"t": 0.8, "text": s[0]["text"]}, {"t": 30.0, "text": s[1]["text"]}]))
    tm = narrate(WORK / "short-probe.json", WORK / "short-probe.wav", WORK / "short-probe-timings.json")
    d0, d1 = tm[0]["dur"], tm[1]["dur"]
    t1 = round(0.8 + d0 + 2.2, 2)
    script = [{"t": 0.8, "text": s[0]["text"]}, {"t": t1, "text": s[1]["text"]}]
    (WORK / "short-script.json").write_text(json.dumps(script))
    (WORK / "short-data.js").write_text("window.SHORT = " + json.dumps({"lines": [{"t": 0.8, "dur": d0, "text": tm[0]["text"]},
                                                                                 {"t": t1, "dur": d1, "text": tm[1]["text"]}], "dur": round(max(15.5, t1 + d1 + 3.4), 2)}) + ";")
    return tm


if __name__ == "__main__":
    main()
