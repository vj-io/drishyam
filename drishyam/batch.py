"""`drishyam batch`: one inspection over a backlog of photos (one drone inspection made 2,536: docs/report.md, section 1).

1. Triage. Every photo gets the agent's own overview, as its job will compute it: decoded, people blurred, then the
   first two events of `Job.run` (pyramid, the L3 saliency map, candidate windows). Its score is the sum of the
   candidates' priors: before any look a region's expected value is its prior (severity 1, no entropy yet), so the
   sum is the photo's expected value in the agent's own units. No new model, no new measurement.
2. Order. Highest score first; ties (photos with no candidate) in file order.
3. Run. The unchanged per-photo `Job` on each photo in that order, with the per-photo caps. Triage and every job are
   charged, on the job's own clock, to one total (the $ cap is the same budget in seconds: the agent models $ from
   seconds); each job gets at most what is left as its seconds cap, and the batch stops when nothing is left. The kill
   switch (`drishyam kill`, Ctrl-C) stops it between photos, and inside a job at its next step (the governor's check).
4. Report. inspection.json, sealed with findings.sign (HMAC), and inspection.md: each photo's status, the human queue
   (every critical finding, by severity then confidence), everything not assessed (photos never reached or unusable,
   and each work order's own list), and the totals.

The batch approves nothing. Each photo's work order is reviewed with `drishyam review`; `drishyam batch --report DIR`
then recomputes the report from the work orders, refusing when the seal does not verify or when a critical finding the
sealed report listed is gone, or changed state without a recorded review. Every decision is one line of
batch_trace.jsonl with its cause and the measured values behind it, as each look is in a job's trace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import time
from dataclasses import replace
from pathlib import Path

from . import findings, imaging, planner, privacy
from .agent import Job
from .cli import fixed_cost_job, write_crops, write_json_atomic
from .findings import DISCLAIMER, SEVERITY, now_iso

PHOTO_TYPES = {".jpg", ".jpeg", ".png"}
SEAL_TTL_S = 10 * 365 * 24 * 3600  # a seal on a report, not a review token: it must outlast the review work


def photo_list(args) -> list[Path]:
    """Folders (their JPEG and PNG files by name: a drone numbers them in capture order), photos, and .txt lists of
    photo paths, one per line, relative to the list's folder."""
    out = []
    for a in map(Path, args):
        if a.is_dir():
            out += sorted(p for p in a.iterdir() if p.suffix.lower() in PHOTO_TYPES and p.is_file())
        elif a.suffix.lower() == ".txt":
            out += [a.parent / line.strip() for line in a.read_text().splitlines() if line.strip()]
        else:
            out.append(a)
    return out


def make_job(img, cfg, kill=lambda: False, job_id=None, cost_model="live", micro_arm="linux-aarch64") -> Job:
    """The Job `drishyam run` makes for one photo: live (this machine's clock) or on the published fixed costs."""
    kw = {"kill": kill, "planner": planner.from_env(), "job_id": job_id}
    return Job(img, cfg, **kw) if cost_model == "live" else fixed_cost_job(img, cfg, micro_arm, kw)


def load(path: Path):
    """(image, privacy summary): the photo as its job sees it, decoded and with people blurred. Raises OSError or
    ValueError (an image the decoder refuses, or a photo of a person) on an unusable photo."""
    img = imaging.decode_image(Path(path).read_bytes())
    return img, privacy.privacy_pass(img)


def triage(img, cfg, make) -> dict:
    """The photo's overview, from its own Job (the first two events of `run`, then the generator is closed)."""
    job = make(img, cfg)
    events = job.run()
    next(events)
    priors = [c["prior"] for c in next(events)["candidates"]]
    events.close()
    return {"candidates": len(priors), "prior_sum": round(sum(priors), 4), "prior_max": max(priors, default=0.0),
            "seconds": job.budget.seconds_used, "pixels": job.budget.pixels_used}


def run_photo(img, priv, cfg, make, root: Path, cost_model: str, micro_arm: str) -> dict:
    """One photo's job, written as `drishyam run` writes it: <root>/<job>/trace.jsonl, work_order.json, crops/."""
    while True:  # 2,536 three-byte ids collide about one batch in five: draw again, as the MCP server does
        job_id = "j_" + secrets.token_hex(3)
        try:
            (root / job_id).mkdir()
            break
        except FileExistsError:
            continue
    job = make(img, cfg, job_id=job_id)
    with (root / job_id / "trace.jsonl").open("w") as f:
        for ev in job.run():
            if ev["type"] == "job":
                ev["privacy"] = {k: v for k, v in priv.items() if k != "boxes"}
                if cost_model == "fixed":
                    ev["cost_model"] = {"kind": "fixed", "micro_arm": micro_arm}
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")
            done = ev
    write_json_atomic(root / job_id / "work_order.json", done["work_order"])
    write_crops(root / job_id, job.findings)
    s = done["summary"]
    return {"run_dir": job_id, "cause": done["cause"], "steps": s["steps"], "findings": s["findings"],
            "critical_pending": s["critical_pending"], "seconds": job.budget.seconds_used,
            "usd": job.budget.usd_used, "pixels": job.budget.pixels_used}


def run_batch(paths, out: Path, cfg, usd_cap=None, seconds_cap=None, make=make_job, kill=lambda: False,
              key: bytes = b"", cost_model="live", micro_arm="linux-aarch64", progress=None) -> Path:
    """Triage, order and run `paths` under one total budget (None: no cap); returns the batch folder under `out`."""
    t_wall = time.perf_counter()
    while True:
        bdir = Path(out) / f"batch-{secrets.token_hex(3)}"
        try:
            bdir.mkdir(parents=True)
            break
        except FileExistsError:
            continue
    make_one = lambda img, c, job_id=None: make(img, c, kill=kill, job_id=job_id, cost_model=cost_model,
                                                 micro_arm=micro_arm)
    budget = min(math.inf if seconds_cap is None else seconds_cap,
                 math.inf if usd_cap is None else usd_cap * 3600 / cfg.usd_per_hour)
    photos = [{"photo": str(p), "index": i, "rank": None} for i, p in enumerate(paths)]
    spent, stop = 0.0, None

    def log(decision, cause, rationale, photo=None, **values):
        rec = {"type": "batch", "batch": bdir.name, "t": now_iso(), "decision": decision,
               **({"photo": photo["photo"]} if photo else {}), "cause": cause, "rationale": rationale,
               "spent_s": round(spent, 6), **values}
        with (bdir / "batch_trace.jsonl").open("a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        if progress:
            progress(rec)

    def unusable(p, exc):
        p.update(outcome="refused" if isinstance(exc, privacy.PeopleDominant) else "unreadable", reason=str(exc))

    for p in photos:
        if kill():
            stop = ("kill-switch", "kill switch set during triage")
            break
        try:
            t = triage(load(p["photo"])[0], cfg, make_one)
        except (OSError, ValueError) as exc:
            unusable(p, exc)
            log("triaged", "unreadable", str(exc), p)
            continue
        p["triage"] = t
        spent += t["seconds"]
        log("triaged", "overview", f"{t['candidates']} candidates, prior sum {t['prior_sum']:g}, max "
            f"{t['prior_max']:g}", p, **t)
    ranked = sorted((p for p in photos if "triage" in p), key=lambda p: (-p["triage"]["prior_sum"], p["index"]))
    if stop:
        log("killed", *stop)
    else:
        log("ordered", "triage-score", "highest sum of candidate priors first (ties: file order)",
            order=[p["index"] for p in ranked])
    for rank, p in enumerate(ranked, 1):
        p["rank"] = rank
        if stop:
            continue
        if kill():
            stop = ("kill-switch", "kill switch set between photos")
        elif spent >= budget:
            stop = ("budget-stop", f"{spent:.3f} s spent of the {budget:.3f} s budget (triage included)")
        if stop:
            log("killed" if stop[0] == "kill-switch" else "budget-stop", *stop)
            continue
        cap = min(cfg.seconds_cap, budget - spent)
        log("started", "rank", f"rank {rank} of {len(ranked)}: prior sum {p['triage']['prior_sum']:g} over "
            f"{p['triage']['candidates']} candidates; seconds cap {cap:.3f} s ("
            f"{'per photo' if cap == cfg.seconds_cap else 'what the batch budget leaves'})", p, seconds_cap=cap)
        try:
            img, priv = load(p["photo"])  # decoded again: a backlog does not fit in memory
        except (OSError, ValueError) as exc:
            unusable(p, exc)
            log("finished", "unreadable", str(exc), p)
            continue
        r = run_photo(img, priv, replace(cfg, seconds_cap=cap), make_one, bdir, cost_model, micro_arm)
        p.update(outcome="assessed", **r)
        spent += r["seconds"]
        log("finished", r["cause"], f"{r['steps']} looks, {r['findings']} findings, {r['critical_pending']} critical "
            f"pending; {r['seconds']:.3f} s", p, **{k: r[k] for k in ("run_dir", "steps", "seconds", "pixels")})
        if r["cause"] == "kill-switch":
            stop = ("kill-switch", "kill switch set while this photo ran: its job halted at its next step")
            log("killed", *stop, p)
    for p in photos:
        if "outcome" not in p:
            p.update(outcome="not-reached", reason=f"not reached: {stop[0]}: {stop[1]}")
    report = {"id": bdir.name, "created": now_iso(), "disclaimer": DISCLAIMER,
              "budget": {"usd_cap": usd_cap, "seconds_cap": seconds_cap,
                         "seconds_budget": None if budget == math.inf else budget, "usd_per_hour": cfg.usd_per_hour,
                         "per_photo": {"pixel_cap": cfg.pixel_cap, "usd_cap": cfg.usd_cap,
                                       "seconds_cap": cfg.seconds_cap},
                         "cost_model": cost_model, **({"micro_arm": micro_arm} if cost_model == "fixed" else {})},
              "stop": stop and {"cause": stop[0], "rationale": stop[1]}, "photos": photos}
    tri = sum(p["triage"]["seconds"] for p in photos if "triage" in p)
    report["spent"] = {"seconds": spent, "usd": spent * cfg.usd_per_hour / 3600, "triage_seconds": tri,
                       "pixels": sum(p.get("pixels", 0) + p.get("triage", {}).get("pixels", 0) for p in photos),
                       "wall_seconds": time.perf_counter() - t_wall}  # decode and privacy blur too: not charged
    write(bdir, derive(bdir, report), key)
    return bdir


# ---- the report -----------------------------------------------------------------------------------
def derive(bdir: Path, report: dict) -> dict:
    """The parts of the report that come from the photos' work orders: each photo's status, the human queue, the
    not-assessed list, the counts and the batch status (blocked, incomplete or ready, as a work order's)."""
    queue, na = [], []
    for p in report["photos"]:
        if "run_dir" not in p:
            p["status"] = "not assessed"
            na.append({"photo": p["photo"], "reason": p["reason"],
                       "action": "run it in a later batch" if p["outcome"] == "not-reached" else "check the file"})
            continue
        wo_path = f"{p['run_dir']}/work_order.json"
        wo = json.loads((bdir / wo_path).read_text())
        p["status"] = wo["status"]
        queue += [{"photo": p["photo"], "work_order": wo_path, "finding": f["id"], "cls": f["cls"],
                   "confidence": f["confidence"], "state": f["state"], "critical_rule": f["critical_rule"],
                   "box": f["box"]} for f in wo["findings"] if f["critical"]]
        na += [{"photo": p["photo"], **n} for n in wo["not_assessed"]]
    queue.sort(key=lambda q: (-SEVERITY[q["cls"]], -q["confidence"]))
    pending = sum(q["state"] == "pending_review" for q in queue)
    statuses = [p["status"] for p in report["photos"]]
    counts = {s: statuses.count(s) for s in ("blocked", "incomplete", "ready", "not assessed")}
    status = "blocked" if pending else "incomplete" if na else "ready"
    return {**report, "status": status, "counts": {"photos": len(statuses), **counts, "critical": len(queue),
                                                   "pending": pending}, "queue": queue, "not_assessed": na}


def _digest(report: dict) -> dict:
    body = {k: v for k, v in report.items() if k != "signature"}
    return {"sha256": hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def write(bdir: Path, report: dict, key: bytes) -> None:
    report["signature"] = findings.sign(_digest(report), key, SEAL_TTL_S)
    write_json_atomic(bdir / "inspection.json", report)
    (bdir / "inspection.md").write_text(markdown(report))


def verify_report(bdir: Path, key: bytes) -> dict:
    """The sealed report; raises findings.SignatureError if anything in it changed since it was signed."""
    report = json.loads((Path(bdir) / "inspection.json").read_text())
    if findings.verify(report.get("signature", ""), key) != _digest(report):
        raise findings.SignatureError("inspection.json does not match its signature")
    return report


def refresh(bdir: Path, key: bytes) -> dict:
    """Recompute the report from the work orders after reviews, and seal it again. Refused (ReviewError) when a
    critical finding the sealed report listed is gone, no longer critical, or left pending_review (or changed a
    decision) without an approval recorded in its work order: only `drishyam review` moves one."""
    bdir = Path(bdir)
    old = verify_report(bdir, key)
    new = derive(bdir, json.loads(json.dumps(old)))
    now = {(q["work_order"], q["finding"]): q for q in new["queue"]}
    changed = 0
    for q in old["queue"]:
        where = f"{q['finding']} in {q['work_order']}"
        if (q["work_order"], q["finding"]) not in now:
            raise findings.ReviewError(f"critical finding {where} is missing from its work order or no longer critical")
        state = now[(q["work_order"], q["finding"])]["state"]
        if state == q["state"]:
            continue
        approvals = json.loads((bdir / q["work_order"]).read_text())["approvals"]
        decided = {"approved": "approve", "rejected": "reject"}.get(state)
        if q["state"] != "pending_review" or not any(
                a["finding_id"] == q["finding"] and a["decision"] == decided for a in approvals):
            raise findings.ReviewError(f"{where} went from {q['state']} to {state} without a recorded review "
                                       "(drishyam review)")
        changed += 1
    write(bdir, new, key)
    with (bdir / "batch_trace.jsonl").open("a") as f:
        f.write(json.dumps({"type": "batch", "batch": new["id"], "t": now_iso(), "decision": "refreshed",
                            "cause": "review", "rationale": f"{changed} decisions recorded since the last report; "
                            f"status {new['status']}"}, separators=(",", ":")) + "\n")
    return new


def markdown(r: dict) -> str:
    c, s = r["counts"], r["spent"]
    name = lambda p: Path(p).name
    cap = r["budget"]["seconds_budget"]
    lines = [f"# Inspection {r['id']}: {r['status']}", "", r["disclaimer"], "",
             f"{c['photos']} photos: {c['blocked']} blocked, {c['incomplete']} incomplete, {c['ready']} ready, "
             f"{c['not assessed']} not assessed. Spent {s['seconds']:.2f} s (${s['usd']:.5f})"
             f"{f' of {cap:.2f} s' if cap is not None else ''}, triage {s['triage_seconds']:.2f} s, "
             f"{s['pixels']:,} px." + (f" Stopped: {r['stop']['cause']}: {r['stop']['rationale']}." if r["stop"] else ""),
             "", f"## Human queue: {c['pending']} of {c['critical']} critical findings pending review", ""]
    if r["queue"]:
        lines += ["| photo | finding | class | confidence | state | work order |", "|---|---|---|---|---|---|"]
        lines += [f"| {name(q['photo'])} | {q['finding']} | {q['cls']} | {q['confidence']:.2f} | {q['state']} | "
                  f"{q['work_order']} |" for q in r["queue"]]
        lines += ["", ("A person decides each one: `drishyam review <this folder>/<work order> <finding> "
                       "approve|reject --by NAME`; `drishyam batch --report <this folder>` then recomputes this report.")]
    lines += ["", f"## Not assessed: {len(r['not_assessed'])}", ""]
    lines += [f"- {name(n['photo'])}: {n['reason']} ({n['action']}"
              f"{', box ' + str(n['box']) if 'box' in n else ''})" for n in r["not_assessed"]]
    lines += ["", "## Photos, in the order run", "", "| rank | photo | candidates | prior sum | status | s | findings |",
              "|---|---|---|---|---|---|---|"]
    for p in sorted(r["photos"], key=lambda p: (p["rank"] is None, p["rank"] or 0, p["index"])):
        t, secs = p.get("triage", {}), round(p["seconds"], 2) if "seconds" in p else "-"
        lines.append(f"| {p['rank'] or '-'} | {name(p['photo'])} | {t.get('candidates', '-')} | "
                     f"{t.get('prior_sum', '-')} | {p['status']} | {secs} | {p.get('findings', '-')} |")
    return "\n".join(lines) + "\n"


def signing_key() -> bytes:
    """DRISHYAM_SIGNING_KEY (32 characters or more, as the web app reads it), else a random key kept owner-only in
    ~/.drishyam/signing_key, made on first use, so a report sealed today still verifies when its reviews come in."""
    if len(env := os.environ.get("DRISHYAM_SIGNING_KEY", "")) >= 32:
        return env.encode()
    path = Path("~/.drishyam/signing_key").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    key = path.read_bytes().strip()
    if len(key) < 32:
        raise ValueError(f"{path} holds a key shorter than 32 bytes; delete it to make a new one")
    return key
