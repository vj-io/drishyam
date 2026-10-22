"""The `drishyam` command line: run a job, review a work order, replay a trace, kill switch, MCP and web servers.

`review` is the only path (with the web UI) that can approve or reject a critical finding. It is a
human-run command: the agent has no shell and no MCP tool that reaches it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import threading
from pathlib import Path

from . import findings, governor, imaging, planner, privacy
from .agent import REPLAY_ACTIONS, Job, JobConfig, gaze_path_from_trace

CAPS = {"pixel_cap": (0.02, 1.0), "usd_cap": (0.0001, 0.05), "seconds_cap": (1.0, 30.0)}  # as the web API
CHAOS = ("p95_spike", "planner_timeout", "planner_malformed")  # as web.CHAOS
USER_ERRORS = (OSError, ValueError, findings.ReviewError)  # bad input, reported without a traceback


def kill_path() -> Path:
    """The kill-switch flag file: its existence halts every job (MCP and CLI) within one step."""
    return Path(governor.kill_path())


def write_json_atomic(path: Path, obj) -> None:
    # ponytail: fixed temp name, so two writers racing on one file can lose an update; a file lock if that matters
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def write_crops(run_dir: Path, items) -> None:
    """crops/<finding id>.jpg for every finding that carries a thumbnail."""
    for f in items:
        if f.crop_jpeg_b64:
            (run_dir / "crops").mkdir(exist_ok=True)
            (run_dir / "crops" / f"{f.id}.jpg").write_bytes(base64.b64decode(f.crop_jpeg_b64))


def read_trace(path) -> list[dict]:
    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # a trace cut short mid-line still replays up to the cut
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _fail(message: str) -> int:
    print(f"drishyam: {message}", file=sys.stderr)
    return 1


# ---- run ------------------------------------------------------------------------------------------
def _progress(ev: dict) -> None:
    t = ev["type"]
    if t == "job":
        b = ev["budget"]
        line = (
            f"job {ev['job']}  {ev['width']}x{ev['height']}  caps {b['pixel_cap_frac']:.0%} px, "
            f"${b['usd_cap']}, {b['seconds_cap']:.0f} s"
        )
    elif t == "overview":
        line = f"overview: {len(ev['candidates'])} candidates"
    elif t == "step":
        d = ev["decision"]
        if d["action"] == "halt":
            line = f"step {ev['step']}  halt  {ev['cause']}: {ev['rationale']}"
            if g := ev.get("governor"):
                line += f"  (governor {g['cause']}: {g['rationale']})"
        else:
            a = d["args"]
            line = (
                f"step {ev['step']}  {d['action']} L{a['level']} ({a['x']},{a['y']})  {ev['cause']}  "
                f"entropy {ev['observation']['entropy']:.2f}  px {ev['pixels_frac']:.1%}"
            )
    elif t == "finding":
        f = ev["finding"]
        line = f"finding {f['id']}  {f['cls']}  state {f['state']}  confidence {f['confidence']}"
        if f.get("critical_step") is not None:
            line += f"  critical from step {f['critical_step']} ({f['critical_area_px']} px^2)"
    elif t == "review_request":
        line = f"review requested for {ev['finding_id']}: {ev['reason']}"
    elif t == "planner":
        line = f"planner: {ev['status']} ({ev['reason']})"
    else:
        line = t
    print(line, file=sys.stderr)


def fixed_cost_job(img, cfg, micro_arm: str, kw: dict):
    """A Job on the published fixed costs, as the evaluations and evals/trace_sample.py run it: every timed op costs
    its p50 from results/<micro_arm>/micro.json on a virtual clock, so the stop rule, the governor and the caps read
    that table, not this machine's load. A repo checkout only: evals/ and results/ are not in the wheel."""
    repo = Path(__file__).resolve().parent.parent
    if not (repo / "evals" / "harness.py").exists():
        raise ValueError("--cost-model fixed needs a repo checkout (evals/ and results/<arm>/micro.json)")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from evals.harness import FixedCostJob, cost_rows, op_costs

    from .segment import default_segmenter

    seg = kw.get("segmenter") or default_segmenter()
    costs = op_costs(cost_rows(micro_arm, seg.name), img.shape[1], img.shape[0])
    return FixedCostJob(img, cfg, costs, **kw)


def cmd_run(a) -> int:
    img = imaging.decode_image(Path(a.image).read_bytes())
    priv = privacy.privacy_pass(img)
    replay, resumed = [], None
    if a.resume:
        trace = read_trace(a.resume)
        replay = [
            [r["decision"]["action"], *(r["decision"]["args"][k] for k in ("x", "y", "level", "size"))]
            for r in trace
            if r.get("type") == "step" and r["decision"]["action"] in REPLAY_ACTIONS
        ]
        resumed = next((r["job"] for r in trace if r.get("type") == "job"), None)

    stop = threading.Event()
    interrupted = []

    def on_sigint(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt  # a second Ctrl-C aborts without waiting for the step
        stop.set()
        interrupted.append(True)
        print("interrupt: halting after this step (Ctrl-C again to abort)", file=sys.stderr)

    cfg = JobConfig(pixel_cap=a.pixel_cap, usd_cap=a.usd_cap, seconds_cap=a.seconds_cap, chaos=a.chaos)
    kw = {
        "kill": lambda: stop.is_set() or kill_path().exists(),
        "replay": replay,
        "planner": planner.from_env(),  # None unless DRISHYAM_PLANNER_URL is set
    }
    job = Job(img, cfg, **kw) if a.cost_model == "live" else fixed_cost_job(img, cfg, a.micro_arm, kw)
    run_dir = Path(a.out) / job.id
    run_dir.mkdir(parents=True)
    old = (
        signal.signal(signal.SIGINT, on_sigint)
        if threading.current_thread() is threading.main_thread()
        else None
    )
    done = None
    try:
        with (run_dir / "trace.jsonl").open("w") as trace_file:
            for ev in job.run():
                if ev["type"] == "job":
                    ev["privacy"] = {k: v for k, v in priv.items() if k != "boxes"}
                    if a.cost_model == "fixed":  # the trace says its times and $ are modelled
                        ev["cost_model"] = {"kind": "fixed", "micro_arm": a.micro_arm}
                    if resumed:
                        ev["resumed_from"] = resumed
                trace_file.write(json.dumps(ev, separators=(",", ":")) + "\n")
                trace_file.flush()
                _progress(ev)
                if ev["type"] == "done":
                    done = ev
    finally:
        if old is not None:
            signal.signal(signal.SIGINT, old)
    wo_path = run_dir / "work_order.json"
    write_json_atomic(wo_path, done["work_order"])
    write_crops(run_dir, job.findings)

    s = done["summary"]
    print(f"job {job.id}: {done['status']} ({done['cause']})")
    print(
        f"steps {s['steps']}  pixels {s['pixels_frac']:.1%}  ${s['usd']}  {s['seconds']} s  "
        f"looked at {done['coverage_looked']:.0%} of the salient area (the survey look not counted)"
    )
    if a.cost_model == "fixed":
        print(f"seconds and $ are modelled from results/{a.micro_arm}/micro.json (--cost-model fixed), not measured")
    print(f"findings {s['findings']}  critical pending review {s['critical_pending']}")
    print(f"work order {wo_path} ({done['work_order']['status']})")
    if na := done["work_order"]["not_assessed"]:
        print(f"not assessed: {len(na)} area{'s' * (len(na) > 1)} listed in the work order; a person should look there")
    if s["critical_pending"]:
        print(
            f"a human decides each critical finding: drishyam review {wo_path} <finding id> approve|reject --by NAME"
        )
    return 130 if interrupted else 0


# ---- batch ----------------------------------------------------------------------------------------
def cmd_batch(a) -> int:
    from . import batch  # it imports this module

    if a.report:
        r = batch.refresh(Path(a.report), batch.signing_key())
        print(f"{r['id']}: {r['status']}; {r['counts']['pending']} of {r['counts']['critical']} critical findings "
              f"pending review; {len(r['not_assessed'])} areas or photos not assessed")
        return 0
    paths = batch.photo_list(a.photos)
    if not paths:
        raise ValueError(f"no JPEG or PNG photos in {' '.join(a.photos) or '(no folder given)'}")
    key = batch.signing_key()
    stop, interrupted = threading.Event(), []

    def on_sigint(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt  # a second Ctrl-C aborts without waiting for the step
        stop.set()
        interrupted.append(True)
        print("interrupt: halting the batch after this step (Ctrl-C again to abort)", file=sys.stderr)

    show = lambda rec: print(f"{rec['decision']}  {Path(rec.get('photo', '')).name}  {rec['cause']}: "
                             f"{rec['rationale']}", file=sys.stderr)
    cfg = JobConfig(pixel_cap=a.photo_pixel_cap, usd_cap=a.photo_usd_cap, seconds_cap=a.photo_seconds_cap)
    rt = None
    if a.router == "nova":
        from . import router

        rt = router.Router(router.Nova())
    main_thread = threading.current_thread() is threading.main_thread()
    old = signal.signal(signal.SIGINT, on_sigint) if main_thread else None
    try:
        bdir = batch.run_batch(paths, Path(a.out), cfg, a.usd_cap, a.seconds_cap,
                               kill=lambda: stop.is_set() or governor.global_kill(), key=key,
                               cost_model=a.cost_model, micro_arm=a.micro_arm, progress=show, router=rt,
                               batch_yield=a.batch_yield)
    finally:
        if old is not None:
            signal.signal(signal.SIGINT, old)
    r = json.loads((bdir / "inspection.json").read_text())
    c = r["counts"]
    print(f"{r['id']}: {r['status']}  {c['photos']} photos: {c['blocked']} blocked, {c['incomplete']} incomplete, "
          f"{c['ready']} ready, {c['not assessed']} not assessed")
    print(f"spent {r['spent']['seconds']:.2f} s (${r['spent']['usd']:.5f}), triage {r['spent']['triage_seconds']:.2f} s"
          + (f"; stopped: {r['stop']['cause']}" if r["stop"] else ""))
    if ro := r.get("router"):
        print(f"router {ro['name']}: {ro['labels']}, {ro['calls']} calls, {ro['cached']} cached, "
              f"{ro['status']['rejected'] + ro['status']['fallback']} fell back; ${ro['usd']:.5f}; {c['cleared']} "
              "cleared photos are listed as not assessed")
    print(f"report {bdir / 'inspection.md'} (signed: {bdir / 'inspection.json'})")
    if c["pending"]:
        print(f"{c['pending']} critical findings wait for a person: drishyam review <work order> <finding id> "
              f"approve|reject --by NAME, then drishyam batch --report {bdir}")
    return 130 if interrupted else 0


def _total(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        v = float("nan")
    if not 0 < v < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive number")
    return v


# ---- review, replay, kill -------------------------------------------------------------------------
def cmd_review(a) -> int:
    path = Path(a.work_order)
    reviewer = findings.clean_reviewer(a.by)
    wo = json.loads(path.read_text())
    match = [f for f in wo["findings"] if f["id"] == a.finding_id]
    if not match:
        raise findings.ReviewError(f"no finding {a.finding_id!r} in {path}")
    findings.decide(match[0], a.decision, actor="human")  # raises unless the finding is pending_review
    wo["approvals"].append(
        {"finding_id": a.finding_id, "decision": a.decision, "reviewer": reviewer, "at": findings.now_iso()}
    )
    wo["status"] = findings.order_status(wo)
    write_json_atomic(path, wo)
    print(f"{a.finding_id} {match[0]['state']} by {reviewer}; work order {wo['status']}")
    return 0


def cmd_replay(a) -> int:
    trace = Path(a.trace)
    path = gaze_path_from_trace(read_trace(trace))
    for i, (x, y, level) in enumerate(path, 1):
        print(f"{i:>3}  x={x} y={y} L{level}")
    wo_file = trace.with_name("work_order.json")
    if not wo_file.exists():
        print(f"{len(path)} looks; no work_order.json beside the trace, so the path is not cross-checked")
        return 0
    if json.loads(wo_file.read_text())["gaze_path"] != path:
        return _fail(f"MISMATCH: the trace's gaze path differs from {wo_file}")
    print(f"{len(path)} looks; the path matches {wo_file.name}")
    return 0


def cmd_kill(_) -> int:
    path = kill_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    print(f"kill switch set: {path}")
    return 0


def cmd_unkill(_) -> int:
    path = kill_path()
    path.unlink(missing_ok=True)
    print(f"kill switch cleared: {path}")
    return 0


# ---- servers --------------------------------------------------------------------------------------
def cmd_mcp(_) -> int:
    try:
        from .mcp_server import main as serve_mcp
    except ModuleNotFoundError as exc:
        return _fail(f"cannot start the MCP server ({exc}); install it with: pip install 'drishyam[mcp]'")
    serve_mcp()
    return 0


def cmd_serve(a) -> int:
    if not 1 <= a.port <= 65535:
        raise ValueError("port must be 1-65535")
    try:
        import uvicorn
    except ModuleNotFoundError:
        return _fail("drishyam serve needs uvicorn: pip install uvicorn")
    os.environ["DRISHYAM_SERVE_STATIC"] = "1"  # web.py mounts public/ when it is imported
    print(f"http://127.0.0.1:{a.port}")
    uvicorn.run("drishyam.web:app", host="127.0.0.1", port=a.port)
    return 0


# ---- parser ---------------------------------------------------------------------------------------
def _cap(name: str):
    lo, hi = CAPS[name]

    def parse(text: str) -> float:
        try:
            v = float(text)
        except ValueError:
            v = float("nan")
        if not lo <= v <= hi:
            raise argparse.ArgumentTypeError(f"must be a number from {lo} to {hi}")
        return v

    return parse


def _chaos(text: str) -> tuple[str, ...]:
    names = tuple(sorted({c for c in text.split(",") if c}))
    if not set(names) <= set(CHAOS):
        raise argparse.ArgumentTypeError(f"must be a comma list of {', '.join(CHAOS)}")
    return names


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drishyam", description="Virtual-PTZ bridge-photo inspector.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser(
        "run", help="inspect one image; writes runs/<job>/trace.jsonl, work_order.json, crops/"
    )
    r.add_argument("image")
    r.add_argument(
        "--pixel-cap", type=_cap("pixel_cap"), default=0.30, help="fraction of L0 pixels (default 0.3)"
    )
    r.add_argument("--usd-cap", type=_cap("usd_cap"), default=0.05, help="modelled $ (default 0.05)")
    r.add_argument("--seconds-cap", type=_cap("seconds_cap"), default=20.0, help="wall seconds (default 20)")
    r.add_argument("--chaos", type=_chaos, default=(), help=f"failure drills: {', '.join(CHAOS)}")
    r.add_argument("--out", default="runs", help="directory for run folders (default runs)")
    r.add_argument(
        "--resume", metavar="TRACE", help="re-execute the decisions of an earlier trace, then continue"
    )
    r.add_argument(
        "--cost-model", choices=("live", "fixed"), default="live",
        help="live: this machine's wall clock (default); fixed: the published per-op costs on a virtual clock, "
        "so the run does not change with load (a repo checkout only)",
    )
    r.add_argument(
        "--micro-arm", default="linux-aarch64", metavar="ARM",
        help="the cost table for --cost-model fixed: results/ARM/micro.json (default linux-aarch64, the published one)",
    )
    r.set_defaults(func=cmd_run)

    b = sub.add_parser(
        "batch", help="inspect a backlog: triage every photo, run the most promising first under one total budget; "
        "writes runs/batch-<id>/inspection.json (signed), inspection.md, batch_trace.jsonl and one run folder per photo"
    )
    b.add_argument("photos", nargs="*", help="folders, photos, or .txt lists of photo paths")
    b.add_argument("--usd-cap", type=_total, help="modelled $ for the whole batch (default: none)")
    b.add_argument("--seconds-cap", type=_total, help="seconds for the whole batch, triage included (default: none)")
    b.add_argument("--photo-pixel-cap", type=_cap("pixel_cap"), default=0.30, help="per photo, as run's --pixel-cap")
    b.add_argument("--photo-usd-cap", type=_cap("usd_cap"), default=0.05, help="per photo, as run's --usd-cap")
    b.add_argument("--photo-seconds-cap", type=_cap("seconds_cap"), default=20.0, help="per photo, as run's --seconds-cap")
    b.add_argument("--out", default="runs", help="directory for the batch folder (default runs)")
    b.add_argument("--cost-model", choices=("live", "fixed"), default="live", help="as run's --cost-model")
    b.add_argument("--micro-arm", default="linux-aarch64", metavar="ARM", help="as run's --micro-arm")
    b.add_argument("--router", choices=("rule", "nova"), default="rule",
                   help="rule (default): triage order only; nova: Amazon Nova Lite on Bedrock labels each overview Clear, "
                   "Requires zoom or Critical (AWS credentials; Clear photos get no job and are listed as not assessed)")
    b.add_argument("--batch-yield", action="store_true",
                   help="under budget pressure, a photo's job ends once its best next look is worth less than the next "
                   "photo's first look (cause batch-yield; what it skipped is listed as not assessed). Off by default")
    b.add_argument("--report", metavar="DIR", help="after reviews: recompute DIR/inspection.json from its work orders")
    b.set_defaults(func=cmd_batch)

    v = sub.add_parser("review", help="a human approves or rejects a pending_review finding")
    v.add_argument("work_order")
    v.add_argument("finding_id")
    v.add_argument("decision", choices=("approve", "reject"))
    v.add_argument("--by", required=True, metavar="NAME", help="reviewer name, 1-64 printable characters")
    v.set_defaults(func=cmd_review)

    y = sub.add_parser("replay", help="print the gaze path of a trace and check it against its work order")
    y.add_argument("trace")
    y.set_defaults(func=cmd_replay)

    sub.add_parser("kill", help="set the global kill switch (halts every job within one step)").set_defaults(
        func=cmd_kill
    )
    sub.add_parser("unkill", help="clear the global kill switch").set_defaults(func=cmd_unkill)
    sub.add_parser("mcp", help="run the MCP server over stdio").set_defaults(func=cmd_mcp)
    s = sub.add_parser("serve", help="run the web app locally (127.0.0.1)")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyError as exc:
        return _fail(f"malformed input file: missing {exc}")
    except USER_ERRORS as exc:
        return _fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
