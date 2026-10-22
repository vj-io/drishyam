"""The MCP server, driven through the protocol (in-memory client and server, full initialize handshake)."""

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from mcp import Client

from drishyam import cli, mcp_server
from drishyam.agent import gaze_path_from_trace

from .conftest import concrete

SAMPLES = Path(__file__).resolve().parent.parent / "public" / "samples"
BRIDGE = SAMPLES / "bridge_03.jpg"  # exposed rebar: a critical finding
TOOLS = {
    "open_image",
    "overview",
    "gaze",
    "enhance",
    "segment",
    "measure",
    "verify",
    "runtime_stats",
    "set_budget",
    "log_finding",
    "draft_work_order",
    "request_human_review",
}

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Everything the server writes or reads lives under tmp_path (or is read-only sample data)."""
    monkeypatch.setenv("DRISHYAM_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DRISHYAM_IMAGE_ROOT", str(SAMPLES))
    monkeypatch.setenv("DRISHYAM_KILL_FILE", str(tmp_path / "KILL"))
    monkeypatch.setenv("DRISHYAM_PIXEL_CAP", "1.0")
    monkeypatch.delenv("DRISHYAM_KILL_SWITCH", raising=False)
    mcp_server.JOBS.clear()
    return tmp_path


class Driver:
    def __init__(self, client):
        self.client, self.used, self.calls = client, set(), 0

    async def raw(self, name, **args):
        self.used.add(name)
        self.calls += name != "open_image"  # every other call lands in the trace (schema errors excepted)
        r = await self.client.call_tool(name, args)
        return r, next((b.text for b in r.content if b.type == "text"), "")

    async def ok(self, name, **args):
        r, text = await self.raw(name, **args)
        assert not r.is_error, text
        return json.loads(text), [b for b in r.content if b.type == "image"]

    async def refused(self, name, containing="", **args):
        r, text = await self.raw(name, **args)
        assert r.is_error and containing in text, text
        return text

    async def open(self, path=BRIDGE):
        return (await self.ok("open_image", path=str(path)))[0]["job"]


@pytest.fixture
async def d():
    async with Client(mcp_server.mcp, mode="legacy") as client:
        yield Driver(client)


def tiny_root(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    cv2.imwrite(str(root / "tiny.jpg"), concrete(300, 400, seed=2))
    return root


def trace_of(tmp_path, job):
    return cli.read_trace(tmp_path / "runs" / job / "trace.jsonl")


def halts_of(tmp_path, job):
    return [t for t in trace_of(tmp_path, job) if t.get("decision", {}).get("action") == "halt"]


async def gaze_with_rebar(d, job):
    """Look at each overview candidate until a segmentation shows exposed rebar."""
    cands = (await d.ok("overview", job=job))[0]["candidates"]
    for c in cands:
        g = (await d.ok("gaze", job=job, x=c["center"][0], y=c["center"][1], level=1))[0]["gaze"]
        if "exposed_rebar" in (await d.ok("segment", job=job, gaze_id=g))[0]["areas"]:
            return g
    pytest.fail("no candidate on bridge_03 segments as exposed rebar")


async def test_tool_list_is_the_allow_list(d):
    names = {t.name for t in (await d.client.list_tools()).tools}
    assert names == TOOLS and len(names) == 12
    assert not any(w in n for n in names for w in ("approve", "reject", "decide", "close", "shell", "write", "resolve",
                                                   "check", "found", "eye", "not_assessed"))
    # and no tool takes an argument that could carry a person's decision or resolution
    params = {p for t in (await d.client.list_tools()).tools for p in t.inputSchema.get("properties", {})}
    assert not params & {"decision", "reviewer", "resolution", "not_assessed", "note", "outcome"}, params


async def test_every_tool_on_a_bridge_photo(d, tmp_path):
    job = await d.open()
    ov, (heat,) = await d.ok("overview", job=job)
    cands = ov["candidates"]
    assert (
        1 <= len(cands) <= 20
        and heat.mime_type == "image/png"
        and base64.b64decode(heat.data)[:4] == b"\x89PNG"
    )

    g, (crop,) = await d.ok("gaze", job=job, x=cands[0]["center"][0], y=cands[0]["center"][1], level=1, yaw=2)
    jpeg = cv2.imdecode(np.frombuffer(base64.b64decode(crop.data), np.uint8), cv2.IMREAD_COLOR)
    assert crop.mime_type == "image/jpeg" and max(jpeg.shape[:2]) <= 512 and g["quality"]["ok"]
    assert g["window"][2:] == [1024, 1024]
    enh, (crop2,) = await d.ok("enhance", job=job, gaze_id=g["gaze"])
    assert enh["gaze"] == g["gaze"] and crop2.mime_type == "image/jpeg"
    fast, _ = await d.ok("segment", job=job, gaze_id=g["gaze"], model="fast")
    assert fast["model"] == "fast" and 0 <= fast["entropy"] <= 1

    gid = await gaze_with_rebar(d, job)  # an accurate segmentation that shows exposed rebar
    m, _ = await d.ok("measure", job=job, gaze_id=gid, cls="exposed_rebar")
    assert m["unit"] == "L0 pixels" and m["measure"]["area_px"] > 0
    v, _ = await d.ok("verify", job=job, gaze_id=gid, why="new exposed_rebar mask at L1;\n confirming at L0")
    assert (
        v["level"] == 0
        and v["against_level"] == 1
        and 0 <= v["iou"] <= 1
        and v["agrees"] == (v["iou"] >= 0.5)
    )

    f, _ = await d.ok("log_finding", job=job, gaze_id=gid, cls="exposed_rebar")
    assert f["finding"]["critical"] and f["finding"]["state"] == "pending_review" and f["awaiting_human"]
    r, _ = await d.ok("request_human_review", job=job, finding_id="f1", reason="rebar  looks\nexposed")
    assert r["recorded"] and r["state"] == "pending_review"
    stats, _ = await d.ok("runtime_stats", job=job)
    assert {"warp", "segment", "saliency"} <= set(stats["p95_ms"]) and stats["budget"]["pixels_used"] > 0
    caps, _ = await d.ok("set_budget", job=job, seconds_cap=10)
    assert caps["seconds_cap"] == 10
    wo, _ = await d.ok("draft_work_order", job=job)
    assert wo["work_order"]["status"] == "blocked"
    assert d.used == TOOLS

    # The finding stays pending and the order blocked, on disk too: no tool can change either.
    on_disk = json.loads((tmp_path / "runs" / job / "work_order.json").read_text())
    assert {x["id"]: x["state"] for x in on_disk["findings"]}["f1"] == "pending_review"
    assert on_disk["status"] == "blocked" and on_disk["approvals"] == []
    assert (tmp_path / "runs" / job / "crops" / "f1.jpg").exists()

    # One trace line per call; its gaze path is the work order's (`drishyam replay` reads it).
    trace = trace_of(tmp_path, job)
    assert trace[0]["type"] == "job" and trace[0]["image"] == "bridge_03.jpg" and len(trace) == 1 + d.calls
    assert trace[-1]["tool"] == "draft_work_order"
    assert gaze_path_from_trace(trace) == on_disk["gaze_path"]
    assert all(t["cause"] and "runtime" in t and "pixels_used" in t for t in trace[1:])
    review = next(t for t in trace if t.get("tool") == "request_human_review")
    assert review["review_request"] == {"finding_id": "f1", "reason": "rebar looks exposed"}
    # The client's `why` is the step's rationale: the trace says what measurement each call answered.
    verify_rec = next(t for t in trace if t.get("tool") == "verify")
    assert verify_rec["rationale"] == "new exposed_rebar mask at L1; confirming at L0"
    assert verify_rec["cause"] == "mcp-call" and verify_rec["verify"]["iou"] == v["iou"]
    assert on_disk["not_assessed"] == [] and wo["work_order"]["not_assessed"] == []


async def test_why_is_optional_bounded_and_printable(d, tmp_path):
    job = await d.open()
    g, _ = await d.ok("gaze", job=job, x=1500, y=1000, level=2, why="prior 0.81, the top overview candidate")
    await d.ok("gaze", job=job, x=1500, y=1000, level=1)  # no why: allowed, rationale stays empty
    r, _ = await d.raw("gaze", job=job, x=1500, y=1000, level=1, why="x" * 201)
    assert r.is_error
    await d.refused("enhance", "printable", job=job, gaze_id=g["gaze"], why="bell\x07")
    await d.ok("set_budget", job=job, seconds_cap=12, why="two looks left; capping the clock")
    looks = [t for t in trace_of(tmp_path, job) if t.get("tool") in ("gaze", "set_budget")]
    assert [t["rationale"] for t in looks] == [
        "prior 0.81, the top overview candidate", "", "two looks left; capping the clock"]


async def test_a_human_decision_survives_a_redraft(d, tmp_path):
    job = await d.open()
    await d.ok("log_finding", job=job, gaze_id=await gaze_with_rebar(d, job), cls="exposed_rebar")
    wo_path = tmp_path / "runs" / job / "work_order.json"
    await d.ok("draft_work_order", job=job)
    assert cli.main(["review", str(wo_path), "f1", "approve", "--by", "A. Engineer"]) == 0
    again, _ = await d.ok("draft_work_order", job=job)  # the agent redrafts; the person's decision must stand
    f1 = next(f for f in again["work_order"]["findings"] if f["id"] == "f1")
    assert f1["state"] == "approved" and again["work_order"]["status"] == "ready"
    assert again["work_order"]["approvals"][0]["reviewer"] == "A. Engineer"


async def test_a_check_by_eye_survives_a_redraft(d, tmp_path, monkeypatch):
    """A person's resolution of a not-assessed entry stands when the agent redrafts, matched by the entry itself; the
    redraft never resolves an entry, and one the person did not resolve stays open."""
    root = tmp_path / "images"
    root.mkdir()
    img = concrete(600, 800, 5)
    img[:180] = 252  # a blown-out band: the overview lists it as not assessed
    cv2.imwrite(str(root / "sky.png"), img)
    monkeypatch.setenv("DRISHYAM_IMAGE_ROOT", str(root))
    job = await d.open(root / "sky.png")
    await d.ok("overview", job=job)
    first = (await d.ok("draft_work_order", job=job))[0]["work_order"]
    assert first["status"] == "incomplete" and len(first["not_assessed"]) == 1
    wo_path = tmp_path / "runs" / job / "work_order.json"
    again = (await d.ok("draft_work_order", job=job))[0]["work_order"]
    assert again["status"] == "incomplete" and "resolution" not in again["not_assessed"][0]  # the agent resolves nothing
    assert cli.main(["review", str(wo_path), "na0", "nothing-found", "--by", "A. Engineer"]) == 0
    again = (await d.ok("draft_work_order", job=job))[0]["work_order"]
    assert again["status"] == "ready" and again["not_assessed"][0]["resolution"]["reviewer"] == "A. Engineer"
    assert again["approvals"][0]["not_assessed"] == 0


async def test_paths_outside_the_roots_are_refused(d, tmp_path, monkeypatch):
    root = tiny_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes((root / "tiny.jpg").read_bytes())
    (root / "link.jpg").symlink_to(outside / "secret.jpg")
    (root / "fake.jpg").write_text("not an image")
    monkeypatch.setenv("DRISHYAM_IMAGE_ROOT", str(root))
    for bad in (
        "../../etc/passwd",
        str(root / ".." / "outside" / "secret.jpg"),
        str(outside / "secret.jpg"),
        str(root / "link.jpg"),
        "/etc/passwd",
        "tiny\x00.jpg",
    ):
        text = await d.refused("open_image", path=bad)
        assert "image roots" in text or "cannot be read" in text, text
    await d.refused("open_image", "not a file", path=str(root))
    await d.refused("open_image", "only JPEG and PNG", path=str(root / "fake.jpg"))
    assert not mcp_server.JOBS and not (tmp_path / "runs").exists()
    assert await d.open(root / "tiny.jpg")  # the same root still opens a real image


async def test_a_view_past_the_border_shows_no_mirrored_crack(d, tmp_path, monkeypatch):
    """A level-1 look at a 800 x 600 image is mostly reflection; a crack 15 px from the left edge must be
    segmented, measured and logged as the crack inside the image only."""
    root = tmp_path / "images"
    root.mkdir()
    img = concrete(600, 800, 5)
    wobble = [(20 + round(5 * np.sin(y / 25)), y) for y in range(40, 561, 8)]  # a crack, not a ruler line
    cv2.polylines(img, [np.array(wobble, np.int32)], False, (55, 55, 58), 3, cv2.LINE_AA)
    cv2.imwrite(str(root / "edge.png"), img)
    monkeypatch.setenv("DRISHYAM_IMAGE_ROOT", str(root))
    job = await d.open(root / "edge.png")
    g, _ = await d.ok("gaze", job=job, x=400, y=300, level=1)
    assert g["window"] == [-112, -212, 1024, 1024]
    seg, _ = await d.ok("segment", job=job, gaze_id=g["gaze"])
    assert 500 < seg["areas"]["crack"] < 2000  # the line, not the line plus its mirror image (about 3800 px)
    f, _ = await d.ok("log_finding", job=job, gaze_id=g["gaze"], cls="crack")
    x, y, w, h = f["finding"]["box"]
    assert 0 <= x <= 15 and 0 <= y and x + w <= 40 and y + h <= 600


async def test_a_straight_line_and_a_bug_hole_are_rejected_with_their_cause(d, tmp_path, monkeypatch):
    """A ruler-straight line is dropped by segment as formwork and comes back (and is traced) as rejected with that
    cause, so log_finding has no crack to log. On bridge_08 a bug hole that segments as crack is refused by
    log_finding as not elongated, as the agent refuses to draft it."""
    from drishyam.segment import PORE_CAUSE, STRAIGHT_CAUSE

    root = tmp_path / "images"
    root.mkdir()
    img = concrete(600, 800, 5)
    cv2.line(img, (380, 60), (390, 560), (55, 55, 58), 3, cv2.LINE_AA)
    cv2.imwrite(str(root / "seam.png"), img)
    monkeypatch.setenv("DRISHYAM_IMAGE_ROOT", f"{root}{os.pathsep}{SAMPLES}")
    job = await d.open(root / "seam.png")
    g, _ = await d.ok("gaze", job=job, x=400, y=300, level=0)
    seg, _ = await d.ok("segment", job=job, gaze_id=g["gaze"])
    (line,) = seg["rejected"]
    assert "crack" not in seg["areas"] and line["cause"] == STRAIGHT_CAUSE and line["residual_px"] < 1.5
    x, y, w, h = line["box"]
    assert 370 <= x and x + w <= 400 and y <= 300 - 200 and y + h >= 300 + 200  # the line, in L0 px
    assert next(t for t in trace_of(tmp_path, job) if t.get("tool") == "segment")["rejected"] == [line]
    await d.refused("log_finding", "no crack in the last segmentation", job=job, gaze_id=g["gaze"], cls="crack")

    job = await d.open(SAMPLES / "bridge_08.jpg")
    g, _ = await d.ok("gaze", job=job, x=2377, y=925, level=0)
    assert "crack" in (await d.ok("segment", job=job, gaze_id=g["gaze"]))[0]["areas"]
    await d.refused("log_finding", PORE_CAUSE, job=job, gaze_id=g["gaze"], cls="crack")


async def test_verify_compares_the_same_class_and_names_a_class_change(d, tmp_path, monkeypatch):
    """verify compares the coarser look's class with the finer look's mask of the SAME class: a spall seen as rust
    at the same place is a class change (iou 0, iou_any ~1), not agreement and not a disagreement about where."""
    from .test_agent import Flip, disc_scene

    root = tmp_path / "images"
    root.mkdir()
    cv2.imwrite(str(root / "disc.png"), disc_scene())
    monkeypatch.setenv("DRISHYAM_IMAGE_ROOT", str(root))
    monkeypatch.setattr(mcp_server, "default_segmenter", lambda: Flip("spalling", "corrosion"))
    job = await d.open(root / "disc.png")
    g, _ = await d.ok("gaze", job=job, x=550, y=450, level=1)
    assert (await d.ok("segment", job=job, gaze_id=g["gaze"]))[0]["best"] == "spalling"
    v, _ = await d.ok("verify", job=job, gaze_id=g["gaze"])
    assert (v["cls"], v["class_now"], v["iou"], v["kind"], v["agrees"]) == ("spalling", "corrosion", 0.0, "class-change", False)
    assert v["iou_any"] >= 0.9
    rec = next(t for t in trace_of(tmp_path, job) if t.get("tool") == "verify")["verify"]
    assert rec == {"iou": 0.0, "class": "spalling", "against_level": 1, "class_now": "corrosion", "iou_any": v["iou_any"]}
    monkeypatch.setattr(mcp_server, "default_segmenter", lambda: Flip("spalling", "spalling"))
    job = await d.open(root / "disc.png")
    g, _ = await d.ok("gaze", job=job, x=550, y=450, level=1)
    await d.ok("segment", job=job, gaze_id=g["gaze"])
    v, _ = await d.ok("verify", job=job, gaze_id=g["gaze"])
    assert v["kind"] == "agree" and v["iou"] == v["iou_any"] >= 0.9 and v["agrees"]


async def test_a_blown_out_band_is_not_assessed_once_the_overview_ran(d, tmp_path, monkeypatch):
    """The overview gives a large blown-out area no candidates, so the work order lists it from the overview."""
    root = tmp_path / "images"
    root.mkdir()
    img = concrete(600, 800, 5)
    img[:180] = 252
    cv2.imwrite(str(root / "sky.png"), img)
    monkeypatch.setenv("DRISHYAM_IMAGE_ROOT", str(root))
    job = await d.open(root / "sky.png")
    assert (await d.ok("draft_work_order", job=job))[0]["work_order"]["not_assessed"] == []  # no overview yet
    cands = (await d.ok("overview", job=job))[0]["candidates"]
    assert all(c["center"][1] > 180 for c in cands)
    (sky,) = (await d.ok("draft_work_order", job=job))[0]["work_order"]["not_assessed"]
    assert sky["step"] == 0 and sky["reason"] == "glare" and sky["box"][:3] == [0, 0, 800] and 176 <= sky["box"][3] <= 184


async def test_a_cap_can_be_lowered_never_raised(d, tmp_path):
    job = await d.open()
    first, _ = await d.ok("set_budget", job=job, pixel_cap=0.5, usd_cap=0.01, seconds_cap=15)
    assert first["pixels_cap"] == 3_375_000 and first["usd_cap"] == 0.01 and first["seconds_cap"] == 15
    await d.ok("set_budget", job=job, pixel_cap=0.5)  # the same value is not a raise
    for raised in ({"pixel_cap": 0.9}, {"usd_cap": 0.02}, {"seconds_cap": 16}):
        await d.refused("set_budget", "only a human may raise", job=job, **raised)
    await d.refused("set_budget", "only a human may raise", job=job, pixel_cap=0.4, seconds_cap=25)
    after, _ = await d.ok(
        "runtime_stats", job=job
    )  # the refused mixed call changed nothing, not even the lower cap
    assert after["budget"]["pixels_cap"] == 3_375_000
    for bad in ({"pixel_cap": 0.001}, {"pixel_cap": 1.5}, {"usd_cap": 1}, {"seconds_cap": 100}, {}):
        r, _ = await d.raw("set_budget", job=job, **bad)
        assert r.is_error
    # A lowered cap halts looking at once, but a capped job can still be written up.
    await d.ok("gaze", job=job, x=100, y=100, level=1)  # 512^2 px used: more than the 0.02 cap below allows
    await d.ok("set_budget", job=job, pixel_cap=0.02)
    await d.refused("gaze", "cap-hit", job=job, x=100, y=100, level=1)
    for name in ("runtime_stats", "draft_work_order"):  # bookkeeping still runs, and its reply says a cap was hit
        reply, _ = await d.ok(name, job=job)
        assert reply["governor"]["cause"] == "cap-hit", name
    halt = halts_of(tmp_path, job)
    assert halt and halt[0]["cause"] == "cap-hit" and halt[0]["tool"] == "gaze"


async def test_kill_switch_halts_the_next_call(d, tmp_path, monkeypatch):
    job = await d.open()
    await d.ok("overview", job=job)
    kill = tmp_path / "KILL"
    kill.touch()
    for name, args in (
        ("runtime_stats", {}),
        ("overview", {}),
        ("draft_work_order", {}),
        ("gaze", {"x": 10, "y": 10, "level": 2}),
    ):
        await d.refused(name, "kill-switch", job=job, **args)
    await d.refused("open_image", "kill-switch", path=str(BRIDGE))
    kill.unlink()
    await d.ok("runtime_stats", job=job)  # the job was halted, not destroyed
    monkeypatch.setenv("DRISHYAM_KILL_SWITCH", "1")
    await d.refused("runtime_stats", "kill-switch", job=job)
    halts = halts_of(tmp_path, job)
    assert len(halts) == 5 and {h["cause"] for h in halts} == {"kill-switch"}


async def test_bad_arguments_are_refused_and_traced(d, tmp_path):
    job = await d.open()
    await d.refused("runtime_stats", "unknown job", job="j_nope")
    await d.refused("segment", "unknown gaze id", job=job, gaze_id="g9")
    await d.refused("gaze", "inside the 3000x2250 image", job=job, x=3000, y=0, level=1)
    for args in ({"level": 4}, {"level": -1}, {"x": -5}, {"yaw": 90}, {"pitch": -46}, {"x": 1.5}):
        r, _ = await d.raw("gaze", job=job, **{"x": 1500, "y": 1000, "level": 1, **args})
        assert r.is_error
    r, _ = await d.raw("segment", job=job, gaze_id="g1", model="slow")
    assert r.is_error
    look, _ = await d.ok("gaze", job=job, x=1500, y=1000, level=0)
    g = look["gaze"]
    await d.refused("measure", "call segment first", job=job, gaze_id=g, cls="crack")
    await d.refused("log_finding", "call segment first", job=job, gaze_id=g, cls="crack")
    await d.refused("verify", "call segment first", job=job, gaze_id=g)
    await d.ok("segment", job=job, gaze_id=g)
    await d.refused("verify", "level 0", job=job, gaze_id=g)
    r, _ = await d.raw("measure", job=job, gaze_id=g, cls="shell")
    assert r.is_error
    await d.refused("request_human_review", "unknown finding", job=job, finding_id="f1", reason="x")
    trace = trace_of(tmp_path, job)
    errors = [t for t in trace if t["type"] == "error"]
    assert len(errors) == 7 and all(e["error"] for e in errors)
    x, y, w, _ = look["window"]
    assert gaze_path_from_trace(trace) == [[x + w // 2, y + w // 2, 0]]  # failed calls are not looks


async def test_a_slow_segmenter_makes_the_governor_downgrade_the_model(d):
    job = await d.open()
    g = (await d.ok("gaze", job=job, x=1840, y=1456, level=1))[0]["gaze"]
    mcp_server.JOBS[job].stats.record(
        "segment:accurate", 900.0
    )  # the p95 spike drill, as in the web chaos mode
    seg, _ = await d.ok("segment", job=job, gaze_id=g, model="accurate")
    assert seg["model"] == "fast" and seg["governor"]["cause"] == "p95>SLA"
    stats, _ = await d.ok("runtime_stats", job=job)
    assert stats["settings"]["model"] == "fast" and stats["settings"]["governor_history"] == ["p95>SLA"]


async def test_starting_caps_are_the_humans_setting(d, monkeypatch, tmp_path):
    monkeypatch.setenv("DRISHYAM_PIXEL_CAP", "0.1")
    info, _ = await d.ok("open_image", path=str(BRIDGE))
    assert info["budget"]["pixel_cap_frac"] == 0.1 and info["budget"]["usd_cap"] == 0.05
    monkeypatch.setenv("DRISHYAM_SECONDS_CAP", "999")
    await d.refused("open_image", "DRISHYAM_SECONDS_CAP", path=str(BRIDGE))


async def test_oldest_job_is_dropped_beyond_the_limit(d, tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "MAX_JOBS", 2)
    tiny = tiny_root(tmp_path) / "tiny.jpg"
    monkeypatch.setenv("DRISHYAM_IMAGE_ROOT", str(tiny.parent))
    jobs = [await d.open(tiny) for _ in range(3)]
    await d.refused("runtime_stats", "unknown job", job=jobs[0])
    await d.ok("runtime_stats", job=jobs[2])
    assert os.path.isdir(tmp_path / "runs" / jobs[0])  # its trace is still on disk
