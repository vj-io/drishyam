"""drishyam.router and `drishyam batch --router nova`: a fake Bedrock client stands in for Nova Lite, so no test calls
AWS. Its reply for a photo is picked from the measurements it is sent (the overview's prior sum): the four sample
photos of tests/test_batch.py (prior sums 0, 0.87, 2.11, 2.22 in file order a, b, c, d) are labelled
Requires zoom, Critical, Requires zoom and Clear, so the router runs b, c, a and clears d, which holds exposed rebar."""

import io
import itertools
import json
import threading
import time
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import pytest

from drishyam import batch, cli, router
from drishyam.agent import JobConfig
from drishyam.planner import Invalid
from tests.test_batch import KEY, PHOTOS, SAMPLES, backlog, by_name, decisions, isolated  # noqa: F401 -- autouse

CFG = JobConfig(pixel_cap=0.2)
USAGE = {"inputTokens": 1000, "outputTokens": 40}


def label_for(facts: dict) -> str:
    s = facts["prior_sum"]
    return "Clear" if s > 2.15 else "Requires zoom" if s > 1.5 else "Critical" if s > 0.5 else "Requires zoom"


class FakeBedrock:
    """`converse` as the Bedrock runtime answers it. `reply(facts)` gives the text, or an exception to raise."""

    def __init__(self, reply=None, delay: float = 0.0) -> None:
        self.reply = reply or (lambda f: json.dumps({"label": label_for(f), "reason": f"prior sum {f['prior_sum']}"}))
        self.delay, self.calls, self.lock = delay, [], threading.Lock()

    def converse(self, **kw):
        content = kw["messages"][0]["content"]
        facts = json.loads(content[1]["text"])
        with self.lock:
            self.calls.append({"facts": facts, "jpeg": content[0]["image"]["source"]["bytes"], "kw": kw})
        time.sleep(self.delay)
        text = self.reply(facts)
        if isinstance(text, Exception):
            raise text
        return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "usage": dict(USAGE)}


def go(tmp, fake, cache=None, **kw):
    rt = router.Router(router.Nova(fake), cache_dir=cache or tmp / "cache")
    bdir = batch.run_batch(batch.photo_list([str(backlog(tmp / "photos"))]), tmp / "runs", CFG, key=KEY.encode(),
                           cost_model="fixed", router=rt, **kw)
    return bdir, json.loads((bdir / "inspection.json").read_text()), cli.read_trace(bdir / "batch_trace.jsonl")


@pytest.fixture(scope="module")
def routed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("routed")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DRISHYAM_KILL_FILE", str(tmp / "never-created"))
        mp.delenv("DRISHYAM_KILL_SWITCH", raising=False)
        fake = FakeBedrock()
        bdir, rep, trace = go(tmp, fake)
        return types.SimpleNamespace(fake=fake, dir=bdir, report=rep, trace=trace)


# ---- the reply ---------------------------------------------------------------------------------------
def test_validate_accepts_exactly_the_schema():
    assert router.validate({"label": "Critical", "reason": "large spall, bars visible"}) == {
        "label": "Critical", "reason": "large spall, bars visible"}
    for bad in ({"label": "Critical"}, {"label": "Critical", "reason": "x", "confidence": 0.9},
                {"label": "clear", "reason": "x"}, {"label": "Unsure", "reason": "x"}, {"label": "Clear", "reason": ""},
                {"label": "Clear", "reason": "x" * 201}, {"label": "Clear", "reason": "a\nb"},
                {"label": "Clear", "reason": "see <img src=x>"}, {"label": "Clear", "reason": 3},
                ["Clear", "x"], "Clear"):
        with pytest.raises(Invalid):
            router.validate(bad)
    # the whole reply as one ```json block (Nova Lite's habit) is the object; anything around or inside it is not
    assert router.parse_reply('```json\n{\n  "label": "Clear",\n  "reason": "x"\n}\n```\n') == {"label": "Clear",
                                                                                               "reason": "x"}
    for bad in ('Here you go:\n```json\n{"label": "Clear", "reason": "x"}\n```',
                '```json\n{"label": "Clear", "reason": "x"}\n```\nI am sure.',
                '```json\n{"label": "Clear", "reason": "x"}\n```\n```json\n{"label": "Critical", "reason": "y"}\n```',
                '{"label": "Clear", "reason": "x"} {"label": "Critical", "reason": "y"}'):
        with pytest.raises(Invalid):
            router.parse_reply(bad)


def test_a_malformed_reply_falls_back_to_the_rule_triage_with_its_cause():
    r = router.consult(router.Nova(FakeBedrock(lambda f: "Sure! The label is Clear.")), b"jpg", {"prior_sum": 1.0})
    assert (r["status"], r["label"], r["cause"]) == ("rejected", router.FALLBACK, "reply is not JSON")
    assert r["usage"] == USAGE  # a rejected reply was still billed
    r = router.consult(router.Nova(FakeBedrock(lambda f: '{"label": "Fine", "reason": "x"}')), b"jpg", {"prior_sum": 1})
    assert r["status"] == "rejected" and "allow-list" in r["cause"] and r["label"] == "Requires zoom"


def test_a_slow_reply_times_out_and_falls_back():
    t0 = time.perf_counter()
    r = router.consult(router.Nova(FakeBedrock(delay=0.5)), b"jpg", {"prior_sum": 2.2}, timeout_s=0.05)
    assert (r["status"], r["cause"], r["label"]) == ("fallback", "timeout", "Requires zoom")
    assert r["latency_ms"] < 400 and time.perf_counter() - t0 < 0.4


def test_an_error_falls_back_with_its_name():
    class ThrottlingException(Exception):
        pass

    r = router.consult(router.Nova(FakeBedrock(lambda f: ThrottlingException("slow down"))), b"jpg", {"prior_sum": 2.2})
    assert (r["status"], r["cause"], r["label"]) == ("fallback", "ThrottlingException", "Requires zoom")


# ---- map and reduce ---------------------------------------------------------------------------------
def test_the_router_sees_the_l3_overview_and_measurements_only(routed):
    rep = routed.report
    assert len(routed.fake.calls) == 4
    for c in routed.fake.calls:
        ov = cv2.imdecode(np.frombuffer(c["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        w, h = c["facts"]["photo_px"]
        assert max(ov.shape[:2]) <= router.MAX_SIDE and abs(ov.shape[1] - w / 8) <= 1 and abs(ov.shape[0] - h / 8) <= 1
        assert set(c["facts"]) == {"photo_px", "overview_px", "candidates", "prior_sum", "prior_max", "saliency",
                                   "quality"}
        assert c["kw"]["modelId"] == router.MODEL_ID and c["kw"]["inferenceConfig"]["maxTokens"] <= 200
        assert not any(Path(p["photo"]).stem in json.dumps(c["kw"], default=str) for p in rep["photos"])


def test_critical_runs_first_then_requires_zoom_and_clear_gets_no_job(routed):
    rep, trace = routed.report, routed.trace
    started = [Path(t["photo"]).name for t in decisions(trace, "started")]
    assert started == ["b_bridge_07.jpg", "c_bridge_03.jpg", "a_real_01.jpg"]  # triage alone runs d, c, b, a
    p = by_name(rep)
    assert [p[n]["route"]["label"] for n in sorted(p)] == ["Requires zoom", "Critical", "Requires zoom", "Clear"]
    assert decisions(trace, "ordered")[0]["cause"] == "router"


def test_a_cleared_photo_is_never_reported_as_assessed(routed):
    bdir, rep, trace = routed.dir, routed.report, routed.trace
    d = by_name(rep)["d_bridge_05.jpg"]
    assert d["outcome"] == "cleared" and d["status"] == "not assessed" and "run_dir" not in d and d["rank"] is None
    assert d["reason"].startswith("cleared by the router (not looked at closely): prior sum")
    na = [n for n in rep["not_assessed"] if n["photo"] == d["photo"]]
    assert len(na) == 1 and na[0]["reason"] == d["reason"]
    assert rep["counts"]["cleared"] == 1 and rep["status"] != "ready"
    assert not any(q["photo"] == d["photo"] for q in rep["queue"])  # its rebar was never looked at: nothing to queue
    assert [Path(t["photo"]).name for t in decisions(trace, "cleared")] == ["d_bridge_05.jpg"]
    md = (bdir / "inspection.md").read_text()
    assert "1 of them cleared by the router, not looked at closely" in md
    assert "- d_bridge_05.jpg: cleared by the router (not looked at closely)" in md
    assert "| - | d_bridge_05.jpg | Clear |" in md
    # recomputing the report from the work orders (after reviews) keeps it not assessed
    assert batch.refresh(bdir, KEY.encode())["photos"][3]["status"] == "not assessed"


def test_the_router_dollars_are_charged_to_the_batch_budget(routed):
    rep = routed.report
    s, ro = rep["spent"], rep["router"]
    call = USAGE["inputTokens"] * router.PRICE["input"] + USAGE["outputTokens"] * router.PRICE["output"]
    assert ro["usd"] == pytest.approx(4 * call) and s["router_usd"] == pytest.approx(4 * call)
    assert s["router_seconds"] == pytest.approx(4 * call * 3600 / CFG.usd_per_hour)
    jobs = sum(p.get("seconds", 0) for p in rep["photos"])
    assert s["seconds"] == pytest.approx(s["triage_seconds"] + jobs + s["router_seconds"])
    assert (ro["calls"], ro["cached"], ro["status"]["accepted"], ro["labels"]["Clear"]) == (4, 0, 4, 1)


def test_replies_are_cached_by_photo_sha256_and_failures_are_not(tmp_path):
    first = FakeBedrock(lambda f: '{"label": "Clear"}' if f["prior_sum"] == 0 else json.dumps(
        {"label": label_for(f), "reason": "ok"}))
    _, rep1, _ = go(tmp_path / "1", first, cache=tmp_path / "cache")
    assert len(first.calls) == 4 and rep1["router"]["status"]["rejected"] == 1
    a = by_name(rep1)["a_real_01.jpg"]["route"]
    assert (a["status"], a["label"]) == ("rejected", "Requires zoom")
    again = FakeBedrock()
    # the same photos under other names: the cache key is the bytes, not the file name
    names = {f"z{i}_{n}": src for i, (n, src) in enumerate(PHOTOS.items())}
    rt = router.Router(router.Nova(again), cache_dir=tmp_path / "cache")
    bdir = batch.run_batch(batch.photo_list([str(backlog(tmp_path / "2" / "photos", names))]), tmp_path / "2" / "runs",
                           CFG, key=KEY.encode(), cost_model="fixed", router=rt)
    rep2 = json.loads((bdir / "inspection.json").read_text())
    assert [c["facts"]["prior_sum"] for c in again.calls] == [0.0]  # only the photo whose reply was rejected
    assert rep2["router"]["cached"] == 3 and rep2["router"]["usd"] == pytest.approx(
        USAGE["inputTokens"] * router.PRICE["input"] + USAGE["outputTokens"] * router.PRICE["output"])
    routes = [p["route"] for p in rep2["photos"]]
    assert [r["cached"] for r in routes] == [False, True, True, True]
    assert [r["label"] for r in routes] == ["Requires zoom", "Critical", "Requires zoom", "Clear"]
    assert decisions(cli.read_trace(bdir / "batch_trace.jsonl"), "routed")[1]["cause"] == "cached"


@pytest.mark.parametrize("after, why", [(2, "kill switch set during triage"),
                                        (4, "kill switch set while the router ran")])
def test_kill_during_the_map_step_runs_no_job_and_clears_nothing(tmp_path, after, why):
    """The kill switch is read before each photo's triage (4 reads), then before each router reply is taken: set after
    `after` reads, it stops the batch with the cause `why`."""
    reads = itertools.count()
    bdir, rep, trace = go(tmp_path, FakeBedrock(), kill=lambda: next(reads) >= after)
    assert rep["stop"] == {"cause": "kill-switch", "rationale": why} and not decisions(trace, "started")
    assert all(p["outcome"] == "not-reached" for p in rep["photos"]) and rep["counts"]["cleared"] == 0
    assert all("not reached: kill-switch" in p["reason"] for p in rep["photos"])
    assert len(decisions(trace, "triaged")) == min(after, 4)
    assert batch.verify_report(bdir, KEY.encode())["id"] == rep["id"]


def test_kill_during_the_reduce_step_halts_the_running_photo(tmp_path):
    flag = threading.Event()

    def make(img, cfg, kill, job_id=None, **kw):
        if job_id:  # a photo's own job, not its triage
            flag.set()
        return batch.make_job(img, cfg, kill=kill, job_id=job_id, **kw)

    _, rep, trace = go(tmp_path, FakeBedrock(), kill=flag.is_set, make=make)
    p = by_name(rep)
    assert [Path(t["photo"]).name for t in decisions(trace, "started")] == ["b_bridge_07.jpg"]
    assert p["b_bridge_07.jpg"]["cause"] == "kill-switch" and rep["stop"]["cause"] == "kill-switch"
    assert p["d_bridge_05.jpg"]["outcome"] == "cleared"
    assert p["c_bridge_03.jpg"]["outcome"] == p["a_real_01.jpg"]["outcome"] == "not-reached"


def test_rule_is_the_default_and_nova_is_a_flag(tmp_path, monkeypatch):
    assert cli._parser().parse_args(["batch", "x"]).router == "rule"
    bdir = batch.run_batch([SAMPLES], tmp_path / "rule", CFG, key=KEY.encode(), kill=lambda: True)  # no router
    assert "router" not in json.loads((bdir / "inspection.json").read_text())
    fake = FakeBedrock()
    monkeypatch.setattr(router.Nova, "__init__", lambda self, client=None: setattr(self, "client", fake))
    monkeypatch.setenv("DRISHYAM_ROUTER_CACHE", str(tmp_path / "cache"))
    with redirect_stdout(io.StringIO()) as so, redirect_stderr(io.StringIO()):
        rc = cli.main(["batch", str(backlog(tmp_path / "photos")), "--photo-pixel-cap", "0.2", "--cost-model", "fixed",
                       "--router", "nova", "--seconds-cap", "1.5", "--out", str(tmp_path / "runs")])
    rep = json.loads(next((tmp_path / "runs").glob("batch-*/inspection.json")).read_text())
    assert rc == 0 and len(fake.calls) == 4 and rep["router"]["name"] == "nova"
    assert "1 cleared photos are listed as not assessed" in so.getvalue()
    assert len(list((tmp_path / "cache").glob("*.json"))) == 4
    assert batch.verify_report(next((tmp_path / "runs").glob("batch-*")), KEY.encode())["router"] == rep["router"]
