"""`drishyam batch`: triage a backlog with the agent's own overview, run the unchanged per-photo Job in that order under
one total budget, and write a signed inspection report that recomputes from the photos' work orders.

The photos are copies of public/samples named so that the folder's order is the reverse of the triage order. Every run
uses the published fixed costs (virtual clock), so durations, caps and budget stops are deterministic."""

import io
import itertools
import json
import os
import shutil
import signal
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from drishyam import batch, cli, findings
from drishyam.agent import Job, JobConfig
from drishyam.findings import SEVERITY

SAMPLES = Path(__file__).resolve().parent.parent / "public" / "samples"
# File order a, b, c, d; their overviews' prior sums are 0 (no candidate), 0.87, 2.11 and 2.22, so triage runs d, c, b, a.
# c and d each hold exposed rebar and a spall: critical findings of two severities.
PHOTOS = {"a_real_01.jpg": "real_01.jpg", "b_bridge_07.jpg": "bridge_07.jpg", "c_bridge_03.jpg": "bridge_03.jpg",
          "d_bridge_05.jpg": "bridge_05.jpg"}
FIXED = ["--photo-pixel-cap", "0.2", "--cost-model", "fixed"]
KEY = "k" * 40


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DRISHYAM_KILL_FILE", str(tmp_path / "KILL"))
    monkeypatch.delenv("DRISHYAM_KILL_SWITCH", raising=False)
    monkeypatch.setenv("DRISHYAM_SIGNING_KEY", KEY)


def backlog(folder: Path, names=PHOTOS) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, src in names.items():
        shutil.copy(SAMPLES / src, folder / name)
    return folder


def run(args, out: Path):
    with redirect_stdout(io.StringIO()) as so, redirect_stderr(io.StringIO()) as se:
        rc = cli.main(["batch", *args, "--out", str(out)])
    dirs = sorted(out.glob("batch-*")) if out.exists() else []
    d = dirs[0] if dirs else None
    return types.SimpleNamespace(rc=rc, dir=d, out=so.getvalue(), err=se.getvalue(),
                                 report=d and json.loads((d / "inspection.json").read_text()),
                                 trace=d and cli.read_trace(d / "batch_trace.jsonl"))


def by_name(rep) -> dict:
    return {Path(p["photo"]).name: p for p in rep["photos"]}


def decisions(trace, kind):
    return [t for t in trace if t["decision"] == kind]


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """One whole batch over the four photos with no total cap: every photo runs, in triage order."""
    tmp = tmp_path_factory.mktemp("batch")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DRISHYAM_KILL_FILE", str(tmp / "never-created"))
        mp.delenv("DRISHYAM_KILL_SWITCH", raising=False)
        mp.setenv("DRISHYAM_SIGNING_KEY", KEY)
        return run([str(backlog(tmp / "photos")), *FIXED], tmp / "runs")


@pytest.fixture
def copy(ran, tmp_path):
    """A private copy of the shared batch, for tests that review or edit it."""
    dest = tmp_path / ran.dir.name
    shutil.copytree(ran.dir, dest)
    return dest


def member(bdir: Path, name: str) -> Path:
    rep = json.loads((bdir / "inspection.json").read_text())
    return bdir / by_name(rep)[name]["run_dir"] / "work_order.json"


def refresh(bdir: Path):
    with redirect_stdout(io.StringIO()) as so, redirect_stderr(io.StringIO()) as se:
        rc = cli.main(["batch", "--report", str(bdir)])
    return rc, so.getvalue() + se.getvalue()


# ---- order ----------------------------------------------------------------------------------------
def test_order_follows_the_triage_score_not_the_file_order(ran):
    assert ran.rc == 0
    photos = by_name(ran.report)
    triaged = decisions(ran.trace, "triaged")
    assert [Path(t["photo"]).name for t in triaged] == sorted(PHOTOS)  # every photo is triaged, in file order
    (ordered,) = decisions(ran.trace, "ordered")
    order = [Path(ran.report["photos"][i]["photo"]).name for i in ordered["order"]]
    assert order == ["d_bridge_05.jpg", "c_bridge_03.jpg", "b_bridge_07.jpg", "a_real_01.jpg"] != sorted(PHOTOS)
    assert [Path(t["photo"]).name for t in decisions(ran.trace, "started")] == order
    assert [photos[n]["rank"] for n in order] == [1, 2, 3, 4]
    for name in order:  # the score is the sum of the priors of the candidates the photo's own job proposed
        p = photos[name]
        overview = next(t for t in cli.read_trace(ran.dir / p["run_dir"] / "trace.jsonl") if t["type"] == "overview")
        assert p["triage"]["candidates"] == len(overview["candidates"])
        assert p["triage"]["prior_sum"] == round(sum(c["prior"] for c in overview["candidates"]), 4)
    assert photos["a_real_01.jpg"]["triage"]["candidates"] == 0
    assert all(t["cause"] and t["rationale"] for t in ran.trace)


def test_every_photo_gets_its_own_work_order_and_the_report_says_where(ran):
    photos = by_name(ran.report)
    for name, p in photos.items():
        wo = json.loads((ran.dir / p["run_dir"] / "work_order.json").read_text())
        assert p["outcome"] == "assessed" and p["status"] == wo["status"], name
    assert photos["c_bridge_03.jpg"]["status"] == photos["d_bridge_05.jpg"]["status"] == "blocked"
    assert ran.report["status"] == "blocked" and ran.report["stop"] is None
    # the zero-candidate photo: a person checks it by eye, so it is listed, never read as clean
    assert any(Path(n["photo"]).name == "a_real_01.jpg" and n["reason"] == "no candidate region was proposed"
               for n in ran.report["not_assessed"])
    md = (ran.dir / "inspection.md").read_text()
    assert "blocked" in md and "Human queue" in md and "Not assessed" in md and findings.DISCLAIMER in md


def test_the_human_queue_holds_every_critical_finding_by_severity_then_confidence(ran):
    queue = ran.report["queue"]
    expected = {(Path(ran.dir / p["run_dir"]).name, f["id"]) for p in ran.report["photos"]
                for f in json.loads((ran.dir / p["run_dir"] / "work_order.json").read_text())["findings"] if f["critical"]}
    assert {(Path(q["work_order"]).parent.name, q["finding"]) for q in queue} == expected and len(queue) >= 4
    keys = [(SEVERITY[q["cls"]], q["confidence"]) for q in queue]
    assert keys == sorted(keys, reverse=True)
    assert len({k[0] for k in keys}) >= 2 and any(a[0] == b[0] for a, b in itertools.pairwise(keys))  # both are used
    assert all(q["state"] == "pending_review" for q in queue)
    assert ran.report["counts"]["pending"] == len(queue)


def test_totals_add_up_triage_and_every_job(ran):
    photos, spent = ran.report["photos"], ran.report["spent"]
    jobs = sum(p["seconds"] for p in photos)
    triage = sum(p["triage"]["seconds"] for p in photos)
    assert spent["triage_seconds"] == pytest.approx(triage) and spent["seconds"] == pytest.approx(jobs + triage)
    assert spent["pixels"] == sum(p["pixels"] + p["triage"]["pixels"] for p in photos)
    assert spent["usd"] == pytest.approx(spent["seconds"] * 0.339 / 3600)


# ---- budget ---------------------------------------------------------------------------------------
def test_budget_stop_hands_the_last_photo_what_is_left_and_lists_the_rest(tmp_path):
    """0.15 s total: triage (4 photos, about 0.06 s) leaves under 0.1 s, so the first photo's job gets that as its
    seconds cap (not its own 20 s), halts after two looks, and the other three are never started."""
    r = run([str(backlog(tmp_path / "photos")), *FIXED, "--seconds-cap", "0.15"], tmp_path / "runs")
    assert r.rc == 0
    (started,) = decisions(r.trace, "started")
    triage = sum(t["seconds"] for t in decisions(r.trace, "triaged"))
    assert started["seconds_cap"] == pytest.approx(0.15 - triage) and started["seconds_cap"] < 0.1
    first = by_name(r.report)["d_bridge_05.jpg"]
    job = cli.read_trace(r.dir / first["run_dir"] / "trace.jsonl")
    assert job[0]["budget"]["seconds_cap"] == pytest.approx(started["seconds_cap"])
    assert job[-1]["cause"] == "cap-hit" and job[-1]["summary"]["steps"] == 2
    (stop,) = decisions(r.trace, "budget-stop")
    assert stop["spent_s"] >= 0.15 and r.report["stop"]["cause"] == "budget-stop"
    for name in ("c_bridge_03.jpg", "b_bridge_07.jpg", "a_real_01.jpg"):
        p = by_name(r.report)[name]
        assert p["outcome"] == "not-reached" and p["status"] == "not assessed" and "run_dir" not in p
        assert any(Path(n["photo"]).name == name and "budget" in n["reason"] for n in r.report["not_assessed"])
    assert r.report["status"] == "blocked"  # the first photo already queued a critical finding


def test_a_usd_total_is_the_same_budget_in_seconds(tmp_path):
    r = run([str(backlog(tmp_path / "photos")), *FIXED, "--usd-cap", str(0.15 * 0.339 / 3600)], tmp_path / "runs")
    (started,) = decisions(r.trace, "started")
    assert r.report["budget"]["seconds_budget"] == pytest.approx(0.15)
    assert len(decisions(r.trace, "budget-stop")) == 1 and started["seconds_cap"] < 0.1


# ---- kill -----------------------------------------------------------------------------------------
def test_kill_between_photos_stops_before_the_next_one(tmp_path, monkeypatch):
    real = batch.run_photo

    def then_kill(*a, **kw):
        out = real(*a, **kw)
        (tmp_path / "KILL").touch()
        return out

    monkeypatch.setattr(batch, "run_photo", then_kill)
    r = run([str(backlog(tmp_path / "photos")), *FIXED], tmp_path / "runs")
    assert r.rc == 0 and len(decisions(r.trace, "started")) == 1
    (killed,) = decisions(r.trace, "killed")
    assert killed["cause"] == "kill-switch" and r.report["stop"]["cause"] == "kill-switch"
    first = by_name(r.report)["d_bridge_05.jpg"]
    assert first["outcome"] == "assessed" and first["cause"] != "kill-switch"  # it finished before the flag was set
    rest = [p for n, p in by_name(r.report).items() if n != "d_bridge_05.jpg"]
    assert all(p["outcome"] == "not-reached" and "kill" in p["reason"] for p in rest)


def test_kill_inside_a_photo_halts_its_job_at_the_next_step(tmp_path, monkeypatch):
    real = Job.run

    def kill_after_first_look(self):
        for ev in real(self):
            yield ev
            if ev["type"] == "step":
                (tmp_path / "KILL").touch()

    monkeypatch.setattr(Job, "run", kill_after_first_look)
    r = run([str(backlog(tmp_path / "photos")), *FIXED], tmp_path / "runs")
    first = by_name(r.report)["d_bridge_05.jpg"]
    done = cli.read_trace(r.dir / first["run_dir"] / "trace.jsonl")[-1]
    assert done["cause"] == "kill-switch" and done["summary"]["steps"] == 1
    assert first["cause"] == "kill-switch" and len(decisions(r.trace, "started")) == 1
    assert decisions(r.trace, "killed")[0]["photo"] == first["photo"]
    # the regions the halt left unlooked are in the batch's not-assessed list, from the photo's own work order
    assert any(n["photo"] == first["photo"] and "kill-switch halted the job" in n["reason"]
               for n in r.report["not_assessed"])
    assert sum(p["outcome"] == "not-reached" for p in r.report["photos"]) == 3


def test_kill_set_before_the_batch_runs_nothing(tmp_path):
    (tmp_path / "KILL").touch()
    r = run([str(backlog(tmp_path / "photos")), *FIXED], tmp_path / "runs")
    assert r.rc == 0 and not decisions(r.trace, "started") and not decisions(r.trace, "triaged")
    assert r.report["stop"]["cause"] == "kill-switch" and r.report["status"] == "incomplete"
    assert all(p["status"] == "not assessed" for p in r.report["photos"])


def test_ctrl_c_halts_the_running_photo_and_still_writes_the_report(tmp_path, monkeypatch):
    real, before, sent = Job.run, signal.getsignal(signal.SIGINT), []

    def interrupt_after_first_look(self):
        for ev in real(self):
            yield ev
            if ev["type"] == "step" and not sent:
                sent.append(True)
                os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(Job, "run", interrupt_after_first_look)
    r = run([str(backlog(tmp_path / "photos")), *FIXED], tmp_path / "runs")
    assert r.rc == 130 and r.report["stop"]["cause"] == "kill-switch"
    assert len(decisions(r.trace, "started")) == 1 and "interrupt" in r.err
    first = by_name(r.report)["d_bridge_05.jpg"]  # Ctrl-C reached the running job, not only the batch loop
    assert first["cause"] == "kill-switch" and first["steps"] == 1
    assert signal.getsignal(signal.SIGINT) is before


# ---- signature and review -------------------------------------------------------------------------
def test_the_report_signature_verifies_and_any_edit_breaks_it(copy, monkeypatch):
    rep = batch.verify_report(copy, KEY.encode())
    assert rep["id"] == copy.name and rep["signature"]
    path = copy / "inspection.json"
    for edit in (lambda r: r["queue"].pop(), lambda r: r["photos"][0].update(status="ready"),
                 lambda r: r.update(status="ready")):
        doc = json.loads(path.read_text())
        edit(doc)
        path.write_text(json.dumps(doc))
        with pytest.raises(findings.SignatureError):
            batch.verify_report(copy, KEY.encode())
        rc, out = refresh(copy)
        assert rc == 1 and "signature" in out
        path.write_text(json.dumps(rep))
    with pytest.raises(findings.SignatureError):
        batch.verify_report(copy, b"x" * 40)  # another key
    monkeypatch.setenv("DRISHYAM_SIGNING_KEY", "y" * 40)
    assert refresh(copy)[0] == 1


def test_reviewing_a_member_work_order_updates_the_batch_report(copy):
    wo = member(copy, "c_bridge_03.jpg")
    pending = [f["id"] for f in json.loads(wo.read_text())["findings"] if f["state"] == "pending_review"]
    assert len(pending) >= 2
    assert cli.main(["review", str(wo), pending[0], "approve", "--by", "A. Engineer"]) == 0
    assert refresh(copy)[0] == 0
    rep = batch.verify_report(copy, KEY.encode())  # re-signed
    q = {(Path(e["work_order"]).parent.name, e["finding"]): e["state"] for e in rep["queue"]}
    assert q[(wo.parent.name, pending[0])] == "approved" and q[(wo.parent.name, pending[1])] == "pending_review"
    assert rep["counts"]["pending"] == len(rep["queue"]) - 1 and rep["status"] == "blocked"
    for f in pending[1:]:
        assert cli.main(["review", str(wo), f, "reject", "--by", "A. Engineer"]) == 0
    assert refresh(copy)[0] == 0
    rep = batch.verify_report(copy, KEY.encode())
    assert by_name(rep)["c_bridge_03.jpg"]["status"] == json.loads(wo.read_text())["status"] != "blocked"
    assert by_name(rep)["d_bridge_05.jpg"]["status"] == "blocked" and rep["status"] == "blocked"
    assert decisions(cli.read_trace(copy / "batch_trace.jsonl"), "refreshed")


def test_a_work_order_edited_around_the_review_is_refused(copy):
    wo = member(copy, "d_bridge_05.jpg")
    original = wo.read_text()
    doc = json.loads(original)
    crit = next(f for f in doc["findings"] if f["state"] == "pending_review")
    crit["state"] = "approved"  # no approval recorded
    wo.write_text(json.dumps(doc))
    rc, out = refresh(copy)
    assert rc == 1 and crit["id"] in out and "review" in out
    doc["findings"] = [f for f in doc["findings"] if f["id"] != crit["id"]]  # a critical finding dropped
    wo.write_text(json.dumps(doc))
    rc, out = refresh(copy)
    assert rc == 1 and crit["id"] in out
    wo.write_text(original)
    # a real review recorded "reject"; the file then says approved: the recorded decision must match the state
    assert cli.main(["review", str(wo), crit["id"], "reject", "--by", "A. Engineer"]) == 0
    doc = json.loads(wo.read_text())
    next(f for f in doc["findings"] if f["id"] == crit["id"])["state"] = "approved"
    wo.write_text(json.dumps(doc))
    rc, out = refresh(copy)
    assert rc == 1 and crit["id"] in out
    wo.write_text(original)
    assert refresh(copy)[0] == 0


def resolve_all(bdir: Path, names, decision="nothing-found"):
    for name in names:
        wo = member(bdir, name)
        doc = json.loads(wo.read_text())
        for f in doc["findings"]:
            if f["state"] == "pending_review":
                assert cli.main(["review", str(wo), f["id"], "approve", "--by", "A. Engineer"]) == 0
        for i, n in enumerate(doc["not_assessed"]):
            if "resolution" not in n:
                assert cli.main(["review", str(wo), f"na{i}", decision, "--by", "A. Engineer"]) == 0


def test_checking_every_entry_by_eye_makes_the_batch_ready(copy):
    """Each entry from a work order names its work order and index; `--report` recomputes with the resolutions, and the
    batch is ready only once every critical finding is decided and every entry is resolved."""
    rep = batch.verify_report(copy, KEY.encode())
    na = rep["not_assessed"]
    assert na and all(Path(copy / n["work_order"]).exists() and "resolution" not in n for n in na)
    for n in na:  # the index is the entry's place in its own work order
        listed = json.loads((copy / n["work_order"]).read_text())["not_assessed"][n["index"]]
        assert listed == {k: v for k, v in n.items() if k not in ("photo", "work_order", "index")}
    resolve_all(copy, ["a_real_01.jpg"], "defect-found")
    assert refresh(copy)[0] == 0
    rep = batch.verify_report(copy, KEY.encode())
    assert by_name(rep)["a_real_01.jpg"]["status"] == "ready" and rep["status"] == "blocked"
    md = (copy / "inspection.md").read_text()
    assert "checked by eye by A. Engineer: defect-found" in md and "drishyam review" in md and "na0" in md
    resolve_all(copy, PHOTOS)
    rc, out = refresh(copy)
    rep = batch.verify_report(copy, KEY.encode())
    assert rc == 0 and rep["status"] == "ready" and "status ready" in out
    assert rep["counts"]["ready"] == len(PHOTOS) and all("resolution" in n for n in rep["not_assessed"])
    first = sum(Path(n["photo"]).name == "a_real_01.jpg" for n in na)  # resolved before the first refresh
    refreshed = decisions(cli.read_trace(copy / "batch_trace.jsonl"), "refreshed")
    assert refreshed[-1]["rationale"].startswith(f"{rep['counts']['critical'] + len(na) - first} decisions")


def test_a_resolution_edited_around_the_review_is_refused(copy):
    """A resolution with no matching recorded one, a not-assessed entry dropped or edited, and a resolution changed
    after it was recorded are each refused; the honest file then recomputes."""
    wo = member(copy, "a_real_01.jpg")
    original = wo.read_text()
    for edit in (lambda d: d["not_assessed"][0].update(resolution={"decision": "nothing-found", "reviewer": "x",
                                                                      "at": "2026-10-22T00:00:00Z", "note": ""}),
                 lambda d: d["not_assessed"].pop(0),
                 lambda d: d["not_assessed"][0].update(reason="looked fine")):
        doc = json.loads(original)
        edit(doc)
        wo.write_text(json.dumps(doc))
        rc, out = refresh(copy)
        assert rc == 1 and "not assessed" in out, out
    wo.write_text(original)
    assert cli.main(["review", str(wo), "na0", "defect-found", "--by", "A. Engineer"]) == 0
    assert refresh(copy)[0] == 0
    doc = json.loads(wo.read_text())
    doc["not_assessed"][0]["resolution"]["decision"] = "nothing-found"  # the recorded resolution says defect-found
    wo.write_text(json.dumps(doc))
    rc, out = refresh(copy)
    assert rc == 1 and "na0" in out


def test_a_report_sealed_before_entries_named_their_work_order_still_recomputes(copy):
    """A 1.1.1 report: its not-assessed entries carry no work_order or index. It recomputes, to the same status."""
    rep = batch.verify_report(copy, KEY.encode())
    old = {**rep, "not_assessed": [{k: v for k, v in n.items() if k not in ("work_order", "index")}
                                   for n in rep["not_assessed"]]}
    batch.write(copy, {k: v for k, v in old.items() if k != "signature"}, KEY.encode())
    assert refresh(copy)[0] == 0
    new = batch.verify_report(copy, KEY.encode())
    assert (new["status"], new["counts"]) == (rep["status"], rep["counts"]) and new["not_assessed"] == rep["not_assessed"]


# ---- inputs ---------------------------------------------------------------------------------------
def test_zero_photos_fail_cleanly(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("not a photo")
    assert cli.main(["batch", str(tmp_path / "empty"), "--out", str(tmp_path / "runs")]) == 1
    assert cli.main(["batch", "--out", str(tmp_path / "runs")]) == 1
    err = capsys.readouterr().err
    assert "no JPEG or PNG photos" in err and "Traceback" not in err and not (tmp_path / "runs").exists()


def test_an_unreadable_photo_is_listed_and_the_rest_still_run(tmp_path):
    folder = backlog(tmp_path / "photos", {"a_real_01.jpg": "real_01.jpg"})
    (folder / "b_broken.jpg").write_bytes(b"\xff\xd8\xff not really a jpeg")
    listing = tmp_path / "list.txt"
    listing.write_text("photos/b_broken.jpg\nphotos/missing.png\nphotos/a_real_01.jpg\n")
    r = run([str(listing), *FIXED], tmp_path / "runs")
    assert r.rc == 0
    photos = by_name(r.report)
    for name in ("b_broken.jpg", "missing.png"):
        assert photos[name]["outcome"] == "unreadable" and photos[name]["status"] == "not assessed"
        assert any(Path(n["photo"]).name == name and n["reason"] == photos[name]["reason"] for n in r.report["not_assessed"])
    assert [t["cause"] for t in decisions(r.trace, "triaged")] == ["unreadable", "unreadable", "overview"]
    assert photos["a_real_01.jpg"]["outcome"] == "assessed" and r.report["status"] == "incomplete"


@pytest.mark.parametrize("on", [False, True])
def test_batch_yield_logs_next_ev_on_every_started_line_and_hands_it_to_the_job(tmp_path, on):
    """--batch-yield at 1.2 s in total (per-photo pixel cap 0.3): before any job has run next_ev is 0 (no mean yet).
    Then the seconds left are fewer than the photos left x the mean job so far, so c's next_ev is the next queued
    photo's (b's) triage prior_max, cited in its rationale, and c's job leaves its photo (batch-yield) with what it did
    not look at listed as not assessed; b's is a's prior_max (0, no candidate) and a is last. Off, the default: next_ev
    0 on every started line and no job yields."""
    r = run([str(backlog(tmp_path / "photos")), "--cost-model", "fixed", "--seconds-cap", "1.2",
             *(["--batch-yield"] if on else [])], tmp_path / "runs")
    assert r.rc == 0
    started, photos = decisions(r.trace, "started"), by_name(r.report)
    prior_max = {t["photo"]: t["prior_max"] for t in decisions(r.trace, "triaged")}
    assert [Path(t["photo"]).name for t in started] == ["d_bridge_05.jpg", "c_bridge_03.jpg", "b_bridge_07.jpg",
                                                        "a_real_01.jpg"]
    nev = [t["next_ev"] for t in started]
    c = photos["c_bridge_03.jpg"]
    if not on:
        assert nev == [0.0] * 4 and all(p["cause"] != "batch-yield" for p in photos.values())
        assert all("batch-yield" not in t["rationale"] for t in started)
        return
    assert nev == [0.0, prior_max[started[2]["photo"]], 0.0, 0.0] and nev[1] > 0
    assert "batch-yield: " in started[1]["rationale"] and f"next_ev {nev[1]:g}" in started[1]["rationale"]
    assert c["cause"] == "batch-yield" and photos["d_bridge_05.jpg"]["cause"] != "batch-yield"
    halt = cli.read_trace(r.dir / c["run_dir"] / "trace.jsonl")[-2]
    assert halt["cause"] == "batch-yield" and f"next_ev {nev[1]:.4g}" in halt["rationale"]
    assert any(n["photo"] == c["photo"] and n["reason"] == "not looked at: batch-yield halted the job first"
               for n in r.report["not_assessed"])


def test_the_batch_yield_replay_matches_a_real_batch_with_the_rule(tmp_path):
    """evals/batch_yield.py replays the batch-yield arm exactly from one full run per photo: at 1.2 s on the four
    samples (c's job yields, a's is cut by the budget), it agrees with run_batch(batch_yield=True) on every job's looks,
    cause and seconds, each next_ev, the critical findings queued and the not-assessed entries."""
    from evals import batch_triage, batch_yield

    folder = backlog(tmp_path / "photos")
    paths = sorted(folder.iterdir())
    photos = [{**batch_yield.photo_task({"id": p.name, "path": str(p), "boxes": []}, make=batch_triage.make),
               "file_index": i} for i, p in enumerate(paths)]
    causes = {}
    for budget in (1.0, 1.2):
        sim = batch_yield.replay(photos, batch_yield.triage_order(photos), budget, True, True, detail=True)
        causes[budget] = [j["cause"] for j in sim["jobs"]]
        bdir = batch.run_batch(paths, tmp_path / f"runs{budget}", JobConfig(thumbs=False), seconds_cap=budget,
                               key=KEY.encode(), cost_model="fixed", micro_arm=batch_triage.MICRO_ARM, batch_yield=True)
        real = json.loads((bdir / "inspection.json").read_text())
        ran = sorted((p for p in real["photos"] if "run_dir" in p), key=lambda p: p["rank"])
        assert [(p["steps"], p["cause"], p["seconds"]) for p in ran] == [
            (j["looks"], j["cause"], j["seconds"]) for j in sim["jobs"]]
        assert [t["next_ev"] for t in decisions(cli.read_trace(bdir / "batch_trace.jsonl"), "started")] == [
            j["next_ev"] for j in sim["jobs"]]
        assert (sim["critical"], sim["not_assessed"], sim["spent_s"]) == (
            real["counts"]["critical"], len(real["not_assessed"]), real["spent"]["seconds"])
    # c yields at both; at 1.0 a's one look leaves nothing of the budget, so the governor halts it after that look
    assert causes[1.0][1] == causes[1.2][1] == "batch-yield" and (causes[1.0][3], causes[1.2][3]) == ("cap-hit", "exhausted")


def test_the_dev_eval_replay_matches_a_real_batch(tmp_path):
    """evals/batch_triage.py replays any order from one full run per photo (a job capped lower takes the same looks
    up to the cap). At 0.8 s the second photo is cut short by the budget: the replay must agree with run_batch on
    which photos ran and on every second spent."""
    from evals import batch_triage

    folder = backlog(tmp_path / "photos")
    paths = sorted(folder.iterdir())
    photos = [{**batch_triage.photo_task({"id": p.name, "path": str(p), "boxes": []}), "file_index": i}
              for i, p in enumerate(paths)]
    order = sorted(range(len(photos)), key=lambda i: (-photos[i]["triage"]["prior_sum"], i))
    sim = batch_triage.replay(photos, order, 0.8, triage=True)
    bdir = batch.run_batch(paths, tmp_path / "runs", JobConfig(thumbs=False), seconds_cap=0.8,
                           key=KEY.encode(), cost_model="fixed", micro_arm=batch_triage.MICRO_ARM)
    real = json.loads((bdir / "inspection.json").read_text())
    assert sim["reached"] == sum("run_dir" in p for p in real["photos"]) == 2
    assert sim["spent_s"] == real["spent"]["seconds"]
    cut = next(p for p in real["photos"] if p["rank"] == 2)
    assert cut["cause"] == "cap-hit" and cut["steps"] < len(photos[order[1]]["post"])  # the budget cut it short


def test_the_signing_key_falls_back_to_an_owner_only_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DRISHYAM_SIGNING_KEY")
    monkeypatch.setenv("HOME", str(tmp_path))
    key = batch.signing_key()
    path = tmp_path / ".drishyam" / "signing_key"
    assert len(key) >= 32 and path.stat().st_mode & 0o777 == 0o600 and batch.signing_key() == key
    monkeypatch.setenv("DRISHYAM_SIGNING_KEY", "short")  # too short to use: the file still decides
    assert batch.signing_key() == key
