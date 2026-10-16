import io
import json
import os
import shutil
import signal
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from drishyam import cli, mcp_server
from drishyam.agent import gaze_path_from_trace

SAMPLE = Path(__file__).resolve().parent.parent / "public" / "samples" / "bridge_03.jpg"  # has exposed rebar
CAP = ["--pixel-cap", "0.2"]


@pytest.fixture(autouse=True)
def isolated_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("DRISHYAM_KILL_FILE", str(tmp_path / "KILL"))
    monkeypatch.delenv("DRISHYAM_KILL_SWITCH", raising=False)


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """One real run, shared: the output of `drishyam run` on bridge_03 with a small pixel cap."""
    out = tmp_path_factory.mktemp("runs")
    with (
        pytest.MonkeyPatch.context() as mp,
        redirect_stdout(io.StringIO()) as so,
        redirect_stderr(io.StringIO()) as se,
    ):
        mp.setenv("DRISHYAM_KILL_FILE", str(out.parent / "never-created"))
        mp.delenv("DRISHYAM_KILL_SWITCH", raising=False)
        rc = cli.main(["run", str(SAMPLE), *CAP, "--out", str(out)])
    (run_dir,) = out.glob("j_*")
    return types.SimpleNamespace(rc=rc, dir=run_dir, out=so.getvalue(), err=se.getvalue())


@pytest.fixture
def copy(ran, tmp_path):
    """A private copy of the shared run, for tests that edit it."""
    dest = tmp_path / "copy" / ran.dir.name
    shutil.copytree(ran.dir, dest)
    return dest


def pending(wo_path):
    return [f["id"] for f in json.loads(wo_path.read_text())["findings"] if f["state"] == "pending_review"]


def review(wo_path, finding, decision="approve", by="A. Engineer"):
    return cli.main(["review", str(wo_path), finding, decision, "--by", by])


def test_run_writes_trace_work_order_and_crops(ran):
    assert ran.rc == 0
    trace = cli.read_trace(ran.dir / "trace.jsonl")
    assert [t["type"] for t in trace[:2]] == ["job", "overview"] and trace[-1]["type"] == "done"
    assert trace[0]["job"] == ran.dir.name and trace[0]["privacy"]["faces_blurred"] == 0
    steps = [t for t in trace if t["type"] == "step"]
    assert steps and steps[-1]["decision"]["action"] == "halt" and all(t["cause"] for t in steps)
    wo = json.loads((ran.dir / "work_order.json").read_text())
    assert wo["job"] == ran.dir.name and wo["status"] == "blocked" and pending(ran.dir / "work_order.json")
    assert wo == trace[-1]["work_order"]
    assert (ran.dir / "crops" / "f1.jpg").read_bytes()[:3] == b"\xff\xd8\xff"
    assert "critical pending review" in ran.out and "drishyam review" in ran.out
    assert "step 1  gaze" in ran.err and not (ran.dir / "work_order.json.tmp").exists()


def test_replay_matches_the_work_order(ran, copy, capsys):
    assert cli.main(["replay", str(ran.dir / "trace.jsonl")]) == 0
    out = capsys.readouterr().out
    path = gaze_path_from_trace(cli.read_trace(ran.dir / "trace.jsonl"))
    assert path and f"{len(path)} looks; the path matches work_order.json" in out and "x=" in out
    wo = copy / "work_order.json"
    doc = json.loads(wo.read_text())
    doc["gaze_path"][0][0] += 1
    wo.write_text(json.dumps(doc))
    assert cli.main(["replay", str(copy / "trace.jsonl")]) == 1
    assert "MISMATCH" in capsys.readouterr().err
    wo.unlink()
    assert cli.main(["replay", str(copy / "trace.jsonl")]) == 0
    assert "not cross-checked" in capsys.readouterr().out


def test_review_approve_unblocks_the_order(copy, capsys):
    wo = copy / "work_order.json"
    critical = pending(wo)
    assert critical and json.loads(wo.read_text())["status"] == "blocked"
    for i, fid in enumerate(critical, 1):
        assert review(wo, fid) == 0
        doc = json.loads(wo.read_text())
        assert {f["id"]: f["state"] for f in doc["findings"]}[fid] == "approved"
        assert doc["status"] == (
            "ready" if i == len(critical) else "blocked"
        )  # ready only after the last one
    assert [(a["finding_id"], a["decision"], a["reviewer"]) for a in doc["approvals"]] == [
        (fid, "approve", "A. Engineer") for fid in critical
    ]
    assert doc["approvals"][0]["at"].endswith("Z") and not pending(wo)
    assert f"{critical[-1]} approved by A. Engineer; work order ready" in capsys.readouterr().out
    assert not (copy / "work_order.json.tmp").exists()


def test_review_reject_and_the_decision_is_final(copy, capsys):
    wo = copy / "work_order.json"
    critical = pending(wo)[0]
    assert review(wo, critical, "reject") == 0
    doc = json.loads(wo.read_text())
    assert next(f["state"] for f in doc["findings"] if f["id"] == critical) == "rejected"
    assert doc["approvals"][0]["decision"] == "reject" and doc["status"] == (
        "blocked" if pending(wo) else "ready"
    )
    before = wo.read_bytes()
    for again in ("approve", "reject"):  # a second decision, either way, fails and changes nothing
        assert review(wo, critical, again, by="B. Boss") == 1
    assert "not pending_review" in capsys.readouterr().err and wo.read_bytes() == before


def test_review_refuses_drafts_unknown_ids_and_bad_reviewer_names(copy, capsys):
    wo = copy / "work_order.json"
    critical = pending(wo)[0]
    before = wo.read_bytes()
    draft = next(f["id"] for f in json.loads(before)["findings"] if f["state"] == "draft")
    assert review(wo, draft) == 1 and review(wo, "f999") == 1
    for name in ("", "   ", "x" * 65, "bell\x07", "two\nlines"):
        assert review(wo, critical, by=name) == 1, repr(name)
    assert wo.read_bytes() == before
    with pytest.raises(SystemExit) as bad:
        cli.main(["review", str(wo), critical, "close", "--by", "A"])  # only approve or reject exist
    assert bad.value.code == 2
    assert "reviewer must be" in capsys.readouterr().err


def test_review_writes_atomically(copy, monkeypatch):
    wo = copy / "work_order.json"
    critical = pending(wo)[0]
    before = wo.read_bytes()
    replaced = []

    def failing_replace(src, dst):
        replaced.append((Path(src).name, Path(dst).name))
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    assert review(wo, critical) == 1
    assert replaced == [("work_order.json.tmp", "work_order.json")] and wo.read_bytes() == before


def test_resume_reproduces_the_first_steps(ran, tmp_path):
    lines = (ran.dir / "trace.jsonl").read_text().splitlines()
    first_steps = [i for i, line in enumerate(lines) if json.loads(line)["type"] == "step"][:2]
    cut = lines[: first_steps[1] + 1]
    truncated = tmp_path / "cut.jsonl"
    truncated.write_text("\n".join(cut) + "\n" + lines[first_steps[1] + 1][:40])  # the run died mid-line
    out = tmp_path / "resumed"
    assert cli.main(["run", str(SAMPLE), *CAP, "--out", str(out), "--resume", str(truncated)]) == 0
    (new_dir,) = out.glob("j_*")
    new, old = cli.read_trace(new_dir / "trace.jsonl"), cli.read_trace(ran.dir / "trace.jsonl")
    assert new[0]["resumed_from"] == ran.dir.name and new_dir.name != ran.dir.name
    assert gaze_path_from_trace(new)[:2] == gaze_path_from_trace(old)[:2]
    replayed = [t for t in new if t["type"] == "step"][:2]
    assert [t["cause"] for t in replayed] == ["replay", "replay"]
    assert [t["decision"] for t in replayed] == [t["decision"] for t in old if t["type"] == "step"][:2]


def test_kill_file_stops_a_run(tmp_path):
    assert cli.main(["kill"]) == 0 and (tmp_path / "KILL").exists()
    out = tmp_path / "out"
    assert cli.main(["run", str(SAMPLE), "--pixel-cap", "0.05", "--out", str(out)]) == 0
    (run_dir,) = out.glob("j_*")
    done = cli.read_trace(run_dir / "trace.jsonl")[-1]
    assert done["status"] == "halted" and done["cause"] == "kill-switch" and done["summary"]["steps"] == 0
    assert (run_dir / "work_order.json").exists()  # a halted job still leaves its state behind
    assert cli.main(["unkill"]) == 0 and not (tmp_path / "KILL").exists()
    assert cli.main(["run", str(SAMPLE), "--pixel-cap", "0.05", "--out", str(out)]) == 0
    (later,) = (d for d in out.glob("j_*") if d != run_dir)
    assert cli.read_trace(later / "trace.jsonl")[-1]["cause"] != "kill-switch"


def test_kill_file_set_mid_run_halts_within_one_step(tmp_path, monkeypatch):
    real_progress = cli._progress

    def kill_after_first_step(ev):
        real_progress(ev)
        if ev["type"] == "step":
            (tmp_path / "KILL").touch()

    monkeypatch.setattr(cli, "_progress", kill_after_first_step)
    assert cli.main(["run", str(SAMPLE), *CAP, "--out", str(tmp_path / "out")]) == 0
    done = cli.read_trace(next((tmp_path / "out").glob("j_*/trace.jsonl")))[-1]
    assert done["cause"] == "kill-switch" and done["summary"]["steps"] == 1


def test_default_kill_file_lives_in_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("DRISHYAM_KILL_FILE")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli.main(["kill"]) == 0 and (tmp_path / ".drishyam" / "KILL").exists()
    assert cli.kill_path() == tmp_path / ".drishyam" / "KILL"
    assert cli.main(["unkill"]) == 0 and cli.main(["unkill"]) == 0  # clearing twice is fine
    assert not (tmp_path / ".drishyam" / "KILL").exists()


def test_ctrl_c_halts_gracefully_and_keeps_state(tmp_path, monkeypatch, capsys):
    real_progress, before = cli._progress, signal.getsignal(signal.SIGINT)
    sent = []

    def interrupt_after_first_step(ev):
        real_progress(ev)
        if ev["type"] == "step" and not sent:
            sent.append(True)
            os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(cli, "_progress", interrupt_after_first_step)
    assert cli.main(["run", str(SAMPLE), *CAP, "--out", str(tmp_path / "out")]) == 130
    (run_dir,) = (tmp_path / "out").glob("j_*")
    done = cli.read_trace(run_dir / "trace.jsonl")[-1]
    assert (
        done["status"] == "halted"
        and done["summary"]["steps"] == 1
        and (run_dir / "work_order.json").exists()
    )
    assert "interrupt: halting after this step" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGINT) is before


def test_chaos_and_cap_arguments(tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(["run", str(SAMPLE), *CAP, "--chaos", "p95_spike", "--out", str(out)]) == 0
    steps = [t for t in cli.read_trace(next(out.glob("j_*/trace.jsonl"))) if t["type"] == "step"]
    assert any(t.get("injected") == "p95_spike" for t in steps)
    for bad in (
        ["--pixel-cap", "5"],
        ["--pixel-cap", "x"],
        ["--usd-cap", "0"],
        ["--seconds-cap", "99"],
        ["--chaos", "nonsense"],
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main(["run", str(SAMPLE), *bad])
        assert exc.value.code == 2
    capsys.readouterr()


def test_unusable_inputs_fail_cleanly(tmp_path, capsys):
    (tmp_path / "notes.txt").write_text("hello")
    assert cli.main(["run", str(tmp_path / "missing.jpg"), "--out", str(tmp_path)]) == 1
    assert cli.main(["run", str(tmp_path / "notes.txt"), "--out", str(tmp_path)]) == 1
    assert cli.main(["review", str(tmp_path / "notes.txt"), "f1", "approve", "--by", "A"]) == 1
    (tmp_path / "wo.json").write_text("{}")
    assert cli.main(["review", str(tmp_path / "wo.json"), "f1", "approve", "--by", "A"]) == 1
    err = capsys.readouterr().err
    assert err.count("drishyam: ") == 4 and "Traceback" not in err


def test_serve_and_mcp_commands_dispatch(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setenv("DRISHYAM_SERVE_STATIC", "0")  # so the test's own env change is undone afterwards
    monkeypatch.setattr(
        uvicorn, "run", lambda *a, **kw: calls.append((a, kw, os.environ.get("DRISHYAM_SERVE_STATIC")))
    )
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert calls == [(("drishyam.web:app",), {"host": "127.0.0.1", "port": 8123}, "1")]
    assert cli.main(["serve", "--port", "70000"]) == 1 and len(calls) == 1
    monkeypatch.setattr(mcp_server, "main", lambda: calls.append("mcp"))
    assert cli.main(["mcp"]) == 0 and calls[-1] == "mcp"
