import time

import pytest

from drishyam.agent import Job, JobConfig, gaze_path_from_trace, validate_replay
from drishyam.findings import ForbiddenTransition, decide

from .conftest import FakeClock


def run(img, clock=None, **cfg):
    job = Job(img, JobConfig(thumbs=False, **cfg), clock=clock or FakeClock(), job_id="j_test")
    return job, list(job.run())


def steps(events):
    return [e for e in events if e["type"] == "step"]


def test_event_order_and_every_step_has_a_cause(scene):
    _job, events = run(scene)
    assert [e["type"] for e in events[:2]] == ["job", "overview"]
    assert events[-1]["type"] == "done"
    assert all(s["cause"] for s in steps(events))
    looks = [s for s in steps(events) if s["decision"]["action"] != "halt"]
    assert looks, "the agent looked at nothing"
    for s in looks:
        assert set(s) >= {"observation", "runtime", "decision", "rationale", "window", "pixels_frac"}


def test_opencv_output_changes_the_next_tool_call(scene):
    """G2: at least one step whose action follows from an OpenCV measurement of the previous look."""
    _, events = run(scene, pixel_cap=1.0)
    caused = [s for s in steps(events) if s["cause"] in ("entropy>tau", "area<A_min", "verify-new-mask", "iou<0.5")]
    assert caused
    s = caused[0]
    prev = next(p for p in steps(events) if p["step"] == s["step"] - 1)
    assert s["decision"]["args"]["level"] == prev["decision"]["args"]["level"] - 1
    # The finer look is centred on what the coarser look measured, so it sits inside that window.
    px, py, pw, ph = prev["window"]
    assert px <= s["decision"]["args"]["x"] <= px + pw and py <= s["decision"]["args"]["y"] <= py + ph
    assert s["region"] == prev["region"]


def test_deterministic_for_a_fixed_clock(scene):
    a = [(s["decision"], s["cause"]) for s in steps(run(scene)[1])]
    b = [(s["decision"], s["cause"]) for s in steps(run(scene)[1])]
    assert a == b


@pytest.mark.parametrize("cap", [0.05, 0.1, 0.3])
def test_pixel_cap_is_never_exceeded(scene, cap):
    _job, events = run(scene, pixel_cap=cap)
    done = events[-1]
    assert done["summary"]["pixels_frac"] <= cap + 1e-9
    if done["cause"] == "cap-hit":
        assert done["status"] == "halted" and done["unvisited"]


def test_seconds_and_usd_caps_halt():
    from .conftest import concrete

    img = concrete(1200, 1600, 1)
    _, ev = run(img, clock=SlowClock(), seconds_cap=1.0, pixel_cap=1.0)
    assert ev[-1]["cause"] == "cap-hit" and ev[-1]["summary"]["seconds"] >= 1.0
    _, ev = run(img, clock=SlowClock(), usd_cap=0.0001, pixel_cap=1.0, seconds_cap=30)
    assert ev[-1]["cause"] == "cap-hit" and ev[-1]["summary"]["usd"] >= 0.0001


class SlowClock(FakeClock):
    def __call__(self):
        self.t += 0.05
        return self.t


def test_kill_switch_halts_within_one_step(scene):
    killed = {"after": 2}
    job = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(),
              kill=lambda: len(job.gaze_path) >= killed["after"])
    events = list(job.run())
    looks = [s for s in steps(events) if s["decision"]["action"] != "halt"]
    assert len(looks) == 2
    assert events[-1]["status"] == "halted" and events[-1]["cause"] == "kill-switch"


def test_forced_p95_spike_changes_next_steps_model(scene):
    _, events = run(scene, pixel_cap=1.0, chaos=("p95_spike",))
    st = steps(events)
    spike = next(s for s in st if s.get("injected") == "p95_spike")
    after = next(s for s in st if s["step"] == spike["step"] + 1)
    assert after["cause"] == "p95>SLA"
    assert spike["model"] == "accurate" and after["model"] == "fast"


def test_planner_drills_fall_back_and_job_continues(scene):
    _, events = run(scene, pixel_cap=1.0, chaos=("planner_timeout", "planner_malformed"), planner_timeout_s=0.05)
    planner = [e for e in events if e["type"] == "planner"]
    assert [(p["step"], p["status"]) for p in planner] == [(3, "fallback"), (6, "rejected")]
    assert planner[0]["reason"] == "timeout"
    causes = {s["step"]: s["cause"] for s in steps(events)}
    assert causes[3] == "planner-fallback" and causes[6] == "planner-fallback"
    assert events[-1]["type"] == "done"


def test_replay_reproduces_gaze_path_from_trace_alone(scene):
    job, events = run(scene, pixel_cap=0.4)
    trace = steps(events)
    path = gaze_path_from_trace(trace)
    assert path == job.gaze_path
    replay = [[s["decision"]["action"], s["decision"]["args"]["x"], s["decision"]["args"]["y"],
               s["decision"]["args"]["level"]] for s in trace if s["decision"]["action"] != "halt"]
    job2 = Job(scene, JobConfig(thumbs=False, pixel_cap=0.4), clock=FakeClock(), replay=replay)
    ev2 = list(job2.run())
    assert job2.gaze_path[: len(path)] == path
    replayed = [s for s in steps(ev2) if s["cause"] == "replay"]
    assert len(replayed) == len(replay)
    for a, b in zip(trace, replayed, strict=False):
        assert a["observation"] == b["observation"]


def test_replay_input_is_validated(scene):
    for bad in ([["shell", 1, 1, 1]], [["gaze", 1.5, 1, 1]], [["gaze", -1, 1, 1]], [["gaze", 1, 1, 9]],
                [["gaze", 1, 1]], "nope", [["gaze", 1, 1, 1]] * 201, [["gaze", True, 1, 1]]):
        with pytest.raises(ValueError):
            validate_replay(bad, scene.shape)


def test_critical_finding_blocks_work_order_and_agent_cannot_close_it(scene):
    job, events = run(scene, pixel_cap=1.0)
    critical = [f for f in job.findings if f.critical]
    assert critical, "the rebar spall should produce a critical finding"
    assert any(e["type"] == "review_request" for e in events)
    assert events[-1]["status"] in ("awaiting_review", "halted")
    assert events[-1]["work_order"]["status"] == "blocked"
    for f in critical:
        d = f.to_dict()
        with pytest.raises(ForbiddenTransition):
            decide(d, "approve", actor="agent")
        assert f.state == "pending_review"


def test_measurements_are_pixels_only(scene):
    job, _ = run(scene, pixel_cap=1.0)
    for f in job.findings:
        assert set(f.measure) == {"length_px", "width_px", "area_px", "orientation_deg"}


def test_large_upload_starts_fast():
    """FR-1: a 4000 x 3000 original starts its job (first event) in well under 2 s."""
    from .conftest import concrete

    img = concrete(3000, 4000, 2)
    t = time.perf_counter()
    first = next(Job(img, JobConfig(thumbs=False)).run())
    assert first["type"] == "job" and first["width"] == 4000
    assert time.perf_counter() - t < 2.0


def test_small_image_still_runs():
    from .conftest import add_crack, concrete

    img = add_crack(concrete(480, 640, 3), (100, 200), n=120)
    _, events = run(img, pixel_cap=1.0)
    assert events[-1]["type"] == "done"
    assert all(s["decision"]["args"]["level"] == 0 for s in steps(events) if s["decision"]["action"] != "halt")


def _three_cracks():
    from .conftest import add_crack, concrete

    img = concrete(1536, 2048, seed=0)
    add_crack(img, (200, 300), width=4)
    add_crack(img, (1300, 1000), n=200, width=4, seed=7, step=(1, 2))
    return add_crack(img, (1500, 200), n=200, width=4, seed=9, step=(2, 3))


def test_tight_budget_skips_noncritical_verify_with_a_cause():
    """Three 512^2 looks, three crack areas: each look goes to unseen ground, and every skipped verify is traced."""
    img = _three_cracks()
    _, tight = run(img, pixel_cap=0.3)
    looks = [s for s in steps(tight) if s["decision"]["action"] != "halt"]
    assert [s["decision"]["action"] for s in looks] == ["gaze"] * 3 and len({s["region"] for s in looks}) == 3
    noted = [(s, k) for s in steps(tight) for k in s.get("skipped", [])]
    assert noted and all(k["cause"] == "looks≤candidates" and s["rationale"].startswith(k["rationale"])
                         for s, k in noted)
    _, loose = run(img, pixel_cap=1.0)
    assert any(s["cause"] == "verify-new-mask" for s in steps(loose))


def test_critical_masks_are_verified_even_when_the_budget_rule_would_skip(monkeypatch):
    from .conftest import add_crack, add_rebar_spall, concrete

    monkeypatch.setattr(Job, "_verify_crowded_out", lambda self, r, *a: {
        "action": "verify", "region": r.id, "level": 0, "cause": "looks≤candidates", "rationale": "forced"})
    img = add_crack(add_rebar_spall(concrete(1536, 2048, seed=1), (900, 700), r=320), (1500, 200), n=200, width=4,
                    seed=9, step=(2, 3))
    job, events = run(img, pixel_cap=1.0)
    critical = {f.region for f in job.findings if f.critical}
    assert any(s["cause"] == "verify-new-mask" and s["region"] in critical for s in steps(events))
    assert not any(k["region"] in critical for s in steps(events) for k in s.get("skipped", []))
