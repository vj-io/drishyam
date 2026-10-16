import time

import cv2
import numpy as np
import pytest

from drishyam import imaging
from drishyam.agent import Job, JobConfig, gaze_path_from_trace, validate_replay
from drishyam.findings import SPALL_CRITICAL_AREA_PX, ForbiddenTransition, decide
from drishyam.segment import CLASSES, finish, segment_view

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


def test_survey_look_comes_first_only_when_a_second_look_is_affordable(scene):
    """Coarse to fine: one look whose window holds every candidate, then finer looks; no survey on a one-look budget."""
    _, events = run(scene, pixel_cap=0.3)
    first, second = steps(events)[:2]
    assert first["cause"] == "survey" and first["decision"]["args"]["level"] > second["decision"]["args"]["level"]
    x, y, w, h = first["window"]
    H, W = scene.shape[:2]
    for bx, by, bw, bh in (c["box"] for c in events[1]["candidates"]):
        assert x <= max(0, bx) and min(W, bx + bw) <= x + w and y <= max(0, by) and min(H, by + bh) <= y + h
    _, events = run(scene, pixel_cap=0.12)  # the overview plus one 512^2 look
    assert next(s["cause"] for s in steps(events)) != "survey"



def test_evidence_trail_keeps_the_look_that_found_it(scene, monkeypatch):
    """A finer look re-centred on a defect refines its finding; the look that found it stays in the trail and box."""
    from drishyam.findings import Finding
    boxes, init, update = {}, Finding.__post_init__, Finding.agent_update

    def spy_init(f):
        init(f)
        boxes[f.id] = [list(f.box)]

    def spy_update(f, **kw):
        boxes[f.id].append(list(kw["box"]))
        update(f, **kw)

    monkeypatch.setattr(Finding, "__post_init__", spy_init)
    monkeypatch.setattr(Finding, "agent_update", spy_update)
    job, events = run(scene, pixel_cap=1.0)
    looks = {s["step"]: s["decision"] for s in steps(events)}
    assert any(looks[s]["action"] in ("escalate", "verify") and looks[s]["args"]["x"] != looks[f.step]["args"]["x"]
               for f in job.findings for s in f.steps[1:]), "no finding was refined by a re-centred finer look"
    for f in job.findings:
        assert f.steps[0] == f.step and f.steps == sorted(set(f.steps)) and len(boxes[f.id]) == len(f.steps)
        x, y, w, h = f.box
        assert all(x <= bx and y <= by and bx + bw <= x + w and by + bh <= y + h for bx, by, bw, bh in boxes[f.id])
        assert f.level == min(looks[s]["args"]["level"] for s in f.steps)
        flags = [e["finding"]["critical"] for e in events if e["type"] == "finding" and e["finding"]["id"] == f.id]
        assert flags == sorted(flags), "a finding went from critical back to non-critical"
    assert [wf["steps"] for wf in events[-1]["work_order"]["findings"]] == [f.steps for f in job.findings]

def inside_image(box, shape) -> bool:
    x, y, w, h = box
    return 0 <= x and 0 <= y and x + w <= shape[1] and y + h <= shape[0]


def test_survey_window_larger_than_the_image_reports_nothing_outside_it(scene, monkeypatch):
    """The 2048-px survey view of a 2000 x 1500 scene is mostly reflection; no label and no box may come from there."""
    from drishyam import agent

    looks = []

    def spy(segmenter, crop, model, valid):
        looks.append((segment_view(segmenter, crop, model, valid), valid))
        return looks[-1][0]

    monkeypatch.setattr(agent, "segment_view", spy)
    job, events = run(scene, pixel_cap=0.3)
    x, y, w, h = steps(events)[0]["window"]
    assert steps(events)[0]["cause"] == "survey" and (x < 0 or y < 0) and (w > scene.shape[1] or h > scene.shape[0])
    assert looks[0][1].mean() < 0.8  # most of the survey view is not the image
    assert all(not seg.labels[~valid].any() for seg, valid in looks)
    assert job.findings
    for f in events[-1]["work_order"]["findings"]:
        assert inside_image(f["box"], scene.shape), f


class DarkIsSpall:
    """Every dark pixel is spalling, crisply: measured areas are exact, so mirrored area shows up 1:1."""

    def predict(self, crop, model="accurate"):
        dark = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 105
        probs = np.zeros((len(CLASSES), *dark.shape), np.float32)
        probs[0], probs[CLASSES.index("spalling")] = ~dark, dark
        return finish(probs, time.perf_counter(), model)


def test_reflection_does_not_inflate_spall_area_to_critical(monkeypatch):
    """A half-disc spall on the top edge, 30k px^2 inside the image, is mirrored by every view that runs past the
    edge; counted twice it is over the 40k critical area, and criticality can never be taken back."""
    from .conftest import concrete

    img = concrete(900, 1100, 6)
    cv2.circle(img, (550, 0), 140, (40, 40, 40), -1)
    true_area = int(np.count_nonzero(img[..., 0] == 40))
    assert true_area < SPALL_CRITICAL_AREA_PX <= 2 * true_area

    def spalls():
        job = Job(img, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), segmenter=DarkIsSpall())
        list(job.run())
        assert any(r["window"][1] < 0 for r in job.trace if r.get("window"))  # some view does run past the edge
        return [f for f in job.findings if f.cls == "spalling"]

    with monkeypatch.context() as m:  # without the footprint (the old behaviour) the mirror makes it critical
        m.setattr(imaging, "footprint", lambda to_l0, l0_size, size: np.ones((size, size), bool))
        assert any(f.critical for f in spalls())
    found = spalls()
    assert found and not any(f.critical for f in found)
    assert all(f.measure["area_px"] <= 1.02 * true_area and inside_image(f.box, img.shape) for f in found)


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


def test_governor_override_on_a_halting_step_reaches_the_trace(scene):
    """PRD rule 5: every override writes a cause. With eps = 1 the stop rule ends the job on the step right
    after the spike, the same step on which the governor switched the model; the halt record carries it."""
    job, events = run(scene, pixel_cap=1.0, chaos=("p95_spike",), eps=1.0)
    spike = next(s for s in steps(events) if s.get("injected") == "p95_spike")
    halt = steps(events)[-1]
    assert halt["step"] == spike["step"] + 1 and halt["cause"] == "stop-rule"  # the case this test exists for
    assert job.settings.history == ["p95>SLA"] and halt["model"] == "fast"
    assert halt["governor"]["cause"] == "p95>SLA" and "model -> fast" in halt["governor"]["rationale"]
    overrides = [s.get("governor", {}).get("cause") or s["cause"] for s in steps(events)]
    assert overrides.count("p95>SLA") == len(job.settings.history)


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
    wo = {f["id"]: f for f in events[-1]["work_order"]["findings"]}
    for f in critical:
        assert f.critical_step in f.steps and wo[f.id]["critical_step"] == f.critical_step  # which look made it critical
        assert wo[f.id]["critical_area_px"] == f.critical_area_px > 0
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
