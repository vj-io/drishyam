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


LOOK_CAUSES = {"survey", "best-score", "entropy>tau", "area<A_min", "verify-new-mask", "iou<0.5"}


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


@pytest.mark.parametrize("cap", [0.3, 1.0])
def test_deterministic_for_a_fixed_clock(scene, cap):
    """Same image, same clock: the same decisions, causes and findings, including the survey look and its trail."""
    def once():
        job, events = run(scene, pixel_cap=cap)
        return ([(s["decision"], s.get("window"), s["cause"], s.get("governor")) for s in steps(events)],
                [f.to_dict() for f in job.findings], events[-1]["cause"])

    a = once()
    assert a[0][0][2] == "survey" and a[1]  # the survey fires on this scene, and something is found
    assert a == once()


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

    def spy(segmenter, crop, model, valid, scale):
        looks.append((segment_view(segmenter, crop, model, valid, scale), valid))
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


def test_seconds_and_usd_caps_halt(scene):
    _, ev = run(scene, clock=SlowClock(), seconds_cap=1.0, pixel_cap=1.0)
    assert ev[-1]["cause"] == "cap-hit" and ev[-1]["summary"]["seconds"] >= 1.0
    _, ev = run(scene, clock=SlowClock(), usd_cap=0.0001, pixel_cap=1.0, seconds_cap=30)
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
    # The override is recorded under `governor`; the look keeps its own (OpenCV or rule) cause.
    assert after["governor"]["cause"] == "p95>SLA" and after["cause"] in LOOK_CAUSES
    assert "model -> fast" in after["governor"]["rationale"] and "p95" not in after["rationale"]
    assert spike["model"] == "accurate" and after["model"] == "fast"


def test_governor_override_on_a_halting_step_reaches_the_trace(scene):
    """PRD rule 5: every override writes a cause. With eps = 0.7 the stop rule ends the job on the step right
    after the spike, the same step on which the governor switched the model; the halt record carries it."""
    job, events = run(scene, pixel_cap=1.0, chaos=("p95_spike",), eps=0.7)
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
    by_step = {s["step"]: s for s in steps(events)}
    # The planner's failure is the step's planner_status; its cause stays the OpenCV reason for that look.
    assert by_step[3]["planner_status"] == {"status": "fallback", "reason": "timeout"}
    assert by_step[6]["planner_status"]["status"] == "rejected" and "allow-list" in by_step[6]["planner_status"]["reason"]
    assert by_step[3]["cause"] in LOOK_CAUSES and by_step[6]["cause"] in LOOK_CAUSES
    assert not any("planner_status" in s for n, s in by_step.items() if n not in (3, 6))
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


def test_planner_cannot_undo_a_governor_downgrade(scene):
    """A planner consulted right after a p95>SLA downgrade asks for the accurate model: refused, and said so."""
    job = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0, chaos=("p95_spike",)), clock=FakeClock(),
              planner=lambda d: '{"strategy": "switch_model", "params": {"model": "accurate"}, "rationale": "x"}')
    events = list(job.run())
    st = steps(events)
    after = next(s for s in st if s["step"] == 5)
    assert after["governor"]["cause"] == "p95>SLA" and after["model"] == "fast"
    refused = [e for e in events if e["type"] == "planner" and e["step"] == 5]
    assert refused and refused[0]["status"] == "rejected" and "p95>SLA" in refused[0]["reason"]


def test_planner_look_change_keeps_the_chosen_looks_cause(scene):
    """change_level queues its own look; the look already chosen on that step keeps its rule cause."""
    plan = '{"strategy": "change_level", "params": {"region_id": "r2", "level": 0}, "rationale": "PLANNER"}'
    job = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), planner=lambda d: plan)
    st = steps(list(job.run()))
    consulted = next(s for s in st if s["step"] == 5)
    assert not consulted["cause"].startswith("planner:")
    assert any(s["cause"] == "planner:change_level" and s["region"] == "r2" for s in st)


def test_replay_carries_the_gaze_size_the_governor_set(scene, monkeypatch):
    """After a warp spike shrinks the gaze, replaying [action, x, y, level, size] reproduces the path exactly."""
    from drishyam import imaging

    clock, real, holder = FakeClock(), imaging.gaze, {}

    def slow_gaze(*a, **k):
        if holder["job"].step == 5:
            clock.t += 0.05  # one slow warp
        return real(*a, **k)

    monkeypatch.setattr("drishyam.agent.imaging.gaze", slow_gaze)
    job = holder["job"] = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=clock)
    trace = steps(list(job.run()))
    monkeypatch.setattr("drishyam.agent.imaging.gaze", real)
    assert any(s.get("governor", {}).get("cause") == "warp-spike" for s in trace) and job.settings.size < 512
    replay = [[s["decision"]["action"], *(s["decision"]["args"][k] for k in ("x", "y", "level", "size"))]
              for s in trace if s["decision"]["action"] != "halt"]
    again = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), replay=replay)
    list(again.run())
    assert again.gaze_path[: len(job.gaze_path)] == job.gaze_path
    with pytest.raises(ValueError):
        validate_replay([["gaze", 10, 10, 1, 9999]], scene.shape)


def test_killed_job_resumes_into_the_same_job(scene):
    full = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock())
    list(full.run())
    for k in (1, 2, 3):
        holder = {}
        cut = holder["job"] = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(),
                                  kill=lambda h=holder, k=k: len(h["job"].gaze_path) >= k)
        trace = steps(list(cut.run()))
        replay = [[s["decision"]["action"], *(s["decision"]["args"][x] for x in ("x", "y", "level", "size"))]
                  for s in trace if s["decision"]["action"] != "halt"]
        resumed = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), replay=replay)
        list(resumed.run())
        assert resumed.gaze_path == full.gaze_path
        assert sorted((f.cls, f.critical) for f in resumed.findings) == sorted((f.cls, f.critical) for f in full.findings)


def test_every_argmax_rationale_states_lambda(scene):
    """The stop rule is visible on every chosen look: its score and the lambda it had to beat."""
    _, events = run(scene, pixel_cap=1.0)
    picked = [s for s in steps(events) if s["cause"] == "best-score"]
    assert picked
    for s in picked:
        score = float(s["rationale"].split("score ")[1].split(":")[0])
        lam = float(s["rationale"].split("lambda ")[1])
        assert score >= lam


def test_a_failed_verification_is_unconfirmed_and_sent_to_a_human(scene):
    """IoU < 0.5 with no further look to settle it: the finding says `verified: false` and a human is asked."""
    job, events = run(scene, pixel_cap=1.0)
    wo = {f["id"]: f for f in events[-1]["work_order"]["findings"]}
    checked = [f for f in job.findings if f.verify_iou is not None]
    assert checked and all(f.verified is (f.verify_iou >= 0.5) for f in checked)
    assert all(wo[f.id]["verify_iou"] == f.verify_iou and wo[f.id]["verified"] is f.verified for f in checked)
    asked = [e for e in events if e["type"] == "review_request" and e["reason"].startswith("unconfirmed: ")]
    assert asked, "the scene has a look that disagrees with its coarser look at L0"
    for e in asked:
        f = next(f for f in job.findings if f.id == e["finding_id"])
        assert f.verified is False and f"iou {f.verify_iou:g} < 0.5 between L" in e["reason"]
        assert f.state == ("pending_review" if f.critical else "draft")  # a request changes no state
    assert all(f.verify_iou is None and f.verified is None for f in job.findings if f not in checked)


def test_a_check_of_an_absorbed_class_counts_for_the_finding_that_absorbed_it():
    """bridge_03: a look's best mask can be a class absorbed into another finding (rust on exposed bars, a crack in
    a spall's rim). Its IoU is evidence about that finding: the finding carries it, and the disagreement review
    names it, so the review and the finding's `verified` never contradict each other."""
    from pathlib import Path

    img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / "bridge_03.jpg"))
    job, events = run(img)
    ids = {f.id: f for f in job.findings}

    def after(s):  # the finding and review events that follow a step record
        rest = events[events.index(s) + 1:]
        return rest[:next(i for i, e in enumerate(rest) if e["type"] not in ("finding", "review_request"))]

    disagreements = [(e, s) for s in steps(events) if s.get("verify") for e in after(s)
                     if e["type"] == "review_request" and f"iou {s['verify']['iou']:g} < 0.5" in e["reason"]]
    assert {e["reason"].startswith("unconfirmed: ") for e, _ in disagreements} == {True, False}
    for e, s in disagreements:
        f = ids[e["finding_id"]]
        assert f.verified is False and f.verify_iou == pytest.approx(s["verify"]["iou"])
    # the L0 verify whose only confident classes were absorbed (bridge_03 step 4) still reaches a human
    s = next(s for s in steps(events) if s["decision"]["action"] == "verify" and s["decision"]["args"]["level"] == 0
             and s["verify"]["iou"] < 0.5)
    (fe,) = [e["finding"] for e in after(s) if e["type"] == "finding"]
    assert fe["verified"] is False and fe["verify_iou"] == pytest.approx(s["verify"]["iou"])
    assert any(e["type"] == "review_request" and e["finding_id"] == fe["id"] and e["reason"].startswith("unconfirmed: ")
               for e in after(s))


def test_review_requests_say_their_reason_in_fields_too():
    """bridge_03 asks for all four kinds of review; each event's iou, levels and classes match its reason."""
    from pathlib import Path

    img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / "bridge_03.jpg"))
    _, events = run(img)
    reqs = [e for e in events if e["type"] == "review_request"]
    kinds = set()
    for e in reqs:
        assert 0 <= e["level_b"] <= 3
        if e["iou"] is not None:
            assert f"iou {e['iou']:g} < 0.5 between L{e['level_a']} and L{e['level_b']}" in e["reason"]
            assert e["level_a"] > e["level_b"] and len(e["classes"]) == 2
            kinds.add("unconfirmed" if e["reason"].startswith("unconfirmed: ") else "disagree")
        elif e["reason"].startswith("ambiguous: "):
            kept, absorbed = e["classes"]
            assert e["reason"].startswith(f"ambiguous: {absorbed} ") and f"inside {kept} " in e["reason"]
            kinds.add("ambiguous")
        else:
            assert e["level_a"] is None and e["reason"] == f"{e['classes'][0]} is critical"
            kinds.add("critical")
    assert kinds == {"critical", "ambiguous", "disagree", "unconfirmed"}


def test_the_clean_concrete_sample_drafts_no_crack_on_its_formwork_groove():
    """bridge_08 has one straight formwork groove, which used to be drafted as a 93 % crack ([2049, 1250, 581, 798]
    in run j_d60376). Every look that sees it now logs it as a straight line, and no finding comes from it: the
    groove is the only line on the photo longer than 100 px."""
    from pathlib import Path

    from drishyam.segment import STRAIGHT_CAUSE

    img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / "bridge_08.jpg"))
    job, events = run(img)
    rejected = [x for s in steps(events) for x in s.get("rejected", [])]
    assert rejected and all(x["cause"] == STRAIGHT_CAUSE and x["residual_px"] < 1.5 for x in rejected)
    assert all(imaging.box_iou(x["box"], [2049, 1250, 581, 798]) > 0 for x in rejected)
    assert not any(f.cls == "crack" and f.measure["length_px"] > 100 for f in job.findings)


def test_review_on_ambiguity_when_a_class_is_absorbed_by_another(scene):
    """Two classes clear A_min on one spot and the agent keeps the dominant one: a human is asked, once."""
    job, events = run(scene, pixel_cap=1.0)
    asked = [e for e in events if e["type"] == "review_request" and e["reason"].startswith("ambiguous: ")]
    assert asked
    ids = {f.id: f for f in job.findings}
    for e in asked:
        f = ids[e["finding_id"]]
        assert f.cls in e["reason"] and f"is mostly inside {f.cls} {f.id}" in e["reason"]
    assert len({e["reason"].split(" px")[0] + e["finding_id"] for e in asked}) == len(asked)


def test_quality_refused_windows_are_not_assessed(scene, monkeypatch):
    """A look the quality gate refuses is written to the work order with its reason and box, inside the image."""
    from drishyam import agent

    real, calls = imaging.quality, []

    def glare_on_second_look(crop):
        calls.append(1)
        q = real(crop)
        return {**q, "ok": False, "reason": "glare"} if len(calls) == 2 else q

    monkeypatch.setattr(agent.imaging, "quality", glare_on_second_look)
    _, events = run(scene, pixel_cap=0.3)
    refused = [s for s in steps(events) if s.get("quality", {}).get("ok") is False]
    (na,) = events[-1]["work_order"]["not_assessed"]
    assert len(refused) == 1 and na["step"] == refused[0]["step"] == 2
    assert na["reason"] == "glare" and na["action"] == "re-shoot" and inside_image(na["box"], scene.shape)
    x, y, w, h = refused[0]["window"]
    bx, by, bw, bh = na["box"]
    assert x <= bx and y <= by and bx + bw <= x + w and by + bh <= y + h


# ---- ablation switches (FR-21): one test each ----------------------------------------------------------
def test_switch_use_uncertainty_drops_the_entropy_term(scene):
    job, _ = run(scene, pixel_cap=0.3)
    r = next(iter(job.regions.values()))
    r.entropy = 0.8
    with_u = job._score(r)
    job.cfg.use_uncertainty = False
    without = job._score(r)
    assert without[1] == pytest.approx(with_u[1] / 1.8) and without[2] == with_u[2]


def test_switch_survey_off_skips_the_survey_look(scene):
    _, events = run(scene, pixel_cap=0.3, survey=False)
    assert steps(events)[0]["cause"] == "best-score" and not any(s["region"] == "s0" for s in steps(events))


def test_switch_verify_off_queues_no_verify_look(scene):
    _, on = run(scene, pixel_cap=1.0)
    _, off = run(scene, pixel_cap=1.0, verify=False)
    assert any(s["decision"]["action"] == "verify" for s in steps(on))
    assert not any(s["decision"]["action"] == "verify" for s in steps(off))


def test_switch_order_random_is_seeded(scene):
    def path(seed):
        _, events = run(scene, pixel_cap=1.0, order="random", seed=seed)
        return [(s["region"], s["cause"]) for s in steps(events)]

    a = path(1)
    assert any(c == "random-order" for _, c in a) and not any(c == "best-score" for _, c in a)
    assert a == path(1) and any(path(seed) != a for seed in (2, 3, 4))


def test_switch_governor_off_keeps_only_kill_and_caps(scene):
    job, events = run(scene, pixel_cap=1.0, chaos=("p95_spike",), governor=False)
    assert not any("governor" in s for s in steps(events)) and job.settings.history == []
    assert {s["model"] for s in steps(events)} == {"accurate"}
    _, capped = run(scene, pixel_cap=0.1, governor=False)
    assert capped[-1]["cause"] == "cap-hit"


class LoadedClock(FakeClock):
    """A machine under load: every read costs 3 ms instead of 1."""

    def __call__(self):
        self.t += 0.003
        return self.t


def test_a_fixed_cost_table_makes_decisions_independent_of_load(scene):
    table = {"warp": 5.0, "segment": 30.0}

    def decisions(clock, **cfg):
        job, events = run(scene, clock=clock, pixel_cap=1.0, **cfg)
        return job, [(s["decision"], s["cause"]) for s in steps(events)]

    job, calm = decisions(FakeClock(), t_hat_ms=table)
    assert job._t_hat() == 35.0
    assert all("t_hat 35 ms" in s["rationale"] for s in job.trace if s["cause"] == "best-score")
    assert decisions(LoadedClock(), t_hat_ms=table)[1] == calm
