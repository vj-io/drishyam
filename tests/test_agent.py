import time

import cv2
import numpy as np
import pytest

from drishyam import imaging
from drishyam.agent import REVIEW_KINDS, Job, JobConfig, gaze_path_from_trace, look_iou, validate_replay
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


class Flip:
    """Dark pixels are one crisp class: `coarse` while the dark blob is small in the crop (a coarse view of it),
    `fine` once it fills more of the crop (a finer view of the same place). A class flip with no change of place."""

    def __init__(self, coarse, fine, k=30000):
        self.coarse, self.fine, self.k = coarse, fine, k

    def predict(self, crop, model="accurate"):
        dark = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 105
        probs = np.zeros((len(CLASSES), *dark.shape), np.float32)
        probs[0], probs[CLASSES.index(self.fine if np.count_nonzero(dark) > self.k else self.coarse)] = ~dark, dark
        return finish(probs, time.perf_counter(), model)


def disc_scene():
    """A dark disc of radius 130 px (53k px^2: a critical area for spalling) on 1100 x 900 concrete."""
    from .conftest import concrete

    img = concrete(900, 1100, 6)
    cv2.circle(img, (550, 450), 130, (40, 40, 40), -1)
    return img


def test_a_class_flip_is_a_class_change_not_a_disagreement():
    """The published 'unconfirmed rebar, IoU 0.003' compared a coarse spalling mask with a fine corrosion mask. A
    check compares the SAME class across levels; when the coarser class is gone but the finer look's best mask
    covers the same place, that is a class change: a human is asked when either class is critical, and the coarser
    finding is unconfirmed (its same-class IoU), never a 'disagreement' about where the defect is."""
    def job(coarse, fine):
        j = Job(disc_scene(), JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), segmenter=Flip(coarse, fine))
        ev = list(j.run())
        (chk,) = [s["verify"] for s in steps(ev) if "verify" in s]
        return j, ev, chk

    j, ev, chk = job("spalling", "corrosion")  # a critical spall at L1, rust at L0, the same pixels
    assert chk["class"] == "spalling" and chk["class_now"] == "corrosion" and chk["against_level"] == 1
    assert chk["iou"] == 0 and chk["iou_any"] >= 0.9
    reqs = [e for e in ev if e["type"] == "review_request"]
    assert [e["kind"] for e in reqs] == ["critical", "class-change"]
    spall = next(f for f in j.findings if f.cls == "spalling")
    assert reqs[1] == {"type": "review_request", "finding_id": spall.id, "kind": "class-change",
                       "reason": f"class change: spalling at L1, corrosion at L0 (same place, iou {chk['iou_any']:g})",
                       "iou": chk["iou_any"], "level_a": 1, "level_b": 0, "classes": ["spalling", "corrosion"]}
    assert spall.critical and spall.verify_iou == 0 and spall.verified is False
    assert any(f.cls == "corrosion" for f in j.findings)  # the finer class is kept

    j, ev, chk = job("corrosion", "efflorescence")  # nothing critical: recorded, no human asked
    assert (chk["class"], chk["class_now"], chk["iou"]) == ("corrosion", "efflorescence", 0) and chk["iou_any"] >= 0.9
    assert not any(e["type"] == "review_request" for e in ev)
    assert next(f for f in j.findings if f.cls == "corrosion").verified is False
    assert any(f.cls == "efflorescence" for f in j.findings)


def test_a_critical_class_seen_only_by_an_escalating_look_still_reaches_a_human():
    """L1 sees an 11 px dark disc as exposed_rebar, 37 px: under A_min, so it escalates and drafts nothing. L0 sees
    the same place as corrosion (149 px) and keeps that. A class change goes to a human when either CLASS is
    critical, so the request names the corrosion finding; its verify_iou stays null (no check of corrosion ran). The
    finding is held in pending_review, so the work order waits for that human and the review API can decide it."""
    from .conftest import concrete

    img = concrete(900, 1100, 6)
    cv2.circle(img, (550, 450), 7, (40, 40, 40), -1)
    job = Job(img, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), segmenter=Flip("exposed_rebar", "corrosion", 80),
              replay=[["gaze", 550, 450, 1]])
    events = list(job.run())
    chk = steps(events)[1]["verify"]
    assert (chk["class"], chk["class_now"], chk["iou"]) == ("exposed_rebar", "corrosion", 0) and chk["iou_any"] >= 0.5
    (f,) = job.findings
    (req,) = [e for e in events if e["type"] == "review_request"]
    assert f.cls == "corrosion" and f.verify_iou is None and f.verified is None
    assert req["kind"] == "class-change" and req["finding_id"] == f.id and req["classes"] == ["exposed_rebar", "corrosion"]
    assert req["reason"] == f"class change: exposed_rebar at L1, corrosion at L0 (same place, iou {chk['iou_any']:g})"
    assert f.critical and f.state == "pending_review" and f.critical_step == 2
    assert f.critical_rule == "class change: exposed_rebar at L1 here (exposed_rebar: always critical)"
    assert [e["finding"]["state"] for e in events if e["type"] == "finding"][-1] == "pending_review"
    done = events[-1]
    assert done["status"] == "awaiting_review" and done["work_order"]["status"] == "blocked"
    (wf,) = done["work_order"]["findings"]
    decide(wf, "approve", "human")
    assert wf["state"] == "approved"


def test_a_class_change_is_critical_by_the_largest_component_as_a_finding_is():
    """Two 25k px^2 dark discs: 51k px^2 of spalling in all, but each spall is under the 40k critical area, so the
    spalling finding is not critical, and nor is the class change to corrosion: it is recorded in the step's verify
    and no human is asked (counting the class's total area asked one)."""
    from .conftest import concrete

    img = concrete(900, 1100, 6)
    for x in (430, 670):
        cv2.circle(img, (x, 450), 90, (40, 40, 40), -1)
    job = Job(img, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), segmenter=Flip("spalling", "corrosion"),
              replay=[["gaze", 550, 450, 1]])
    events = list(job.run())
    chk = steps(events)[1]["verify"]
    assert (chk["class"], chk["class_now"]) == ("spalling", "corrosion") and chk["iou_any"] >= 0.5
    spall = next(f for f in job.findings if f.cls == "spalling")
    assert not spall.critical and spall.measure["area_px"] < SPALL_CRITICAL_AREA_PX < 2 * spall.measure["area_px"]
    assert not any(e["type"] == "review_request" for e in events) and events[-1]["work_order"]["status"] == "ready"


def test_a_check_compares_nothing_when_the_coarser_mask_lies_outside_the_finer_view():
    """None means nothing was compared: no pixel of the coarser mask lies in the finer view. A finer view that shows
    the same class somewhere else is not a disagreement about it (that read iou 0, 'disagree')."""
    prev, cur = np.zeros((64, 64), np.uint8), np.zeros((64, 64), np.uint8)
    prev[10:20, 10:20] = 1  # an L1 view: crop px x 2 = L0 px, so this mask is at L0 20..40
    cur[30:40, 30:40] = 1
    to_l1 = np.diag([2.0, 2.0, 1.0])
    inside = np.array([[1.0, 0, 60], [0, 1, 60], [0, 0, 1]])  # an L0 view of L0 60..124, inside the L1 view (0..128)
    assert look_iou(prev, to_l1, cur, inside) is None
    assert look_iou(prev, to_l1, cur, np.eye(3)) is not None  # the coarser mask is in this view: compared


def test_a_finer_seed_is_never_overwritten_by_a_coarser_look(scene):
    """An L1 look, then an L3 look, both show the same unvisited candidate: it keeps the L1 seed (the finer view of
    it), and the L3 look does not list it under `seeded`."""
    probe = Job(scene, JobConfig(thumbs=False), clock=FakeClock())
    it = probe.run()
    next(it)
    cands = next(it)["candidates"]
    c = cands[0]
    x, y = c["box"][0] + c["box"][2] // 2 + 60, c["box"][1] + c["box"][3] // 2 + 60  # near it, not its centre
    job = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), replay=[["gaze", x, y, 1], ["gaze", x, y, 3]])
    st = steps(list(job.run()))
    fine = [z for z in st[0]["seeded"] if z["region"] == c["id"]]
    assert fine and fine[0]["from_level"] == 1
    assert st[1]["decision"]["args"]["level"] == 3 and st[1]["seeded"]  # the L3 look seeds others it shows ...
    assert not any(z["region"] == c["id"] for z in st[1]["seeded"])  # ... but not the one an L1 look already showed


def test_a_look_that_sees_under_half_of_a_candidates_window_neither_shows_nor_seeds_it(scene):
    """An L0 look covers at most a quarter of an L1 candidate's first-look window: it neither shows the candidate (so
    the floor still waits for a look that sees it) nor seeds it from that quarter; an L1 look that covers most of
    the window then seeds it, at L1."""
    probe = Job(scene, JobConfig(thumbs=False), clock=FakeClock())
    it = probe.run()
    next(it)
    c = next(it)["candidates"][0]
    x, y = c["box"][0] + c["box"][2] // 2 + 100, c["box"][1] + c["box"][3] // 2  # near it, not its centre
    job = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), replay=[["gaze", x, y, 0], ["gaze", x, y, 1]])
    st = steps(list(job.run()))
    assert st[0]["decision"]["args"]["level"] == 0 and st[0]["quality"]["ok"]
    assert not any(z["region"] == c["id"] for z in st[0].get("seeded", []))
    assert [z["from_level"] for z in st[1]["seeded"] if z["region"] == c["id"]] == [1]


def test_the_floor_never_stops_a_job_while_a_candidate_has_had_only_a_coarse_view(scene):
    """At a floor halt, every unvisited candidate still worth a look (unseen > 0) has been shown by a look at least
    as fine as its own first look: one that only the survey, or no look, has shown keeps the job going."""
    from pathlib import Path

    b01 = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / "bridge_01.jpg"))
    floors = 0
    for img in (scene, b01):
        for cap in (0.3, 1.0):
            job, events = run(img, pixel_cap=cap)
            if not steps(events)[-1]["rationale"].endswith("(floor: eps x v0 / t_hat)"):
                continue
            floors += 1
            for r in job.regions.values():
                level, win = job._first_look(r)
                if not r.looks and not r.done and job._unseen(win, level) > 0:
                    assert r.shown_level is not None and r.shown_level <= level, (cap, r.id, r.shown_level)
    assert floors


def test_a_finding_that_becomes_critical_on_a_later_look_is_sent_to_a_human():
    """An L0 look sees only part of a 53k px^2 dark disc (spalling, under the 40k critical area); an L1 look then
    sees all of it and the same finding becomes critical: a 'critical' request follows that look, as for a new one."""
    img = disc_scene()
    job = Job(img, JobConfig(thumbs=False, pixel_cap=1.0, verify=False), clock=FakeClock(), segmenter=DarkIsSpall(),
              replay=[["gaze", 1000, 450, 0], ["gaze", 550, 450, 1]])
    events = list(job.run())
    (f,) = [f for f in job.findings if f.cls == "spalling"]
    assert f.step == 1 and f.critical and f.critical_step == 2 and f.state == "pending_review"
    reqs = [e for e in events if e["type"] == "review_request" and e["kind"] == "critical"]
    assert [(e["finding_id"], e["reason"], e["level_b"]) for e in reqs] == [(f.id, "spalling is critical", 1)]


def test_seeded_uncertainty_and_class_reorder_the_argmax(scene):
    """The judges found 81 of 81 argmax scores with severity 1 and uncertainty 0: entropy and class were set only on
    regions already looked at, and only unvisited ones are scored. Every look now seeds the unvisited candidates it
    shows, the argmax reads those seeds (each pick's rationale names them), and that changes the order of looks."""
    from drishyam.findings import SEVERITY

    def run_(**cfg):
        _, events = run(scene, pixel_cap=1.0, v0=0.0, **cfg)
        return steps(events)

    seeded, plain = run_(), run_(seed_from_looks=False)
    picks = lambda st: [s["region"] for s in st if s["cause"] == "best-score"]
    assert picks(seeded) != picks(plain)
    assert all("severity 1 (seeding off), uncertainty 0.00," in s["rationale"] for s in plain if s["cause"] == "best-score")
    assert not any("seeded" in s for s in plain)
    latest, terms = {}, []
    for s in seeded:
        if s["cause"] == "best-score":
            seed = latest[s["region"]]  # the survey showed every candidate of this scene
            sev = SEVERITY.get(seed["cls"], 1.0)
            assert (f"severity {sev:g} ({seed['cls'] or 'no class'} seen at L{seed['from_level']}), "
                    f"uncertainty {seed['entropy']:.2f}," in s["rationale"])
            terms.append((sev, seed["entropy"]))
        for x in s.get("seeded", []):
            assert x["region"] != s["region"] and x["from_level"] == s["decision"]["args"]["level"]
            if x["region"] in latest:
                assert x["from_level"] <= latest[x["region"]]["from_level"]  # a coarser look never overwrites
            latest[x["region"]] = x
    assert terms and any(sev != 1 for sev, _ in terms) and any(u > 0 for _, u in terms)
    assert steps(run(scene, pixel_cap=1.0)[1])[0]["seeded"]  # the survey seeds first of all


def test_a_clean_photo_stops_on_the_floor_once_a_first_level_look_has_shown_what_is_left():
    """lambda was eps x value / cost: 0 until something is found, so a photo with nothing to find could stop only at
    its cap or once every candidate had a look. The floor eps x v0 / t_hat ends it on the rule when the best
    candidate left is worth less than a look, and the halt says the floor was the binding term. Here r2's L1 look
    shows r3 (seeded at L1, its own first-look level, with no class): the job stops instead of looking at r3. Without
    the floor it looks at r3 too."""
    from evals import synth

    img, gt, _ = synth.render(2, 1600, 1200, {"distractors": ["water_stain"]})
    assert not gt

    def once(**cfg):
        _, events = run(img, pixel_cap=1.0, **cfg)
        return events[-1], steps(events)

    done, st = once()
    assert done["cause"] == "stop-rule" and st[-1]["rationale"].endswith("(floor: eps x v0 / t_hat)")
    assert any(x["from_level"] == 1 and x["cls"] is None for s in st for x in s.get("seeded", []))
    assert done["summary"]["pixels_frac"] < 1 and done["summary"]["findings"] == 0
    done0 = once(v0=0.0)[0]
    assert done0["cause"] != "stop-rule" and done0["summary"]["steps"] > done["summary"]["steps"]


def test_a_coarse_look_alone_never_puts_the_stop_floor_on():
    """bridge_01: five 1-2 px cracks, at the web default 30 % cap. The L3 survey cannot resolve them and seeds every
    candidate 'no class, uncertainty 0'; when that alone put the floor on, the job stopped after the survey with
    nothing found. The floor needs a look at least as fine as the candidate's own first look (L1 here)."""
    from pathlib import Path

    img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / "bridge_01.jpg"))
    job, events = run(img)
    st = steps(events)
    assert st[0]["cause"] == "survey" and st[0]["seeded"] and all(x["cls"] is None for x in st[0]["seeded"])
    assert len(job.gaze_path) > 1 and any(f.cls == "crack" for f in job.findings)
    if st[-1]["rationale"].endswith("(floor: eps x v0 / t_hat)"):
        assert any(s["decision"]["args"]["level"] <= 1 for s in st[:-1])


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
    """PRD rule 5: every override writes a cause. With three candidates the stop rule's floor ends the job on the
    step right after the spike, the same step on which the governor switched the model; the halt record carries it."""
    job, events = run(scene, pixel_cap=1.0, chaos=("p95_spike",), max_candidates=3)
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


def test_a_drill_the_job_ends_before_is_reported_not_silently_skipped(scene):
    """Each drill fires on a fixed step. With the default config at a 30 % cap the scene ends after three looks,
    before the spike (step 4) and the malformed planner (step 6): the done event names them, so a drill that tested
    nothing never reads as passed. At the full cap every drill fires and the list is empty."""
    drills = ("p95_spike", "planner_timeout", "planner_malformed")
    for cap, missed in ((0.3, ["p95_spike", "planner_malformed"]), (1.0, [])):
        _, events = run(scene, pixel_cap=cap, chaos=drills, planner_timeout_s=0.05)
        fired = {e["status"] for e in events if e["type"] == "planner"}
        spiked = any(s.get("injected") == "p95_spike" for s in steps(events))
        assert events[-1]["drills_not_reached"] == missed
        assert spiked is ("p95_spike" not in missed) and ("rejected" in fired) is ("planner_malformed" not in missed)
        assert "fallback" in fired
    assert run(scene, pixel_cap=0.3)[1][-1]["drills_not_reached"] == []


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


class Part(Flip):
    """Like Flip, but the finer view labels only the left quarter of the dark disc: the same class, elsewhere in
    extent. A geometric disagreement (no class change)."""

    def predict(self, crop, model="accurate"):
        out = super().predict(crop, model)
        if out.areas[self.fine] > self.k:  # the finer view
            xs = np.nonzero(out.labels)[1]
            keep = np.arange(crop.shape[1]) < xs.min() + (xs.max() - xs.min()) // 4
            probs = out.probs.copy()
            probs[0][:, ~keep] += probs[CLASSES.index(self.fine)][:, ~keep]
            probs[CLASSES.index(self.fine)][:, ~keep] = 0
            out = finish(probs, time.perf_counter(), model)
        return out


def test_verification_fields_agree_everywhere_on_the_scene(scene):
    """On a realistic scene and several budgets: `verified` is `verify_iou >= 0.5` wherever a check ran, the work
    order mirrors both fields, and a finding no check compared has both null."""
    for cap in (0.3, 1.0):
        for extra in ({}, {"v0": 0.0}):
            job, events = run(scene, pixel_cap=cap, **extra)
            wo = {f["id"]: f for f in events[-1]["work_order"]["findings"]}
            for f in job.findings:
                assert f.verified is (None if f.verify_iou is None else f.verify_iou >= 0.5)
                assert wo[f.id]["verify_iou"] == f.verify_iou and wo[f.id]["verified"] is f.verified


def test_a_failed_verification_is_unconfirmed_and_sent_to_a_human():
    """The kind and fields of a same-class disagreement: the finer look's mask of the same class covers another
    extent (IoU < 0.5) at L0, where no further look can settle it. A non-critical finding is unconfirmed (`verified:
    false`) and a human is asked (`unconfirmed`); a critical one is sent at once (`disagree`). A request changes no
    state."""
    for cls, kind, reason in (("corrosion", "unconfirmed", "unconfirmed: "), ("spalling", "disagree", "")):
        job = Job(disc_scene(), JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), segmenter=Part(cls, cls))
        events = list(job.run())
        (chk,) = [s["verify"] for s in steps(events) if "verify" in s]
        assert chk["class"] == chk["class_now"] == cls and chk["iou"] == chk["iou_any"] < 0.5
        (f,) = job.findings
        assert f.verify_iou == chk["iou"] and f.verified is False
        wo = events[-1]["work_order"]["findings"][0]
        assert wo["verify_iou"] == f.verify_iou and wo["verified"] is False
        asked = [e for e in events if e["type"] == "review_request" and e["kind"] == kind]
        assert [e["reason"] for e in asked] == [f"{reason}iou {chk['iou']:g} < 0.5 between L1 and L0"]
        assert asked[0]["finding_id"] == f.id and asked[0]["classes"] == [cls, cls] and asked[0]["iou"] == chk["iou"]
        assert f.state == ("pending_review" if f.critical else "draft") and f.critical is (cls == "spalling")


def test_a_check_judges_only_a_finding_of_the_class_it_compared():
    """bridge_03 and bridge_05: a look's compared class can be absorbed into a finding of another class (rust on
    exposed bars, a crack in a spall's rim). Its IoU says nothing about that finding's own class, so the finding that
    absorbed it keeps its verify_iou, and no disagreement names it: a finding's verify_iou is always the same-class
    IoU of a check of the class it holds, and every disagreement names a finding of the compared class."""
    from pathlib import Path

    for name in ("bridge_03", "bridge_05"):
        img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / f"{name}.jpg"))
        job, events = run(img)
        ids, st = {f.id: f for f in job.findings}, steps(events)
        checks = [s["verify"] for s in st if "verify" in s]
        for f in job.findings:
            if f.verify_iou is not None:
                assert any(c["class"] == f.cls and c["iou"] == f.verify_iou for c in checks), (name, f.id, f.cls)
        reqs = [e for e in events if e["type"] == "review_request"]
        for e in reqs:
            if e["kind"] in ("disagree", "unconfirmed"):
                assert ids[e["finding_id"]].cls == e["classes"][0] == e["classes"][1], (name, e)
            elif e["kind"] == "class-change":
                assert ids[e["finding_id"]].cls in e["classes"], (name, e)
        # the case this test exists for: a compared class that this look absorbed into a finding of another class
        absorbed = [(s, e) for s in st if "verify" in s for e in events[events.index(s) + 1:][:12]
                    if e["type"] == "review_request" and e["kind"] == "ambiguous" and e["classes"][1] == s["verify"]["class"]
                    and e["level_b"] == s["decision"]["args"]["level"]]
        if name == "bridge_03":
            assert absorbed, "bridge_03 has a check of a class absorbed into another finding"


def test_review_requests_say_their_reason_in_fields_too():
    """bridge_03 and bridge_05 between them ask for four kinds of review; each event's kind, iou, levels and classes
    match its reason."""
    from pathlib import Path

    kinds = set()
    for name in ("bridge_03", "bridge_05"):
        img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / f"{name}.jpg"))
        _, events = run(img)
        for e in (e for e in events if e["type"] == "review_request"):
            assert 0 <= e["level_b"] <= 3 and e["kind"] in REVIEW_KINDS
            kinds.add(e["kind"])
            if e["kind"] in ("disagree", "unconfirmed"):
                prefix = "unconfirmed: " if e["kind"] == "unconfirmed" else ""
                assert e["reason"] == f"{prefix}iou {e['iou']:g} < 0.5 between L{e['level_a']} and L{e['level_b']}"
                assert e["level_a"] > e["level_b"] and len(e["classes"]) == 2 and e["classes"][0] == e["classes"][1]
            elif e["kind"] == "class-change":
                assert e["reason"].startswith(f"class change: {e['classes'][0]} at L{e['level_a']}, {e['classes'][1]} at ")
            elif e["kind"] == "ambiguous":
                kept, absorbed = e["classes"]
                assert e["reason"].startswith(f"ambiguous: {absorbed} ") and f"inside {kept} " in e["reason"]
            else:
                assert e["level_a"] is None and e["iou"] is None and e["reason"] == f"{e['classes'][0]} is critical"
    assert kinds == {"critical", "ambiguous", "disagree", "unconfirmed"}


def test_the_clean_concrete_sample_drafts_no_crack():
    """bridge_08 has one straight formwork groove, which used to be drafted as a 93 % crack ([2049, 1250, 581, 798]
    in run j_d60376), and bug holes that classical-v1 labels crack. Every look that sees the groove logs it as a
    straight line; a bug hole is logged as not elongated (a blob, under PORE_ASPECT widths long). No crack is drafted."""
    from pathlib import Path

    from drishyam.segment import PORE_ASPECT, PORE_CAUSE, STRAIGHT_CAUSE

    img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / "bridge_08.jpg"))
    job, events = run(img)
    rejected = [x for s in steps(events) for x in s.get("rejected", [])]
    lines = [x for x in rejected if x["cause"] == STRAIGHT_CAUSE]
    pores = [x for x in rejected if x["cause"] == PORE_CAUSE]
    assert lines and pores and len(lines) + len(pores) == len(rejected)
    assert all(x["residual_px"] < 1.5 and imaging.box_iou(x["box"], [2049, 1250, 581, 798]) > 0 for x in lines)
    assert all(x["length_px"] < PORE_ASPECT * x["width_px"] and inside_image(x["box"], img.shape) for x in pores)
    assert not any(f.cls == "crack" for f in job.findings)


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
    # absorb off (the dacl10k profile): the class that was absorbed is a finding of its own, and nobody is asked
    job, events = run(scene, pixel_cap=1.0, absorb=False)
    assert not any(e["type"] == "review_request" and e["kind"] == "ambiguous" for e in events)
    assert {e["reason"].split(":")[1].split()[0] for e in asked} <= {f.cls for f in job.findings}


@pytest.mark.parametrize("band", [0.25, 0.35])
def test_a_blown_out_band_is_not_assessed_and_takes_no_looks_from_a_crack(band):
    """A blown-out band (sky) over a quarter or a third of the photo is marked not assessed from the overview, and
    neither it nor its edge yields candidates: the crack below it is still found at the 30 % cap. A small glare patch
    still gets a look, so the quality gate marks it (the 'Glare and blur' sample)."""
    from pathlib import Path

    from .conftest import add_crack, concrete

    img = concrete(1536, 2048, 7)
    img[: int(1536 * band)] = 252
    add_crack(img, (500, 900), n=300, width=3)
    job, events = run(img, pixel_cap=0.3)
    assert any(f.cls == "crack" and f.box[1] > 900 for f in job.findings)
    sky = events[-1]["work_order"]["not_assessed"][0]
    assert sky == {"step": 0, "box": [0, 0, 2048, sky["box"][3]], "reason": "glare", "action": "re-shoot"}
    assert abs(sky["box"][3] - 1536 * band) <= 8  # one overview px
    assert all(c["box"][1] + c["box"][3] // 2 > 1536 * band for c in events[1]["candidates"])  # no centre in the band
    img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / "bridge_07.jpg"))
    _, events = run(img, pixel_cap=0.3)
    (glare,) = events[-1]["work_order"]["not_assessed"]
    assert glare["step"] > 0 and glare["reason"] == "glare"


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



# ---- per-segmenter profiles (agent.PROFILES) ---------------------------------------------------------------
ONNX = "onnx:drishyam_dacl10k.onnx"


def named(name, sha256=None):
    """The classical engine under another segmenter's name and weights (by default the weights its profile was
    chosen with): profiles are looked up by both."""
    from drishyam.agent import WEIGHTS
    from drishyam.segment import ClassicalSegmenter

    return type("Named", (ClassicalSegmenter,), {"name": name, "sha256": sha256 or WEIGHTS.get(name)})()


def test_the_classical_profile_is_the_agent_every_classical_number_ran_with(scene):
    """classical-v1 resolves to exactly the JobConfig defaults from before profiles existed (first look L1, any level,
    absorption on), so no classical result moves (results/dev/onnx_profile_sweep.json: the byte-for-byte rerun)."""
    from dataclasses import asdict

    from drishyam.agent import PROFILES
    from drishyam.governor import USD_PER_HOUR
    from drishyam.planner import TIMEOUT_S

    assert PROFILES["classical-v1"] == {"min_view": 0.0, "absorb": True}
    job = Job(scene, JobConfig(), clock=FakeClock())  # the default segmenter here: the classical engine
    assert asdict(job.cfg) == {
        "pixel_cap": 0.3, "usd_cap": 0.05, "seconds_cap": 20.0, "size": 512, "first_level": 1, "min_level": 0,
        "min_view": 0.0, "profile": "classical-v1", "tau": 0.5, "a_min": 64, "eps": 0.25, "sla_ms": 500.0,
        "max_candidates": 20, "chaos": (), "planner_every": 5, "planner_timeout_s": TIMEOUT_S,
        "usd_per_hour": USD_PER_HOUR, "thumbs": True, "max_steps": 200, "use_uncertainty": True,
        "seed_from_looks": True, "v0": 2.0, "survey": True, "verify": True, "absorb": True, "order": "score", "seed": 0,
        "governor": True, "t_hat_ms": None}


def test_a_job_takes_its_segmenters_profile_by_name_and_a_knob_set_explicitly_wins(scene, monkeypatch):
    from drishyam.agent import PROFILES, profile_for
    from drishyam.segment import ClassicalSegmenter
    from evals import harness

    assert profile_for(ClassicalSegmenter()) == profile_for(object()) == profile_for(named("onnx:other.onnx")) \
        == "classical-v1"  # a segmenter nobody calibrated keeps the original defaults
    assert profile_for(named(ONNX)) == ONNX
    # the same file name with other weights (a retrained model installed over it) was not what the profile measured
    assert profile_for(named(ONNX, sha256="0" * 64)) == "classical-v1"
    assert profile_for(type("NoSha", (ClassicalSegmenter,), {"name": ONNX})()) == "classical-v1"
    cfg = Job(scene, JobConfig(), segmenter=named(ONNX)).cfg
    assert cfg.profile == ONNX and {k: getattr(cfg, k) for k in PROFILES[ONNX]} == PROFILES[ONNX]
    cfg = Job(scene, JobConfig(min_view=0.0), segmenter=named(ONNX)).cfg  # a dev sweep's variant
    assert (cfg.min_view, cfg.absorb) == (0.0, PROFILES[ONNX]["absorb"])
    cfg = Job(scene, JobConfig(profile="classical-v1"), segmenter=named(ONNX)).cfg
    assert (cfg.profile, cfg.min_view, cfg.absorb) == ("classical-v1", 0.0, True)
    # The eval harness wraps the segmenter to record its looks: the wrapper must carry the name and the weights, or
    # every ONNX eval would silently run the classical profile.
    monkeypatch.setattr(harness, "default_segmenter", lambda: named(ONNX))
    rec = harness.RecordingSegmenter()
    assert profile_for(rec) == ONNX and (rec.name, rec.sha256) == (ONNX, named(ONNX).sha256)


def test_an_onnx_segmenter_hashes_its_weights_once(tmp_path, monkeypatch):
    """The profile check reads the weights file's sha256; every job loads its own segmenter, so the hash is cached per
    path, mtime and size, and a file replaced in place is hashed again."""
    import hashlib

    from drishyam import segment

    monkeypatch.setattr(segment.OnnxSegmenter, "_load", staticmethod(lambda path: None))
    w = tmp_path / "drishyam_dacl10k.onnx"
    w.write_bytes(b"weights v1")
    calls = []
    real = hashlib.file_digest
    monkeypatch.setattr(hashlib, "file_digest", lambda *a: calls.append(1) or real(*a))
    a, b = segment.OnnxSegmenter(str(w)), segment.OnnxSegmenter(str(w))
    assert a.sha256 == b.sha256 == hashlib.sha256(b"weights v1").hexdigest() and len(calls) == 1
    w.write_bytes(b"weights v2, retrained")
    assert segment.OnnxSegmenter(str(w)).sha256 == hashlib.sha256(b"weights v2, retrained").hexdigest()


class Unsure:
    """Dark pixels are spalling at p 0.55: a mask that clears A_min, at an entropy near 1 (> tau) at every level."""

    p = 0.55

    def __init__(self, name):
        from drishyam.agent import WEIGHTS

        self.name, self.sha256 = name, WEIGHTS.get(name)

    def predict(self, crop, model="accurate"):
        dark = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 105).astype(np.float32)
        probs = np.zeros((len(CLASSES), *dark.shape), np.float32)
        probs[0], probs[CLASSES.index("spalling")] = 1 - self.p * dark, self.p * dark
        return finish(probs, time.perf_counter(), model)


class Sure(Unsure):
    """Dark pixels are spalling at p 0.97: a confident mask (entropy < tau) at every level."""

    p = 0.97


class Speck(Unsure):
    """A confident 6 x 6 spalling speck at the centre of every view: 36 px, under A_min, at entropy 0."""

    def predict(self, crop, model="accurate"):
        h, w = crop.shape[:2]
        probs = np.zeros((len(CLASSES), h, w), np.float32)
        probs[0] = 1
        probs[0, h // 2 - 3 : h // 2 + 3, w // 2 - 3 : w // 2 + 3] = 0.02
        probs[CLASSES.index("spalling"), h // 2 - 3 : h // 2 + 3, w // 2 - 3 : w // 2 + 3] = 0.98
        return finish(probs, time.perf_counter(), model)


class Haze(Unsure):
    """Spalling at p 0.4 everywhere: background wins every pixel (no class), yet every pixel is damage-like, so the
    entropy is 0.97 (> tau)."""

    def predict(self, crop, model="accurate"):
        probs = np.zeros((len(CLASSES), *crop.shape[:2]), np.float32)
        probs[0], probs[CLASSES.index("spalling")] = 0.6, 0.4
        return finish(probs, time.perf_counter(), model)


def spots(w, h):
    """Concrete with four dark spots: the saliency candidates the looks go to."""
    from .conftest import concrete

    img = concrete(h, w, 9)
    for fx, fy in ((0.15, 0.2), (0.6, 0.25), (0.4, 0.65), (0.75, 0.75)):
        cv2.circle(img, (int(fx * w), int(fy * h)), w // 50, (45, 45, 45), -1)
    return img


def onnx_looks(img, seg, **cfg):
    job = Job(img, JobConfig(thumbs=False, pixel_cap=1.0, **cfg), clock=FakeClock(), segmenter=seg)
    ev = list(job.run())
    return job, [s for s in steps(ev) if s["decision"]["action"] != "halt"]


@pytest.mark.parametrize("w, h, finest", [(4608, 3456, 2), (2048, 1536, 1)])
def test_the_onnx_profile_never_looks_finer_than_a_share_of_the_photo(w, h, finest):
    """The trained model is reliable on views that span a good share of the photo: under its profile no look the
    rules choose, first or follow-up, is finer than the finest level whose 512-px window spans min_view (0.3) of the
    long side, though the measurements ask for finer looks (entropy > tau at every level): L2 on a CODEBRIM-sized
    4608-px photo, L1 on a 2048-px one. (Replayed looks and gazes the governor shrank are exempt: api-contract.md.)"""
    img = spots(w, h)
    job, st = onnx_looks(img, Unsure(ONNX))
    assert 512 * 2**finest >= job.cfg.min_view * w > 512 * 2 ** (finest - 1)
    assert st and min(s["decision"]["args"]["level"] for s in st) == finest
    held = [s for s in st if s["observation"]["entropy"] > job.cfg.tau and s["decision"]["args"]["level"] == finest]
    assert held and all(f"L{finest} is the finest level for {ONNX} (min_view 0.3): entropy " in s["rationale"]
                        and "spalling drafted here; no finer look" in s["rationale"] for s in held)
    assert {s["profile"] for s in job.trace} == {ONNX} and job.work_order["profile"] == ONNX
    assert job.work_order["weights_sha256"] == Unsure(ONNX).sha256
    _, st = onnx_looks(img, Unsure(ONNX), profile="classical-v1")  # the same measurements, original defaults
    assert min(s["decision"]["args"]["level"] for s in st) < finest


@pytest.mark.parametrize("seg, said", [
    (Speck(ONNX), "spalling area 36 px < A_min 64, spalling not drafted; no finer look"),
    (Haze(ONNX), "entropy 0.97 > 0.5, no class; no finer look")])
def test_a_held_look_says_what_it_drafted_and_never_claims_a_draft_it_did_not_make(seg, said):
    """A look at the finest level that would have escalated says why (a mask under A_min, or entropy above tau) and
    what it drafted: here nothing, so no step may say 'drafted here' and no finding exists."""
    job, st = onnx_looks(spots(2048, 1536), seg)
    held = [s for s in st if s["decision"]["args"]["level"] == 1]
    assert held and all(f"; L1 is the finest level for {ONNX} (min_view 0.3): {said}" in s["rationale"] for s in held)
    assert not any("drafted here" in s["rationale"] for s in st) and not job.findings
    assert any(s["cause"] in ("entropy>tau", "area<A_min") for s in st)  # coarser looks still escalate to L1


def test_a_mask_drafted_at_the_finest_level_gets_no_verify_and_says_so():
    """The rules confirm a new mask one level finer; at the profile's finest level there is none, so the look says
    the mask was not verified, no verify look goes to its region and its finding keeps verify_iou null. Under the
    original defaults the same masks are verified at L0."""
    img = spots(2048, 1536)
    job, st = onnx_looks(img, Sure(ONNX))
    note = f"not verified: L1 is the finest level for {ONNX} (min_view 0.3), no finer look"
    noted = [s for s in st if note in s["rationale"]]
    assert noted and all(s["decision"]["args"]["level"] == 1 and s["cause"] == "best-score" for s in noted)
    regions = {s["region"] for s in noted}
    assert not any(s["decision"]["action"] == "verify" and s["region"] in regions for s in st)
    assert all(f.verify_iou is None for f in job.findings if f.region in regions)
    _, st = onnx_looks(img, Sure(ONNX), profile="classical-v1")
    assert any(s["cause"] == "verify-new-mask" and s["decision"]["args"]["level"] == 0 for s in st)
    assert not any("not verified" in s["rationale"] for s in st)


def test_a_planner_cannot_zoom_past_the_profiles_finest_level():
    """A planner's change_level finer than the profile's finest level runs at that level, and the look says so."""
    import json

    def planner(digest):
        return json.dumps({"strategy": "change_level", "rationale": "look closer",
                           "params": {"region_id": digest["top_candidates"][0]["id"], "level": 0}})

    job = Job(spots(2048, 1536), JobConfig(thumbs=False, pixel_cap=1.0, planner_every=2), clock=FakeClock(),
              segmenter=Unsure(ONNX), planner=planner)
    asked = [s for s in steps(list(job.run())) if s["cause"] == "planner:change_level"]
    assert asked and all(s["decision"]["args"]["level"] == 1 and s["rationale"].startswith(
        "look closer (L0 is finer than this segmenter's finest L1)") for s in asked)
