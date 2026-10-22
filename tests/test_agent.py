import time
from pathlib import Path

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


def test_fill_shrinks_only_the_look_that_no_longer_fits(scene):
    """fill: three 512-px looks leave 0.022 of this photo under a 30 % cap; the next look takes it at 256 px (a
    multiple of 32, at least MIN_GAZE) instead of the job halting, and says so; the cap still holds."""
    plain, filled = (steps(run(scene, fill=f)[1]) for f in (False, True))
    looks = [s for s in filled if s["decision"]["action"] != "halt"]
    assert [s["decision"] for s in looks[:-1]] == [s["decision"] for s in plain if s["decision"]["action"] != "halt"]
    assert looks[-1]["decision"]["args"]["size"] == 256 and "what the pixel cap leaves" in looks[-1]["rationale"]
    assert plain[-1]["pixels_frac"] < filled[-1]["pixels_frac"] <= 0.3 and filled[-1]["cause"] == "cap-hit"


def test_tau_scout_sets_escalation_at_coarse_looks_only(scene):
    """tau_scout is the entropy threshold of a look at L2 or coarser; finer looks keep tau."""
    def acts(**kw):
        return [(s["decision"]["action"], s["decision"]["args"].get("level"), s["cause"]) for s in steps(run(scene, **kw)[1])]

    base = acts(tau_scout=0.5)  # tau at every level: the agent before tau_scout
    assert acts() == base and base[1][2] == "verify-new-mask"  # the L2 survey's entropy is 0.44, under 0.5 and 0.7
    assert acts(tau_scout=0.4)[1] == ("escalate", 1, "entropy>tau")  # ... so a lower scout threshold zooms on it
    # finer looks keep tau: the L1 look's entropy 0.80 is under a 0.9 scout threshold and still escalates to L0
    assert acts(tau_scout=0.9, pixel_cap=1.0) == acts(tau_scout=0.5, pixel_cap=1.0)
    assert ("escalate", 0, "entropy>tau") in acts(pixel_cap=1.0)


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
    full_done = list(full.run())[-1]
    for k in (1, 2, 3):
        holder = {}
        cut = holder["job"] = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(),
                                  kill=lambda h=holder, k=k: len(h["job"].gaze_path) >= k)
        trace = steps(list(cut.run()))
        replay = [[s["decision"]["action"], *(s["decision"]["args"][x] for x in ("x", "y", "level", "size"))]
                  for s in trace if s["decision"]["action"] != "halt"]
        resumed = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(), replay=replay)
        done = list(resumed.run())[-1]
        assert resumed.gaze_path == full.gaze_path
        assert done["coverage_looked"] == full_done["coverage_looked"]  # the replayed survey is no closer a look
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
    (na,) = [a for a in events[-1]["work_order"]["not_assessed"] if a["action"] == "re-shoot"]  # not the unvisited
    assert len(refused) == 1 and na["step"] == refused[0]["step"] == 2
    assert na["reason"] == "glare" and na["action"] == "re-shoot" and inside_image(na["box"], scene.shape)
    x, y, w, h = refused[0]["window"]
    bx, by, bw, bh = na["box"]
    assert x <= bx and y <= by and bx + bw <= x + w and by + bh <= y + h


@pytest.mark.parametrize("fallback", [True, False])
def test_a_photo_with_no_candidate_region_is_not_assessed_not_ready(fallback):
    """The team's wall photo IMG_2332 (public/samples/real_01.jpg): the overview proposes no candidate. With the
    fallback (the default) one whole-photo L2 look runs, without it no look; either way that is not a clean wall: the
    whole photo is not assessed and the work order is not ready. The job's own cause stays exhausted."""
    img = cv2.imread(str(Path(__file__).resolve().parent.parent / "public" / "samples" / "real_01.jpg"))
    H, W = img.shape[:2]
    _, events = run(img, fallback_l2=fallback)
    done, wo = events[-1], events[-1]["work_order"]
    assert events[1]["candidates"] == [] and (done["summary"]["steps"] > 0) == fallback and done["cause"] == "exhausted"
    assert {"step": 0, "box": [0, 0, W, H], "reason": "no candidate region was proposed",
            "action": "check-by-eye"} in wo["not_assessed"]
    assert wo["status"] in ("incomplete", "blocked")
    if not fallback:
        assert done["status"] == "completed" and len(wo["not_assessed"]) == 1 and done["coverage_looked"] == 0.0


def unvisited_not_assessed(done, shape):
    """The work order's not_assessed entries for candidates no look reached, checked against `unvisited`."""
    na = [a for a in done["work_order"]["not_assessed"] if "region" in a]
    assert all(a["action"] == "check-by-eye" and inside_image(a["box"], shape) for a in na)
    assert sorted(a["region"] for a in na) == sorted(r for r in done["unvisited"] if r.startswith("r"))
    return na


@pytest.mark.parametrize("after", [1, 2])
def test_a_killed_job_lists_every_unvisited_candidate_as_not_assessed(scene, after):
    """KILL after the survey look (and after one look past it): every candidate no look of its own reached goes to
    the work order as not assessed, with the halt as its reason, so the order is not ready; coverage_looked leaves
    out what only the survey saw, while coverage (any level, survey included) keeps its old meaning."""
    job = Job(scene, JobConfig(thumbs=False, pixel_cap=1.0), clock=FakeClock(),
              kill=lambda: len(job.gaze_path) >= after)
    events = list(job.run())
    done = events[-1]
    assert (done["status"], done["cause"]) == ("halted", "kill-switch") and steps(events)[0]["cause"] == "survey"
    na = unvisited_not_assessed(done, scene.shape)
    assert na and all(a["reason"] == "not looked at: kill-switch halted the job first" for a in na)
    assert done["work_order"]["status"] == "incomplete"
    assert done["coverage"] == 1.0 and done["coverage_looked"] < 1.0
    assert (done["coverage_looked"] == 0.0) == (after == 1)  # the survey alone looked at nothing closely


@pytest.mark.parametrize("cap", [0.1, 0.3])
def test_a_cap_hit_job_lists_every_unvisited_candidate_as_not_assessed(scene, cap):
    """A cap that halts the job (before any look at 0.1, after three at 0.3) leaves its unvisited candidates not
    assessed in the work order, with the cap as the reason."""
    _, events = run(scene, pixel_cap=cap)
    done = events[-1]
    assert (done["status"], done["cause"]) == ("halted", "cap-hit")
    na = unvisited_not_assessed(done, scene.shape)
    assert na and all(a["reason"] == "not looked at: cap-hit halted the job first" for a in na)
    assert done["work_order"]["status"] == "incomplete" and done["coverage_looked"] < 1.0


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
    absorption on), so no classical result moves (results/dev/onnx_profile_sweep.json: the byte-for-byte rerun).
    tau_scout, fill and survey_value stay off: measured, not adopted (results/dev/tune_sweep.json). fallback_l2 is on
    for every profile since its dev run (results/dev/zero_candidate_fallback.json, measured on v1); the classical numbers
    ran before it, with it off, and it changes only a photo with no candidate. v1-policy2's knobs are off for it,
    batch-yield's next_ev is 0 (off) for every profile, and so is the re-blur gate (results/dev/blur_gate.json)."""
    from dataclasses import asdict

    from drishyam.agent import PROFILES
    from drishyam.governor import USD_PER_HOUR
    from drishyam.planner import TIMEOUT_S

    assert PROFILES["classical-v1"] == {"min_view": 0.0, "absorb": True}
    job = Job(scene, JobConfig(), clock=FakeClock())  # the default segmenter here: the classical engine
    assert asdict(job.cfg) == {
        "pixel_cap": 0.3, "usd_cap": 0.05, "seconds_cap": 20.0, "size": 512, "first_level": 1, "min_level": 0,
        "min_view": 0.0, "profile": "classical-v1", "tau": 0.5, "tau_scout": None, "a_min": 64, "eps": 0.25,
        "sla_ms": 500.0, "max_candidates": 20, "chaos": (), "planner_every": 5, "planner_timeout_s": TIMEOUT_S,
        "usd_per_hour": USD_PER_HOUR, "thumbs": True, "max_steps": 200, "use_uncertainty": True,
        "seed_from_looks": True, "v0": 2.0, "survey": True, "verify": True, "absorb": True, "order": "score", "seed": 0,
        "governor": True, "fill": False, "survey_value": True,
        "t_hat_ms": None, "fallback_l2": True, "tau_confirm_lo": 0.35, "tau_confirm_hi": 0.65,
        "grid": False, "crit_weight": 1.0, "crit_floor": 0.0, "crit_confirm": "off", "merge_critical": False,
        "next_ev": 0.0, "blur_gate": "laplacian"}


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
    # pinned: the ONNX profile as it was measured (results/dev/onnx_profile_sweep.json), tau at every level
    assert PROFILES[ONNX] == {"min_view": 0.3, "absorb": False} and (cfg.tau, cfg.tau_scout) == (0.5, None)
    cfg = Job(scene, JobConfig(min_view=0.0), segmenter=named(ONNX)).cfg  # a dev sweep's variant
    assert (cfg.min_view, cfg.absorb) == (0.0, PROFILES[ONNX]["absorb"])
    cfg = Job(scene, JobConfig(profile="classical-v1"), segmenter=named(ONNX)).cfg
    assert (cfg.profile, cfg.min_view, cfg.absorb) == ("classical-v1", 0.0, True)
    # The eval harness wraps the segmenter to record its looks: the wrapper must carry the name and the weights, or
    # every ONNX eval would silently run the classical profile.
    monkeypatch.setattr(harness, "default_segmenter", lambda: named(ONNX))
    rec = harness.RecordingSegmenter()
    assert profile_for(rec) == ONNX and (rec.name, rec.sha256) == (ONNX, named(ONNX).sha256)


def test_the_governor_limit_is_per_profile_and_only_v1_has_its_own(scene):
    """agent.SLA_MS_BY_PROFILE: v1 on Lambda measured 0.56-1.14 s per look at p95, so its limit is 1500 ms and live jobs
    keep the 512-px net the held-out runs used; every other profile keeps 500 ms; an explicit sla_ms wins."""
    from drishyam.agent import SLA_MS, SLA_MS_BY_PROFILE

    assert SLA_MS == 500.0 and SLA_MS_BY_PROFILE == {ONNX: 1500.0}
    assert Job(scene, JobConfig(), segmenter=named(ONNX)).cfg.sla_ms == 1500.0
    assert Job(scene, JobConfig(), clock=FakeClock()).cfg.sla_ms == 500.0  # classical
    assert Job(scene, JobConfig(), segmenter=named("onnx:drishyam_dacl10k_v2a.onnx")).cfg.sla_ms == 500.0
    assert Job(scene, JobConfig(), segmenter=named(ONNX, sha256="0" * 64)).cfg.sla_ms == 500.0  # other weights
    assert Job(scene, JobConfig(sla_ms=250.0), segmenter=named(ONNX)).cfg.sla_ms == 250.0
    # The slow-segmenter drill still trips the governor under v1's limit: its injected look scales with the limit.
    job = Job(scene, JobConfig(chaos=("p95_spike",), pixel_cap=1.0, thumbs=False), segmenter=named(ONNX), clock=FakeClock())
    st = steps(list(job.run()))
    spike = next(s for s in st if s.get("injected") == "p95_spike")
    after = next(s for s in st if s["step"] == spike["step"] + 1)
    assert after["governor"]["cause"] == "p95>SLA" and "SLA 1500 ms" in after["governor"]["rationale"]


def test_the_v2a_profile_is_the_one_its_dev_sweep_chose_and_only_for_its_weights(scene):
    """results/dev/onnx_v2a_profile_sweep.json: no candidate passed the rule, so v2a keeps v1's values; v1's weights
    under v2a's file name (or v2a's under v1's) were not what either profile measured."""
    import json
    from pathlib import Path

    from drishyam.agent import PROFILES, WEIGHTS, profile_for

    v2a = "onnx:drishyam_dacl10k_v2a.onnx"
    assert PROFILES[v2a] == {"min_view": 0.3, "absorb": False} == PROFILES[ONNX]
    assert WEIGHTS[v2a] == "95b503c2d205567c09b19154ff77a5117889b7aba0b19733ed1c178dc4ab1ad1"
    assert profile_for(named(v2a)) == v2a
    assert profile_for(named(v2a, sha256=WEIGHTS[ONNX])) == profile_for(named(ONNX, sha256=WEIGHTS[v2a])) == "classical-v1"
    cfg = Job(scene, JobConfig(), segmenter=named(v2a)).cfg
    assert (cfg.profile, cfg.min_view, cfg.absorb, cfg.tau, cfg.tau_scout) == (v2a, 0.3, False, 0.5, None)
    sweep = json.loads((Path(__file__).parents[1] / "results/dev/onnx_v2a_profile_sweep.json").read_text())
    assert sweep["chosen"]["profile"][v2a] == PROFILES[v2a] and sweep["chosen_variant"] == "ref"
    assert not any(e["verdict"]["passes"] for e in sweep["evidence"].values())


def test_the_v2b_profile_is_the_one_its_dev_sweep_chose_and_only_for_its_weights(scene):
    """results/dev/onnx_v2b_profile_sweep.json: no candidate passed the rule, so v2b keeps v1's values too; the other
    two models' weights under v2b's file name (or v2b's under theirs) were not what any profile measured."""
    import json
    from pathlib import Path

    from drishyam.agent import PROFILES, WEIGHTS, profile_for

    v2a, v2b = "onnx:drishyam_dacl10k_v2a.onnx", "onnx:drishyam_dacl10k_v2b.onnx"
    assert PROFILES[v2b] == {"min_view": 0.3, "absorb": False} == PROFILES[ONNX]
    assert WEIGHTS[v2b] == "47e4b687d179efc9d221ef05eca441d24b0f025f9ab252c0a60c64455ff99207"
    assert profile_for(named(v2b)) == v2b
    assert {profile_for(named(v2b, sha256=WEIGHTS[m])) for m in (ONNX, v2a)} == {"classical-v1"}
    assert {profile_for(named(m, sha256=WEIGHTS[v2b])) for m in (ONNX, v2a)} == {"classical-v1"}
    cfg = Job(scene, JobConfig(), segmenter=named(v2b)).cfg
    assert (cfg.profile, cfg.min_view, cfg.absorb, cfg.tau, cfg.tau_scout) == (v2b, 0.3, False, 0.5, None)
    sweep = json.loads((Path(__file__).parents[1] / "results/dev/onnx_v2b_profile_sweep.json").read_text())
    assert sweep["chosen"]["profile"][v2b] == PROFILES[v2b] and sweep["chosen_variant"] == "ref"
    assert sweep["chosen"]["weights"][v2b] == WEIGHTS[v2b]
    assert not any(e["verdict"]["passes"] for e in sweep["evidence"].values())


def test_the_v2c_profile_is_the_one_its_dev_sweep_chose_and_only_for_its_weights(scene):
    """results/dev/onnx_v2c_profile_sweep.json: no candidate passed the rule, so v2c keeps v1's values too; the other
    three models' weights under v2c's file name (or v2c's under theirs) were not what any profile measured."""
    import json
    from pathlib import Path

    from drishyam.agent import PROFILES, WEIGHTS, profile_for

    others = (ONNX, "onnx:drishyam_dacl10k_v2a.onnx", "onnx:drishyam_dacl10k_v2b.onnx")
    v2c = "onnx:drishyam_dacl10k_v2c.onnx"
    assert PROFILES[v2c] == {"min_view": 0.3, "absorb": False} == PROFILES[ONNX]
    assert WEIGHTS[v2c] == "24b4d561fb012cb525131e16fa211b528a2e30c90fd9b9e76d98e29edbf18d21"
    assert profile_for(named(v2c)) == v2c
    assert {profile_for(named(v2c, sha256=WEIGHTS[m])) for m in others} == {"classical-v1"}
    assert {profile_for(named(m, sha256=WEIGHTS[v2c])) for m in others} == {"classical-v1"}
    cfg = Job(scene, JobConfig(), segmenter=named(v2c)).cfg
    assert (cfg.profile, cfg.min_view, cfg.absorb, cfg.tau, cfg.tau_scout) == (v2c, 0.3, False, 0.5, None)
    sweep = json.loads((Path(__file__).parents[1] / "results/dev/onnx_v2c_profile_sweep.json").read_text())
    assert sweep["chosen"]["profile"][v2c] == PROFILES[v2c] and sweep["chosen_variant"] == "ref"
    assert sweep["chosen"]["weights"][v2c] == WEIGHTS[v2c]
    assert not any(e["verdict"]["passes"] for e in sweep["evidence"].values())


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


# ---- zero-candidate fallback (JobConfig.fallback_l2) ---------------------------------------------------------------
FALLBACK_COSTS = {"pyrDown": 30.0, "saliency": 15.0, "warp": 2.0, "quality": 3.0, "segment:accurate": 560.0,
                  "segment:fast": 190.0}  # a fixed cost table (ms per call) for evals.harness.FixedCostJob
FOLLOW_UPS = {"entropy>tau", "area<A_min", "verify-new-mask", "iou<0.5"}  # the rules' own follow-up looks
SAMPLE_PHOTOS = sorted((Path(__file__).resolve().parent.parent / "public" / "samples").glob("*.jpg"))


def soft_disc(w, h):
    """A grainy grey photo with one soft dark disc: its saliency stays under the 0.5 threshold, so the overview proposes
    no candidate, while a segmenter that reads dark pixels (`Sure`) segments the disc; the grain passes the quality
    gate's sharpness test even at L2."""
    m = np.zeros((h, w), np.float32)
    cv2.circle(m, (w // 2, h // 3), w // 12, 1.0, -1)
    g = 170 - 80 * cv2.GaussianBlur(m, (0, 0), 20) + np.random.default_rng(0).normal(0, 16, (h, w))
    return np.repeat(np.clip(g, 0, 255)[..., None], 3, axis=2).astype(np.uint8)


def fixed_job(img, seg=None, kill=lambda: False, replay=None, **cfg):
    from evals.harness import FixedCostJob

    return FixedCostJob(img, JobConfig(thumbs=False, **cfg), FALLBACK_COSTS, segmenter=seg, kill=kill, replay=replay,
                        job_id="j_test")


def looks_of(events):
    return [s for s in steps(events) if s["decision"]["action"] != "halt"]


def covers(window, shape) -> bool:
    x, y, w, h = window
    return x <= 0 and y <= 0 and x + w >= shape[1] and y + h >= shape[0]


def test_with_the_flag_a_photo_with_no_candidate_gets_one_whole_photo_l2_look():
    """No candidate: with fallback_l2 the job spends one look at L2 whose window spans the whole photo, with the
    profile's accurate net, cause fallback-l2, its pixels and fixed-cost ms charged like any look's; what it drafts is
    reported like any look's finding (a critical one goes to a human) and the photo stays not assessed. Under v1's
    profile L2 is the finest level on a 4000-px photo, so it is the only look. Off: no look, as before."""
    from evals import harness

    img, (H, W) = soft_disc(4000, 3000), (3000, 4000)
    job = fixed_job(img, Sure(ONNX), fallback_l2=True)
    ev = list(job.run())
    ov, done = ev[1], ev[-1]
    assert ov["candidates"] == [] and job.cfg.profile == ONNX
    (s,) = looks_of(ev)
    assert (s["cause"], s["decision"]["action"], s["model"]) == ("fallback-l2", "gaze", "accurate")
    a = s["decision"]["args"]
    assert (a["level"], a["size"]) == (2, 1000) and covers(s["window"], img.shape)  # 1000 px at L2: 4000 L0 px
    assert s["pixels_used"] == ov["pixels_used"] + 1000**2
    assert s["seconds_used"] == pytest.approx((30 + 15 + 2 + 3 + 560) / 1000)
    (f,) = job.findings
    assert f.cls == "spalling" and f.step == s["step"] and f.critical and f.state == "pending_review"
    assert any(e["type"] == "review_request" and e["finding_id"] == f.id and e["kind"] == "critical" for e in ev)
    assert (done["status"], done["cause"]) == ("awaiting_review", "exhausted")
    assert {"step": 0, "box": [0, 0, W, H], "reason": "no candidate region was proposed",
            "action": "check-by-eye"} in done["work_order"]["not_assessed"] and done["work_order"]["status"] == "blocked"
    assert done["coverage_looked"] == 0.0  # one whole-photo look is not a close one
    off = list(fixed_job(img, Sure(ONNX), fallback_l2=False).run())
    assert looks_of(off) == [] and off[-1]["summary"]["findings"] == 0
    # the eval path (evals.dev_sweep, evals.harness): the switch reaches the job, and the look's boxes are scored
    r = harness.run_drishyam(img, 0.3, FALLBACK_COSTS, segmenter=Sure(ONNX), fallback_l2=True)
    assert r["switches"]["fallback_l2"] is True and [t["cause"] for t in r["trace"]] == ["fallback-l2", "exhausted"]
    assert r["preds"] and {p["cls"] for p in r["preds"]} == {"spalling"} and r["wall_s"] == pytest.approx(0.61)
    r = harness.run_drishyam(img, 0.3, FALLBACK_COSTS, segmenter=Sure(ONNX), fallback_l2=False)
    assert r["switches"]["fallback_l2"] is False and [t["cause"] for t in r["trace"]] == ["exhausted"]
    assert harness.run_drishyam(img, 0.3, FALLBACK_COSTS, segmenter=Sure(ONNX))["switches"]["fallback_l2"] is True


@pytest.mark.parametrize("name, size, cap", [("real_01", 750, 0.3), ("concrete_1024", 512, 1.0)])
def test_the_fallback_look_is_the_first_and_only_whole_photo_look_its_follow_ups_are_the_rules(name, size, cap):
    """The team's wall photo (public/samples/real_01.jpg) and a small plain one propose no candidate. With the flag,
    exactly one look has cause fallback-l2: the first, at L2, spanning the photo (750 px for the 3000-px photo; never
    a smaller gaze than 512 px, which at L2 already spans 2048 px, so on a 1024 x 768 photo it is 33 % of the pixels
    and only a cap above that lets it run); any other look is a follow-up the usual rules queue (escalate, verify) at
    the normal 512-px gaze."""
    from .conftest import concrete

    img = cv2.imread(str(SAMPLE_PHOTOS[0].parent / "real_01.jpg")) if name == "real_01" else concrete(768, 1024, 2)
    ev = run(img, fallback_l2=True, pixel_cap=cap)[1]
    st = looks_of(ev)
    assert ev[1]["candidates"] == [] and st[0]["cause"] == "fallback-l2"
    assert [s["cause"] for s in st].count("fallback-l2") == 1
    assert (st[0]["decision"]["args"]["level"], st[0]["decision"]["args"]["size"]) == (2, size)
    assert covers(st[0]["window"], img.shape)
    assert all(s["cause"] in FOLLOW_UPS and s["decision"]["args"]["size"] == 512 for s in st[1:])


def test_the_fallback_never_acts_on_a_photo_with_a_candidate(scene):
    """The fallback acts only when the overview proposes nothing: with a candidate, the flag changes no event."""
    import json

    on, off = (json.dumps(_no_wall_clock(run(scene, fallback_l2=f)[1]), sort_keys=True) for f in (True, False))
    assert on == off and '"fallback-l2"' not in on


@pytest.mark.parametrize("stop", [{"kill": True}, {"pixel_cap": 0.05}, {"seconds_cap": 0.04}, {"usd_cap": 1e-6}])
def test_the_fallback_look_obeys_kill_and_every_cap(stop):
    """The governor checks the fallback look like any other before it runs: KILL, or a pixel cap it would exceed (1000
    x 1000 px at L2 is 8.3 % of a 4000 x 3000 photo), or the seconds or $ caps already spent (the overview costs 45 ms
    on this table), halts the job with no look, and the whole photo stays not assessed."""
    stop = dict(stop)
    kill = stop.pop("kill", False)
    job = fixed_job(soft_disc(4000, 3000), Sure(ONNX), kill=lambda: kill, fallback_l2=True, **stop)
    ev = list(job.run())
    done = ev[-1]
    assert looks_of(ev) == [] and not job.findings
    assert (done["status"], done["cause"]) == ("halted", "kill-switch" if kill else "cap-hit")
    assert done["summary"]["pixels_used"] <= job.budget.pixels_cap
    assert done["work_order"]["not_assessed"][-1]["reason"] == "no candidate region was proposed"


def test_a_job_killed_after_the_fallback_look_resumes_into_the_same_job():
    """KILL after the fallback look, then resume with the trace's decisions ([action, x, y, level, size]): the replay
    accepts the look's own size (larger than a normal gaze), the follow-up it queues runs at 512 px, and the resumed
    job is the uninterrupted one, its photo still not assessed."""
    img = soft_disc(3000, 2250)  # L1 is the finest level under v1's profile here: the drafted mask is verified there
    full = fixed_job(img, Sure(ONNX), fallback_l2=True)
    full_done = list(full.run())[-1]
    assert [s["cause"] for s in full.trace][:2] == ["fallback-l2", "verify-new-mask"]
    holder = {}
    cut = holder["job"] = fixed_job(img, Sure(ONNX), kill=lambda: len(holder["job"].gaze_path) >= 1, fallback_l2=True)
    ev = list(cut.run())
    assert ev[-1]["cause"] == "kill-switch" and ev[-1]["coverage_looked"] == 0.0
    replay = [[s["decision"]["action"], *(s["decision"]["args"][k] for k in ("x", "y", "level", "size"))]
              for s in looks_of(ev)]
    assert replay == [["gaze", 1500, 1125, 2, 750]]
    resumed = fixed_job(img, Sure(ONNX), replay=replay, fallback_l2=True)
    done = list(resumed.run())[-1]
    assert resumed.gaze_path == full.gaze_path
    assert [s["decision"]["args"]["size"] for s in resumed.trace[:2]] == [750, 512]
    assert [(f.cls, f.box) for f in resumed.findings] == [(f.cls, f.box) for f in full.findings]
    assert done["work_order"]["not_assessed"] == full_done["work_order"]["not_assessed"]
    with pytest.raises(ValueError):  # only the whole-photo L2 look may be larger than a gaze
        validate_replay([["gaze", 1500, 1125, 1, 750]], img.shape)
    with pytest.raises(ValueError):
        validate_replay([["gaze", 1500, 1125, 2, 751]], img.shape)


def without_fallback(monkeypatch):
    """drishyam.agent without its lines marked `# fallback`: the agent before the zero-candidate fallback, provided
    every line the fallback added carries the mark (and no line it needs was edited)."""
    import re
    import sys
    import types

    from drishyam import agent

    src = Path(agent.__file__).read_text()
    kept = "".join(ln for ln in src.splitlines(keepends=True) if not re.search(r"#\s*fallback\b", ln))
    mod = types.ModuleType("drishyam._agent_without_fallback")
    mod.__package__ = "drishyam"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(kept, "drishyam/agent.py without the fallback", "exec"), mod.__dict__)  # noqa: S102 -- our own source
    assert "fallback_l2" in agent.JobConfig.__dataclass_fields__ and "fallback_l2" not in mod.JobConfig.__dataclass_fields__
    return mod


def _no_wall_clock(x):
    if isinstance(x, dict):
        return {k: _no_wall_clock(v) for k, v in x.items() if k not in ("t", "cpu_pct", "created")}
    return [_no_wall_clock(v) for v in x] if isinstance(x, list) else x


@pytest.mark.parametrize("name", [*(p.stem for p in SAMPLE_PHOTOS), "soft_disc", "soft_disc@v1"])
def test_with_the_flag_off_every_event_is_the_agents_before_the_fallback(name, monkeypatch):
    """Every bundled sample (real_01 has no candidate) and a synthetic photo with none, on the classical engine (and the
    disc under v1's profile): with fallback_l2 off, the whole event stream (thumbnails, timings on the job's clock)
    equals the agent's with the fallback's lines removed, the agent every number published before it ran with."""
    import json

    from drishyam import agent

    off = without_fallback(monkeypatch)
    paths = {p.stem: p for p in SAMPLE_PHOTOS}
    img = cv2.imread(str(paths[name])) if name in paths else soft_disc(3000, 2250)
    seg = Sure(ONNX) if name.endswith("@v1") else None
    streams = [json.dumps(_no_wall_clock(list(mod.Job(img, mod.JobConfig(**kw), clock=FakeClock(), job_id="j_test",
                                                       segmenter=seg).run())), sort_keys=True)
               for mod, kw in ((agent, {"fallback_l2": False}), (off, {}))]
    assert streams[0] == streams[1]


# ---- v1-policy2 (JobConfig grid, crit_weight, crit_floor, crit_confirm, merge_critical) ---------------------------
P2_OFF = {"grid": False, "crit_weight": 1.0, "crit_floor": 0.0, "crit_confirm": "off", "merge_critical": False}


def blots(w, h):
    """Concrete with two large dark discs: under `Unsure` each is a spalling mask far above the critical area at every
    level (about 106,000 L0 px on a 4608-px photo), at confidence 0.55."""
    from .conftest import concrete

    img = concrete(h, w, 5)
    for fx, fy in ((0.3, 0.4), (0.7, 0.62)):
        cv2.circle(img, (int(fx * w), int(fy * h)), w // 25, (40, 40, 40), -1)
    return img


class Blink(Unsure):
    """`Unsure` on odd calls, all background on even ones: the look right after a look that saw a mask sees nothing."""

    def __init__(self, name):
        super().__init__(name)
        self.n = 0

    def predict(self, crop, model="accurate"):
        self.n += 1
        if self.n % 2:
            return super().predict(crop, model)
        probs = np.zeros((len(CLASSES), *crop.shape[:2]), np.float32)
        probs[0] = 1
        return finish(probs, time.perf_counter(), model)


def needs_look(done):
    return [e for e in done["work_order"]["not_assessed"] if e.get("kind") == "needs-a-look"]


def test_the_grid_cells_are_the_tiling_baselines_tiles():
    """The grid a policy2 first look is placed on is evals.harness.slice_boxes (SAHI's 20 %-overlap 512-px tiles), the
    tiling baseline's own, on any level shape, including one smaller than a tile."""
    from drishyam.agent import _tiles
    from evals import harness

    for h, w in ((864, 1152), (384, 512), (1536, 2048), (1000, 1500), (300, 700), (512, 512)):
        assert _tiles(h, w, 512) == harness.slice_boxes(h, w, 512)


def test_with_grid_each_first_look_is_one_tiling_cell_that_holds_a_candidate():
    """grid: the regions are the cells of the first level's tiling grid that hold a candidate's centre (each candidate
    counted in the first cell, in tile order, that holds it), so no two first looks share a cell; every first look's
    window is its cell. Off: one region per candidate, as before."""
    from evals import harness

    img = spots(4608, 3456)
    job = fixed_job(img, Sure(ONNX), pixel_cap=1.0, grid=True)
    ev = list(job.run())
    cells = [[4 * x0, 4 * y0, 4 * (x1 - x0), 4 * (y1 - y0)] for x0, y0, x1, y1 in harness.slice_boxes(864, 1152)]
    inside = lambda c, b: b[0] <= c["box"][0] + c["box"][2] // 2 < b[0] + b[2] and b[1] <= c["box"][1] + c["box"][3] // 2 < b[1] + b[3]
    cands = ev[1]["candidates"]
    want = [b for b in cells if any(inside(c, b) and next(x for x in cells if inside(c, x)) == b for c in cands)]
    grid = [r for r in job.regions.values() if r.box is not None]
    assert cands and len(grid) < len(cands) and sorted(map(list, (r.box for r in grid))) == sorted(want)
    firsts = [s for s in looks_of(ev) if s["cause"] == "best-score"]
    assert firsts and all(s["window"] == list(job.regions[s["region"]].box) for s in firsts)
    assert len({s["region"] for s in firsts}) == len(firsts)
    off = fixed_job(img, Sure(ONNX), pixel_cap=1.0, grid=False)
    list(off.run())
    assert {r.id for r in off.regions.values() if r.box is not None} == {c["id"] for c in cands}


def test_grid_counts_each_candidate_in_the_first_tile_that_holds_it():
    """Tiles overlap by 20 %: a candidate in an overlap joins the first tile (in tile order) that holds it, once, so
    a tile whose only candidates an earlier tile took gets no region, and a region's prior is its own candidates'."""
    from drishyam.agent import _grid_regions
    from drishyam.imaging import Candidate

    cands = [Candidate("r1", 1800, 600, (0, 0, 1, 1), 0.9), Candidate("r2", 3000, 600, (0, 0, 1, 1), 0.6)]
    regions = _grid_regions(cands, (864, 1152), 2, 512)  # L2 of 4608 x 3456: tiles at L0 x 0, 1640, 2560 (2048 wide)
    assert {k: (r.prior, r.box, (r.cx, r.cy)) for k, r in regions.items()} == {
        "g1": (0.9, (0, 0, 2048, 2048), (1024, 1024)), "g2": (0.6, (1640, 0, 2048, 2048), (2664, 1024))}


def test_crit_weight_scales_the_look_score_of_a_region_seen_as_a_critical_class_only(scene):
    """crit_weight multiplies the severity in a region's expected value when the region was seen (seeded or looked at)
    as exposed_rebar or spalling, and leaves every other class, and an unseen region, as before."""
    from dataclasses import replace

    job = Job(scene, JobConfig(thumbs=False, crit_weight=1.0), clock=FakeClock())
    it = job.run()
    next(e for e in it if e["type"] == "step")
    r = next(r for r in job.regions.values() if not r.looks and job._score(r)[2] > 0)

    def ev(cls, w):
        r.cls, job.cfg = cls, replace(job.cfg, crit_weight=w)
        return job._score(r)[1]

    for cls, ratio in (("exposed_rebar", 2.5), ("spalling", 2.5), ("crack", 1), ("corrosion", 1), ("efflorescence", 1),
                       (None, 1)):
        assert ev(cls, 2.5) == pytest.approx(ratio * ev(cls, 1.0)) and ev(cls, 1.0) > 0


def rect_seg(cls, x0, y0, x1, y1, p=0.9):
    probs = np.zeros((len(CLASSES), 512, 512), np.float32)
    probs[0] = 1
    probs[0, y0:y1, x0:x1], probs[CLASSES.index(cls), y0:y1, x0:x1] = 1 - p, p
    return finish(probs, time.perf_counter(), "accurate")


@pytest.mark.parametrize("merge", [False, True])
def test_merge_critical_makes_a_critical_mask_inside_a_critical_finding_of_its_class_join_it(merge):
    """Two looks at two regions see one large spall: the coarser box holds the finer one (IoU 0.13, so the rule that
    joins same-class masks at IoU > 0.3 keeps them apart). merge_critical: the second joins the first, one critical
    finding for a person instead of two; a critical mask of another class, or a mask of the class under the critical
    area, never joins."""
    from drishyam.agent import Region
    from evals.harness import gaze_to_l0

    img = np.full((3000, 4000, 3), 128, np.uint8)
    job = Job(img, JobConfig(thumbs=False, merge_critical=merge, crit_floor=0.0), clock=FakeClock())
    crop = np.zeros((512, 512, 3), np.uint8)
    looks = [(Region("r1", 1000, 1000, 0.5, 1), "spalling", (100, 100, 300, 300), (1000, 1000, 1)),
             (Region("r2", 1500, 1500, 0.5, 2), "spalling", (31, 31, 306, 306), (1500, 1500, 2)),
             (Region("r3", 1500, 1500, 0.5, 2), "exposed_rebar", (31, 31, 306, 306), (1500, 1500, 2)),
             (Region("r4", 1000, 1000, 0.5, 1), "spalling", (150, 150, 240, 240), (1000, 1000, 1))]
    kinds = []
    for job.step, (r, cls, rect, (x, y, lv)) in enumerate(looks, start=1):
        kinds.append(job._log_finding(r, cls, rect_seg(cls, *rect), crop, gaze_to_l0(x, y, lv, 512), lv)[1])
    spall = [f for f in job.findings if f.cls == "spalling"]
    assert [f.critical for f in spall] == ([True] if merge else [True, True]) + [False]
    assert kinds == (["new", "update", "new", "new"] if merge else ["new", "new", "new", "new"])
    if merge:
        assert spall[0].steps == [1, 2] and spall[0].box[2] >= 1100
    assert [f.cls for f in job.findings if f.critical] == ["spalling"] * (1 if merge else 2) + ["exposed_rebar"]


def test_under_the_floor_a_critical_mask_goes_to_a_person_as_needs_a_look_not_as_a_critical_finding():
    """crit_floor without a confirm look: a mask that would make a critical finding (here spalling far above the
    critical area) but whose confidence (0.55) is under the floor is drafted as no finding at all; the work order lists
    it as not assessed, kind needs-a-look, with its class, confidence and box, action check-by-eye, so the order is
    never ready. A confident mask (0.97) is a critical finding as before; with the floor off, so is the unsure one."""
    img = blots(4608, 3456)
    job = fixed_job(img, Unsure(ONNX), pixel_cap=1.0, crit_floor=0.6, crit_confirm="off")
    ev = list(job.run())
    done, look = ev[-1], needs_look(ev[-1])
    assert look and all(e["cls"] == "spalling" and e["confidence"] == pytest.approx(0.55, abs=1e-3)
                        and e["action"] == "check-by-eye" and "under the floor 0.6" in e["reason"]
                        and e["box"][2] * e["box"][3] >= SPALL_CRITICAL_AREA_PX for e in look)
    assert {e["step"] for e in look} <= {s["step"] for s in looks_of(ev)}
    assert not any(f.critical for f in job.findings)
    assert not any(e["type"] == "review_request" and e["kind"] == "critical" for e in ev)
    assert done["work_order"]["status"] == "incomplete" and not any(s["cause"] == "confirm-critical" for s in job.trace)
    for seg, floor in ((Sure(ONNX), 0.6), (Unsure(ONNX), 0.0)):
        j = fixed_job(img, seg, pixel_cap=1.0, crit_floor=floor, crit_confirm="off")
        d = list(j.run())[-1]
        assert any(f.critical and f.cls == "spalling" for f in j.findings) and not needs_look(d)
        assert d["work_order"]["status"] == "blocked"


class UnsureRust(Unsure):
    """`Unsure`, plus a confident 40 x 40 px corrosion patch at the same place in every view: another class every
    look drafts."""

    def predict(self, crop, model="accurate"):
        out = super().predict(crop, model)
        probs = out.probs.copy()
        probs[:, 460:500, 460:500] = 0
        probs[0, 460:500, 460:500], probs[CLASSES.index("corrosion"), 460:500, 460:500] = 0.1, 0.9
        return finish(probs, time.perf_counter(), model)


def test_a_held_mask_gets_one_confirm_look_at_its_level_recentred_on_it_and_is_critical_when_it_agrees():
    """crit_confirm shift: a held mask queues one verify look (cause confirm-critical) at the same level, centred on the
    mask; when that look's mask of the class overlaps the held one (IoU >= 0.5) the class is drafted from it as a
    critical finding, verified, and sent to a person; a confirm look drafts no other class (here corrosion)."""
    img = blots(4608, 3456)
    job = fixed_job(img, UnsureRust(ONNX), pixel_cap=1.0, crit_floor=0.6, crit_confirm="shift")
    ev = list(job.run())
    st = looks_of(ev)
    conf = [s for s in st if s["cause"] == "confirm-critical"]
    assert conf and all(s["decision"]["action"] == "verify" and s["decision"]["args"]["level"] == 2 for s in conf)
    held = {s["step"]: s for s in st}
    for s in conf:  # the look that held the mask ran at the same level, and the confirm look is centred on the mask
        h = held[s["step"] - 1]
        assert h["decision"]["args"]["level"] == 2 and "held for a confirm look" in h["rationale"]
    crit = [f for f in job.findings if f.critical]
    assert crit and all(f.step in {s["step"] for s in conf} and f.verified and f.state == "pending_review" for f in crit)
    assert {f.id for f in crit} <= {e["finding_id"] for e in ev if e["type"] == "review_request" and e["kind"] == "critical"}
    assert not needs_look(ev[-1])
    assert all({f.cls for f in job.findings if s["step"] in f.steps} <= {"spalling"} for s in conf)


def test_a_held_mask_its_confirm_look_does_not_see_again_goes_to_a_person_as_needs_a_look():
    """The confirm look sees nothing where the held mask was (`Blink`): no critical finding is drafted from it, and
    the held mask is listed needs-a-look, saying its confirm look did not confirm it."""
    job = fixed_job(blots(4608, 3456), Blink(ONNX), pixel_cap=1.0, crit_floor=0.6, crit_confirm="shift")
    ev = list(job.run())
    conf = [s for s in looks_of(ev) if s["cause"] == "confirm-critical"]
    look = needs_look(ev[-1])
    assert conf and len(look) == len(conf) and not any(f.critical for f in job.findings)
    assert all("not confirmed by its L2 confirm look" in e["reason"] and e["step"] + 1 in {s["step"] for s in conf}
               for e in look)


class Shrink(Unsure):
    """`Unsure` on odd calls; on even ones a confident (0.97) spall on only the middle of each dark disc, under the
    critical area: the look right after a held mask confirms it by confidence but measures it smaller."""

    def __init__(self, name):
        super().__init__(name)
        self.n = 0

    def predict(self, crop, model="accurate"):
        self.n += 1
        if self.n % 2:
            return super().predict(crop, model)
        dark = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 105).astype(np.uint8)
        core = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (53, 53))).astype(np.float32)
        probs = np.zeros((len(CLASSES), *dark.shape), np.float32)
        probs[0], probs[CLASSES.index("spalling")] = 1 - 0.97 * core, 0.97 * core
        return finish(probs, time.perf_counter(), model)


def test_a_confirmed_mask_keeps_the_criticality_its_held_look_measured():
    """A confirm look that confirms a held spall but measures it under the critical area still sends it to a person:
    the finding is held critical by the area the held look measured (criticality only rises)."""
    job = fixed_job(blots(4608, 3456), Shrink(ONNX), pixel_cap=1.0, crit_floor=0.6, crit_confirm="shift")
    ev = list(job.run())
    conf = {s["step"] for s in looks_of(ev) if s["cause"] == "confirm-critical"}
    crit = [f for f in job.findings if f.critical]
    assert conf and crit and all(f.critical_step in conf and f.critical_area_px >= SPALL_CRITICAL_AREA_PX
                                 and f.measure["area_px"] < SPALL_CRITICAL_AREA_PX and f.state == "pending_review"
                                 for f in crit)
    assert {f.id for f in crit} <= {e["finding_id"] for e in ev if e["type"] == "review_request" and e["kind"] == "critical"}


def test_a_finer_confirm_look_goes_one_level_past_min_view_and_no_other_look_does():
    """crit_confirm finer: the confirm look runs one level finer than the look that held the mask, L1 on a 4608-px
    photo, although v1's min_view makes L2 the finest level for every other look."""
    job = fixed_job(blots(4608, 3456), Unsure(ONNX), pixel_cap=1.0, crit_floor=0.6, crit_confirm="finer")
    st = looks_of(list(job.run()))
    conf = [s for s in st if s["cause"] == "confirm-critical"]
    assert conf and all(s["decision"]["args"]["level"] == 1 for s in conf)
    assert all(s["decision"]["args"]["level"] >= 2 for s in st if s["cause"] != "confirm-critical")


def test_a_held_mask_whose_confirm_look_never_runs_goes_to_a_person():
    """A job that ends before a held mask's confirm look (here max_steps) lists the mask needs-a-look."""
    full = fixed_job(blots(4608, 3456), Unsure(ONNX), pixel_cap=1.0, crit_floor=0.6, crit_confirm="shift")
    first = next(s["step"] for s in looks_of(list(full.run())) if s["cause"] == "confirm-critical")
    cut = fixed_job(blots(4608, 3456), Unsure(ONNX), pixel_cap=1.0, crit_floor=0.6, crit_confirm="shift",
                    max_steps=first - 1)
    look = needs_look(list(cut.run())[-1])
    assert any(e["step"] == first - 1 and "the job ended before its confirm look" in e["reason"] for e in look)


def test_the_eval_scores_a_needs_a_look_mask_apart_from_the_drafted_boxes():
    """evals.harness: the boxes of needs-a-look masks (per component, as drafted boxes are) are `preds_look`, never in
    `preds`, so critical boxes count what goes to sign-off; the job's critical findings (sign-offs), review requests and
    needs-a-look entries are counted, and `preds_all` is both tiers, what a person is shown."""
    from drishyam.findings import is_critical
    from evals import harness

    img = blots(4608, 3456)
    on = harness.run_drishyam(img, 1.0, FALLBACK_COSTS, segmenter=Unsure(ONNX), **{**P2_OFF, "crit_floor": 0.6})
    off = harness.run_drishyam(img, 1.0, FALLBACK_COSTS, segmenter=Unsure(ONNX), **P2_OFF)
    assert on["preds_look"] and {p["cls"] for p in on["preds_look"]} == {"spalling"} and on["needs_look"] > 0
    assert not any(p["cls"] == "spalling" and is_critical(p["cls"], p["area"]) for p in on["preds"])
    assert on["preds_all"] == on["preds"] + on["preds_look"] and on["signoffs"] == 0
    assert off["preds_look"] == [] and off["needs_look"] == 0 and off["preds_all"] == off["preds"]
    assert 0 < off["signoffs"] <= sum(f["cls"] == "spalling" for f in off["preds_union"]) and off["reviews"] >= off["signoffs"]
    assert {k: on["switches"][k] for k in P2_OFF} == {**P2_OFF, "crit_floor": 0.6}


def test_v1_runs_the_policy2_its_dev_rule_chose_and_no_other_profile_does(scene):
    """v1's profile (its name and released weights) takes g_s60, the configuration results/dev/policy2_sweep.json
    chose; the classical engine, v2a-v2c and v1's name with other weights keep every knob off; a knob set explicitly
    wins."""
    import json

    from drishyam.agent import POLICY2, POLICY2_OFF, WEIGHTS
    from evals import policy2

    chosen = json.loads((Path(__file__).resolve().parents[1] / "results" / "dev" / "policy2_sweep.json").read_text())
    assert chosen["verdict"]["adopted"] and POLICY2 == {ONNX: policy2.ELIGIBLE[chosen["verdict"]["chosen"]]}
    knobs = lambda seg=None, **kw: {k: getattr(Job(scene, JobConfig(**kw), segmenter=seg).cfg, k) for k in POLICY2_OFF}
    assert knobs(Sure(ONNX)) == POLICY2[ONNX] and knobs() == POLICY2_OFF
    for name in WEIGHTS:
        if name != ONNX:
            assert knobs(Sure(name)) == POLICY2_OFF
    other = Sure(ONNX)
    other.sha256 = "0" * 64
    assert knobs(other) == POLICY2_OFF
    assert knobs(Sure(ONNX), crit_floor=0.0, grid=False) == {**POLICY2[ONNX], "crit_floor": 0.0, "grid": False}


def without_policy2(monkeypatch):
    """drishyam.agent without its lines marked `# policy2`: the agent before v1-policy2."""
    import re
    import sys
    import types

    from drishyam import agent

    src = Path(agent.__file__).read_text()
    kept = "".join(ln for ln in src.splitlines(keepends=True) if not re.search(r"#\s*policy2\b", ln))
    assert kept != src
    mod = types.ModuleType("drishyam._agent_without_policy2")
    mod.__package__ = "drishyam"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(kept, "drishyam/agent.py without policy2", "exec"), mod.__dict__)  # noqa: S102 -- our own source
    assert "crit_floor" in agent.JobConfig.__dataclass_fields__ and "crit_floor" not in mod.JobConfig.__dataclass_fields__
    return mod


@pytest.mark.parametrize("name", [*(p.stem for p in SAMPLE_PHOTOS), "blots@v1", "spots@v1", "soft_disc@v1"])
def test_with_policy2_off_every_event_is_the_agents_before_policy2(name, monkeypatch):
    """Every bundled sample on the classical engine, and three synthetic photos under v1's profile: with every policy2
    knob off, the whole event stream equals the agent's with policy2's lines removed."""
    import json

    from drishyam import agent

    off = without_policy2(monkeypatch)
    paths = {p.stem: p for p in SAMPLE_PHOTOS}
    img = (cv2.imread(str(paths[name])) if name in paths else
           {"blots": blots, "spots": spots, "soft_disc": soft_disc}[name.split("@")[0]](3000, 2250))
    seg = (Unsure if name.startswith("blots") else Sure)(ONNX) if name.endswith("@v1") else None
    streams = [json.dumps(_no_wall_clock(list(mod.Job(img, mod.JobConfig(**kw), clock=FakeClock(), job_id="j_test",
                                                       segmenter=seg).run())), sort_keys=True)
               for mod, kw in ((agent, P2_OFF), (off, {}))]
    assert streams[0] == streams[1]


# ---- batch-yield (JobConfig.next_ev): the patch-leaving rule `drishyam batch --batch-yield` sets under pressure -----
def without_batch_yield(monkeypatch):
    """drishyam.agent without its lines marked `# batch-yield`: the agent before the rule."""
    import re
    import sys
    import types

    from drishyam import agent

    src = Path(agent.__file__).read_text()
    kept = "".join(ln for ln in src.splitlines(keepends=True) if not re.search(r"#\s*batch-yield\b", ln))
    assert kept != src
    mod = types.ModuleType("drishyam._agent_without_batch_yield")
    mod.__package__ = "drishyam"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(kept, "drishyam/agent.py without batch-yield", "exec"), mod.__dict__)  # noqa: S102 -- our own source
    assert "next_ev" in agent.JobConfig.__dataclass_fields__ and "next_ev" not in mod.JobConfig.__dataclass_fields__
    return mod


def without_value(events):
    """The stream with the `value` field taken off the step records, once it is checked to be on exactly the
    best-score ones (the only trace change next_ev 0 makes)."""
    out = []
    for e in events:
        if e["type"] == "step":
            assert ("value" in e) == (e["cause"] == "best-score"), (e["cause"], sorted(e))
            e = {k: v for k, v in e.items() if k != "value"}
        out.append(e)
    return out


@pytest.mark.parametrize("name", [*(p.stem for p in SAMPLE_PHOTOS), "blots@v1", "spots@v1", "soft_disc@v1"])
def test_with_next_ev_0_every_event_is_the_agents_before_batch_yield_but_the_value_field(name, monkeypatch):
    """Every bundled sample on the classical engine, and three synthetic photos under v1's profile (policy2 on): at
    next_ev 0, the default, the whole event stream equals the agent's with batch-yield's lines removed, apart from the
    `value` field on best-score steps."""
    import json

    from drishyam import agent

    off = without_batch_yield(monkeypatch)
    paths = {p.stem: p for p in SAMPLE_PHOTOS}
    img = (cv2.imread(str(paths[name])) if name in paths else
           {"blots": blots, "spots": spots, "soft_disc": soft_disc}[name.split("@")[0]](3000, 2250))
    seg = (Unsure if name.startswith("blots") else Sure)(ONNX) if name.endswith("@v1") else None
    ours, before = (list(mod.Job(img, mod.JobConfig(), clock=FakeClock(), job_id="j_test", segmenter=seg).run())
                    for mod in (agent, off))
    assert any(s["cause"] == "best-score" for s in steps(ours)) or not ours[1]["candidates"]
    assert (json.dumps(_no_wall_clock(without_value(ours)), sort_keys=True)
            == json.dumps(_no_wall_clock(before), sort_keys=True))


def yield_job(img, seg=None, **cfg):
    job = Job(img, JobConfig(thumbs=False, **cfg), clock=FakeClock(), job_id="j_test", segmenter=seg)
    return job, list(job.run())


def test_a_next_ev_above_every_prior_halts_after_one_look_and_lists_what_it_skipped():
    """next_ev 100 is above any look's value (a prior is at most 1). The first look still runs (no look yet), then the
    next best-score look is worth less: the job halts, cause batch-yield, its rationale giving both values, and every
    candidate it did not look at is not assessed, so nothing it skipped reads as clean."""
    import re

    img = spots(4608, 3456)
    full = looks_of(yield_job(img, Sure(ONNX), survey=False, verify=False)[1])
    _, events = yield_job(img, Sure(ONNX), survey=False, verify=False, next_ev=100.0)
    looks, done = looks_of(events), events[-1]
    (halt,) = [s for s in steps(events) if s["decision"]["action"] == "halt"]
    assert [s["cause"] for s in looks] == ["best-score"] and _no_wall_clock(looks) == _no_wall_clock(full[:1])
    assert (done["status"], done["cause"], halt["cause"]) == ("halted", "batch-yield", "batch-yield")
    value, nev = map(float, re.fullmatch(r"best look's value (\S+) < next_ev (\S+) \(the next photo's first look's "
                                         r"value\)", halt["rationale"]).groups())
    assert nev == 100.0 and value == pytest.approx(full[1]["value"], rel=1e-3) and full[1]["cause"] == "best-score"
    na = [a for a in done["work_order"]["not_assessed"] if "region" in a]
    assert len(na) >= 4 and sorted(a["region"] for a in na) == sorted(done["unvisited"])
    assert all(a["reason"] == "not looked at: batch-yield halted the job first" and a["action"] == "check-by-eye"
               for a in na)
    assert done["work_order"]["status"] == "incomplete"


@pytest.mark.parametrize("case", ["scene", "scene-no-survey", "blots@v1"])
def test_batch_yield_never_cuts_a_forced_look(scene, case):
    """At next_ev 100 every look the full job takes before its second best-score choice still runs: the survey and
    the rules' follow-ups after it (verify-new-mask, entropy>tau, v1-policy2's confirm-critical; each case has at
    least one), and the halt comes only where the argmax would have chosen."""
    img, seg, kw = {"scene": (scene, None, {}), "scene-no-survey": (scene, None, {"survey": False}),
                    "blots@v1": (blots(4608, 3456), Unsure(ONNX), {"survey": False})}[case]
    full, cut = (looks_of(yield_job(img, seg, next_ev=v, **kw)[1]) for v in (0.0, 100.0))
    m = next(i for i, s in enumerate(full) if i and s["cause"] == "best-score")
    assert m >= 2 and all(s["cause"] != "best-score" for s in cut[1:])
    assert _no_wall_clock(cut) == _no_wall_clock(full[:m])


def test_batch_yield_halts_only_on_a_best_score_look_strictly_below_next_ev():
    """next_ev equal to the least value of any best-score look after the first: no look is below it, so the job runs
    as at next_ev 0 to its own end; the next float above it halts exactly at that look."""
    img = spots(4608, 3456)
    full = looks_of(yield_job(img, Sure(ONNX), survey=False, verify=False)[1])
    vals = [s["value"] for s in full[1:] if s["cause"] == "best-score"]
    low = min(vals)
    at = next(i for i, s in enumerate(full) if i and s.get("value") == low)
    assert 1 < at and vals.count(low) == 1
    _, same = yield_job(img, Sure(ONNX), survey=False, verify=False, next_ev=low)
    assert _no_wall_clock(looks_of(same)) == _no_wall_clock(full) and same[-1]["cause"] == "exhausted"
    _, cut = yield_job(img, Sure(ONNX), survey=False, verify=False, next_ev=float(np.nextafter(low, 1.0)))
    assert _no_wall_clock(looks_of(cut)) == _no_wall_clock(full[:at]) and cut[-1]["cause"] == "batch-yield"


def test_the_next_photos_first_look_is_worth_its_triage_prior_max_under_v1_grid():
    """next_ev is the next photo's triage prior_max: its first look's value with severity 1, entropy 0 and unseen 1.
    Under v1-policy2 the regions are grid tiles whose prior is their best candidate's, so the best tile's prior is
    still the photo's best candidate prior, and with nothing seen yet (no survey) the first look is worth exactly that.
    (With the survey, which v1 runs, the first best-score look's value also carries what the survey seeded.)"""
    from drishyam import batch

    img = spots(4608, 3456)
    tri = batch.triage(img, JobConfig(thumbs=False),
                       lambda im, cfg, **kw: Job(im, cfg, clock=FakeClock(), segmenter=Sure(ONNX), **kw))
    job, events = yield_job(img, Sure(ONNX), survey=False)
    grid = [r for r in job.regions.values() if r.box is not None]
    assert job.cfg.grid and grid and all(r.id.startswith("g") for r in grid) and len(grid) < tri["candidates"]
    assert max(r.prior for r in grid) == tri["prior_max"] > 0
    assert looks_of(events)[0]["cause"] == "best-score" and looks_of(events)[0]["value"] == tri["prior_max"]


# ---- blur gate (JobConfig.blur_gate): the quality gate's Laplacian variance, or the re-blur ratio (imaging.reblur) --
def without_blur_gate(monkeypatch):
    """drishyam.agent without its lines marked `# blur-gate`: the agent before the knob."""
    import re
    import sys
    import types

    from drishyam import agent

    src = Path(agent.__file__).read_text()
    kept = "".join(ln for ln in src.splitlines(keepends=True) if not re.search(r"#\s*blur-gate\b", ln))
    assert kept != src
    mod = types.ModuleType("drishyam._agent_without_blur_gate")
    mod.__package__ = "drishyam"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(kept, "drishyam/agent.py without the blur gate", "exec"), mod.__dict__)  # noqa: S102 -- our own source
    assert "blur_gate" in agent.JobConfig.__dataclass_fields__ and "blur_gate" not in mod.JobConfig.__dataclass_fields__
    return mod


@pytest.mark.parametrize("name", [*(p.stem for p in SAMPLE_PHOTOS), "blots@v1", "spots@v1", "soft_disc@v1"])
def test_with_the_laplacian_gate_every_event_is_the_agents_before_the_knob(name, monkeypatch):
    """Every bundled sample on the classical engine, and three synthetic photos under v1's profile: with blur_gate
    "laplacian", the default, the whole event stream equals the agent's with the knob's lines removed."""
    import json

    from drishyam import agent

    off = without_blur_gate(monkeypatch)
    paths = {p.stem: p for p in SAMPLE_PHOTOS}
    img = (cv2.imread(str(paths[name])) if name in paths else
           {"blots": blots, "spots": spots, "soft_disc": soft_disc}[name.split("@")[0]](3000, 2250))
    seg = (Unsure if name.startswith("blots") else Sure)(ONNX) if name.endswith("@v1") else None
    streams = [json.dumps(_no_wall_clock(list(mod.Job(img, mod.JobConfig(**kw), clock=FakeClock(), job_id="j_test",
                                                       segmenter=seg).run())), sort_keys=True)
               for mod, kw in ((agent, {}), (agent, {"blur_gate": "laplacian"}), (off, {}))]
    assert streams[0] == streams[1] == streams[2]


def faint_wall(w, h):
    """Sharp concrete grain at a fifth of `concrete`'s contrast: a smooth painted wall in flat light, in focus."""
    from .conftest import concrete

    g = concrete(h, w, 4).astype(np.float32)
    return np.clip(g.mean() + (g - g.mean()) * 0.2, 0, 255).astype(np.uint8)


def test_the_reblur_gate_keeps_a_sharp_faint_wall_and_still_refuses_a_blurred_one():
    """real_01's case without real_01: every look at a sharp, faint wall (it has no candidate: one whole-photo L2 look)
    is refused as blur by the Laplacian gate (a not-assessed entry, re-shoot) and passed by the re-blur gate, whose
    ratio each step then records. A wall at full contrast blurred by sigma 8 L0 px (1 view px at L3, 4 at L1) is
    refused at every look by both."""
    from evals import harness

    from .conftest import concrete

    sharp = faint_wall(4000, 3000)
    for img, passes in ((sharp, True), (cv2.GaussianBlur(concrete(3000, 4000, 4), (0, 0), 8.0), False)):
        for gate in ("laplacian", "reblur"):
            ev = list(fixed_job(img, blur_gate=gate).run())
            looks, refused = looks_of(ev), [e for e in ev[-1]["work_order"]["not_assessed"] if e["reason"] == "blur"]
            assert looks and all(("reblur" in s["observation"]) == (gate == "reblur") for s in looks)
            if passes and gate == "reblur":
                assert all(s["quality"]["ok"] for s in looks) and not refused
            else:
                assert not any(s["quality"]["ok"] for s in looks) and len(refused) == len(looks)
                assert all(e["action"] == "re-shoot" for e in refused)
    r = harness.run_drishyam(sharp, 0.3, FALLBACK_COSTS, blur_gate="reblur")
    assert r["switches"]["blur_gate"] == "reblur" and all(t["q_ok"] for t in r["trace"])
    assert harness.run_drishyam(sharp, 0.3, FALLBACK_COSTS)["switches"]["blur_gate"] == "laplacian"
