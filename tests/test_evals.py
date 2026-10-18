"""Evaluation harness, CODEBRIM loader and parity tool: matching rule, tiling accounting, an end-to-end tiny run."""
import gzip
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from bench import parity
from drishyam.segment import CLASSES, finish
from evals import codebrim, failures, harness, synth


class BlankSegmenter:
    """Predicts 'all background' instantly, so pixel accounting is tested without paying for segmentation."""

    def predict(self, crop, model="accurate"):
        h, w = crop.shape[:2]
        probs = np.zeros((len(CLASSES), h, w), np.float32)
        probs[0] = 1
        out = finish(probs, 0.0, model)
        out.net_px = None  # an engine that does not say what it computed on: costs fall back to pixels read
        return out


class NetSegmenter(BlankSegmenter):
    """Feeds every crop to a 64x64 network input, as an ONNX net resizes to its own, and says so (net_px)."""

    def predict(self, crop, model="accurate"):
        out = super().predict(crop, model)
        out.net_px = 64 * 64
        return out


# ---------------------------------------------------------------- matching
def test_match_by_iou_or_centre_and_class():
    gt = {"cls": "crack", "box": [100, 100, 100, 100]}
    p = lambda cls, box: [{"cls": cls, "box": box}]
    assert harness.found(gt, p("crack", [100, 100, 100, 100]))  # perfect overlap
    assert harness.found(gt, p("crack", [150, 100, 100, 100]))  # IoU 1/3 >= 0.3
    assert not harness.found(gt, p("crack", [170, 100, 100, 100]))  # IoU 0.18 and centre (150,150) outside
    assert harness.found(gt, p("crack", [0, 0, 400, 400]))  # IoU tiny, but it contains the GT centre
    assert not harness.found(gt, [])  # nothing predicted
    assert not harness.found(gt, p("spalling", [100, 100, 100, 100]))  # right place, wrong class
    assert not harness.found(gt, p("crack", [0, 0, 400, 400]), centre=False)  # strict variant ignores containment
    assert harness.hits([140, 140, 20, 20], [100, 100, 100, 100]) and not harness.hits([0, 0, 50, 50], [100, 100, 100, 100])


def test_critical_false_accepts():
    rebar = {"cls": "exposed_rebar", "box": [0, 0, 100, 100]}
    small_spall = {"cls": "spalling", "box": [300, 0, 100, 100]}  # box 10,000 px^2: below the critical area
    big_spall = {"cls": "spalling", "box": [500, 0, 300, 300]}
    gts = [rebar, small_spall, big_spall]
    preds = [{"cls": "exposed_rebar", "box": [0, 0, 100, 100], "area": 5000},  # matches rebar GT
             {"cls": "exposed_rebar", "box": [900, 900, 50, 50], "area": 100},  # nothing there
             {"cls": "spalling", "box": [300, 0, 100, 100], "area": 60_000},  # critical by area, GT spall is not
             {"cls": "spalling", "box": [500, 0, 300, 300], "area": 60_000},  # matches the big spall
             {"cls": "crack", "box": [0, 0, 10, 10], "area": 10}]  # never critical
    assert harness.critical_counts(preds, gts) == (4, 2)


def test_chance_baseline_precision_and_intervals():
    """Each policy's own boxes moved to random places must score below the real thing; precision, FP/image,
    strata and the paired bootstrap come out of the same per-image counts."""
    gts = [{"cls": "crack", "box": [100 + 7 * i, 200, 60, 40], "hairline": i % 2 == 0} for i in range(20)]
    records = [{"id": f"s{i}", "boxes": [g]} for i, g in enumerate(gts)]
    hit = lambda g: {"cls": "crack", "box": list(g["box"]), "area": 100, "conf": 0.9}
    miss = {"cls": "crack", "box": [900, 900, 30, 30], "area": 100, "conf": 0.5}
    results = [{"width": 2000, "height": 1500, "configs": {
        "good": {"preds": [hit(g), miss], "pixels_frac": 0.3, "wall_s": 0.1},
        "half": {"preds": [hit(g)] if i % 2 else [miss], "pixels_frac": 0.3, "wall_s": 0.1},
        "dup": {"preds": [hit(g), {**hit(g), "conf": 0.8}], "pixels_frac": 0.3, "wall_s": 0.1}}} for i, g in enumerate(gts)]
    good, half, dup = (harness.score_config(k, records, results) for k in ("good", "half", "dup"))
    assert good["recall"] == 1.0 and good["chance_recall"] < 0.2 and good["chance_recall_iou_only"] <= good["chance_recall"]
    assert good["precision"] == 0.5 and good["fp_per_image"] == 1.0 and good["predictions_per_image"] == 2
    # one GT box re-reported twice: one match, one false positive (one-to-one), not two hits
    assert dup["precision"] == 0.5 and dup["fp_per_image"] == 1.0 and dup["hits_per_matched_gt"] == 2.0
    assert good["hits_per_matched_gt"] == 1.0
    assert half["recall"] == 0.5 and half["strata"] == {"hairline_crack": 0.0, "small_box": 0.5}
    assert good["strata_gt"] == {"hairline_crack": 10, "small_box": 20}
    lo, hi = harness.boot_ci(harness.recall_of(half), len(records))
    assert lo < 0.5 < hi and harness.boot_ci(harness.recall_of(good), len(records)) == [1.0, 1.0]
    assert harness.ratio_ci(half, good, len(records)) == [lo, hi]  # paired: the same resamples for both policies
    assert harness.score_config("good", records, results)["chance_recall"] == good["chance_recall"]  # seeded


def test_target_met_on_the_point_estimate_but_not_by_the_interval():
    """19 of 20 boxes against tiling's 20 of 20 is 0.95x, but a bootstrap over 20 images reaches below 0.9x."""
    row = lambda f: [f, f, 0, 0, 1]
    pol = lambda found, px: {"recall": sum(found) / 20, "recall_iou_only": sum(found) / 20, "pixels_frac": px,
                             "per_image": [row(f) for f in found]}
    m = {"drishyam@0.3": pol([1] * 19 + [0], 0.3), "tiling@L0": pol([1] * 20, 1.67), "downscale": pol([1] * 20, 0.25)}
    hs = harness.headline_stats(m, 20)
    assert hs["recall_ratio_vs_tiling"] == 0.95 and hs["meets_target"] and not hs["meets_target_ci"]
    assert hs["recall_ratio_vs_tiling_ci"][0] < 0.9 and hs["recall_ratio_iou_only_vs_tiling"] == 0.95


def test_fixed_cost_model_is_load_independent(scene):
    """--cost-model fixed: every timed op costs its table value, so two runs of one image make identical decisions."""
    rows = harness.cost_rows("local-macos-x86", "classical-v1")
    costs = harness.op_costs(rows, scene.shape[1], scene.shape[0])
    a, b = (harness.run_drishyam(scene, 1.0, costs) for _ in range(2))
    assert a["trace"] == b["trace"] and a["preds"] == b["preds"] and a["wall_s"] == b["wall_s"] > 0
    looks = sum(t["action"] != "halt" for t in a["trace"])
    per_look = costs["warp"] + costs["quality"] + costs["segment:accurate"]
    assert a["wall_s"] == pytest.approx((costs["pyrDown"] + costs["saliency"] + looks * per_look) / 1000)
    big = harness.op_costs(rows, 4608, 3456)  # nearest timed size 4000x3000: per-image ops scale with the pixels
    ref = {r["op"]: r["p50_ms"] for r in rows if r["size"] == "4000x3000"}
    assert big["pyrDown"] == pytest.approx(ref["pyrDown"] * 4608 * 3456 / 12e6) and big["warp"] == ref["warp"]
    with pytest.raises(SystemExit, match="segments with onnx"):
        harness.cost_rows("local-macos-x86", "onnx:m.onnx")


# ---------------------------------------------------------------- tiling and downscale accounting
@pytest.mark.parametrize("h,w", [(1536, 2048), (768, 1024), (500, 600), (300, 400), (3000, 4000)])
def test_tiles_cover_the_image_with_the_right_overlap(h, w):
    boxes = harness.slice_boxes(h, w)
    cover = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert x1 - x0 == min(512, w) and y1 - y0 == min(512, h)  # edge tiles are pulled back, not shrunk
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    xs = sorted({b[0] for b in boxes})
    if w >= 922:  # at least three columns, so the second tile is a regular one and not the pulled-back edge tile
        assert xs[1] - xs[0] == 512 - int(0.2 * 512)  # stride = tile x (1 - overlap): a 20 % overlap
    assert len(harness.slice_boxes(1536, 2048)) == 20  # 5 columns x 4 rows


def test_tiling_pixel_accounting():
    img = np.zeros((768, 1024, 3), np.uint8)
    l0 = harness.run_tiling(img, BlankSegmenter(), 0)
    tiles = harness.slice_boxes(768, 1024)
    assert len(tiles) == 6
    assert l0["pixels_frac"] == pytest.approx(len(tiles) * 512 * 512 / (768 * 1024))  # overlap counted
    l1 = harness.run_tiling(img, BlankSegmenter(), 1)  # level 1 is 512x384: one 512x384 tile
    assert l1["pixels_frac"] == pytest.approx(512 * 384 / (768 * 1024))
    assert l1["preds"] == [] and l0["wall_s"] >= 0


def test_downscale_pixel_accounting():
    big = harness.run_downscale(np.zeros((1500, 2000, 3), np.uint8), BlankSegmenter())
    assert big["pixels_frac"] == pytest.approx(1024 * 768 / (1500 * 2000), rel=1e-3)  # 1024 px on the long side
    small = harness.run_downscale(np.zeros((300, 400, 3), np.uint8), BlankSegmenter())
    assert small["pixels_frac"] == 1.0  # never upscaled


def test_net_pixels_are_reported_and_charged_when_the_segmenter_gives_them():
    img = np.zeros((768, 1024, 3), np.uint8)
    costs = {"segment:accurate": 100.0, "downscale": 5.0, "pyrDown": 5.0}
    blank, net = (harness.run_downscale(img, s, costs) for s in (BlankSegmenter(), NetSegmenter()))
    assert blank["net_pixels_frac"] is None and blank["wall_s"] == pytest.approx(0.1 * 768 * 1024 / 512**2)  # by pixels read
    assert net["pixels_frac"] == 1.0 and net["net_pixels_frac"] == pytest.approx(64 * 64 / (768 * 1024))
    assert net["wall_s"] == pytest.approx(0.1 * 64 * 64 / 512**2)  # charged what the net was fed
    tiles = harness.run_tiling(img, NetSegmenter(), 0, costs)
    assert tiles["net_pixels_frac"] == pytest.approx(6 * 64 * 64 / (768 * 1024)) and tiles["models"] == {"accurate": 6}
    assert harness.net_px([NetSegmenter().predict(img), BlankSegmenter().predict(img)]) is None  # one unknown: unknown


def test_predictions_map_back_to_l0():
    h, w = 128, 128
    probs = np.zeros((len(CLASSES), h, w), np.float32)
    probs[0] = 1
    probs[0, 20:40, 30:70], probs[1, 20:40, 30:70] = 0, 1  # a crack blob at crop (30, 20) 40x20
    seg = finish(probs, 0.0, "accurate")
    to_l0 = np.array([[2.0, 0, 2 * 100], [0, 2.0, 2 * 50], [0, 0, 1]])  # tile at (100, 50) of pyramid level 1
    (comp,) = harness.preds_from_seg(seg, to_l0)
    assert comp["cls"] == "crack" and comp["box"] == [2 * 130, 2 * 70, 80, 40] and comp["area"] == 4 * 800
    assert harness.union_from_seg(seg, to_l0)[0]["box"] == comp["box"]


def test_drishyam_components_reproduce_finding_boxes(scene):
    """The rebuilt per-component boxes of every look in a finding's trail must sit inside its own (union) box."""
    rec = harness.RecordingSegmenter()
    job = harness.Job(scene, harness.JobConfig(pixel_cap=0.3, thumbs=False), segmenter=rec)
    list(job.run())
    comps = harness.finding_components(job, rec.log)
    assert job.findings and comps
    for c in comps:
        f = next(f for f in job.findings if f.cls == c["cls"] and c["step"] in f.steps)
        fx, fy, fw, fh = f.box
        assert c["box"][0] >= fx - 2 and c["box"][1] >= fy - 2
        assert c["box"][0] + c["box"][2] <= fx + fw + 2 and c["box"][1] + c["box"][3] <= fy + fh + 2


def test_drishyam_components_leave_out_what_the_job_dropped_as_a_straight_line(monkeypatch):
    """bridge_08's formwork groove is dropped as a straight line on every look that sees it. With the pore rule off,
    two bug holes become crack findings, one on a look that also saw the groove: the rebuilt per-component boxes
    hold the bug holes and never the groove the trace lists as rejected."""
    from drishyam import imaging, segment

    monkeypatch.setattr(segment, "PORE_ASPECT", 0.0)
    img = cv2.imread(str(harness.REPO / "public" / "samples" / "bridge_08.jpg"))
    costs = harness.op_costs(harness.cost_rows("local-macos-x86", "classical-v1"), img.shape[1], img.shape[0])
    rec = harness.RecordingSegmenter()
    job = harness.FixedCostJob(img, harness.JobConfig(pixel_cap=0.3, thumbs=False), costs, segmenter=rec)
    list(job.run())
    comps = harness.finding_components(job, rec.log)
    rejected = {r["step"]: [x["box"] for x in r["rejected"]] for r in job.trace if r.get("rejected")}
    assert comps and any(s in rejected for f in job.findings for s in f.steps)  # a finding's look also saw the groove
    assert all(imaging.box_iou(c["box"], b) < 0.3 for c in comps for bs in rejected.values() for b in bs)


def test_chaos_drills_recover(scene):
    # Real clock, real stop rule: the job may halt on the step right after the p95 spike; the governor's
    # model switch is then recorded on that halt record (`governor`), which the drill accepts.
    out = harness.run_chaos(scene, "planner_timeout")
    assert out["fired"] and out["ok"] and out["detail"]["reason"] == "timeout"
    spike = harness.run_chaos(scene, "p95_spike")
    assert spike["fired"] and spike["ok"]
    costs = harness.op_costs(harness.cost_rows("local-macos-x86", "classical-v1"), scene.shape[1], scene.shape[0])
    for kind in ("planner_timeout", "p95_spike"):  # on the fixed costs a drill takes the same path on any machine load
        a, b = (harness.run_chaos(scene, kind, costs) for _ in range(2))
        assert a == b and a["fired"] and a["ok"]


def test_governor_activity_counts_overrides_noted_on_halt_records():
    trace = [{"action": "gaze", "cause": "survey"}, {"action": "gaze", "cause": "p95>SLA"},
             {"action": "halt", "cause": "stop-rule", "governor": "warp-spike"}]
    res = [{"width": 100, "height": 100, "configs": {"drishyam@0.3": {"trace": trace, "pixels_frac": 0.2, "halt_cause": "stop-rule"}}}]
    gov = harness.agent_metrics(res, [], (0.3,), 1)["governor_activity_in_eval"]
    assert gov == {"p95>SLA": 1, "warp-spike": 1, "usd≥80%": 0}


def test_a_confounded_comparison_is_named():
    """A policy comparison is like for like only with the governor off, no model switch, and the accurate net throughout."""
    am = {"governor_activity_in_eval": {"p95>SLA": 0}, "model_usage": {"downscale": {"accurate": 3}, "drishyam@0.3": {"accurate": 9}}}
    ok = {"agent_config": {"governor": False}, "agent_metrics": am}
    assert harness.confounded(ok) is None
    assert "governor was on" in harness.confounded({**ok, "agent_config": {"governor": True}})
    assert "governor was on" in harness.confounded({"agent_metrics": am})  # summaries from before the field ran it on
    gov = {**ok, "agent_metrics": {**am, "governor_activity_in_eval": {"p95>SLA": 800}}}
    assert "800 times" in harness.confounded(gov)  # the trained-model run that compared a 256-px fp16 net with the 512 one
    fast = {**ok, "agent_metrics": {**am, "model_usage": {"drishyam@0.3": {"accurate": 2, "fast": 7}}}}
    assert "fast" in harness.confounded(fast)


def test_run_drishyam_records_the_switches_the_job_read(scene):
    out = harness.run_drishyam(scene, 0.3)
    assert out["switches"] == harness.switches("full") and out["switches"]["governor"] is False
    assert set(out["models"]) == {"accurate"} and sum(out["models"].values()) == sum(t["action"] != "halt" for t in out["trace"])
    assert harness.run_drishyam(scene, 0.3, governor=True)["switches"]["governor"] is True


def test_ablations_assert_each_variant_ran_its_switches(tmp_path):
    data = tmp_path / "abl"
    synth.build_dataset(data, n=1, seed=1, width=1024, height=768)  # 1024 px: room for looks at cap 1.0, too small for a survey
    entry = harness.run_ablations(str(data), n=1, out_dir=tmp_path / "pub", results_dir=tmp_path / "raw", workers=1,
                                  caps=(1.0,), seed=1, log=lambda *_: None)
    doc = json.loads((tmp_path / "pub" / "ablations.json").read_text())
    assert doc["datasets"] == {"abl": json.loads(json.dumps(entry))} and "NaN" not in json.dumps(doc)
    rows = entry["by_cap"]["1.0"]
    assert set(rows) == set(harness.ABLATIONS) == set(entry["variants"]) and len(harness.ABLATIONS) == 12  # full + 11
    assert rows["full"]["ratio_vs_full"] == 1.0 and rows["full"]["vs_full"] == "no measurable difference"
    # a score term moves the stop floor too: its floor-free twin is scored against no-floor, and every row says how it ended
    for v in ("no-seed+no-floor", "no-uncertainty+no-floor"):
        assert rows[v]["ref"] == "no-floor" and len(rows[v]["ratio_vs_ref_ci"]) == 2 and "vs_ref" in rows[v]
    assert all(sum(r["halt_causes"].values()) == 1 for r in rows.values()) and "no-floor" in entry["coupling"]
    assert rows["no-verify"]["verify_looks"] == rows["no-survey"]["survey_looks"] == 0
    assert set(rows["no-survey"]["model_usage"]) == {"accurate"} and len(rows["full"]["ratio_vs_full_ci"]) == 2
    assert entry["variants"]["random-order-s2"] == {**harness.switches("full"), "order": "random", "seed": 2}
    assert entry["random_order"][0].startswith("At a 100% pixel cap, random order found")
    assert "the seeds pooled per image" in entry["random_order"][0] and "multiple comparisons" in entry["random_order"][0]
    assert entry["provenance"]["dataset_gt_sha256"] == harness.dataset_sha256(harness.load_records(data, 1, seed=1)[0])["dataset_gt_sha256"]
    saved = json.loads((tmp_path / "raw" / "ablation_per_image.json").read_text())["results"]
    harness.check_variants(saved, (1.0,))
    cfg = saved[0]["configs"]["no-survey@1.0"]
    cfg["switches"]["survey"] = True  # labelled no-survey, ran the full agent
    with pytest.raises(SystemExit, match="not its label"):
        harness.check_variants(saved, (1.0,))
    cfg["switches"]["survey"] = False
    cfg["trace"].insert(0, {"action": "gaze", "cause": "survey"})  # right switches, but the trace has a survey look
    with pytest.raises(SystemExit, match="trace shows a look"):
        harness.check_variants(saved, (1.0,))
    cfg["trace"].pop(0)
    harness.check_variants(saved, (1.0,))
    saved[0]["configs"]["no-seed@1.0"]["trace"][0]["seeded"] = 2  # labelled no-seed, but a look seeded candidates
    with pytest.raises(SystemExit, match="trace shows a look"):
        harness.check_variants(saved, (1.0,))


def test_no_seed_and_no_floor_ablations_switch_one_term_off_and_change_the_run(scene):
    """'no-seed' (no look seeds an unvisited candidate) and 'no-floor' (v0 = 0: lambda is 0 until something is found)
    each flip one switch, and each changes what the agent does: without seeds the argmax order changes; without the
    floor a photo with nothing to find runs on to its cap or until every candidate is looked at."""
    from drishyam.agent import JobConfig

    full = harness.switches("full")
    assert harness.switches("no-seed") == {**full, "seed_from_looks": False} and full["seed_from_looks"] is True
    assert harness.switches("no-floor") == {**full, "v0": 0.0} and full["v0"] == JobConfig().v0 > 0
    looks = lambda out: [(t["action"], t["x"], t["y"], t["level"]) for t in out["trace"]]
    full_run, no_seed = (harness.run_drishyam(scene, 1.0, **harness.ABLATIONS[v]) for v in ("full", "no-seed"))
    assert no_seed["switches"] == harness.switches("no-seed") and looks(no_seed) != looks(full_run)
    clean = synth.render(2, 1600, 1200, {"distractors": ["water_stain"]})[0]  # nothing to find
    full_run, no_floor = (harness.run_drishyam(clean, 1.0, **harness.ABLATIONS[v]) for v in ("full", "no-floor"))
    assert no_floor["switches"] == harness.switches("no-floor") and full_run["halt_cause"] == "stop-rule"
    assert len(looks(no_floor)) > len(looks(full_run)) and no_floor["halt_cause"] != "stop-rule"


def test_random_order_matches_only_inside_the_margin():
    # a 6-image review run: every seed below the full agent, every CI reaching 1.0 only at its top
    rows = {"full": {"recall": 0.625}, **{f"random-order-s{s}": {"recall": r, "ratio_vs_full_ci": ci}
                                          for s, (r, ci) in enumerate([(0.583, [0.81, 1.0]), (0.542, [0.73, 1.0]), (0.542, [0.6, 1.0])])}}
    note = lambda ci: harness.random_order_note(rows, 1.0, ci, 6)
    assert "no measurable difference at n=6, but the CI reaches down to 0.75" in note([0.75, 1.0]) and "matches the" not in note([0.75, 1.0])
    assert "0.81-1.00, 0.73-1.00, 0.60-1.00" in note([0.75, 1.0])
    assert "so random order matches the full agent" in note([0.97, 1.03])
    assert "below the full agent" in note([0.6, 0.9]) and "undefined" in note([None, None])
    m = {f"random-order-s{s}@1.0": {"per_image": [[s, 0, 0, 0, 4], [2, 0, 0, 0, 4]]} for s in (0, 1, 2)}
    assert harness.pooled_seeds(m, 1.0)["per_image"].tolist() == [[1, 0, 0, 0, 4], [2, 0, 0, 0, 4]]


def test_a_clean_set_is_scored_as_false_alarms(tmp_path):
    data = tmp_path / "clean"
    synth.build_dataset(data, n=2, seed=11, width=640, height=480, clean=True)  # + bridge_08, last: n=2 leaves it out
    s = harness.run(str(data), n=2, out_dir=tmp_path / "pub", results_dir=tmp_path / "raw", workers=1, chaos=(0, 0, 0),
                    seed=11, log=lambda *_: None)
    assert "NaN" not in (tmp_path / "pub" / "summary.json").read_text() and s["headline_stats"] is None
    assert s["dataset"].startswith("synthetic clean set, 2 images")
    fa = s["false_alarms"]
    assert set(fa) == {"downscale", *(f"tiling@L{lv}" for lv in harness.TILE_LEVELS), *(f"drishyam@{c}" for c in harness.BUDGETS)}
    assert all(0 <= v["images_with_critical"] <= v["images_with_any"] <= 1 for v in fa.values())
    raw = json.loads((tmp_path / "raw" / "eval_per_image.json").read_text())["results"]
    for k, v in fa.items():  # every policy in both units: the headline never sets findings beside component boxes
        if k != "classes_reported":
            assert v["union_per_image"] == round(sum(len(r["configs"][k]["preds_union"]) for r in raw) / 2, 2)
            assert v["boxes_per_image"] == round(sum(len(r["configs"][k]["preds"]) for r in raw) / 2, 2)
    d = s["policies"]["drishyam"][3]
    assert d["recall"] is None and d["recall_ci"] == [None, None]
    d, ds = fa["drishyam@0.3"], fa["downscale"]
    assert f"per-component boxes per image, Drishyam at a 30% pixel cap reported {d['boxes_per_image']:.2f}" in s["headline"]
    assert f"(a Drishyam finding), {d['union_per_image']:.2f}, {ds['union_per_image']:.2f} and" in s["headline"]
    box = {"cls": "exposed_rebar", "box": [0, 0, 9, 9], "area": 81, "conf": 0.9}
    one = harness.false_alarms([{"id": "bridge_08"}], [{"configs": {"downscale": {"preds": [box], "preds_union": [box]}}}], ["downscale"], "components")
    assert one["downscale"]["critical_per_image"] == 1.0 and one["classes_reported"] == {"bridge_08": {"downscale": {"exposed_rebar": 1}}}


def test_thread_count_is_read_back_and_build_lines_quoted(monkeypatch):
    from bench import micro

    lines = micro.build_lines()
    assert "cv2_baseline" in lines and set(lines) <= set(micro.BUILD_LINES.values())
    assert all(v in cv2.getBuildInformation() for v in lines.values())  # verbatim
    assert lines.items() <= micro.provenance().items()
    before = cv2.getNumThreads()
    try:
        assert micro.set_threads(1) == 1 == cv2.getNumThreads()  # on macOS's GCD only setNumThreads(0) gives one
    finally:
        cv2.setNumThreads(before)
    monkeypatch.setattr(micro.cv2, "setNumThreads", lambda n: None)  # a backend that ignores the request
    monkeypatch.setattr(micro.cv2, "getNumThreads", lambda: 16)
    with pytest.raises(SystemExit, match=r"getNumThreads\(\) = 16"):
        micro.set_threads(1)
    with pytest.raises(SystemExit, match="max-load 0"):
        micro.check_load(0)
    assert len(micro.check_load()) == 3


# ---------------------------------------------------------------- end to end
def test_three_image_run_writes_valid_summary_and_gallery(tmp_path):
    data = tmp_path / "data"
    synth.build_dataset(data, n=3, seed=7, width=640, height=480)
    pub, raw = tmp_path / "pub", tmp_path / "raw"
    summary = harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), log=lambda *_: None)
    on_disk = json.loads((pub / "summary.json").read_text())
    assert on_disk == json.loads(json.dumps(summary))
    assert list(on_disk) == ["dataset", "n_images", "segmenter", "agent_config", "cost_model", "policies", "headline_stats",
                             "strata", "per_class", "agent_metrics", "micro", "provenance", "headline"]
    assert on_disk["n_images"] == 3 and on_disk["dataset"].startswith("synthetic-3 at 640x480")
    assert on_disk["agent_config"]["governor"] is False and harness.confounded(on_disk) is None
    assert on_disk["cost_model"]["kind"] == "fixed" and on_disk["cost_model"]["ms_per_call"]["segment:accurate"] > 0
    pol = on_disk["policies"]
    assert [len(pol[k]) for k in ("downscale", "tiling", "drishyam")] == [1, 3, 6]
    for points in pol.values():
        for p in points:
            assert set(p) == {"pixels_frac", "net_pixels_frac", "recall", "recall_ci", "recall_iou_only", "recall_iou_only_ci",
                              "chance_recall", "chance_recall_iou_only", "precision", "fp_per_image", "boxes_per_image",
                              "hits_per_matched_gt", "usd"}
            assert 0 <= p["recall"] <= 1 and p["pixels_frac"] > 0 and p["usd"] >= 0
            assert p["net_pixels_frac"] is None or p["net_pixels_frac"] >= 0  # None until the segmenter reports net_px
            assert p["recall_ci"][0] <= p["recall"] <= p["recall_ci"][1]
    hs = on_disk["headline_stats"]
    assert hs["budget"] == 0.3 and len(hs["recall_ratio_vs_tiling_ci"]) == 2 and isinstance(hs["meets_target"], bool)
    assert isinstance(hs["meets_target_ci"], bool) and len(hs["recall_ratio_iou_only_vs_tiling_ci"]) == 2
    assert set(on_disk["strata"]["recall"]["downscale"]) == {"hairline_crack", "small_box"}
    assert {"drishyam@0.3", "tiling@1.0", "downscale"} <= set(on_disk["per_class"])
    assert set(on_disk["per_class"]["downscale"]) == set(synth.CLASSES)
    prov = on_disk["provenance"]
    assert set(prov) - {"cv2_custom_hal", "cv2_parallel_framework", "cv2_dispatch", "cv2_baseline"} == {
        "cv2", "build_info_sha256", "machine", "cpu", "python", "timestamp", "build", "load_avg", "load_avg_start", "load_avg_end",
        "dataset_gt_sha256", "dataset_images_sha256", "dataset_generator_version"}
    assert prov["load_avg"] == prov["load_avg_start"] and len(prov["load_avg_end"]) == 3  # start and end, named as in bench/
    assert prov["dataset_gt_sha256"] == harness.dataset_sha256(harness.load_records(data, 3)[0])["dataset_gt_sha256"]
    am = on_disk["agent_metrics"]
    assert am["steps_with_cause_pct"] == 100.0 and set(am["box_convention"]["recall"]) == {"components", "union", "as_specified"}
    assert am["timing"]["cv2_threads_per_worker"] == harness.cv2.getNumThreads()  # inline: this process, read back
    assert set(am["model_usage"]["drishyam@1.0"]) == {"accurate"} and am["model_usage"]["tiling@L0"] == {"accurate": 3 * len(harness.slice_boxes(480, 640))}
    head = on_disk["headline"]
    assert isinstance(head, str) and head.count(". ") == 1  # two sentences
    for baseline in ("20%-overlap 512-px tiling", "1024-px downscale", "L1 tiling", "L2 tiling", "IoU >= 0.3"):
        assert baseline in head, baseline
    gallery = json.loads((pub / "failures.json").read_text())
    for g in gallery:
        assert set(g) == {"image", "id", "class", "category", "what_happened", "rule_that_would_catch_it"}
        assert (pub / g["image"]).exists()  # paths in failures.json are relative to public/results/ (the page prepends results/)
    assert (raw / "eval_raw.json").exists() and (raw / "eval_per_image.json").exists()
    cache = raw / "eval_per_image.json"
    assert gzip.decompress((pub / "per_image.json.gz").read_bytes()) == cache.read_bytes()
    again = harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), reuse=True, log=lambda *_: None)
    assert again["policies"]["drishyam"] == summary["policies"]["drishyam"]  # same predictions, re-scored
    fresh = harness.run(str(data), n=3, out_dir=tmp_path / "pub2", results_dir=tmp_path / "raw2", workers=1, chaos=(0, 0, 0),
                        log=lambda *_: None)
    assert fresh["policies"] == summary["policies"]  # fixed costs: a second full run decides and scores the same
    saved = json.loads(cache.read_text())
    assert saved["segmenter"] == summary["segmenter"] == "classical-v1" and saved["cost_model"] == "fixed"
    with pytest.raises(SystemExit, match="cost-model fixed, not live"):
        harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), reuse=True,
                    cost_model="live", log=lambda *_: None)
    with pytest.raises(SystemExit, match="seed 7, not --seed 8"):
        harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), seed=8, log=lambda *_: None)
    with pytest.raises(SystemExit, match="ran the governor off"):
        harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), reuse=True, governor=True,
                    log=lambda *_: None)
    cache.write_text(json.dumps({**saved, "dataset": {**saved["dataset"], "dataset_gt_sha256": "0" * 64}}))  # other GT
    with pytest.raises(SystemExit, match="other images or ground truth"):
        harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), reuse=True, log=lambda *_: None)
    cache.write_text(json.dumps({**saved, "segmenter": "onnx:m.onnx"}))  # an ONNX run, re-scored without DRISHYAM_ONNX
    with pytest.raises(SystemExit, match="onnx:m.onnx predictions"):
        harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), reuse=True, log=lambda *_: None)
    ann = json.loads((data / "annotations.json").read_text())
    (data / "annotations.json").write_text(json.dumps({k: v for k, v in ann.items() if k != "generator_version"}))
    with pytest.raises(SystemExit, match="generator version None"):  # a set built before the version (and the hairline flag)
        harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), log=lambda *_: None)


def test_gallery_picks_a_real_false_alarm(tmp_path):
    path = tmp_path / "s0001.jpg"
    cv2.imwrite(str(path), np.full((600, 800, 3), 150, np.uint8))
    rec = {"id": "s0001", "path": str(path), "boxes": [], "distractors": [{"kind": "graffiti", "box": [100, 100, 200, 200]}]}
    pred = {"cls": "corrosion", "box": [120, 120, 100, 100], "area": 5000, "conf": 0.8, "step": 1, "level": 1, "region": "r1"}
    quiet = {"preds": [], "trace": []}
    res = {"configs": {"drishyam@0.3": {"preds": [pred], "trace": []}, "drishyam@1.0": quiet, "tiling@L0": quiet}}
    (row,) = failures.build_gallery([rec], [res], tmp_path / "pub", log=lambda *_: None)
    assert row["category"] == "Graffiti read as damage" and "corrosion" in row["what_happened"]
    assert (tmp_path / "pub" / "failures" / "f01.jpg").exists() and row["image"] == "failures/f01.jpg"


def test_render_case_is_narrow_enough(tmp_path):
    img = np.full((1536, 2048, 3), 150, np.uint8)
    case = {"marks": [{"kind": "gt", "label": "GT crack", "box": [100, 100, 900, 700]},
                      {"kind": "pred", "label": "crack 0.9", "box": [150, 120, 200, 200]}]}
    trace = [{"action": "gaze", "x": 400, "y": 300, "level": 1, "window": [0, 0, 1024, 1024]},
             {"action": "halt", "x": None, "y": None, "level": None, "window": None}]
    out = tmp_path / "f01.jpg"
    failures.render_case(img, case, trace, "f01  Test  |  s0001  |  drishyam@0.3", out)
    shown = cv2.imread(str(out))
    assert shown is not None and shown.shape[1] <= 900


# ---------------------------------------------------------------- CODEBRIM
XML = """<annotation>
  <filename>image_0000001.jpg</filename>
  <size><width>6000</width><height>4000</height><depth>3</depth></size>
  <object><name>defect</name><bndbox><xmin>100</xmin><ymin>200</ymin><xmax>400</xmax><ymax>260</ymax></bndbox>
    <Defect><Background>0</Background><Crack>1</Crack><Spallation>0</Spallation><Efflorescence>0</Efflorescence>
    <ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain></Defect></object>
  <object><name>defect</name><bndbox><xmin>1000</xmin><ymin>1000</ymin><xmax>1500</xmax><ymax>1800</ymax></bndbox>
    <Defect><Background>0</Background><Crack>0</Crack><Spallation>1</Spallation><Efflorescence>0</Efflorescence>
    <ExposedBars>1</ExposedBars><CorrosionStain>1</CorrosionStain></Defect></object>
  <object><name>defect</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
    <Defect><Background>1</Background><Crack>0</Crack><Spallation>0</Spallation><Efflorescence>0</Efflorescence>
    <ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain></Defect></object>
</annotation>"""  # the real release's layout: the flags sit in <Defect>


def test_codebrim_xml_parsing_maps_flags_to_our_classes():
    a = codebrim.parse_xml(XML)
    assert (a["file"], a["width"], a["height"]) == ("image_0000001.jpg", 6000, 4000)
    assert a["boxes"] == [{"cls": "crack", "box": [100, 200, 300, 60]},
                          {"cls": "spalling", "box": [1000, 1000, 500, 800]},  # one box, three flags: three GT entries
                          {"cls": "exposed_rebar", "box": [1000, 1000, 500, 800]},
                          {"cls": "corrosion", "box": [1000, 1000, 500, 800]}]  # the background-only box is dropped


def test_codebrim_load_and_seeded_selection(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    rng = np.random.default_rng(0)
    for i in range(30):
        flags = {c: int(rng.random() < 0.3) for c in ("Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain")}
        body = "".join(f"<{k}>{v}</{k}>" for k, v in flags.items())
        (tmp_path / "images" / f"image_{i:07d}.jpg").write_bytes(b"x")
        (tmp_path / "annotations" / f"image_{i:07d}.xml").write_text(
            f"<annotation><filename>image_{i:07d}.jpg</filename><size><width>10</width><height>10</height></size>"
            f"<object><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>"
            f"<Defect>{body}</Defect></object></annotation>")
    fork = tmp_path / "__MACOSX" / "annotations"  # the zip ships binary macOS resource forks named like the XML
    fork.mkdir(parents=True)
    (fork / "._image_0000000.xml").write_bytes(b"\x00\x05\x16\x07\xff\xfe")
    records = codebrim.load(tmp_path)
    assert len(records) == 30 and records[0]["id"] == "image_0000000"
    subset, short = codebrim.select_eval(records, n=12, seed=7, min_per_class=3)
    assert len(subset) == 12 and not short
    for c in codebrim.CLASSES:
        assert sum(b["cls"] == c for r in subset for b in r["boxes"]) >= 3
    assert [r["id"] for r in subset] == [r["id"] for r in codebrim.select_eval(records, n=12, seed=7, min_per_class=3)[0]]
    out = tmp_path / "eval.txt"
    codebrim.write_list(subset, out)
    assert [r["id"] for r in codebrim.load_eval(tmp_path, out)] == [r["id"] for r in subset]



def test_codebrim_unpack_survives_wrapped_offsets(tmp_path):
    """The release zip's offsets wrap at 2**32; every entry must still come out intact, forks skipped, paths flattened."""
    zp = tmp_path / "release.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("original_dataset/images/image_0000001.jpg", b"\xff\xd8jpeg")
        z.writestr("original_dataset/annotations/image_0000001.xml", XML)
        z.writestr("__MACOSX/original_dataset/annotations/._image_0000001.xml", b"\x00\x05\x16\x07")
        z.writestr("../../escape.xml", "<annotation/>")
    with zipfile.ZipFile(zp) as z:
        for i, info in enumerate(z.infolist()):  # how zipfile sees the real file: some +2**32, some as stored
            info.header_offset += 2**32 if i % 2 == 0 else 0
        got = dict(codebrim._members(z, zp.stat().st_size))
    assert got == {"image_0000001.jpg": b"\xff\xd8jpeg", "image_0000001.xml": XML.encode(), "escape.xml": b"<annotation/>"}
    assert codebrim.unpack(zp, tmp_path / "out") == 3
    assert sorted(p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*.*")) == \
        ["annotations/escape.xml", "annotations/image_0000001.xml", "images/image_0000001.jpg"]

def test_harness_refuses_to_publish_codebrim_crops(tmp_path):
    """CODEBRIM's licence bars redistributing its images, so the gallery crops must never land under public/."""
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "image_0000001.jpg").write_bytes(b"x")
    (tmp_path / "image_0000001.xml").write_text(XML)
    (tmp_path / "eval200.txt").write_text("image_0000001.jpg\n")
    with pytest.raises(SystemExit, match="refusing"):
        harness.run(tmp_path, n=1, out_dir=tmp_path / "public" / "results", results_dir=tmp_path / "raw", workers=1)
    repo = Path(harness.__file__).resolve().parents[1]
    with pytest.raises(SystemExit, match="refusing"):  # anywhere in the repo that git would pick up
        harness.run(tmp_path, n=1, out_dir=repo / "docs" / "codebrim", results_dir=tmp_path / "raw", workers=1)
    assert not (repo / "docs" / "codebrim").exists()

def test_leakage_finds_a_resized_rotated_copy(tmp_path):
    from evals import leakage

    a, b = (synth.generate(s, 640, 480)[0] for s in (1, 2))
    (tmp_path / "train").mkdir()
    cv2.imwrite(str(tmp_path / "a.jpg"), a)
    cv2.imwrite(str(tmp_path / "b.jpg"), b)
    cv2.imwrite(str(tmp_path / "train" / "copy.jpg"), cv2.rotate(cv2.resize(a, (320, 240)), cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite(str(tmp_path / "train" / "other.jpg"), synth.generate(3, 640, 480)[0])
    rep = leakage.near_duplicates([tmp_path / "a.jpg", tmp_path / "b.jpg"], sorted((tmp_path / "train").iterdir()))
    assert rep["eval_images"] == 2 and rep["train_images"] == 2 and rep["near_duplicates"] == rep["candidates"] == 1
    assert rep["pairs"][0][:2] == ["a.jpg", "copy.jpg"] and rep["pairs"][0][2] <= 8 and rep["pairs"][0][3] > 0.9
    cal = leakage.calibrate([tmp_path / "a.jpg", tmp_path / "b.jpg"], [tmp_path / "train" / "other.jpg"], n_copies=2)
    assert cal["copies"]["n"] == 2 and cal["copies"]["jpeg50"]["found"] == 2 and cal["other_photos"]["n"] == 2
    assert cal["other_photos"]["confirmed"] == 0
    out = tmp_path / "leak.json"
    assert leakage.main(["--dacl10k", str(tmp_path / "absent"), "--out", str(out)]) == {"skipped": f"no dacl10k images under {tmp_path / 'absent'}"}
    assert json.loads(out.read_text())["skipped"]


def test_leakage_keeps_unrelated_low_texture_photos_apart():
    """Two different plain walls under the same lighting gradient: a 9x8 dHash sees only the gradient (distance near 0),
    the lighting-free pHash and the NCC check must not call them copies."""
    from evals import leakage

    ramp = np.linspace(60, 200, 800, dtype=np.float32)[None, :].repeat(600, 0)
    wall = lambda seed: np.clip(ramp + cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 12, (600, 800)).astype(np.float32),
                                                         (0, 0), 3), 0, 255).astype(np.uint8)
    a, b = wall(1), wall(2)
    def old(g):  # the 9x8 dHash this check replaced
        s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
        return np.packbits(s[:, :-1] > s[:, 1:])
    assert np.unpackbits(old(a) ^ old(b)).sum() <= 4  # what the old check would have flagged
    assert leakage.nearest(a, leakage.phash(b)[None])[2] > leakage.MAX_DIST and leakage.ncc(a, b) < leakage.MIN_NCC
    assert leakage.nearest(a, leakage.phash(cv2.resize(a, (400, 300)))[None])[2] <= leakage.MAX_DIST  # a copy still is one


def test_model_eval_refuses_a_mislabelled_confounded_or_other_data_arm(tmp_path):
    from evals import model_eval

    data = {"dataset_gt_sha256": "a" * 64, "dataset_images_sha256": "b" * 64}
    ok = {"segmenter": "onnx:m.onnx", "agent_config": {"governor": False}, "provenance": data,
          "agent_metrics": {"governor_activity_in_eval": {"p95>SLA": 0}, "model_usage": {"drishyam@0.3": {"accurate": 5}}}}
    assert model_eval.check(ok, "onnx:m.onnx", "x") == ok
    with pytest.raises(SystemExit, match="scored with classical-v1, not onnx:m.onnx"):
        model_eval.check({**ok, "segmenter": "classical-v1"}, "onnx:m.onnx", "runs/model-eval/onnx-synth")
    slow = {**ok, "agent_metrics": {**ok["agent_metrics"], "governor_activity_in_eval": {"p95>SLA": 800}}}
    with pytest.raises(SystemExit, match="not a like-for-like.*800 times"):  # the governor swapped in the fp16 net
        model_eval.check(slow, "onnx:m.onnx", "runs/model-eval/onnx-synth")
    assert model_eval.same_data(ok, {**ok}) and not model_eval.same_data(ok, {**ok, "provenance": {**data, "dataset_gt_sha256": "c"}})
    assert not model_eval.same_data({"provenance": {}}, {"provenance": {}})  # unhashed summaries never match
    arm64 = {"cost_model": {"kind": "fixed", "source": "results/linux-aarch64/micro.json job_ops (p50 ms per call)"}}
    assert model_eval.same_costs(arm64, "linux-aarch64") and not model_eval.same_costs(arm64, "local-macos-x86")
    leak = tmp_path / "leakage.json"
    leak.write_text(json.dumps({"skipped": "no dacl10k images under x"}))
    with pytest.raises(SystemExit, match="refusing to publish a CODEBRIM number"):
        model_eval.leakage_block(leak)
    rep = {"eval_images": 200, "train_images": 9000, "max_dist": 32, "min_ncc": 0.8, "candidates": 3, "near_duplicates": 1,
           "hash_at_or_below": {"8": 0}, "pairs": [["a.jpg", "b.jpg", 4, 0.95]], "calibration": {"separated": True}}
    leak.write_text(json.dumps(rep))
    assert model_eval.leakage_block(leak) == {k: v for k, v in rep.items() if k != "pairs"}  # counts and calibration, no file names
    onnx, cb = tmp_path / "m.onnx", tmp_path / "codebrim"
    onnx.write_bytes(b"")
    cb.mkdir()
    (cb / "eval200.txt").write_text("image_0000001.jpg\n")
    with pytest.raises(SystemExit, match="needs the dacl10k leakage check"):  # refused before anything runs
        model_eval.main(["--onnx", str(onnx), "--codebrim", str(cb), "--publish"])


# ---------------------------------------------------------------- parity
def test_parity_compare_saved_files(tmp_path):
    a = np.zeros((2, 8, 8), np.uint8)
    a[:, :4] = 1  # half crack
    b = a.copy()
    b[1, :2] = 0  # 16 of the 128 crack pixels become background in the second crop
    crops = np.zeros((2, 8, 8, 3), np.uint8)
    for name, labels in (("a.npz", a), ("b.npz", b)):
        np.savez(tmp_path / name, crops=crops, labels=labels, classes=np.array(CLASSES), meta=json.dumps(
            {"cv2": "x", "build": "y", "cpu": "z"}))
    res = parity.compare_files(str(tmp_path / "a.npz"), str(tmp_path / "b.npz"))
    assert res["crops"] == 2 and res["inputs_identical"] and res["identical_crops"] == 1
    assert res["pixel_agreement"] == pytest.approx(1 - 16 / 128)
    assert res["worst_crop_agreement"] == pytest.approx(1 - 16 / 64)
    assert res["class_iou"]["crack"] == pytest.approx(48 / 64) and res["class_iou"]["spalling"] is None
    with pytest.raises(ValueError):
        parity.compare(a, b[:1])


def test_codebrim_dev_split_is_disjoint_from_the_held_out_list():
    """Tuning reads evals/codebrim_dev100.txt only: 100 images, none of them in the held-out eval200 list."""
    evals_dir = Path(codebrim.__file__).parent
    dev = (evals_dir / "codebrim_dev100.txt").read_text().split()
    held = set((evals_dir / "codebrim_eval200.txt").read_text().split())
    assert len(dev) == len(set(dev)) == 100 and not held & set(dev)
    records = [{"id": f"image_{i}", "path": f"/x/image_{i}.jpg", "boxes": []} for i in range(10)]
    picked = codebrim.select_dev(records, ["image_1.jpg", "image_2.jpg"], n=5, seed=3)
    assert len(picked) == 5 and not {"image_1", "image_2"} & {r["id"] for r in picked}
    assert picked == codebrim.select_dev(records, ["image_1.jpg", "image_2.jpg"], n=5, seed=3)


def test_dev_sweep_reads_variants_and_only_dev_splits():
    from evals import dev_sweep

    assert dev_sweep.parse("a=v0:2,seed_from_looks:False;b=") == {"a": {"v0": 2.0, "seed_from_looks": False}, "b": {}}
    assert {d for d, _, _ in dev_sweep.SPLITS.values()} == {"data/synth_dev", "data/synth_clean_dev"}  # never held out
