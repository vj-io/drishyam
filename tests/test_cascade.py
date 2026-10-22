"""The optional two-model cascade (drishyam.cascade): when the confirm net looks again, what it is charged, and that a
job without a confirm net is the agent as it was before the cascade, event for event."""

import json
import re
import sys
import time
import types
from dataclasses import replace
from inspect import signature
from pathlib import Path

import cv2
import numpy as np
import pytest

from drishyam import agent, cascade
from drishyam.agent import Job, JobConfig
from drishyam.governor import Budget
from drishyam.prefetch import Prefetcher
from drishyam.segment import CLASSES, ClassicalSegmenter, OnnxSegmenter, Segmentation, finish
from evals import dev_sweep, harness, synth

from .conftest import FakeClock

COSTS = {"pyrDown": 2.0, "saliency": 2.0, "warp": 1.0, "quality": 1.0, "segment:accurate": 10.0, "segment:fast": 5.0,
         "segment:confirm": 700.0}
WALL_CLOCK = {"t", "cpu_pct", "created"}  # ISO timestamps and process CPU share: the only fields not on the job's clock
SAMPLES = Path(__file__).resolve().parent.parent / "public" / "samples"


class Painted:
    """A stand-in confirm net: background everywhere but, with `cls`, one 40 x 40 block of that class at the crop's
    centre. It records the model each call asked for."""

    def __init__(self, cls=None, name=None, sha256="c" * 64):
        self.cls, self.name, self.sha256, self.models = cls, name or f"stub:{cls or 'blank'}", sha256, []

    def predict(self, crop, model="accurate"):
        self.models.append(model)
        h, w = crop.shape[:2]
        probs = np.zeros((len(CLASSES), h, w), np.float32)
        probs[0] = 1
        if self.cls:
            rows, cols = slice(h // 2 - 20, h // 2 + 20), slice(w // 2 - 20, w // 2 + 20)
            probs[:, rows, cols] = 0
            probs[CLASSES.index(self.cls), rows, cols] = 1
        return finish(probs, time.perf_counter(), model)


def seg_of(entropy, **blocks):
    """A 512-px look's segmentation with this `entropy` and, per class, one h x w block (its only component)."""
    labels, y = np.zeros((512, 512), np.uint8), 0
    for cls, (h, w) in blocks.items():
        labels[y : y + h, :w] = CLASSES.index(cls)
        y += h + 10
    areas = {c: int(np.count_nonzero(labels == i)) for i, c in enumerate(CLASSES) if i}
    return Segmentation(np.zeros((len(CLASSES), 1, 1), np.float32), labels, entropy, areas, dict.fromkeys(areas, 0.9),
                        0.0, "accurate")


def looks(events):
    return [e for e in events if e["type"] == "step" and e["decision"]["action"] != "halt"]


def ran(step) -> bool:
    return "latency_ms" in step.get("confirm", {})


def skipped(steps, cap: str) -> bool:
    return any(s.get("confirm", {}).get("skipped", "").startswith(cap) for s in steps)


def fixed(img, costs=COSTS, cfg=None, **kw):
    """A cascade job (classical scout) on the fixed cost table `costs`, a virtual clock, governor off as in every eval."""
    job = harness.FixedCostJob(img, cfg or JobConfig(pixel_cap=1.0, governor=False, thumbs=False), dict(costs),
                               job_id="j_test", segmenter=ClassicalSegmenter(), **kw)
    return job, list(job.run())


@pytest.fixture
def no_cached_confirm():
    cascade.default_confirm.cache_clear()
    yield
    cascade.default_confirm.cache_clear()


# ---------------------------------------------------------------- when the confirm net looks again
def test_the_confirm_net_looks_again_when_the_scout_is_unsure_or_drafted_a_critical_class():
    cfg = JobConfig()
    assert (cfg.tau_confirm_lo, cfg.tau_confirm_hi) == (0.35, 0.65)  # both ends in the band
    for e, want in ((0.0, None), (0.34, None), (0.35, "ambiguous"), (0.5, "ambiguous"), (0.65, "ambiguous"),
                    (0.66, None), (0.95, None)):
        assert cascade.reason(seg_of(e, crack=(4, 100)), 0, cfg) == want, e
    # a critical class the scout would draft (a mask of at least A_min) goes to the confirm net whatever the entropy
    assert cascade.reason(seg_of(0.95, exposed_rebar=(8, 8)), 0, cfg) == "critical"
    assert cascade.reason(seg_of(0.5, exposed_rebar=(8, 8)), 0, cfg) == "critical"  # named first when both hold
    assert cascade.reason(seg_of(0.95, exposed_rebar=(7, 9)), 0, cfg) is None  # 63 px < A_min 64: not drafted
    # spalling is critical from 40,000 L0 px² of its largest component: 50 x 50 crop px at L2 (x16), not at L1 (x4)
    assert cascade.reason(seg_of(0.1, spalling=(50, 50)), 2, cfg) == "critical"
    assert cascade.reason(seg_of(0.1, spalling=(50, 50)), 1, cfg) is None
    assert cascade.reason(seg_of(0.1, spalling=(50, 50)), 1, replace(cfg, tau_confirm_lo=0.1)) == "ambiguous"


# ---------------------------------------------------------------- the budget gate
def test_the_budget_gate_admits_a_confirm_only_inside_every_cap():
    look = 512 * 512
    b = Budget(10 * look, 1.0, 20.0)
    b.pixels_used = 8 * look  # this look takes the ninth crop's worth of the cap, the confirm the tenth: exactly full
    assert cascade.fits(b, 512, 19.0, 1000.0) is None  # and 20 s on the clock: at the seconds cap, not past it
    b.pixels_used += 1
    assert cascade.fits(b, 512, 0.0, 1.0).startswith("pixel cap")
    b.pixels_used -= 1
    assert cascade.fits(b, 512, 19.0, 1000.1).startswith("seconds cap")
    b.usd_cap = 19.5 * b.usd_per_hour / 3600  # $ is seconds x the hourly rate: a cap below 20 s of it binds first
    assert cascade.fits(b, 512, 19.0, 600.0).startswith("$ cap")


def test_a_confirm_that_would_break_a_cap_is_skipped_and_the_step_says_which(scene):
    job, ev = fixed(scene, cfg=JobConfig(pixel_cap=0.3, governor=False, thumbs=False), confirm=Painted())
    steps = looks(ev)
    assert any(map(ran, steps)) and skipped(steps, "pixel cap")
    assert max(s["pixels_used"] for s in steps) <= job.budget.pixels_cap
    # The first confirm is predicted at the scout's cost (no confirm timed yet) and takes 15 s; any later one would end
    # past the 20 s cap, so it is skipped, and the job's clock never passes the cap.
    job, ev = fixed(scene, {**COSTS, "segment:confirm": 15000.0}, confirm=Painted())
    steps = looks(ev)
    assert sum(map(ran, steps)) == 1 and skipped(steps, "seconds cap")
    assert max(s["seconds_used"] for s in steps) <= job.cfg.seconds_cap
    assert all(s["latency_ms"] == 1.0 + 10.0 for s in steps if not ran(s))  # a skipped confirm costs nothing


# ---------------------------------------------------------------- what a confirm is charged
def test_a_confirm_is_charged_as_its_own_op_and_spends_the_jobs_own_budget(scene):
    job, ev = fixed(scene, confirm=Painted())
    steps = looks(ev)
    done = [s for s in steps if ran(s)]
    assert done and len(done) < len(steps)  # the confirm net looked again at some looks, not all
    before = next(e for e in ev if e["type"] == "overview")["pixels_used"]
    for s in steps:  # a confirmed look reads its crop twice, against the one pixel cap
        assert s["pixels_used"] - before == s["decision"]["args"]["size"] ** 2 * (1 + ran(s))
        assert s["latency_ms"] == 1.0 + 10.0 + 700.0 * ran(s)  # the warp, the scout, and the confirm at its own p50
        before = s["pixels_used"]
    assert all(s["confirm"]["latency_ms"] == 700.0 for s in done)
    assert job.now == pytest.approx((2 + 2 + 12 * len(steps) + 700 * len(done)) / 1000)  # each op once, on one clock
    p50 = ev[-2]["runtime"]["p50_ms"]  # the halt record
    assert (p50["segment:accurate"], p50["segment:confirm"]) == (10.0, 700.0)


def test_on_a_fixed_predicted_cost_table_the_stop_rules_spend_counts_each_confirm_at_its_own_cost(scene):
    """With JobConfig.t_hat_ms the stop rule's spend is t_hat per look (35 ms here) plus segment:confirm's cost per
    confirm, so its lambda (eps x value found / spend; the floor is off) falls as the confirm net costs more."""
    def run(ms):
        cfg = JobConfig(pixel_cap=1.0, thumbs=False, v0=1e-12, t_hat_ms={"warp": 5.0, "segment": 30.0, "segment:confirm": ms})
        steps = looks(list(Job(scene, cfg, clock=FakeClock(), job_id="j_test", confirm=Painted()).run()))
        i, lam = next((i, float(m[1])) for i, s in enumerate(steps) if (m := re.search(r"lambda ([-+.e\d]+)", s["rationale"]))
                      and float(m[1]) > 0)
        return [s["decision"] for s in steps], lam, 35.0 * i + ms * sum(map(ran, steps[:i]))
    (cheap, lam1, spend1), (dear, lam700, spend700) = run(1.0), run(700.0)
    assert cheap == dear and spend700 > spend1  # the same looks, and some confirmed before lambda was read
    assert lam700 / lam1 == pytest.approx(spend1 / spend700, rel=0.01)


def test_a_confirmed_look_keeps_the_confirm_nets_segmentation_and_the_ladder_moves_the_scout_only(scene):
    net = Painted("corrosion")
    job = Job(scene, JobConfig(pixel_cap=1.0, thumbs=False, chaos=("p95_spike",)), clock=FakeClock(), job_id="j_test",
              confirm=net)
    steps = looks(list(job.run()))
    done = [s for s in steps if ran(s)]
    assert done and all(s["observation"]["classes"] == {"corrosion": 1600} for s in done)  # the confirm's, not the scout's
    assert all(s["confirm"]["scout"]["classes"] and s["confirm"]["net"] == "stub:corrosion" for s in done)
    assert {s["model"] for s in steps} == {"accurate", "fast"}  # the governor moved the scout down its ladder (p95>SLA)
    assert set(net.models) == {"accurate"}  # never the confirm net
    assert job.work_order["confirm_sha256"] == "c" * 64  # the work order names both nets' weights


def test_a_prefetched_look_still_gets_the_confirm_nets_own_segmentation(scene):
    """The prefetcher serves a look's scout segmentation once; the confirm on the same crop is computed, not served."""
    job = Job(scene, JobConfig(pixel_cap=1.0, thumbs=False), clock=FakeClock(), job_id="j_test", confirm=Painted())
    job.prefetch = Prefetcher()
    try:
        steps = looks(list(job.run()))
    finally:
        job.prefetch.close()
    assert job.prefetch.counts["seg_hit"] and any(map(ran, steps))
    assert all(s["observation"]["entropy"] == 0 and not s["observation"]["classes"] for s in steps if ran(s))


# ---------------------------------------------------------------- off by default
def without_cascade(monkeypatch):
    """drishyam.agent without its lines marked `# cascade`, as a sibling module in the drishyam package: the agent
    before the cascade, provided every line the cascade added carries the mark."""
    src = Path(agent.__file__).read_text()
    kept = "".join(ln for ln in src.splitlines(keepends=True) if not re.search(r"#\s*cascade\b", ln))
    mod = types.ModuleType("drishyam._agent_without_cascade")
    mod.__package__ = "drishyam"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(kept, "drishyam/agent.py without the cascade", "exec"), mod.__dict__)  # noqa: S102 -- our own source
    assert "confirm" in signature(agent.Job).parameters and "confirm" not in signature(mod.Job).parameters
    return mod


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in WALL_CLOCK}
    return [_strip(v) for v in x] if isinstance(x, list) else x


IMAGES = {**{p.stem: (lambda p=p: cv2.imread(str(p))) for p in sorted(SAMPLES.glob("bridge_*.jpg"))},
          "synth_9101": lambda: synth.generate(9101, 2048, 1536)[0],
          "synth_9102_wide": lambda: synth.generate(9102, 3000, 1000)[0]}  # looks past the image's edge
ONNX = "onnx:drishyam_dacl10k.onnx"
CASES = [*((name, "classical", None) for name in IMAGES),
         *((name, "onnx", p) for name in ("bridge_03", "bridge_06", "synth_9102_wide") for p in (None, ONNX))]


@pytest.mark.parametrize(("name", "engine", "profile"), CASES)
def test_without_a_confirm_net_every_event_is_the_agents_before_the_cascade(name, engine, profile, tiny_onnx,
                                                                              monkeypatch, no_cached_confirm):
    """The bundled samples and two synthetic photos, on the classical engine and on an ONNX net under its own
    (classical-v1) profile and the dacl10k one: the whole event stream (thumbnails, timings on the job's clock) of a
    job given no confirm net equals the agent's with the cascade's lines removed. DRISHYAM_ONNX_CONFIRM is set
    meanwhile: a Job never reads it (`cascade.default_confirm` is for bench.micro and the dev evals)."""
    monkeypatch.setenv("DRISHYAM_ONNX_CONFIRM", tiny_onnx)
    off, img = without_cascade(monkeypatch), IMAGES[name]()
    seg = OnnxSegmenter(tiny_onnx) if engine == "onnx" else ClassicalSegmenter()
    streams = []
    for mod in (agent, off):
        job = mod.Job(img, mod.JobConfig(profile=profile), clock=FakeClock(), job_id="j_test", segmenter=seg)
        streams.append(json.dumps(_strip(list(job.run())), sort_keys=True))
    assert streams[0] == streams[1] and not re.search(r'"confirm(_sha256)?":', streams[0])  # no cascade record


def test_the_confirm_net_is_named_by_its_own_variable_only(tiny_onnx, monkeypatch, no_cached_confirm):
    monkeypatch.delenv("DRISHYAM_ONNX_CONFIRM", raising=False)
    monkeypatch.setenv("DRISHYAM_ONNX", tiny_onnx)  # the scout's variable never names a confirm net
    assert cascade.default_confirm() is None
    monkeypatch.setenv("DRISHYAM_ONNX_CONFIRM", tiny_onnx)
    cascade.default_confirm.cache_clear()
    net = cascade.default_confirm()
    assert isinstance(net, OnnxSegmenter) and net.files == (tiny_onnx, None)  # its accurate net only


# ---------------------------------------------------------------- the eval and the cost table
def test_each_dev_sweep_arm_runs_the_nets_it_is_named_for(tmp_path, monkeypatch):
    img = synth.generate(9101, 1024, 768)[0]
    cv2.imwrite(str(tmp_path / "a.png"), img)
    rows = [{"op": op, "size": "1024x768", "p50_ms": ms} for op, ms in COSTS.items()]
    monkeypatch.setattr(dev_sweep, "default_segmenter", ClassicalSegmenter)
    net = Painted("corrosion", ONNX, agent.WEIGHTS[ONNX])  # a confirm net whose own profile is not the scout's
    assert agent.profile_for(net) == ONNX != agent.profile_for(ClassicalSegmenter())
    monkeypatch.setattr(cascade, "default_confirm", lambda: net)
    variants = dev_sweep.parse("scout=;confirm=net:confirm;cascade=net:cascade")
    res = dev_sweep.task({"id": "a", "path": str(tmp_path / "a.png")}, variants, rows, [1.0])["configs"]
    scout, alone, both = res["scout@1.0"], res["confirm@1.0"], res["cascade@1.0"]
    assert (scout["segmenter"], scout["confirm"]) == ("classical-v1", None) and not any("confirm" in t for t in scout["trace"])
    # the confirm net alone: every look charged its p50, under the scout's profile so only the net differs
    assert (alone["segmenter"], alone["confirm"], alone["profile"]) == (ONNX, None, "classical-v1")
    assert alone["wall_s"] == pytest.approx((2 + 2 + 702 * len(looks_of(alone))) / 1000)
    assert (both["segmenter"], both["confirm"], both["profile"]) == ("classical-v1", ONNX, "classical-v1")
    confirmed = {t["step"] for t in both["trace"] if "latency_ms" in t.get("confirm", {})}
    assert confirmed and both["wall_s"] == pytest.approx((4 + 12 * len(looks_of(both)) + 700 * len(confirmed)) / 1000)
    # Findings from a confirmed look are rebuilt from the confirm's segmentation (its corrosion block), not the scout's
    comps = [p for p in both["preds"] if p["step"] in confirmed]
    assert comps and all(p["cls"] == "corrosion" and p["box"][2:] == [40 * 2 ** p["level"]] * 2 for p in comps)
    monkeypatch.setattr(cascade, "default_confirm", lambda: None)
    with pytest.raises(SystemExit, match="needs a confirm net"):
        dev_sweep.task({"id": "a", "path": str(tmp_path / "a.png")}, variants, rows, [1.0])


def test_a_dev_sweep_row_reports_modelled_ms_and_recall_per_modelled_second_paired_with_scout_only():
    gt, miss = {"cls": "crack", "box": [100, 100, 50, 50]}, {"cls": "crack", "box": [900, 700, 50, 50], "conf": 0.5, "area": 9}
    hit = {**gt, "conf": 0.9, "area": 9}
    records = [{"id": str(i), "boxes": [gt]} for i in range(4)]
    cfg = lambda preds, s: {"preds": preds, "preds_union": preds, "wall_s": s, "pixels_frac": 0.5, "net_pixels_frac": 0.5}
    results = [{"id": str(i), "width": 1024, "height": 768,
                "configs": {"scout@1.0": cfg([hit] if i < 2 else [], 1.0), "cascade@1.0": cfg([hit] if i < 3 else [miss], 2.0 + i)}}
               for i in range(4)]
    scored = {k: harness.score_config(k, records, results) for k in ("scout@1.0", "cascade@1.0")}
    r = dev_sweep.row("cascade@1.0", records, results, scored, "scout@1.0")
    assert (r["recall"], r["ms_per_job"], r["recall_per_s"]) == (0.75, 3500.0, round(0.75 / 3.5, 4))
    assert (r["ms_vs_ref"], r["recall_per_s_vs_ref"]) == (3.5, round(0.75 * 1.0 / (0.5 * 3.5), 4))
    assert r["ms_vs_ref_ci"][0] >= 2.0 and r["ms_vs_ref_ci"][1] <= 5.0  # every resample costs 2-5 x scout-only's
    assert r["boxes_diff_vs_ref_ci"][0] >= 0 and r["boxes_diff_vs_ref_ci"][1] <= 1  # per photo: 0 or 1 more box
    assert dev_sweep.row("scout@1.0", records, results, scored, "scout@1.0").get("ref") is None  # the reference itself


def looks_of(config: dict) -> list:
    return [t for t in config["trace"] if t["action"] != "halt"]


def test_a_cost_table_must_have_timed_the_same_confirm_net(tmp_path, monkeypatch):
    from bench import micro

    monkeypatch.setattr(harness, "REPO", tmp_path)
    rows = [{"op": "segment:accurate", "size": "1024x768", "p50_ms": 10.0}]
    p = tmp_path / "results" / "a" / "micro.json"
    p.parent.mkdir(parents=True)
    for jobs, ok in (({"segmenter": "classical-v1"}, False), ({"segmenter": "classical-v1", "confirm": "onnx:x.onnx"}, False),
                     ({"segmenter": "classical-v1", "confirm": "onnx:c.onnx"}, True)):
        p.write_text(json.dumps({"job_ops": {**jobs, "results": rows}}))
        assert harness.cost_rows("a", "classical-v1") == rows  # the scout's ops are there either way
        if ok:
            assert harness.cost_rows("a", "classical-v1", "onnx:c.onnx") == rows
        else:
            with pytest.raises(SystemExit, match="onnx:c.onnx"):
                harness.cost_rows("a", "classical-v1", "onnx:c.onnx")
    # bench.micro times the confirm under the trace's op name, and only when there is one
    assert "segment:confirm" not in micro.job_ops(1024, 768, ClassicalSegmenter())
    net = Painted("corrosion")
    out = micro.job_ops(1024, 768, ClassicalSegmenter(), net)["segment:confirm"]()
    assert out.areas["corrosion"] == 1600 and net.models == ["accurate"]


def test_a_gain_and_a_harm_against_scout_only_are_read_by_the_same_ci_standard():
    from evals.model_eval import readings

    read = lambda split, **cis: readings(split, cis)
    # a ratio's CI must exclude 1 on either side, a difference's 0: touching it is no change, gain or harm alike
    assert read("synth_dev", vs_ref_ci=[1.0, 1.2], recall_per_s_vs_ref_ci=[0.8, 1.0], ms_vs_ref_ci=[1.0, 1.0]) == {
        "recall": "no clear change", "recall_per_s": "no clear change", "ms_per_job": "no clear change"}
    assert read("synth_dev", vs_ref_ci=[1.01, 1.2], vs_ref_iou_only_ci=[0.8, 0.99], ms_vs_ref_ci=[0.9, 0.95],
                recall_per_s_vs_ref_ci=[0.7, 0.8]) == {"recall": "gain", "recall_iou_only": "harm", "ms_per_job": "gain",
                                                       "recall_per_s": "harm"}
    assert read("real", ms_vs_ref_ci=[1.2, 1.3], chance_vs_ref_ci=[1.1, 2.0], boxes_diff_vs_ref_ci=[0.1, 0.5]) == {
        "ms_per_job": "harm", "chance_recall": "higher", "boxes_per_image": "higher"}
    # on the clean split every box is a false alarm: more is a harm, fewer a gain
    assert read("clean", boxes_diff_vs_ref_ci=[0.1, 0.5], images_with_any_vs_ref_ci=[-0.2, -0.1]) == {
        "boxes_per_image": "harm", "images_with_any": "gain"}
    assert read("clean", images_with_any_vs_ref_ci=[-0.1, 0.0], vs_ref_ci=[None, None]) == {"images_with_any": "no clear change"}


def test_the_cascade_eval_times_the_pair_and_sweeps_the_dev_splits_only(tmp_path, monkeypatch):
    from evals import model_eval

    assert model_eval.pair_tag(Path("drishyam_dacl10k_v2a.onnx"), Path("drishyam_dacl10k_v2b.onnx")) == "dacl10k_v2a+v2b"
    assert model_eval.pair_tag(Path("drishyam_dacl10k.fp16.onnx"), Path("drishyam_dacl10k.onnx")) == "dacl10k.fp16+dacl10k"
    scout, confirm = tmp_path / "drishyam_dacl10k_v2a.onnx", tmp_path / "drishyam_dacl10k_v2b.onnx"
    scout.write_bytes(b"")
    confirm.write_bytes(b"")
    (tmp_path / "results" / "dev").mkdir(parents=True)
    nets = {"scout": ("onnx:drishyam_dacl10k_v2a.onnx", None), "confirm": ("onnx:drishyam_dacl10k_v2b.onnx", None),
            "cascade": ("onnx:drishyam_dacl10k_v2a.onnx", "onnx:drishyam_dacl10k_v2b.onnx")}
    calls, wrong = [], set()

    def run(cmd, env, cwd, check):
        calls.append((cmd, env))
        if "evals.dev_sweep" in cmd:
            row = {"recall": 0.5, "recall_iou_only": 0.4, "chance_recall": 0.1, "ms_per_job": 900.0, "recall_per_s": 0.6,
                   "boxes_per_image": 3.0, "images_with_any": 1.0, "profile": "classical-v1"}
            rows = {f"{k}@{cap}": {**row, "segmenter": s, "confirm": None if "net" in wrong and k == "cascade" else c,
                                   **({"profile": ONNX} if "profile" in wrong and k == "confirm" else {}),
                                   **({"ref": f"scout@{cap}", "vs_ref_ci": [0.9, 0.99]} if k != "scout" else {})}
                    for k, (s, c) in nets.items() for cap in (0.3, 1.0)}
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"n": 1, "rows": rows}))

    monkeypatch.setattr(model_eval, "REPO", tmp_path)
    monkeypatch.setattr(model_eval.subprocess, "run", run)
    argv = ["--onnx", str(scout), "--confirm", str(confirm), "--codebrim", str(tmp_path / "no-codebrim"), "--label", "x"]
    model_eval.main(argv)
    (micro, env), *sweeps = calls
    assert micro[1:5] == ["-m", "bench.micro", "--arm", "local-macos-x86-dacl10k_v2a+v2b"]
    assert (env["DRISHYAM_ONNX"], env["DRISHYAM_ONNX_CONFIRM"]) == (str(scout), str(confirm))
    assert [c[3] for c, _ in sweeps] == ["synth_dev", "clean"]  # no CODEBRIM here, and never a held-out set
    assert all(c[4:6] == ["0.3,1.0", model_eval.CASCADE_ARMS] and c[c.index("--workers") + 1] == "4" for c, _ in sweeps)
    assert all(e["DRISHYAM_ONNX_CONFIRM"] == str(confirm) for _, e in sweeps)
    out = json.loads((tmp_path / "results" / "dev" / "cascade_dacl10k_v2a+v2b.json").read_text())
    assert out["label"] == "x" and set(out["splits"]) == {"synth_dev", "clean"}
    assert out["readings"]["clean"] == {k: {"recall": "harm"} for k in ("confirm@0.3", "confirm@1.0", "cascade@0.3", "cascade@1.0")}
    wrong.add("profile")  # an arm under another profile is refused: the nets would not be all that differs
    with pytest.raises(SystemExit, match="different profiles"):
        model_eval.main(argv)
    wrong.add("net")  # a cascade arm that ran no confirm net is refused, not written up
    with pytest.raises(SystemExit, match="cascade@0.3 ran"):
        model_eval.main(argv)
    with pytest.raises(SystemExit, match="nothing is published"):
        model_eval.main([*argv, "--publish"])
