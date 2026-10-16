"""Evaluation harness: three looking policies on the same images, one segmenter, one matching rule.

    python -m evals.harness --data data/synth200 --n 200 --out public/results

Every policy uses `drishyam.segment.ClassicalSegmenter`, so only the looking policy differs:
  downscale   one read of the whole image resized to 1024 px on the long side, segmented once
  tiling      512 px tiles with 20 % overlap at L0, plus the same grid read from pyramid levels 1 and 2
  drishyam    `Job` with pixel_cap in BUDGETS, thumbs off

A ground-truth box is FOUND when a prediction of the same class overlaps it at IoU >= 0.3 or contains its
centre (PRD). Pixels are counted as pixels fed to the segmenter (tile overlap and re-looks included, the
saliency overview included for Drishyam) over the L0 pixel count, so exhaustive 20 %-overlap tiling costs
1.67x the image. $/image = wall seconds x $0.339/3600.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from bench.micro import provenance
from drishyam import imaging
from drishyam.agent import PLANNER_MALFORMED_STEP, PLANNER_TIMEOUT_STEP, Job, JobConfig
from drishyam.findings import SPALL_CRITICAL_AREA_PX, is_critical
from drishyam.governor import USD_PER_HOUR
from drishyam.segment import DEFECTS, ClassicalSegmenter, restrict
from evals import codebrim, synth

CLASSES = DEFECTS  # crack, spalling, efflorescence, exposed_rebar, corrosion
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
HEADLINE_BUDGET = 0.30
TILE, OVERLAP, TILE_LEVELS = 512, 0.2, (2, 1, 0)  # coarsest grid first
LONG_SIDE = 1024
IOU_MIN = 0.3
GOVERNOR_CAUSES = ("p95>SLA", "warp-spike", "usd≥80%")


# ---------------------------------------------------------------- matching
def contains_centre(pred_box, gt_box) -> bool:
    cx, cy = gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3] / 2
    return pred_box[0] <= cx <= pred_box[0] + pred_box[2] and pred_box[1] <= cy <= pred_box[1] + pred_box[3]


def hits(pred_box, gt_box, centre: bool = True) -> bool:
    """The PRD rule: IoU >= 0.3, or the prediction contains the GT centre (`centre=False` is the strict variant)."""
    return imaging.box_iou(pred_box, gt_box) >= IOU_MIN or (centre and contains_centre(pred_box, gt_box))


def found(gt: dict, preds: list[dict], centre: bool = True) -> bool:
    return any(p["cls"] == gt["cls"] and hits(p["box"], gt["box"], centre) for p in preds)


def gt_critical(gt: dict) -> bool:
    """Exposed rebar, or spalling whose box is at least the critical area (GT has boxes, not masks)."""
    return is_critical(gt["cls"], gt["box"][2] * gt["box"][3])


def critical_counts(preds: list[dict], gts: list[dict]) -> tuple[int, int]:
    """(critical predictions, those matching no critical GT box of their class)."""
    crit_gt = [g for g in gts if gt_critical(g)]
    crit = [p for p in preds if is_critical(p["cls"], p["area"])]
    return len(crit), sum(not any(g["cls"] == p["cls"] and hits(p["box"], g["box"]) for g in crit_gt) for p in crit)


# ---------------------------------------------------------------- policies
def slice_boxes(h: int, w: int, size: int = TILE, overlap: float = OVERLAP) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) tiles covering an h x w image; mirrors SAHI's `slice_image` / `get_slice_bboxes`:
    a stride of size x (1 - overlap) and a last row/column pulled back to the edge instead of shrunk."""
    ov = int(overlap * size)
    boxes, y_min, y_max = [], 0, 0
    while y_max < h:
        x_min = x_max = 0
        y_max = y_min + size
        while x_max < w:
            x_max = x_min + size
            if y_max > h or x_max > w:
                xe, ye = min(w, x_max), min(h, y_max)
                boxes.append((max(0, xe - size), max(0, ye - size), xe, ye))
            else:
                boxes.append((x_min, y_min, x_max, y_max))
            x_min = x_max - ov
        y_min = y_max - ov
    return boxes


def preds_from_seg(seg, to_l0: np.ndarray) -> list[dict]:
    """One L0 box per connected component of each class mask (crop -> L0 through `to_l0`)."""
    det, out = abs(float(np.linalg.det(to_l0[:2, :2]))), []
    for ci, cls in enumerate(DEFECTS, start=1):
        if not seg.areas[cls]:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(seg.mask(cls), connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = (int(v) for v in stats[i])
            (ax, ay), (bx, by) = imaging.map_points(to_l0, [[x, y], [x + w, y + h]])
            out.append({"cls": cls, "box": [round(ax), round(ay), round(bx - ax), round(by - ay)],
                        "area": round(a * det), "conf": round(float(seg.probs[ci][lab == i].mean()), 3)})
    return out


def union_from_seg(seg, to_l0: np.ndarray) -> list[dict]:
    """One L0 box per class: the bounding box of all that class's pixels in the view (what a Drishyam finding reports)."""
    det, out = abs(float(np.linalg.det(to_l0[:2, :2]))), []
    for cls in DEFECTS:
        if not seg.areas[cls]:
            continue
        mask = seg.mask(cls)
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        (ax, ay), (bx, by) = imaging.map_points(to_l0, [[x, y], [x + w, y + h]])
        biggest = int(cv2.connectedComponentsWithStats(mask, connectivity=8)[2][1:, cv2.CC_STAT_AREA].max())
        out.append({"cls": cls, "box": [round(ax), round(ay), round(bx - ax), round(by - ay)],
                    "area": round(biggest * det), "conf": round(seg.confidence[cls], 3)})
    return out


def merge_boxes(preds: list[dict], ios: float = 0.5) -> list[dict]:
    """Greedy same-class merge of boxes whose intersection over the smaller box is >= `ios` (SAHI's GREEDYNMM).
    ponytail: the merged area is a sum minus the smaller overlap, an estimate; exact would need the masks."""
    out: list[dict] = []
    for p in sorted(preds, key=lambda p: -p["conf"]):
        px0, py0, pw, ph = p["box"]
        for q in out:
            if q["cls"] != p["cls"]:
                continue
            qx0, qy0, qw, qh = q["box"]
            iw = min(px0 + pw, qx0 + qw) - max(px0, qx0)
            ih = min(py0 + ph, qy0 + qh) - max(py0, qy0)
            inter = max(0, iw) * max(0, ih)
            if inter and inter / max(1, min(pw * ph, qw * qh)) >= ios:
                x0, y0 = min(px0, qx0), min(py0, qy0)
                q["box"] = [x0, y0, max(px0 + pw, qx0 + qw) - x0, max(py0 + ph, qy0 + qh) - y0]
                q["area"] += p["area"] - min(q["area"], p["area"], inter)
                break
        else:
            out.append(dict(p))
    return out


def run_downscale(img: np.ndarray, seg) -> dict:
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    k = min(1.0, LONG_SIDE / max(h, w))
    small = img if k == 1.0 else cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    out, to_l0 = seg.predict(small), np.diag([1 / k, 1 / k, 1.0])
    return {"preds": preds_from_seg(out, to_l0), "preds_union": union_from_seg(out, to_l0), "pixels_frac": small.shape[0] * small.shape[1] / (h * w),
            "wall_s": time.perf_counter() - t0}


def run_tiling(img: np.ndarray, seg, level: int) -> dict:
    """SAHI-style slicing of pyramid level `level` (0 = full resolution)."""
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    base, s = imaging.build_pyramid(img, level + 1)[level], 2.0**level
    preds, union, px = [], [], 0
    for x0, y0, x1, y1 in slice_boxes(*base.shape[:2]):
        crop = np.ascontiguousarray(base[y0:y1, x0:x1])
        px += crop.shape[0] * crop.shape[1]
        out, to_l0 = seg.predict(crop), np.array([[s, 0, s * x0], [0, s, s * y0], [0, 0, 1.0]])
        preds += preds_from_seg(out, to_l0)
        union += union_from_seg(out, to_l0)
    return {"preds": merge_boxes(preds), "preds_union": merge_boxes(union), "pixels_frac": px / (h * w), "wall_s": time.perf_counter() - t0}


class RecordingSegmenter:
    """ClassicalSegmenter that keeps every Segmentation, so per-component boxes can be rebuilt after a job."""

    def __init__(self) -> None:
        self.inner, self.log = ClassicalSegmenter(), []

    def predict(self, crop, model="accurate"):
        self.log.append(self.inner.predict(crop, model))
        return self.log[-1]


def gaze_to_l0(x: int, y: int, level: int, size: int) -> np.ndarray:
    """The crop -> L0 matrix of a yaw = pitch = 0 gaze (a pure translation at pyramid scale), as `imaging.gaze` builds it."""
    s = 2.0**level
    return np.array([[s, 0, s * (x / s - size / 2)], [0, s, s * (y / s - size / 2)], [0, 0, 1.0]])


def finding_components(job: Job, segs: list) -> list[dict]:
    """Drishyam's evidence in the baselines' box semantics: per-component boxes of the classes behind each
    finding, from every look in its evidence trail. Lets the harness score both box conventions (see `run`)."""
    looks = [r for r in job.trace if r["decision"]["action"] != "halt"]
    out, wh = [], job.image.shape[1::-1]
    for rec, seg in zip(looks, segs, strict=True):
        classes = {f.cls for f in job.findings if rec["step"] in f.steps}
        if classes:
            a = rec["decision"]["args"]
            to_l0 = gaze_to_l0(a["x"], a["y"], a["level"], a["size"])
            seen = restrict(seg, imaging.footprint(to_l0, wh, a["size"]))  # what the job saw: no mirrored pixels
            out += [{**p, "step": rec["step"], "level": a["level"]} for p in preds_from_seg(seen, to_l0) if p["cls"] in classes]
    return out


def run_drishyam(img: np.ndarray, cap: float) -> dict:
    t0 = time.perf_counter()
    rec_seg = RecordingSegmenter()
    job = Job(img, JobConfig(pixel_cap=cap, thumbs=False), segmenter=rec_seg)
    done = list(job.run())[-1]
    wall = time.perf_counter() - t0
    preds = [{"cls": f.cls, "box": list(f.box), "area": int(f.measure["area_px"]), "conf": round(f.confidence, 3),
              "step": f.step, "level": f.level, "region": f.region} for f in job.findings]
    trace = [{"step": r["step"], "action": r["decision"]["action"], "cause": r["cause"],
              "x": r["decision"]["args"].get("x"), "y": r["decision"]["args"].get("y"),
              "level": r["decision"]["args"].get("level"), "window": r.get("window"),
              "q_ok": r.get("quality", {}).get("ok", True), "q_reason": r.get("quality", {}).get("reason"),
              "sharpness": r.get("observation", {}).get("sharpness"), "model": r.get("model")} for r in job.trace]
    return {"preds": finding_components(job, rec_seg.log), "preds_union": preds,
            "pixels_frac": done["summary"]["pixels_frac"], "wall_s": wall, "trace": trace,
            "status": done["status"], "halt_cause": done["cause"]}


def eval_image(rec: dict, budgets=BUDGETS) -> dict:
    """Every policy on one image (a worker task)."""
    img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
    seg = ClassicalSegmenter()
    cfg = {"downscale": run_downscale(img, seg)}
    for lvl in TILE_LEVELS:
        cfg[f"tiling@L{lvl}"] = run_tiling(img, seg, lvl)
    for cap in budgets:
        cfg[f"drishyam@{cap}"] = run_drishyam(img, cap)
    return {"id": rec["id"], "width": img.shape[1], "height": img.shape[0], "configs": cfg}


# ---------------------------------------------------------------- failure drills
def run_chaos(img: np.ndarray, kind: str) -> dict:
    """One Drishyam job with a injected fault; did it recover / did the governor respond?"""
    over = {"planner_timeout_s": 0.2} if kind == "planner_timeout" else {}
    job = Job(img, JobConfig(pixel_cap=1.0, thumbs=False, chaos=(kind,), **over))
    events = list(job.run())
    steps = [e for e in events if e["type"] == "step" and e["decision"]["action"] != "halt"]
    completed = events[-1]["type"] == "done"
    out = {"kind": kind, "steps": len(steps), "completed": completed, "fired": False, "ok": False}
    if kind in ("planner_timeout", "planner_malformed"):
        at = PLANNER_TIMEOUT_STEP if kind == "planner_timeout" else PLANNER_MALFORMED_STEP
        ev = next((e for e in events if e["type"] == "planner" and e["step"] == at), None)
        step = next((s for s in steps if s["step"] == at), None)
        out["fired"] = ev is not None
        out["ok"] = bool(ev and ev["proposal"] is None and step and completed)  # rule planner carried on
        out["detail"] = {"status": ev and ev["status"], "reason": ev and ev["reason"], "step_cause": step and step["cause"]}
    else:
        spike = next((s for s in steps if s.get("injected") == "p95_spike"), None)
        # The next record: a look, or a halt record (stop rule, exhausted) that carries the override under `governor`.
        after = next((e for e in events if spike and e["type"] == "step" and e["step"] == spike["step"] + 1), None)
        cause = after and (after.get("governor") or after)["cause"]
        out["fired"] = spike is not None
        out["ok"] = bool(cause == "p95>SLA" and (after["model"] != spike["model"] or job.settings.level_bias > 0))
        out["detail"] = {"next_step_cause": cause, "next_action": after and after["decision"]["action"],
                         "model": [spike and spike["model"], after and after["model"]]}
    return out


def chaos_task(arg: tuple[dict, str]) -> dict:
    rec, kind = arg
    return {"id": rec["id"], **run_chaos(cv2.imread(rec["path"], cv2.IMREAD_COLOR), kind)}


# ---------------------------------------------------------------- pool
def _init_worker() -> None:
    cv2.setNumThreads(1)  # one thread per worker process: what a vCPU-per-worker deployment would run
    warm = np.random.default_rng(0).integers(0, 255, (768, 1024, 3)).astype(np.uint8)
    ClassicalSegmenter().predict(warm[:512, :512].copy())
    list(Job(warm, JobConfig(pixel_cap=0.5, thumbs=False)).run())


def pool_map(fn, items: list, workers: int, log, label: str) -> list:
    """Ordered results; spawn-based process pool, or inline when workers <= 1."""
    t0, out = time.time(), [None] * len(items)
    every = max(1, len(items) // 10)
    if workers <= 1:
        for i, item in enumerate(items):
            out[i] = fn(item)
            if (i + 1) % every == 0:
                log(f"  {label}: {i + 1}/{len(items)}  {time.time() - t0:.0f}s")
        return out
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        futs = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for n, fut in enumerate(as_completed(futs), start=1):
            out[futs[fut]] = fut.result()
            if n % every == 0 or n == len(items):
                el = time.time() - t0
                log(f"  {label}: {n}/{len(items)}  {el:.0f}s elapsed, ~{el / n * (len(items) - n):.0f}s left")
    return out


# ---------------------------------------------------------------- scoring
# How each policy's output becomes boxes. `preds` is per connected component, `preds_union` one box per class per
# view (a Drishyam finding). The union box swallows more GT centres, so mixing conventions between policies flatters
# whichever uses it; the primary scoring therefore uses one convention for all, and the as-specified mix is reported too.
CONVENTIONS = {"components": lambda key: "preds", "union": lambda key: "preds_union",
               "as_specified": lambda key: "preds_union" if key.startswith("drishyam") else "preds"}


def load_records(data_dir, n: int, kind: str = "auto") -> tuple[list[dict], str]:
    d = Path(data_dir)
    if kind == "auto":
        kind = "synth" if (d / "annotations.json").exists() or not (d / "eval200.txt").exists() else "codebrim"
    if kind == "synth":
        if not (d / "annotations.json").exists():
            print(f"{d}/annotations.json missing: building synthetic-{n} (about 4 minutes)", flush=True)
            synth.build_dataset(d, n=n, seed=7)
        recs = [{**r, "path": str(d / r["file"])} for r in json.loads((d / "annotations.json").read_text())["images"]]
        recs = recs[:n]
        return recs, f"synthetic-{len(recs)} (labelled synthetic)"
    recs = codebrim.load_eval(d, d / "eval200.txt")[:n]
    return recs, f"CODEBRIM eval subset ({len(recs)} images)"


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def score_config(key: str, records: list[dict], results: list[dict], field: str = "preds") -> dict:
    tot, fnd, strict = ({c: 0 for c in CLASSES} for _ in range(3))
    crit = crit_false = n_preds = 0
    px, wall, flags = [], [], {}
    for rec, res in zip(records, results, strict=True):
        c = res["configs"][key]
        preds = c[field]
        flags[rec["id"]] = [found(g, preds) for g in rec["boxes"]]
        for g, f in zip(rec["boxes"], flags[rec["id"]], strict=True):
            tot[g["cls"]] += 1
            fnd[g["cls"]] += f
            strict[g["cls"]] += found(g, preds, centre=False)
        k, bad = critical_counts(preds, rec["boxes"])
        crit, crit_false, n_preds = crit + k, crit_false + bad, n_preds + len(preds)
        px.append(c["pixels_frac"])
        wall.append(c["wall_s"])
    ratio = lambda a, b: a / b if b else None
    return {"recall": ratio(sum(fnd.values()), sum(tot.values())),
            "recall_iou_only": ratio(sum(strict.values()), sum(tot.values())),
            "per_class": {c: ratio(fnd[c], tot[c]) for c in CLASSES}, "gt_per_class": tot,
            "pixels_frac": float(np.mean(px)), "wall_s": float(np.mean(wall)),
            "usd": float(np.mean(wall)) * USD_PER_HOUR / 3600,
            "critical_predictions": crit, "critical_false": crit_false, "false_accept_rate": ratio(crit_false, crit),
            "predictions_per_image": n_preds / len(records), "flags": flags}


def agent_metrics(results: list[dict], chaos: list[dict], budgets, workers: int) -> dict:
    by_budget, causes, total = {}, 0, 0
    for cap in budgets:
        tr = [r["configs"][f"drishyam@{cap}"] for r in results]
        looks = [sum(t["action"] != "halt" for t in c["trace"]) for c in tr]
        acts = [t["action"] for c in tr for t in c["trace"]]
        n_looks = max(1, sum(looks))
        frac = [c["pixels_frac"] for c in tr]
        px = [c["pixels_frac"] * r["width"] * r["height"] for c, r in zip(tr, results, strict=True)]
        by_budget[str(cap)] = {
            "escalation_rate": _r(acts.count("escalate") / n_looks), "verify_rate": _r(acts.count("verify") / n_looks),
            "steps_p50": _r(np.percentile(looks, 50), 1), "steps_p95": _r(np.percentile(looks, 95), 1),
            "pixels_frac_p50": _r(np.percentile(frac, 50)), "pixels_frac_p95": _r(np.percentile(frac, 95)),
            "pixels_p50": round(float(np.percentile(px, 50))), "pixels_p95": round(float(np.percentile(px, 95))),
            "halt_causes": {k: sum(c["halt_cause"] == k for c in tr) for k in sorted({c["halt_cause"] for c in tr})}}
        causes += sum(bool(t["cause"]) for c in tr for t in c["trace"])
        total += sum(len(c["trace"]) for c in tr)
    gov = {k: sum(t["cause"] == k for r in results for cap in budgets for t in r["configs"][f"drishyam@{cap}"]["trace"])
           for k in GOVERNOR_CAUSES}
    by_kind = lambda k: [c for c in chaos if c["kind"] == k]
    tally = lambda k: {"injected": len(by_kind(k)), "fired": sum(c["fired"] for c in by_kind(k)),
                       "recovered": sum(c["ok"] for c in by_kind(k))}
    timeout, malformed, spike = tally("planner_timeout"), tally("planner_malformed"), tally("p95_spike")
    head = by_budget[str(HEADLINE_BUDGET)]
    return {
        "headline_budget": HEADLINE_BUDGET,
        "escalation_rate": head["escalation_rate"],
        "steps_per_image": {"p50": head["steps_p50"], "p95": head["steps_p95"]},
        "pixels_per_image": {"p50": head["pixels_p50"], "p95": head["pixels_p95"],
                             "p50_frac": head["pixels_frac_p50"], "p95_frac": head["pixels_frac_p95"]},
        "steps_with_cause_pct": _r(100 * causes / max(1, total), 2),
        "by_budget": by_budget,
        "planner_recovery": {"timeout": timeout, "malformed": malformed,
                             "recovered_total": timeout["recovered"] + malformed["recovered"],
                             "injected_total": timeout["injected"] + malformed["injected"]},
        "governor_response": spike,
        "governor_activity_in_eval": gov,
        "timing": {"workers": workers, "cv2_threads_per_worker": 1 if workers > 1 else cv2.getNumThreads(),
                   "note": "wall times are per image while the other workers were busy; read ratios, not absolutes"},
    }


def micro_rows(arm: str, results_dir) -> list[dict]:
    p = Path(results_dir) / arm / "micro.json"
    if not p.exists():
        return []
    return [{"op": r["op"], "size": r["size"], "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"]}
            for r in json.loads(p.read_text())["results"] if r["size"] == "4000x3000"]


def point(m: dict) -> dict:
    return {"pixels_frac": _r(m["pixels_frac"]), "recall": _r(m["recall"]), "usd": _r(m["usd"], 6)}


def headline(scored: dict, primary: str, n: int, label: str) -> str:
    """Two sentences: the hypothesis test against L0 tiling, then every other baseline, as recall ratios."""
    m, key = scored[primary], f"drishyam@{HEADLINE_BUDGET}"
    ratio_of = lambda sc, base, f="recall": sc[key][f] / sc[base][f] if sc[base][f] else float("nan")
    d, t, s = m[key], m["tiling@L0"], m["downscale"]
    px = d["pixels_frac"] / t["pixels_frac"]
    verdict = "meets" if ratio_of(m, "tiling@L0") >= 0.9 and px <= 0.3 else "does not meet"
    names = {"components": "per-component", "union": "one-box-per-class", "as_specified": "as-specified mixed"}
    others = ", ".join(f"{ratio_of(sc, 'tiling@L0'):.2f}x with {names[name]} boxes" for name, sc in scored.items() if name != primary)
    best = max((f"tiling@L{lv}" for lv in TILE_LEVELS), key=lambda k: m[k]["recall"])
    grids = " and ".join(f"{ratio_of(m, k):.2f}x L{k[-1]} tiling ({m[k]['recall']:.2f} at {m[k]['pixels_frac']:.2f}"
                         f"{', the best grid' if k == best else ''})" for k in ("tiling@L1", "tiling@L2"))
    return (f"On {n} {label}, Drishyam at a {HEADLINE_BUDGET:.0%} pixel cap found {d['recall']:.2f} of ground-truth boxes using "
            f"{d['pixels_frac']:.2f} of the image's pixels, versus {t['recall']:.2f} for 20%-overlap 512-px tiling using "
            f"{t['pixels_frac']:.2f}: {ratio_of(m, 'tiling@L0'):.2f}x its recall ({others}) at {px:.2f}x its pixels, which "
            f"{verdict} the 0.9x-at-0.3x hypothesis. It is {ratio_of(m, 'downscale'):.2f}x a single 1024-px downscale "
            f"({s['recall']:.2f} at {s['pixels_frac']:.2f} of the pixels), {grids}, and {ratio_of(m, 'tiling@L0', 'recall_iou_only'):.2f}x "
            f"L0 tiling when a match needs IoU >= {IOU_MIN} (no contains-the-centre clause).")


# ---------------------------------------------------------------- run
def run(data_dir="data/synth200", n=200, out_dir="public/results", results_dir="results", workers=None,
        chaos=(20, 20, 10), kind="auto", micro_arm="local-macos-x86", primary_boxes="components", reuse=False,
        log=lambda msg: print(msg, flush=True)) -> dict:
    from evals import failures

    t_start = time.time()
    workers = max(1, min(8, (mp.cpu_count() or 2) // 2)) if workers is None else workers
    records, label = load_records(data_dir, n, kind)
    log(f"{len(records)} images, {workers} worker(s); policies: downscale, tiling L2/L1/L0, drishyam {BUDGETS}")
    cache = Path(results_dir) / "eval_per_image.json"
    if reuse and cache.exists():  # re-score and re-draw from the last run's predictions and traces
        saved = json.loads(cache.read_text())
        assert saved["ids"] == [r["id"] for r in records], "cached run is for a different image list"
        results = saved["results"]
        log(f"reusing {cache}")
    else:
        results = pool_map(eval_image, records, workers, log, "eval")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"ids": [r["id"] for r in records], "results": results},
                                    default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    keys = ["downscale", *(f"tiling@L{lv}" for lv in TILE_LEVELS), *(f"drishyam@{c}" for c in BUDGETS)]
    scored = {conv: {k: score_config(k, records, results, field(k)) for k in keys} for conv, field in CONVENTIONS.items()}
    metrics = scored[primary_boxes]

    # fault drills run on images where the full-budget job went long enough to reach the injection step
    full = f"drishyam@{max(BUDGETS)}"
    eligible = [rec for rec, res in zip(records, results, strict=True)
                if sum(t["action"] != "halt" for t in res["configs"][full]["trace"]) >= 7]
    wanted = [("planner_timeout", chaos[0]), ("planner_malformed", chaos[1]), ("p95_spike", chaos[2])]
    jobs = [(eligible[i % len(eligible)], k) for k, cnt in wanted for i in range(cnt)] if eligible else []
    log(f"fault drills: {len(jobs)} jobs on {len(eligible)} long-running images")
    drills = pool_map(chaos_task, jobs, workers, log, "drills") if jobs else []

    am = agent_metrics(results, drills, BUDGETS, workers)
    nominal = {"tiling@L0": "1.0", "tiling@L1": "0.25", "tiling@L2": "0.0625"}  # nominal budget of each grid
    per_class = {"downscale": {c: _r(v) for c, v in metrics["downscale"]["per_class"].items()}}
    for k, m in metrics.items():
        if k.startswith("tiling"):
            per_class[f"tiling@{nominal[k]}"] = {c: _r(v) for c, v in m["per_class"].items()}
        elif k.startswith("drishyam"):
            per_class[k] = {c: _r(v) for c, v in m["per_class"].items()}
    am["false_accept_critical"] = {k: {"critical_predictions": m["critical_predictions"], "false": m["critical_false"],
                                       "rate": _r(m["false_accept_rate"])} for k, m in metrics.items()}
    am["false_accept_critical"]["note"] = (f"a critical prediction is exposed_rebar or spalling >= {SPALL_CRITICAL_AREA_PX} px^2; "
                                          "false = matches no critical GT box (spalling GT is critical at box area >= the same threshold)")

    am["box_convention"] = {
        "primary": primary_boxes,
        "note": ("components: every policy reports one box per connected component; union: every policy reports one box per "
                 "class per view (a Drishyam finding); as_specified: findings boxes for Drishyam, components for the baselines. "
                 "Union boxes contain more GT centres, so as_specified favours Drishyam. iou_only drops the contains-centre clause."),
        **{what: {conv: {k: _r(m[field]) for k, m in sc.items()} for conv, sc in scored.items()}
           for what, field in (("recall", "recall"), ("recall_iou_only", "recall_iou_only"),
                               ("false_accept_rate", "false_accept_rate"))},
        "predictions_per_image": {conv: {k: _r(m["predictions_per_image"], 1) for k, m in sc.items()} for conv, sc in scored.items()},
    }
    prov = provenance()
    keep = {k: prov[k] for k in ("cv2", "build_info_sha256", "machine", "cpu", "python", "timestamp", "build")}
    keep["machine"] += (f", {prov['logical_cpus']} logical CPUs; eval ran {workers} worker(s) x "
                        f"{am['timing']['cv2_threads_per_worker']} cv2 thread(s); micro arm {micro_arm}")
    summary = {
        "dataset": label,
        "n_images": len(records),
        "policies": {
            "downscale": [point(metrics["downscale"])],
            "tiling": [point(metrics[f"tiling@L{lv}"]) for lv in TILE_LEVELS],
            "drishyam": [point(metrics[f"drishyam@{c}"]) for c in BUDGETS],
        },
        "per_class": per_class,
        "agent_metrics": am,
        "micro": micro_rows(micro_arm, results_dir),
        "provenance": keep,
        "headline": headline(scored, primary_boxes, len(records), "labelled synthetic images" if label.startswith("synthetic") else "CODEBRIM images"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    gallery = failures.build_gallery(records, results, out, log=log)
    (out / "failures.json").write_text(json.dumps(gallery, indent=1))

    raw = Path(results_dir)
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "eval_raw.json").write_text(json.dumps({
        "wall_s": round(time.time() - t_start, 1), "workers": workers, "provenance": prov,
        "metrics": scored, "drills": drills, "gallery_cases": len(gallery)}, indent=1, default=float))
    log(f"done in {time.time() - t_start:.0f}s; wrote {out}/summary.json, failures.json ({len(gallery)} cases)")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.harness", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/synth200")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="public/results")
    ap.add_argument("--results-dir", default="results", help="raw run JSON (git-ignored)")
    ap.add_argument("--workers", type=int, default=None, help="default: min(8, cpus/2); 1 runs inline")
    ap.add_argument("--chaos", default="20,20,10", help="planner_timeout,planner_malformed,p95_spike job counts")
    ap.add_argument("--kind", choices=("auto", "synth", "codebrim"), default="auto")
    ap.add_argument("--micro-arm", default="local-macos-x86", help="results/<arm>/micro.json to embed")
    ap.add_argument("--reuse", action="store_true", help="re-score and re-draw from results/eval_per_image.json (last run)")
    ap.add_argument("--primary-boxes", choices=tuple(CONVENTIONS), default="components",
                    help="box convention behind policies/per_class/headline; all three are in agent_metrics.box_convention")
    a = ap.parse_args(argv)
    run(a.data, a.n, a.out, a.results_dir, a.workers, tuple(int(x) for x in a.chaos.split(",")), a.kind, a.micro_arm,
        a.primary_boxes, a.reuse)


if __name__ == "__main__":
    main()
