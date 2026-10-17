"""Failure gallery: real misses and false alarms found in the evaluation run, drawn with the gaze path.

Nothing is staged. Each case is a ground-truth box, a prediction, a distractor record (graffiti, formwork
line, water stain, glare, blur region) or a quality-gate event that the run itself produced; the text states
what the numbers say and the rule that would have caught it. The rules are proposals, not implemented code.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from drishyam.findings import is_critical

# BGR, the site's drawing palette: ground truth in sky, predictions in callout amber, on blueprint deep.
SKY, AMBER, WHITE, INK = (255, 200, 156), (87, 200, 255), (255, 241, 232), (85, 42, 8)
PREFIX = "failures"  # image paths in failures.json, relative to public/results/ (what public/js/results.js prepends)
GAZE_CFGS = ("drishyam@0.3", "drishyam@1.0")
MAX_W, MAX_H = 900, 800
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _inside(a, b) -> float:
    """Fraction of box a (x, y, w, h) lying inside box b."""
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / max(1, a[2] * a[3])


def _centre_in(gt_box, window) -> bool:
    return (window[0] <= gt_box[0] + gt_box[2] / 2 <= window[0] + window[2]
            and window[1] <= gt_box[1] + gt_box[3] / 2 <= window[1] + window[3])


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def _mark(kind, label, box):
    return {"kind": kind, "label": label, "box": list(box)}


def _visits(trace, gt_box):
    return [t for t in trace if t["window"] and _centre_in(gt_box, t["window"])]


# ---------------------------------------------------------------- finders: each yields cases for one image
def _false_alarm(kind, classes, label):
    """A finding inside a distractor's bounding box that sits on no ground-truth box of any class (so it is not
    real damage that merely happens to lie under the distractor's box)."""
    def find(rec, res):
        from evals.harness import hits

        for cfg in GAZE_CFGS:
            for p in res["configs"][cfg]["preds"]:
                if p["cls"] not in classes or max(p["box"][2:]) < 40 or any(hits(p["box"], g["box"]) or _inside(p["box"], g["box"]) >= 0.3
                                                  for g in rec["boxes"]):
                    continue
                for d in (d for d in rec["distractors"] if d["kind"] == kind):
                    share = _inside(p["box"], d["box"])
                    if share >= 0.5:
                        crit = is_critical(p["cls"], p["area"])
                        yield {"id": rec["id"], "cls": p["cls"], "cfg": cfg, "score": 10 * crit + share + p["conf"],
                               "marks": [_mark("pred", f"{p['cls']} {p['conf']:.2f}", p["box"]), _mark("other", label, d["box"])],
                               "what": (f"{rec['id']}, {cfg}: {_a(p['cls'])} finding (confidence {p['conf']:.2f}, step {p['step']} at "
                                        f"L{p['level']}) lies within the bounding box of {_a(label)} patch and matches no "
                                        f"ground-truth box of any class." + (" It is critical, so it would stop at human review." if crit else ""))}
    return find


def _glare_gate(rec, res):
    from evals.harness import found

    glares = [d["box"] for d in rec["distractors"] if d["kind"] == "glare"]
    for cfg in GAZE_CFGS:
        c = res["configs"][cfg]
        for t in c["trace"]:
            hit = next((g for g in glares if t["window"] and not t["q_ok"] and _inside(g, t["window"]) >= 0.2), None)
            if hit is None:
                continue
            inside = [g for g in rec["boxes"] if _centre_in(g["box"], t["window"])]
            missed = [g for g in inside if not found(g, c["preds"])]
            gt = (missed or inside or [None])[0]
            yield {"id": rec["id"], "cls": gt and gt["cls"], "cfg": cfg, "score": 5 * len(missed) + len(inside),
                   "marks": ([_mark("gt", f"GT {gt['cls']}", gt["box"])] if gt else [])
                   + [_mark("other", "glare", hit), _mark("other", f"gaze {t['step']} (refused)", t["window"])],
                   "what": (f"{rec['id']}, {cfg}: step {t['step']} (L{t['level']}) was refused by the quality gate "
                            f"({t['q_reason']}, sharpness {t['sharpness']}) because the window covers a blown-out highlight. "
                            f"{len(inside)} ground-truth box(es) have their centre in that window"
                            + (f" ({', '.join(sorted({g['cls'] for g in inside}))}); {len(missed)} not found by the finished job. "
                               if inside else ". ")
                            + "The gate did what it should; the cost is an area the work order silently leaves unassessed.")}
            break


def _blurred_crack(rec, res):
    from evals.harness import found

    blurs = [d["box"] for d in rec["distractors"] if d["kind"] == "blur_region"]
    for g in (g for g in rec["boxes"] if g["cls"] == "crack"):
        for b in (b for b in blurs if _centre_in(g["box"], b)):
            for cfg in reversed(GAZE_CFGS):  # prefer the full-budget run: the budget is not the excuse there
                c = res["configs"][cfg]
                if found(g, c["preds"]):
                    continue
                seen = _visits(c["trace"], g["box"])
                gated = [t for t in seen if not t["q_ok"]]
                how = ("no gaze window ever covered its centre." if not seen else
                       f"it was looked at (step {seen[0]['step']}, L{seen[0]['level']}) and the quality gate refused the view "
                       f"({gated[0]['q_reason']}, sharpness {gated[0]['sharpness']})." if gated else
                       f"it was looked at (step {seen[0]['step']}, L{seen[0]['level']}, sharpness {seen[0]['sharpness']}), "
                       "the gate accepted the view, and no crack mask came out of it.")
                yield {"id": rec["id"], "cls": "crack", "cfg": cfg, "score": cfg.endswith("1.0") + len(seen) / 10,
                       "marks": [_mark("gt", "GT crack", g["box"]), _mark("other", "blurred patch", b)],
                       "what": f"{rec['id']}, {cfg}: a {g['box'][2]}x{g['box'][3]} px crack inside a blurred patch was not found; {how}"}
                break


def _budget_miss(rec, res):
    from evals.harness import found

    small = res["configs"]["drishyam@0.3"]
    for g in (g for g in rec["boxes"] if g["cls"] == "crack" and not found(g, small["preds"])):
        by = [n for n in ("drishyam@1.0", "tiling@L0") if found(g, res["configs"][n]["preds"])]
        if not by:
            continue
        seen = _visits(small["trace"], g["box"])
        looks = sum(t["action"] != "halt" for t in small["trace"])
        how = (f"it was looked at (step {seen[0]['step']}, L{seen[0]['level']}) but no crack mask survived." if seen else
               f"its window was never visited: the job ended after {looks} looks ({small['halt_cause']}) before the "
               "saliency ranking reached it.")
        yield {"id": rec["id"], "cls": "crack", "cfg": "drishyam@0.3", "score": 10 * len(by) - min(g["box"][2], g["box"][3]) / 100,
               "marks": [_mark("gt", "GT crack", g["box"])],
               "what": (f"{rec['id']}, drishyam@0.3: a {g['box'][2]}x{g['box'][3]} px crack box is missed under the 30% cap "
                        f"but found by {' and '.join(by)}; {how}")}


def _confusion(rec, res):
    from evals.harness import found, hits

    group = ("exposed_rebar", "spalling", "corrosion")
    for cfg in GAZE_CFGS:
        preds = res["configs"][cfg]["preds"]
        for g in (g for g in rec["boxes"] if g["cls"] in group and not found(g, preds)):
            wrong = [p for p in preds if p["cls"] in group and p["cls"] != g["cls"] and hits(p["box"], g["box"])]
            if wrong:
                p = max(wrong, key=lambda p: p["conf"] * p["area"])
                yield {"id": rec["id"], "cls": g["cls"], "cfg": cfg, "score": 10 * (g["cls"] == "exposed_rebar") + p["conf"],
                       "marks": [_mark("gt", f"GT {g['cls']}", g["box"]), _mark("pred", f"{p['cls']} {p['conf']:.2f}", p["box"])],
                       "what": (f"{rec['id']}, {cfg}: ground truth is {g['cls']}, the job reported {p['cls']} there "
                                f"(confidence {p['conf']:.2f}, step {p['step']} at L{p['level']}) and no {g['cls']} finding overlaps the box.")}
                break


CATEGORIES = [
    ("Graffiti read as damage", _false_alarm("graffiti", ("crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"), "graffiti"),
     ("Chroma gate before drafting: spray paint is far more saturated than concrete or rust, so a candidate whose median "
     "saturation sits well above the surrounding concrete and outside the rust hue band is dropped as non-structural. Not implemented.")),
    ("Formwork line read as crack", _false_alarm("formwork_line", ("crack",), "formwork line"),
     ("Straightness test on the crack mask: a centreline that fits one straight line within about 1.5 px over a length of "
     "20x its width is a formwork seam, not a crack, because real cracks meander (cv2.fitLine residual, minAreaRect aspect). Not implemented.")),
    ("Water stain read as damage", _false_alarm("water_stain", ("crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"), "water stain"),
     ("Texture test before a spall may be raised: a spall floor is rough, a wet stain is smooth, so require the Laplacian "
     "variance inside the mask to exceed the surrounding concrete's. Critical spalling still stops at human review. Not implemented.")),
    ("Glare caught by the quality gate", _glare_gate,
     ("The gate (more than 20% blown-out pixels) works. The missing rule is bookkeeping: a refused window is written to the "
     "work order as 'not assessed' with a re-shoot suggestion, so the area never reads as clean. Not implemented.")),
    ("Crack hidden in blur", _blurred_crack,
     ("A local sharpness map: where Laplacian variance falls below the gate, report the area as unreadable and ask for a re-shoot; "
     "where the gate accepted the view, scale the sharpness threshold to the expected crack width. Not implemented.")),
    ("Crack missed under a small budget", _budget_miss,
     ("Reserve a small share of the pixel cap for a coarse sweep of low-saliency regions, or lower the stop-rule epsilon so the "
     "job does not stop while unvisited candidates remain. Both trade pixels for recall. Not implemented.")),
    ("Rebar, spall and stain confusion", _confusion,
     ("Shape and co-occurrence rule: exposed bars need rust inside a spall rim and a linear shape (minAreaRect aspect above 3); "
     "rust with no spall rim is corrosion. Where two classes clear the threshold on one region, send it to human review instead "
     "of choosing. The rust-near-spall part exists; the review-on-ambiguity step does not.")),
]


# ---------------------------------------------------------------- drawing
def _fit(lo, hi, size, limit):
    """Window of `size` around [lo, hi], shifted (not shrunk) to stay inside [0, limit]."""
    size = int(min(size, limit))
    start = round((lo + hi) / 2 - size / 2)
    return max(0, min(start, limit - size)), size


def _text(img, s, org, color, scale=0.55):
    """Text on a dark plate, so it reads over any texture."""
    (w, h), base = cv2.getTextSize(s, FONT, scale, 1)
    cv2.rectangle(img, (org[0] - 2, org[1] - h - 2), (org[0] + w + 2, org[1] + base), INK, -1)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def _inset(img, region, marks, gaze, width=240):
    """Whole frame in miniature: the region shown, the numbered gaze path, the boxes."""
    H, W = img.shape[:2]
    k = width / W
    ins = cv2.resize(img, (width, round(H * k)), interpolation=cv2.INTER_AREA)
    Q = lambda x, y: (round(x * k), round(y * k))
    for t in gaze:
        if t["window"]:
            w = t["window"]
            cv2.rectangle(ins, Q(w[0], w[1]), Q(w[0] + w[2], w[1] + w[3]), (200, 170, 120), 1)
    if len(gaze) > 1:
        cv2.polylines(ins, [np.array([Q(t["x"], t["y"]) for t in gaze], np.int32)], False, WHITE, 1, cv2.LINE_AA)
    for m in marks:
        color = {"gt": SKY, "pred": AMBER}.get(m["kind"])
        if color:
            b = m["box"]
            cv2.rectangle(ins, Q(b[0], b[1]), Q(b[0] + b[2], b[1] + b[3]), color, 1, cv2.LINE_AA)
    for i, t in enumerate(gaze, start=1):
        c = Q(t["x"], t["y"])
        cv2.circle(ins, c, 7, INK, -1, cv2.LINE_AA)
        cv2.putText(ins, str(i), (c[0] - (3 if i < 10 else 6), c[1] + 3), FONT, 0.3, WHITE, 1, cv2.LINE_AA)
    rx, ry, rw, rh = region
    cv2.rectangle(ins, Q(rx, ry), Q(rx + rw, ry + rh), WHITE, 2)
    return cv2.copyMakeBorder(ins, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=WHITE)


def render_case(img: np.ndarray, case: dict, trace: list[dict], title: str, path: Path) -> None:
    H, W = img.shape[:2]
    boxes = [m["box"] for m in case["marks"]]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    pad = max(60, 0.3 * max(x1 - x0, y1 - y0))
    rx, rw = _fit(x0 - pad, x1 + pad, max(x1 - x0 + 2 * pad, 480), W)
    ry, rh = _fit(y0 - pad, y1 + pad, max(y1 - y0 + 2 * pad, 360), H)
    k = min(MAX_W / rw, MAX_H / rh, 2.0)
    view = cv2.resize(img[ry:ry + rh, rx:rx + rw], (round(rw * k), round(rh * k)),
                      interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    P = lambda x, y: (round((x - rx) * k), round((y - ry) * k))

    def rect(box, color, thick):
        cv2.rectangle(view, P(box[0], box[1]), P(box[0] + box[2], box[1] + box[3]), color, thick, cv2.LINE_AA)

    def label(box, s, color):
        ax, ay = P(box[0], box[1])
        _text(view, s, (max(2, ax), ay - 6 if ay > 16 else P(box[0], box[1] + box[3])[1] + 16), color)

    gaze = [t for t in trace if t["action"] != "halt" and t["x"] is not None]
    for t in gaze:
        if t["window"]:
            rect(t["window"], (200, 170, 120), 1)
    pts = [P(t["x"], t["y"]) for t in gaze]
    for a, b in pairwise(pts):  # path segments, clipped to this view
        ok, pa, pb = cv2.clipLine((0, 0, view.shape[1], view.shape[0]), a, b)
        if ok:
            cv2.line(view, pa, pb, WHITE, 1, cv2.LINE_AA)
    for m in case["marks"]:
        if m["kind"] == "other":
            rect(m["box"], WHITE, 1)
            label(m["box"], m["label"], WHITE)
    for m in case["marks"]:
        if m["kind"] != "other":
            color = SKY if m["kind"] == "gt" else AMBER
            rect(m["box"], color, 3)
            label(m["box"], m["label"], color)
    for i, (px, py) in enumerate(pts, start=1):
        if 0 <= px < view.shape[1] and 0 <= py < view.shape[0]:
            cv2.circle(view, (px, py), 11, INK, -1, cv2.LINE_AA)
            cv2.circle(view, (px, py), 11, WHITE, 1, cv2.LINE_AA)
            _text(view, str(i), (px - (4 if i < 10 else 9), py + 4), WHITE, 0.42)
    ins = _inset(img, (rx, ry, rw, rh), case["marks"], gaze)
    mx0, my0 = min(P(b[0], b[1])[0] for b in boxes), min(P(b[0], b[1])[1] for b in boxes)
    mx1, my1 = max(P(b[0] + b[2], b[1] + b[3])[0] for b in boxes), max(P(b[0] + b[2], b[1] + b[3])[1] for b in boxes)
    vh, vw = view.shape[:2]
    corners = [(vw - ins.shape[1] - 8, vh - ins.shape[0] - 8), (8, vh - ins.shape[0] - 8), (vw - ins.shape[1] - 8, 8), (8, 8)]
    overlap = lambda c: max(0, min(mx1, c[0] + ins.shape[1]) - max(mx0, c[0])) * max(0, min(my1, c[1] + ins.shape[0]) - max(my0, c[1]))
    cx, cy = min(corners, key=overlap)
    if ins.shape[0] + 16 < vh and ins.shape[1] + 16 < vw:
        view[cy:cy + ins.shape[0], cx:cx + ins.shape[1]] = ins
    legend = (("ground truth", SKY), ("prediction", AMBER), ("gaze order", WHITE))
    need = 370 + cv2.getTextSize(title, FONT, 0.5, 1)[0][0]
    rows = 1 if need < view.shape[1] else 2  # narrow crops: legend goes on its own line
    bar = np.full((34 if rows == 1 else 56, view.shape[1], 3), INK, np.uint8)
    cv2.putText(bar, title, (8, 22), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    x, y = (bar.shape[1] - 330, 11) if rows == 1 else (8, 33)
    for s, color in legend:
        cv2.rectangle(bar, (x, y), (x + 12, y + 12), color, -1)
        cv2.putText(bar, s, (x + 18, y + 11), FONT, 0.42, WHITE, 1, cv2.LINE_AA)
        x += 24 + 8 * len(s)
    cv2.imwrite(str(path), np.vstack([bar, view]), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- assembly
def build_gallery(records: list[dict], results: list[dict], out_dir, log=print) -> list[dict]:
    """Pick up to 2-4 real cases per category (>= 10 in total when the run produced them), render, return the JSON rows."""
    out = Path(out_dir) / "failures"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("f[0-9][0-9].jpg"):
        old.unlink()
    pairs = list(zip(records, results, strict=True))
    pools = []
    for title, finder, rule in CATEGORIES:
        cases = sorted((c for rec, res in pairs for c in finder(rec, res)), key=lambda c: (-c["score"], c["id"]))
        seen, uniq = set(), []
        for c in cases:
            if c["id"] not in seen:
                seen.add(c["id"])
                uniq.append(c)
        pools.append((title, rule, uniq))
    for quota in (2, 3, 4):
        chosen = [(t, r, u[:quota]) for t, r, u in pools]
        if sum(len(c) for _, _, c in chosen) >= 10:
            break
    by_id = {rec["id"]: (rec, res) for rec, res in pairs}
    rows, n = [], 0
    for title, rule, cases in chosen:
        log(f"  gallery: {title}: {len(cases)} case(s)")
        for c in cases:
            n += 1
            rec, res = by_id[c["id"]]
            trace = res["configs"][c["cfg"]].get("trace", [])
            render_case(cv2.imread(rec["path"], cv2.IMREAD_COLOR), c, trace, f"f{n:02d}  {title}  |  {c['id']}  |  {c['cfg']}",
                        out / f"f{n:02d}.jpg")
            rows.append({"image": f"{PREFIX}/f{n:02d}.jpg", "id": c["id"], "class": c["cls"], "category": title,
                         "what_happened": c["what"], "rule_that_would_catch_it": rule})
    return rows
