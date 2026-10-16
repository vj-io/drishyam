"""The inspection loop: overview, then look / escalate / verify until the stop rule or a cap.

`Job.run()` yields the NDJSON events in docs/api-contract.md. Every step's
decision is a logged function of an OpenCV output (entropy, mask area, IoU,
quality) or a runtime statistic (p95, $, pixels), and carries a `cause`.
"""

from __future__ import annotations

import math
import secrets
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import imaging
from .findings import SEVERITY, Finding, now_iso, work_order
from .governor import USD_PER_HOUR, Budget, RuntimeStats, Settings, check
from .measure import measure
from .planner import TIMEOUT_S, ChaosPlanner, consult
from .segment import CLASSES, DEFECT_PX_BG_MAX, MIN_BLOB_PX, default_segmenter

SPIKE_STEP = 4  # chaos p95_spike lands here; the governor reacts on the next step
SPIKE_MS = 900.0
PLANNER_TIMEOUT_STEP = 3
PLANNER_MALFORMED_STEP = 6
REPLAY_ACTIONS = ("gaze", "escalate", "verify")
DOMINATED_BY = {"corrosion": ("exposed_rebar",), "crack": ("spalling", "exposed_rebar"),
                "efflorescence": ("spalling", "exposed_rebar")}
MAX_REPLAY = 200


@dataclass
class JobConfig:
    pixel_cap: float = 0.30
    usd_cap: float = 0.05
    seconds_cap: float = 20.0
    size: int = 512
    first_level: int = 1
    tau: float = 0.5  # entropy above this escalates
    a_min: int = 64  # crop px; a smaller mask escalates
    eps: float = 0.25  # stop-rule slope factor
    sla_ms: float = 500.0
    max_candidates: int = 20
    chaos: tuple[str, ...] = ()
    planner_every: int = 5
    planner_timeout_s: float = TIMEOUT_S
    usd_per_hour: float = USD_PER_HOUR
    thumbs: bool = True
    max_steps: int = 200


@dataclass
class Region:
    id: str
    cx: int
    cy: int
    prior: float
    next_level: int
    entropy: float = 0.0
    cls: str | None = None
    looks: list = field(default_factory=list)  # (level, class mask, to_l0)
    done: bool = False
    verified: bool = False
    extra_used: bool = False


def _rect_on(shape, window, level_scale):
    """Window [x, y, w, h] in L0 -> slice bounds on a map whose pixels are `level_scale` L0 px."""
    x, y, w, h = window
    H, W = shape
    x0, y0 = max(0, int(x) // level_scale), max(0, int(y) // level_scale)
    x1, y1 = min(W, math.ceil((x + w) / level_scale)), min(H, math.ceil((y + h) / level_scale))
    return y0, y1, x0, x1


class Job:
    def __init__(self, image: np.ndarray, config: JobConfig | None = None, *, job_id: str | None = None,
                 segmenter=None, planner=None, kill=lambda: False, clock=time.perf_counter,
                 replay: list | None = None) -> None:
        self.image = image
        self.cfg = config or JobConfig()
        self.id = job_id or "j_" + secrets.token_hex(3)
        self.segmenter = segmenter or default_segmenter()
        self.planner = planner
        self.kill = kill
        self.clock = clock
        self.replay = validate_replay(replay or [], image.shape)
        self.findings: list[Finding] = []
        self.gaze_path: list[list[int]] = []
        self.trace: list[dict] = []

    # ---- helpers -------------------------------------------------------------------------------
    def _timed(self, op, fn, *args, **kw):
        t0 = self.clock()
        out = fn(*args, **kw)
        ms = (self.clock() - t0) * 1000
        self.stats.record(op, ms)
        return out, ms

    def _window(self, cx, cy, level, size):
        half = size * 2**level / 2
        H, W = self.image.shape[:2]
        cx = int(min(max(cx, half), W - half)) if W > 2 * half else W // 2
        cy = int(min(max(cy, half), H - half)) if H > 2 * half else H // 2
        return cx, cy, [int(cx - half), int(cy - half), int(2 * half), int(2 * half)]

    def _unseen(self, window, level) -> float:
        seen = np.logical_or.reduce(self.seen[: level + 1])
        y0, y1, x0, x1 = _rect_on(seen.shape, window, self.ov_scale)
        patch = seen[y0:y1, x0:x1]
        return 1.0 - float(patch.mean()) if patch.size else 0.0

    def _t_hat(self) -> float:
        """Predicted ms for one look; a model with no samples yet borrows the median of all segment calls."""
        seg_prior = self.stats.median("segment") or 30.0
        return self.stats.p("warp", 95, 5.0) + self.stats.p(f"segment:{self.settings.model}", 95, seg_prior)

    def _score(self, r: Region):
        level = min(r.next_level + self.settings.level_bias, self.max_level)
        _, _, win = self._window(r.cx, r.cy, level, self.settings.size)
        a = self._unseen(win, level)
        ev = r.prior * SEVERITY.get(r.cls, 1.0) * (1 + r.entropy) * a
        return ev / max(self._t_hat(), 1e-3), ev, a

    def _verify_crowded_out(self, r: Region, seg, level: int, value: float, cost_ms: float) -> dict | None:
        """Rule 3 under a pixel budget, for a non-critical mask. A verify re-reads covered ground, so it is scored
        like a region with the prior replaced by the doubt left in the mask (1 - confidence), and it keeps its look
        only if the looks left after this one also cover the unseen ground of every unvisited candidate that
        outscores it and the stop rule's lambda. Returns the skip record for the trace, or None to verify."""
        b, size = self.budget, self.settings.size
        looks_left = (b.pixels_cap - b.pixels_used) // size**2 - 1
        ev = SEVERITY[r.cls] * (1 + seg.entropy) * (1 - seg.confidence[r.cls])
        score = ev / max(self._t_hat(), 1e-3)
        bar = max(score, self.cfg.eps * value / cost_ms if cost_ms else 0.0)
        need, win_px, n = np.zeros(self.seen[0].shape, bool), 1.0, 0
        for o in self.regions.values():
            if o.done or o.looks or self._score(o)[0] < bar:
                continue
            lvl = min(o.next_level + self.settings.level_bias, self.max_level)
            y0, y1, x0, x1 = _rect_on(need.shape, self._window(o.cx, o.cy, lvl, size)[2], self.ov_scale)
            need[y0:y1, x0:x1] |= ~np.logical_or.reduce(self.seen[: lvl + 1])[y0:y1, x0:x1]
            win_px, n = max(win_px, (size * 2**lvl / self.ov_scale) ** 2), n + 1
        # ponytail: looks needed = unseen area / one window, a lower bound (looks overlap); errs toward verifying
        looks_needed = math.ceil(np.count_nonzero(need) / win_px)
        if not 0 < looks_left <= looks_needed:
            return None
        return {"action": "verify", "region": r.id, "level": level - 1, "cause": "looks≤candidates",
                "rationale": f"skipped L{level - 1} verify of {r.cls} on {r.id} (score {score:.2e}): "
                             f"{looks_left} look(s) left, {n} unvisited candidate(s) score higher and need "
                             f"{looks_needed} look(s) to cover"}

    def digest(self) -> dict:
        """Numbers and fixed names only: what the optional LLM planner is allowed to see."""
        b, s = self.budget, self.settings
        live = sorted((r for r in self.regions.values() if not r.done), key=lambda r: -self._score(r)[0])[:5]
        counts = {c: 0 for c in CLASSES[1:]}
        for f in self.findings:
            counts[f.cls] += 1
        return {
            "step": self.step, "max_level": self.max_level, "model": s.model,
            "looks_left": max(0, (b.pixels_cap - b.pixels_used) // (s.size * s.size)),
            "pixels_left_frac": round(1 - b.pixels_used / max(b.pixels_cap, 1), 4),
            "usd_left": round(b.usd_cap - b.usd_used, 6), "seconds_left": round(b.seconds_cap - b.seconds_used, 2),
            "p95_ms": {"warp": round(self.stats.p("warp", 95), 1),
                       "segment": round(self.stats.p(f"segment:{s.model}", 95), 1)},
            "top_candidates": [{"id": r.id, "score": round(self._score(r)[0], 6), "entropy": r.entropy,
                                "level": r.next_level, "prior": r.prior} for r in live],
            "counts_by_class": counts,
            "critical_pending": sum(f.state == "pending_review" for f in self.findings),
        }

    def _plan(self):
        """Consult the planner when due. Returns (event or None, cause or None, halt?)."""
        cfg, planner = self.cfg, None
        if "planner_timeout" in cfg.chaos and self.step == PLANNER_TIMEOUT_STEP:
            planner = ChaosPlanner("timeout")
        elif "planner_malformed" in cfg.chaos and self.step == PLANNER_MALFORMED_STEP:
            planner = ChaosPlanner("malformed")
        elif self.planner and self.step and self.step % cfg.planner_every == 0:
            planner = self.planner
        if planner is None:
            return None, None, False
        proposal, status, reason = consult(planner, self.digest(), cfg.planner_timeout_s)
        event = {"type": "planner", "step": self.step, "status": status, "reason": reason, "proposal": proposal}
        if proposal is None:
            return event, "planner-fallback", False
        strategy, p = proposal["strategy"], proposal["params"]
        if strategy == "switch_model":
            self.settings.model = "accurate" if p["model"] == "accurate" else "fast"
        elif strategy == "batch_remaining":
            self.settings.batch = True
        elif strategy in ("change_level", "reverify"):
            r = self.regions[p["region_id"]]
            level = p.get("level", max(0, (r.looks[-1][0] if r.looks else r.next_level) - 1))
            r.done = False
            self.forced.appendleft(("verify" if strategy == "reverify" else "gaze", r, level,
                                    f"planner:{strategy}", proposal["rationale"][:120]))
        elif strategy == "stop":
            return event, "planner:stop", True
        return event, (f"planner:{strategy}" if strategy != "continue" else None), False

    def _log_finding(self, r, cls, seg, crop, to_l0, level):
        mask = seg.mask(cls)
        m = measure(mask, 2**level, linear=cls == "crack")
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        corners = imaging.map_points(to_l0, [[x, y], [x + w, y + h]])
        box = [int(corners[0][0]), int(corners[0][1]), int(corners[1][0] - corners[0][0]),
               int(corners[1][1] - corners[0][1])]
        pad = max(8, max(w, h) // 3)
        y0, x0 = max(0, y - pad), max(0, x - pad)
        view = crop[y0 : y + h + pad, x0 : x + w + pad].copy()
        cs, _ = cv2.findContours(mask[y0 : y + h + pad, x0 : x + w + pad], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, cs, -1, (216, 79, 255), max(1, view.shape[0] // 160))
        thumb = imaging.jpeg_b64(view, 256) if self.cfg.thumbs else None
        conf = seg.confidence[cls]
        # One physical defect, one finding: rust on exposed bars is the bar, a spall's rim is the spall.
        if any(f.cls in DOMINATED_BY.get(cls, ()) and _inside(box, f.box) >= 0.5 for f in self.findings):
            return None, False
        same = [f for f in self.findings if f.cls == cls and (f.region == r.id or imaging.box_iou(f.box, box) > 0.3)]
        if same:
            f = same[0]
            if level <= f.level:
                f.agent_update(confidence=conf, measure=m, box=box, step=self.step, level=level, crop_jpeg_b64=thumb)
            return f, False
        f = Finding(f"f{len(self.findings) + 1}", cls, conf, m, box, self.step, level, r.id, crop_jpeg_b64=thumb)
        self.findings.append(f)
        return f, True

    # ---- the loop ------------------------------------------------------------------------------
    def run(self):
        cfg, clock = self.cfg, self.clock
        t_start = clock()
        self.stats = RuntimeStats(clock=clock)
        H, W = self.image.shape[:2]
        pyr, _ = self._timed("pyrDown", imaging.build_pyramid, self.image)
        self.max_level = min(3, len(pyr) - 1)
        self.budget = Budget(int(cfg.pixel_cap * W * H), cfg.usd_cap, cfg.seconds_cap, usd_per_hour=cfg.usd_per_hour)
        self.settings = Settings(size=cfg.size)
        first = max(0, min(cfg.first_level, int(math.log2(max(W, H) / cfg.size)) if max(W, H) > cfg.size else 0))
        yield {"type": "job", "job": self.id, "width": W, "height": H,
               "levels": [[p.shape[1], p.shape[0]] for p in pyr],
               "budget": {"pixel_cap_frac": cfg.pixel_cap, "pixels_cap": self.budget.pixels_cap,
                          "usd_cap": cfg.usd_cap, "seconds_cap": cfg.seconds_cap}}

        ov = imaging.OVERVIEW_LEVEL
        self.ov_scale = 2**ov
        sal, _ = self._timed("saliency", imaging.saliency, pyr[ov])
        cands = imaging.find_candidates(sal, ov, cfg.size * 2**first, (W, H), cfg.max_candidates)
        self.budget.spend(int(sal.size), 0)
        self.budget.seconds_used = clock() - t_start
        yield {"type": "overview", "heatmap_png_b64": imaging.png_b64((sal * 255).astype(np.uint8)),
               "candidates": [{"id": c.id, "box": list(c.box), "prior": c.prior} for c in cands],
               "pixels_used": self.budget.pixels_used, "pixels_frac": round(self.budget.pixels_used / (W * H), 4)}

        self.regions = {c.id: Region(c.id, c.cx, c.cy, c.prior, first) for c in cands}
        self.seen = [np.zeros(sal.shape, bool) for _ in range(len(pyr))]
        self.forced: deque = deque()
        for i, (action, x, y, level) in enumerate(self.replay):
            r = next((r for r in self.regions.values() if (r.cx, r.cy) == (x, y)), None)
            if r is None:
                r = self.regions.setdefault(f"x{i + 1}", Region(f"x{i + 1}", x, y, 0.0, level))
            self.forced.append((action, r, level, "replay", "replayed from an earlier trace"))
        n_replay = len(self.forced)
        value, cost_ms, halt_cause, halt_why = 0.0, 0.0, None, ""
        skipped: list[dict] = []  # verifies not taken; written into the next trace record
        self.step = 0

        while self.step < cfg.max_steps:
            self.step += 1
            step_t0 = clock()
            replaying = n_replay > 0
            # 1. governor first: it may halt, or change the model, level or gaze size, before a look is chosen
            verdict = check(self.budget, self.stats, self.settings, self.settings.size ** 2, cfg.sla_ms, self.kill())
            if verdict.halt:
                halt_cause, halt_why = verdict.cause, verdict.rationale
                break
            governor_cause = verdict.cause
            # 2. choose the next look
            if self.forced:
                action, r, level, cause, why = self.forced.popleft()
                n_replay -= 1 if replaying else 0
            else:
                live = [r for r in self.regions.values() if not r.done and not r.looks]
                scored = sorted(((self._score(r), r) for r in live), key=lambda t: (-t[0][0], -t[0][2], t[1].cy, t[1].cx))
                scored = [(s, r) for s, r in scored if s[2] > 0]
                if not scored:
                    halt_cause, halt_why = "exhausted", "every candidate has been looked at"
                    break
                (best, _, a), r = scored[0]
                lam = cfg.eps * value / cost_ms if cost_ms else 0.0
                if best < lam:
                    halt_cause = "stop-rule"
                    halt_why = f"best score {best:.2e} < lambda {lam:.2e} (eps x value/cost)"
                    break
                action, level, cause = "gaze", r.next_level, "best-score"
                why = f"argmax score {best:.2e}: prior {r.prior:.2f}, unseen {a:.2f}, t_hat {self._t_hat():.0f} ms"
            if governor_cause:
                why = f"{verdict.rationale}; {why}"
            if not replaying:
                level = min(level + self.settings.level_bias, self.max_level)
            if self.settings.critical_only and action == "verify" and not any(
                    f.critical and f.region == r.id for f in self.findings):
                r.done = True
                skipped.append({"action": "verify", "region": r.id, "level": level, "cause": "usd≥80%",
                                "rationale": f"skipped L{level} verify on {r.id}: verifying critical classes only"})
                self.step -= 1
                continue
            # 3. optional planner
            planner_cause = None
            if not replaying:
                event, planner_cause, stop = self._plan()
                if event:
                    yield event
                if stop:
                    halt_cause, halt_why = planner_cause, "planner proposed stop; validator accepted"
                    break
            # 4. the look itself
            size = self.settings.size
            cx, cy, window = self._window(r.cx, r.cy, level, size)
            (crop, to_l0), warp_ms = self._timed("warp", imaging.gaze, pyr, cx, cy, level, size=size)
            q, _ = self._timed("quality", imaging.quality, crop)
            model = self.settings.model
            seg, seg_ms = self._timed(f"segment:{model}", self.segmenter.predict, crop, model)
            self.stats.record("segment", seg_ms)
            injected = None
            if "p95_spike" in cfg.chaos and self.step == SPIKE_STEP:
                self.stats.record(f"segment:{model}", SPIKE_MS)
                self.stats.record("segment", SPIKE_MS)
                injected = "p95_spike"
            y0, y1, x0, x1 = _rect_on(self.seen[level].shape, window, self.ov_scale)
            self.seen[level][y0:y1, x0:x1] = True
            sal_max = round(float(sal[y0:y1, x0:x1].max()), 4) if y1 > y0 and x1 > x0 else 0.0

            # 5. what the measurements say to do next
            best = seg.best if q["ok"] else None
            area = seg.areas[best] if best else 0
            r.entropy = seg.entropy
            mask = seg.mask(best) if best else np.zeros(seg.labels.shape, np.uint8)
            prev = next((lk for lk in reversed(r.looks) if lk[1].any()), None)
            iou = look_iou(prev[1], prev[2], mask, to_l0) if (prev is not None and best) else None
            prev_level = prev[0] if prev is not None else None
            r.looks.append((level, mask, to_l0))
            new_events, follow, skip = [], None, None
            batch = self.settings.batch
            if not q["ok"]:
                r.done = True
            elif level > 0 and not batch and seg.entropy > cfg.tau:
                follow = ("escalate", r, level - 1, "entropy>tau", f"entropy {seg.entropy:.2f} > {cfg.tau} at L{level}")
            elif level > 0 and not batch and best and area < cfg.a_min:
                follow = ("escalate", r, level - 1, "area<A_min", f"{best} area {area} px < {cfg.a_min} at L{level}")
            elif best and (present := sorted((c for c, a in seg.areas.items() if a >= cfg.a_min),
                                             key=lambda c: -SEVERITY[c])):
                # Every class with a confident mask becomes evidence, most severe first.
                r.cls = present[0]
                critical = None
                # Realised value of the look, in the same units as a region's expected value
                # (one class's severity), so the stop rule compares like with like.
                value += max(SEVERITY[c] * seg.confidence[c] for c in present)
                for c in present:
                    f, is_new = self._log_finding(r, c, seg, crop, to_l0, level)
                    if f is None:
                        continue
                    new_events.append({"type": "finding", "finding": f.to_dict()})
                    if f.critical:
                        critical = critical or f
                        if is_new:
                            new_events.append({"type": "review_request", "finding_id": f.id,
                                               "reason": f"{c} is critical"})
                disagree = f"iou {iou:.2f} < 0.5 between L{prev_level} and L{level}" if iou is not None else ""
                if iou is not None and iou < 0.5 and critical:
                    f = critical
                    new_events.append({"type": "review_request", "finding_id": f.id, "reason": disagree})
                    r.done = True
                elif iou is not None and iou < 0.5 and not r.extra_used and level > 0:
                    r.extra_used = True
                    follow = ("verify", r, level - 1, "iou<0.5", disagree)
                elif iou is None and level > 0 and not batch and not r.verified:
                    # Critical masks are always confirmed (or sent to a human); others compete for the budget.
                    if not critical and (not replaying or n_replay == 0):
                        skip = self._verify_crowded_out(r, seg, level, value, cost_ms + (clock() - step_t0) * 1000)
                    if skip:
                        r.done = True
                    else:
                        follow = ("verify", r, level - 1, "verify-new-mask",
                                  f"new {best} mask at L{level}; confirming at L{level - 1}")
                else:
                    r.verified = iou is not None and iou >= 0.5
                    r.done = True
            else:
                r.done = True
            # While replaying, the trace already holds the follow-ups; only the last replayed look may queue one.
            if follow and (not replaying or n_replay == 0):
                r.cx, r.cy = _focus(seg, to_l0, (cx, cy))
                self.forced.appendleft(follow)
            elif follow:
                r.done = True

            # 6. account and log
            step_s = clock() - step_t0
            cost_ms += step_s * 1000
            usd_before = self.budget.usd_used
            self.budget.pixels_used += size * size
            self.budget.seconds_used = clock() - t_start
            self.gaze_path.append([cx, cy, level])
            rec = {
                "type": "step", "job": self.id, "step": self.step, "t": now_iso(), "region": r.id,
                "decision": {"action": action, "args": {"x": cx, "y": cy, "level": level, "yaw": 0, "pitch": 0,
                                                        "size": size}},
                "window": window,
                "observation": {"entropy": seg.entropy, "saliency_max": sal_max, "mask_area_px": int(sum(seg.areas.values())),
                                "sharpness": q["sharpness"], "exposure": q["exposure"],
                                "classes": {c: a for c, a in seg.areas.items() if a}},
                "quality": {"ok": q["ok"], "reason": q["reason"]},
                "runtime": self.stats.snapshot(self.budget.usd_used),
                "rationale": why if q["ok"] else f"{why}; quality gate: {q['reason']}",
                "cause": governor_cause or planner_cause or cause,
                "planner": "llm" if planner_cause and planner_cause.startswith("planner:") else "rule",
                "model": model, "latency_ms": round(warp_ms + seg_ms, 1),
                "cost_usd": round(self.budget.usd_used - usd_before, 8),
                "pixels_used": self.budget.pixels_used,
                "pixels_frac": round(self.budget.pixels_used / (W * H), 4),
                "usd_used": round(self.budget.usd_used, 6), "seconds_used": round(self.budget.seconds_used, 3),
                "injected": injected,
            }
            if iou is not None:
                rec["verify"] = {"iou": iou, "against_level": prev_level}
            if cfg.thumbs:
                rec["crop_jpeg_b64"] = imaging.jpeg_b64(crop, 160, 70)
            _note_skipped(rec, skipped)
            if skip:
                skipped.append(skip)
            self.trace.append(rec)
            yield rec
            yield from new_events
        else:
            halt_cause, halt_why = "cap-hit", f"max_steps {cfg.max_steps} reached"

        self.budget.seconds_used = clock() - t_start
        halted = halt_cause in ("cap-hit", "kill-switch")
        rec = {"type": "step", "job": self.id, "step": self.step, "t": now_iso(), "region": None,
               "decision": {"action": "halt", "args": {}}, "rationale": halt_why, "cause": halt_cause,
               "planner": "rule", "model": self.settings.model, "pixels_used": self.budget.pixels_used,
               "pixels_frac": round(self.budget.pixels_used / (W * H), 4),
               "usd_used": round(self.budget.usd_used, 6), "seconds_used": round(self.budget.seconds_used, 3),
               "runtime": self.stats.snapshot(self.budget.usd_used)}
        _note_skipped(rec, skipped)
        self.trace.append(rec)
        yield rec
        pending = sum(f.state == "pending_review" for f in self.findings)
        status = "halted" if halted else ("awaiting_review" if pending else "completed")
        seen_any = np.logical_or.reduce(self.seen)
        coverage = float((sal * seen_any).sum() / max(sal.sum(), 1e-6))
        self.work_order = work_order(self.id, self.findings, self.gaze_path)
        yield {"type": "done", "status": status, "cause": halt_cause,
               "summary": {"steps": len(self.gaze_path), "pixels_used": self.budget.pixels_used,
                           "pixels_frac": round(self.budget.pixels_used / (W * H), 4),
                           "usd": round(self.budget.usd_used, 6), "seconds": round(self.budget.seconds_used, 3),
                           "findings": len(self.findings), "critical_pending": pending},
               "coverage": round(coverage, 4),
               "unvisited": [r.id for r in self.regions.values() if not r.looks],
               "work_order": self.work_order}


def _note_skipped(rec: dict, skipped: list[dict]) -> None:
    """Verifies not taken go into the next trace record, each with its cause, ahead of that record's rationale."""
    if skipped:
        rec["skipped"] = skipped[:]
        rec["rationale"] = "; ".join([*(s["rationale"] for s in skipped), rec["rationale"]])
        skipped.clear()


def _inside(a, b) -> float:
    """Fraction of box a's area that lies inside box b."""
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / max(1, a[2] * a[3])


def look_iou(prev_mask: np.ndarray, prev_to_l0: np.ndarray, cur_mask: np.ndarray, cur_to_l0: np.ndarray) -> float:
    """Agreement of two looks' masks, compared in the finer (current) look's frame with a 2-px tolerance."""
    size = cur_mask.shape[::-1]
    m = np.linalg.inv(cur_to_l0) @ prev_to_l0
    warped = cv2.warpPerspective(prev_mask, m, size, flags=cv2.INTER_NEAREST)
    footprint = cv2.warpPerspective(np.ones_like(prev_mask), m, size, flags=cv2.INTER_NEAREST) > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a = cv2.dilate(cur_mask, k).astype(bool) & footprint
    b = cv2.dilate(warped, k).astype(bool)
    union = np.count_nonzero(a | b)
    return round(np.count_nonzero(a & b) / union, 3) if union else 1.0


def _focus(seg, to_l0, fallback: tuple[int, int]) -> tuple[int, int]:
    """Where the next, finer look should centre: the defect pixels of this look, else the
    pixels the segmenter is unsure about, else the same centre."""
    ys, xs = np.nonzero(seg.labels)
    if xs.size < MIN_BLOB_PX:
        ys, xs = np.nonzero(seg.probs[0] < DEFECT_PX_BG_MAX)
    if xs.size < MIN_BLOB_PX:
        return fallback
    (fx, fy), = imaging.map_points(to_l0, [[float(xs.mean()), float(ys.mean())]])
    return int(fx), int(fy)


def validate_replay(replay: list, shape) -> list[tuple[str, int, int, int]]:
    """Earlier decisions to re-execute. Client-supplied, so every field is checked."""
    if not isinstance(replay, list) or len(replay) > MAX_REPLAY:
        raise ValueError(f"replay must be a list of at most {MAX_REPLAY} decisions")
    H, W = shape[:2]
    out = []
    for item in replay:
        if not (isinstance(item, (list, tuple)) and len(item) == 4 and item[0] in REPLAY_ACTIONS
                and all(isinstance(v, int) and not isinstance(v, bool) for v in item[1:])):
            raise ValueError("each replay decision is [action, x, y, level] with integer x, y, level")
        action, x, y, level = item
        if not (0 <= x < W and 0 <= y < H and 0 <= level <= 3):
            raise ValueError("replay decision outside the image or level range")
        out.append((action, x, y, level))
    return out


def gaze_path_from_trace(lines) -> list[list[int]]:
    """Replay the gaze path from trace JSON alone (FR-15)."""
    return [[r["decision"]["args"]["x"], r["decision"]["args"]["y"], r["decision"]["args"]["level"]]
            for r in lines if r.get("type") == "step" and r["decision"]["action"] != "halt"]
