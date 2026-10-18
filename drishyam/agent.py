"""The inspection loop: overview, then look / escalate / verify until the stop rule or a cap.

`Job.run()` yields the NDJSON events in docs/api-contract.md. Every step's
decision is a logged function of an OpenCV output (entropy, mask area, IoU,
quality) or a runtime statistic (p95, $, pixels), and carries a `cause`.
"""

from __future__ import annotations

import math
import random
import secrets
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import imaging
from .findings import IOU_AGREE, SEVERITY, Finding, now_iso, work_order
from .governor import MIN_GAZE, USD_PER_HOUR, Budget, RuntimeStats, Settings, check
from .measure import measure
from .planner import TIMEOUT_S, ChaosPlanner, consult
from .segment import (
    CLASSES,
    DEFECT_PX_BG_MAX,
    MIN_BLOB_PX,
    PORE_CAUSE,
    _entropy,
    default_segmenter,
    elongated,
    rejected_lines,
    segment_view,
)

SPIKE_STEP = 4  # chaos p95_spike lands here; the governor reacts on the next step
SPIKE_MS = 900.0
PLANNER_TIMEOUT_STEP = 3
PLANNER_MALFORMED_STEP = 6
REPLAY_ACTIONS = ("gaze", "escalate", "verify")
DOMINATED_BY = {"corrosion": ("exposed_rebar",), "crack": ("spalling", "exposed_rebar"),
                "efflorescence": ("spalling", "exposed_rebar")}
MAX_REPLAY = 200
V0 = 2.0  # JobConfig.v0 default: the largest v0 tried (0-4) that lost no recall on the dev splits (synth_dev, codebrim_dev100)


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
    # Ablation switches (FR-21); the defaults are the agent as published.
    use_uncertainty: bool = True  # the (1 + entropy) term of a region's score
    seed_from_looks: bool = True  # every look sets the entropy and class of the unvisited candidates it shows
    # The value of a look that finds nothing, in severity units: the stop threshold never falls below eps x v0 per
    # look, so a photo with nothing to find can stop on the rule (0: no floor, the rule as first published).
    v0: float = V0
    survey: bool = True  # one coarse look that covers every candidate before zooming
    verify: bool = True  # verify looks: a new mask, or a finer look that disagrees with the coarser one
    order: str = "score"  # "random": a seeded random pick among unseen candidates instead of the argmax
    seed: int = 0
    governor: bool = True  # False: kill switch and caps only; no model, level, gaze-size or critical-only override
    # A fixed predicted cost per look, {"warp": ms, "segment": ms, "segment:<model>": ms}, instead of live p95s:
    # scores and the stop rule then do not move with machine load. The caps still run on the real clock.
    t_hat_ms: dict | None = None


@dataclass
class Look:
    level: int
    labels: np.ndarray  # class index per crop pixel, as segment_view left it (nothing past the image, no straight line)
    to_l0: np.ndarray
    cls: str | None  # the look's best class
    finding: str | None = None  # the finding that class's evidence went to (new, update or absorbed)

    def mask(self, cls: str) -> np.ndarray:
        return (self.labels == CLASSES.index(cls)).astype(np.uint8)


@dataclass
class Region:
    id: str
    cx: int
    cy: int
    prior: float
    next_level: int
    entropy: float = 0.0  # a look's entropy here: its own look's, or a seed's (`Job._seed`) while it has none
    cls: str | None = None  # likewise the most severe class seen here
    box: tuple | None = None  # L0 window of its first look (candidates only): what a seed reads
    shown_level: int | None = None  # level of the finest look that has shown it (and seeded it); None: none has
    looks: list = field(default_factory=list)  # Look per look at this region
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
        self.replay = validate_replay(replay or [], image.shape, self.cfg.size)
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
        if t := self.cfg.t_hat_ms:
            return t["warp"] + t.get(f"segment:{self.settings.model}", t["segment"])
        seg_prior = self.stats.median("segment") or 30.0
        return self.stats.p("warp", 95, 5.0) + self.stats.p(f"segment:{self.settings.model}", 95, seg_prior)

    def _score(self, r: Region):
        """(score, expected value, unseen fraction, severity, uncertainty): ev = prior x severity x (1 + u) x unseen."""
        level = min(r.next_level + self.settings.level_bias, self.max_level)
        _, _, win = self._window(r.cx, r.cy, level, self.settings.size)
        a = self._unseen(win, level)
        u = r.entropy if self.cfg.use_uncertainty else 0.0
        s = SEVERITY.get(r.cls, 1.0)
        ev = r.prior * s * (1 + u) * a
        return ev / max(self._t_hat(), 1e-3), ev, a, s, u

    def _seed(self, seg, to_l0, valid, level) -> list[dict]:
        """After a look the quality gate passed: every unvisited candidate whose centre lies in the look's valid
        footprint has been shown (the stop floor needs that), and takes its uncertainty and class from what the look
        saw inside that candidate's box: the binary damage-vs-background entropy over that sub-window
        (`segment._entropy`, as a look reports it) and the most severe class whose area there would clear A_min at
        the candidate's own first-look level (None if none). A finer look overwrites a coarser seed, never the
        reverse; a seed is not a look (the candidate stays unvisited). seed_from_looks=False: shown, not seeded."""
        inv, size, out = np.linalg.inv(to_l0), valid.shape[0], []
        for r in self.regions.values():
            if r.looks or r.done or r.box is None or (r.shown_level is not None and level > r.shown_level):
                continue
            (u, v), = imaging.map_points(inv, [[r.cx, r.cy]])
            if not (0 <= u < size and 0 <= v < size and valid[int(v), int(u)]):
                continue
            r.shown_level = level
            if not self.cfg.seed_from_looks:
                continue
            x, y, w, h = r.box
            c = imaging.map_points(inv, [[x, y], [x + w, y + h]])
            (x0, y0), (x1, y1) = np.clip(np.floor(c.min(0)), 0, size).astype(int), np.clip(np.ceil(c.max(0)), 0, size).astype(int)
            labels = seg.labels[y0:y1, x0:x1]
            min_area = max(MIN_BLOB_PX, self.cfg.a_min * 4.0 ** (r.next_level - level))
            cls = max((k for k in CLASSES[1:] if np.count_nonzero(labels == CLASSES.index(k)) >= min_area),
                      key=SEVERITY.get, default=None)
            r.entropy, r.cls = _entropy(seg.probs[:, y0:y1, x0:x1]), cls
            out.append({"region": r.id, "entropy": r.entropy, "cls": cls, "from_level": level})
        return out

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
        """Consult the planner when due. Returns (planner event or None, halt?)."""
        cfg, planner = self.cfg, None
        if "planner_timeout" in cfg.chaos and self.step == PLANNER_TIMEOUT_STEP:
            planner = ChaosPlanner("timeout")
        elif "planner_malformed" in cfg.chaos and self.step == PLANNER_MALFORMED_STEP:
            planner = ChaosPlanner("malformed")
        elif self.planner and self.step and self.step % cfg.planner_every == 0:
            planner = self.planner
        if planner is None:
            return None, False
        proposal, status, reason = consult(planner, self.digest(), cfg.planner_timeout_s)
        event = {"type": "planner", "step": self.step, "status": status, "reason": reason, "proposal": proposal}
        if proposal is None:
            return event, False
        strategy, p = proposal["strategy"], proposal["params"]
        if strategy == "switch_model" and p["model"] == "accurate" and "p95>SLA" in self.settings.history:
            # The governor downgraded the model for latency; a planner cannot quietly undo that.
            event.update(status="rejected", reason="governor downgraded the model (p95>SLA)", proposal=None)
            return event, False
        if strategy == "switch_model":
            self.settings.model = "accurate" if p["model"] == "accurate" else "fast"
        elif strategy == "batch_remaining":
            self.settings.batch = True
        elif strategy in ("change_level", "reverify"):
            r = self.regions[p["region_id"]]
            level = p.get("level", max(0, (r.looks[-1].level if r.looks else r.next_level) - 1))
            r.done = False
            self.forced.appendleft(("verify" if strategy == "reverify" else "gaze", r, level,
                                    f"planner:{strategy}", proposal["rationale"][:120]))
        elif strategy == "stop":
            return event, True
        # change_level / reverify queue their own look, which carries planner:<strategy> as its cause.
        return event, False

    def _present(self, seg, to_l0, level, rejected: list) -> list[str]:
        """The classes a look drafts, most severe first: every class with a mask of at least A_min, except a crack
        whose measure is a blob, not a line (`elongated`): that one is appended to `rejected` instead."""
        present = sorted((c for c, a in seg.areas.items() if a >= self.cfg.a_min), key=lambda c: -SEVERITY[c])
        if "crack" in present and not elongated(m := measure(mask := seg.mask("crack"), 2**level)):
            present.remove("crack")
            rejected.append({"cls": "crack", "cause": PORE_CAUSE, "length_px": m["length_px"], "width_px": m["width_px"],
                             "box": imaging.l0_box(cv2.boundingRect(cv2.findNonZero(mask)), to_l0, self.image.shape[1::-1])})
        return present

    def _log_finding(self, r, cls, seg, crop, to_l0, level, verify_iou=None):
        """(finding, "new" | "update"), or (the finding that absorbed it, "absorbed")."""
        mask = seg.mask(cls)
        m = measure(mask, 2**level, linear=cls == "crack")
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        box = imaging.l0_box((x, y, w, h), to_l0, self.image.shape[1::-1])
        pad = max(8, max(w, h) // 3)
        y0, x0 = max(0, y - pad), max(0, x - pad)
        view = crop[y0 : y + h + pad, x0 : x + w + pad].copy()
        cs, _ = cv2.findContours(mask[y0 : y + h + pad, x0 : x + w + pad], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, cs, -1, (87, 200, 255), max(2, view.shape[0] // 64))  # BGR callout amber #FFC857, ~2 px at card size
        thumb = imaging.jpeg_b64(view, 256) if self.cfg.thumbs else None
        conf = seg.confidence[cls]
        # One physical defect, one finding: rust on exposed bars is the bar, a spall's rim is the spall.
        if dom := next((f for f in self.findings if f.cls in DOMINATED_BY.get(cls, ()) and _inside(box, f.box) >= 0.5),
                       None):
            if verify_iou is not None:  # the check was of this defect, seen as the class it absorbed
                dom.set_verify(verify_iou)
            return dom, "absorbed"
        same = [f for f in self.findings if f.cls == cls and (f.region == r.id or imaging.box_iou(f.box, box) > 0.3)]
        if same:
            f = same[0]
            f.agent_update(confidence=conf, measure=m, box=box, step=self.step, level=level, crop_jpeg_b64=thumb,
                           verify_iou=verify_iou)
            return f, "update"
        f = Finding(f"f{len(self.findings) + 1}", cls, conf, m, box, self.step, level, r.id, crop_jpeg_b64=thumb,
                    verify_iou=verify_iou)
        self.findings.append(f)
        return f, "new"

    def _settle(self, r, chk, level, best, target, now_f, critical, drafted, new_events):
        """After a look that queued no escalation: settle its comparison with the coarser look (`chk`, from
        `compare`), queue the next look at this region or mark it done. Returns the follow-up look or None.
        `target`: the finding the compared class lives in; `now_f`: this look's finding of its best class;
        `critical`: this look's first critical finding; `drafted`: whether this look drafted anything."""
        cfg = self.cfg
        if chk is None:
            if drafted and level > 0 and not self.settings.batch and not r.verified and cfg.verify:
                return ("verify", r, level - 1, "verify-new-mask",
                        f"new {best} mask at L{level}; confirming at L{level - 1}")
            r.done = True
            return None
        k, a, iou = chk["class"], chk["against_level"], chk["iou"]
        if chk["kind"] == "agree":
            r.verified = r.done = True
            return None
        if chk["kind"] == "class-change":
            # The finer look sees the same place as another class: it keeps that class, the coarser one is recorded
            # (the step's verify.class; the coarser finding is unconfirmed). A human decides if either is critical.
            r.done = True
            if crit := next((f for f in (target, now_f) if f is not None and f.critical), None):
                new_events.append(review(crit.id, f"class change: {k} at L{a}, {chk['class_now']} at L{level} (same "
                                         f"place, iou {chk['iou_any']:g})", level, [k, chk["class_now"]],
                                         chk["iou_any"], a, kind="class-change"))
            return None
        disagree = f"iou {iou:g} < {IOU_AGREE} between L{a} and L{level}"
        if crit := (target if target is not None and target.critical else critical):
            new_events.append(review(crit.id, disagree, level, [k, k], iou, a, kind="disagree"))
            r.done = True
        elif not r.extra_used and level > 0 and cfg.verify:
            r.extra_used = True
            return ("verify", r, level - 1, "iou<0.5", disagree)
        else:
            r.done = True
            if target is not None:  # no further look will settle it
                new_events.append(review(target.id, f"unconfirmed: {disagree}", level, [k, k], iou, a,
                                         kind="unconfirmed"))
        return None

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

        self.regions = {c.id: Region(c.id, c.cx, c.cy, c.prior, first, box=c.box) for c in cands}
        self.seen = [np.zeros(sal.shape, bool) for _ in range(len(pyr))]
        self.forced: deque = deque()
        replay_meta: deque = deque()  # (x, y, size) of each replayed look, applied when it runs
        prev_r = None
        for i, (action, x, y, level, rsize) in enumerate(self.replay):
            # escalate / verify always ran on the region of the look just before them (rule follow-ups)
            r = prev_r if action != "gaze" and prev_r is not None else None
            r = r or next((r for r in self.regions.values() if (r.cx, r.cy) == (x, y)), None)
            if r is None:
                r = self.regions.setdefault(f"x{i + 1}", Region(f"x{i + 1}", x, y, 0.0, level))
            self.forced.append((action, r, level, "replay", "replayed from an earlier trace"))
            replay_meta.append((x, y, rsize))
            prev_r = r
        n_replay = len(self.forced)
        looks = (self.budget.pixels_cap - self.budget.pixels_used) // cfg.size**2
        if not self.replay and cfg.survey and (s := survey_look(cands, (W, H), cfg.size, first, self.max_level,
                                                               looks)):
            r = self.regions["s0"] = Region("s0", s[0], s[1], max(c.prior for c in cands), s[2])
            self.forced.append(("gaze", r, s[2], "survey", (f"one L{s[2]} look covers all {len(cands)} candidates "
                                                           f"before zooming; {looks} looks affordable")))
        value, cost_ms, halt_cause, halt_why = 0.0, 0.0, None, ""
        govs: list = []  # governor overrides not yet written to the trace (PRD rule 5: every override writes a cause)
        rng = random.Random(cfg.seed)  # only the order="random" ablation draws from it
        self.not_assessed = imaging.unseen_areas(pyr[ov], self.ov_scale, (W, H))  # too large to spend looks on
        asked: set = set()  # (finding id, class) pairs already sent to review as ambiguous
        self.followups = 0  # events that follow the step record just yielded (its findings and review requests)
        self.step = 0

        while self.step < cfg.max_steps:
            self.step += 1
            step_t0 = clock()
            replaying = n_replay > 0
            # 1. governor first: it may halt, or change the model, level or gaze size, before a look is chosen
            verdict = check(self.budget, self.stats, self.settings, self.settings.size ** 2, cfg.sla_ms, self.kill(),
                            overrides=cfg.governor)
            if verdict.halt:
                halt_cause, halt_why = verdict.cause, verdict.rationale
                break
            if verdict.cause:
                govs.append(verdict)
            # 2. choose the next look
            if self.forced:
                action, r, level, cause, why = self.forced.popleft()
                if replaying:
                    n_replay -= 1
                    r.cx, r.cy, rsize = replay_meta.popleft()  # the recorded centre of this very look
                    if rsize is not None:
                        self.settings.size = rsize  # the gaze size the governor had set when this look first ran
            else:
                live = [r for r in self.regions.values() if not r.done and not r.looks]
                scored = sorted(((self._score(r), r) for r in live), key=lambda t: (-t[0][0], -t[0][2], t[1].cy, t[1].cx))
                scored = [(s, r) for s, r in scored if s[2] > 0]
                if not scored:
                    halt_cause, halt_why = "exhausted", "every candidate has been looked at"
                    break
                (best, _, a, sev, u), r = scored[0]
                # lambda: eps x the value found per ms so far, but never below eps x v0 per look (the floor), so a
                # photo with nothing to find can still stop on the rule rather than spend its whole budget. The floor
                # holds only once a look has shown the best candidate (a seed): while no look has, its value is
                # unknown, not small, so the job never stops on a saliency prior alone (nor before it has seen anything).
                lam_found = cfg.eps * value / cost_ms if cost_ms else 0.0
                lam_floor = cfg.eps * cfg.v0 / max(self._t_hat(), 1e-3) if r.shown_level is not None else 0.0
                lam = max(lam_found, lam_floor)
                if best < lam:
                    halt_cause = "stop-rule"
                    term = "floor: eps x v0 / t_hat" if lam_floor >= lam_found else "eps x value/cost"
                    halt_why = f"best score {best:.2e} < lambda {lam:.2e} ({term})"
                    break
                pick, cause = "argmax", "best-score"
                if cfg.order == "random":  # ablation: any unseen candidate; the stop rule still reads the best score
                    (best, _, a, sev, u), r = rng.choice(scored)
                    pick, cause = f"random pick (seed {cfg.seed}),", "random-order"
                action, level = "gaze", r.next_level
                seen_as = ("not seen yet" if r.shown_level is None else "seeding off" if not cfg.seed_from_looks
                           else f"{r.cls or 'no class'} seen at L{r.shown_level}")
                why = (f"{pick} score {best:.2e}: prior {r.prior:.2f}, severity {sev:g} ({seen_as}), uncertainty {u:.2f}, "
                       f"unseen {a:.2f}, t_hat {self._t_hat():.0f} ms; lambda {lam:.2e}")
            if not replaying:
                level = min(level + self.settings.level_bias, self.max_level)
            if self.settings.critical_only and action == "verify" and not any(
                    f.critical and f.region == r.id for f in self.findings):
                r.done = True
                self.step -= 1
                continue
            # 3. optional planner: its outcome goes to planner_status; the look keeps its own cause
            planner_status = None
            if not replaying:
                event, stop = self._plan()
                if event:
                    yield event
                    planner_status = {"status": event["status"], "reason": event["reason"]}
                    if event["proposal"]:
                        planner_status["strategy"] = event["proposal"]["strategy"]
                if stop:
                    halt_cause, halt_why = "planner:stop", "planner proposed stop; validator accepted"
                    break
            # 4. the look itself
            size = self.settings.size
            cx, cy, window = self._window(r.cx, r.cy, level, size)
            (crop, to_l0), warp_ms = self._timed("warp", imaging.gaze, pyr, cx, cy, level, size=size)
            q, _ = self._timed("quality", imaging.quality, crop)
            model = self.settings.model
            # Background wherever the view runs past the image: gaze fills that by reflection (a mirrored defect).
            valid = imaging.footprint(to_l0, (W, H), size)
            seg, seg_ms = self._timed(f"segment:{model}", segment_view, self.segmenter, crop, model, valid, 2**level)
            self.stats.record("segment", seg_ms)
            # Crack components segment_view dropped as straight lines (formwork, joints): logged, never drafted.
            rejected = rejected_lines(seg, to_l0, (W, H))
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
            # The finer look against the last coarser look that saw a class here, class by class (PRD rule 3).
            prev = next((lk for lk in reversed(r.looks) if lk.cls), None) if q["ok"] else None
            chk = compare(prev, seg, to_l0) if prev is not None else None
            look = Look(level, seg.labels, to_l0, best)
            r.looks.append(look)
            seeded = self._seed(seg, to_l0, valid, level) if q["ok"] else []
            new_events, follow, drafted, critical = [], None, {}, None
            batch = self.settings.batch
            if not q["ok"]:
                r.done = True
                self.not_assessed.append({"step": self.step, "box": imaging.l0_box((0, 0, size, size), to_l0, (W, H)),
                                          "reason": q["reason"], "action": "re-shoot"})
            elif level > 0 and not batch and seg.entropy > cfg.tau:
                follow = ("escalate", r, level - 1, "entropy>tau", f"entropy {seg.entropy:.2f} > {cfg.tau} at L{level}")
            elif level > 0 and not batch and best and area < cfg.a_min:
                follow = ("escalate", r, level - 1, "area<A_min", f"{best} area {area} px < {cfg.a_min} at L{level}")
            else:
                if best and (present := self._present(seg, to_l0, level, rejected)):
                    # Every class with a confident mask becomes evidence, most severe first.
                    r.cls = present[0]
                    # Realised value of the look, in the same units as a region's expected value
                    # (one class's severity), so the stop rule compares like with like.
                    value += max(SEVERITY[c] * seg.confidence[c] for c in present)
                    for c in present:
                        # The check compared class chk["class"] across the two looks: evidence about that class's finding.
                        f, kind = self._log_finding(r, c, seg, crop, to_l0, level,
                                                    chk["iou"] if chk and c == chk["class"] else None)
                        drafted[c] = f
                        if c == best:
                            look.finding = f.id
                        if kind == "absorbed":
                            if chk and c == chk["class"]:  # the absorbing finding now carries this look's check
                                new_events.append({"type": "finding", "finding": f.to_dict()})
                            # Review on ambiguity: two classes clear A_min on one spot and the agent kept one of them.
                            if (f.id, c) not in asked:
                                asked.add((f.id, c))
                                new_events.append(review(f.id, (
                                    f"ambiguous: {c} {seg.areas[c]} px (>= A_min {cfg.a_min}) is mostly inside {f.cls} "
                                    f"{f.id}; recorded as {f.cls} only"), level, [f.cls, c], kind="ambiguous"))
                            continue
                        new_events.append({"type": "finding", "finding": f.to_dict()})
                        if f.critical:
                            critical = critical or f
                            if kind == "new":
                                new_events.append(review(f.id, f"{c} is critical", level, [c], kind="critical"))
                # The finding the check is about: the one this look drafted the compared class into, else the one the
                # coarser look drafted it into (the class is gone or too small here: it still gets this verdict).
                target = None
                if chk:
                    target = drafted.get(chk["class"]) or next((f for f in self.findings if f.id == prev.finding), None)
                    if target is not None and chk["class"] not in drafted:
                        target.set_verify(chk["iou"])
                        new_events.append({"type": "finding", "finding": target.to_dict()})
                follow = self._settle(r, chk, level, best, target, drafted.get(best), critical, bool(drafted), new_events)

            # While replaying, the trace already holds the follow-ups; only the last replayed look may queue one.
            if follow and (not replaying or n_replay == 0):
                r.cx, r.cy = _focus(seg, to_l0, (cx, cy))
                self.forced.appendleft(follow)
            elif follow:
                r.done = True

            # 6. account and log
            step_s = clock() - step_t0
            cost_ms += self._t_hat() if cfg.t_hat_ms else step_s * 1000
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
                "cause": cause,
                "planner": "llm" if cause.startswith("planner:") or (planner_status or {}).get("strategy") in (
                    "switch_model", "batch_remaining") else "rule",
                "model": model, "latency_ms": round(warp_ms + seg_ms, 1),
                "cost_usd": round(self.budget.usd_used - usd_before, 8),
                "pixels_used": self.budget.pixels_used,
                "pixels_frac": round(self.budget.pixels_used / (W * H), 4),
                "usd_used": round(self.budget.usd_used, 6), "seconds_used": round(self.budget.seconds_used, 3),
                "injected": injected,
            }
            if chk:
                rec["verify"] = {k: chk[k] for k in ("iou", "class", "against_level", "class_now", "iou_any")}
            if seeded:
                rec["seeded"] = seeded
            if rejected:
                rec["rejected"] = rejected
            if planner_status:
                rec["planner_status"] = planner_status
            if cfg.thumbs:
                rec["crop_jpeg_b64"] = imaging.jpeg_b64(crop, 160, 70)
            if govs:
                rec["governor"] = {"cause": govs[0].cause, "causes": [g.cause for g in govs],
                                   "rationale": "; ".join(g.rationale for g in govs)}
            self.trace.append(rec)
            govs = []
            self.followups = len(new_events)
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
        if govs:  # the governor changed settings on the step that then halted: the override still gets its cause
            rec["governor"] = {"cause": govs[0].cause, "causes": [g.cause for g in govs],
                               "rationale": "; ".join(g.rationale for g in govs)}
        self.trace.append(rec)
        yield rec
        pending = sum(f.state == "pending_review" for f in self.findings)
        status = "halted" if halted else ("awaiting_review" if pending else "completed")
        seen_any = np.logical_or.reduce(self.seen)
        coverage = float((sal * seen_any).sum() / max(sal.sum(), 1e-6))
        self.work_order = work_order(self.id, self.findings, self.gaze_path, self.not_assessed)
        yield {"type": "done", "status": status, "cause": halt_cause,
               "summary": {"steps": len(self.gaze_path), "pixels_used": self.budget.pixels_used,
                           "pixels_frac": round(self.budget.pixels_used / (W * H), 4),
                           "usd": round(self.budget.usd_used, 6), "seconds": round(self.budget.seconds_used, 3),
                           "findings": len(self.findings), "critical_pending": pending},
               "coverage": round(coverage, 4),
               "unvisited": [r.id for r in self.regions.values() if not r.looks],
               "work_order": self.work_order}


def survey_look(cands, l0_size, size, first, max_level, looks) -> tuple[int, int, int] | None:
    """Binoculars before the zoom: (x, y, level) of one look whose window covers every candidate window.
    Only when it is coarser than a first look and the budget leaves a second look to act on what it shows
    (a coarse look whose uncertainty can never be resolved is a wasted look); otherwise None."""
    if looks < 2 or not cands:
        return None
    W, H = l0_size
    x0, y0 = max(0, min(c.box[0] for c in cands)), max(0, min(c.box[1] for c in cands))
    x1, y1 = min(W, max(c.box[0] + c.box[2] for c in cands)), min(H, max(c.box[1] + c.box[3] for c in cands))
    level = min(max_level, math.ceil(math.log2(max(x1 - x0, y1 - y0, 1) / size)))
    return ((x0 + x1) // 2, (y0 + y1) // 2, level) if level > first else None


REVIEW_KINDS = ("critical", "ambiguous", "disagree", "class-change", "unconfirmed")


def review(finding_id: str, reason: str, level: int, classes: list, iou: float | None = None,
           level_a: int | None = None, kind: str = "critical") -> dict:
    """A review_request event. `reason` is the sentence traces and MCP clients read; the rest says the same in
    fields: `kind` (REVIEW_KINDS), level_b the look that asked, level_a the coarser look it was compared with (iou:
    their same-class mask IoU, or for a class change the IoU of the two looks' best masks), and `classes` is
    [class at level_a, class at level_b] for a comparison, [kept, absorbed] for an ambiguity, [class] for a new
    critical finding."""
    assert kind in REVIEW_KINDS, kind
    return {"type": "review_request", "finding_id": finding_id, "kind": kind, "reason": reason, "iou": iou,
            "level_a": level_a, "level_b": level, "classes": classes}


def compare(prev: Look, seg, to_l0: np.ndarray) -> dict | None:
    """A finer look against the coarser look `prev`, class by class: `iou` compares prev's class mask with this
    look's mask of the SAME class (gone here: 0), `iou_any` the two looks' best masks whatever their classes. `kind`:
    agree (iou >= IOU_AGREE); class-change (the same place, seen as another class: iou_any >= IOU_AGREE); else
    disagree. None when the two cannot be compared (prev's mask lies outside this view, which shows none of it)."""
    k, now = prev.cls, seg.best
    iou = look_iou(prev.mask(k), prev.to_l0, seg.mask(k), to_l0)
    if iou is None:
        return None
    now_mask = seg.mask(now) if now else np.zeros(seg.labels.shape, np.uint8)  # this look may see no defect at all
    iou_any = look_iou(prev.mask(k), prev.to_l0, now_mask, to_l0) if now != k else iou
    kind = ("agree" if iou >= IOU_AGREE else "class-change" if now and iou_any is not None and iou_any >= IOU_AGREE
            else "disagree")
    return {"iou": iou, "class": k, "against_level": prev.level, "class_now": now, "iou_any": iou_any, "kind": kind}


def _inside(a, b) -> float:
    """Fraction of box a's area that lies inside box b."""
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / max(1, a[2] * a[3])


def look_iou(prev_mask: np.ndarray, prev_to_l0: np.ndarray, cur_mask: np.ndarray, cur_to_l0: np.ndarray) -> float | None:
    """Agreement of two looks' masks, compared in the finer (current) look's frame with a 2-px tolerance. None when
    neither mask has a pixel there (the coarser mask lies outside this view): nothing was compared, so nothing agrees."""
    size = cur_mask.shape[::-1]
    m = np.linalg.inv(cur_to_l0) @ prev_to_l0
    warped = cv2.warpPerspective(prev_mask, m, size, flags=cv2.INTER_NEAREST)
    footprint = cv2.warpPerspective(np.ones_like(prev_mask), m, size, flags=cv2.INTER_NEAREST) > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a = cv2.dilate(cur_mask, k).astype(bool) & footprint
    b = cv2.dilate(warped, k).astype(bool)
    union = np.count_nonzero(a | b)
    return round(float(np.count_nonzero(a & b) / union), 3) if union else None


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


def validate_replay(replay: list, shape, max_size: int = 512) -> list[tuple[str, int, int, int, int | None]]:
    """Earlier decisions to re-execute: [action, x, y, level] or [action, x, y, level, size].
    Client-supplied, so every field is checked."""
    if not isinstance(replay, list) or len(replay) > MAX_REPLAY:
        raise ValueError(f"replay must be a list of at most {MAX_REPLAY} decisions")
    H, W = shape[:2]
    out = []
    for item in replay:
        if not (isinstance(item, (list, tuple)) and len(item) in (4, 5) and item[0] in REPLAY_ACTIONS
                and all(isinstance(v, int) and not isinstance(v, bool) for v in item[1:])):
            raise ValueError("each replay decision is [action, x, y, level(, size)] with integers")
        action, x, y, level, *rest = item
        size = rest[0] if rest else None
        if not (0 <= x < W and 0 <= y < H and 0 <= level <= 3):
            raise ValueError("replay decision outside the image or level range")
        if size is not None and not MIN_GAZE <= size <= max_size:
            raise ValueError(f"replay gaze size must be {MIN_GAZE}..{max_size}")
        out.append((action, x, y, level, size))
    return out


def gaze_path_from_trace(lines) -> list[list[int]]:
    """Replay the gaze path from trace JSON alone (FR-15)."""
    return [[r["decision"]["args"]["x"], r["decision"]["args"]["y"], r["decision"]["args"]["level"]]
            for r in lines if r.get("type") == "step" and r["decision"]["action"] != "halt"]
