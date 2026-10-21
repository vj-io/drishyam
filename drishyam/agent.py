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
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from . import imaging
from .cascade import confirm_look, ms_hat  # cascade
from .findings import IOU_AGREE, SEVERITY, Finding, critical_rule, is_critical, now_iso, work_order
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
SPIKE_X = 1.8  # chaos p95_spike records a look at 1.8x the job's governor limit (900 ms at the default 500 ms)
PLANNER_TIMEOUT_STEP = 3
PLANNER_MALFORMED_STEP = 6
REPLAY_ACTIONS = ("gaze", "escalate", "verify")
DOMINATED_BY = {"corrosion": ("exposed_rebar",), "crack": ("spalling", "exposed_rebar"),
                "efflorescence": ("spalling", "exposed_rebar")}
MAX_REPLAY = 200
# JobConfig.v0 default: the largest v0 tried (0-4) that lost no recall at the 30 % cap on the dev splits (synth_dev,
# codebrim_dev100). Uncapped it costs recall on synth_dev (0.555 against 0.565 for v0 <= 1, at 0.45 of the pixels
# instead of 0.48-0.50) and saves false alarms on the clean dev split (15.8 boxes per photo instead of 20.3);
# results/dev/v0_sweep.json.
V0 = 2.0
# Where each segmenter is reliable, by its name (JobConfig knobs left None take its values; a value set explicitly
# always wins): min_view, the smallest share of the photo's long side a look may span, which bounds the first look and
# every escalation and verify look; absorb, whether a crack or rust inside a spall's or bar's finding joins it;
# tau_scout, the entropy threshold of a look at L2 or coarser (tau keeps every finer look; unset, tau everywhere).
# classical-v1, and any segmenter not listed, keeps the agent as published: any level, absorption on.
# The dacl10k ONNX values were chosen on the dev splits only (results/dev/onnx_profile_sweep.json, 30 % cap, paired
# 95 % CIs). On CODEBRIM dev100 photos (mostly 4608 px) L1 looks (0.22 of the side) add boxes, not recall: L2 as
# first and finest level found 0.397 at 33 boxes per photo against 0.403 at 85 for L1; with absorption off, 0.473
# against 0.415 for the original levels (L1 first, L0 finest; x1.14, CI 1.07-1.22). Absorption had dropped real
# cracks the model segmented beside a spall (x1.19 with it off, CI 1.11-1.31). On synth_dev (2048 px) L1 looks (0.5
# of the side) do add recall: L2 alone found x0.88 of the old agent's (CI 0.82-0.93). Any share above 0.222 (L1 at
# 4608 px) up to 0.341 (L2 at 6000 px) keeps L2 the finest level on those photos and allows L1 on 2048-px ones, but
# other dev100 sizes have breakpoints inside that range (L0 of 2272 px spans 0.225, L0 of 2048 px 0.25, L1 of 3648 px
# 0.281, L0 of 1600 px 0.32), so the value still moves levels: 0.3 makes L2 the finest on the 3 photos of 3648 px and
# L1 on the 2048- and 2272-px ones, and allows L0 on the 2 of 1600 px. 0.3 found x1.18 a 1024-px downscale on
# CODEBRIM (CI 1.07-1.31) and x1.03 the old agent on synth_dev (CI 0.98-1.07), at no more false alarms on the clean
# split. 0.25 (L1 on the 3648-px photos, L0 on the 2048-px ones) showed no measurable difference on either at the
# 30 % cap; uncapped, its L0 looks found a little more on synth_dev (0.3 against it: x0.98, CI 0.95-1.00).
# A mask first drafted at the finest level gets no verify look (no finer level to confirm it at; its verify_iou stays
# null, and the step's rationale says so): the dev numbers above ran that way, and a verify at the same level (a
# shifted window) was not built or measured.
# tau_scout was measured for classical-v1 on the dev splits and not adopted (results/dev/tune_sweep.json, fixed costs,
# paired 95 % CIs, against tau 0.5 at every level). A coarse look that escalates drafts nothing of what it saw; with
# 0.7 a survey whose entropy is in (0.5, 0.7] drafts and is verified one level finer instead. At the 30 % cap that
# found x1.03 on synth_dev (CI 0.98-1.09, not a measurable gain) and x1.22 on CODEBRIM dev100 (1.09-1.43): 0.153 to
# 0.186, where the same boxes at random positions find 0.088 and 0.099 (mean of 30 draws; IoU-only 0.066 to 0.086
# against 0.019 and 0.023). It sent 242 critical boxes to the human gate on CODEBRIM dev100 instead of 178 (239 and 176
# matching no ground truth), put 8 more boxes on 2 clean photos of 30 (12.40 per photo against 12.13), and any value in
# [0.692, 0.703) gives the same dev rows, the upper edge a clean dev survey's entropy. Lower values lost recall (0.3:
# x0.94 on synth_dev, CI 0.88-0.99); above 0.7 no dev recall moved and more clean photos held a box.
# The retrained v2a model (drishyam_dacl10k_v2a.onnx) keeps v1's values: no candidate passed the rule stated before its
# sweep (results/dev/onnx_v2a_profile_sweep.json: a recall gain over them at the 30 % cap on synth_dev and on CODEBRIM
# dev100, each paired 95 % CI above 1, with no more clean false alarms by the same standard). Against them, any level
# (min_view 0) found x1.01 on synth_dev (CI 0.97-1.05) and x0.79 on CODEBRIM (0.70-0.88); 0.15 x1.01 and x1.02
# (0.95-1.10) at 2.2x the boxes on CODEBRIM; absorption on x0.98 (0.94-1.00) and x0.89 (0.82-0.95).
# The v2b model (drishyam_dacl10k_v2b.onnx, an EfficientNet-B2 UNet) keeps them too, under the same rule stated before
# its sweep (results/dev/onnx_v2b_profile_sweep.json): no candidate found more on synth_dev (x1.00, CI 0.97-1.03 or
# narrower). On CODEBRIM any level found x0.66 (0.56-0.77), absorption on x0.84 (0.75-0.91), 0.15 x0.98 (0.90-1.06) at
# 2.4x the boxes.
# The v2c model (drishyam_dacl10k_v2c.onnx, v2a's MobileNetV3-Large UNet trained with rare classes weighted harder)
# keeps them too, under the same rule stated before its sweep (results/dev/onnx_v2c_profile_sweep.json): no candidate
# found more on synth_dev (x0.98, CI 0.95-1.02; absorption on 0.95-1.00; old 0.92-1.01). On CODEBRIM 0.15 found x1.09
# (1.01-1.20) at 2.3x the boxes, a gain on that split alone, which the rule does not take; any level lost recall
# (x0.92, 0.85-0.99), as did absorption on (x0.88, 0.82-0.93).
PROFILES = {
    "classical-v1": {"min_view": 0.0, "absorb": True},
    "onnx:drishyam_dacl10k.onnx": {"min_view": 0.3, "absorb": False},
    "onnx:drishyam_dacl10k_v2a.onnx": {"min_view": 0.3, "absorb": False},
    "onnx:drishyam_dacl10k_v2b.onnx": {"min_view": 0.3, "absorb": False},
    "onnx:drishyam_dacl10k_v2c.onnx": {"min_view": 0.3, "absorb": False},
}


# The weights each ONNX profile was chosen with (the file's sha256): the same file name with other weights (a retrained
# model installed as models/drishyam_dacl10k.onnx) was not measured, so it gets classical-v1.
WEIGHTS = {"onnx:drishyam_dacl10k.onnx": "c49416c10087f3c1baa8443b53fc839a3d8eb227ff27a1b4f7be1da964d6ed4a",
           "onnx:drishyam_dacl10k_v2a.onnx": "95b503c2d205567c09b19154ff77a5117889b7aba0b19733ed1c178dc4ab1ad1",
           "onnx:drishyam_dacl10k_v2b.onnx": "47e4b687d179efc9d221ef05eca441d24b0f025f9ab252c0a60c64455ff99207",
           "onnx:drishyam_dacl10k_v2c.onnx": "24b4d561fb012cb525131e16fa211b528a2e30c90fd9b9e76d98e29edbf18d21"}


# The governor's limit on the segmenter's p95 per look (`governor.check`), per profile; any other profile gets SLA_MS.
# Not a dev-chosen value: on Lambda arm64 one v1 look (512 px, FP32) measured 0.56-1.14 s at p95 over the smoke runs
# of 2026-10-21 (docs/aws.md), so at 500 ms every live v1 job left the 512-px net the held-out runs measured (governor
# off) for the 256-px FP16 net after its first looks. 1500 ms is that p95 plus headroom, set once from latency, never
# from recall; a slower step than that still switches.
SLA_MS = 500.0
SLA_MS_BY_PROFILE = {"onnx:drishyam_dacl10k.onnx": 1500.0}

# policy2: v1-policy2's knobs (JobConfig grid, crit_weight, crit_floor, crit_confirm, merge_critical) by profile, for a
# policy2: JobConfig field left None; a profile not listed, or a knob it does not set, takes POLICY2_OFF, the agent before
# policy2: them. results/dev/policy2_sweep.json states the rule, grid and dev runs they were chosen by.
# policy2: v1 takes g_s60, the one configuration the rule chose of the 8 it listed (results/dev/policy2_sweep.json, 30 %
# policy2: cap, paired 95 % CIs against v1 as it was): on CODEBRIM dev100 sign-offs per photo 2.12 to 1.34, critical
# policy2: boxes 7.24 to 6.38, recall of what a person is shown x1.02 (0.99-1.05), critical-class recall x0.98
# policy2: (0.95-1.02); nothing measurable moved on synth_dev or the clean split. It does not match L2 tiling.
POLICY2: dict = {"onnx:drishyam_dacl10k.onnx": {"grid": True, "crit_weight": 2.0, "crit_floor": 0.6,  # policy2
                                               "crit_confirm": "shift", "merge_critical": True}}  # policy2
POLICY2_OFF = {"grid": False, "crit_weight": 1.0, "crit_floor": 0.0, "crit_confirm": "off", "merge_critical": False}  # policy2
CRIT_CLASSES = ("exposed_rebar", "spalling")  # policy2: the classes findings.is_critical can make critical


def profile_for(segmenter) -> str:
    """The PROFILES key of a segmenter (its `name`), or classical-v1 for one not calibrated: a name not listed, or
    weights (its `sha256`) other than those the profile was chosen with (WEIGHTS)."""
    name = getattr(segmenter, "name", None)
    ok = name in PROFILES and WEIGHTS.get(name) in (None, getattr(segmenter, "sha256", None))
    return name if ok else "classical-v1"


@dataclass
class JobConfig:
    pixel_cap: float = 0.30
    usd_cap: float = 0.05
    seconds_cap: float = 20.0
    size: int = 512
    first_level: int = 1
    min_level: int = 0  # the finest level an escalation or a verify look may go to (min_view may raise it)
    min_view: float | None = None  # a look spans at least this share of the photo's long side (0: any); None: profile's
    profile: str | None = None  # a PROFILES key; None: the segmenter's own (`profile_for`)
    tau: float = 0.5  # entropy above this escalates (at L2 or coarser too, unless tau_scout is set)
    tau_scout: float | None = None  # the same for a look at L2 or coarser; None: the profile's (PROFILES), else tau
    a_min: int = 64  # crop px; a smaller mask escalates
    eps: float = 0.25  # stop-rule slope factor
    sla_ms: float | None = None  # the governor's segment p95 limit; None: SLA_MS_BY_PROFILE's for the profile, else SLA_MS
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
    absorb: bool | None = None  # a crack or rust mostly inside a spall's or bar's finding joins it (DOMINATED_BY)
    order: str = "score"  # "random": a seeded random pick among unseen candidates instead of the argmax
    seed: int = 0
    governor: bool = True  # False: kill switch and caps only; no model, level, gaze-size or critical-only override
    # When the next look no longer fits under the pixel cap, it shrinks to the largest gaze (a multiple of 32 px, at
    # least MIN_GAZE) that does, instead of the job halting on cap-hit; the stop rule still decides whether it is taken.
    fill: bool = False
    # Whether the survey's finds (and its cost) count toward the stop rule's value found per ms. It sees the whole photo,
    # so its value is not what a look at one candidate yields; False: only looks at candidates set that term.
    survey_value: bool = True
    # A fixed predicted cost per look, {"warp": ms, "segment": ms, "segment:<model>": ms}, instead of live p95s:
    # scores and the stop rule then do not move with machine load. The caps still run on the real clock.
    t_hat_ms: dict | None = None
    # fallback: when the overview proposes no candidate, one look at L2 whose window spans the whole photo before the
    # fallback: job ends (cause fallback-l2), then the rules' follow-ups; off, such a photo gets no look at all. On for
    # fallback: every profile since it passed the rule stated before its dev run (results/dev/zero_candidate_fallback.json,
    # fallback: v1 at the 30 % cap; it acted on 1 dev photo of 190); every number published before it ran with it off.
    fallback_l2: bool = True  # fallback
    # cascade: with a confirm net (Job(confirm=...), drishyam.cascade; none by default), the scout entropy band, both
    # cascade: ends included, in which the confirm net segments a look again. Not tuned: it straddles tau.
    tau_confirm_lo: float = 0.35  # cascade
    tau_confirm_hi: float = 0.65  # cascade
    # policy2: v1-policy2 (None: the profile's value in POLICY2, else POLICY2_OFF's):
    grid: bool | None = None  # policy2: first looks on the first level's tile grid (`_tiles`), one per tile holding a candidate
    crit_weight: float | None = None  # policy2: x the look score's severity of a region seen as a CRIT_CLASSES class
    crit_floor: float | None = None  # policy2: a mask that would make a new critical finding, under this confidence, is held
    crit_confirm: str | None = None  # policy2: a held mask's confirm look: "shift" (its level, recentred), "finer" or "off"
    merge_critical: bool | None = None  # policy2: a critical mask mostly inside a critical finding of its class joins it


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
    box: tuple | None = None  # its overview box (candidates only; None for the survey and replayed regions)
    shown_level: int | None = None  # level of the finest look that has shown it (and seeded it); None: none has
    looks: list = field(default_factory=list)  # Look per look at this region
    done: bool = False
    verified: bool = False
    extra_used: bool = False
    held: list = field(default_factory=list)  # policy2: (class, Look, confidence, step, area) its confirm look decides


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
                 confirm=None,  # cascade: the confirm net (drishyam.cascade); None, the default: no cascade
                 replay: list | None = None) -> None:
        self.image = image
        self.segmenter = segmenter or default_segmenter()
        self.confirm = confirm  # cascade
        cfg = config or JobConfig()
        name = cfg.profile or profile_for(self.segmenter)
        self.cfg = replace(cfg, profile=name, **{k: v for k, v in PROFILES[name].items() if getattr(cfg, k) is None})
        if self.cfg.sla_ms is None:
            self.cfg = replace(self.cfg, sla_ms=SLA_MS_BY_PROFILE.get(name, SLA_MS))
        self.cfg = replace(self.cfg, **{k: v for k, v in {**POLICY2_OFF, **POLICY2.get(name, {})}.items()  # policy2
                                         if getattr(self.cfg, k) is None})  # policy2
        self.id = job_id or "j_" + secrets.token_hex(3)
        self.planner = planner
        self.kill = kill
        self.clock = clock
        self.replay = validate_replay(replay or [], image.shape, self.cfg.size)
        self.findings: list[Finding] = []
        self.gaze_path: list[list[int]] = []
        self.trace: list[dict] = []

    # ---- helpers -------------------------------------------------------------------------------
    prefetch = None  # off; bench/loop.py sets a drishyam.prefetch.Prefetcher, which serves warps and segmentations

    def _timed(self, op, fn, *args, **kw):
        t0 = self.clock()
        out = fn(*args, **kw) if self.prefetch is None else self.prefetch(self, op, fn, args, kw)
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

    def _first_look(self, r: Region) -> tuple[int, list]:
        """(level, L0 window) of the look the argmax would send to unvisited region r now."""
        level = min(r.next_level + self.settings.level_bias, self.max_level)
        return level, self._window(r.cx, r.cy, level, self.settings.size)[2]

    def _score(self, r: Region):
        """(score, expected value, unseen fraction, severity, uncertainty): ev = prior x severity x (1 + u) x unseen."""
        level, win = self._first_look(r)
        a = self._unseen(win, level)
        u = r.entropy if self.cfg.use_uncertainty else 0.0
        s = SEVERITY.get(r.cls, 1.0)
        s *= self.cfg.crit_weight if r.cls in CRIT_CLASSES else 1.0  # policy2
        ev = r.prior * s * (1 + u) * a
        return ev / max(self._t_hat(), 1e-3), ev, a, s, u

    def _seed(self, seg, to_l0, valid, level) -> list[dict]:
        """After a look the quality gate passed: every unvisited candidate whose own first-look window
        (`_first_look`, clamped to the image as that look will be) lies at least half inside the look's valid
        footprint has been shown (the stop floor needs that), and takes its uncertainty and class from what the look
        saw inside that window: the binary damage-vs-background entropy over that sub-window
        (`segment._entropy`, as a look reports it) and the most severe class whose area there would clear A_min at
        the candidate's own first-look level (None if none). A look that sees less of the window (a finer look
        covers at most a quarter of it) neither shows nor seeds it. A finer look overwrites a coarser seed, never
        the reverse; a seed is not a look (the candidate stays unvisited). seed_from_looks=False: shown, not seeded."""
        inv, size, out = np.linalg.inv(to_l0), valid.shape[0], []
        for r in self.regions.values():
            if r.looks or r.done or r.box is None or (r.shown_level is not None and level > r.shown_level):
                continue
            first, (x, y, w, h) = self._first_look(r)
            c = imaging.map_points(inv, [[x, y], [x + w, y + h]])
            lo, hi = np.floor(c.min(0)), np.ceil(c.max(0))
            (x0, y0), (x1, y1) = np.clip(lo, 0, size).astype(int), np.clip(hi, 0, size).astype(int)
            if np.count_nonzero(valid[y0:y1, x0:x1]) < 0.5 * float(np.prod(hi - lo)):
                continue
            r.shown_level = level
            if not self.cfg.seed_from_looks:
                continue
            labels = seg.labels[y0:y1, x0:x1]
            min_area = max(MIN_BLOB_PX, self.cfg.a_min * 4.0 ** (first - level))
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
            self.drills.add("planner_timeout")
        elif "planner_malformed" in cfg.chaos and self.step == PLANNER_MALFORMED_STEP:
            planner = ChaosPlanner("malformed")
            self.drills.add("planner_malformed")
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
            asked = p.get("level", (r.looks[-1].level if r.looks else r.next_level) - 1)
            level = max(self.min_level, asked)
            why = proposal["rationale"][:120] + (f" (L{asked} is finer than this segmenter's finest L{level})"
                                                 if level > asked else "")
            r.done = False
            self.forced.appendleft(("verify" if strategy == "reverify" else "gaze", r, level, f"planner:{strategy}", why))
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
        # One physical defect, one finding: rust on exposed bars is the bar, a spall's rim is the spall. A check of the
        # absorbed class is not a check of the dominating finding's class, so it leaves that finding's verify alone.
        if self.cfg.absorb and (dom := next((f for f in self.findings if f.cls in DOMINATED_BY.get(cls, ())
                                             and _inside(box, f.box) >= 0.5), None)):
            return dom, "absorbed"
        same = [f for f in self.findings if f.cls == cls and (f.region == r.id or imaging.box_iou(f.box, box) > 0.3)]
        if not same and self.cfg.merge_critical and is_critical(cls, m["area_px"]):  # policy2: one finding per critical
            same = [f for f in self.findings if f.cls == cls and f.critical  # policy2: defect: a mask mostly inside (IoS
                    and max(_inside(box, f.box), _inside(f.box, box)) >= 0.5]  # policy2: >= 0.5) one of its class joins it
        if same:
            f = same[0]
            f.agent_update(confidence=conf, measure=m, box=box, step=self.step, level=level, crop_jpeg_b64=thumb,
                           verify_iou=verify_iou)
            return f, "update"
        f = Finding(f"f{len(self.findings) + 1}", cls, conf, m, box, self.step, level, r.id, crop_jpeg_b64=thumb,
                    verify_iou=verify_iou)
        self.findings.append(f)
        return f, "new"

    def _triage(self, r, cause, seg, to_l0, level, present, first, replaying):  # policy2
        # policy2: v1-policy2's gate before a person. A confirm look (cause confirm-critical, at the region made for it)
        # policy2: drafts only the held classes it confirms: its own mask at or above crit_floor, or one overlapping the
        # policy2: held mask at IoU >= IOU_AGREE; every other held class goes to a person as needs-a-look. Any other look
        # policy2: holds each mask that would make a new critical finding under crit_floor and queues one confirm look
        # policy2: for them (crit_confirm), or without one sends them to a person at once. Returns (to draft, a note).
        cfg = self.cfg  # policy2
        if cause == "confirm-critical":  # policy2
            self.pending.discard(r.id)  # policy2
            ok = [k for k, lk, *_ in r.held if k in present and (seg.confidence[k] >= cfg.crit_floor or (look_iou(  # policy2
                lk.mask(k), lk.to_l0, seg.mask(k), to_l0) or 0.0) >= IOU_AGREE)]  # policy2
            self.not_assessed += [self._needs_look(k, lk, c, s, f"not confirmed by its L{level} confirm look")  # policy2
                                  for k, lk, c, s, _ in r.held if k not in ok]  # policy2
            return ok, f"; confirm look: {', '.join(ok) or 'none'} of {', '.join(h[0] for h in r.held)} confirmed"  # policy2
        held = [c for c in present if seg.confidence[c] < cfg.crit_floor and self._new_critical(r, c, seg, to_l0, level)]  # policy2
        if not held:  # policy2
            return present, ""  # policy2
        look = Look(level, seg.labels, to_l0, held[0])  # policy2
        items = [(c, look, seg.confidence[c], self.step, measure(seg.mask(c), 2**level, linear=False)["area_px"])  # policy2
                 for c in held]  # policy2
        lvl = min(level - (cfg.crit_confirm == "finer"), first)  # policy2: never coarser than a first look
        names = ", ".join(f"{c} {seg.confidence[c]:.2f}" for c in held)  # policy2
        keep = [c for c in present if c not in held]  # policy2
        if cfg.crit_confirm in ("shift", "finer") and lvl >= 0 and cfg.verify and not self.settings.batch and not replaying:  # policy2
            ys, xs = np.nonzero(np.isin(seg.labels, [CLASSES.index(c) for c in held]))  # policy2
            (fx, fy), = imaging.map_points(to_l0, [[float(xs.mean()), float(ys.mean())]])  # policy2
            rc = Region(f"{r.id}.c{self.step}", int(fx), int(fy), r.prior, lvl, looks=[look], extra_used=True, held=items)  # policy2
            self.regions[rc.id] = rc  # policy2
            self.pending.add(rc.id)  # policy2
            why = f"{names} under the floor {cfg.crit_floor:g} at L{level} (step {self.step}): confirming at L{lvl}"  # policy2
            self.forced.append(("verify", rc, lvl, "confirm-critical", why + ", centred on the mask"))  # policy2
            return keep, f"; {names} under the floor {cfg.crit_floor:g}: held for a confirm look at L{lvl}"  # policy2
        self.not_assessed += [self._needs_look(c, look, cf, s, "no confirm look") for c, _, cf, s, _ in items]  # policy2
        return keep, f"; {names} under the floor {cfg.crit_floor:g}: sent to a person, needs a look"  # policy2

    def _new_critical(self, r, c, seg, to_l0, level) -> bool:  # policy2
        # policy2: whether this look's mask of c would put a critical finding before a person that is not there yet:
        # policy2: critical by area as _log_finding measures it, and joining no critical finding of c by its rules
        mask = seg.mask(c)  # policy2
        if not is_critical(c, measure(mask, 2**level, linear=c == "crack")["area_px"]):  # policy2
            return False  # policy2
        box = imaging.l0_box(cv2.boundingRect(cv2.findNonZero(mask)), to_l0, self.image.shape[1::-1])  # policy2
        return not any(f.cls == c and f.critical and (f.region == r.id or imaging.box_iou(f.box, box) > 0.3 or (  # policy2
            self.cfg.merge_critical and max(_inside(box, f.box), _inside(f.box, box)) >= 0.5)) for f in self.findings)  # policy2

    def _needs_look(self, cls, look, conf, step, why) -> dict:  # policy2
        # policy2: a held mask as a not-assessed entry: a person checks it by eye, and the work order is never ready meanwhile
        box = imaging.l0_box(cv2.boundingRect(cv2.findNonZero(look.mask(cls))), look.to_l0, self.image.shape[1::-1])  # policy2
        return {"step": step, "box": box, "reason": f"{cls} at L{look.level}, confidence {conf:.2f} under the floor "  # policy2
                f"{self.cfg.crit_floor:g}: not a critical finding; {why}", "action": "check-by-eye",  # policy2
                "kind": "needs-a-look", "cls": cls, "confidence": round(float(conf), 3)}  # policy2

    def _settle(self, r, chk, level, best, target, now_f, critical, drafted, new_events, areas):
        """After a look that queued no escalation: settle its comparison with the coarser look (`chk`, from
        `compare`), queue the next look at this region or mark it done. Returns the follow-up look or None.
        `target`: the finding of the compared class itself (never one that absorbed it); `now_f`: this look's
        finding of its best class; `critical`: this look's first critical finding of its own class (a human reviews
        this region anyway); `drafted`: whether this look drafted anything; `areas` (class changes only): the area in
        L0 px of the compared class at the coarser look and of this look's best class, each its largest component as
        a finding measures it, which say whether a class change is critical.
        Returns (the follow-up look or None, a note for this look's rationale: a verify the rules call for that the
        profile's finest level rules out, else "")."""
        cfg = self.cfg
        # A verify look goes one level finer; at the profile's finest level (min_view) there is none to go to.
        finest = (f"L{level} is the finest level for {cfg.profile} (min_view {cfg.min_view:g})"
                  if 0 < level == self.min_level else None)
        if chk is None:
            note = ""
            if drafted and not self.settings.batch and not r.verified and cfg.verify:
                if level > self.min_level:
                    return ("verify", r, level - 1, "verify-new-mask",
                            f"new {best} mask at L{level}; confirming at L{level - 1}"), ""
                if finest:
                    note = f"; new {best} mask not verified: {finest}, no finer look"
            r.done = True
            return None, note
        k, a, iou = chk["class"], chk["against_level"], chk["iou"]
        if chk["kind"] == "agree":
            r.verified = r.done = True
            return None, ""
        if chk["kind"] == "class-change":
            # The finer look sees the same place as another class: it keeps that class, the coarser one is recorded
            # (the step's verify.class; the coarser finding is unconfirmed). A human decides when either CLASS is
            # critical, also when the coarser look drafted nothing (it escalated): the request then names the finer
            # look's finding. When no finding named is critical, that finding is held for the human like a critical
            # one (`Finding.hold`), so the work order waits for the decision. Neither look drafted anything (both
            # masks under A_min): no finding, no request.
            r.done = True
            now = chk["class_now"]
            named = [f for f in (target, now_f) if f is not None]
            crit_cls = next(((c, lv, ar) for c, lv, ar in ((k, a, areas[0]), (now, level, areas[1]))
                             if is_critical(c, ar)), None)
            if named and (any(f.critical for f in named) or crit_cls):
                crit = next((f for f in named if f.critical), None)
                if crit is None:
                    crit, (c, lv, ar) = named[0], crit_cls
                    crit.hold(self.step, ar, f"class change: {c} at L{lv} here ({critical_rule(c, ar)})")
                    new_events.append({"type": "finding", "finding": crit.to_dict()})
                new_events.append(review(crit.id, f"class change: {k} at L{a}, {now} at L{level} (same place, iou "
                                         f"{chk['iou_any']:g})", level, [k, now], chk["iou_any"], a, kind="class-change"))
            return None, ""
        disagree = f"iou {iou:g} < {IOU_AGREE} between L{a} and L{level}"
        note = ""
        if target is not None and target.critical:  # a critical finding goes to a human at once
            new_events.append(review(target.id, disagree, level, [k, k], iou, a, kind="disagree"))
            r.done = True
        elif not r.extra_used and level > self.min_level and cfg.verify and critical is None:  # (a critical finding here: a human
            r.extra_used = True  # reviews this region anyway, so no look is spent to settle the disagreement)
            return ("verify", r, level - 1, "iou<0.5", disagree), ""
        else:
            r.done = True
            if not r.extra_used and finest and cfg.verify and critical is None:
                note = f"; {disagree} not settled: {finest}, no finer look"
            if target is not None:  # no further look will settle it
                new_events.append(review(target.id, f"unconfirmed: {disagree}", level, [k, k], iou, a,
                                         kind="unconfirmed"))
        return None, note

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
        fits = int(math.log2(max(W, H) / cfg.size)) if max(W, H) > cfg.size else 0  # coarsest level a look fits in
        first = max(0, min(cfg.first_level, fits))
        # min_view: no look the rules or a planner choose, first or follow-up, is finer than the finest level whose
        # window spans that share of the long side (0: any level), capped where a look still fits in the photo. A
        # replayed look keeps its recorded level, and a gaze the governor shrank (warp-spike) keeps its level.
        wide = min(fits, self.max_level, max(0, math.ceil(math.log2(cfg.min_view * max(W, H) / cfg.size)))) \
            if cfg.min_view else 0
        first = max(first, wide)
        self.min_level = max(wide, min(cfg.min_level, first))
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
        if cfg.grid and cands:  # policy2: one first look per tile of the first level's grid that holds a candidate
            self.regions = _grid_regions(cands, pyr[first].shape[:2], first, cfg.size)  # policy2
        self.pending: set = set()  # policy2: confirm regions whose confirm look has not run
        self.seen = [np.zeros(sal.shape, bool) for _ in range(len(pyr))]
        self.looked = np.zeros(sal.shape, bool)  # seen by a look other than the survey: coverage_looked
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
        if not cands and not self.replay:  # nothing proposed is not a clean photo: no look will run, so a person looks
            self.not_assessed.append({"step": 0, "box": [0, 0, W, H], "reason": "no candidate region was proposed",
                                      "action": "check-by-eye"})
        # fallback: the overview's L3 pyramid can blur a hairline crack away, so a photo the prior is silent on gets one
        # fallback: look at L2 spanning it (a gaze of the long side / 4 px, never under `size`), governed and capped
        # fallback: like any look; it stays not assessed whatever that look drafts. A resumed job replays the look.
        whole = max(cfg.size, math.ceil(max(W, H) / 4))  # fallback
        if cfg.fallback_l2 and not cands and self.replay and self.replay[0][3:] == (2, whole):  # fallback
            self.not_assessed.append({"step": 0, "box": [0, 0, W, H],  # fallback
                                      "reason": "no candidate region was proposed", "action": "check-by-eye"})  # fallback
        elif cfg.fallback_l2 and not cands and not self.replay:  # fallback
            self.settings.size = whole  # fallback
            r = self.regions["w0"] = Region("w0", W // 2, H // 2, 0.0, 2)  # fallback
            why = f"no candidate region was proposed: one L2 look of {self.settings.size} px spans the photo"  # fallback
            self.forced.append(("gaze", r, 2, "fallback-l2", why))  # fallback
        asked: set = set()  # (finding id, class) pairs already sent to review as ambiguous
        self.followups = 0  # events that follow the step record just yielded (its findings and review requests)
        self.drills: set = set()  # the chaos drills that fired (each on its fixed step)
        self.step = 0

        while self.step < cfg.max_steps:
            self.step += 1
            step_t0 = clock()
            replaying = n_replay > 0
            left, fill = self.budget.pixels_cap - self.budget.pixels_used, ""
            if cfg.fill and not replaying and self.settings.size ** 2 > left >= MIN_GAZE ** 2:
                self.settings.size = math.isqrt(left) // 32 * 32
                fill = f"; gaze {self.settings.size} px: what the pixel cap leaves ({left} px)"
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
                # holds only once, for EVERY candidate still worth a look (unseen > 0), a look at least as fine as its
                # own first look has shown it (a seed): while one has not, its value is unknown, not small. A coarser
                # look (the survey) cannot resolve what that look would (a hairline crack at L3), so the job never
                # stops on a saliency prior and a coarse view alone (nor before it has seen anything).
                lam_found = cfg.eps * value / cost_ms if cost_ms else 0.0
                fine = all(c.shown_level is not None and c.shown_level <= self._first_look(c)[0] for _, c in scored)
                lam_floor = cfg.eps * cfg.v0 / max(self._t_hat(), 1e-3) if fine else 0.0
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
            why += fill
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
            self.settings.size = min(size, cfg.size)  # fallback: a whole-photo look's gaze is its own, not the next look's
            cx, cy, window = self._window(r.cx, r.cy, level, size)
            (crop, to_l0), warp_ms = self._timed("warp", imaging.gaze, pyr, cx, cy, level, size=size)
            q, _ = self._timed("quality", imaging.quality, crop)
            model = self.settings.model
            # Background wherever the view runs past the image: gaze fills that by reflection (a mirrored defect).
            valid = imaging.footprint(to_l0, (W, H), size)
            seg, seg_ms = self._timed(f"segment:{model}", segment_view, self.segmenter, crop, model, valid, 2**level)
            self.stats.record("segment", seg_ms)
            # cascade: the confirm net looks again at an unsure or critical look, if the caps leave room for it
            seg, confirmed = (confirm_look(self, seg, crop, level, valid, clock() - t_start)  # cascade
                              if self.confirm is not None and q["ok"] else (seg, None))  # cascade
            ran = confirmed is not None and "skipped" not in confirmed  # cascade
            # Crack components segment_view dropped as straight lines (formwork, joints): logged, never drafted.
            rejected = rejected_lines(seg, to_l0, (W, H))
            injected = None
            if "p95_spike" in cfg.chaos and self.step == SPIKE_STEP:
                self.stats.record(f"segment:{model}", SPIKE_X * cfg.sla_ms)
                self.stats.record("segment", SPIKE_X * cfg.sla_ms)
                injected = "p95_spike"
                self.drills.add(injected)
            y0, y1, x0, x1 = _rect_on(self.seen[level].shape, window, self.ov_scale)
            self.seen[level][y0:y1, x0:x1] = True
            if cause != "survey" and not (cause == "replay" and self.step == 1 and level > first):  # nor its replay
                self.looked[y0:y1, x0:x1] = True
            if cause == "fallback-l2":  # fallback: the first look, and a whole-photo one is no closer than the survey
                self.looked[:] = False  # fallback
            sal_max = round(float(sal[y0:y1, x0:x1].max()), 4) if y1 > y0 and x1 > x0 else 0.0

            # 5. what the measurements say to do next
            best = seg.best if q["ok"] else None
            area = seg.areas[best] if best else 0
            tau = cfg.tau if level < 2 or cfg.tau_scout is None else cfg.tau_scout
            r.entropy = seg.entropy
            # The finer look against the last coarser look that saw a class here, class by class (PRD rule 3).
            prev = next((lk for lk in reversed(r.looks) if lk.cls), None) if q["ok"] else None
            chk = compare(prev, seg, to_l0) if prev is not None else None
            look = Look(level, seg.labels, to_l0, best)
            r.looks.append(look)
            seeded = self._seed(seg, to_l0, valid, level) if q["ok"] else []
            new_events, follow, drafted, critical = [], None, {}, None
            batch = self.settings.batch
            batch = batch or cause == "confirm-critical"  # policy2: a confirm look never escalates
            if not q["ok"]:
                r.done = True
                self.not_assessed.append({"step": self.step, "box": imaging.l0_box((0, 0, size, size), to_l0, (W, H)),
                                          "reason": q["reason"], "action": "re-shoot"})
            elif level > self.min_level and not batch and seg.entropy > tau:
                follow = ("escalate", r, level - 1, "entropy>tau", f"entropy {seg.entropy:.2f} > {tau} at L{level}")
            elif level > self.min_level and not batch and best and area < cfg.a_min:
                follow = ("escalate", r, level - 1, "area<A_min", f"{best} area {area} px < {cfg.a_min} at L{level}")
            else:
                present = self._present(seg, to_l0, level, rejected) if best else []
                present, held_note = self._triage(r, cause, seg, to_l0, level, present, first, replaying)  # policy2
                why += held_note  # policy2
                if 0 < level == self.min_level and not batch and (seg.entropy > tau or (best and area < cfg.a_min)):
                    # the escalation the rules call for, held: what this look saw, and what it drafted of it
                    saw = (f"entropy {seg.entropy:.2f} > {tau}" if seg.entropy > tau
                           else f"{best} area {area} px < A_min {cfg.a_min}")
                    kept = f"{', '.join(present)} drafted here" if present else f"{best} not drafted" if best else "no class"
                    why += (f"; L{level} is the finest level for {cfg.profile} (min_view {cfg.min_view:g}): {saw}, "
                            f"{kept}; no finer look")
                if present:
                    # Every class with a confident mask becomes evidence, most severe first.
                    r.cls = present[0]
                    # Realised value of the look, in the same units as a region's expected value
                    # (one class's severity), so the stop rule compares like with like.
                    if cause != "survey" or cfg.survey_value:
                        value += max(SEVERITY[c] * seg.confidence[c] for c in present)
                    for c in present:
                        # The check compared class chk["class"] across the two looks: evidence about that class's finding.
                        f, kind = self._log_finding(r, c, seg, crop, to_l0, level,
                                                    chk["iou"] if chk and c == chk["class"] else None)
                        if cause == "confirm-critical" and kind != "absorbed" and not f.critical:  # policy2: the held
                            f.hold(self.step, a := next(h[4] for h in r.held if h[0] == c), critical_rule(c, a))  # policy2: look's area
                        drafted[c] = f
                        if c == best:
                            look.finding = f.id
                        if kind == "absorbed":
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
                            if f.critical_step == self.step:  # new, or an update that made it critical
                                new_events.append(review(f.id, f"{c} is critical", level, [c], kind="critical"))
                # The finding the check is about: a finding OF the compared class, the one this look drafted it into,
                # else the one the coarser look drafted it into (the class is gone, too small or absorbed here: it
                # still gets this verdict). A finding that absorbed the class holds another class: never its verdict.
                target, areas = None, (0, 0)
                if chk:
                    k = chk["class"]
                    target = drafted.get(k) if k in drafted and drafted[k].cls == k else None
                    if target is None and (target := next((f for f in self.findings
                                                           if f.id == prev.finding and f.cls == k), None)):
                        target.set_verify(chk["iou"])
                        new_events.append({"type": "finding", "finding": target.to_dict()})
                    if chk["kind"] == "class-change":  # in L0 px, by the rule findings use: the largest component
                        areas = (measure(prev.mask(k), 2**prev.level, linear=False)["area_px"],
                                 measure(seg.mask(best), 2**level, linear=False)["area_px"])
                follow, note = self._settle(r, chk, level, best, target, drafted.get(best), critical, bool(drafted),
                                            new_events, areas)
                why += note

            # While replaying, the trace already holds the follow-ups; only the last replayed look may queue one.
            if follow and (not replaying or n_replay == 0):
                r.cx, r.cy = _focus(seg, to_l0, (cx, cy))
                self.forced.appendleft(follow)
            elif follow:
                r.done = True

            # 6. account and log
            step_s = clock() - step_t0
            if cause != "survey" or cfg.survey_value:
                cost_ms += self._t_hat() if cfg.t_hat_ms else step_s * 1000
                cost_ms += ms_hat(self) if ran and cfg.t_hat_ms else 0.0  # cascade: step_s holds it otherwise
            usd_before = self.budget.usd_used
            self.budget.pixels_used += size * size
            self.budget.pixels_used += size * size if ran else 0  # cascade: the confirm read the crop again
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
                "model": model, "profile": cfg.profile, "latency_ms": round(warp_ms + seg_ms, 1),
                "cost_usd": round(self.budget.usd_used - usd_before, 8),
                "pixels_used": self.budget.pixels_used,
                "pixels_frac": round(self.budget.pixels_used / (W * H), 4),
                "usd_used": round(self.budget.usd_used, 6), "seconds_used": round(self.budget.seconds_used, 3),
                "injected": injected,
            }
            if confirmed:  # cascade
                rec |= {"confirm": confirmed,  # cascade
                        "latency_ms": round(warp_ms + seg_ms + confirmed.get("latency_ms", 0.0), 1)}  # cascade
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
               "planner": "rule", "model": self.settings.model, "profile": cfg.profile,
               "pixels_used": self.budget.pixels_used,
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
        if halted:  # a candidate the halt left without a look of its own is not assessed, never clean
            self.not_assessed += [{"step": self.step, "box": imaging.l0_box(r.box, np.eye(3), (W, H)),
                                   "reason": f"not looked at: {halt_cause} halted the job first",
                                   "action": "check-by-eye", "region": r.id}
                                  for r in self.regions.values() if not r.looks and r.box is not None]
        self.not_assessed += [self._needs_look(k, lk, c, s, "the job ended before its confirm look")  # policy2
                              for rid in sorted(self.pending) for k, lk, c, s, _ in self.regions[rid].held]  # policy2
        self.work_order = work_order(self.id, self.findings, self.gaze_path, self.not_assessed)
        self.work_order["profile"] = cfg.profile  # the levels it looked at were this segmenter's (PROFILES)
        self.work_order["weights_sha256"] = getattr(self.segmenter, "sha256", None)  # the model that looked (WEIGHTS)
        if self.confirm is not None:  # cascade
            self.work_order["confirm_sha256"] = getattr(self.confirm, "sha256", None)  # cascade
        yield {"type": "done", "status": status, "cause": halt_cause,
               "summary": {"steps": len(self.gaze_path), "pixels_used": self.budget.pixels_used,
                           "pixels_frac": round(self.budget.pixels_used / (W * H), 4),
                           "usd": round(self.budget.usd_used, 6), "seconds": round(self.budget.seconds_used, 3),
                           "findings": len(self.findings), "critical_pending": pending},
               "coverage": round(coverage, 4),  # seen at any level, the survey included
               "coverage_looked": round(float((sal * self.looked).sum() / max(sal.sum(), 1e-6)), 4),
               "unvisited": [r.id for r in self.regions.values() if not r.looks],
               # requested drills whose step the job never reached (it ended first): they tested nothing
               "drills_not_reached": [d for d in cfg.chaos if d not in self.drills],
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


def _tiles(h: int, w: int, size: int, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:  # policy2
    # policy2: (x0, y0, x1, y1) tiles of an h x w level image, as evals.harness.slice_boxes cuts them (SAHI's
    # policy2: get_slice_bboxes: stride size x (1 - overlap), the last row and column pulled back to the edge): the
    # policy2: tiling baseline's own grid, so a grid look sees what a tile sees
    ov, boxes, y_min, y_max = int(overlap * size), [], 0, 0  # policy2
    while y_max < h:  # policy2
        x_min = x_max = 0  # policy2
        y_max = y_min + size  # policy2
        while x_max < w:  # policy2
            x_max = x_min + size  # policy2
            xe, ye = min(w, x_max), min(h, y_max)  # policy2
            boxes.append((max(0, xe - size), max(0, ye - size), xe, ye))  # policy2
            x_min = x_max - ov  # policy2
        y_min = y_max - ov  # policy2
    return boxes  # policy2


def _grid_regions(cands, shape, level: int, size: int) -> dict:  # policy2
    # policy2: one Region per tile of the level's grid (`_tiles` of a level image of `shape`) holding a candidate's
    # policy2: centre, each candidate counted in the first tile (in tile order) that holds it; the region's centre and
    # policy2: box are the tile's in L0 px, its prior its candidates' best
    s, out, left = 2**level, {}, list(cands)  # policy2
    for i, (x0, y0, x1, y1) in enumerate(_tiles(*shape, size)):  # policy2
        mine = [c for c in left if x0 * s <= c.cx < x1 * s and y0 * s <= c.cy < y1 * s]  # policy2
        if mine:  # policy2
            left = [c for c in left if c not in mine]  # policy2
            out[f"g{i + 1}"] = Region(f"g{i + 1}", (x0 + x1) * s // 2, (y0 + y1) * s // 2, max(c.prior for c in mine),  # policy2
                                      level, box=(x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s))  # policy2
    return out  # policy2


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
    disagree. None when the two cannot be compared: no pixel of prev's mask lies inside this view."""
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
    the coarser mask has no pixel in this view (it lies outside it): nothing was compared, so nothing agrees or
    disagrees, whatever this look shows elsewhere."""
    size = cur_mask.shape[::-1]
    m = np.linalg.inv(cur_to_l0) @ prev_to_l0
    warped = cv2.warpPerspective(prev_mask, m, size, flags=cv2.INTER_NEAREST)
    if not warped.any():
        return None
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
        if level == 2 and size == max(max_size, math.ceil(max(W, H) / 4)):  # fallback: the whole-photo look's gaze
            out.append((action, x, y, level, size))  # fallback
            continue  # fallback
        if size is not None and not MIN_GAZE <= size <= max_size:
            raise ValueError(f"replay gaze size must be {MIN_GAZE}..{max_size}")
        out.append((action, x, y, level, size))
    return out


def gaze_path_from_trace(lines) -> list[list[int]]:
    """Replay the gaze path from trace JSON alone (FR-15)."""
    return [[r["decision"]["args"]["x"], r["decision"]["args"]["y"], r["decision"]["args"]["level"]]
            for r in lines if r.get("type") == "step" and r["decision"]["action"] != "halt"]
