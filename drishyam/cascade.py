"""An optional two-model cascade: a fast scout segments every look, a stronger net confirms only the looks that need it.

Off unless a Job is given a confirm net (`Job(confirm=...)`). DRISHYAM_ONNX_CONFIRM names one for bench.micro and the
dev evals (`default_confirm`); no app path (web API, CLI, MCP server) passes one. After the scout segments a look that
passed the quality gate, the confirm net segments the same crop when the scout drafted a critical class ("critical")
or its entropy lies in [tau_confirm_lo, tau_confirm_hi] ("ambiguous"), and the look keeps the confirm's segmentation
for everything after: escalation, drafting, seeding and the step's observation. There is no second budget: the confirm
reads the crop a second time, so its pixels count against the pixel cap, and its time runs on the job's clock (the
seconds and $ caps, and the stop rule's spend). A confirm that would break a cap is skipped and the step says which.
It is timed as its own op, segment:confirm, so the fixed cost model charges each net its own p50, and it always runs
the confirm's accurate net: the governor's model ladder moves the scout only.
"""

from __future__ import annotations

import functools
import os

from .findings import is_critical
from .measure import measure
from .segment import OnnxSegmenter, segment_view


@functools.cache
def default_confirm():
    """The confirm net DRISHYAM_ONNX_CONFIRM names (its accurate net only), else None: no cascade."""
    path = os.environ.get("DRISHYAM_ONNX_CONFIRM")
    return OnnxSegmenter(path) if path else None


def reason(seg, level: int, cfg) -> str | None:
    """Why the confirm net segments this look again: "critical" when the scout drafted a critical class (a mask of at
    least A_min whose largest component is critical by the findings rule), else "ambiguous" when the scout's entropy
    lies in [tau_confirm_lo, tau_confirm_hi], else None."""
    if any(a >= cfg.a_min and is_critical(c, measure(seg.mask(c), 2**level, linear=False)["area_px"])
           for c, a in seg.areas.items()):
        return "critical"
    return "ambiguous" if cfg.tau_confirm_lo <= seg.entropy <= cfg.tau_confirm_hi else None


def ms_hat(job) -> float:
    """Predicted ms of one confirm: its fixed cost (JobConfig.t_hat_ms), else its p95 so far, else (none timed yet)
    the scout's median segment."""
    if t := job.cfg.t_hat_ms:
        return t.get("segment:confirm", t["segment"])
    return job.stats.p("segment:confirm", 95, job.stats.median("segment") or 30.0)


def fits(budget, size: int, elapsed_s: float, ms: float) -> str | None:
    """None when a confirm of a size x size look fits under the job's caps, else the cap it would break: its pixels on
    top of the look's own (not yet counted), and `ms` more on the clock after `elapsed_s`."""
    left = budget.pixels_cap - budget.pixels_used - size * size
    if size * size > left:
        return f"pixel cap: {left} px left after this look, {size * size} needed"
    s = elapsed_s + ms / 1000
    if s > budget.seconds_cap:
        return f"seconds cap: {s:.1f} s > {budget.seconds_cap:g} s"
    if (usd := s * budget.usd_per_hour / 3600) > budget.usd_cap:
        return f"$ cap: ${usd:.5f} > ${budget.usd_cap:.5f}"
    return None


def confirm_look(job, seg, crop, level: int, valid, elapsed_s: float):
    """(the segmentation the look keeps, the step's `confirm` record or None when the scout gave no reason)."""
    why = reason(seg, level, job.cfg)
    if why is None:
        return seg, None
    rec = {"reason": why, "net": job.confirm.name,
           "scout": {"entropy": seg.entropy, "classes": {c: a for c, a in seg.areas.items() if a}}}
    if full := fits(job.budget, crop.shape[0], elapsed_s, ms_hat(job)):
        return seg, {**rec, "skipped": full}
    out, ms = job._timed("segment:confirm", segment_view, job.confirm, crop, "accurate", valid, 2**level)
    return out, {**rec, "latency_ms": round(ms, 1)}
