"""Budget governor and live runtime statistics.

The governor runs before every step. It reads the caps and the measured per-op
latencies and may halt the job, switch the model, coarsen the level or shrink
the gaze; every override returns a trace `cause` so the decision is auditable.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

USD_PER_HOUR = 0.339  # c8g.2xlarge on-demand + COOL software fee, us-east-1
MODEL_LADDER = ("accurate", "fast")
MIN_GAZE = 256
WARP_SPIKE_FLOOR_MS = 10.0  # a 2x jump of 1 ms is timer noise, not a spike


class RuntimeStats:
    """Rolling per-op latency window plus process CPU share since the job began."""

    def __init__(self, window: int = 50, clock=time.perf_counter) -> None:
        self.clock = clock
        self.samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._wall0, self._cpu0 = clock(), time.process_time()

    def record(self, op: str, ms: float) -> None:
        self.samples[op].append(float(ms))

    def reset(self, op: str) -> None:
        self.samples[op].clear()

    def p(self, op: str, q: float, default: float = 0.0) -> float:
        s = self.samples.get(op)
        return float(np.percentile(s, q)) if s else default

    def median(self, op: str) -> float:
        return self.p(op, 50)

    def cpu_pct(self) -> float:
        wall = self.clock() - self._wall0
        return round(100 * (time.process_time() - self._cpu0) / wall, 1) if wall > 0 else 0.0

    def snapshot(self, usd: float) -> dict:
        ops = sorted(self.samples)
        return {
            "p50_ms": {op: round(self.p(op, 50), 1) for op in ops},
            "p95_ms": {op: round(self.p(op, 95), 1) for op in ops},
            "cpu_pct": self.cpu_pct(),
            "usd": round(usd, 6),
        }


def kill_path() -> str:
    return os.environ.get("DRISHYAM_KILL_FILE") or os.path.expanduser("~/.drishyam/KILL")


def global_kill() -> bool:
    """Operator-wide stop: an env flag, or a file (`drishyam kill`) whose existence halts every job."""
    return os.environ.get("DRISHYAM_KILL_SWITCH") == "1" or os.path.exists(kill_path())


@dataclass
class Budget:
    pixels_cap: int
    usd_cap: float
    seconds_cap: float
    pixels_used: int = 0
    seconds_used: float = 0.0
    usd_per_hour: float = USD_PER_HOUR

    @property
    def usd_used(self) -> float:
        return self.seconds_used * self.usd_per_hour / 3600

    def spend(self, pixels: int, seconds: float) -> float:
        before = self.usd_used
        self.pixels_used += pixels
        self.seconds_used += seconds
        return self.usd_used - before


@dataclass
class Settings:
    """What the governor may change. The policy reads these before each look."""

    model: str = "accurate"
    level_bias: int = 0  # added to every planned level (coarser) after an SLA breach
    size: int = 512
    critical_only: bool = False
    batch: bool = False
    history: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    halt: bool = False
    cause: str | None = None
    rationale: str = ""


def check(budget: Budget, stats: RuntimeStats, settings: Settings, next_pixels: int, sla_ms: float,
          kill: bool) -> Verdict:
    """Run the governor table from the PRD, in priority order."""
    if kill or global_kill():
        return Verdict(True, "kill-switch", "kill switch set; state preserved for resume")
    if budget.pixels_used + next_pixels > budget.pixels_cap:
        return Verdict(True, "cap-hit", f"next look needs {next_pixels} px; "
                       f"{budget.pixels_cap - budget.pixels_used} px left of the cap")
    if budget.usd_used >= budget.usd_cap:
        return Verdict(True, "cap-hit", f"${budget.usd_used:.5f} spent of ${budget.usd_cap:.5f}")
    if budget.seconds_used >= budget.seconds_cap:
        return Verdict(True, "cap-hit", f"{budget.seconds_used:.1f} s used of {budget.seconds_cap:.0f} s")

    seg_p95 = stats.p(f"segment:{settings.model}", 95)
    if seg_p95 > sla_ms:
        if settings.model != MODEL_LADDER[-1]:
            settings.model = MODEL_LADDER[MODEL_LADDER.index(settings.model) + 1]
            change = f"model -> {settings.model}"
        else:
            settings.level_bias += 1
            change = f"one level coarser (bias {settings.level_bias})"
        stats.reset(f"segment:{settings.model}")
        settings.history.append("p95>SLA")
        return Verdict(False, "p95>SLA", f"p95(segment) {seg_p95:.0f} ms > SLA {sla_ms:.0f} ms; {change}")

    warp = stats.samples.get("warp")
    warp_p95, warp_med = stats.p("warp", 95), stats.median("warp")
    if (warp and len(warp) >= 5 and warp_p95 > 2 * warp_med and warp_p95 - warp_med > WARP_SPIKE_FLOOR_MS
            and settings.size > MIN_GAZE):
        p95 = warp_p95
        settings.size = max(MIN_GAZE, settings.size * 3 // 4)
        settings.batch = True
        stats.reset("warp")
        settings.history.append("warp-spike")
        return Verdict(False, "warp-spike", f"p95(warp) {p95:.1f} ms > 2x median; gaze {settings.size} px, batching")

    if not settings.critical_only and budget.usd_used >= 0.8 * budget.usd_cap:
        settings.critical_only = True
        settings.history.append("usd>=80%")
        return Verdict(False, "usd≥80%", "80% of the $ cap spent; verifying critical classes only")
    return Verdict()
