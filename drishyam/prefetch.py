"""Asynchronous look prefetching, off by default (bench/loop.py turns it on; nothing in the app does).

While the agent segments its current look, one background thread warps, and with `segment=True` also segments, the
looks it is likely to take next: the head of its queue of forced looks (the survey, a planner's or a replayed look)
and the `k` best unvisited candidates by the agent's own score. Each result is cached by (cx, cy, level, size, model)
and handed back when the agent asks for that very look, if the worker has started it (one not yet started is cancelled
and counted `late`: a hit then would wait behind the look ahead of it); any other look misses and runs inline as before.

Decisions cannot change: a cached result is the same function (`imaging.gaze`, `segment.segment_view`) of the same
inputs, computed earlier on another thread, and the prefetcher only reads the job. The clock is what changes: the
agent times its wait, not the work. With live costs (t-hat from measured p95s, the governor's latency rules) a faster
step can change what the agent does next, as a faster machine would; on a fixed cost table or a deterministic clock
the trace is byte-identical (tests/test_prefetch.py).

    job = Job(img, cfg)
    job.prefetch = Prefetcher(k=2)
    events = list(job.run())
    job.prefetch.close()
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor

from . import imaging
from .segment import segment_view


class Prefetcher:
    """`Job.prefetch`: called by `Job._timed` for every timed op. `segmenter`: the instance the background thread
    segments with (None: the job's own; an OnnxSegmenter serialises its forwards on a lock, so give it a second one)."""

    def __init__(self, k: int = 2, segment: bool = True, segmenter=None) -> None:
        self.k, self.segment, self.segmenter = k, segment, segmenter
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="prefetch")
        self.cache: dict[tuple, tuple] = {}  # key -> (pool future, warp future, segment future or None)
        self.crop, self.seg = None, None  # the look in progress: its crop, and its segmentation's future on a hit
        self.counts: Counter = Counter()  # warp_hit, warp_miss, seg_hit, submitted, dropped, late (a miss, see below)

    def __call__(self, job, op, fn, args, kw):
        if op == "warp":
            pyr, cx, cy, level = args
            key = (cx, cy, level, kw["size"], job.settings.model)
            pf, wf, sf = self.cache.pop(key, (None, None, None))
            if pf is not None and pf.cancel():  # still queued behind another look on the one worker: waiting would
                wf = sf = None  # cost that look's warp and forward, so it is a miss, computed inline
                self.counts["late"] += 1
            self.counts["warp_hit" if wf else "warp_miss"] += 1
            out = wf.result() if wf else fn(*args, **kw)
            self.crop, self.seg = out[0], sf
            self._ahead(job, pyr, key)  # the background works while this look is segmented
            return out
        if op.startswith("segment:") and self.seg is not None and args[1] is self.crop:
            self.counts["seg_hit"] += 1
            sf, self.seg = self.seg, None  # one call per look: a cascade's confirm on the same crop is computed
            return sf.result()
        return fn(*args, **kw)

    def _ahead(self, job, pyr, now: tuple) -> None:
        """Queue the next likely looks, as the agent's loop would choose and place them; drop cached looks no longer
        among them (a queued one is cancelled, a running one finishes unread)."""
        size, model, s = job.settings.size, job.settings.model, job.settings
        nxt = [job.forced[0][1:3]] if job.forced else []
        live = [r for r in job.regions.values() if not r.done and not r.looks]
        scored = sorted(((job._score(r), r) for r in live), key=lambda t: (-t[0][0], -t[0][2], t[1].cy, t[1].cx))
        nxt += [(r, r.next_level) for sc, r in scored if sc[2] > 0]
        keys: list[tuple] = []
        for r, level in nxt:
            level = min(level + s.level_bias, job.max_level)
            key = (*job._window(r.cx, r.cy, level, size)[:2], level, size, model)
            if key != now and key not in keys:
                keys.append(key)
            if len(keys) == self.k + bool(job.forced):
                break
        for key in set(self.cache) - set(keys):
            self.cache.pop(key)[0].cancel()
            self.counts["dropped"] += 1
        wh, seg = job.image.shape[1::-1], self.segmenter or job.segmenter
        for key in keys:
            if key not in self.cache:
                wf, sf = Future(), Future() if self.segment else None
                self.cache[key] = (self.pool.submit(_work, pyr, key, wh, seg, wf, sf), wf, sf)
                self.counts["submitted"] += 1

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.cache.clear()


def _work(pyr, key, wh, seg, wf: Future, sf: Future | None) -> None:
    """The look `key` exactly as the agent's loop computes it: gaze, then segment_view inside the view's footprint."""
    cx, cy, level, size, model = key
    try:
        crop, to_l0 = imaging.gaze(pyr, cx, cy, level, size=size)
        wf.set_result((crop, to_l0))
        if sf is not None:
            sf.set_result(segment_view(seg, crop, model, imaging.footprint(to_l0, wh, size), 2**level))
    except Exception as exc:  # noqa: BLE001 -- re-raised by .result() in the agent, where the inline call would raise
        for f in (wf, sf):
            if f is not None and not f.done():
                f.set_exception(exc)
