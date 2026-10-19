"""Look prefetching (drishyam.prefetch) changes when work runs, never what the agent decides."""

import json
import threading
from collections import Counter
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from drishyam import prefetch
from drishyam.agent import Job, JobConfig
from drishyam.prefetch import Prefetcher
from evals import synth
from tests.conftest import FakeClock

WALL_CLOCK = {"t", "cpu_pct", "created"}  # ISO timestamps and process CPU share: the only fields not on the job's clock


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in WALL_CLOCK}
    return [_strip(v) for v in x] if isinstance(x, list) else x


class Spy(Prefetcher):
    """Checks each look served from the cache against the agent's own call for it: the warp's (crop, to_l0) against
    the inline gaze, and every input the background passed to segment_view (segmenter, crop, model, footprint, scale)
    against the ones the agent passes, so a wrong input fails even where it would not change this photo's output."""

    def __init__(self, seen: dict, **kw) -> None:
        super().__init__(**kw)
        self.seen, self.checked = seen, Counter()

    def __call__(self, job, op, fn, args, kw):
        if op == "warp":
            hits = self.counts["warp_hit"]
            out = super().__call__(job, op, fn, args, kw)
            if self.counts["warp_hit"] > hits:
                crop, to_l0 = fn(*args, **kw)
                assert np.array_equal(out[0], crop) and np.array_equal(out[1], to_l0)
                self.checked["warp"] += 1
            return out
        if op.startswith("segment:") and self.seg is not None and args[1] is self.crop:
            crop, seg, model, valid, scale = self.seen[id(args[1])]
            assert crop is args[1] and seg is args[0], op
            assert (model, scale) == (args[2], args[4]), ((model, scale), args[2::2])
            assert np.array_equal(valid, args[3]), "the image footprint differs"
            self.checked["segment"] += 1
        return super().__call__(job, op, fn, args, kw)


def _run(img, prefetch=None):
    """The default agent (governor on, live costs, thumbnails on) on a deterministic clock."""
    job = Job(img, JobConfig(), clock=FakeClock(), job_id="j_test")
    job.prefetch = prefetch
    try:
        return job, json.dumps(_strip(list(job.run())), sort_keys=True)
    finally:
        if prefetch:
            prefetch.close()


IMAGES = {"bridge_03": lambda: cv2.imread(str(Path(__file__).resolve().parent.parent / "public/samples/bridge_03.jpg")),
          **{f"synth_{s}": (lambda s=s: synth.generate(s, 2048, 1536)[0]) for s in (9101, 9102)},
          "synth_9102_wide": lambda: synth.generate(9102, 3000, 1000)[0]}  # a served look runs past the image's edge


def test_prefetch_leaves_every_decision_byte_identical(monkeypatch):
    """Warp and segment prefetching (segment=False is the warp half of the same path) on bridge_03 and three synthetic
    photos: the whole event stream, timings included, is identical with it on and off, and every look served from the
    cache was computed from the agent's own inputs."""
    seen, real = {}, prefetch.segment_view

    def record(seg, crop, model, valid, scale=1.0):  # the background's call, keyed by its crop (kept alive here)
        seen[id(crop)] = (crop, seg, model, valid, scale)
        return real(seg, crop, model, valid, scale)

    monkeypatch.setattr(prefetch, "segment_view", record)
    counts, checked = Counter(), Counter()
    for name, load in IMAGES.items():
        img = load()
        base, events = _run(img)
        pf = Spy(seen, k=2, segment=True)
        job, events_pf = _run(img, pf)
        assert events_pf == events, name  # every event: gaze path, causes, rationales, findings, work order
        assert job.gaze_path == base.gaze_path and len(job.gaze_path) >= 3, name
        assert [f.to_dict() for f in job.findings] == [f.to_dict() for f in base.findings] and job.findings, name
        counts.update(pf.counts)
        checked.update(pf.checked)
    assert counts["warp_hit"] > 0 and counts["seg_hit"] > 0, counts  # the cache was used: the identity is not vacuous
    assert checked["warp"] == counts["warp_hit"] and checked["segment"] == counts["seg_hit"], (checked, counts)


def test_a_look_still_queued_is_computed_inline_not_waited_for():
    """One worker, busy with another look: the asked-for look has not started, so the agent computes it inline (a
    `late` miss) instead of waiting behind the other look's warp and forward."""
    pf = Prefetcher(k=2)
    pf._ahead = lambda *a: None  # only the hit path is under test
    gate = threading.Event()
    timer = threading.Timer(2.0, gate.set)  # without the cancel, the agent would wait this long for the cached look
    timer.start()
    pf.pool.submit(gate.wait)
    wf = Future()
    key = (100, 80, 1, 512, "fast")
    pf.cache[key] = (pf.pool.submit(wf.set_result, ("cached", None)), wf, Future())
    out = pf(SimpleNamespace(settings=SimpleNamespace(model="fast")), "warp", lambda *a, **kw: ("inline", None),
             (None, 100, 80, 1), {"size": 512})
    gate.set()
    timer.cancel()
    pf.close()
    assert out == ("inline", None) and pf.seg is None, out
    assert pf.counts == Counter(late=1, warp_miss=1), pf.counts
