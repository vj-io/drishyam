"""Look prefetching (drishyam.prefetch) changes when work runs, never what the agent decides."""

import json
from collections import Counter
from pathlib import Path

import cv2

from drishyam.agent import Job, JobConfig
from drishyam.prefetch import Prefetcher
from evals import synth
from tests.conftest import FakeClock

WALL_CLOCK = {"t", "cpu_pct", "created"}  # ISO timestamps and process CPU share: the only fields not on the job's clock


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in WALL_CLOCK}
    return [_strip(v) for v in x] if isinstance(x, list) else x


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
          **{f"synth_{s}": (lambda s=s: synth.generate(s, 2048, 1536)[0]) for s in (9101, 9102)}}


def test_prefetch_leaves_every_decision_byte_identical():
    """Warp and segment prefetching (segment=False is the warp half of the same path) on bridge_03 and two synthetic
    photos: the whole event stream, timings included, is identical with it on and off."""
    counts = Counter()
    for name, load in IMAGES.items():
        img = load()
        base, events = _run(img)
        pf = Prefetcher(k=2, segment=True)
        job, events_pf = _run(img, pf)
        assert events_pf == events, name  # every event: gaze path, causes, rationales, findings, work order
        assert job.gaze_path == base.gaze_path and len(job.gaze_path) >= 3, name
        assert [f.to_dict() for f in job.findings] == [f.to_dict() for f in base.findings] and job.findings, name
        counts.update(pf.counts)
    assert counts["warp_hit"] > 0 and counts["seg_hit"] > 0, counts  # the cache was used: the identity is not vacuous
