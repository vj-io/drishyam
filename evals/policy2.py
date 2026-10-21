"""v1-policy2: the HARD subset of an evaluation set, fixed before any v1-policy2 run (results/dev/policy2_sweep.json).

A photo is HARD when either holds:
  small-critical  a ground-truth box of a critical class (exposed_rebar or spalling, the classes findings.is_critical
                  can make critical) covers under HARD_AREA_FRAC (1 %) of the photo's pixels: the defect a person must
                  see is small at the scale the agent first looks at;
  no-candidate    the overview proposes no candidate region (drishyam.imaging.find_candidates on the L3 saliency map,
                  the agent's own call): the zero-candidate case the fallback exists for.
Both read only the labels and the image itself (its saliency, a fixed OpenCV function), never any policy's output, so
the same photos are hard for every policy compared on them. The 1 % cut was set from the dev splits' label sizes only
(the 50th percentile of a critical-class box is 1.3 % of a synth_dev photo and 3.3 % of a CODEBRIM dev100 one): 21 of
60 synth_dev photos and 20 of 100 dev100 photos hold such a box. Large cracks are not a critical class here
(findings.is_critical never makes a crack critical), so they do not make a photo hard.
"""
from __future__ import annotations

import cv2

from drishyam import imaging

CRITICAL_CLASSES = ("exposed_rebar", "spalling")
HARD_AREA_FRAC = 0.01


def small_critical(rec: dict) -> list[dict]:
    """The photo's ground-truth boxes of a critical class under HARD_AREA_FRAC of its pixels."""
    area = rec["width"] * rec["height"]
    return [g for g in rec["boxes"] if g["cls"] in CRITICAL_CLASSES and g["box"][2] * g["box"][3] < HARD_AREA_FRAC * area]


def no_candidate(img) -> bool:
    """The overview proposes nothing: the same saliency and proposal call as Job.run (the window size only splits a
    salient blob into windows, so it cannot change whether there is one)."""
    pyr = imaging.build_pyramid(img)
    return not imaging.find_candidates(imaging.saliency(pyr[imaging.OVERVIEW_LEVEL]), imaging.OVERVIEW_LEVEL, 1024,
                                       img.shape[1::-1])


def hard_reasons(rec: dict, img=None) -> list[str]:
    """Why a photo is hard (empty: not hard). `img`: the decoded photo, else read from rec['path']."""
    img = cv2.imread(rec["path"], cv2.IMREAD_COLOR) if img is None else img
    return ["small-critical"] * bool(small_critical(rec)) + ["no-candidate"] * no_candidate(img)


def hard_subset(records: list[dict]) -> dict[str, list[str]]:
    """{photo id: reasons} for the hard photos of `records`, in record order."""
    return {r["id"]: why for r in records if (why := hard_reasons(r))}
