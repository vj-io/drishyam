"""Crack geometry in pixels: length, mean width, area and orientation.

The centreline comes from hit-or-miss thinning (stock OpenCV imgproc ops; KleidiCV
HAL coverage of these calls is not measured); component labelling and contours are
plain imgproc. Length is the smoothed arc length of that 1-px line and
mean width is area / length. Everything is reported in L0 image pixels; there
are no millimetres anywhere in Drishyam.
"""

from __future__ import annotations

import cv2
import numpy as np

_BASE = ([[-1, -1, -1], [0, 1, 0], [1, 1, 1]], [[0, -1, -1], [1, 1, -1], [0, 1, 0]])
_THIN_KERNELS = [np.rot90(np.array(k, np.int8), r).copy() for k in _BASE for r in range(4)]


def thin(mask: np.ndarray) -> np.ndarray:
    """1-px 8-connected centreline of a 0/1 mask (classic hit-or-miss thinning)."""
    img = (mask > 0).astype(np.uint8)
    while True:
        before = cv2.countNonZero(img)
        for k in _THIN_KERNELS:
            img &= ~cv2.morphologyEx(img, cv2.MORPH_HITMISS, k)
        if cv2.countNonZero(img) == before:
            return img


def measure(mask: np.ndarray, scale: float = 1.0, linear: bool = True) -> dict:
    """Geometry of the largest component of `mask`, scaled from crop to L0 pixels.

    `linear` (cracks) measures length along the thinned centreline; for area defects
    (spalling, stains) length is the long side of the minimum-area rectangle, which
    skips thinning a large blob, by far the most expensive step.
    """
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return {"length_px": 0.0, "width_px": 0.0, "area_px": 0, "orientation_deg": 0.0}
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[biggest, :4]
    comp = (lab[y : y + h, x : x + w] == biggest).astype(np.uint8)  # thin inside the bounding box only
    area = int(stats[biggest, cv2.CC_STAT_AREA])
    (_, _), (rw, rh), angle = cv2.minAreaRect(cv2.findNonZero(comp))
    if linear:
        padded = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(thin(padded), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # A closed contour around a 1-px curve walks it twice; approxPolyDP removes the
        # staircase bias of 8-connected steps (about +8% on a 30-degree line otherwise).
        length = max(1.0, sum(cv2.arcLength(cv2.approxPolyDP(c, 1.0, True), True) for c in contours) / 2)
    else:
        length = max(1.0, float(max(rw, rh)))
    if rw < rh:
        angle += 90.0
    return {
        "length_px": round(length * scale, 1),
        "width_px": round(area / length * scale, 2),
        "area_px": round(area * scale * scale),
        "orientation_deg": round(angle % 180.0, 1),
    }
