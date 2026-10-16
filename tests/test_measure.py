import math

import cv2
import numpy as np
import pytest

from drishyam.measure import measure, thin


@pytest.mark.parametrize("thickness", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("angle", [0, 30, 45, 63, 90])
def test_straight_crack_within_10_percent(thickness, angle):
    L = 400
    m = np.zeros((600, 600), np.uint8)
    dx, dy = math.cos(math.radians(angle)) * L / 2, math.sin(math.radians(angle)) * L / 2
    cv2.line(m, (int(300 - dx), int(300 - dy)), (int(300 + dx), int(300 + dy)), 1, thickness)
    true_width = np.count_nonzero(m) / L  # what was actually rasterised
    r = measure(m)
    assert abs(r["length_px"] - L) / L < 0.10
    assert abs(r["width_px"] - true_width) / true_width < 0.10
    assert abs(((r["orientation_deg"] - angle) + 90) % 180 - 90) < 3


def test_curved_crack_and_scale():
    m = np.zeros((600, 700), np.uint8)
    pts = np.array([[50 + i, 300 + int(80 * math.sin(i / 60))] for i in range(500)], np.int32)
    cv2.polylines(m, [pts], False, 1, 4)
    true_len = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    r1 = measure(m)
    assert abs(r1["length_px"] - true_len) / true_len < 0.10
    r2 = measure(m, scale=2)  # a crop from pyramid level 1 reports L0 pixels
    assert r2["length_px"] == pytest.approx(2 * r1["length_px"], rel=1e-3)
    assert r2["area_px"] == pytest.approx(4 * np.count_nonzero(m), abs=1)


def test_largest_component_only_and_empty():
    m = np.zeros((200, 200), np.uint8)
    cv2.line(m, (10, 10), (190, 10), 1, 3)
    cv2.circle(m, (100, 120), 3, 1, -1)
    assert measure(m)["length_px"] > 150
    assert measure(np.zeros((50, 50), np.uint8))["area_px"] == 0


def test_thin_is_one_pixel_wide():
    m = np.zeros((100, 100), np.uint8)
    cv2.rectangle(m, (10, 40), (90, 60), 1, -1)
    sk = thin(m)
    assert 0 < np.count_nonzero(sk) < 130  # an 80 px centreline plus short corner spurs
    assert np.count_nonzero(sk[:, 50]) == 1
