import threading

import cv2
import numpy as np
import pytest


class FakeClock:
    """Advances 1 ms per read, so latencies depend only on call order: runs are deterministic."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.001
        return self.t


def concrete(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = sum(
        cv2.resize(rng.normal(0, 1, (max(2, h // s), max(2, w // s))).astype(np.float32), (w, h),
                   interpolation=cv2.INTER_CUBIC) * a
        for s, a in ((2, 4.0), (8, 5.0), (64, 8.0))
    )
    g = np.clip(170 + tex, 0, 255).astype(np.uint8)
    return cv2.merge([g, g, np.clip(g.astype(np.int16) + 3, 0, 255).astype(np.uint8)])


def add_crack(img, start, n=260, width=3, seed=1, step=(3, 1)):
    rng = np.random.default_rng(seed)
    pts = np.cumsum(rng.normal(0, 1.5, (n, 2)), axis=0) + start
    pts[:, 0] += np.arange(n) * step[0]
    pts[:, 1] += np.arange(n) * step[1]
    cv2.polylines(img, [pts.astype(np.int32)], False, (55, 55, 58), width, cv2.LINE_AA)
    return img


def add_rebar_spall(img, center, r=150):
    cx, cy = center
    rng = np.random.default_rng(3)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rad = r * (0.75 + 0.25 * rng.random(24))
    poly = np.stack([cx + rad * np.cos(ang), cy + 0.7 * rad * np.sin(ang)], 1).astype(np.int32)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    rough = (rng.normal(0, 28, img.shape[:2])).astype(np.float32)
    dark = np.clip(img[..., 0].astype(np.float32) - 60 + rough, 0, 255).astype(np.uint8)
    for c in range(3):
        img[..., c][mask > 0] = dark[mask > 0]
    for k in (-1, 1):
        y = cy + k * r // 4
        cv2.line(img, (cx - r // 2, y), (cx + r // 2, y), (35, 75, 150), 14, cv2.LINE_AA)
    return img


def threaded_mismatches(fn, inputs, threads=8, rounds=6) -> list[str]:
    """Calls of `fn` from `threads` threads at once (a barrier per round), each on a different one
    of `inputs` than its neighbour, whose result is not the single-threaded one. [] when `fn` is
    safe to share between request threads."""
    ref = [fn(x) for x in inputs]
    barrier, bad = threading.Barrier(threads), []

    def work(t):
        for r in range(rounds):
            k = (t + r) % len(inputs)
            barrier.wait()
            try:
                if fn(inputs[k]) != ref[k]:
                    bad.append(f"thread {t} round {r}: wrong result")
            except Exception as exc:  # noqa: BLE001 -- a raise under contention is a mismatch; a dead thread hangs the barrier
                bad.append(f"thread {t} round {r}: {exc!r:.120}")

    pool = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
    for th in pool:
        th.start()
    for th in pool:
        th.join()
    return bad


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture(scope="session")
def scene():
    img = concrete(1500, 2000, seed=4)
    add_crack(img, (300, 400), width=3)
    add_crack(img, (1100, 900), n=200, width=1, seed=7, step=(1, 2))
    add_rebar_spall(img, (1500, 450))
    return img
