"""Metric A: the pipeline's OpenCV vision kernels in isolation, ops/s and p50/p95 ms at three frame sizes, on one
thread and on all of them.

    python -m bench.kernels --arm local-macos-x86 [--threads 1,all] [--repeats 30] [--max-load 4]

Writes results/<arm>/kernels.json. The frames are synthetic bridge photos (evals.synth seed 9000, not an eval set) at
1024x768, 2048x1536 and 4000x3000. Kernels: the 5-level pyrDown pyramid, CLAHE on the L channel of LAB, the Scharr
gradient structure tensor (three Gaussian-smoothed products), 15 px black-hat plus top-hat, Canny, connected
components plus external contours with minAreaRect and fitLine on every contour of at least 50 points (of the
black-hat's top 1 %; `frame_facts` records how many), a 512x512 oblique gaze (warpPerspective, yaw 8, pitch 4: its output size is
fixed, so it barely moves with the frame), and blobFromImage of the frame to a 512x512 ImageNet-normalised tensor.
DNN forwards have fixed inputs, so they are timed once per thread count (`dnn`): NanoDet-Plus at 416, and the dacl10k
segmenter at 256 and 512 when DRISHYAM_ONNX (and DRISHYAM_ONNX_FP16) name it (bench.micro.dnn_ops, ENGINE_NEW).

`all` is OpenCV's own default for this process, read before anything sets it: the cgroup-aware CPU count on Linux.
macOS's GCD backend runs every count above 1 on all logical CPUs and cannot be held to 2 or 4 (bench.micro.set_threads
refuses such a count there), so a Mac measures 1 and all only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bench.micro import check_load, dnn_ops, provenance, set_threads, time_op
from drishyam import imaging
from evals import synth

ALL = cv2.getNumThreads()  # OpenCV's default thread count, before set_threads changes it
SIZES = ((1024, 768), (2048, 1536), (4000, 3000))
SEED = 9000  # bench.parity's seed: synthetic, outside every eval split
MIN_CONTOUR = 50
WARMUP = 3
MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)  # ImageNet, RGB order (swapRB)


def structure_tensor(gray: np.ndarray) -> list[np.ndarray]:
    gx, gy = cv2.Scharr(gray, cv2.CV_32F, 1, 0), cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return [cv2.GaussianBlur(cv2.multiply(a, b), (0, 0), 2.0) for a, b in ((gx, gx), (gx, gy), (gy, gy))]


def geometry(mask: np.ndarray) -> int:
    cv2.connectedComponentsWithStats(mask, connectivity=8)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = [c for c in cs if len(c) >= MIN_CONTOUR]
    for c in big:
        cv2.minAreaRect(c)
        cv2.fitLine(c, cv2.DIST_HUBER, 0, 0.01, 0.01)
    return len(big)


def kernels(img: np.ndarray) -> tuple[dict, dict]:
    """name -> zero-argument callable on one frame, and facts about the frame's inputs."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ell)
    mask = cv2.compare(bh, float(np.percentile(bh, 99)), cv2.CMP_GT)  # the strongest 1 %: thin dark lines, specks
    pyr = imaging.build_pyramid(img)
    table = {
        "pyramid": lambda: imaging.build_pyramid(img),
        "clahe_lab": lambda: imaging.enhance(img),
        "scharr_tensor": lambda: structure_tensor(gray),
        "blackhat_tophat": lambda: (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ell),
                                    cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ell)),
        "canny": lambda: cv2.Canny(gray, 50, 150),
        "components_contours": lambda: geometry(mask),
        "warp_gaze": lambda: imaging.gaze(pyr, w / 2, h / 2, 0, yaw=8.0, pitch=4.0, size=512),
        "blob": lambda: cv2.dnn.blobFromImage(img, 1 / 255, (512, 512), MEAN, swapRB=True),
    }
    return table, {"contours": geometry(mask), "mask_frac": round(float(np.count_nonzero(mask)) / mask.size, 4)}


def row(fn, repeats: int, **key) -> dict:
    r = time_op(fn, repeats, WARMUP)
    out = {**key, **{k: r[k] for k in ("n", "p50_ms", "p95_ms", "mean_ms", "min_ms")},
           "ops_per_s": round(1000 / r["mean_ms"], 2)}
    print(" ".join(f"{v}" for v in key.values()), f"p50 {r['p50_ms']:.2f} ms  p95 {r['p95_ms']:.2f} ms", flush=True)
    return out


def run(arm: str, threads: list[int], repeats: int = 30, out_dir="results", sizes=SIZES,
        max_load: float | None = None) -> dict:
    check_load(max_load)
    prov = provenance()
    frames = {f"{w}x{h}": synth.generate(SEED, w, h)[0] for w, h in sizes}
    rows, dnn, facts = [], [], {}
    for t in threads:
        got = set_threads(t)
        for size, img in frames.items():
            table, facts[size] = kernels(img)
            rows += [row(fn, repeats, kernel=name, size=size, threads=got) for name, fn in table.items()]
        dnn += [row(fn, repeats, model=name, size=f"{s}x{s}", threads=got) for name, (s, fn) in dnn_ops().items()]
    doc = {"arm": arm, "config": {"repeats": repeats, "warmup": WARMUP, "threads": threads, "all_threads": ALL,
                                  "sizes": list(frames), "frames": f"evals.synth.generate({SEED}, w, h)",
                                  "min_contour_points": MIN_CONTOUR, "frame_facts": facts},
           "provenance": prov, "results": rows, "dnn": dnn, "load_avg_end": check_load()}
    out = Path(out_dir) / arm / "kernels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.kernels", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, help="results/<arm>/kernels.json")
    ap.add_argument("--threads", default="1,all", help="comma-separated cv2 thread counts; all = OpenCV's default")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-load", type=float, help="exit before timing if the 1-min load average is at or above this")
    a = ap.parse_args(argv)
    threads = [ALL if t == "all" else int(t) for t in a.threads.split(",")]
    run(a.arm, threads, a.repeats, a.out_dir, max_load=a.max_load)


if __name__ == "__main__":
    main()
