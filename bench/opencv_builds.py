"""The pipeline's own stages on other OpenCV builds against the locked 5.0 wheel, on one machine: what 5.0 itself
changes against the latest 4.x, and what the x86 wheel's IPP HAL does, in time and in output.

    uv run python -m bench.opencv_builds --arm local-macos-x86 [--other 4.14.0.94] [--rounds 24] [--repeats 5]

Variants, each a worker process (`python -m bench.opencv_builds --worker`, run from the repository root): `5.0` (this
environment's wheel, the reference), `5.0 null` (the same again: the noise floor), `5.0 IPP off` (the same with
cv2.ipp.setUseIPP(False); only where the build has IPP), and each --other opencv-python-headless version, installed
with this NumPy into runs/opencv_builds/<version>/ by `uv` (the first run downloads it). Every round runs each
variant once, the order rotated by one per round; a worker times each stage --repeats times on one thread and
reports its median and the sha256 of its output. The inputs are made once by the reference and saved, so a stage's
output differs only where the stage computes differently (decode excepted: its input is the JPEG). Per stage and
variant: the median ms over rounds, the ratio to 5.0 per round as a geometric mean with a 95 % bootstrap interval,
and whether the output equals 5.0's byte for byte.

Stages: decode (imaging.decode_image of bridge_03.jpg, 3000x2250), pyramid, saliency (L3), gaze (L1, straight) and
gaze_oblique (L0, yaw 8 pitch 4), quality and enhance (the L1 crop), segment (ClassicalSegmenter.predict, accurate),
measure (the crop's largest crack, measure.measure: thinning and contours), nanodet and yunet (privacy.py's two
detectors on the 640-px copy; on 5.0 readNet's ENGINE_AUTO picks the new engine, on 4.x the only one).
Writes results/<arm>/opencv_builds.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "runs" / "opencv_builds"
PHOTO = REPO / "public" / "samples" / "bridge_03.jpg"
BOOT = 2000
# What each build offers, by name (docs/bench.md's "new in 5.0" column is this, 5.0 against 4.x): DNN engines and
# profiling, element types, and calls the pipeline uses or tried.
API = ("dnn.ENGINE_NEW", "dnn.ENGINE_CLASSIC", "dnn.Net.setProfilingMode", "dnn.DNN_PROFILE_DETAILED",
       "dnn.Net.setTracingMode", "dnn.Net.getPerfProfile", "CV_Bool", "CV_32U", "CV_64S", "reduceArgMax", "compare",
       "hasNonZero", "findContoursLinkRuns", "ALGO_HINT_APPROX", "ipp.setUseIPP")


def has(module, name: str) -> bool:
    for part in name.split("."):
        if not hasattr(module, part):
            return False
        module = getattr(module, part)
    return True


def digest(x) -> str:
    h = hashlib.sha256()
    for a in x if isinstance(x, (list, tuple)) else [x]:
        h.update(np.ascontiguousarray(a).tobytes() if isinstance(a, np.ndarray) else json.dumps(a).encode())
    return h.hexdigest()[:16]


def stages(inp: dict) -> dict:
    """name -> zero-argument call on the saved inputs; each returns what its output digest is taken of."""
    from drishyam import imaging, measure, privacy
    from drishyam.segment import ClassicalSegmenter

    pyr, crop, mask, small = [inp[f"pyr{i}"] for i in range(5)], inp["crop"], inp["mask"], inp["small"]
    data = PHOTO.read_bytes()
    cx, cy = (int(v) for v in inp["centre"])
    det = privacy.detectors()
    return {"decode": lambda: imaging.decode_image(data),
            "pyramid": lambda: imaging.build_pyramid(pyr[0]),
            "saliency": lambda: imaging.saliency(pyr[3]),
            "gaze": lambda: imaging.gaze(pyr, cx, cy, 1)[0],
            "gaze_oblique": lambda: imaging.gaze(pyr, cx, cy, 0, yaw=8.0, pitch=4.0)[0],
            "quality": lambda: [imaging.quality(crop)],
            "enhance": lambda: imaging.enhance(crop),
            "segment": lambda: ClassicalSegmenter().predict(crop).probs,
            "measure": lambda: [measure.measure(mask, 2.0)],
            "nanodet": lambda: [det.objects(small)],
            "yunet": lambda: [det.faces(small)]}


def worker(inputs: str, repeats: int, ipp: bool) -> None:
    import cv2

    from bench.micro import build_lines, set_threads

    set_threads(1)
    if not ipp:
        cv2.ipp.setUseIPP(False)
    inp = dict(np.load(inputs))
    out = {"cv2": cv2.__version__, "numpy": np.__version__, "ipp": cv2.ipp.useIPP(), **build_lines(),
           "api": {name: has(cv2, name) for name in API}, "stages": {}}
    try:  # a NumPy bool array as an operation mask: CV_Bool
        cv2.copyTo(np.ones((2, 2), np.float32), np.ones((2, 2), bool))
        out["api"]["bool mask"] = True
    except cv2.error:
        out["api"]["bool mask"] = False
    for name, fn in stages(inp).items():
        res = fn()
        ms = []
        for _ in range(repeats):
            t = time.perf_counter_ns()
            fn()
            ms.append((time.perf_counter_ns() - t) / 1e6)
        out["stages"][name] = {"ms": float(np.median(ms)), "out": digest(res)}
    print(json.dumps(out))


def make_inputs(path: Path) -> None:
    """The reference's inputs for every worker: the decoded photo's pyramid, an L1 crop on a crack, its crack mask, the
    detectors' 640-px copy."""
    import cv2

    from drishyam import imaging
    from drishyam.segment import ClassicalSegmenter

    img = imaging.decode_image(PHOTO.read_bytes())
    pyr = imaging.build_pyramid(img)
    cx, cy = 2620, 1081  # the centre of bridge_03's crack box
    crop = imaging.gaze(pyr, cx, cy, 1)[0]
    mask = ClassicalSegmenter().predict(crop).mask("crack")
    k = 640 / max(img.shape[:2])
    small = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"pyr{i}": p for i, p in enumerate(pyr)}, crop=crop, mask=mask, small=small,
             centre=np.array([cx, cy]))


def env_python(version: str) -> str:
    """A Python with opencv-python-headless==version and this NumPy, made once under runs/opencv_builds/."""
    env = WORK / version
    py = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        subprocess.run(["uv", "venv", "--python", sys.executable, str(env)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(py), f"opencv-python-headless=={version}",
                        f"numpy=={np.__version__}"], check=True)
    return str(py)


def ratio(a, b) -> dict:
    d = np.log(np.asarray(a) / np.asarray(b))
    boots = d[np.random.default_rng(0).integers(0, len(d), (BOOT, len(d)))].mean(1)
    lo, hi = np.exp(np.percentile(boots, [2.5, 97.5]))
    return {"ratio": round(float(np.exp(d.mean())), 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}


def run(arm: str, others=("4.14.0.94",), rounds: int = 24, repeats: int = 5, out_dir="results",
        max_load: float | None = None) -> dict:
    import cv2

    from bench.micro import check_load, provenance

    check_load(max_load)
    prov = provenance()
    inputs = WORK / "inputs.npz"
    make_inputs(inputs)
    me = sys.executable
    variants = {"5.0": [me], "5.0 null": [me]}
    if cv2.ipp.useIPP():
        variants["5.0 IPP off"] = [me, "--no-ipp"]
    variants.update({v: [env_python(v)] for v in others})
    names, res = list(variants), {v: [] for v in variants}
    for r in range(rounds):
        for v in names[r % len(names):] + names[: r % len(names)]:
            cmd = [variants[v][0], "-m", "bench.opencv_builds", "--worker", str(inputs), "--repeats", str(repeats),
                   *variants[v][1:]]
            res[v].append(json.loads(subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                                    check=True).stdout.splitlines()[-1]))
        print(f"round {r + 1}/{rounds}", {v: round(sum(s["ms"] for s in res[v][-1]["stages"].values()), 1)
                                           for v in names}, flush=True)
    ref = res["5.0"]
    table = {}
    for v in names:
        first = res[v][0]
        row = {k: first[k] for k in first if k != "stages"} | {"stages": {}}
        for s in first["stages"]:
            ms = [x["stages"][s]["ms"] for x in res[v]]
            row["stages"][s] = {"p50_ms": round(float(np.median(ms)), 3),
                                "same_output": len({x["stages"][s]["out"] for x in res[v]} |
                                                   {x["stages"][s]["out"] for x in ref}) == 1,
                                **({} if v == "5.0" else {"vs_5.0": ratio(ms, [x["stages"][s]["ms"] for x in ref])})}
        table[v] = row
        print(v, {s: (x["p50_ms"], x.get("vs_5.0", {}).get("ratio"), x["same_output"]) for s, x in row["stages"].items()})
    doc = {"arm": arm, "config": {"rounds": rounds, "repeats": repeats, "threads": 1, "photo": str(PHOTO.relative_to(REPO)),
                                  "others": list(others), "boot": BOOT},
           "provenance": prov, "variants": table, "load_avg_end": check_load()}
    out = Path(out_dir) / arm / "opencv_builds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.opencv_builds", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm")
    ap.add_argument("--other", nargs="*", default=["4.14.0.94"], help="opencv-python-headless versions to compare")
    ap.add_argument("--rounds", type=int, default=24, help="a multiple of the variant count keeps the order balanced")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-load", type=float)
    ap.add_argument("--worker", metavar="INPUTS", help=argparse.SUPPRESS)
    ap.add_argument("--no-ipp", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        worker(a.worker, a.repeats, not a.no_ipp)
    else:
        if not a.arm:
            ap.error("--arm is required")
        run(a.arm, a.other, a.rounds, a.repeats, a.out_dir, a.max_load)


if __name__ == "__main__":
    main()
