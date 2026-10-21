"""OpenCV calls in Drishyam's own pipeline against the forms they replace: whether the agent's traces stay the same
to the byte, and what each costs; with the v1 weights, v1's time per layer on both DNN engines (OpenCV 5's
profiling mode) and the agent with its FP32 net on ENGINE_CLASSIC.

    uv run python -m bench.opencv5_features --arm local-macos-x86 [--threads 1,all] [--rounds 21] [--looks 24]
        [--dev 4] [--onnx models/drishyam_dacl10k.onnx models/drishyam_dacl10k.fp16.onnx | --onnx none] [--max-load L]

Photos (the identity set): every demo sample (public/samples: bridge_01-08, synthetic; real_01-02, real photos) and
the first --dev images of the synth_dev split as `make eval-dev` builds it (evals.synth seed 4242: 2048x1536, JPEG
quality 92), each decoded as an upload is (imaging.decode_image). No CODEBRIM image.

- `identity`: the agent (JobConfig() defaults, a clock that advances 1 ms per read, thumbnails on) on every photo
  with drishyam.segment's flags all off (the NumPy forms), off again (the null), each flag alone and all of them; its
  event stream, with the wall-clock fields stripped as tests/test_prefetch.py strips them, must be identical to the
  byte. Classical engine, and v1 (ENGINE_NEW) when its files are there.
- `timing`: each flag's unit on the looks those runs made (every k-th, --looks of them): finish() on the
  segmenter's probabilities for CV_ARGMAX and CV_COMPARE, restrict() on the looks that run past the image for
  CV_BOOL_MASKS and CV_COMPARE, and the whole segment_view() for all three at once (classical engine only: a v1
  look is its forward, which the `engine` part times); off is all three off, on is
  that flag alone (or all three). --rounds rounds, each running off, on and off again (the null slot) once over
  every input, in an order rotated by one per round; per round on/off and null/off, reported as the geometric mean
  with a 95 % bootstrap interval over the rounds (paired: a round's variants share the machine's load).
- `profile` (v1): the FP32 net at 512 and 256 on ENGINE_NEW with Net.setProfilingMode(DNN_PROFILE_DETAILED) and on
  ENGINE_CLASSIC (a static-shape copy, segment._classic), --rounds forwards each, interleaved; every layer's time
  from getPerfProfile(), its median over rounds, summed by what the layer is (from the ONNX graph: depthwise or
  other convolution and its kernel, or the op type; a layer the new engine adds is named by its own type).
- `engine` (v1): the agent with OnnxSegmenter(engine="classic") against engine="new" on every photo: identical
  traces or not, and on every look the new engine made, max |dprob| and the pixels whose argmax differs; and the
  accurate looks' segment_view() timed in rounds as above (null: the new engine again).

Writes results/<arm>/opencv5_features.json with bench.micro's provenance block.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from bench.micro import check_load, provenance, set_threads
from bench.opencv5_engines import blobs, opencv_log
from bench.parity import make_crops
from drishyam import agent, imaging, segment
from drishyam.agent import Job, JobConfig
from drishyam.segment import ClassicalSegmenter, OnnxSegmenter
from evals import synth

REPO = Path(__file__).resolve().parent.parent
ALL = cv2.getNumThreads()
V1 = ("models/drishyam_dacl10k.onnx", "models/drishyam_dacl10k.fp16.onnx")
FLAGS = ("CV_ARGMAX", "CV_COMPARE", "CV_BOOL_MASKS")
UNITS = {"CV_ARGMAX": ("finish",), "CV_COMPARE": ("finish", "restrict"), "CV_BOOL_MASKS": ("restrict",),
         "all": ("segment_view",)}
DEV_SEED, DEV_N = 4242, 60  # Makefile eval-dev: data/synth_dev, `evals.synth dataset --n 60 --seed 4242`
WALL_CLOCK = {"t", "cpu_pct", "created"}  # tests/test_prefetch.py: the only fields not on the job's clock
BOOT = 2000


class Clock:
    """1 ms per read (tests.conftest.FakeClock): a run's latencies depend only on call order."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.001
        return self.t


def photos(dev: int) -> dict[str, np.ndarray]:
    out = {p.stem: imaging.decode_image(p.read_bytes()) for p in sorted((REPO / "public" / "samples").glob("*.jpg"))}
    for i, spec in enumerate(synth.stratify(DEV_N, DEV_SEED)[:dev]):
        img = synth.render(DEV_SEED * 100_000 + i, 2048, 1536, spec)[0]
        out[f"synth_dev/s{i + 1:04d}"] = imaging.decode_image(
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
    return out


@contextlib.contextmanager
def flags(on=()):
    """drishyam.segment's flags: those in `on` True, the rest False; restored after."""
    saved = {f: getattr(segment, f) for f in FLAGS}
    for f in FLAGS:
        setattr(segment, f, f in on)
    try:
        yield
    finally:
        for f, v in saved.items():
            setattr(segment, f, v)


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in WALL_CLOCK}
    return [strip(v) for v in x] if isinstance(x, list) else x


def trace(img: np.ndarray, seg, looks: list | None = None) -> str:
    """The agent's event stream on `img`, wall-clock fields stripped; `looks` collects each segment_view call's
    (crop, model, footprint, scale)."""
    real = agent.segment_view

    def record(s, crop, model, valid, scale=1.0):
        looks.append((crop, model, valid, scale))
        return real(s, crop, model, valid, scale)

    if looks is not None:
        agent.segment_view = record
    try:
        return json.dumps(strip(list(Job(img, JobConfig(), clock=Clock(), job_id="j_bench", segmenter=seg).run())),
                          sort_keys=True)
    finally:
        agent.segment_view = real


def identity(seg, imgs: dict) -> tuple[dict, list]:
    """Each variant's traces against all flags off, photo by photo; and the looks the off runs made."""
    variants = {"null": (), **{f: (f,) for f in FLAGS}, "all": FLAGS}
    looks, same, events = [], {v: [] for v in variants}, 0
    for name, img in imgs.items():
        with flags():
            base = trace(img, seg, looks)
        events += len(json.loads(base))
        for v, on in variants.items():
            with flags(on):
                if trace(img, seg) == base:
                    same[v].append(name)
        print(f"identity {seg.name:32s} {name:20s}", {v: name in s for v, s in same.items()}, flush=True)
    return {"photos": len(imgs), "events": events, "looks": len(looks),
            "identical": {v: len(s) for v, s in same.items()},
            "differ": {v: sorted(set(imgs) - set(s)) for v, s in same.items()}}, looks


def interleaved(fns: dict, rounds: int, warmup: int = 1) -> dict[str, np.ndarray]:
    """ms per call of each zero-argument fn, `rounds` rounds; a round calls each once, the order rotated by one."""
    names = list(fns)
    for f in fns.values():
        for _ in range(warmup):
            f()
    ms = {k: np.empty(rounds) for k in names}
    for r in range(rounds):
        for k in names[r % len(names):] + names[: r % len(names)]:
            t = time.perf_counter_ns()
            fns[k]()
            ms[k][r] = (time.perf_counter_ns() - t) / 1e6
    return ms


def ratio(a: np.ndarray, b: np.ndarray) -> dict:
    """a/b per round: the geometric mean and its 95 % bootstrap interval over rounds (seeded)."""
    d = np.log(a / b)
    boots = d[np.random.default_rng(0).integers(0, len(d), (BOOT, len(d)))].mean(1)
    lo, hi = np.exp(np.percentile(boots, [2.5, 97.5]))
    return {"ratio": round(float(np.exp(d.mean())), 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}


def compare(ms: dict, ref: str = "off") -> dict:
    out = {f"{k}_p50_ms": round(float(np.median(v)), 3) for k, v in ms.items()}
    return out | {k: ratio(v, ms[ref]) for k, v in ms.items() if k != ref}


def timing(seg, looks: list, n_looks: int, rounds: int, threads: list[int], whole_look: bool = True) -> dict:
    """Each flag's units (UNITS) on every k-th look: off (all flags off), on (that flag, or all), null (off again).
    `whole_look` False leaves segment_view out (v1's look is its forward: the engine part times that)."""
    every = lambda xs: xs[:: max(1, len(xs) // n_looks)][:n_looks]
    picked, rim = every(looks), every([lk for lk in looks if not lk[2].all()])  # rim: the looks past the image
    with flags():
        raw = [(seg.predict(crop, model), valid) for crop, model, valid, _ in picked]
        edge = [(seg.predict(crop, model), valid) for crop, model, valid, _ in rim]
    inputs = {"finish": [(s.probs, s.model) for s, _ in raw], "restrict": edge,
              "segment_view": picked if whole_look else []}
    call = {"finish": lambda xs: [segment.finish(p, 0.0, m) for p, m in xs],
            "restrict": lambda xs: [segment.restrict(s, v) for s, v in xs],
            "segment_view": lambda xs: [segment.segment_view(seg, c, m, v, k) for c, m, v, k in xs]}
    out = {}
    for flag, units in UNITS.items():
        on = FLAGS if flag == "all" else (flag,)
        out[flag] = {}
        for unit in units:
            xs = inputs[unit]
            row = {"inputs": len(xs)}
            if xs:
                for t in threads:
                    got = set_threads(t)

                    def run(f, xs=xs, unit=unit):
                        def go():
                            with flags(f):
                                call[unit](xs)
                        return go
                    row[f"threads{got}"] = compare(interleaved({"off": run(()), "on": run(on), "null": run(())}, rounds))
                    print(f"timing {seg.name:32s} {flag:13s} {unit:12s} threads {got}", row[f"threads{got}"], flush=True)
            out[flag][unit] = row
    set_threads(ALL)
    return out


def kinds(path: str) -> dict[str, str]:
    """ONNX node name -> what it computes: 'depthwise conv 3x3', 'conv 1x1' and so on, or its op type."""
    import onnx

    g = onnx.load(path, load_external_data=False).graph
    dims = {i.name: list(i.dims) for i in g.initializer}
    out = {}
    for n in g.node:
        out[n.name] = n.op_type
        if n.op_type == "Conv" and (w := dims.get(n.input[1])):
            group = next((a.i for a in n.attribute if a.name == "group"), 1)
            out[n.name] = f"{'depthwise ' if group > 1 and w[1] == 1 else ''}conv {'x'.join(map(str, w[2:]))}"
    return out


def profile(path: str, sizes, threads: list[int], rounds: int) -> dict:
    """v1's time per layer on ENGINE_NEW (profiling mode DETAILED) and ENGINE_CLASSIC (static copy), by layer kind."""
    kind, out = kinds(path), {}
    for size in sizes:
        log = []
        with opencv_log(log):
            new = cv2.dnn.readNetFromONNX(path, engine=cv2.dnn.ENGINE_NEW)
            new.setProfilingMode(cv2.dnn.DNN_PROFILE_DETAILED)
            nets = {"new": new, "classic": segment._classic(path, size)}
        blob = blobs(make_crops(1), size)[0]
        label = {}
        for e, net in nets.items():
            names = net.getLayerNames()
            label[e] = [kind.get(n.removeprefix("onnx_node!"), net.getLayer(net.getLayerId(n)).type) for n in names]
            net.setInput(blob)
        out[size] = {}
        for t in threads:
            got = set_threads(t)
            for net in nets.values():
                net.forward()
            wall, layers = {e: [] for e in nets}, {e: [] for e in nets}
            for r in range(rounds):
                for e in list(nets)[r % 2:] + list(nets)[: r % 2]:
                    t0 = time.perf_counter_ns()
                    nets[e].forward()
                    wall[e].append((time.perf_counter_ns() - t0) / 1e6)
                    layers[e].append(np.asarray(nets[e].getPerfProfile()[1], float).ravel() * 1000 / cv2.getTickFrequency())
            row = {}
            for e in nets:
                med = np.median(layers[e], axis=0)
                by = {}
                for k, v in zip(label[e], med, strict=True):
                    n, ms = by.get(k, (0, 0.0))
                    by[k] = (n + 1, ms + float(v))
                row[e] = {"layers": len(med), "wall_p50_ms": round(float(np.median(wall[e])), 2),
                          "profiled_ms": round(float(med.sum()), 2),
                          "by_kind": {k: {"layers": n, "ms": round(ms, 2)}
                                      for k, (n, ms) in sorted(by.items(), key=lambda kv: -kv[1][1])}}
            out[size][f"threads{got}"] = row
            print(f"profile {size} threads {got}", {e: (r["wall_p50_ms"], r["profiled_ms"]) for e, r in row.items()},
                  flush=True)
        out[size]["log"] = log
    set_threads(ALL)
    return out


def engine(files, imgs: dict, n_looks: int, rounds: int, threads: list[int]) -> dict:
    """The agent on ENGINE_CLASSIC (engine="classic") against ENGINE_NEW: traces, per-look parity and look time."""
    segs = {e: OnnxSegmenter(files[0], files[1] if len(files) > 1 else None, engine=e) for e in ("new", "classic")}
    same, looks = [], []
    for name, img in imgs.items():
        mine = []
        base, other = trace(img, segs["new"], mine), trace(img, segs["classic"])
        same += [name] if base == other else []
        looks += mine
        print(f"engine {name:20s} identical {base == other}", flush=True)
    worst, flipped, px = 0.0, 0, 0
    for crop, model, _, _ in looks:
        a, b = (segs[e].predict(crop, model) for e in ("new", "classic"))
        worst = max(worst, float(np.abs(a.probs - b.probs).max()))
        flipped += int(np.count_nonzero(a.labels != b.labels))
        px += a.labels.size
    acc = [lk for lk in looks if lk[1] == "accurate"]
    picked = acc[:: max(1, len(acc) // n_looks)][:n_looks]
    view = lambda e: lambda: [segment.segment_view(segs[e], c, m, v, k) for c, m, v, k in picked]
    out = {"photos": len(imgs), "identical": len(same), "differ": sorted(set(imgs) - set(same)), "looks": len(looks),
           "accurate_looks": len(acc), "max_dprob": worst, "argmax_differs_px": flipped, "pixels": px,
           "timed_looks": len(picked)}
    for t in threads:
        got = set_threads(t)
        out[f"threads{got}"] = compare(interleaved({"off": view("new"), "on": view("classic"), "null": view("new")},
                                                   rounds))
        print(f"engine timing threads {got}", out[f"threads{got}"], flush=True)
    set_threads(ALL)
    return out


def run(arm: str, threads: list[int], rounds: int = 21, n_looks: int = 24, dev: int = 4, files=V1,
        out_dir="results", max_load: float | None = None, sizes=(512, 256)) -> dict:
    check_load(max_load)
    prov = provenance()
    imgs = photos(dev)
    files = [f for f in files if Path(f).is_file()] if files else []
    segs = {"classical": ClassicalSegmenter()}
    if files:
        segs["v1"] = OnnxSegmenter(files[0], files[1] if len(files) > 1 else None, engine="new")
    doc = {"arm": arm, "config": {"threads": threads, "all_threads": ALL, "rounds": rounds, "looks": n_looks,
                                  "photos": list(imgs), "dev": f"evals.synth seed {DEV_SEED}, first {dev} of {DEV_N}",
                                  "onnx": files, "defaults": {f: getattr(segment, f) for f in FLAGS},
                                  "units": UNITS, "boot": BOOT},
           "provenance": prov, "identity": {}, "timing": {}}
    for name, seg in segs.items():
        doc["identity"][name], looks = identity(seg, imgs)
        doc["timing"][name] = timing(seg, looks, n_looks, rounds, threads, whole_look=name == "classical")
    if files:
        doc["profile"] = profile(files[0], sizes, threads, max(3, rounds // 2))
        doc["engine"] = engine(files, imgs, max(1, n_looks // 3), max(3, rounds // 2), threads)
    doc["load_avg_end"] = check_load()
    out = Path(out_dir) / arm / "opencv5_features.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.opencv5_features", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--threads", default="1,all", help="comma-separated cv2 thread counts; all = OpenCV's default")
    ap.add_argument("--rounds", type=int, default=21, help="a multiple of 3 keeps off, on and null balanced")
    ap.add_argument("--looks", type=int, default=24, help="looks each unit is timed on")
    ap.add_argument("--dev", type=int, default=4, help="synth_dev images in the identity set")
    ap.add_argument("--onnx", nargs="+", default=list(V1), help="v1 FP32 [FP16]; 'none' skips the v1 parts")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-load", type=float)
    a = ap.parse_args(argv)
    threads = [ALL if t == "all" else int(t) for t in a.threads.split(",")]
    run(a.arm, threads, a.rounds, a.looks, a.dev, [] if a.onnx == ["none"] else a.onnx, a.out_dir, a.max_load)


if __name__ == "__main__":
    main()
