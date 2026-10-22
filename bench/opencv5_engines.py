"""OpenCV 5's two DNN engines on the v1 segmenter: which reads it, how fast each runs it, and whether they agree.

    uv run python -m bench.opencv5_engines --arm local-macos-x86 [--onnx A.onnx B.onnx] [--repeats 20] [--crops 8]

Every ONNX given (default: the v1 release files models/drishyam_dacl10k.onnx and .fp16.onnx, never committed), plus a
static-shape copy of each per input size, written to --work (runs/opencv5_engines/, git-ignored): the graph input's
symbolic H and W set to that size and nothing else changed. Each is read with cv2.dnn.readNetFromONNX(path,
engine=ENGINE_CLASSIC) and (engine=ENGINE_NEW), the one the agent uses; a failure to load or to run a forward is
recorded with the exact exception and the lines OpenCV logged. For each pair that runs, at 512x512 and 256x256 (the
agent's accurate and fast sizes): forward p50/p95 on 1 thread and all threads, timed in interleaved rounds (one forward
of every net per round, so a change in load lands on all of them) with the reference net in a second slot as well
(`null_forward_threads*`: the noise floor), and parity, the max |dprob| (softmax
over the class axis) and argmax agreement over --crops synthetic crops (bench.parity.make_crops, preprocessed as
OnnxSegmenter does a square crop) against the first file on ENGINE_NEW (the deployed net) and against the same file
on the other engine. Also each file's symbolic input dims and its Shape nodes, the ones ENGINE_CLASSIC stops at.
Writes results/<arm>/opencv5_engines.json with bench.micro's provenance block (getBuildInformation's Baseline,
Dispatched code generation, Parallel framework and Custom HAL lines verbatim).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import onnx

from bench.int8 import tensor
from bench.micro import check_load, provenance, set_threads
from bench.parity import make_crops

ALL = cv2.getNumThreads()
ENGINES = {"new": cv2.dnn.ENGINE_NEW, "classic": cv2.dnn.ENGINE_CLASSIC}  # new first: the reference row comes first
SIZES = (512, 256)
V1 = ("models/drishyam_dacl10k.onnx", "models/drishyam_dacl10k.fp16.onnx")


@contextlib.contextmanager
def opencv_log(lines: list):
    """Collect what OpenCV writes to fd 2 (its logger) while the block runs, one stripped line each."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as f:
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            f.seek(0)
            lines += [ln.strip() for ln in f.read().decode(errors="replace").splitlines() if ln.strip()]


def graph(path: str) -> dict:
    """The input dims as declared (a name where symbolic) and each Shape node with the op chain to the first Resize."""
    g = onnx.load(path, load_external_data=False).graph
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    chains = []
    for n in (n for n in g.node if n.op_type == "Shape"):
        ops, cur = [], n
        while cur.op_type != "Resize" and users.get(cur.output[0]):
            cur = users[cur.output[0]][0]
            ops.append(cur.op_type)
        chains.append({"node": n.name, "input": n.input[0], "chain": " -> ".join(["Shape", *ops])})
    return {"inputs": {i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in g.input},
            "nodes": len(g.node), "shape_nodes": chains}


def static_copy(path: str, size: int, work: Path) -> str:
    """`path` with every symbolic input dim after the batch set to `size`; nothing else in the graph changes."""
    m = onnx.load(path)
    for d in m.graph.input[0].type.tensor_type.shape.dim[2:]:
        if d.dim_param:
            d.dim_value = size  # assigning the value clears dim_param (a oneof)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{Path(path).name.removesuffix('.onnx')}.static{size}.onnx"
    onnx.save(m, out)
    return str(out)


def blobs(crops: np.ndarray, size: int) -> list[np.ndarray]:
    """Square crops as OnnxSegmenter feeds them at `size`: INTER_AREA to it, RGB, /255, ImageNet mean and std."""
    return [tensor(c if c.shape[0] == size else cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA))
            for c in crops]


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)


def probs(net, inputs: list[np.ndarray]) -> np.ndarray:
    out = []
    for b in inputs:
        net.setInput(b)
        out.append(softmax(np.asarray(net.forward())[0].astype(np.float64)).astype(np.float32))
    return np.stack(out)


def parity(p: np.ndarray, ref: np.ndarray) -> dict:
    return {"max_dprob": float(np.abs(p - ref).max()),
            "argmax_agreement": round(float((p.argmax(1) == ref.argmax(1)).mean()), 6)}


def attempt(path: str, engine: str, inputs: list[np.ndarray]) -> tuple[dict, object, np.ndarray | None]:
    """(row, net, softmax outputs); on a failure the net is None and the row says why and whether it loaded."""
    log, net = [], None
    try:
        with opencv_log(log):
            net = cv2.dnn.readNetFromONNX(path, engine=ENGINES[engine])
        with opencv_log(log):
            p = probs(net, inputs)
    except cv2.error as exc:
        return {"loads": net is not None, "runs": False, "error": str(exc).strip(), "log": log}, None, None
    return {"loads": True, "runs": True, "log": log, "output_shape": list(p.shape[1:])}, net, p


def interleaved(nets: list, blob: np.ndarray, repeats: int, warmup: int = 3) -> list[dict]:
    """Forward latency of each net, timed in rounds (one forward of every net per round) so that a change in the
    machine's load lands on all of them alike: [{n, p50_ms, p95_ms, mean_ms}] in the order given."""
    for net in nets:
        net.setInput(blob)
        for _ in range(warmup):
            net.forward()
    ms = np.empty((repeats, len(nets)))
    for r in range(repeats):
        for i, net in enumerate(nets):
            t = time.perf_counter_ns()
            net.forward()
            ms[r, i] = (time.perf_counter_ns() - t) / 1e6
    pct = lambda c, q: round(float(np.percentile(c, q)), 3)
    return [{"n": repeats, "p50_ms": pct(c, 50), "p95_ms": pct(c, 95), "mean_ms": round(float(c.mean()), 3)}
            for c in ms.T]


def run(arm: str, files=V1, repeats: int = 20, crops: int = 8, out_dir="results", work="runs/opencv5_engines",
        sizes=SIZES, max_load: float | None = None) -> dict:
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        raise SystemExit(f"missing {missing}: the v1 files are release assets (gh release download v1.1.0 "
                         "--repo vj-io/drishyam --pattern 'drishyam_dacl10k*' --dir models)")
    check_load(max_load)
    prov = provenance()
    meta = {}
    for f in files:
        with open(f, "rb") as fh:
            meta[Path(f).name] = {"path": f, "bytes": os.path.getsize(f),
                                  "sha256": hashlib.file_digest(fh, "sha256").hexdigest(), "graph": graph(f)}
    ref_name = Path(files[0]).name
    images = make_crops(crops)
    set_threads(ALL)
    rows = []
    for size in sizes:
        inputs, ref, timed = blobs(images, size), None, []
        for f in files:
            for variant, path in ((Path(f).name, f), (Path(f).name + " static", static_copy(f, size, Path(work)))):
                outs = {}
                for engine in ENGINES:
                    row, net, p = attempt(path, engine, inputs)
                    row = {"file": variant, "path": path, "engine": engine, "size": size, **row}
                    if p is not None:
                        if (variant, engine) == (ref_name, "new"):
                            ref = p
                            timed.append((row, "null_", net))  # the same net in a second slot: the noise floor
                        row["vs_reference"] = "reference" if p is ref else None if ref is None else parity(p, ref)
                        outs[engine] = p
                        timed.append((row, "", net))
                    rows.append(row)
                    said = "runs" if p is not None else " ".join(row["error"].split())[-160:]
                    print(f"{variant:34s} {engine:7s} {size:4d}", said, json.dumps(row.get("vs_reference")), flush=True)
                if len(outs) == 2:
                    for r in rows[-2:]:
                        r["vs_other_engine"] = parity(outs["classic"], outs["new"])
        for t in (1, ALL):
            got = set_threads(t)
            for (row, key, _), ms in zip(timed, interleaved([net for _, _, net in timed], inputs[0], repeats),
                                         strict=True):
                row[f"{key}forward_threads{got}"] = ms
                print(f"{row['file']:34s} {row['engine']:7s} {size:4d} {key}forward_threads{got}", ms, flush=True)
        set_threads(ALL)
    doc = {"arm": arm, "config": {"repeats": repeats, "warmup": 3, "crops": crops, "sizes": list(sizes),
                                  "all_threads": ALL, "reference": f"{ref_name} on ENGINE_NEW",
                                  "timing": "interleaved: each round runs one forward of every net at that size",
                                  "inputs": "bench.parity.make_crops (evals.synth seeds 9000 on, 512 px), INTER_AREA "
                                            "to the size, as OnnxSegmenter preprocesses a square crop",
                                  "static_copies": str(work)},
           "provenance": prov, "load_avg_end": check_load(), "files": meta, "rows": rows,
           "onnx_runtime_in_build": next((ln.split(":", 1)[1].strip() for ln in cv2.getBuildInformation().splitlines()
                                          if ln.strip().startswith("ONNX Runtime:")), None)}  # ENGINE_ORT needs YES
    out = Path(out_dir) / arm / "opencv5_engines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.opencv5_engines", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--onnx", nargs="+", default=list(V1), help="the reference first (default: the v1 files)")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--crops", type=int, default=8)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--work", default="runs/opencv5_engines", help="where the static-shape copies are written")
    ap.add_argument("--max-load", type=float)
    a = ap.parse_args(argv)
    run(a.arm, a.onnx, a.repeats, a.crops, a.out_dir, a.work, max_load=a.max_load)


if __name__ == "__main__":
    main()
