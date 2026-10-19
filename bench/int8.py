"""INT8 versions of the dacl10k segmenter, and whether OpenCV 5's DNN can run them.

    DRISHYAM_ONNX=/path/drishyam_dacl10k.onnx uv run --with onnxruntime python -m bench.int8 --arm local-macos-x86

Quantizes the FP32 ONNX (DRISHYAM_ONNX or --onnx; the weights are never committed) with onnxruntime.quantization into
runs/int8/ (git-ignored): `quantize_dynamic` (int8 weights, activations quantized at run time), and two static models
calibrated on 8 synthetic crops, the QDQ format asked for (QuantizeLinear/DequantizeLinear pairs) and, for reference,
the QOperator format (QLinearConv and friends). Then, with OpenCV alone, it reads every model with ENGINE_NEW and
ENGINE_CLASSIC and records the exact error where reading or a forward fails; for each pair that runs: 512x512 forward
p50/p95 at 1 thread and all threads, and agreement with FP32 on ENGINE_NEW over 20 other synthetic crops
(bench.parity.compare: pixel agreement and per-class IoU of the label maps, through the agent's own OnnxSegmenter pre-
and post-processing). onnxruntime is a bench tool here, never on the live path. Writes results/<arm>/int8.json.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx

from bench.micro import check_load, provenance, set_threads, time_op
from bench.parity import compare, make_crops
from drishyam.segment import OnnxSegmenter

ALL = cv2.getNumThreads()
CALIB, EVAL = 8, 20  # crops 0-7 calibrate, 8-27 measure agreement


def tensor(crop: np.ndarray) -> np.ndarray:
    """A 512x512 crop as OnnxSegmenter feeds it (no letterbox needed at 512): RGB, /255, ImageNet mean and std."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
    mean, std = np.array((0.485, 0.456, 0.406), np.float32), np.array((0.229, 0.224, 0.225), np.float32)
    return ((rgb - mean[:, None, None]) / std[:, None, None])[None]


QUANT = {"dynamic": "quantize_dynamic, int8 weights (per tensor)",
         "static_qdq": "quantize_static QDQ, int8 weights per channel, uint8 activations",
         "static_qoperator": "quantize_static QOperator, int8 weights per tensor, uint8 activations"}


def quantize(fp32: str, out: Path, calib: np.ndarray) -> tuple[dict[str, str], dict[str, str]]:
    """({name: path} of the models written, {name: exact error} of those onnxruntime could not quantize)."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    feed = [{onnx.load(fp32, load_external_data=False).graph.input[0].name: tensor(c)} for c in calib]

    class Crops(CalibrationDataReader):
        def __init__(self):
            self.it = iter(feed)

        def get_next(self):
            return next(self.it, None)

    out.mkdir(parents=True, exist_ok=True)
    pre = str(out / "fp32_pre.onnx")
    quant_pre_process(fp32, pre)  # shape inference and graph cleanup, as onnxruntime recommends before static
    static = {"activation_type": QuantType.QUInt8, "weight_type": QuantType.QInt8}
    # QOperator per channel stops in onnxruntime 1.23's bias requantization (a shape broadcast error), so per tensor
    jobs = {"dynamic": lambda p: quantize_dynamic(fp32, p, weight_type=QuantType.QInt8),
            "static_qdq": lambda p: quantize_static(pre, p, Crops(), quant_format=QuantFormat.QDQ, per_channel=True,
                                                    **static),
            "static_qoperator": lambda p: quantize_static(pre, p, Crops(), quant_format=QuantFormat.QOperator,
                                                          per_channel=False, **static)}
    paths, failed = {}, {}
    for name, job in jobs.items():
        try:
            job(p := str(out / f"int8_{name}.onnx"))
            paths[name] = p
        except Exception as exc:  # noqa: BLE001 -- recorded: which quantization onnxruntime itself could not do
            failed[name] = f"{type(exc).__name__}: {exc}"
    return paths, failed


class ClassicOnnxSegmenter(OnnxSegmenter):
    """OnnxSegmenter on the classic engine (the agent's reads ENGINE_NEW)."""

    @staticmethod
    def _load(path):
        return cv2.dnn.readNetFromONNX(path, engine=cv2.dnn.ENGINE_CLASSIC)


SEGMENTERS = {"new": OnnxSegmenter, "classic": ClassicOnnxSegmenter}


def evaluate(path: str, engine: str, crops: np.ndarray, ref: np.ndarray | None, repeats: int):
    """(row, label maps) of one model on one engine; a failure to read or run is recorded with its exact message."""
    try:
        seg = SEGMENTERS[engine](path)
        labels = np.stack([seg.predict(c, "accurate").labels for c in crops])
    except (cv2.error, ValueError) as exc:
        return {"runs": False, "error": str(exc).strip()}, None
    net, row = seg.nets["accurate"], {"runs": True}
    net.setInput(tensor(crops[0]))
    for t in (1, ALL):
        got = set_threads(t)  # before timing (an assignment evaluates its value before its subscript)
        row[f"forward512_threads{got}"] = {k: v for k, v in time_op(net.forward, repeats, 3).items()
                                           if k in ("n", "p50_ms", "p95_ms", "mean_ms")}
    set_threads(1)
    row["vs_fp32_new"] = None if ref is None else compare(ref, labels)
    return row, labels


def run(arm: str, fp32: str, repeats: int = 20, out_dir="results", work="runs/int8", max_load=None) -> dict:
    check_load(max_load)
    prov = provenance()
    crops = make_crops(CALIB + EVAL)
    paths, failed = quantize(fp32, Path(work), crops[:CALIB])
    models = {"fp32": fp32, **paths}
    set_threads(1)
    ref, rows = None, {f"{k}/quantize": {"how": QUANT[k], "quantized": False, "error": e} for k, e in failed.items()}
    for name, path in models.items():
        g = onnx.load(path)
        info = {"how": QUANT.get(name, "the trained model"), "path": path, "bytes": os.path.getsize(path),
                "op_types": dict(Counter(n.op_type for n in g.graph.node).most_common(12))}
        for engine in SEGMENTERS:
            row, labels = evaluate(path, engine, crops[CALIB:], ref, repeats)
            if name == "fp32" and engine == "new":
                ref, row["vs_fp32_new"] = labels, "reference"
            rows[f"{name}/{engine}"] = {**info, "engine": engine, **row}
            print(name, engine, json.dumps({k: v for k, v in row.items() if k != "vs_fp32_new"})[:300], flush=True)
    doc = {"arm": arm, "config": {"calibration_crops": CALIB, "eval_crops": EVAL, "repeats": repeats,
                                  "crops": "bench.parity.make_crops (evals.synth seeds 9000-9027, 512 px)",
                                  "all_threads": ALL},
           "provenance": prov, "load_avg_end": check_load(), "models": rows}
    out = Path(out_dir) / arm / "int8.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.int8", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--onnx", default=os.environ.get("DRISHYAM_ONNX"), help="the FP32 model (default: DRISHYAM_ONNX)")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-load", type=float)
    a = ap.parse_args(argv)
    if not a.onnx:
        ap.error("--onnx or DRISHYAM_ONNX must name the FP32 model")
    run(a.arm, a.onnx, a.repeats, a.out_dir, max_load=a.max_load)


if __name__ == "__main__":
    main()
