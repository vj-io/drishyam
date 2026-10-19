"""Checkpoint -> Drishyam ONNX (FR-4): (1,3,H,W) ImageNet-normalised RGB in, (1,6,H,W) logits out.

The 19 dacl10k sigmoid channels are merged to Drishyam's 6 classes inside the graph (dacl_common.DrishyamHead,
mapping from class_map.json), so OnnxSegmenter needs no class_map. H and W are dynamic (multiples of 32).
Writes <out>/drishyam_dacl10k.onnx (FP32), drishyam_dacl10k.fp16.onnx (weights in FP16, FP32 inputs/outputs) and
drishyam_dacl10k_19.onnx (FP32, the 19 dacl10k logits before the merge: another class_map can be tried on it, on dev
data, without retraining; Drishyam does not load it).

    python training/export_onnx.py --ckpt runs/dacl10k/best.pt --out models/
"""
from __future__ import annotations

import argparse
import inspect
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dacl_common import DrishyamHead, load_checkpoint, load_class_map

NAME = "drishyam_dacl10k"


def _export(model, path: Path, size: int, slim: bool) -> None:
    import onnx

    dummy = torch.randn(1, 3, size, size)
    # The TorchScript exporter, the one checked against OpenCV DNN here. torch >= 2.9 defaults to dynamo=True
    # (a different graph, and it needs onnxscript); torch 2.2.2 has no such argument.
    legacy = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(path), opset_version=17, input_names=["input"],
                          output_names=["logits"], dynamic_axes={"input": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}},
                          **legacy)
    if slim:
        try:
            import onnxslim

            onnxslim.slim(str(path), str(path))  # constant-folds the shape ops OpenCV DNN dislikes
        except ImportError:
            print("onnxslim not installed: skipping graph cleanup (pip install onnxslim)")
    onnx.checker.check_model(onnx.load(str(path)))


def export(ckpt: str, out: Path, size: int = 512, slim: bool = True, fp16: bool = True) -> dict[str, Path]:
    import onnx

    names, _, groups = load_class_map()
    net, ck = load_checkpoint(ckpt)
    if ck["names"] != names:
        raise SystemExit("checkpoint channels do not match class_map.json 'dacl10k_classes'")
    out.mkdir(parents=True, exist_ok=True)
    paths = {"fp32": out / f"{NAME}.onnx"}
    _export(DrishyamHead(net, groups).eval(), paths["fp32"], size, slim)
    if fp16:
        try:
            from onnxconverter_common import float16

            with warnings.catch_warnings():  # "float32 number 1e-08 will be truncated": harmless underflow of tiny weights
                warnings.simplefilter("ignore")
                half = float16.convert_float_to_float16(onnx.load(str(paths["fp32"])), keep_io_types=True)
            paths["fp16"] = out / f"{NAME}.fp16.onnx"
            onnx.save(half, str(paths["fp16"]))
        except ImportError:
            print("onnxconverter-common not installed: no FP16 variant (pip install onnxconverter-common)")
    paths["19"] = out / f"{NAME}_19.onnx"
    _export(net, paths["19"], size, slim)
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="best.pt / last.pt from train_dacl10k.py")
    p.add_argument("--out", default="models", help="output directory")
    p.add_argument("--size", type=int, default=512, help="trace size (the graph stays dynamic)")
    p.add_argument("--no-slim", action="store_true")
    p.add_argument("--no-fp16", action="store_true")
    a = p.parse_args(argv)
    paths = export(a.ckpt, Path(a.out), a.size, not a.no_slim, not a.no_fp16)
    for k, v in paths.items():
        print(f"{k}: {v}  {v.stat().st_size / 1e6:.2f} MB")
    try:  # the check that matters: does the OpenCV 5 DNN new engine accept the graph, at 2 sizes?
        import cv2
        import numpy as np

        for k, v in paths.items():
            net = cv2.dnn.readNetFromONNX(str(v), engine=cv2.dnn.ENGINE_NEW)
            for s in (256, a.size):
                net.setInput(np.zeros((1, 3, s, s), np.float32))
                print(f"OpenCV {cv2.__version__} DNN ENGINE_NEW {k} @ {s}: out {tuple(np.asarray(net.forward()).shape)}")
    except ImportError:
        print("cv2 not installed: skipped the OpenCV DNN load check")


if __name__ == "__main__":
    main()
