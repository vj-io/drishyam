"""Repro: OpenCV 5.0.0 ENGINE_CLASSIC cannot import an ONNX model whose Shape node reads a dynamic H x W input.

    python bench/repro/classic_shape.py [--out DIR]

Needs only `opencv-python-headless` 5.0.0 and `onnx`. Builds three tiny models with onnx.helper (opset 17), the
pattern PyTorch exports for `F.interpolate(y, size=x.shape[-2:])` in a UNet decoder:

    x [1, 3, H, W] -> AveragePool 2x2 -> y [1, 3, H/2, W/2]
    Shape(x) -> Gather(2), Gather(3) -> Unsqueeze -> Concat([1, 3], h, w) -> Resize(y, sizes) -> z [1, 3, H, W]

`dynamic_hw` declares H and W symbolic (dim_param "h", "w"), as `torch.onnx.export(dynamic_axes=...)` writes them;
`dynamic_batch` declares only the batch symbolic; `static` fixes 1x3x64x48. Each is read with
cv2.dnn.readNetFromONNX(path, engine=...) for ENGINE_CLASSIC and ENGINE_NEW; every net that loads runs one 1x3x64x48
forward, compared with the same graph in NumPy. Prints one line per model and engine; exits 0 either way.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SHAPE = (1, 3, 64, 48)
DIMS = {"dynamic_hw": [1, 3, "h", "w"], "dynamic_batch": ["n", 3, 64, 48], "static": list(SHAPE)}
ENGINES = {"classic": cv2.dnn.ENGINE_CLASSIC, "new": cv2.dnn.ENGINE_NEW}


def build(path: str, dims: list) -> str:
    """The Shape -> Gather -> Unsqueeze -> Concat -> Resize model above, input declared with `dims`, saved to `path`."""
    i64 = lambda name, v: numpy_helper.from_array(np.array(v, np.int64), name)
    nodes = [
        helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Shape", ["x"], ["s"], name="shape_of_x"),
        helper.make_node("Constant", [], ["i2"], value=i64("i2", 2)),  # 0-d indices, as the exporter writes them
        helper.make_node("Constant", [], ["i3"], value=i64("i3", 3)),
        helper.make_node("Gather", ["s", "i2"], ["h"], axis=0),
        helper.make_node("Gather", ["s", "i3"], ["w"], axis=0),
        helper.make_node("Unsqueeze", ["h", "axes"], ["h1"]),
        helper.make_node("Unsqueeze", ["w", "axes"], ["w1"]),
        helper.make_node("Concat", ["nc", "h1", "w1"], ["sizes"], axis=0),
        helper.make_node("Resize", ["y", "", "", "sizes"], ["z"], mode="nearest",
                         coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
    ]
    graph = helper.make_graph(nodes, "shape_resize", [helper.make_tensor_value_info("x", TensorProto.FLOAT, dims)],
                              [helper.make_tensor_value_info("z", TensorProto.FLOAT, dims)],
                              [i64("axes", [0]), i64("nc", [1, 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    return path


def reference(x: np.ndarray) -> np.ndarray:
    """The same graph in NumPy: 2x2 average pool, then nearest (floor, asymmetric) back to H x W."""
    n, c, h, w = x.shape
    y = x[:, :, : h // 2 * 2, : w // 2 * 2].reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))
    return y[:, :, (np.arange(h) * (h // 2) // h)][:, :, :, (np.arange(w) * (w // 2) // w)]


def run(path: str, engine: str, x: np.ndarray) -> str:
    try:
        net = cv2.dnn.readNetFromONNX(path, engine=ENGINES[engine])
    except cv2.error as exc:
        return "fails to load: " + " ".join(str(exc).split())
    net.setInput(x)
    z = np.asarray(net.forward())
    return f"loads; forward {tuple(z.shape)}, max |z - numpy| {np.abs(z - reference(x)).max():.2e}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="where the three models are written (default: a new temporary directory)")
    a = ap.parse_args(argv)
    out = Path(a.out or tempfile.mkdtemp(prefix="classic_shape_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"cv2 {cv2.__version__}, onnx {onnx.__version__}, numpy {np.__version__}, "
          f"python {sys.version.split()[0]}; models in {out}")
    x = np.random.default_rng(0).standard_normal(SHAPE).astype(np.float32)
    for name, dims in DIMS.items():
        path = build(str(out / f"{name}.onnx"), dims)
        for engine in ENGINES:
            print(f"{name:13s} {dims!s:20s} ENGINE_{engine.upper():7s} {run(path, engine, x)}", flush=True)


if __name__ == "__main__":
    main()
