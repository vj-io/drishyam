"""FR-4 acceptance: the exported ONNX on OpenCV 5 DNN (ENGINE_NEW), through drishyam.segment.OnnxSegmenter,
must match the PyTorch checkpoint on 20 fixed crops.

    python training/parity.py --ckpt runs/dacl10k/best.pt --onnx models/drishyam_dacl10k.onnx \
        [--onnx-fp16 models/drishyam_dacl10k.fp16.onnx] [--onnx-19 models/drishyam_dacl10k_19.onnx]

Compared per crop: probabilities (softmax over the 6 logits, as OnnxSegmenter returns them) and the raw DNN
logits. Crops are 512 x 512 windows of public/samples/*.jpg at fixed positions. Each ONNX runs at 512 (the
"accurate" path) and at 256 (the "fast" path), which also checks that H x W really is dynamic.
`--onnx-19` adds a row for the 19-class logits ONNX (raw cv2.dnn at 512 against the checkpoint's net without the
merge; probabilities are the 19 sigmoids): DrishyamHead discards the channels mapped to background, so the rows
above never see them. It is an FP32 row.
Exit status: 0 when every row is within its tolerance (max |dprob|: FP32 1e-3, FP16 5e-2 by default), 1 when an
FP32 row is not (do not install), FP16_OVER (3) when FP32 passes and only FP16 is over (install the FP32 file alone:
the FP16 margin shrinks as training goes on, so it must not hold back a good FP32 model).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from dacl_common import MEAN, STD, DrishyamHead, load_checkpoint, load_class_map

from drishyam.segment import OnnxSegmenter

FP16_OVER = 3


def crops(samples: Path, n: int, size: int, seed: int) -> list[np.ndarray]:
    files = sorted(samples.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no *.jpg in {samples}")
    imgs = [cv2.imread(str(f)) for f in files]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        img = imgs[i % len(imgs)]
        h, w = img.shape[:2]
        y, x = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
        out.append(img[y:y + size, x:x + size].copy())
    return out


def torch_reference(model, crop: np.ndarray, size: int):
    """(6, size, size) logits and the (6, h, w) probabilities, preprocessed like OnnxSegmenter.predict."""
    h, w = crop.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    blob = ((rgb.astype(np.float32).transpose(2, 0, 1) / 255.0 - MEAN.reshape(3, 1, 1)) / STD.reshape(3, 1, 1))[None]
    with torch.no_grad():
        z = model(torch.from_numpy(np.ascontiguousarray(blob))).numpy()[0]
    e = np.exp(z - z.max(axis=0, keepdims=True))
    probs = e / e.sum(axis=0, keepdims=True)
    return blob, z, np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in probs])


def run(ckpt: str, onnx32: str, onnx16: str | None = None, samples: Path = ROOT / "public" / "samples",
        n: int = 20, crop: int = 512, seed: int = 0, tol: float = 1e-3, tol16: float = 5e-2,
        onnx19: str | None = None) -> list[dict]:
    _, _, groups = load_class_map()
    net, _ = load_checkpoint(ckpt)
    model = DrishyamHead(net, groups).eval()
    cs = crops(samples, n, crop, seed)
    seg32 = OnnxSegmenter(onnx32)
    assert np.allclose(seg32.mean.ravel(), MEAN) and np.allclose(seg32.std.ravel(), STD), "normalisation differs"
    cases = [("fp32 accurate", seg32, "accurate", 512, tol), ("fp32 fast", seg32, "fast", 256, tol)]
    if onnx16:
        cases += [("fp16 accurate", OnnxSegmenter(onnx16), "accurate", 512, tol16),
                  ("fp16 fast", OnnxSegmenter(onnx32, fp16_path=onnx16), "fast", 256, tol16)]
    rows = []
    for name, seg, mode, size, limit in cases:
        dp = dz = zmax = 0.0
        agree = []
        dnn = seg.nets["fp16" if mode == "fast" else "accurate"]  # the net OnnxSegmenter.predict will use
        for c in cs:
            blob, z, p_ref = torch_reference(model, c, size)
            s = seg.predict(c, mode)
            dp = max(dp, float(np.abs(s.probs - p_ref).max()))
            dnn.setInput(blob)
            dz = max(dz, float(np.abs(np.asarray(dnn.forward())[0] - z).max()))
            zmax = max(zmax, float(np.abs(z).max()))
            agree.append(float((s.probs.argmax(0) == p_ref.argmax(0)).mean()))
        rows.append({"case": name, "size": size, "max_dprob": dp, "max_dlogit": dz, "max_abs_logit": zmax,
                     "label_agreement": float(np.mean(agree)), "tol": limit, "ok": dp <= limit})
    if onnx19:  # the 19 logits before the merge, every channel (multi-label: sigmoid probabilities, agreement per channel)
        dnn, dp, dz, zmax, agree = cv2.dnn.readNetFromONNX(onnx19, engine=cv2.dnn.ENGINE_NEW), 0.0, 0.0, 0.0, []
        for c in cs:
            blob, z, _ = torch_reference(net, c, crop)
            dnn.setInput(blob)
            y = np.asarray(dnn.forward())[0]
            dp = max(dp, float(np.abs(1 / (1 + np.exp(-y)) - 1 / (1 + np.exp(-z))).max()))
            dz, zmax = max(dz, float(np.abs(y - z).max())), max(zmax, float(np.abs(z).max()))
            agree.append(float(((y > 0) == (z > 0)).mean()))
        rows.append({"case": "fp32 19 logits", "size": crop, "max_dprob": dp, "max_dlogit": dz, "max_abs_logit": zmax,
                     "label_agreement": float(np.mean(agree)), "tol": tol, "ok": dp <= tol})
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--onnx", required=True, help="FP32 ONNX from export_onnx.py")
    p.add_argument("--onnx-fp16")
    p.add_argument("--onnx-19", help="19-class logits ONNX from export_onnx.py (an FP32 row of its own)")
    p.add_argument("--samples", default=str(ROOT / "public" / "samples"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-3, help="max |dprob| allowed for FP32")
    p.add_argument("--tol-fp16", type=float, default=5e-2)
    a = p.parse_args(argv)
    rows = run(a.ckpt, a.onnx, a.onnx_fp16, Path(a.samples), a.n, tol=a.tol, tol16=a.tol_fp16, onnx19=a.onnx_19)
    print(f"OpenCV {cv2.__version__} DNN ENGINE_NEW vs torch {torch.__version__}, {a.n} crops")
    print(f"{'case':<15}{'size':>5}{'max|dprob|':>12}{'max|dlogit|':>13}{'max|logit|':>12}{'label agree':>13}{'tol':>8}")
    for r in rows:
        print(f"{r['case']:<15}{r['size']:>5}{r['max_dprob']:>12.2e}{r['max_dlogit']:>13.2e}{r['max_abs_logit']:>12.2f}"
              f"{r['label_agreement']:>13.5f}{r['tol']:>8.0e}  {'ok' if r['ok'] else 'FAIL'}")
    if not all(r["ok"] for r in rows if r["case"].startswith("fp32")):
        print("FP32 over tolerance: do not install these weights")
        return 1
    if not all(r["ok"] for r in rows):
        print("FP32 ok; FP16 over tolerance: install the FP32 ONNX alone (Drishyam's fast looks then use it too)")
        return FP16_OVER
    return 0


if __name__ == "__main__":
    sys.exit(main())
