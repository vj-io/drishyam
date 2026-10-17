"""The dacl10k training/export pipeline (training/). The class-map test needs only the standard library; the rest
runs when torch + segmentation_models_pytorch + timm + onnx are installed (the main venv has none, so they skip)
and covers: polygon rasterisation, the 19 -> 6 merge, a 3-step synthetic train, ONNX export, OpenCV DNN load and
PyTorch parity.
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from drishyam.segment import CLASSES, OnnxSegmenter

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"

# dacl10k-toolkit Dacl10kDataset.TARGET_LIST, copied by hand so the test checks the JSON against the source.
TARGET_LIST = ["Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket", "Hollowareas", "Cavity",
               "Spalling", "Graffiti", "Weathering", "Restformwork", "ExposedRebars",
               "Bearing", "EJoint", "Drainage", "PEquipment", "JTape", "WConccor"]

needs_torch = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("torch", "segmentation_models_pytorch", "timm", "onnx")),
    reason="needs torch, segmentation_models_pytorch, timm and onnx (see training/README.md)",
)


def test_class_map_is_the_official_dacl10k_list_mapped_onto_drishyam():
    cm = json.loads((TRAINING / "class_map.json").read_text())
    assert cm["dacl10k_classes"] == TARGET_LIST and tuple(cm["drishyam_classes"]) == CLASSES
    assert set(cm["map"]) == set(TARGET_LIST) and set(cm["map"].values()) <= set(CLASSES)
    assert set(cm["map"].values()) >= set(CLASSES)  # every Drishyam class, background included, has a source
    assert {k for k, v in cm["map"].items() if v == "crack"} == {"Crack", "ACrack"}
    assert cm["map"]["Rust"] == "corrosion" and cm["map"]["ExposedRebars"] == "exposed_rebar"


def test_kaggle_notebook_is_built_from_the_current_script():
    sys.path.insert(0, str(TRAINING))
    import make_notebook

    assert make_notebook.main(["--check"]) == 0, "stale notebook: run python training/make_notebook.py"
    nb = json.loads((TRAINING / "drishyam_dacl10k_kaggle.ipynb").read_text())
    setup = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("# Setup"))
    assert nb["nbformat"] == 4 and all(repr(rel) in setup for rel in make_notebook.CARRIED)


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(TRAINING))
    import dacl_common
    import export_onnx
    import parity
    import train_dacl10k

    return dacl_common, train_dacl10k, export_onnx, parity


@needs_torch
def test_head_is_zero_background_plus_max_over_mapped_channels(mods):
    import torch

    dacl_common = mods[0]
    names, _, groups = dacl_common.load_class_map()

    class Fixed(torch.nn.Module):
        def forward(self, x):
            return x

    z = torch.full((1, 19, 2, 2), -9.0)
    z[0, names.index("ACrack")] = 3.0  # merged into crack
    z[0, names.index("Crack")] = 1.0  # max, not sum
    z[0, names.index("Graffiti")] = 7.0  # mapped to background: no evidence for any defect
    out = dacl_common.DrishyamHead(Fixed(), groups)(z)
    assert out.shape == (1, 6, 2, 2)
    assert torch.all(out[:, 0] == 0) and torch.allclose(out[:, 1], torch.full((1, 2, 2), 3.0))
    assert torch.all(out[:, 2:] <= -9.0)
    p = out.softmax(1)[0, 1, 0, 0].item()  # one active class: softmax([0, s]) is sigmoid(s)
    assert abs(p - torch.sigmoid(torch.tensor(3.0)).item()) < 0.001  # the other classes' -9 logits leak ~1e-4


@needs_torch
def test_dataset_rasterises_labelme_polygons_per_class(mods, tmp_path):
    import torch

    dacl_common, train = mods[0], mods[1]
    names, _, _ = dacl_common.load_class_map()
    root = tmp_path / "dacl10k_v2_devphase"  # an unzipped archive keeps one wrapper folder
    (root / "annotations" / "train").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "train" / "a.jpg"), np.full((60, 120, 3), 128, np.uint8))
    shapes = [  # left half Crack, the same left half also Rust (multi-label), a right square Spalling
        {"label": "Crack", "points": [[0, 0], [60, 0], [60, 60], [0, 60]], "shape_type": "polygon"},
        {"label": "Rust", "points": [[0, 0], [60, 0], [60, 60], [0, 60]], "shape_type": "polygon"},
        {"label": "Spalling", "points": [[80, 10], [110, 10], [110, 50], [80, 50]], "shape_type": "polygon"},
        {"label": "NotAClass", "points": [[0, 0], [5, 0], [5, 5]], "shape_type": "polygon"},
    ]
    (root / "annotations" / "train" / "a.json").write_text(
        json.dumps({"imageWidth": 120, "imageHeight": 60, "imageName": "a.jpg", "shapes": shapes}))
    assert train.find_root(str(tmp_path)) == root
    kaggle = tmp_path / "input" / "datasets" / "owner" / "slug"  # a Kaggle mount may add more wrapper folders
    kaggle.mkdir(parents=True)
    root = root.rename(kaggle / root.name)
    assert train.find_root(str(tmp_path / "input")) == root
    x, m = train.Dacl10k(root, "train", 64, names, train=False, seed=0)[0]
    assert x.shape == (3, 64, 64) and m.shape == (19, 64, 64)
    crack, rust, spall = (m[names.index(n)] for n in ("Crack", "Rust", "Spalling"))
    assert crack[:, :28].mean() > 0.95 and crack[:, 36:].sum() == 0  # left half only
    assert torch.equal(crack, rust)  # one pixel, two classes
    assert spall[:, :36].sum() == 0 and spall[:, 36:].sum() > 0
    assert m.sum() == crack.sum() + rust.sum() + spall.sum()  # the unknown label drew nothing


@pytest.fixture(scope="module")
def exported(mods, tmp_path_factory):
    """3 synthetic training steps -> checkpoint -> FP32 (+FP16) ONNX."""
    _, train, export_onnx, _ = mods
    run = tmp_path_factory.mktemp("run")
    train.main(["--synthetic", "--max-steps", "3", "--size", "128", "--batch-size", "2", "--out", str(run)])
    paths = export_onnx.export(str(run / "last.pt"), run / "onnx", size=128)
    return run / "last.pt", paths


@needs_torch
def test_synthetic_smoke_writes_checkpoint_and_metrics(exported):
    ckpt, _ = exported
    assert ckpt.exists() and (ckpt.parent / "best.pt").exists()
    history = json.loads((ckpt.parent / "metrics.json").read_text())
    assert history[-1]["step"] == 3 and set(history[-1]["iou19"]) == set(TARGET_LIST)


@needs_torch
def test_onnx_loads_on_opencv_dnn_with_dynamic_hw_and_six_class_output(exported):
    _, paths = exported
    for p in paths.values():
        net = cv2.dnn.readNetFromONNX(str(p), engine=cv2.dnn.ENGINE_NEW)
        for h, w in ((64, 96), (128, 128), (256, 160)):
            net.setInput(np.zeros((1, 3, h, w), np.float32))
            assert np.asarray(net.forward()).shape == (1, 6, h, w)
    seg = OnnxSegmenter(str(paths["fp32"]), input_size=128).predict(np.full((200, 300, 3), 90, np.uint8))
    assert seg.probs.shape == (6, 200, 300) and np.allclose(seg.probs.sum(axis=0), 1, atol=1e-4)


@needs_torch
def test_onnxsegmenter_matches_pytorch(exported, mods):
    ckpt, paths = exported
    fp16 = paths.get("fp16")
    rows = mods[3].run(str(ckpt), str(paths["fp32"]), str(fp16) if fp16 else None, n=2)
    assert all(r["ok"] for r in rows), rows
    assert rows[0]["max_dprob"] < 1e-3
