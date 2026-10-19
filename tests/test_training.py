"""The dacl10k training/export pipeline (training/). The class-map test needs only the standard library; the rest
runs when torch + segmentation_models_pytorch + timm + onnx are installed (the main venv has none, so they skip)
and covers: polygon rasterisation (thin cracks at low zoom), the v2 crops and loss, the 19 -> 6 merge, a 3-step
synthetic train, resume refusal, ONNX export (merged and 19-class), OpenCV DNN load and PyTorch parity.
"""

import importlib.util
import json
import re
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
    assert f'SOURCE_SHA256 = "{make_notebook.source_sha256()}"' in setup  # what run_info.json records on Kaggle
    # one notebook per VARIANTS key of the script, and they differ in the VARIANT line alone
    src = (TRAINING / "train_dacl10k.py").read_text()
    variants = re.findall(r'^    "(\w+)": ', src.split("VARIANTS = {")[1].split("\n}")[0], re.MULTILINE)
    assert variants == list(make_notebook.NOTEBOOKS) == ["v2a", "v2b"]
    a, b = (make_notebook.NOTEBOOKS[v].read_text().splitlines() for v in variants)
    assert [(x, y) for x, y in zip(a, b) if x != y] == [('    "VARIANT = \\"v2a\\"\\n",', '    "VARIANT = \\"v2b\\"\\n",')]


def test_kaggle_kernel_metadata_per_variant():
    """training/kaggle/<variant>/kernel-metadata.json: the kernel scripts/kaggle_train.sh pushes, with its own notebook."""
    sys.path.insert(0, str(TRAINING))
    import make_notebook

    for variant, notebook in make_notebook.NOTEBOOKS.items():
        folder = TRAINING / "kaggle" / variant
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        assert meta["id"] == f"vaibhav2105/drishyam-dacl10k-train-{variant}"
        assert meta["title"].lower().replace(" ", "-") == meta["id"].split("/")[1]  # Kaggle derives the slug from it
        assert (folder / meta["code_file"]).resolve() == notebook.resolve()
        assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
        assert meta["dataset_sources"] == ["winterqi131/dacl10k-v2-devphase"]
        assert meta["kernel_type"] == "notebook" and meta["language"] == "python"
        assert not any("codebrim" in json.dumps(v).lower() for v in meta.values())  # never a training input


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
    assert m.dtype == torch.uint8  # a quarter of float32 through the workers' shared memory; floated on the device
    crack, rust, spall = (m[names.index(n)] for n in ("Crack", "Rust", "Spalling"))
    assert crack[:, :28].float().mean() > 0.95 and crack[:, 36:].sum() == 0  # left half only
    assert torch.equal(crack, rust)  # one pixel, two classes
    assert spall[:, :36].sum() == 0 and spall[:, 36:].sum() > 0
    assert m.sum() == crack.sum() + rust.sum() + spall.sum()  # the unknown label drew nothing
    ann = root / "annotations" / "train" / "a.json"  # the JSON's size disagrees with the image: refuse, never misplace
    ann.write_text(ann.read_text().replace('"imageWidth": 120', '"imageWidth": 60'))
    with pytest.raises(ValueError, match="decodes to 120x60"):
        train.Dacl10k(root, "train", 64, names, train=False, seed=0)[0]
    # A phone photo stored 120x60 with EXIF orientation 6 is a 60x120 picture, and labelme drew its polygons (and
    # wrote its JSON size) in that rotated frame, as dacl10k_v2_train_3544.jpg is: it must load, not raise.
    from PIL import Image

    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(np.full((60, 120, 3), 128, np.uint8)).save(root / "images" / "train" / "a.jpg", exif=exif)
    ann.write_text(json.dumps({"imageWidth": 60, "imageHeight": 120, "imageName": "a.jpg", "shapes": shapes[:1]}))
    x, m = train.Dacl10k(root, "train", 64, names, train=False, seed=0)[0]
    assert x.shape == (3, 64, 64) and m[names.index("Crack")].sum() > 0


@needs_torch
def test_one_pixel_crack_survives_a_whole_photo_view(mods, tmp_path):
    """A 1 px wide crack in a 4000 px photo, seen whole at 512 px (7.8x down): recipe v2 draws the polygon at 4x the
    target and max-pools, so every target pixel the crack crosses is on. Against the exact max-pool of a fillPoly at
    source resolution (the truth below), over 200 random 1-3 px cracks (2026-10-19): v2 recall 0.90, precision 0.93;
    drawn straight at 512 (v1), recall 0.56, precision 0.70 (OpenCV still draws a sub-pixel polygon's edge, so v1
    keeps a 1 px line, but misses or misplaces much of a crack's footprint)."""
    dacl_common, train = mods[0], mods[1]
    names, _, _ = dacl_common.load_class_map()
    root = tmp_path / "d"
    (root / "annotations" / "train").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "train" / "a.jpg"), np.full((3000, 4000, 3), 128, np.uint8))
    shapes = [{"label": "Crack", "points": [[100, 1500.3], [3900, 1500.3], [3900, 1501.3], [100, 1501.3]]},  # 1 px
              {"label": "Crack", "points": [[100, 100], [2900, 2900], [2901, 2900], [101, 100]]}]  # 1 px, 45 degrees
    (root / "annotations" / "train" / "a.json").write_text(
        json.dumps({"imageWidth": 4000, "imageHeight": 3000, "imageName": "a.jpg", "shapes": shapes}))
    _, m = train.Dacl10k(root, "train", 512, names, train=False, seed=0)[0]  # the whole photo, recipe v2's raster
    crack = m[names.index("Crack")].numpy()
    assert crack.shape == (512, 512) and m.sum() == crack.sum()
    row = round(1500.8 * 512 / 3000)
    assert crack[row - 1 : row + 2, 13:499].any(0).all()  # unbroken, x 100..3900 of 4000
    diag = [crack[round(y * 512 / 3000) - 1 : round(y * 512 / 3000) + 2, round(x * 512 / 4000) - 1 :
                  round(x * 512 / 4000) + 2].any() for x, y in zip(range(100, 2900, 25), range(100, 2900, 25))]
    assert all(diag)
    # What v2 changes: the footprint. Truth without rasterise: the polygons filled at source resolution, then a target
    # pixel is on when any source pixel it covers is. v1 (k = 1) keeps a line too, so presence alone would pass on
    # it; recall separates them.
    src = np.zeros((3000, 4000), np.uint8)
    for s in shapes:
        cv2.fillPoly(src, [np.round(np.asarray(s["points"]) * 16).astype(np.int32)], 1, shift=4)
    yy, xx = np.nonzero(src)
    truth = np.zeros((512, 512), bool)
    truth[yy * 512 // 3000, xx * 512 // 4000] = True
    v1 = train.rasterise(shapes, {"Crack": 0}, (0, 0, 4000, 3000), 512, 1)[0] > 0
    recall = lambda m: (m & truth).sum() / truth.sum()
    assert v1.sum() > 0 and recall(v1) < 0.8 and recall(crack > 0) > 0.9


@needs_torch
def test_v2_crops_span_the_gaze_scales_and_find_the_labels(mods):
    """Recipe v2's train windows: 15 % whole photos; crops log-uniform from 0.75 x --size to the short side, nearly
    square; most of the label-centred ones contain the labelled spot."""
    train = mods[1]
    ds = train.Dacl10k.__new__(train.Dacl10k)  # crop_box needs no files
    ds.size, ds.index, ds.crop_mode, ds.crop_p, ds.crop_label_p, ds.crop_min_px = 512, {"Crack": 0}, "v2", 0.85, 0.6, 384
    shapes = [{"label": "Crack", "points": [[3500, 2500], [3510, 2500], [3510, 2510], [3500, 2510]]}]
    boxes = [ds.crop_box(4000, 3000, shapes, np.random.default_rng((0, 0, i))) for i in range(2000)]
    crops = [b for b in boxes if b[2:] != (4000, 3000)]
    assert 0.12 < 1 - len(crops) / len(boxes) < 0.18
    sides = np.array([np.sqrt(cw * ch) for _, _, cw, ch in crops])
    assert sides.min() >= 384 * 0.9 and sides.max() <= 3000 and np.median(sides) < 1200  # log-uniform 384..3000
    assert all(1 / 1.25 <= cw / ch <= 1.25 for _, _, cw, ch in crops if max(cw, ch) < 3000)
    hit = np.mean([x0 <= 3505 < x0 + cw and y0 <= 2505 < y0 + ch for x0, y0, cw, ch in crops])
    assert hit > 0.6  # 60 % centred on it, plus the large uniform ones that contain it anyway


@needs_torch
def test_loss_v1_is_bce_plus_dice_and_v2_weights_rare_defect_channels_up(mods, tmp_path):
    import segmentation_models_pytorch as smp
    import torch

    dacl_common, train = mods[0], mods[1]
    names, _, groups = dacl_common.load_class_map()
    g = torch.Generator().manual_seed(0)
    z, y = torch.randn(2, 19, 8, 8, generator=g), (torch.rand(2, 19, 8, 8, generator=g) > 0.8).float()
    dice = smp.losses.DiceLoss("multilabel", from_logits=True)
    v1 = train.seg_loss(z, y, dice, 0.0, None, 1.0, 1.0)
    assert torch.allclose(v1, torch.nn.BCEWithLogitsLoss()(z, y) + dice(z, y))
    assert train.seg_loss(z, y, dice, 2.0, None, 1.0, 0.0) < train.seg_loss(z, y, dice, 0.0, None, 1.0, 0.0)
    # pixel shares from the JSONs alone (one 100 x 100 photo): a 20 px square per channel (share 0.04, the median),
    # but Crack 2 px, ExposedRebars 10 px, Spalling 50 px, and Bearing (dropped to background) 2 px
    (tmp_path / "annotations" / "train").mkdir(parents=True)
    side = {**dict.fromkeys(names, 20), "Crack": 2, "ExposedRebars": 10, "Spalling": 50, "Bearing": 2}
    shapes = [{"label": n, "points": [[0, 0], [s, 0], [s, s], [0, s]]} for n, s in side.items()]
    (tmp_path / "annotations" / "train" / "a.json").write_text(
        json.dumps({"imageWidth": 100, "imageHeight": 100, "imageName": "a.jpg", "shapes": shapes}))
    ds = train.Dacl10k(tmp_path, "train", 64, names, train=True, seed=0)
    share, w = train.class_weights(ds, groups, 0.5, 4.0)
    i = names.index
    assert share[i("Crack")] == pytest.approx(4e-4) and share[i("Spalling")] == pytest.approx(0.25)
    assert w[i("Crack")] == 4.0  # sqrt(0.04 / 4e-4) = 10, clipped to --class-weight-max
    assert w[i("ExposedRebars")] == pytest.approx(2.0)  # sqrt(0.04 / 0.01)
    assert w[i("Spalling")] == 1.0 and w[i("Rust")] == 1.0  # common: never weighted down
    assert w[i("Bearing")] == 1.0  # rare, but dropped to background: not weighted


@needs_torch
def test_resume_auto_finds_last_pt_through_a_symlinked_input(mods, tmp_path, capsys):
    train = mods[1]
    (tmp_path / "mnt" / "v1" / "run").mkdir(parents=True)
    (tmp_path / "mnt" / "v1" / "run" / "last.pt").write_bytes(b"ck")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "notebook").symlink_to(tmp_path / "mnt" / "v1")  # pathlib's ** would not enter it
    out = tmp_path / "out"
    out.mkdir()
    assert train.auto_resume(out, tmp_path / "input") == str(out / "last.pt")
    assert (out / "last.pt").read_bytes() == b"ck"
    assert train.auto_resume(tmp_path / "empty", tmp_path / "mnt" / "none") is None
    assert "no last.pt" in capsys.readouterr().out


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
    (info,) = json.loads((ckpt.parent / "run_info.json").read_text())  # one session
    sys.path.insert(0, str(TRAINING))
    import make_notebook
    import timm
    import torch

    assert info["versions"]["torch"] == torch.__version__.split("+")[0] and info["versions"]["timm"] == timm.__version__
    assert info["versions"]["cv2"] == cv2.__version__ and info["source_sha256"] == make_notebook.source_sha256()
    assert info["git_sha"] is None or len(info["git_sha"]) == 40
    assert info["flags"]["recipe"] == "v2" and "scale_mean" in info["select_rule"] and len(info["class_weights"]) == 19
    best = torch.load(ckpt.parent / "best.pt", map_location="cpu")  # chosen on Drishyam's five defects over scales
    assert best["scale_mean"] == max(h["scale_mean"] for h in history) == best["best"]
    assert set(history[-1]["scales"]) == {"128", "256", "512", "whole"}  # --size 128: L0, L1, L2 and the whole photo
    assert history[-1]["scales"]["whole"]["miou5"] == history[-1]["val_miou5"]


@needs_torch
def test_resume_refuses_a_checkpoint_trained_with_other_flags(exported, mods, tmp_path):
    ckpt, _ = exported
    train = mods[1]
    args = ["--synthetic", "--max-steps", "3", "--size", "128", "--batch-size", "2", "--out", str(tmp_path)]
    with pytest.raises(SystemExit, match="--recipe v2 \\(now v1\\), --epochs 60 \\(now 30\\)"):
        train.main([*args, "--resume", str(ckpt), "--recipe", "v1"])
    with pytest.raises(SystemExit, match="--encoder tu-mobilenetv3_large_100 \\(now resnet34\\)"):
        train.main([*args, "--resume", str(ckpt), "--encoder", "resnet34"])
    v1 = {"arch": "unet", "encoder": "tu-mobilenetv3_large_100", "size": 512, "epochs": 30}  # saved before --recipe
    assert train.resume_mismatch(v1, train.parse(["--synthetic", "--recipe", "v1"])) == []
    assert train.resume_mismatch(v1, train.parse(["--synthetic"]))[0] == "--recipe v1 (now v2)"


@needs_torch
def test_onnx_loads_on_opencv_dnn_with_dynamic_hw_and_six_class_output(exported):
    _, paths = exported
    assert set(paths) == {"fp32", "fp16", "19"}
    for k, p in paths.items():
        net = cv2.dnn.readNetFromONNX(str(p), engine=cv2.dnn.ENGINE_NEW)
        for h, w in ((64, 96), (128, 128), (256, 160)):
            net.setInput(np.zeros((1, 3, h, w), np.float32))
            assert np.asarray(net.forward()).shape == (1, 19 if k == "19" else 6, h, w)
    seg = OnnxSegmenter(str(paths["fp32"]), input_size=128).predict(np.full((200, 300, 3), 90, np.uint8))
    assert seg.probs.shape == (6, 200, 300) and np.allclose(seg.probs.sum(axis=0), 1, atol=1e-4)


@needs_torch
def test_onnxsegmenter_matches_pytorch(exported, mods):
    ckpt, paths = exported
    fp16 = paths.get("fp16")
    rows = mods[3].run(str(ckpt), str(paths["fp32"]), str(fp16) if fp16 else None, n=2, onnx19=str(paths["19"]))
    assert all(r["ok"] for r in rows), rows
    assert rows[0]["max_dprob"] < 1e-3
    assert rows[-1]["case"] == "fp32 19 logits" and rows[-1]["max_dlogit"] < 1e-3  # every channel, merged or not


@needs_torch
def test_fp16_over_tolerance_does_not_fail_a_good_fp32_model(exported, mods):
    ckpt, paths = exported
    if "fp16" not in paths:
        pytest.skip("no FP16 export (onnxconverter-common missing)")
    parity = mods[3]
    args = ["--ckpt", str(ckpt), "--onnx", str(paths["fp32"]), "--onnx-fp16", str(paths["fp16"]), "--n", "1"]
    assert parity.main([*args, "--tol-fp16", "0"]) == parity.FP16_OVER  # train_dacl10k.py then zips FP32 alone
    assert parity.main([*args, "--tol", "0"]) == 1
