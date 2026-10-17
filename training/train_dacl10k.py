"""Train a dacl10k segmenter for Drishyam (FR-4): smp UNet / DeepLabV3+ on a timm MobileNetV3 encoder.

This file is also the Kaggle notebook: training/make_notebook.py turns its `# %%` cells into
training/drishyam_dacl10k_kaggle.ipynb. Run headless, it runs the same cells top to bottom (train, export FP32 +
FP16 ONNX, parity, zip); imported (tests), it only defines things.
    python training/train_dacl10k.py --data /path/to/dacl10k --out runs/dacl10k
Smoke test, no dataset and no downloads:
    python training/train_dacl10k.py --synthetic --max-steps 3 --size 128 --batch-size 2 --out /tmp/run

The head is multi-label (19 sigmoid channels, dacl10k-toolkit TARGET_LIST order from class_map.json), loss is
BCE + Dice, and the 19 -> 6 merge for Drishyam happens at export time (export_onnx.py), not here.
"""
# %% [markdown]
# # Drishyam: train the dacl10k segmenter on Kaggle
#
# Generated from `training/train_dacl10k.py` by `training/make_notebook.py` (edit the script, not this notebook).
# It trains a UNet + MobileNetV3 on dacl10k, exports the FP32 and FP16 ONNX that Drishyam loads, checks them
# against PyTorch on OpenCV 5 DNN, and zips `best.onnx`, `best_fp16.onnx` and `metrics.json`.
#
# 1. **Data**: *Add Input* > Datasets > search `dacl10k-v2-devphase` and add **winterqi131/dacl10k-v2-devphase**
#    (the official v2 devphase files: 6,935 train and 975 validation images with labelme JSON; the identical
#    **sristi29raj/dacl10k-dataset** also works). Both are third-party re-uploads; the data's own licence is
#    CC BY-NC 4.0 (its `LICENSE.md`). Neither available? Download the official zip to `/tmp` in a cell:
#    `!wget -q https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip -P /tmp`
#    then `!unzip -q /tmp/dacl10k_v2_devphase.zip -d /tmp/dacl10k` and put `--data /tmp/dacl10k` in the
#    settings cell (not `/kaggle/working`: that is saved as output).
# 2. **Settings** (right panel): Accelerator **GPU T4 x2** (one GPU is used), Internet **on** (pip and the
#    ImageNet encoder weights; needs a phone-verified account).
# 3. **Save Version > Save & Run All (Commit)**. It runs in the background (you can close the tab) for up to
#    12 h; 30 epochs are estimated at 2 to 4 h on a T4, and `--hours 11` stops early enough to export anyway.
# 4. When the version is done: its **Output** tab > `run/drishyam_dacl10k.zip` > download.
# 5. Stopped early ("resume to continue" in the log)? New version: *Add Input* > Your Work > this notebook's
#    output, then Save & Run All again; `--resume auto` carries on from its `last.pt`.
# 6. On your machine, in the repo: `unzip drishyam_dacl10k.zip -d /tmp/m && scripts/install_model.sh
#    /tmp/m/best.onnx /tmp/m/best_fp16.onnx` (checks, smoke jobs, before/after eval; see training/README.md).
#
# The weights are trained on CC BY-NC 4.0 data, so they are for non-commercial use.

# %% Imports
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# %% Setup: the repo files and packages the cells below need
# In the notebook make_notebook.py fills REPO_FILES with the training helpers and drishyam/segment.py (for the
# parity check), so it runs on Kaggle without the repo; run from the repo, the files beside this script are used.
REPO_FILES: dict[str, str] = {}
NOTEBOOK = "__file__" not in globals()
KAGGLE = Path("/kaggle/working").is_dir()
CV5 = Path(tempfile.gettempdir()) / "cv5"  # OpenCV 5 for the DNN checks, kept apart from Kaggle's OpenCV 4
if REPO_FILES:
    HERE = Path(tempfile.gettempdir()) / "drishyam" / "training"
    for rel, text in REPO_FILES.items():
        (HERE.parent / rel).parent.mkdir(parents=True, exist_ok=True)
        (HERE.parent / rel).write_text(text)
else:
    HERE = Path("training").resolve() if NOTEBOOK else Path(__file__).resolve().parent
if KAGGLE and __name__ == "__main__":
    # Pinned: what this pipeline adds; torch, torchvision, timm and numpy stay Kaggle's own (pinning them would swap
    # the CUDA build). onnx is pinned too: Kaggle forces protobuf 5.29.5, which recent onnx wheels cannot load, and
    # this one pulls protobuf >= 6.31 (only unused TensorFlow-side packages mind). OpenCV 5 goes to its own folder:
    # Kaggle holds cv2 at 4.x, and two OpenCV wheels in one site-packages overwrite each other's cv2/. Training
    # only needs 4.x; the DNN checks (export and parity, run as child processes) need 5.
    pins = ["segmentation-models-pytorch==0.5.0", "onnx==1.23.2", "onnxslim==0.1.98", "onnxconverter-common==1.16.0"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=True)
    if not (CV5 / "cv2").is_dir():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(CV5),
                        "opencv-python-headless==5.0.0.93"], check=True)
sys.path.insert(0, str(HERE))
from dacl_common import MEAN, STD, build_model, load_class_map


# %% Data
def find_root(path: str) -> Path:
    """The folder holding annotations/ and images/ (an unzip or a Kaggle mount adds wrapper folders)."""
    p = Path(path)
    for depth in range(5):
        for c in sorted(p.glob("/".join(["*"] * depth))) if depth else [p]:
            if (c / "annotations" / "train").is_dir():
                return c
    raise FileNotFoundError(f"no annotations/train under {path}: on Kaggle, Add Input > winterqi131/dacl10k-v2-devphase;"
                            " otherwise see training/README.md for the layout")


class Dacl10k(Dataset):
    """Official split: annotations/{train,validation}/*.json (labelme polygons) + images/<split>/*.jpg.

    Images are resized to size x size with INTER_AREA, exactly like OnnxSegmenter does at inference (and
    like the dacl10k paper's 512 x 512 baselines). Train samples are, 70% of the time, a random zoomed crop
    of the full-resolution photo first, which is what the virtual PTZ feeds the model. Polygons are
    rasterised straight at the target size, one channel per class (a pixel can carry several classes).
    """

    def __init__(self, root: Path, split: str, size: int, names: list[str], train: bool, seed: int):
        self.ann = sorted((root / "annotations" / split).glob("*.json"))
        if not self.ann:
            raise FileNotFoundError(f"no annotations in {root / 'annotations' / split}")
        self.img_dir = root / "images" / split
        self.size, self.train, self.seed, self.epoch = size, train, seed, 0
        self.index = {n: i for i, n in enumerate(names)}

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(self, i: int):
        d = json.loads(self.ann[i].read_text())
        img = cv2.imread(str(self.img_dir / d["imageName"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.img_dir / d["imageName"])
        h, w = img.shape[:2]
        S = self.size
        rng = np.random.default_rng((self.seed, self.epoch, i))  # same sample + epoch => same augmentation
        x0 = y0 = 0
        cw, ch = w, h
        if self.train and rng.random() < 0.7:
            side = rng.uniform(0.3, 1.0)
            cw, ch = max(S // 4, int(w * side)), max(S // 4, int(h * side))
            cw, ch = min(cw, w), min(ch, h)
            x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        x = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (S, S), interpolation=cv2.INTER_AREA)
        mask = np.zeros((len(self.index), S, S), np.uint8)
        scale = (S / cw, S / ch)
        for s in d["shapes"]:
            k = self.index.get(s["label"])
            if k is not None and len(s["points"]) >= 3:
                pts = (np.asarray(s["points"], np.float64) - (x0, y0)) * scale
                cv2.fillPoly(mask[k], [np.round(pts).astype(np.int32)], 1)
        if self.train:
            if rng.random() < 0.5:
                x, mask = x[:, ::-1], mask[:, :, ::-1]
            if rng.random() < 0.8:  # brightness / contrast
                x = np.clip(x.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25), 0, 255)
        rgb = cv2.cvtColor(np.ascontiguousarray(x).astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()), torch.from_numpy(np.ascontiguousarray(mask)).float()


class Synthetic(Dataset):
    """Fabricated noise images with blocky random masks. Smoke-testing only: it proves the pipeline runs."""

    def __init__(self, n: int, size: int, classes: int, seed: int):
        self.n, self.size, self.classes, self.seed, self.epoch = n, size, classes, seed, 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        x = torch.randn(3, self.size, self.size, generator=g)
        m = (torch.rand(self.classes, self.size // 16, self.size // 16, generator=g) > 0.92).float()
        return x, nn.functional.interpolate(m[None], size=(self.size, self.size), mode="nearest")[0]


# %% Metrics: IoU per class accumulated over the whole validation set, as in the dacl10k paper
def _counts(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    dims = (0, 2, 3)
    return torch.stack([(pred & gt).sum(dims), (pred & ~gt).sum(dims), (~pred & gt).sum(dims)])  # tp, fp, fn


def _iou(c: torch.Tensor) -> torch.Tensor:
    tp, fp, fn = c.double()
    den = tp + fp + fn
    return torch.where(den > 0, tp / den.clamp(min=1), torch.full_like(den, float("nan")))  # nan: class never seen


def _merge(t: torch.Tensor, groups) -> torch.Tensor:
    return torch.stack([t[:, g].any(1) for g in groups], 1)  # a Drishyam class is on if any mapped channel is


@torch.no_grad()
def evaluate(model, loader, device, groups, amp, max_batches=0):
    """(IoU for the 19 dacl10k classes, IoU for the 5 Drishyam defects after the class_map merge)."""
    model.eval()
    c19 = c5 = 0
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x.to(device))
        pred, gt = logits.float() > 0, y.to(device) > 0.5  # logit > 0 is sigmoid > 0.5
        c19 = c19 + _counts(pred, gt)
        c5 = c5 + _counts(_merge(pred, groups), _merge(gt, groups))
    return _iou(c19).cpu().numpy(), _iou(c5).cpu().numpy()


# %% Training loop
def parse(argv=None):
    p = argparse.ArgumentParser(description="Train a dacl10k segmenter for Drishyam (see training/README.md)")
    p.add_argument("--data", help="dacl10k root (annotations/ + images/); required unless --synthetic")
    p.add_argument("--out", default="runs/dacl10k", help="checkpoints (last.pt, best.pt) and metrics.json")
    p.add_argument("--arch", choices=["unet", "deeplabv3plus"], default="unet")
    p.add_argument("--encoder", default="tu-mobilenetv3_large_100")
    p.add_argument("--encoder-weights", default="imagenet", help="'imagenet' (downloads) or 'none'")
    p.add_argument("--size", type=int, default=512, help="train/val resolution, a multiple of 32")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # macOS: DataLoader worker processes hang at interpreter exit here (torch 2.2.2 + Python 3.12.13, reproduced with
    # a 10-line script unrelated to this one), so default to in-process loading there.
    p.add_argument("--workers", type=int, default=0 if sys.platform == "darwin" else min(4, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (smoke runs); 0 = off")
    p.add_argument("--resume", help="a last.pt to continue from, or 'auto': <out>/last.pt if there is one, else a "
                   "last.pt under /kaggle/input (an earlier notebook version's output added as input)")
    p.add_argument("--hours", type=float, default=0, help="start no epoch that would end after this many hours "
                   "(a Kaggle session ends at 12); 0 = off. Resume to continue")
    p.add_argument("--synthetic", action="store_true", help="fabricated tiny dataset, no downloads (smoke test)")
    p.add_argument("--log-every", type=int, default=50)
    a = p.parse_args(argv)
    if not a.synthetic and not a.data:
        p.error("--data is required unless --synthetic")
    if a.size % 32:
        p.error("--size must be a multiple of 32")
    return a


def auto_resume(out: Path, search: Path = Path("/kaggle/input")) -> str | None:
    """`--resume auto`: <out>/last.pt, else the newest last.pt under `search`, copied into <out> with the
    best.pt and metrics.json beside it so the history carries on."""
    if (out / "last.pt").exists():
        return str(out / "last.pt")
    found = sorted(search.glob("**/last.pt"), key=lambda f: f.stat().st_mtime) if search.is_dir() else []
    if not found:
        return None
    for name in ("last.pt", "best.pt", "metrics.json"):
        if (found[-1].parent / name).exists():
            shutil.copy2(found[-1].parent / name, out / name)
    print(f"resuming from {found[-1]}")
    return str(out / "last.pt")


def main(argv=None):
    import segmentation_models_pytorch as smp

    a = parse(argv)
    t_run = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    names, _, groups = load_class_map()
    C = len(names)

    if a.synthetic:
        a.encoder_weights, a.workers = "none", 0
        train_ds = Synthetic(16, a.size, C, a.seed)
        val_ds = Synthetic(8, a.size, C, a.seed + 1)
    else:
        root = find_root(a.data)
        train_ds = Dacl10k(root, "train", a.size, names, True, a.seed)
        val_ds = Dacl10k(root, "validation", a.size, names, False, a.seed)
    kw = {"num_workers": a.workers, "pin_memory": amp}
    train_dl = DataLoader(train_ds, a.batch_size, shuffle=True, drop_last=True,
                          generator=torch.Generator().manual_seed(a.seed), **kw)
    val_dl = DataLoader(val_ds, a.batch_size, shuffle=False, **kw)

    model = build_model(a.arch, a.encoder, None if a.encoder_weights == "none" else a.encoder_weights, C).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.max_steps or a.epochs * len(train_dl)
    warm = min(2 * len(train_dl), total // 10)  # the paper's baselines: cosine schedule, 2-epoch warm-up
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    # torch.amp.GradScaler is torch >= 2.3 (Kaggle); torch.cuda.amp's is deprecated there but is all 2.2.2 (the Mac) has
    scaler = (torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp))
    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss("multilabel", from_logits=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.resume == "auto":
        a.resume = auto_resume(out)
    start, step, best, history = 0, 0, -1.0, []
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, step, best = ck["epoch"] + 1, ck["step"], ck.get("best", -1.0)
        history = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else []
    print(f"{a.arch}/{a.encoder}  device={device.type} amp={amp}  train={len(train_ds)} val={len(val_ds)}  "
          f"steps/epoch={len(train_dl)}  total_steps={total}  start epoch {start}")

    for epoch in range(start, a.epochs):
        train_ds.epoch = epoch
        model.train()
        t0, run = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            logits = logits.float()
            loss = bce(logits, y) + dice(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run = loss.item() if run == 0.0 else 0.98 * run + 0.02 * loss.item()
            if step % a.log_every == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if a.max_steps and step >= a.max_steps:
                break

        iou19, iou5 = evaluate(model, val_dl, device, groups, amp, max_batches=2 if a.max_steps else 0)
        miou = float(np.nanmean(iou19))
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s  val mIoU(19) {miou:.4f}  "
              f"Drishyam-5 mean {float(np.nanmean(iou5)):.4f}")
        for n, v in zip(names, iou19):
            print(f"  {n:<14}{v:.3f}")
        for n, v in zip(["crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"], iou5):
            print(f"  [merged] {n:<14}{v:.3f}")
        history.append({"epoch": epoch, "step": step, "train_loss": run, "val_miou19": miou,
                        "iou19": dict(zip(names, map(float, iou19))), "iou5": list(map(float, iou5))})
        best = max(best, miou)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "best": best, "arch": a.arch,
              "encoder": a.encoder, "classes": C, "names": names, "val_miou19": miou, "args": vars(a)}
        torch.save(ck, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", out / "last.pt")  # every epoch; a killed session never leaves a torn file
        if miou >= best:
            torch.save(ck, out / "best.pt")
        (out / "metrics.json").write_text(json.dumps(history, indent=1))
        if a.max_steps and step >= a.max_steps:
            break
        if a.hours and epoch + 1 < a.epochs and time.time() - t_run + (time.time() - t0) > a.hours * 3600:
            print(f"--hours {a.hours}: stopping after epoch {epoch}, the next would end too late; resume to continue")
            break
    print(f"checkpoints in {out}  (best val mIoU {best:.4f})")


def tool(script: str, *args) -> int:
    """Run a training/ script in a child process and show its output; on Kaggle with OpenCV 5 first on the path."""
    env = dict(os.environ)
    if KAGGLE:
        env["PYTHONPATH"] = os.pathsep.join([str(CV5), *filter(None, [env.get("PYTHONPATH")])])
    r = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=False, env=env,
                       capture_output=True, text=True)
    print(r.stdout + r.stderr)
    return r.returncode


# %% Settings
# The notebook runs with these flags (or DACL_FLAGS, if set); a headless run takes them from its command line.
# Every flag is in parse() above. --hours 11 leaves time to export inside Kaggle's 12 h; --resume auto picks up
# a run that stopped (this session's /kaggle/working/run, or an earlier version's output added as input).
NOTEBOOK_FLAGS = "--data /kaggle/input --out /kaggle/working/run --resume auto --hours 11"
if __name__ == "__main__":  # true in a notebook too: importing this module (tests) skips every cell from here
    FLAGS = shlex.split(os.environ.get("DACL_FLAGS", NOTEBOOK_FLAGS)) if NOTEBOOK else sys.argv[1:]
    A = parse(FLAGS)
    OUT, ONNX = Path(A.out), Path(A.out) / "onnx"

# %% Data and GPU check
if __name__ == "__main__":
    ROOT = None if A.synthetic else find_root(A.data)
    if ROOT:
        n_train, n_val = (len(list((ROOT / "annotations" / s).glob("*.json"))) for s in ("train", "validation"))
        full = (n_train, n_val) == (6935, 975)
        print(f"dacl10k at {ROOT}: {n_train} train, {n_val} validation"
              + ("" if full else "  (v2 devphase has 6935 and 975: is this the whole set?)"))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 2**30:.0f} GB  (torch {torch.__version__}, cv2 {cv2.__version__})")
    elif KAGGLE and not A.synthetic:
        raise SystemExit("no GPU: Settings > Accelerator > GPU T4 x2, then run again")
    else:
        print(f"no GPU: training on CPU, fine for --synthetic smoke runs only (torch {torch.__version__})")

# %% Train: <out>/last.pt and metrics.json every epoch, best.pt at the best validation mIoU
if __name__ == "__main__":
    main(FLAGS)

# %% Export: FP32 and FP16 ONNX with the 19 -> 6 merge in the graph, loaded on OpenCV 5 DNN at 256 and 512
if __name__ == "__main__":
    CKPT = OUT / "best.pt" if (OUT / "best.pt").exists() else OUT / "last.pt"
    if tool("export_onnx.py", "--ckpt", CKPT, "--out", ONNX):
        raise SystemExit("export failed: see the output above")
    FP32, FP16 = ONNX / "drishyam_dacl10k.onnx", ONNX / "drishyam_dacl10k.fp16.onnx"

# %% Parity: OnnxSegmenter on OpenCV 5 DNN vs PyTorch, 4 crops (validation images, or noise for --synthetic)
if __name__ == "__main__":
    samples = Path(tempfile.mkdtemp())
    val = sorted((ROOT / "images" / "validation").glob("*.jpg"))[:4] if ROOT else []
    noise = np.random.default_rng(0)
    for i in range(4):
        img = cv2.imread(str(val[i])) if i < len(val) else noise.integers(0, 256, (576, 768, 3), dtype=np.uint8)
        cv2.imwrite(str(samples / f"{i}.jpg"), cv2.resize(img, (768, 576), interpolation=cv2.INTER_AREA))
    PARITY_OK = tool("parity.py", "--ckpt", CKPT, "--onnx", FP32, *(["--onnx-fp16", FP16] if FP16.exists() else []),
                     "--samples", samples, "--n", 4) == 0

# %% Package: <out>/drishyam_dacl10k.zip holds best.onnx, best_fp16.onnx and metrics.json
if __name__ == "__main__":
    ZIP = OUT / "drishyam_dacl10k.zip"
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for src, name in ((FP32, "best.onnx"), (FP16, "best_fp16.onnx"), (OUT / "metrics.json", "metrics.json")):
            if src.exists():
                z.write(src, name)
    print(f"{ZIP}: {', '.join(z.namelist())}, {ZIP.stat().st_size / 1e6:.1f} MB")
    print("parity ok" if PARITY_OK else "PARITY FAILED (table above): do not install these weights")
    print("next, on your machine in the repo, with the zip downloaded there:")
    print("  unzip drishyam_dacl10k.zip -d /tmp/dacl-model")
    print("  scripts/install_model.sh /tmp/dacl-model/best.onnx /tmp/dacl-model/best_fp16.onnx")
    if not NOTEBOOK and not PARITY_OK:
        sys.exit(1)
