"""Train a dacl10k segmenter for Drishyam (FR-4): smp UNet / DeepLabV3+ on a timm MobileNetV3 encoder.

Kaggle (GPU on, Internet on for the ImageNet encoder weights):
    !python training/train_dacl10k.py --data /kaggle/working/dacl10k --out /kaggle/working/run
Also runs cell by cell (`# %%` cells; call `main([...])` with the same flags as a list).
Smoke test, no dataset and no downloads:   python training/train_dacl10k.py --synthetic --max-steps 3

The head is multi-label (19 sigmoid channels, dacl10k-toolkit TARGET_LIST order from class_map.json), loss is
BCE + Dice, and the 19 -> 6 merge for Drishyam happens at export time (export_onnx.py), not here.
"""
# %% Imports
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path("training")))
from dacl_common import MEAN, STD, build_model, load_class_map


# %% Data
def find_root(path: str) -> Path:
    """The folder holding annotations/ and images/ (the zip may add one or two wrapper folders)."""
    p = Path(path)
    for c in (p, *sorted(p.glob("*")), *sorted(p.glob("*/*"))):
        if (c / "annotations" / "train").is_dir():
            return c
    raise FileNotFoundError(f"no annotations/train under {path}; see training/README.md for the layout")


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


# %% Train
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
    p.add_argument("--resume", help="a last.pt to continue from")
    p.add_argument("--synthetic", action="store_true", help="fabricated tiny dataset, no downloads (smoke test)")
    p.add_argument("--log-every", type=int, default=50)
    a = p.parse_args(argv)
    if not a.synthetic and not a.data:
        p.error("--data is required unless --synthetic")
    if a.size % 32:
        p.error("--size must be a multiple of 32")
    return a


def main(argv=None):
    import segmentation_models_pytorch as smp

    a = parse(argv)
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
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss("multilabel", from_logits=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
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
          f"steps/epoch={len(train_dl)}  total_steps={total}")

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
    print(f"checkpoints in {out}  (best val mIoU {best:.4f})")


if __name__ == "__main__":
    main()
