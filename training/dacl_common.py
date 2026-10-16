"""Shared by train / export / parity: the class map, the model factory and the 19 -> 6 merge head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet; drishyam.segment.OnnxSegmenter normalises the same way
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASS_MAP = Path(__file__).with_name("class_map.json")


def load_class_map(path: Path = CLASS_MAP):
    """(dacl10k names, drishyam names, groups) where groups[c - 1] lists the dacl10k channels merged into
    Drishyam class c (c = 1..5; background is the constant-0 logit, never a group)."""
    cm = json.loads(Path(path).read_text())
    dacl, dri, mapping = cm["dacl10k_classes"], cm["drishyam_classes"], cm["map"]
    if set(mapping) != set(dacl) or len(set(dacl)) != len(dacl):
        raise ValueError("class_map.json: 'map' keys must be exactly the dacl10k classes")
    if not set(mapping.values()) <= set(dri) or dri[0] != "background":
        raise ValueError("class_map.json: 'map' values must be Drishyam classes, index 0 = background")
    groups = [[i for i, n in enumerate(dacl) if mapping[n] == c] for c in dri[1:]]
    if not all(groups):
        raise ValueError("class_map.json: every Drishyam defect class needs at least one dacl10k channel")
    return dacl, dri, groups


def build_model(arch: str = "unet", encoder: str = "tu-mobilenetv3_large_100",
                encoder_weights: str | None = "imagenet", classes: int = 19) -> nn.Module:
    """Plain multi-label segmenter: (B,3,H,W) ImageNet-normalised RGB -> (B,classes,H,W) logits.
    H and W must be multiples of 32 (encoder stride)."""
    import segmentation_models_pytorch as smp

    cls = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[arch]
    return cls(encoder_name=encoder, encoder_weights=encoder_weights, classes=classes, activation=None)


class DrishyamHead(nn.Module):
    """19 sigmoid logits -> 6 softmax logits, inside the graph so the ONNX needs no class_map.

    Channel 0 (background) is the constant logit 0 and defect c is the max of its mapped channels' logits
    (= the sigmoid of the max, since sigmoid is monotone). Then softmax([0, s_1..s_5]) gives, for a pixel with
    a single active class, exactly sigmoid(s_c) for that class and 1 - sigmoid(s_c) for background; when two
    classes are both active they split the mass in proportion to their odds. A dacl10k class mapped to
    background simply contributes no evidence. This is what `OnnxSegmenter`'s softmax expects.
    """

    def __init__(self, net: nn.Module, groups: list[list[int]]):
        super().__init__()
        self.net = net
        self.n_groups = len(groups)
        for i, g in enumerate(groups):
            self.register_buffer(f"g{i}", torch.tensor(g, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        outs = [z[:, :1] * 0.0]  # keeps the dynamic H x W; a constant would need a Shape/Expand chain
        for i in range(self.n_groups):
            outs.append(torch.index_select(z, 1, getattr(self, f"g{i}")).amax(dim=1, keepdim=True))
        return torch.cat(outs, dim=1)


def load_checkpoint(path: str, device="cpu"):
    """Returns (plain 19-channel net in eval mode, checkpoint dict). Never downloads encoder weights."""
    ck = torch.load(path, map_location=device)
    net = build_model(ck["arch"], ck["encoder"], None, ck["classes"])
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck
