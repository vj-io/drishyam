# Models (the two bundled ones are third-party, permissive licences; `drishyam_dacl10k` is CC BY-NC 4.0 and not bundled)

| File | Source | Licence | sha256 |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo, YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT (`LICENSE.yunet`) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `object_detection_nanodet_2022nov.onnx` | [OpenCV Zoo, NanoDet-Plus](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet) | Apache 2.0 (`LICENSE.nanodet`) | `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186` |

Both are used only by `drishyam/privacy.py` to blur faces, people and vehicles
before an image is analysed or displayed. Fonts in `public/fonts/` are SIL OFL 1.1
(`public/fonts/OFL-plex.txt`, `public/fonts/OFL-archivo.txt`).

The trained segmenter `drishyam_dacl10k.onnx` (and `.fp16.onnx`), built by `training/` and installed here by
`scripts/install_model.sh`, is not a permissive model: it is trained on dacl10k, so its weights are
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial, with attribution;
`LICENSE.drishyam_dacl10k`). Its weight files are not committed to this repository, and the live demo does not use
it: the deployed `/api/health` reports segmenter `classical-v1`.

## Trained segmenter: `drishyam_dacl10k`

Fine-tuned on dacl10k with `training/drishyam_dacl10k_kaggle.ipynb` (UNet, timm MobileNetV3-Large encoder) and installed
2026-10-18 by `scripts/install_model.sh`. Input `(1, 3, H, W)` ImageNet-normalised RGB, output `(1, 6, H, W)`
logits in `drishyam.segment.CLASSES` order (the 19 -> 6 merge is in the graph); `DRISHYAM_ONNX` / `DRISHYAM_ONNX_FP16`
select it (`training/README.md`).

| File | Size | sha256 |
| --- | --- | --- |
| `drishyam_dacl10k.onnx` | 26.8 MB | `c49416c10087f3c1baa8443b53fc839a3d8eb227ff27a1b4f7be1da964d6ed4a` |
| `drishyam_dacl10k.fp16.onnx` | 13.5 MB | `f9b06fabbe47a5614c4f61b9170a21c81165d0fd3caf173942ae6ee6e47cb976` |

Epoch 23 of 24, chosen on the mean validation IoU of Drishyam's five defect classes: **0.359** (crack 0.408, spalling 0.397, efflorescence 0.284, exposed_rebar 0.329, corrosion 0.378). That is the split that chose the epoch, so the number is optimistic, and no held-out number exists (dacl10k's test labels are withheld). The merged classes are unions (`training/class_map.json`): crack is dacl10k Crack or ACrack (alligator crack), whose own IoUs at that epoch are 0.212 and 0.443. Validation mIoU over the 19 dacl10k classes at that epoch: 0.353. For reference, not as a like-for-like comparison: the dacl10k paper's DeepLabv3+ with a MobileNetV3-Large encoder and no auxiliary loss scores 0.320 on the validation split ([arXiv:2309.00460](https://arxiv.org/abs/2309.00460), Table 3). Per-epoch numbers: `drishyam_dacl10k.metrics.json`.

No `run_info.json` came with the weights (notebooks built before it was added do not write one).

What is known about this run without it: the notebook ran on Colab (the cell in `training/README.md`, section 4; the
install commit 03af85b says so). `drishyam_dacl10k.metrics.json` holds 24 epochs (0 to 23) of 433 steps each, which is
6,935 // 16, the whole dacl10k train split at the default batch size 16 with `drop_last`. Epoch 23 is the last of
them, so the selection above picked the final checkpoint. The exported graph is a UNet (its decoder nodes are
`decoder/blocks.0` to `.4`). Not recorded: package versions, the notebook source hash, the flags, the seed and the
run time.

**Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial; not the permissive licences
above and not the repo's MIT.** The weights are adapted from (trained on) [dacl10k](https://github.com/phiyodr/dacl10k-toolkit)
(Flotzinger et al., WACV 2024, [arXiv:2309.00460](https://arxiv.org/abs/2309.00460); CC BY-NC 4.0), so they carry its
non-commercial terms and its attribution (`LICENSE.drishyam_dacl10k`). No dacl10k images or labels are included. The
ImageNet starting point (timm `mobilenetv3_large_100`) is Apache 2.0.
