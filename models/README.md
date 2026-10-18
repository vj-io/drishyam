# Bundled models (third-party; permissive licences, except the CC BY-NC 4.0 `drishyam_dacl10k`)

| File | Source | Licence | sha256 |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo, YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT (`LICENSE.yunet`) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `object_detection_nanodet_2022nov.onnx` | [OpenCV Zoo, NanoDet-Plus](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet) | Apache 2.0 (`LICENSE.nanodet`) | `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186` |

Both are used only by `drishyam/privacy.py` to blur faces, people and vehicles
before an image is analysed or displayed. Fonts in `public/fonts/` are SIL OFL 1.1
(`public/fonts/OFL-plex.txt`, `public/fonts/OFL-archivo.txt`).

The trained segmenter `drishyam_dacl10k.onnx` (and `.fp16.onnx`), built by `training/` and installed here by
`scripts/install_model.sh`, is not a permissive model: it is trained on dacl10k, so its weights are
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial, with attribution). Until it is
installed, no such file is in this folder.

## Trained segmenter: `drishyam_dacl10k`

Fine-tuned on dacl10k with `training/drishyam_dacl10k_kaggle.ipynb` (UNet, timm MobileNetV3-Large encoder) and installed
2026-10-18 by `scripts/install_model.sh`. Input `(1, 3, H, W)` ImageNet-normalised RGB, output `(1, 6, H, W)`
logits in `drishyam.segment.CLASSES` order (the 19 -> 6 merge is in the graph); `DRISHYAM_ONNX` / `DRISHYAM_ONNX_FP16`
select it (`training/README.md`).

| File | Size | sha256 |
| --- | --- | --- |
| `drishyam_dacl10k.onnx` | 26.8 MB | `c49416c10087f3c1baa8443b53fc839a3d8eb227ff27a1b4f7be1da964d6ed4a` |
| `drishyam_dacl10k.fp16.onnx` | 13.5 MB | `f9b06fabbe47a5614c4f61b9170a21c81165d0fd3caf173942ae6ee6e47cb976` |

Epoch 23 of 24, chosen on the mean validation IoU of Drishyam's five defect classes: **0.359** (crack 0.408, spalling 0.397, efflorescence 0.284, exposed_rebar 0.329, corrosion 0.378). Validation mIoU over the 19 dacl10k classes at that epoch: 0.353 (the paper's MobileNetV3 baseline: 0.320). Per-epoch numbers: `drishyam_dacl10k.metrics.json`.

No `run_info.json` came with the weights (notebooks built before it was added do not write one).

**Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial; not the permissive licences
above and not the repo's MIT.** The weights are adapted from (trained on) [dacl10k](https://github.com/phiyodr/dacl10k-toolkit)
(Flotzinger et al., WACV 2024, [arXiv:2309.00460](https://arxiv.org/abs/2309.00460); CC BY-NC 4.0), so they carry its
non-commercial terms and its attribution. No dacl10k images or labels are included. The ImageNet starting point (timm
`mobilenetv3_large_100`) is Apache 2.0.
