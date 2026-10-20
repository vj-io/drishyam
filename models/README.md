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

What is known about this run without it (`drishyam_dacl10k.provenance.json`): the notebook ran on Colab (the cell in `training/README.md`, section 4; the install commit 03af85b says so). `drishyam_dacl10k.metrics.json` holds 24 epochs (0 to 23) of 433 steps each, which is 6,935 // 16, the whole dacl10k train split at the default batch size 16 with `drop_last`. Epoch 23 is the last of them, so the selection above picked the final checkpoint. The exported graph is a UNet (its decoder nodes are `decoder/blocks.0` to `.4`). Not recorded: package versions, the notebook source hash, the flags, the seed and the run time.

**Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial; not the permissive licences
above and not the repo's MIT.** The weights are adapted from (trained on) [dacl10k](https://github.com/phiyodr/dacl10k-toolkit)
(Flotzinger et al., WACV 2024, [arXiv:2309.00460](https://arxiv.org/abs/2309.00460); CC BY-NC 4.0), so they carry its
non-commercial terms and its attribution (`LICENSE.drishyam_dacl10k`). No dacl10k images or labels are included. The
ImageNet starting point (timm `mobilenetv3_large_100`) is Apache 2.0.

## Trained segmenter: `drishyam_dacl10k_v2a`

Fine-tuned on dacl10k with `training/drishyam_dacl10k_kaggle.ipynb` (recipe v2, variant v2a: unet, timm `mobilenetv3_large_100` encoder) and installed
2026-10-19 by `scripts/install_model.sh`. Input `(1, 3, H, W)` ImageNet-normalised RGB, output `(1, 6, H, W)`
logits in `drishyam.segment.CLASSES` order (the 19 -> 6 merge is in the graph); `DRISHYAM_ONNX` / `DRISHYAM_ONNX_FP16`
select it (`training/README.md`).

| File | Size | sha256 |
| --- | --- | --- |
| `drishyam_dacl10k_v2a.onnx` | 26.8 MB | `95b503c2d205567c09b19154ff77a5117889b7aba0b19733ed1c178dc4ab1ad1` |
| `drishyam_dacl10k_v2a.fp16.onnx` | 13.5 MB | `0f5fa16e9a65d34ebc3229b78f7d09f9011c7fcaf6935598e86be2e3504a1892` |
| `eval/drishyam_dacl10k_v2a_19.onnx` | 26.8 MB | `844690e08aa85055768f26404aa93251073bdf807cb6ba574405e39a34182ebb` |

Epoch 59 of 60, chosen on the mean validation IoU of Drishyam's five defect classes over 4 views (square crops of 512, 1024, 2048 source px, and the whole photo): **0.391** (512 px 0.402, 1024 px 0.398, 2048 px 0.392, whole 0.371); on whole photos alone 0.371 (crack 0.438, spalling 0.378, efflorescence 0.317, exposed_rebar 0.345, corrosion 0.380). That is the split that chose the epoch, so the number is optimistic, and no held-out number exists (dacl10k's test labels are withheld). The merged classes are unions (`training/class_map.json`): crack is dacl10k Crack or ACrack (alligator crack), whose own IoUs at that epoch are 0.232 and 0.466. Validation mIoU over the 19 dacl10k classes at that epoch: 0.353. For reference, not as a like-for-like comparison: the dacl10k paper's DeepLabv3+ with a MobileNetV3-Large encoder and no auxiliary loss scores 0.320 on the validation split ([arXiv:2309.00460](https://arxiv.org/abs/2309.00460), Table 3). Per-epoch numbers: `drishyam_dacl10k_v2a.metrics.json`.

Trained with Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, segmentation-models-pytorch 0.5.0, onnx 1.23.2, onnxslim 0.1.98, onnxconverter-common 1.16.0, numpy 2.1.3, cv2 4.14.0, cv2_export 5.0.0.93 (1 session(s); every one is in `drishyam_dacl10k_v2a.run_info.json`). Source: no git checkout (Kaggle); notebook sources sha256 `15fa5c66887d08ed6582dbe51dbd0e6ba42a427a7c043632555e678aa28b7468` (`training/make_notebook.py`'s `source_sha256()` at the commit the notebook was built from). Training flags: that file's `flags`; best.pt: the epoch with the highest scale_mean, the mean over 4 views (square crops of 512, 1024, 2048 source px, each clamped to the photo's short side, 60 % centred on a labelled pixel, seeded and the same every epoch, and the whole photo) of the validation mean IoU of Drishyam's 5 defects.

**Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial; not the permissive licences
above and not the repo's MIT.** The weights are adapted from (trained on) [dacl10k](https://github.com/phiyodr/dacl10k-toolkit)
(Flotzinger et al., WACV 2024, [arXiv:2309.00460](https://arxiv.org/abs/2309.00460); CC BY-NC 4.0), so they carry its
non-commercial terms and its attribution (`LICENSE.drishyam_dacl10k`). No dacl10k images or labels are included. The
ImageNet starting point (timm `mobilenetv3_large_100`) is Apache 2.0.

## Trained segmenter: `drishyam_dacl10k_v2b`

Fine-tuned on dacl10k with `training/drishyam_dacl10k_kaggle_v2b.ipynb` (recipe v2, variant v2b: unet, timm `efficientnet_b2` encoder) and installed
2026-10-19 by `scripts/install_model.sh`. Input `(1, 3, H, W)` ImageNet-normalised RGB, output `(1, 6, H, W)`
logits in `drishyam.segment.CLASSES` order (the 19 -> 6 merge is in the graph); `DRISHYAM_ONNX` / `DRISHYAM_ONNX_FP16`
select it (`training/README.md`).

| File | Size | sha256 |
| --- | --- | --- |
| `drishyam_dacl10k_v2b.onnx` | 38.3 MB | `47e4b687d179efc9d221ef05eca441d24b0f025f9ab252c0a60c64455ff99207` |
| `drishyam_dacl10k_v2b.fp16.onnx` | 19.3 MB | `3f8b8648ccfd465e8bd668cf0b2e8d7c6085252426ff048c256dee69d3922838` |
| `eval/drishyam_dacl10k_v2b_19.onnx` | 38.3 MB | `4b679ffafbc1b76cfe95057740be6e0a78ea397eb63ac761dd12a0b493151dbb` |

Epoch 58 of 60, chosen on the mean validation IoU of Drishyam's five defect classes over 4 views (square crops of 512, 1024, 2048 source px, and the whole photo): **0.390** (512 px 0.381, 1024 px 0.407, 2048 px 0.401, whole 0.370); on whole photos alone 0.370 (crack 0.420, spalling 0.390, efflorescence 0.308, exposed_rebar 0.352, corrosion 0.378). That is the split that chose the epoch, so the number is optimistic, and no held-out number exists (dacl10k's test labels are withheld). The merged classes are unions (`training/class_map.json`): crack is dacl10k Crack or ACrack (alligator crack), whose own IoUs at that epoch are 0.243 and 0.441. Validation mIoU over the 19 dacl10k classes at that epoch: 0.352. For reference, not as a like-for-like comparison: the dacl10k paper's DeepLabv3+ with a MobileNetV3-Large encoder and no auxiliary loss scores 0.320 on the validation split ([arXiv:2309.00460](https://arxiv.org/abs/2309.00460), Table 3). Per-epoch numbers: `drishyam_dacl10k_v2b.metrics.json`.

Trained with Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, segmentation-models-pytorch 0.5.0, onnx 1.23.2, onnxslim 0.1.98, onnxconverter-common 1.16.0, numpy 2.1.3, cv2 4.14.0, cv2_export 5.0.0.93 (1 session(s); every one is in `drishyam_dacl10k_v2b.run_info.json`). Source: no git checkout (Kaggle); notebook sources sha256 `15fa5c66887d08ed6582dbe51dbd0e6ba42a427a7c043632555e678aa28b7468` (`training/make_notebook.py`'s `source_sha256()` at the commit the notebook was built from). Training flags: that file's `flags`; best.pt: the epoch with the highest scale_mean, the mean over 4 views (square crops of 512, 1024, 2048 source px, each clamped to the photo's short side, 60 % centred on a labelled pixel, seeded and the same every epoch, and the whole photo) of the validation mean IoU of Drishyam's 5 defects.

**Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial; not the permissive licences
above and not the repo's MIT.** The weights are adapted from (trained on) [dacl10k](https://github.com/phiyodr/dacl10k-toolkit)
(Flotzinger et al., WACV 2024, [arXiv:2309.00460](https://arxiv.org/abs/2309.00460); CC BY-NC 4.0), so they carry its
non-commercial terms and its attribution (`LICENSE.drishyam_dacl10k`). No dacl10k images or labels are included. The
ImageNet starting point (timm `efficientnet_b2`) is Apache 2.0.

## Trained segmenter: `drishyam_dacl10k_v2c`

Fine-tuned on dacl10k with `training/drishyam_dacl10k_kaggle_v2c.ipynb` (recipe v2, variant v2c: unet, timm `mobilenetv3_large_100` encoder) and installed
2026-10-20 by `scripts/install_model.sh`. Input `(1, 3, H, W)` ImageNet-normalised RGB, output `(1, 6, H, W)`
logits in `drishyam.segment.CLASSES` order (the 19 -> 6 merge is in the graph); `DRISHYAM_ONNX` / `DRISHYAM_ONNX_FP16`
select it (`training/README.md`).

| File | Size | sha256 |
| --- | --- | --- |
| `drishyam_dacl10k_v2c.onnx` | 26.8 MB | `24b4d561fb012cb525131e16fa211b528a2e30c90fd9b9e76d98e29edbf18d21` |
| `drishyam_dacl10k_v2c.fp16.onnx` | 13.5 MB | `fe45fea70cd09522b01d186d04735e847b1a3adfddb6d9cfec4c7df700d53fbc` |
| `eval/drishyam_dacl10k_v2c_19.onnx` | 26.8 MB | `94e815b55450b7176c55e7bc8edbe092cb16fba75929980536ae339564dd0c93` |

Epoch 53 of 60, chosen on the mean validation IoU of Drishyam's five defect classes over 4 views (square crops of 512, 1024, 2048 source px, and the whole photo): **0.378** (512 px 0.389, 1024 px 0.390, 2048 px 0.380, whole 0.355); on whole photos alone 0.355 (crack 0.398, spalling 0.382, efflorescence 0.301, exposed_rebar 0.335, corrosion 0.361). That is the split that chose the epoch, so the number is optimistic, and no held-out number exists (dacl10k's test labels are withheld). The merged classes are unions (`training/class_map.json`): crack is dacl10k Crack or ACrack (alligator crack), whose own IoUs at that epoch are 0.241 and 0.411. Validation mIoU over the 19 dacl10k classes at that epoch: 0.350. For reference, not as a like-for-like comparison: the dacl10k paper's DeepLabv3+ with a MobileNetV3-Large encoder and no auxiliary loss scores 0.320 on the validation split ([arXiv:2309.00460](https://arxiv.org/abs/2309.00460), Table 3). Per-epoch numbers: `drishyam_dacl10k_v2c.metrics.json`.

Trained with Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0+cu130, timm 1.0.30, segmentation-models-pytorch 0.5.0, onnx 1.23.2, onnxslim 0.1.98, onnxconverter-common 1.16.0, numpy 2.1.3, cv2 4.14.0, cv2_export 5.0.0.93 (4 session(s); every one is in `drishyam_dacl10k_v2c.run_info.json`). Source: no git checkout (Kaggle); notebook sources sha256 `f25e7ea28fa29bc0f6d1087f0fb8c70ca1ffbf78dbe403d45e21e9fbed901cf5` (`training/make_notebook.py`'s `source_sha256()` at the commit the notebook was built from). Training flags: that file's `flags`; best.pt: the epoch with the highest scale_mean, the mean over 4 views (square crops of 512, 1024, 2048 source px, each clamped to the photo's short side, 60 % centred on a labelled pixel, seeded and the same every epoch, and the whole photo) of the validation mean IoU of Drishyam's 5 defects.

**Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial; not the permissive licences
above and not the repo's MIT.** The weights are adapted from (trained on) [dacl10k](https://github.com/phiyodr/dacl10k-toolkit)
(Flotzinger et al., WACV 2024, [arXiv:2309.00460](https://arxiv.org/abs/2309.00460); CC BY-NC 4.0), so they carry its
non-commercial terms and its attribution (`LICENSE.drishyam_dacl10k`). No dacl10k images or labels are included. The
ImageNet starting point (timm `mobilenetv3_large_100`) is Apache 2.0.
