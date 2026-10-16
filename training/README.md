# Drishyam segmenter: dacl10k training and ONNX export (FR-4)

Fine-tune a MobileNetV3 UNet on dacl10k on Kaggle, export one ONNX that outputs Drishyam's six classes directly,
and prove it matches PyTorch when run through `drishyam.segment.OnnxSegmenter` on OpenCV 5 DNN `ENGINE_NEW`.

| File | Job |
|---|---|
| `class_map.json` | The 19 dacl10k classes and their merge onto Drishyam's six (committed and reported, per the PRD) |
| `dacl_common.py` | Model factory, class-map loader, and `DrishyamHead` (the 19 to 6 merge that is exported in the graph) |
| `train_dacl10k.py` | Kaggle-ready training script, also usable cell by cell (`# %%`); `--synthetic` smoke mode |
| `export_onnx.py` | Checkpoint to FP32 and FP16 ONNX, opset 17, dynamic H x W, onnxslim, OpenCV DNN load check |
| `parity.py` | FR-4 acceptance: 20 fixed crops, PyTorch vs `OnnxSegmenter` |
| `../tests/test_training.py` | Class-map check always; loader, merge, 3-step train, export, DNN load and parity when torch is installed |

## 1. Released dacl10k weights (checked 2026-10-17)

The PRD says to look before training. Nothing is usable as a drop-in; nothing was downloaded.

- **Hugging Face Hub models**: no model repository matches `dacl10k`, `dacl`, or `bridge damage` segmentation
  (searched the models API; the one hit, `fatemehhAmin/Concrete_Bridge_Damage_Detection`, is not dacl10k).
- **[phiyodr/dacl10k-toolkit](https://github.com/phiyodr/dacl10k-toolkit)** (the dataset authors): a `Dataset` class and
  utilities only; no weights, no training script.
- **[HF Space phiyodr/dacl-challenge](https://huggingface.co/spaces/phiyodr/dacl-challenge)** (the dataset authors): ships
  `runs/2023-08-31_rich-paper-12/best_model_cpu.pth` (54.8 MB). Per its `app.py` it is a SegFormer `nvidia/mit-b1`
  with 19 sigmoid channels in the same class order, 512 x 512 input, ImageNet normalisation, threshold 0.5, saved as a
  whole pickled `nn.Module` (loading it executes pickle code: only load it in a throwaway Kaggle session). The Space
  declares no licence; SegFormer's base weights are NVIDIA's research-only licence and dacl10k is CC BY-NC 4.0, so
  treat it as non-commercial. It is the only credible public weight set. It cannot be fine-tuned into this
  pipeline (a different architecture), and SegFormer's attention graph is a risky fit for OpenCV DNN with dynamic H x W,
  so I did not build on it. Use: a teacher for sanity checks or distillation if our MobileNetV3 lands far below
  the paper's 0.320 validation mIoU.
- **[Sarim-MBZUAI/Architect_eye](https://github.com/Sarim-MBZUAI/Architect_eye)**: FPN + EfficientNet-B4, weights in a SharePoint
  folder, no licence stated. Not used.
- [Voxel51/dacl10k](https://huggingface.co/datasets/Voxel51/dacl10k) is a FiftyOne copy of the data (8,922 samples),
  not the official split layout. `train_dacl10k.py` reads the official layout below.

So the model is trained here: UNet (`--arch deeplabv3plus` is also supported) with a timm `mobilenetv3_large_100`
encoder (ImageNet weights, MIT/Apache-2.0 code), via `segmentation_models_pytorch`.

## 2. Class mapping (`class_map.json`)

Source of the 19 names and their channel order: `Dacl10kDataset.TARGET_LIST` in
`dacl10k/dacl10kdataset.py` of the toolkit, and its README "Labels/Classes"; definitions from Flotzinger et al., WACV 2024
([arXiv:2309.00460](https://arxiv.org/abs/2309.00460)), appendix tables 7 to 9. The README lists WConccor
(Washouts/Concrete corrosion) with the six "objects" although the paper defines it as a surface-erosion damage.

| ch | dacl10k | Drishyam | Why |
|---|---|---|---|
| 0 | Crack | crack | 1:1 |
| 1 | ACrack (alligator crack) | crack | A crack pattern; the PRD scores one crack class |
| 2 | Wetspot | background (dropped) | Moisture patch, no Drishyam defect |
| 3 | Efflorescence | efflorescence | 1:1 |
| 4 | Rust | corrosion | Surface rust / stain = CODEBRIM "corrosion stain" |
| 5 | Rockpocket | spalling (merged) | Exposed coarse aggregate: concrete loss from poor compaction |
| 6 | Hollowareas | spalling (merged) | Delamination found by hammering, usually corroding rebar beneath; "about to spall" |
| 7 | Cavity | spalling (merged) | Small air holes; the weakest merge (will add some false spalling) |
| 8 | Spalling | spalling | 1:1 |
| 9 | Graffiti | background (dropped) | A mark that must not read as damage (PRD failure-case list) |
| 10 | Weathering | background (dropped) | Dirt, soot, plant growth; also a PRD false-positive source |
| 11 | Restformwork | background (dropped) | Leftover formwork, a PRD false-positive source |
| 12 | ExposedRebars | exposed_rebar | 1:1 |
| 13 | Bearing | background (dropped) | Structural object |
| 14 | EJoint | background (dropped) | Structural object |
| 15 | Drainage | background (dropped) | Structural object |
| 16 | PEquipment | background (dropped) | Structural object |
| 17 | JTape | background (dropped) | Structural object |
| 18 | WConccor | spalling (merged) | Planar surface erosion; the PRD calls out spalling vs rockpocket vs washout confusion, and the best paper model scores only 0.085 / 0.121 IoU on it (val / test), so a separate class would be mostly noise |

Counts: 5 kept 1:1 (Crack, Spalling, Efflorescence, ExposedRebars, and Rust renamed to corrosion), 1 merged into crack
(ACrack), 4 merged into spalling (Rockpocket, Hollowareas, Cavity, WConccor), 9 dropped to
background. Hollowareas, Cavity and WConccor are judgement calls. Because the merge is applied at export time, changing any
row means editing `class_map.json` and re-running `export_onnx.py`; no retraining.

## 3. How the multi-label head becomes six probabilities

dacl10k is multi-label (a pixel can be Crack and Rust and Weathering), so training uses 19 independent sigmoid channels
(BCE + Dice). The current `OnnxSegmenter` applies a **softmax** to the model output and optionally sums
probabilities through `class_map`. That is wrong for this head: the 19 channels do not compete, there is no
background channel (clean concrete would still be forced to spread its mass over 19 defect-ish classes), and summing
sigmoid-derived probabilities of merged classes over-counts. So the merge is exported inside the graph
(`dacl_common.DrishyamHead`):

- background logit = constant 0;
- each Drishyam class logit = max of its mapped channels' logits (max of logits equals sigmoid of the max);
- the ONNX output is `(1, 6, H, W)` logits in `CLASSES` order, and `OnnxSegmenter`'s softmax then gives, for a pixel with one
  active class, exactly `sigmoid(logit)` for it and `1 - sigmoid` for background. Two simultaneously active classes
  split the mass by odds, which is what a 6-way label needs. Dropped classes contribute no evidence.

`OnnxSegmenter` needs no change and no `class_map`.

## 4. Data

dacl10k v2 "devphase": 6,935 train + 975 validation images with labelme polygons (test labels are withheld, so
validation mIoU is the only local number). Licence **CC BY-NC 4.0**: keep it out of the repo and off EC2.

```
<root>/annotations/{train,validation}/*.json
<root>/images/{train,validation}/*.jpg
```

`--data` may point at a folder one or two levels above `<root>` (what unzipping or a Kaggle dataset gives you).
Download (toolkit README; sha256 `dcbcd5fb82699076a2c7f3a72492a9ef798870e0ca1f0c9399360f273ea95260`):

```
!wget -q https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip
!echo "dcbcd5fb82699076a2c7f3a72492a9ef798870e0ca1f0c9399360f273ea95260  dacl10k_v2_devphase.zip" | sha256sum -c
!unzip -q dacl10k_v2_devphase.zip -d /kaggle/working/dacl10k
```

I could not confirm that the AWS link is live from here (an S3 `HEAD` returned 403); if `wget` fails, get the zip from the toolkit
README or the dacl.ai page and attach it as a private Kaggle dataset instead (then `--data /kaggle/input/<name>`).

## 5. Train on Kaggle

Notebook settings: Accelerator GPU (T4 or P100; the script uses one GPU), Internet on (ImageNet encoder weights
download from the HF hub; so does `pip`).

```
!pip install -q segmentation-models-pytorch onnx onnxslim onnxconverter-common
# put the training/ folder (with class_map.json) next to the notebook: a Kaggle dataset or a git clone
!python training/train_dacl10k.py --data /kaggle/working/dacl10k --out /kaggle/working/run
```

Defaults follow the paper's baselines (512 x 512, 30 epochs, cosine schedule, 2-epoch warm-up) with BCE + Dice, AdamW
(lr 1e-3), batch 16, AMP on GPU, seed 0, horizontal flip, brightness/contrast jitter, and 70% of train samples as a random
zoomed crop of the full-resolution photo (the virtual-PTZ regime). Lower `--batch-size` on an out-of-memory error.

- Every epoch writes `last.pt` (atomic), `best.pt` (best validation mIoU over 19 classes) and `metrics.json`. A 12-hour
  session that dies resumes with `--resume /kaggle/working/run/last.pt` (same flags). Save `/kaggle/working/run` as
  the notebook output.
- Each epoch prints the IoU of all 19 classes and the mean (the paper's validation mIoU, accumulated over the whole
  set), then the same for the five merged Drishyam defect classes (a Drishyam class is on if any mapped channel is).
- Expected time: **roughly 2 to 4 GPU hours for 30 epochs** on a T4 (an estimate from model size, not measured: this
  Mac has no GPU). That is well inside the 30 h/week quota; the data loader (4 vCPUs) should keep up.
- Target: the paper's MobileNetV3 DeepLabV3+ scores 0.320 validation mIoU; anything near 0.3 is fit for purpose.
  `--arch deeplabv3plus` is the like-for-like model (4.7 M params against UNet's 6.7 M).

## 6. Export and parity

```
python training/export_onnx.py --ckpt run/best.pt --out models
python training/parity.py --ckpt run/best.pt --onnx models/drishyam_dacl10k.onnx \
       --onnx-fp16 models/drishyam_dacl10k.fp16.onnx
```

`export_onnx.py` writes `drishyam_dacl10k.onnx` (opset 17, input `(1,3,H,W)` ImageNet-normalised RGB, output `(1,6,H,W)`
logits, H and W dynamic, multiples of 32) and `drishyam_dacl10k.fp16.onnx` (FP16 weights, FP32 inputs and outputs; needs
`onnxconverter-common`), runs `onnxslim` when installed, prints both sizes, and loads each file on
`cv2.dnn.readNetFromONNX(..., engine=cv2.dnn.ENGINE_NEW)` at 256 and 512 as a smoke check.

`parity.py` is the FR-4 acceptance check and exits non-zero on failure. It takes 20 fixed 512 x 512 crops of
`public/samples/*.jpg` and compares `OnnxSegmenter` (probabilities) with PyTorch at 512 (the accurate path) and 256 (the fast
path, which also proves the graph is dynamic), for FP32 (tolerance max |dprob| 1e-3) and FP16 (5e-2). It also reports
max |dlogit| of the raw DNN output, the stricter range-independent number (|dprob| is at most about half of it).
Run it where torch and OpenCV 5 coexist (below), with the repo checked out (it imports `drishyam.segment`).

## 7. Point Drishyam at the model

```
export DRISHYAM_ONNX=/path/to/drishyam_dacl10k.onnx
export DRISHYAM_ONNX_FP16=/path/to/drishyam_dacl10k.fp16.onnx   # optional: the "fast" / FP16 path
```

These map to `OnnxSegmenter(path, fp16_path)`. No `class_map` is needed because the ONNX already outputs the six classes.
(The orchestrator wires the variables into the app; this README only documents them.)

Publish to the Hugging Face Hub so the EC2 box can pull it (the weights inherit dacl10k's non-commercial licence, so say
**CC BY-NC 4.0 (trained on dacl10k)** in the model card, with the `metrics.json` table):

```
hf auth login
hf repo create <user>/drishyam-dacl10k-unet-mnv3 --repo-type model
hf upload <user>/drishyam-dacl10k-unet-mnv3 models/drishyam_dacl10k.onnx
hf upload <user>/drishyam-dacl10k-unet-mnv3 models/drishyam_dacl10k.fp16.onnx
hf upload <user>/drishyam-dacl10k-unet-mnv3 training/class_map.json
```

## 8. Smoke test on the Intel Mac (no dataset, no downloads)

PyTorch's last x86-64 macOS wheels are 2.2.2 (Python 3.12 or older), which needs numpy < 2, while the repo pins
`opencv-python-headless==5.0.0.93`, which declares numpy >= 2 but runs on numpy 1.26 (checked: resize, DNN). So install
OpenCV with `--no-deps`:

```
uv venv --python 3.12 /tmp/torchenv
uv pip install --python /tmp/torchenv/bin/python torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/torchenv/bin/python "numpy<2" segmentation-models-pytorch timm onnx onnxslim onnxconverter-common pytest
uv pip install --python /tmp/torchenv/bin/python --no-deps opencv-python-headless==5.0.0.93

/tmp/torchenv/bin/python training/train_dacl10k.py --synthetic --max-steps 3 --size 128 --batch-size 2 --out /tmp/run
/tmp/torchenv/bin/python training/export_onnx.py --ckpt /tmp/run/last.pt --out /tmp/onnx
/tmp/torchenv/bin/python training/parity.py --ckpt /tmp/run/last.pt --onnx /tmp/onnx/drishyam_dacl10k.onnx --onnx-fp16 /tmp/onnx/drishyam_dacl10k.fp16.onnx
/tmp/torchenv/bin/python -m pytest tests/test_training.py
```

`--synthetic` fabricates random images and masks and forces `--encoder-weights none`: it proves the pipeline runs,
and says nothing about accuracy. On macOS `--workers` defaults to 0: with torch 2.2.2 on Python 3.12.13 a process that used
DataLoader worker processes never exits (a 10-line DataLoader script with no code from here reproduces it), though training
itself completes; Linux/Kaggle defaults to 4 workers and I did not test it there. The main repo venv has no torch, so `tests/test_training.py` runs only its class-map
test there and skips the rest.
