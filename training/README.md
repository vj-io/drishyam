# Drishyam segmenter: dacl10k training and ONNX export (FR-4)

Fine-tune a UNet on dacl10k on Kaggle (or Colab, section 4), export one ONNX that outputs Drishyam's six classes directly,
and prove it matches PyTorch when run through `drishyam.segment.OnnxSegmenter` on OpenCV 5 DNN `ENGINE_NEW`.
Training follows recipe v2 (section 5) in two variants, one Kaggle kernel each (section 5a, "Kaggle from the
command line"): **v2a**, the MobileNetV3-Large UNet Drishyam deploys today, and **v2b**, an EfficientNet-B2 UNet
(section 5b). `--recipe v1` reproduces the first model.

## Kaggle in 6 steps

The notebook carries everything it needs (the training helpers and `drishyam/segment.py` are embedded), so nothing
has to be pushed or uploaded first. It needs a Kaggle account with phone verification (for Internet and the GPU).

1. kaggle.com > Create > New Notebook > File > **Import Notebook** > upload `training/drishyam_dacl10k_kaggle.ipynb`
   (variant v2a; `drishyam_dacl10k_kaggle_v2b.ipynb` is v2b). From a terminal instead: section 5a.
2. **Add Input** > Datasets > search `dacl10k-v2-devphase` > add **winterqi131/dacl10k-v2-devphase** (checked
   2026-10-17: the official v2 devphase layout, 6,935 train + 975 validation labelme JSON with their images, 5.1 GB;
   **sristi29raj/dacl10k-dataset** is the same set, same byte count). Both are third-party re-uploads of data
   licensed **CC BY-NC 4.0** (their `LICENSE.md`), whatever licence the Kaggle page shows. No mirror? See section 4.
3. Right panel > Session options: Accelerator **GPU T4 x2**, Internet **on**.
4. **Save Version > Save & Run All (Commit)**. It runs in the background (closing the tab is fine), checkpoints
   `run/last.pt`, `best.pt` and `metrics.json` every epoch, and stops starting epochs at 11 h (`--hours 11`), leaving
   time to export inside the 12 h limit. Recipe v2 asks for 60 epochs; the time per epoch is not measured (the log and
   `metrics.json` give each epoch's seconds), so a run may need a second session (step 5).
5. When the version is complete: its **Output** tab > `run/drishyam_dacl10k.zip` (`best.onnx`, `best_fp16.onnx`,
   `best_19.onnx`, `metrics.json`, `run_info.json`; about 60 MB for v2a, 90 MB for v2b) > download. The log ends with the parity table and `parity ok`. The verdict is
   FP32's: if only FP16 is over its tolerance, the log says so and the zip holds `best.onnx` without `best_fp16.onnx`.
   If it says "resume to continue", make a new version with **Add Input > Your Work > this notebook's output** and run all again:
   `--resume auto` copies that `last.pt` and carries on.
6. Hand back, in the repo (the main `.venv`, no torch needed), giving the zip's path as the browser saved it:

   ```
   scripts/install_model.sh ~/Downloads/drishyam_dacl10k.zip
   ```

   If your browser unpacked it (Safari does by default), give the folder instead:

   ```
   scripts/install_model.sh ~/Downloads/drishyam_dacl10k
   ```

   It unpacks the zip into a fresh temporary folder, loads both files on OpenCV 5 DNN `ENGINE_NEW` and through
   `OnnxSegmenter` before touching anything (FP16 against FP32 too; if FP16 is over its tolerance it is dropped and
   FP32 is installed alone, which Drishyam's fast looks then use as well), copies them to
   `models/NAME.onnx` and `.fp16.onnx` (and `metrics.json` and `run_info.json` to `models/NAME.metrics.json` and `.run_info.json`), writes their sha256, validation mIoU, the run's package versions, flags and source, and **CC BY-NC 4.0** licence into
   `models/README.md` (a section per NAME), runs three `drishyam run` jobs with `DRISHYAM_ONNX` set, runs
   `evals.harness` on `data/synth200` twice (classical, then the model) into `runs/model-eval/` (never
   `public/results`), prints the before/after recall table with each run's pixel fraction, and prints the `git` and
   `scripts/aws.sh deploy` commands that would put the model on the live site
   (it runs none of them). 200 images is the published eval; `--n 20` is a quicker look. The ONNX run is CPU-bound:
   on the 16-thread Intel Mac a 512 px look took 0.25 s and an image's policies about 46 s against 6 s classical
   (smoke model, 8 workers), so 200 images should take roughly 25 minutes, against 3 for the classical run.

   NAME is `drishyam_dacl10k` for a v1 run and `drishyam_dacl10k_<variant>` (`_v2a`, `_v2b`) for a v2 run (its
   `run_info.json` says which), so the v2 models install **beside** v1 and none overwrites another's files, cost table
   (`results/<arm>-dacl10k_v2a/`) or eval folders (`runs/model-eval/onnx-v2a-*`). `best_19.onnx` goes to
   `models/eval/NAME_19.onnx`, which `scripts/build_lambda.sh` does not package. Every `models/*.onnx` is packaged,
   though: with three segmenters installed, move the ones you are not deploying out of `models/` before
   `scripts/aws.sh deploy`, or the package can pass Lambda's 250 MB limit (the script says so).

Edit `train_dacl10k.py`, never the notebooks: `python training/make_notebook.py` rebuilds them (one per variant), and
`tests/test_training.py` fails while a committed notebook is stale. The notebook's flags are the
`NOTEBOOK_FLAGS` line of its settings cell plus its variant's line of `VARIANTS`; every flag is in `parse()`.

| File | Job |
|---|---|
| `class_map.json` | The 19 dacl10k classes and their merge onto Drishyam's six (committed and reported, per the PRD) |
| `dacl_common.py` | Model factory, class-map loader, and `DrishyamHead` (the 19 to 6 merge that is exported in the graph) |
| `train_dacl10k.py` | Training script and the notebook's source (`# %%` cells): train, export, parity, zip; `--synthetic` smoke mode |
| `make_notebook.py` | Builds one notebook per variant from those cells, embedding the helper files; `--check` for staleness |
| `drishyam_dacl10k_kaggle.ipynb`, `drishyam_dacl10k_kaggle_v2b.ipynb` | The generated Kaggle notebooks, v2a and v2b (do not edit by hand) |
| `kaggle/<variant>/kernel-metadata.json` | Each variant's Kaggle kernel: id, notebook, GPU, internet, dataset (section 5a) |
| `../scripts/kaggle_train.sh` | `push`, `status`, `output` of a variant's kernel with the kaggle CLI (section 5a) |
| `../scripts/install_model.sh` | The hand-back: checks, installs into `models/`, smoke jobs, before/after eval, deploy commands |
| `export_onnx.py` | Checkpoint to FP32 and FP16 ONNX, plus the 19-class logits ONNX; opset 17, dynamic H x W, onnxslim, OpenCV DNN load check |
| `parity.py` | FR-4 acceptance: 20 fixed crops, PyTorch vs `OnnxSegmenter` |
| `../tests/test_training.py` | Class-map, notebook and kernel-metadata checks always; loader, thin-crack masks, v2 crops and loss, merge, 3-step train, resume refusal, export, DNN load and parity when torch is installed |

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
  the paper's 0.320 validation mIoU for DeepLabv3+ with MobileNetV3-Large (arXiv:2309.00460, Table 3).
- **[Sarim-MBZUAI/Architect_eye](https://github.com/Sarim-MBZUAI/Architect_eye)**: FPN + EfficientNet-B4, weights in a SharePoint
  folder, no licence stated. Not used.
- [Voxel51/dacl10k](https://huggingface.co/datasets/Voxel51/dacl10k) is a FiftyOne copy of the data (8,922 samples),
  not the official split layout. `train_dacl10k.py` reads the official layout below.

So the model is trained here: UNet (`--arch deeplabv3plus` is also supported) with a timm `mobilenetv3_large_100`
encoder (ImageNet weights, MIT/Apache-2.0 code; variant v2b uses `efficientnet_b2`, section 5b), via
`segmentation_models_pytorch`.

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
| 18 | WConccor | spalling (merged) | Planar surface erosion; the PRD calls out spalling vs rockpocket vs washout confusion, and the best paper model (FPN, EfficientNet-B4, auxiliary loss) scores only 0.085 / 0.121 IoU on it (val / test; the paper's Table 4), so a separate class would be mostly noise |

Counts: 5 kept 1:1 (Crack, Spalling, Efflorescence, ExposedRebars, and Rust renamed to corrosion), 1 merged into crack
(ACrack), 4 merged into spalling (Rockpocket, Hollowareas, Cavity, WConccor), 9 dropped to
background. Hollowareas, Cavity and WConccor are judgement calls. Because the merge is applied at export time, changing any
row means editing `class_map.json` and re-running `export_onnx.py`; no retraining.

Proposed 2026-10-19 and **not** adopted: Wetspot and Weathering to corrosion. dacl10k defines Wetspot as moisture and
Weathering as surface deterioration (dirt, soot, plant growth), not rust, so merging them would label every damp or
dirty patch as corrosion on the strength of a guess. Measure it instead: recipe v2 runs also export
`best_19.onnx`, the 19 logits before any merge (installed as `models/eval/NAME_19.onnx`), so any mapping can be
scored on dev data by taking the max of its channels' logits (what `DrishyamHead` does) without retraining; change
this table only with that result. (One training-time coupling: v2's loss weights, section 5, up-weight only channels
merged into a defect, so Wetspot and Weathering were trained at weight 1.)

## 3. How the multi-label head becomes six probabilities

dacl10k is multi-label (a pixel can be Crack and Rust and Weathering), so training uses 19 independent sigmoid channels
(focal BCE + Dice in recipe v2, BCE + Dice in v1). The current `OnnxSegmenter` applies a **softmax** to the model output and optionally sums
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
validation mIoU is the only local number; the epoch is chosen on it too, so it is not a held-out score). Licence
**CC BY-NC 4.0**: keep the data out of the repo.

```
<root>/annotations/{train,validation}/*.json
<root>/images/{train,validation}/*.jpg
```

Images are read with their EXIF orientation applied (labelme applied it when the polygons were drawn: the 10 of 7,910 photos with a 90-degree EXIF rotation are labelled in the rotated frame), and an
image whose decoded size differs from its JSON's `imageWidth` x `imageHeight` stops training with its name instead of
training on misplaced masks.

`--data` may point at a folder one or two levels above `<root>` (what unzipping or a Kaggle dataset gives you).
On Kaggle, attach a mirror (step 2 above; the notebook searches `/kaggle/input` up to four folders deep).
Elsewhere, fetch the same Kaggle dataset with `kagglehub` (public, so no token is needed) and pass its path to
`--data`. The official AWS zip from the toolkit README now answers "All access to this object has been disabled".

On **Colab** (a T4 runtime), the same notebook runs with one cell added at the top: it creates `/kaggle/working`
(which switches on the notebook's own setup: the pins and OpenCV 5 in its own folder), downloads the data and adds
flags (`DACL_FLAGS` goes after the notebook's own flags, so `--data`, `--out` and `--hours` there win and the
variant's line still applies). Free Colab sessions end without warning and lose `/content`, so cap the run and
download the zip promptly:

```python
import os, kagglehub
os.makedirs("/kaggle/working", exist_ok=True)
DATA = kagglehub.dataset_download("winterqi131/dacl10k-v2-devphase")
os.environ["DACL_FLAGS"] = f"--data {DATA} --out /content/run --hours 3"
```

## 5. Train on Kaggle

The notebook (steps above) installs what it adds, pinned: `segmentation-models-pytorch==0.5.0`, `onnx==1.23.2`,
`onnxslim==0.1.98`, `onnxconverter-common==1.16.0` (the versions verified on the Mac). torch, torchvision, timm and
numpy stay Kaggle's own (pinning them would replace the CUDA build); each run's `run_info.json` records the versions it used. onnx is pinned because Kaggle force-installs
protobuf 5.29.5, which recent onnx wheels refuse to load; this onnx pulls protobuf >= 6.31 into the session. Kaggle holds `cv2` below 5 on purpose (its OpenCV wheels share one `cv2/` folder),
so `opencv-python-headless==5.0.0.93` goes to its own `--target` folder and only the export and parity child
processes put it first on `PYTHONPATH`; training's imread/resize/fillPoly run on Kaggle's 4.x. Checked against
the image definition in Kaggle/docker-python (2026-10-17: Python 3.13, a 2026-09 Colab base, protobuf 5.29.5):
`export_onnx.py` asks for `dynamo=False` because torch 2.9+ defaults to the dynamo exporter. Replayed in a
`python:3.13-slim` container with torch 2.14.1 (CPU), cv2 4.14 and protobuf forced to 5.29.5 (where the image's own
onnx no longer imports), the notebook ran end to end with `DACL_FLAGS="--synthetic --max-steps 3 --size 128
--batch-size 2 --out /kaggle/working/run"`: install, train, export, parity ok, zip. Not covered by that replay: CUDA and
AMP, the real dataset mount, and the run time. That replay was of the v1 notebook and was not repeated for v2: both
v2 notebooks ran end to end on the Mac only (torch 2.2.2, OpenCV 5 throughout; section 8), so v2's added OpenCV calls
(`fillPoly`, `polylines`, `contourArea`) have not run on Kaggle's OpenCV 4.x.

Common to both recipes: 512 x 512, cosine schedule with a 2-epoch warm-up (the paper's baselines), AdamW (lr 1e-3),
batch 16, AMP on GPU, seed 0, horizontal flip, brightness/contrast jitter. Lower `--batch-size` on an out-of-memory error.

**Recipe v2** is the default; `--recipe v1` restores every default of the first model (the code up to c36e375).
A recipe only sets defaults: any flag given wins.

| | v1 (the installed model) | v2 |
|---|---|---|
| train windows | 70 %: a crop of 30-100 % of each side; else the whole photo | 85 %: a square-ish crop (aspect within 1.2) whose side is log-uniform from 0.75 x `--size` to the photo's short side, 60 % of them centred on a pixel of a random labelled polygon; else the whole photo (`--crop-mode`, `--crop-p`, `--crop-min-px`, `--crop-label-p`) |
| masks | polygons drawn at the target size | drawn at up to 4x the target, outline included, and max-pooled (`--raster-k`) |
| loss | BCE + Dice | focal BCE (gamma 2) with per-channel weights, + Dice (`--focal-gamma`, `--bce-weight`, `--dice-weight`, `--class-weights`, `--class-weight-power`, `--class-weight-max`) |
| epochs | 30 | 60 (`--epochs`) |
| `best.pt` | highest whole-photo `val_miou5` | highest `scale_mean`: mean `val_miou5` over four views (`--select`) |

Why: Drishyam's gaze windows are 512 x 2^L source px. L0, 512 px, is about 13 % of a 4000 px photo's side, a scale
v1's crops (30 to 100 % of the photo) never showed the model, and on the CODEBRIM dev split the v1 model's L0 tiling
scored 0.38 against 0.55 at L1 and 0.51 at L2.

- **Crops**: a v2 crop is cut in source px and resized with `INTER_AREA`, as `OnnxSegmenter` resizes a gaze view, so
  every pyramid level the agent looks at (and the whole photo) is in the training distribution. Centring 60 % of crops
  on a labelled polygon (uniform over polygons, so thin, numerous cracks get their share against large spalls)
  keeps the fine crops from being mostly background.
- **Masks**: a downscaled window's polygons are filled, outline included, at min(4, the downscale factor) x the target
  size and max-pooled, so a target pixel is on when any of its sub-pixels is covered. Against the max-pool of a
  source-resolution raster, over 200 random 1 to 3 px cracks in a 4000 x 3000 photo seen whole at 512 px
  (2026-10-19): drawn straight at 512 (v1) recall 0.59, precision 0.73; v2 recall 0.93, precision 0.96.
  OpenCV's `fillPoly` still draws a sub-pixel polygon's edge, so v1 did not drop 1 px cracks outright; it missed or
  misplaced much of their footprint. `tests/test_training.py` checks that a 1 px crack (straight and at 45 degrees)
  in a 4000 px photo is unbroken in the 512 px whole-photo mask. The other side of it: at 8x down a 1 px crack keeps
  an eighth of its contrast in the image, so v2 asks the model to mark lines it can barely see; whether that helps
  is for the dev eval.
- **Loss**: each channel merged into a Drishyam defect gets the weight clip(sqrt(median share / share), 1, 4), from
  its share of train pixels (polygon areas from the JSONs, no image read; printed at the start of each session,
  rarest first, and saved in `run_info.json` as `train_pixel_share` and `class_weights`). Crack and ExposedRebars,
  the rarest of them, get the most. Channels dropped to background keep 1, and nothing is weighted down. A weight
  multiplies a channel's positive and negative terms alike, so it changes how much the channel drives the shared
  features, not where its sigmoid crosses 0.5 (BCE's `pos_weight` would move that threshold, and the merged ONNX's
  labels with it). Focal (gamma 2) treats both labels alike, so its optimum also crosses 0.5 where the true
  probability does, but it pulls probabilities toward 0.5: the agent's entropy-driven zoom will see less confident
  pixels than v1 gave it, so recheck the thresholds tuned on v1 on dev data. Dice is per class already and is not
  weighted. Untuned: the term weights are 1 and 1 and no sweep was run.
- **Validation, every epoch**: the 975 validation photos four ways: square crops of 512, 1024 and 2048 source px
  (clamped to the short side; positions seeded, the same every epoch and whatever the training flags, 60 % centred on
  a labelled pixel), and the whole photo. The log prints each class's IoU per view; `metrics.json` holds every number
  (`scales`: per view the 19-class and merged mIoU and IoUs), the whole-photo fields v1 had (`val_miou19`,
  `val_miou5`, `iou19`, `iou5`), `scale_mean`, and each epoch's train and validation seconds (`epoch_s`, `val_s`).
  `run_info.json` records the rule in words (`select_rule`) and every flag (`flags`). v2's validation masks are
  max-pooled too, so its numbers are not like-for-like with v1's 0.353 / 0.359 (thin classes have more positive
  pixels); the comparison that counts is the CODEBRIM dev eval of the installed ONNX.
- **Schedule and resume**: 60 epochs under `--hours 11` (no epoch starts that would end after 11 h, leaving the export
  inside Kaggle's 12 h). The time per epoch is not measured; if the run stops early, `scripts/kaggle_train.sh push
  <variant> --resume` (or step 5 above) continues it. A resume (`--resume`, or `auto`) refuses a checkpoint trained
  with other `RESUME_KEYS` flags (recipe, arch, encoder, size, epochs, batch, lr, crop, raster, loss and selection
  flags) and names each one that differs, rather than changing the schedule, data or selection mid-run; a v1
  checkpoint counts as `--recipe v1`. Resuming across notebook versions works when those flags match.

- Each session first appends to `run_info.json`: its first epoch, Python, the torch, torchvision, timm,
  segmentation-models-pytorch, onnx, onnxslim, onnxconverter-common, numpy and OpenCV (training and export) versions
  as installed, the git SHA when run from a checkout, and `source_sha256`, which `make_notebook.py` writes into the
  notebook (Kaggle has no checkout): `source_sha256()` from `training/make_notebook.py` at a commit gives the same
  value when the notebook was built there.
- Every epoch writes `last.pt` (atomic), `best.pt` and `metrics.json`. `best.pt` is the checkpoint that is exported
  and shipped. `--resume auto` continues from `<out>/last.pt`, or from a `last.pt` under `/kaggle/input` (an earlier
  version's output). The notebook uses it, with `--hours 11`.
- The 30 h/week GPU quota holds the two variants' first sessions (up to 12 h each) with about 6 h to spare, so a
  resumed second session may have to wait for the next week's quota.

## 5a. Kaggle from the command line

Two kernels train in parallel, one per variant: `vaibhav2105/drishyam-dacl10k-train-v2a` (UNet + MobileNetV3-Large,
the deployable one) and `vaibhav2105/drishyam-dacl10k-train-v2b` (UNet + EfficientNet-B2, section 5b). Each has
`training/kaggle/<variant>/kernel-metadata.json`: private, GPU, internet on, the dataset
`winterqi131/dacl10k-v2-devphase`, and its generated notebook. The notebooks differ in one line, `VARIANT = "v2a"`,
which picks that variant's flag line from `VARIANTS` in the settings cell.

It needs the kaggle CLI (2.2.4 checked; when none is on `PATH` the script runs `uvx kaggle==2.2.4`) signed in as the
kernels' owner, whose account is phone-verified (GPU and internet). The CLI reads its own credentials; the script
never reads, prints or passes them.

```
scripts/kaggle_train.sh push v2a
scripts/kaggle_train.sh push v2b
scripts/kaggle_train.sh status v2a
scripts/kaggle_train.sh output v2a
```

`push` uploads the notebook as a new kernel version, which Kaggle queues and runs; `status` says queued, running,
complete or error; `output` downloads into a fresh `runs/kaggle/v2a/<time>/` (`run/drishyam_dacl10k.zip` and the
kernel log) and prints the install command, which installs the model as `models/drishyam_dacl10k_v2a.onnx`:

```
scripts/install_model.sh runs/kaggle/v2a/<time>/run/drishyam_dacl10k.zip
```

(No comments inside these blocks: macOS's default zsh passes a pasted `# ...` to the command as arguments.)

- `push` refuses a stale notebook (`make_notebook.py --check`) and a metadata file whose notebook is another
  variant's. `push <variant> --resume` attaches the kernel's own latest output as an input (`kernel_sources`), so
  `--resume auto` carries on from its `last.pt` after a run stopped by `--hours 11`.
- What is uploaded: the notebook and its metadata, nothing else. The kernel reads dacl10k (CC BY-NC 4.0) from its
  Kaggle mirror. CODEBRIM is never uploaded or used for training: its licence bars sharing the images, and it is the
  evaluation set.
- The accelerator: kaggle 2.2.4's `kernels push` sends the metadata's `machine_shape` (its `--accelerator` option
  overrides it), and the kagglesdk 0.1.37 it installs documents `NvidiaTeslaT4`, `NvidiaTeslaP100` and `Tpu1VmV38`
  as the supported values (`ApiSaveKernelRequest`), so the metadata asks for `NvidiaTeslaT4` with `enable_gpu` true.
  Not checked: whether that shape is one T4 or the UI's "GPU T4 x2" (one GPU is used either way), and how many GPU
  sessions Kaggle runs at once (if v2b queues behind v2a, it starts when v2a ends).
- `output` downloads only the zip (and the kernel log the CLI always writes); `status` and `output` take the
  latest version.
- Reference: the paper's DeepLabv3+ with a MobileNetV3-Large encoder scores 0.320 validation mIoU without auxiliary
  loss and 0.329 with it (arXiv:2309.00460, Table 3). Ours is measured on the split that chose its epoch, so a number
  near that is a sanity check, not a held-out result; whether the model helps is for the CODEBRIM eval to show.
  `--arch deeplabv3plus` is the like-for-like model (4.7 M params against UNet's 6.7 M).

## 5b. Variant B: the encoder, and what a look costs on a CPU

v2b is the same recipe and UNet decoder on timm `efficientnet_b2` (ImageNet weights, `efficientnet_b2.ra_in1k`,
Apache 2.0, downloaded on Kaggle like v2a's): one flag, `--encoder tu-efficientnet_b2`, in its `VARIANTS` line.
Candidates, each exported by `export_onnx.py` (onnxslim, FP16) and run through `OnnxSegmenter` on OpenCV 5.0.0 DNN
`ENGINE_NEW` on this Mac (Intel i9-9880H, 8 cores / 16 threads), one 512 px look, untrained weights (the time does not
depend on them), median of 8 after 2 warm-ups, `cv2.setNumThreads(1)`, 2026-10-19 under other jobs' load (load
average 13 to 15):

| UNet encoder | params | FP32 ONNX | 1 thread, 512 px | 1 thread, FP16 256 px | parity (max abs dlogit, one crop) |
|---|---|---|---|---|---|
| MobileNetV3-Large (v1, v2a) | 6.7 M | 27 MB | 773 ms | 185 ms | 2.4e-06 |
| EfficientNet-B0 | 5.8 M | 23 MB | 742 ms | 224 ms | 4.1e-07 |
| **EfficientNet-B2 (v2b)** | 9.5 M | 38 MB | 976 ms (1.26x) | 378 ms | 5.4e-07 |
| ResNet-34 | 24.4 M | 98 MB | 1286 ms (1.66x) | 421 ms | 9.0e-04 |

The UNet decoder is the same in all four and, judging by these numbers, most of the cost: EfficientNet-B0 costs what MobileNetV3 does
and B2 only about a quarter more. B2 is the strongest of the four on ImageNet (timm: about 80 % top-1 against about
76 % for MobileNetV3-Large-100, 78 % for B0 and 73 % for ResNet-34) at that price; ResNet-34 costs two thirds more and
is 3.6 times the file.

Measured again on the smoke runs' exported files (the pipeline's own export, recipe v2), A and B interleaved so
load hits both alike (`time.thread_time`, the CPU time of the one thread doing the work, since the load average was
30 to 70), 24 looks each:

| | v2a | v2b | B / A |
|---|---|---|---|
| 1 thread, 512 px (accurate) | 622 ms CPU | 767 ms CPU | 1.24 |
| 1 thread, FP16 at 256 px (fast) | 199 ms | 234 ms | 1.18 |
| all threads, 512 px (wall) | 214 ms | 263 ms | 1.23 |

A second interleaved run, while other jobs' load fell from 35 to 18, gave slower absolute times (867 and 1117 ms of
CPU on one thread: hyperthread neighbours slow a core) and the same ratios: 1.28, 1.19 and 1.25.

The deployed MobileNetV3 UNet (v2a's architecture) takes about 560 ms a look on one Lambda thread (the figure in the
v2 brief, not re-measured here); at 1.24 to 1.28x, v2b would take about 700 ms there (an estimate from this ratio,
not measured on Lambda). Both smoke runs' ONNX loaded on `ENGINE_NEW` and matched PyTorch
through `OnnxSegmenter` (4 crops at 512 and 256: FP32 max |dprob| 3e-07 for v2b, 6e-07 for v2a; FP16 1.4e-04 and
4.1e-04), on near-untrained weights (max |logit| under 1.1): the FP16 margin shrinks as a model trains, and the
notebook then ships FP32 alone.

Training memory, the tensors autograd keeps per 512 px sample (fp32, CPU): MobileNetV3 427 MiB, B2 849 MiB. v1 trained
MobileNetV3 at batch 16 on a T4; B2 at batch 16 needs about twice that, which should fit a 15 GB T4 under AMP but is
not measured. If v2b stops with CUDA out of memory, add `--batch-size 12` to its `VARIANTS` line (and keep v2a's 16).

## 6. Export and parity

```
python training/export_onnx.py --ckpt run/best.pt --out models
python training/parity.py --ckpt run/best.pt --onnx models/drishyam_dacl10k.onnx \
       --onnx-fp16 models/drishyam_dacl10k.fp16.onnx
```

`export_onnx.py` writes `drishyam_dacl10k.onnx` (opset 17, input `(1,3,H,W)` ImageNet-normalised RGB, output `(1,6,H,W)`
logits, H and W dynamic, multiples of 32) and `drishyam_dacl10k.fp16.onnx` (FP16 weights, FP32 inputs and outputs; needs
`onnxconverter-common`) and `drishyam_dacl10k_19.onnx` (the 19 dacl10k logits before the merge, `(1,19,H,W)`, for
scoring other class mappings; zipped as `best_19.onnx`), runs `onnxslim` when installed, prints the sizes, and loads
each file on `cv2.dnn.readNetFromONNX(..., engine=cv2.dnn.ENGINE_NEW)` at 256 and 512 as a smoke check.

`parity.py` is the FR-4 acceptance check. It exits 1 when FP32 is over its tolerance (do not install) and 3 when
FP32 passes and only FP16 is over (install the FP32 file alone: the FP16 margin shrinks as the model trains). It
takes 20 fixed 512 x 512 crops of `public/samples/*.jpg` and compares `OnnxSegmenter` (probabilities) with PyTorch at 512 (the accurate path) and 256 (the fast
path, which also proves the graph is dynamic), for FP32 (tolerance max |dprob| 1e-3) and FP16 (5e-2). It also reports
max |dlogit| of the raw DNN output, the stricter range-independent number (|dprob| is at most about half of it).
Run it where torch and OpenCV 5 coexist (below), with the repo checked out (it imports `drishyam.segment`).

## 7. Point Drishyam at the model

`scripts/install_model.sh` (step 6 above) does this and prints the deploy command for the live site. By hand:

```
export DRISHYAM_ONNX=/path/to/drishyam_dacl10k.onnx
```

Optional, for the "fast" / FP16 path:

```
export DRISHYAM_ONNX_FP16=/path/to/drishyam_dacl10k.fp16.onnx
```

These map to `OnnxSegmenter(path, fp16_path)`. No `class_map` is needed because the ONNX already outputs the six classes.
`drishyam.segment.default_segmenter` picks these variables up automatically. A v2 model is used the same way
(`models/drishyam_dacl10k_v2a.onnx` and `.fp16.onnx`); `make eval-model ONNX=models/drishyam_dacl10k_v2a.onnx` scores
it on CODEBRIM and synth200 against the classical engine with its own cost table and run folders
(`evals/model_eval.py`), so v1, v2a and v2b can be compared side by side.

On the live site (AWS, `docs/aws.md`) the model is chosen at deploy time: `scripts/build_lambda.sh` packages
every `models/*.onnx`, and `ONNX` (optionally `ONNX_FP16`) sets the function's `DRISHYAM_ONNX`
(`DRISHYAM_ONNX_FP16`) to that file:

```
ONNX=models/drishyam_dacl10k.onnx ONNX_FP16=models/drishyam_dacl10k.fp16.onnx scripts/aws.sh deploy
```

To take the model off the live site later, deploy again without them:

```
scripts/aws.sh deploy
```

To publish the weights outside git, for example on the Hugging Face Hub (they inherit dacl10k's non-commercial licence,
so say **CC BY-NC 4.0 (trained on dacl10k)** in the model card, with the `metrics.json` table, and upload the licence
file with them):

```
hf auth login
hf repo create <user>/drishyam-dacl10k-unet-mnv3 --repo-type model
hf upload <user>/drishyam-dacl10k-unet-mnv3 models/drishyam_dacl10k.onnx
hf upload <user>/drishyam-dacl10k-unet-mnv3 models/drishyam_dacl10k.fp16.onnx
hf upload <user>/drishyam-dacl10k-unet-mnv3 training/class_map.json
hf upload <user>/drishyam-dacl10k-unet-mnv3 models/LICENSE.drishyam_dacl10k
```

## 8. Smoke test on the Intel Mac (no dataset, no downloads)

PyTorch's last x86-64 macOS wheels are 2.2.2 (Python 3.12 or older), which needs numpy < 2, while the repo pins
`opencv-python-headless==5.0.0.93`, which declares numpy >= 2 but runs on numpy 1.26 (checked: resize, DNN). So install
OpenCV with `--no-deps`:

```
uv venv --python 3.12 /tmp/torchenv
uv pip install --python /tmp/torchenv/bin/python torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/torchenv/bin/python "numpy<2" segmentation-models-pytorch==0.5.0 timm==1.0.30 onnx==1.23.2 onnxslim==0.1.98 onnxconverter-common==1.16.0 pytest
uv pip install --python /tmp/torchenv/bin/python --no-deps opencv-python-headless==5.0.0.93

/tmp/torchenv/bin/python training/train_dacl10k.py --synthetic --max-steps 3 --size 128 --batch-size 2 --variant v2a --out /tmp/run
/tmp/torchenv/bin/python training/train_dacl10k.py --synthetic --max-steps 3 --size 128 --batch-size 2 --variant v2b \
    --encoder tu-efficientnet_b2 --out /tmp/run_b
/tmp/torchenv/bin/python -m pytest tests/test_training.py
scripts/install_model.sh /tmp/run/drishyam_dacl10k.zip --n 3
```

These are the notebook's pins, plus timm 1.0.30, the version all 14 tests in `tests/test_training.py` passed with on
torch 2.2.2 (2026-10-19). On Kaggle torch, torchvision and timm stay the image's own; which versions a run used is in
its `run_info.json` (zipped with the weights, and copied into `models/README.md` by `scripts/install_model.sh`).

The install of a v2a zip writes `models/drishyam_dacl10k_v2a.*` (a v1 zip, `--recipe v1`, writes
`models/drishyam_dacl10k.*`), replacing any real trained weights of that name, which git does not hold: run the smoke
test in a checkout without them, or copy them aside first. Then undo the install (smoke weights are noise):

```
git checkout models/README.md
rm -rf models/drishyam_dacl10k_v2a.* models/eval/drishyam_dacl10k_v2a_19.onnx results/local-macos-x86-dacl10k_v2a
```

The first command is the notebook's whole path (train, export into `/tmp/run/onnx`, parity on noise crops,
`/tmp/run/drishyam_dacl10k.zip`); `export_onnx.py` and `parity.py` also run on their own (section 6).

`--synthetic` fabricates random images and masks and forces `--encoder-weights none`: it proves the pipeline runs,
and says nothing about accuracy. On macOS `--workers` defaults to 0: with torch 2.2.2 on Python 3.12.13 a process that used
DataLoader worker processes never exits (a 10-line DataLoader script with no code from here reproduces it), though training
itself completes; Linux/Kaggle defaults to 4 workers and I did not test it there. The main repo venv has no torch, so `tests/test_training.py` runs only its class-map,
notebook and kernel-metadata tests there and skips the rest.
