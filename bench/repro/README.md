# OpenCV 5.0.0 DNN: three problems found with Drishyam's segmenters

Written so the owner can file them upstream; nothing here has been filed. Measured with the stock
`opencv-python-headless` 5.0.0.93 wheel (cv2 5.0.0), onnx 1.23.2, NumPy 2.5.4, Python 3.13.16, on macOS x86_64
(Intel i9-9880H). The Linux x86_64 and aarch64 runs are the `engines` and `features` jobs of
`.github/workflows/bench-arm64.yml` (runs 39617514621 and 39665440222).

## 1. `ENGINE_CLASSIC` cannot import a `Shape` node whose input has a symbolic H or W

Repro, with nothing but OpenCV and onnx installed:

```bash
python bench/repro/classic_shape.py --out /tmp/classic_shape
```

[classic_shape.py](classic_shape.py) builds three tiny opset-17 models with `onnx.helper`: the pattern PyTorch
exports for `F.interpolate(y, size=x.shape[-2:])`, which is what a UNet decoder does at every skip connection:

    x [1, 3, H, W] -> AveragePool 2x2 -> y
    Shape(x) -> Gather(2), Gather(3) -> Unsqueeze -> Concat([1, 3], h, w) -> Resize(y, sizes=...) -> z [1, 3, H, W]

declared with symbolic H and W (`dynamic_hw`, as `torch.onnx.export(dynamic_axes=...)` writes them), with only the
batch symbolic (`dynamic_batch`), and fully static (`static`). It reads each with
`cv2.dnn.readNetFromONNX(path, engine=...)` on both engines and checks every forward against NumPy. Output (paths to
the build tree shortened to `<opencv>`):

    [ERROR:0@0.312] global onnx_importer.cpp:2590 parseShape DNN/ONNX(Shape): dynamic 'zero' shapes are not supported, input x [<empty>]
    [ERROR:0@0.312] global onnx_importer.cpp:1095 handleNode DNN/ONNX: ERROR during processing node with 1 inputs and 1 outputs: [Shape]:(onnx_node!shape_of_x) from domain='ai.onnx'
    dynamic_hw    [1, 3, 'h', 'w']     ENGINE_CLASSIC fails to load: OpenCV(5.0.0) <opencv>/modules/dnn/src/onnx/onnx_importer.cpp:1114: error: (-2:Unspecified error) in function 'handleNode' > Node [Shape@ai.onnx]:(onnx_node!shape_of_x) parse error: OpenCV(5.0.0) <opencv>/modules/dnn/src/onnx/onnx_importer.cpp:2591: error: (-215:Assertion failed) !isDynamicShape in function 'parseShape' >
    dynamic_hw    [1, 3, 'h', 'w']     ENGINE_NEW     loads; forward (1, 3, 64, 48), max |z - numpy| 1.19e-07
    dynamic_batch ['n', 3, 64, 48]     ENGINE_CLASSIC loads; forward (1, 3, 64, 48), max |z - numpy| 1.19e-07
    dynamic_batch ['n', 3, 64, 48]     ENGINE_NEW     loads; forward (1, 3, 64, 48), max |z - numpy| 1.19e-07
    static        [1, 3, 64, 48]       ENGINE_CLASSIC loads; forward (1, 3, 64, 48), max |z - numpy| 1.19e-07
    static        [1, 3, 64, 48]       ENGINE_NEW     loads; forward (1, 3, 64, 48), max |z - numpy| 1.19e-07

**Cause.** In the 5.0.0 source (`modules/dnn/src/onnx/onnx_importer.cpp`), the classic importer stores each graph
input dim as its `dim_value()`, which is 0 for a `dim_param` dim; a symbolic batch dim alone does not mark the
input dynamic, and the batch is then set to 1, which is why `dynamic_batch` loads. `parseShape` folds a `Shape` node
into a constant at import time and asserts that no dim of its input is 0 (`isDynamicShape |= (sz == 0)`), so a
symbolic H or W stops the import at the first `Shape` that reads it, directly or through layers whose shapes were
propagated from the input. The log prints that shape as `[<empty>]`. The same check and message are in the 4.x
branch's importer, so this is the classic importer's long-standing limit, not a 5.0 regression; `ENGINE_NEW` (and
the default `ENGINE_AUTO`, which tries it first) evaluates `Shape` at run time.

**In Drishyam's v1 model.** The v1 segmenter (`drishyam_dacl10k.onnx`, a UNet on MobileNetV3-Large, PyTorch 2.11
export, opset 17) declares its input `[1, 3, 'h', 'w']` on purpose: the agent runs it at 512x512 (accurate) and its
FP16 copy at 256x256 (fast). It has five such chains, `/net/decoder/Shape` on the graph input and `Shape_2`, `_4`,
`_6` and `_8` on encoder outputs, each `Shape -> Gather -> Unsqueeze -> Concat -> Resize` (nearest, asymmetric,
floor). The FP32 file stops at `/net/decoder/Shape` (input `input [<empty>]`), the FP16 file at
`/net/decoder/Shape_2` (input `/net/encoder/model/blocks.0/blocks.0.0/Add_output_0 [<empty>]`), because the two
files order their nodes differently. `bench/opencv5_engines.py` records both errors and logs whole
(`results/local-macos-x86/opencv5_engines.json`).

**Workaround, measured.** Setting the input's two symbolic dims to the size the net will run at, and changing
nothing else (`bench.opencv5_engines.static_copy`), makes the FP32 file load and run on `ENGINE_CLASSIC`: its
softmax output differs from the deployed file on `ENGINE_NEW` by at most 2.1e-5 at 512 and 1.4e-5 at 256 (8 crops,
every argmax equal). A static copy runs at that size only, so the agent would need one per size. The FP16 file's
static copy loads on `ENGINE_CLASSIC` but its first forward fails in the stem convolution's type check (a second,
separate limit; not checked against 4.x):

    OpenCV(5.0.0) <opencv>/modules/dnn/src/layer.cpp:274: error: (-2:Unspecified error) in function 'virtual void cv::dnn::dnn5_v20260605::Layer::getTypes(<arguments shortened>) const'
    >     'input == CV_32F || input == CV_64F || input == CV_8S || input == CV_8U || input == CV_64S'
    > where
    >     'input' is 7 (CV_16FC1)

after the log line `[Convolution]:(onnx_node!/net/encoder/model/conv_stem/Conv): getMemoryShapes() throws exception`.

**What an upstream report could ask for.** Not a crash and not a regression, so a feature request or a docs note:
either `ENGINE_CLASSIC` keeps a symbolic dim as a run-time `Shape`, or its error names the symbolic dims and points
to `ENGINE_NEW` (the message today shows the shape as `[<empty>]` and does not say which dims are symbolic). The FP16
convolution type check on `ENGINE_CLASSIC` is a separate report. Search the tracker first: the
"dynamic 'zero' shapes are not supported" message predates 5.0. Attach the script's output, the three `.onnx` files
it writes and the `getBuildInformation()` lines in `opencv5_engines.json`'s `provenance`.

## 2. A DeepLabV3+ export did not match PyTorch on `ENGINE_NEW` (not reproducible from this repository)

What is recorded, all of it in the message of commit 95741ba and a comment in `training/train_dacl10k.py` (the
`VARIANTS` table): a DeepLabV3+ (`--arch deeplabv3plus`, `segmentation_models_pytorch` `DeepLabV3Plus` on the
default `tu-mobilenetv3_large_100` encoder unless a flag changed it) was tried for v2c and dropped because "its
merged FP32 export fails parity on OpenCV 5.0.0 ENGINE_NEW (max|dprob| 0.85) while its FP16 and 19-logit exports
pass". "Merged" means the export with `training/dacl_common.py`'s `DrishyamHead` in the graph (19 dacl10k logits to
6 classes: a zero background logit, `z[:, :1] * 0.0`, then a channel `index_select` and `amax` per class, then
`cat`); "19-logit" is the bare network. The check is `training/parity.py`: the ONNX through `OnnxSegmenter` on
`ENGINE_NEW` against the PyTorch checkpoint, max |dprob| over 20 crops of the demo photos at 512 (accurate) and 256
(fast), tolerance 1e-3 for FP32 and 5e-2 for FP16.

What is not known: which checkpoint (trained or not, which epoch), the crop sizes and input that failed, the
exported file (no file named for DeepLabV3+ is in the main checkout, `runs/` included) and its opset, whether `ENGINE_CLASSIC` (or a static copy of
the file) gives the PyTorch answer, and which node diverges. The pattern (the FP32 merged graph fails while the
FP16 merged graph and the FP32 unmerged graph pass, and the same head passes on v1's UNet) points at an `ENGINE_NEW`
fusion or layout path that the DeepLabV3+ decoder and the merge head trigger together, but that is a hypothesis,
not a finding. To isolate it: export an untrained `DeepLabV3Plus` with and without `DrishyamHead` (`torch`,
`segmentation_models_pytorch` and `timm` are not in this repository's environment), run `training/parity.py` on
both engines, then bisect the graph by exposing intermediate outputs (`onnx.utils.extract_model`) until the first
tensor that differs. Until that is done there is nothing to file.

## 3. `ENGINE_NEW` runs v1's 3x3 convolutions about 3.2 times slower than `ENGINE_CLASSIC` on aarch64

A performance finding, not a bug: the owner kept `ENGINE_NEW` live, so it is reported here. On GitHub's Neoverse-N2
runner (stock 5.0.0 aarch64 wheel, 1 thread, v1 at 512), `Net.setProfilingMode(DNN_PROFILE_DETAILED)` puts v1's twelve
3x3 convolutions at 800.6 ms on `ENGINE_NEW` against 248.5 ms on `ENGINE_CLASSIC` (a static-shape copy of the same
file; at the agent level the traces were identical and max |dprob| 1.5e-5). That is more than the whole gap between
the two forwards; the 1x1 convolutions take the same time on both, and `ENGINE_NEW` is the faster on the 5x5 depthwise
ones. v1's looks on `ENGINE_CLASSIC` took x0.39 of `ENGINE_NEW`'s time on 1 thread and x0.44 on 2. On the x86 runner
the 3x3 convolutions took 388.6 against 208.5 ms. Which kernel each engine picks for them is not recorded. The numbers
are in `results/linux-aarch64/opencv5_features.json` (`profile`, `engine`) and docs/bench.md, "OpenCV 5 features in
the pipeline". An upstream report could attach that JSON and ask whether `ENGINE_NEW`'s 3x3 convolution on aarch64
misses the path `ENGINE_CLASSIC` takes (KleidiCV, carotene or its own NEON code).
