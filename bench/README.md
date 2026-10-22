# Benchmarks

Three arms separate what COOL (KleidiCV-accelerated OpenCV on Graviton) buys from what the hardware buys. Everything here is documentation plus local scripts: **nothing in this repo touches AWS**; you launch the arms yourself.

| Arm | Instance | OpenCV | Isolates |
| --- | --- | --- | --- |
| A | c8g.2xlarge, COOL AMI | COOL build, from the `/opt/cool` venv | the claimed workload |
| B | c8g.2xlarge, Ubuntu 24.04 | stock `opencv-python-headless` 5.x wheel, pinned | COOL vs. Graviton alone (same hardware as A) |
| C | c7i.2xlarge, Ubuntu 24.04 | the same stock wheel version, x86 build | Graviton vs. x86 (B's wheel version built for x86, so build differences count too) |

A vs. B is the COOL effect, B vs. C is the hardware effect, A vs. C is what a buyer sees. Hourly cost before any speed-up: c8g.2xlarge $0.339 (on-demand $0.319 + $0.02 COOL fee), c7i.2xlarge $0.357; `--usd-per-hour` sets the rate per arm.

## What runs

`bench/run_all.sh <arm>` writes `results/<arm>/`:

- **`micro.json`** (`bench/micro.py`): p50/p95/p99 ms of each op on the gaze and saliency path (`pyrDown`, `warpPerspective`, `remap`, `resize`, `GaussianBlur`, `Sobel`, `adaptiveThreshold`, `morphologyEx`, CLAHE, `inRange`, `cvtColor`, plus `resize_area` and an HSV `cvtColor`) at 1024x768, 2048x1536 and 4000x3000, with warm-up, 50 repeats and a fixed `cv2.setNumThreads` (default 1). The count is read back with `getNumThreads()` and the run exits if it did not take; `config` records `requested_threads` and `cv2_threads`. On macOS's GCD backend `getNumThreads()` reports the CPU count for any n > 0; `setNumThreads(1)` still runs sequentially (one OS thread, measured with `ps -M`), and `--threads 1` sets 0 there so the recorded value is 1 (`THREADS` above 1 exits there, since the read-back cannot confirm it). `results/local-macos-x86/micro.json` predates this read-back: it records `requested_threads` 1 and `cv2_threads` 16 from that read-back, but ran single-threaded. Its `job_ops` were timed at a 1-minute load average of 23.8, which is the reason to re-time it at a load below 4. The two gaze ops render the 512x512 view the agent really asks for, so their cost does not grow with the source size. Its `dnn` block times bare OpenCV DNN forwards: the NanoDet-Plus privacy detector on its 416x416 input, and, when `DRISHYAM_ONNX` / `DRISHYAM_ONNX_FP16` name a segmentation ONNX, that model read with `ENGINE_NEW` at 256 and 512 (`onnx:fp32@512`, `onnx:fp16@256`, ...). `python -m bench.micro --arm <arm> --dnn-only` re-times only those rows.
- **`macro.json`** (`bench/macro.py`): the full Drishyam job (rule planner, no thumbnails) on the eval images at a fixed pixel cap: images/s, gazes/s, per-image p50/p95/p99, CPU use, `$/1k images` = (instance $/h) x hours / images, and an Amdahl split of the time into KleidiCV-covered imgproc ops, segmentation, `measure()` thinning, image decode and the remainder. With `WORKERS="1 2 4 8"` it also writes `macro_w<N>.json` for the thread-scaling sweep (N single-threaded worker processes).
- **`kernels.json`** (`bench/kernels.py`, Metric A) and **`loop.json`** (`bench/loop.py`, Metric B): the pipeline's OpenCV kernels in isolation (ops/s, p50/p95 ms at three sizes, one thread and all), and the whole agent loop with and without asynchronous look prefetching (`drishyam/prefetch.py`) at 1, 2 and 4 threads. `bench/int8.py` (not in `run_all.sh`: it needs the dacl10k weights and onnxruntime) quantizes the segmenter and checks whether OpenCV's DNN runs it. What each measures, how to reproduce it and the numbers: [docs/bench.md](../docs/bench.md).
- **`opencv5_engines.json`** (`bench/opencv5_engines.py`, not in `run_all.sh`: it needs the v1 weights): whether
  OpenCV 5's `ENGINE_CLASSIC` and `ENGINE_NEW` read the v1 FP32 and FP16 files and a static-shape copy of each, their
  forward p50/p95 at 512 and 256 on one thread and all, and max |dprob| against the deployed net. [repro/](repro/)
  holds a weights-free repro of the classic engine's failure for an upstream report.
- **`opencv5_features.json`** (`bench/opencv5_features.py`, not in `run_all.sh`): each OpenCV flag of
  `drishyam/segment.py` against the NumPy form it replaces (the agent's traces compared byte for byte on the demo
  samples and a synth_dev slice, and its unit timed in interleaved rounds with a null slot); with the v1 weights, v1's
  time per layer on `ENGINE_NEW` (OpenCV 5's profiling mode) and `ENGINE_CLASSIC`, and the agent on `ENGINE_CLASSIC`.
- **`opencv_builds.json`** (`bench/opencv_builds.py`, not in `run_all.sh`): the pipeline's stages under
  opencv-python-headless 4.14 (installed by `uv` into `runs/`) and with the IPP HAL off, against the locked 5.0
  wheel, in interleaved worker processes: time, and whether each stage's output is 5.0's.
- **`parity.npz`** (`bench/parity.py --save`): label maps of 24 fixed crops. Compare two arms with `python -m bench.parity --compare results/A/parity.npz results/B/parity.npz` for pixel agreement and per-class IoU. Pass `PARITY_INPUTS=results/A/parity.npz` when running B and C so every arm segments the *same pixels*; otherwise each arm renders its own crops with its own OpenCV and the comparison includes renderer differences (`inputs_identical` in the output says which).

Every JSON carries a provenance block: `cv2.__version__`, the sha256 of `cv2.getBuildInformation()` and its Baseline, Dispatched code generation, Parallel framework and Custom HAL lines verbatim (`cv2_baseline`, `cv2_dispatch`, `cv2_parallel_framework`, `cv2_custom_hal`), `cv2.__file__` and whether it lives under `/opt/cool`, platform, CPU model, Python and NumPy versions, thread count, timestamp, the 1/5/15-minute load average at the start (`load_avg`) and at the end (`load_avg_end`, beside the provenance block here; the eval summaries and `ablations.json` write `load_avg_start` and `load_avg_end` in it, with `load_avg` equal to the start) and `BENCH_LABEL` as `label` when set. With `DRISHYAM_ON_EC2=1` it also records instance type, AMI id and instance id from IMDSv2; without that variable IMDS is never called. `MAX_LOAD=4` makes micro and macro exit before timing if the 1-minute load average is 4 or more; the check runs in the timing process itself, so other work cannot start between the check and the launch.

## Arm64 on GitHub's runner

`.github/workflows/bench-arm64.yml` is a manual (`workflow_dispatch`) job that runs `bench/run_all.sh linux-aarch64` on GitHub's hosted `ubuntu-24.04-arm` runner and uploads `results/linux-aarch64/micro.json`, `macro.json`, `kernels.json` and `loop.json` as the `bench-linux-aarch64` artifact; the same job runs `bench/run_all.sh linux-x86_64` on the x86 `ubuntu-24.04` runner (artifact `bench-linux-x86_64`) for the Arm-against-x86 comparison ([docs/bench.md](../docs/bench.md)). That runner is a GitHub Actions VM, not the deployed Lambda, and the files say so in `provenance.label`; copy them into `results/linux-aarch64/` to publish them, never over its `micro.json`, the fixed cost table ([docs/bench.md](../docs/bench.md) has the commands and run 39220769299's results). They time the stock aarch64 wheel from `uv.lock`; its `cv2_custom_hal` line records which HAL that wheel was built with. The segmenter rows time the classical engine: the dacl10k ONNX files are not in the repository, so no trained-model DNN row runs there until they are fetched in the job (a release asset) or committed. The optional `engines` job (`-f engines=true`) does fetch them from the v1.1.0 release, checks their sha256, and runs only `bench/opencv5_engines.py` (artifacts `opencv5-engines-linux-aarch64` and `opencv5-engines-linux-x86_64`); with `-f features=true` it also runs `bench/opencv5_features.py --threads 1,2,4` (artifacts `opencv5-features-*`). The `builds` job (`-f builds=true`, no weights) runs `bench/opencv_builds.py` (artifacts `opencv-builds-*`).

The DNN-only workload (the v1 segmenter's forward pass at 512x512 and 256x256, FP32 and FP16) is not in `run_all.sh`: `bench/opencv5_engines.py` times it, locally and in the `engines` job.

## Running the arms on AWS

[aws_runbook.md](aws_runbook.md) is the copy-paste sequence: a $50/$100 budget with a stop action, an instance role limited to SSM plus one S3 prefix, one shared dataset and parity-crop file uploaded from your laptop, the three launches (each instance terminates itself after 3 hours), `run_all.sh` on each arm through SSM ([aws_arm.sh](aws_arm.sh)), the copy back and the termination. Expect about $1 per arm at most. Back on the laptop, `python -m bench.compare_arms results/A results/B results/C` folds the three folders into `docs/cool_benchmark.md` and `public/results/cool.json` (FIG. 4.3 on the website). It reads each arm's `parity.npz`, which git does not keep, so keep those files to re-run it. It refuses arms that ran different OpenCV major versions or different images.

Run the arms on otherwise idle instances (`MAX_LOAD=1` refuses a busy one). Micro numbers are single-threaded by default so that a kernel's speed-up is not blurred by thread scheduling; use `THREADS=8` as a second pass if you want multi-threaded latency (Linux; GCD on macOS cannot be held to 8).

## Reading the numbers

- Report what the JSON says. A micro speed-up on one op is not an end-to-end speed-up: use the Amdahl split in `macro.json` to see how much of a job those ops are.
- On GitHub's arm64 runner (Neoverse-N2, one thread, the stock wheel with KleidiCV 26.03; 200-image macro, `results/linux-aarch64/macro.json`) the split puts 8 % of a job in KleidiCV-covered imgproc, 74 % in the classical segmenter, 4 % in crack thinning, 7 % in decode and 7 % in Python overhead, so accelerating the covered ops 1.8× makes a whole job about 1.04× faster, whatever the micro table shows. The local x86 machine (8-image macro, `results/local-macos-x86/macro.json`: 9 %, 69 %, 3 %, 8 % and 12 %) gives the same 1.04×. Neither is a Graviton instance.
- Throughput is measured from the first job's start to the last job's end, so process start-up is excluded; `$/1k images` bills the whole instance, so a single worker on an 8-vCPU instance is the pessimistic figure and the `WORKERS` sweep is the realistic one.
