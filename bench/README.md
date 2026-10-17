# Benchmarks

Three arms separate what COOL (KleidiCV-accelerated OpenCV on Graviton) buys from what the hardware buys. Everything here is documentation plus local scripts: **nothing in this repo touches AWS**; you launch the arms yourself.

| Arm | Instance | OpenCV | Isolates |
| --- | --- | --- | --- |
| A | c8g.2xlarge, COOL AMI | COOL build, from the `/opt/cool` venv | the claimed workload |
| B | c8g.2xlarge, Ubuntu 24.04 | stock `opencv-python-headless` 5.x wheel, pinned | COOL vs. Graviton alone (same hardware as A) |
| C | c7i.2xlarge, Ubuntu 24.04 | the same stock wheel, x86 | Graviton vs. x86 (same build as B) |

A vs. B is the COOL effect, B vs. C is the hardware effect, A vs. C is what a buyer sees. Hourly cost before any speed-up: c8g.2xlarge $0.339 (on-demand $0.319 + $0.02 COOL fee), c7i.2xlarge $0.357; `--usd-per-hour` sets the rate per arm.

## What runs

`bench/run_all.sh <arm>` writes `results/<arm>/`:

- **`micro.json`** (`bench/micro.py`): p50/p95/p99 ms of each op on the gaze and saliency path (`pyrDown`, `warpPerspective`, `remap`, `resize`, `GaussianBlur`, `Sobel`, `adaptiveThreshold`, `morphologyEx`, CLAHE, `inRange`, `cvtColor`, plus `resize_area` and an HSV `cvtColor`) at 1024x768, 2048x1536 and 4000x3000, with warm-up, 50 repeats and a fixed `cv2.setNumThreads` (default 1, recorded in the file). The two gaze ops render the 512x512 view the agent really asks for, so their cost does not grow with the source size.
- **`macro.json`** (`bench/macro.py`): the full Drishyam job (rule planner, no thumbnails) on the eval images at a fixed pixel cap: images/s, gazes/s, per-image p50/p95/p99, CPU use, `$/1k images` = (instance $/h) x hours / images, and an Amdahl split of the time into KleidiCV-covered imgproc ops, segmentation, `measure()` thinning, image decode and the remainder. With `WORKERS="1 2 4 8"` it also writes `macro_w<N>.json` for the thread-scaling sweep (N single-threaded worker processes).
- **`parity.npz`** (`bench/parity.py --save`): label maps of 24 fixed crops. Compare two arms with `python -m bench.parity --compare results/A/parity.npz results/B/parity.npz` for pixel agreement and per-class IoU. Pass `PARITY_INPUTS=results/A/parity.npz` when running B and C so every arm segments the *same pixels*; otherwise each arm renders its own crops with its own OpenCV and the comparison includes renderer differences (`inputs_identical` in the output says which).

Every JSON carries a provenance block: `cv2.__version__`, the sha256 of `cv2.getBuildInformation()`, `cv2.__file__` and whether it lives under `/opt/cool`, platform, CPU model, Python and NumPy versions, thread count, timestamp. With `DRISHYAM_ON_EC2=1` it also records instance type, AMI id and instance id from IMDSv2; without that variable IMDS is never called.

The DNN-only workload (segmenter forward pass at 512x512, FP32 and FP16) is not here: the shipped segmenter is the classical one, and the ONNX engine is benchmarked once the trained model lands.

## Running the arms on AWS

[aws_runbook.md](aws_runbook.md) is the copy-paste sequence: a $50/$100 budget with a stop action, an instance role limited to SSM plus one S3 prefix, one shared dataset and parity-crop file uploaded from your laptop, the three launches (each instance terminates itself after 3 hours), `run_all.sh` on each arm through SSM ([aws_arm.sh](aws_arm.sh)), the copy back and the termination. Expect about $1 per arm at most. Back on the laptop, `python -m bench.compare_arms results/A results/B results/C` folds the three folders into `docs/cool_benchmark.md` and `public/results/cool.json`. It refuses arms that ran different OpenCV major versions or different images.

Run the arms on otherwise idle instances. Micro numbers are single-threaded by default so that a kernel's speed-up is not blurred by thread scheduling; use `THREADS=8` as a second pass if you want multi-threaded latency.

## Reading the numbers

- Report what the JSON says. A micro speed-up on one op is not an end-to-end speed-up: use the Amdahl split in `macro.json` to see how much of a job those ops are.
- On the local x86 machine (8-image macro, `results/local-macos-x86/macro.json`) a job spends 9 % of its time in KleidiCV-covered imgproc, 69 % in the classical segmenter, 3 % in crack thinning, 8 % in decode and 12 % in Python overhead. If the same holds on Graviton, accelerating the covered ops 1.8× makes a whole job about 1.04× faster, whatever the micro table shows.
- Throughput is measured from the first job's start to the last job's end, so process start-up is excluded; `$/1k images` bills the whole instance, so a single worker on an 8-vCPU instance is the pessimistic figure and the `WORKERS` sweep is the realistic one.
