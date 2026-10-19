# Benchmarks: OpenCV kernels, the agent loop, Arm against x86, Lambda threads, INT8

Two metrics, each with the machine, its load and the file that holds the numbers. Nothing in this page ran on Lambda
or on any AWS instance: this workstream does not deploy or write to AWS.

- **Metric A, kernels** (`bench/kernels.py` -> `results/<arm>/kernels.json`): the pipeline's OpenCV vision kernels in
  isolation, ops/s and p50/p95 ms at 1024x768, 2048x1536 and 4000x3000, on one thread and on all of them, plus the
  DNN forwards.
- **Metric B, the agent loop** (`bench/loop.py` -> `results/<arm>/loop.json`): whole jobs, with and without
  asynchronous look prefetching (`drishyam/prefetch.py`): jobs/s, p50/p95 job wall time and step latency at 1, 2 and 4
  threads.

## Where each number comes from

| Where | CPU, threading | OpenCV | What ran | Files |
| --- | --- | --- | --- | --- |
| This Mac | Intel Core i9-9880H, x86, 8 cores / 16 threads; macOS GCD | opencv-python-headless 5.0.0 wheel (`uv.lock`), IPP HAL | Metric A, Metric B (1 thread and all 16), dacl10k latency, INT8 | `results/local-macos-x86/{kernels,loop,loop_onnx,int8,int8_repeat}.json` |
| GitHub `ubuntu-24.04-arm` | Arm (an earlier run reported Neoverse-N2); Linux pthreads | the same version, aarch64 wheel (KleidiCV and carotene HAL) | Metric A, Metric B at 1, 2, 4 threads, micro, macro | `results/linux-aarch64/` once the owner runs the workflow |
| GitHub `ubuntu-24.04` | x86; Linux pthreads | the same version, x86_64 wheel | the same | `results/linux-x86_64/` once the owner runs the workflow |
| Lambda arm64, 3008 MB | Graviton, 1.7 vCPU of time (below) | the same version, aarch64 wheel | not measured here | the two smoke runs in `docs/report.md` section 5 are the only Lambda timings |

**COOL and AISPL.** The AWS India (AISPL) account could not subscribe to the COOL Marketplace AMI (the agreement was
terminated; a support case is open), so COOL on a Graviton4 instance was not run. The Arm evidence is the deployed
Lambda arm64 function (Graviton) and GitHub's arm64 runner measured against its x86 runner with the same pinned wheel.
`bench/aws_runbook.md` keeps the three-arm COOL plan for when the subscription works.

**This Mac was shared.** Every time the processes were listed during these runs, another workstream had six
single-threaded workers busy on the Mac's 8 cores, and each JSON records the 1/5/15-minute load average at its start
and end (`provenance.load_avg`, `load_avg_end`; 7 to 19 at the 1-minute mark). One-thread numbers are the more
trustworthy; the all-thread numbers had about two free cores plus hyper-threads and understate an idle machine.

Absolute numbers swing between runs far more than the load average suggests. The same Metric B command (classical
segmenter, prefetch off) gave 0.94 jobs/s on one thread in its first run (commit 1670078's `loop.json`) and 0.76 in
the second (the `loop.json` here); on sixteen threads 0.89 and 1.33; a review re-run on one thread, not kept, gave
2.34. CPU seconds per job moved with it (1.04, 1.26 and 0.43 on one thread), so the same work took up to about 3 times
the CPU time, not only more waiting. Every absolute number on this page, Metric A's kernels and the DNN forwards
included, carries that uncertainty: only ratios within one run are comparable, and a difference between variants
smaller than the null arm's own movement in that run (Metric B's `warp`) is noise. macOS's GCD backend runs any
`cv2.setNumThreads` above 1 on every logical CPU, so this Mac measures 1 and all only; 2 and 4 threads come from the
Linux runners.

## Metric A: kernels in isolation

    python -m bench.kernels --arm local-macos-x86 --threads 1,all

Frames are synthetic bridge photos (`evals.synth.generate(9000, w, h)`, not an evaluation split). Each kernel is timed
30 times after 3 warm-ups; ops/s is 1000 / mean ms. `all` is OpenCV's own default thread count (16 here).

- `pyramid`: the agent's 5-level `pyrDown` chain (`imaging.build_pyramid`).
- `clahe_lab`: CLAHE on the L channel of LAB and back (`imaging.enhance`).
- `scharr_tensor`: Scharr x and y gradients and the three Gaussian-smoothed products of the structure tensor.
- `blackhat_tophat`: 15 px elliptical black-hat and top-hat (crack enhancement).
- `canny`: `cv2.Canny(gray, 50, 150)`.
- `components_contours`: `connectedComponentsWithStats`, external `findContours`, then `minAreaRect` and `fitLine`
  (Huber) on every contour of at least 50 points of the black-hat's strongest 1 % (6, 9 and 29 such contours).
- `warp_gaze`: one oblique 512x512 gaze (`imaging.gaze`, yaw 8, pitch 4: `warpPerspective`). Its output size is fixed,
  so its cost barely moves with the frame.
- `blob`: `cv2.dnn.blobFromImage` of the whole frame to a 512x512 ImageNet-normalised tensor.

This Mac, `results/local-macos-x86/kernels.json` (load average 8.3 at the start, 11.3 at the end). The times are one
run's, with the run-to-run swing above (up to about 3 times); the thread ratios within the run are the comparable part:

| kernel | 1024x768, 1 thread, p50 / p95 ms | 16 threads, p50 ms | 2048x1536, 1 thread | 16 threads | 4000x3000, 1 thread | 16 threads |
| --- | --- | --- | --- | --- | --- | --- |
| pyramid | 2.4 / 2.7 | 0.8 (x3.1) | 11.1 / 14.4 | 3.0 (x3.7) | 41.4 / 45.1 | 10.1 (x4.1) |
| clahe_lab | 14.9 / 16.5 | 5.3 (x2.8) | 67.0 / 80.4 | 19.0 (x3.5) | 220.4 / 246.6 | 55.5 (x4.0) |
| scharr_tensor | 17.5 / 23.8 | 20.1 (x0.9) | 53.0 / 71.7 | 33.1 (x1.6) | 252.0 / 283.0 | 129.5 (x1.9) |
| blackhat_tophat | 7.1 / 8.2 | 8.0 (x0.9) | 34.9 / 40.6 | 15.0 (x2.3) | 131.5 / 140.3 | 42.6 (x3.1) |
| canny | 3.8 / 4.7 | 2.8 (x1.4) | 12.8 / 14.6 | 7.4 (x1.7) | 42.4 / 47.5 | 15.3 (x2.8) |
| components_contours | 8.2 / 9.9 | 7.2 (x1.1) | 32.4 / 38.1 | 27.6 (x1.2) | 102.8 / 111.9 | 86.6 (x1.2) |
| warp_gaze | 2.4 / 2.9 | 0.5 (x5.2) | 2.3 / 3.3 | 0.5 (x4.7) | 2.1 / 2.9 | 0.6 (x3.7) |
| blob | 3.5 / 5.1 | 4.1 (x0.9) | 3.7 / 7.7 | 3.7 (x1.0) | 4.9 / 8.8 | 5.3 (x0.9) |

The x factor is the 1-thread p50 over the 16-thread p50. Read with the load above: below x1 is what threads cost on
this busy machine at the small frames; an idle machine was not measured. `components_contours` hardly scales: its
contour walk and the per-contour Python loop are sequential (not profiled further here).

DNN forwards (fixed inputs; NanoDet-Plus as `privacy.py` runs it, the dacl10k segmenter read with `ENGINE_NEW` from
`DRISHYAM_ONNX`, which points at the uncommitted weights):

| forward | input | 1 thread, p50 / p95 ms | 16 threads, p50 / p95 ms |
| --- | --- | --- | --- |
| NanoDet-Plus | 416x416 | 44.9 / 50.3 | 23.2 / 29.2 |
| dacl10k FP32 (the `fast` size) | 256x256 | 177.2 / 199.3 | 64.9 / 73.3 |
| dacl10k FP32 (`accurate`) | 512x512 | 704.9 / 814.5 | 198.2 / 209.2 |

## Metric B: the agent loop, with and without prefetching

    python -m bench.loop --arm local-macos-x86 --threads 1,all
    DRISHYAM_ONNX=/path/to/drishyam_dacl10k.onnx python -m bench.loop --arm local-macos-x86 --threads 1,all --synth 2

**What prefetching does.** While the agent segments its current look, one background thread (`concurrent.futures`)
warps, and in the `warp+segment` variant also segments, the looks it is likely to take next: the head of its queue of
forced looks and the 2 best unvisited candidates by the agent's own score. Results are cached by
(cx, cy, level, size, model) and handed back only when the agent asks for exactly that look and the worker has started
it. A cached look still queued behind another (one worker) is cancelled and computed inline, counted `late`: waiting
for it would cost the other look's warp and forward first. The hook is one call in
`Job._timed`, off by default (`Job.prefetch = None`); nothing in the app turns it on.

**Decisions do not change.** A cached look is the same function of the same inputs, computed earlier on another
thread, and the prefetcher only reads the job. `tests/test_prefetch.py` runs bridge_03 and two synthetic photos with
the default agent (governor on, live costs, thumbnails on) on a deterministic clock and requires the whole event
stream, gaze path, causes, rationales, findings and work order, to be byte-identical with prefetching on and off, and
the cache to have been hit. It also checks every look served from the cache against the agent's own call for it:
the warp's crop and homography against an inline `imaging.gaze`, and the background's `segment_view` inputs
(segmenter, crop, model, image footprint, straight-line scale) against the agent's. A 3000x1000 synthetic photo gives
it a served look that runs past the image's edge. Versions with the wrong model, no footprint, the wrong scale or an
offset gaze each fail it, and a second test fails without the `late` cancel. On a real clock the
agent times its wait, not the work, so in the live cost mode a faster step can change what it does next, as a faster
machine would; that is why Metric B holds decisions fixed (governor overrides off, a fixed t-hat whose value cancels
out of the argmax and the stop rule, a 120 s seconds cap) and checks per photo that every variant's gaze path and
findings equal prefetch-off's (`decisions_identical`).

**What is measured.** The eight demo photos and 8 synthetic 2048x1536 photos (seeds 9100 to 9107, outside every eval
split), decoded before timing, 30 % pixel cap, thumbnails off, every photo once per variant, interleaved and rotated.
jobs/s is photos over summed job wall time in one process; step latency is the wall time between consecutive step
records; CPU s per job is process CPU time, so speculation the agent never used is counted.

This Mac, two rounds (`results/local-macos-x86/loop.json`, classical segmenter, 16 photos, load average 9.4 at the
start):

| threads | prefetch | jobs/s | job p50 / p95 ms | step p50 / p95 ms | CPU s per job | looks served from the cache | speculative looks (dropped unread) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | off | 0.755 | 1086 / 2548 | 281 / 455 | 1.26 | | |
| 1 | warp | 0.760 | 1075 / 2547 | 283 / 453 | 1.26 | 38 of 142 (0.27) | 116 (28) |
| 1 | warp+segment | 0.868 | 1061 / 2062 | 250 / 463 | 1.75 | 38 of 142 (0.27) | 116 (28) |
| 16 | off | 1.333 | 599 / 1317 | 155 / 232 | 0.94 | | |
| 16 | warp | 1.296 | 574 / 1358 | 156 / 238 | 0.97 | 38 of 142 (0.27) | 116 (28) |
| 16 | warp+segment | 1.415 | 608 / 1079 | 170 / 268 | 1.38 | 38 of 142 (0.27) | 116 (28) |

The dacl10k segmenter (`results/local-macos-x86/loop_onnx.json`, the 8 samples and 2 synthetic photos, two rounds,
load average 10.6 at the start and 18.7 at the end):

| threads | prefetch | jobs/s | job p50 / p95 ms | step p50 / p95 ms | CPU s per job | looks served from the cache | speculative looks (dropped unread) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | off | 0.277 | 2997 / 6822 | 874 / 1132 | 3.56 | | |
| 1 | warp | 0.283 | 3106 / 6379 | 868 / 1030 | 3.50 | 42 of 78 (0.54) | 80 (10) |
| 1 | warp+segment | 0.368 | 2345 / 4659 | 845 / 1137 | 4.99 | 40 of 78 (0.51), 2 `late` | 80 (10) |
| 16 | off | 0.856 | 1097 / 2055 | 282 / 369 | 7.73 | | |
| 16 | warp | 0.862 | 1095 / 1968 | 282 / 361 | 7.80 | 42 of 78 (0.54) | 80 (10) |
| 16 | warp+segment | 0.887 | 1046 / 1713 | 304 / 579 | 9.87 | 40 of 78 (0.51), 2 `late` | 80 (10) |

Every variant made the same decisions as prefetch-off on every photo, at both thread counts.

**How to read it.** The `warp` variant is the null arm: a warp costs about 2 ms of a 150 to 1000 ms step, so
prefetching it cannot move any metric by more than about 1 %, yet against off it moved metrics by up to 9 % in this
run and up to 22 % in the first (commit 1670078). Every metric of `warp+segment` below is read against it, losses
included, and the first run is given where it differs:

- **Classical segmenter: no reliable gain, 40 to 60 % more CPU, slower steps on 16 threads.** This run: jobs/s +15 %
  on one thread and +6 % on sixteen, job p95 -19 % and -18 %, CPU per job +39 % and +47 % (the null arm within 4 %
  on all of these). On sixteen threads steps got slower: step p50 +10 % and p95 +16 % (null +1 % and +3 %). The first
  run of the same code (no look was ever `late` with this segmenter, so the change below does not touch it) gave
  jobs/s -2 % and +3 % and, on one thread, losses: job p50 +18 % and step p95 +33 % (null +8 % and +22 %); on sixteen,
  step p95 +10 % (null -13 %). A review re-run on one thread, not kept, also lost: job p50 +22 % and step p95 +26 %
  (null -1 % and +2 %), CPU +55 %. So in every run CPU rose 40 to 60 % and job p95 fell 7 to 19 %; jobs/s went
  either way (-2 % to +15 %); the one-thread job p50 and step p95 rose by 18 to 33 % in two of the three runs and
  stayed within 2 % in this one. Why there is little to gain: of one round's 71 looks, 15 were each photo's first
  look (the survey: nothing runs before it to overlap with), 35 were escalations and verifications, whose centre comes
  from the current look's mask and so cannot be known before that look is segmented, and 21 were the argmax's picks,
  of which the cache held 19 (a one-off count with the same photos and settings). Photos end after 4 to 5 looks at the
  30 % cap, and a segmentation of 100 to 200 ms leaves little to hide.
- **dacl10k on one thread: +33 % jobs/s** (0.368 against 0.277; the first run +34 %, an earlier single-round run, not
  kept, +25 %), job p50 -22 % and p95 -32 %, for 40 % more CPU per job. Step latency did not improve: step p50 -3 %
  and p95 +0 % (null -1 % and -9 %; the first run -5 % and -20 % with null -6 % and -22 %). The gain is in whole jobs:
  its profile's looks are more often argmax picks (22 of 39 per round; 21 of them were in the cache, 20 served and 1
  still queued), and each hit hides a forward of about 0.7 s behind the current one.
- **dacl10k on all 16 threads: a loss at the step.** jobs/s +4 % (the first run -1 %) and job p95 -17 % (the first
  run 0 %), but step p95 rose from 369 to 579 ms, +57 %, past the 500 ms SLA (the first run 291 to 495 ms, +70 %; the
  null arm -2 % and +2 %), for 28 % more CPU. A one-off split of the segment op (same photos and settings, two rounds,
  not kept) shows where: off, p50 263 / p95 365 ms; with prefetching, the 39 looks served from the cache waited
  p50 43 / p95 268 ms, and the 39 computed inline took p50 381 / p95 560 ms (worst 941): the main thread's forward
  runs while a background forward takes cores from it.

**Head-of-line waits.** The first version handed back any cached look, even one the single worker had not started
because it was still busy with another look's warp and forward, so such a hit could take about one forward longer than
computing the look inline (a review probe measured 0.8 to 1.0 s of warp-hit waiting per job on photos with one such
hit). Such a look is now `late`: cancelled and computed inline. It happened to 2 of the 42 cached dacl10k looks at
each thread count and to none of the classical ones. The 16-thread step p95 rise remained after the change (+57 %
against +70 % before), so these waits were not its cause.

So prefetching pays only where a look is expensive and a core is idle while it runs: one inference thread with a spare
core, and there it shortens whole jobs, not steps. With every core busy it makes steps slower. On Lambda's 1.7 vCPU
the second thread would have at most about 0.7 of a core, and with the classical segmenter the deployed function runs
it buys no reliable gain for 40 to 60 % more CPU; it stays off.

## Arm against x86 on GitHub

`.github/workflows/bench-arm64.yml` (manual only) runs `bench/run_all.sh` on `ubuntu-24.04-arm` (`linux-aarch64`) and
`ubuntu-24.04` (`linux-x86_64`): micro, macro, parity, Metric A at 1 thread and all, Metric B at 1, 2 and 4 threads.
Both install the same `opencv-python-headless` version from `uv.lock` with `--locked` (each architecture's own wheel
of it); every JSON quotes the build's baseline, dispatch, parallel framework and HAL lines and the runner's
`logical_cpus`, and `provenance.label` says it is a GitHub runner, not Lambda. The dacl10k weights are not in the
repository, so no trained-model row runs there.

The owner runs it, from the repository root:

    gh workflow run bench-arm64.yml -f n=200
    gh run list --workflow bench-arm64.yml --limit 1
    gh run watch RUN_ID
    gh run download RUN_ID -n bench-linux-aarch64 -D runs/gh/linux-aarch64
    gh run download RUN_ID -n bench-linux-x86_64 -D runs/gh/linux-x86_64

Each artifact holds `micro.json`, `macro.json`, `kernels.json` and `loop.json` at its top level. Copy the files into
`results/linux-aarch64/` and `results/linux-x86_64/` to publish them:

    cp runs/gh/linux-aarch64/kernels.json runs/gh/linux-aarch64/loop.json results/linux-aarch64/
    cp runs/gh/linux-x86_64/*.json results/linux-x86_64/

Copy `results/linux-aarch64/micro.json` and `macro.json` only on purpose: the evaluation harness charges its fixed
costs from that `micro.json` (`MICRO_ARM=linux-aarch64` in the Makefile), so replacing it changes the modelled wall
times of every later eval run (not its decisions). The macro step builds and runs `data/synth200` on the runner, as
the workflow always has.

## Lambda: threads and the 500 ms step

**Memory buys CPU.** "At 1,769 MB, a function has the equivalent of one vCPU (one vCPU-second of credits per second)"
([Lambda memory](https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html)); memory goes up to
10,240 MB, which AWS announced as up to 6 vCPUs. The function's 3008 MB is therefore 3008 / 1769 = 1.70 vCPU-seconds
per second: no thread count can make it more than 1.7 times faster than one thread. How many CPUs the function's
Linux reports (`os.cpu_count()`, which OpenCV's default follows) was not measured.

**`DRISHYAM_CV_THREADS`.** `drishyam/web.py` reads it at startup and calls `cv2.setNumThreads` with it; the stack sets
it to 2 for the function (`infra/drishyam_stack.py`), the whole-CPU count nearest 1.7. When it is set, the cold start
logs `DRISHYAM_CV_THREADS=2: cv2.getNumThreads() = N`, which is how to check on the function that it took. Unset, as
everywhere but the stack, nothing changes. Not deployed by this workstream.

**What the dacl10k FP32 512 net would need.** The governor compares the p95 of the segment op (the forward plus its
post-processing) with `sla_ms` 500 and switches to the 256 px `fast` net above it; the PRD's target (NFR-1) is a gaze
step, warp plus FP32 segment at 512, of at most 500 ms p95.

On this Mac a dacl10k segment op is the forward plus about 62 ms of post-processing on one thread (softmax, label
map, connected components; a one-off cProfile of three 512 px `segment_view` calls: 929 ms per call, 868 ms of it in
`forward`). So a step fits 500 ms p95 when the forward's p95 is at most about 500 - 62 - 3 (warp) = 435 ms at the
thread count the function runs. (Whole agent steps take longer than the segment op, about 0.9 to 1.0 s at the median
on one thread in Metric B's two dacl10k runs; the rest lies outside the segment op, in the agent's own Python
(findings, measurement, seeding, comparisons; not broken down here), which the governor's check does not count.)

| dacl10k forward, this Mac | 1 thread, p50 / p95 ms | 16 threads (GCD), p50 / p95 ms | fits a 435 ms p95? |
| --- | --- | --- | --- |
| FP32 at 512 (`accurate`) | 704.9 / 814.5 | 198.2 / 209.2 | undetermined on one thread (below); yes on 16 |
| FP32 at 256 (`fast`) | 177.2 / 199.3 | 64.9 / 73.3 | yes |

2 and 4 threads cannot be set here (GCD); a one-off probe of the 512 forward gave 601 ms at `setNumThreads(1)` and
157, 160 and 166 ms at 2, 4 and 16, the same work on all 16 logical CPUs each time. On the function, 2 threads on
1.7 vCPU can at best divide the one-thread time by 1.7, so the 512 net can fit only if its one-thread forward p95 on
the function's core is under 1.7 x 435 = 740 ms. That is the necessary condition, not a sufficient one, since thread
scaling on a CPU quota is below ideal.

This Mac does not decide it. Its one-thread 512 forward p95s were 764 ms (`int8.json`), 815 ms (`kernels.json`) and
871 ms (`int8_repeat.json`), and a review run, not kept, gave 974 and 1140 ms: all above 740 ms, but all taken at a
load average of 7 to 15, while single forwards ran as fast as 541 ms (the review run), 601 ms (the probe above) and
668 ms (`kernels.json`'s minimum). On an idle core of this speed the p95 could fall on either side of 740 ms. If the
512 net does not fit, the governor switches to the 256 net, which fits with room (one-thread p95 199 ms). Whether a
Graviton core is faster or slower than this i9's for this net was not measured (the weights are not on the GitHub
runner, and nothing ran on Lambda); only a measurement on Graviton or on the function settles it.

## INT8: can OpenCV 5's DNN run a quantized dacl10k?

    DRISHYAM_ONNX=/path/to/drishyam_dacl10k.onnx uv run --with onnxruntime python -m bench.int8 --arm local-macos-x86

`bench/int8.py` quantizes the FP32 model with onnxruntime 1.23.2 into `runs/int8/` (git-ignored; onnxruntime is a
bench tool here and stays off the live path): `quantize_dynamic` (int8 weights), and `quantize_static` calibrated on 8
synthetic 512 px crops in the QDQ format and, for reference, the QOperator format (per tensor: per channel stops inside
onnxruntime 1.23.2's bias requantization). It reads each with `ENGINE_NEW` and `ENGINE_CLASSIC`, times a 512x512
forward 20 times, and compares the label maps with FP32 on `ENGINE_NEW` over 20 other synthetic crops through the
agent's own `OnnxSegmenter` pre- and post-processing (`bench.parity.compare`). This Mac,
`results/local-macos-x86/int8.json` (load average 7.8 at the start; a repeat is below the table):

| model | size | `ENGINE_NEW` | `ENGINE_CLASSIC` | 512 forward, 1 thread, p50 / p95 ms | 16 threads | pixel agreement with FP32 | class IoU with FP32: crack, spalling, corrosion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32, the trained model | 26.8 MB | runs | fails (a) | 686.7 / 764.0 | 186.4 / 209.3 | (reference) | (reference) |
| `quantize_dynamic` | 7.0 MB | fails (b) | fails (c) | | | | |
| `quantize_static`, QDQ | 7.3 MB | runs | fails (a) | 1066.6 / 1137.8 | 314.6 / 341.4 | 0.993 (worst crop 0.963) | 0.37, 0.83, 0.87 |
| `quantize_static`, QOperator | 6.9 MB | runs | fails (a) | 1391.1 / 1823.6 | 522.0 / 635.0 | 0.970 (worst crop 0.913) | 0.52, 0.14, 0.14 |

Neither model labelled any efflorescence or exposed rebar on these crops, so those IoUs are undefined. The exact
errors (the `OpenCV(5.0.0) <build path>` prefix trimmed; `int8.json` has them whole):

- (a) every model on `ENGINE_CLASSIC`, the FP32 one included: `Node [Shape@ai.onnx]:(onnx_node!/net/decoder/Shape)
  parse error: ... onnx_importer.cpp:2591: error: (-215:Assertion failed) !isDynamicShape in function 'parseShape'`,
  after the log line `DNN/ONNX(Shape): dynamic 'zero' shapes are not supported`. The classic importer stops at the
  decoder's `Shape` node before it reaches any quantized op.
- (b) `ENGINE_NEW` on the dynamic model: the new parser logs `the model ... cannot be loaded with the new parser.
  Trying the older parser. Unsupported operations: DynamicQuantizeLinear`, the older parser fails as in (c), and the
  returned net is empty: `net.cpp:80: error: (-215:Assertion failed) !empty() in function 'forward'`.
- (c) `Node [DynamicQuantizeLinear@ai.onnx]:(onnx_node!input_QuantizeLinear) parse error: ... net_impl.hpp:187:
  error: (-2:Unspecified error) Can't create layer "onnx_node!input_QuantizeLinear" of type "DynamicQuantizeLinear"
  in function 'getLayerInstance'`.

A repeat the same day (`results/local-macos-x86/int8_repeat.json`, load average 15.4 at the start, its quantized
files in a scratch directory) reproduced every model size, error message, pixel agreement and IoU exactly. Its 512
forwards (p50 / p95 ms) were slower throughout: FP32 786.6 / 871.3 at one thread and 203.7 / 220.7 at sixteen; QDQ 1894.6 /
2000.1 and 664.8 / 780.6; QOperator 2099.0 / 2435.0 and 630.7 / 717.2.

So OpenCV 5.0.0 runs the static INT8 models only on `ENGINE_NEW`, and on this x86 machine both are slower than FP32:
1.5 to 2.7 times at one thread and 1.7 to 3.3 times at sixteen (p50 ratios; `int8.json` x1.55 and x2.0, then x1.7
and x2.8; the repeat x2.4 and x2.7, then x3.3 and x3.1). Which of the two is slower did not reproduce (at sixteen
threads the repeat reversed the order, and a review repeat, not kept, reversed it at one thread, x1.60 against x1.50),
so they are not ranked. They are also less faithful where it matters most: the QDQ model's crack masks overlap
FP32's at IoU 0.37, because a crack is a few pixels wide and quantization moves its edge. INT8 does not help the SLA
above on this machine; onnxruntime stays out of the live path.

## What is not measured

- Lambda itself: no timing in this page comes from the function; the GitHub arm64 runner is a different VM.
- 2 and 4 threads on this Mac (GCD), and any number on an idle Mac: the machine was shared throughout.
- The GitHub Arm and x86 runs: the workflow is ready and not triggered; the owner runs it.
- The dacl10k net on Arm, FP32 or INT8: the x86 INT8 result above does not predict it.
