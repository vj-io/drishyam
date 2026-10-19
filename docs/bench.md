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
| GitHub `ubuntu-24.04-arm` | Neoverse-N2, 4 logical CPUs; Linux pthreads | the same version, aarch64 wheel (KleidiCV and carotene HAL) | Metric A, Metric B at 1, 2, 4 threads, micro, macro | `results/linux-aarch64/`: `micro.json` (the fixed cost table, 2026-10-18); `kernels.json`, `loop.json`, `macro.json` and `micro_run39220769299.json` from run 39220769299 |
| GitHub `ubuntu-24.04` | AMD EPYC 7763, 4 logical CPUs; Linux pthreads | the same version, x86_64 wheel (IPP HAL) | the same | `results/linux-x86_64/{micro,macro,kernels,loop}.json`, run 39220769299 |
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

Each artifact holds `micro.json`, `macro.json`, `kernels.json` and `loop.json` at its top level. To publish a run:

    cp runs/gh/linux-aarch64/{kernels,loop,macro}.json results/linux-aarch64/
    cp runs/gh/linux-aarch64/micro.json results/linux-aarch64/micro_runRUN_ID.json
    cp runs/gh/linux-x86_64/*.json results/linux-x86_64/

Never copy a run's aarch64 `micro.json` over `results/linux-aarch64/micro.json`: the evaluation harness charges its
fixed costs from that file (`MICRO_ARM=linux-aarch64` in the Makefile), so replacing it changes the modelled wall
times of every later eval run (not its decisions); a run's own table is kept beside it under the run's id. A new
`macro.json` changes the job split that README.md, docs/report.md and bench/README.md quote: `tests/test_bench_doc.py`
fails until they match it, and checks every table below, and every figure the text takes from the run's files and the
cost table, against the JSON (not the Mac's figures quoted from above, nor the earlier macro run's, which is only in
git history). The macro step builds and runs `data/synth200` on the runner, as the workflow always has (timings only;
nothing is scored).

### Run 39220769299

GitHub Actions run 39220769299 (`bench-arm64.yml` on `release/v1.1`, `n=200`; the run id is the owner's, the JSON does
not record it) ran both jobs at the same time on 2026-10-19: the files' start timestamps run from 08:10:33 to 08:13:26
UTC. It is one run per runner and one round of Metric B. The only repeat is the aarch64 micro: the cost table,
`results/linux-aarch64/micro.json`, ran a day earlier on the same build hash. Of its 39 op timings (p50), 37 moved by
less than 3.5 %, and `pyrDown` and the `INTER_AREA` resize at 4000x3000 ran 6.5 % and 18.1 % faster in this run (its
`job_ops` are under "The fixed cost table" below). x86, the kernels, the loop and the macro ran once, so how far they
move between runs on GitHub's VMs is not known (the Mac above moved up to about 3 times).

| | `linux-aarch64` | `linux-x86_64` |
| --- | --- | --- |
| runner (`provenance.label`) | GitHub ubuntu-24.04-arm runner, not Lambda | GitHub ubuntu-24.04 runner, not Lambda |
| CPU (`provenance.cpu`), logical CPUs | Neoverse-N2, 4 | AMD EPYC 7763 64-Core Processor, 4 |
| OS | Linux 6.17.0-1022-azure, glibc 2.39 | the same |
| OpenCV | 5.0.0, stock (not COOL) | 5.0.0, stock (not COOL) |
| `build_info_sha256` | `4460ffe25568d7f496997f0593f568f4d91ee4740c97f057b2986c237e8a386c` | `0ef08003c31127b3b5d34410ab13483c511e73b99e278b1a8abd6ddcb265699a` |
| Baseline; dispatched | NEON FP16; NEON_DOTPROD NEON_FP16 NEON_BF16 | SSE SSE2 SSE3; SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX |
| Custom HAL | YES (carotene (ver 0.0.1) KleidiCV (ver 26.03)) | YES (ipp (ver 0.0.1)) |
| parallel framework | pthreads | pthreads |
| Python; NumPy | 3.13.16; 2.5.4 | 3.13.16; 2.5.4 |
| 1-minute load average, lowest to highest | 0.68 to 1.41 | 1.02 to 1.62 |

The load averages are every reading in the run's four files (start and end of each, and of the micro file's
`job_ops` and DNN blocks). A benchmark on one thread keeps one CPU busy, so about 1 on 4 logical CPUs is the
benchmark itself; the highest readings came after the 4-thread kernels.

**Like with like.** The same wheel version (5.0.0.93, `uv.lock`), Python, NumPy, kernel, code and inputs, and 4
logical CPUs each. The builds differ, as they must across architectures (the hashes differ; the aarch64 one equals the
hash in `results/linux-aarch64/micro.json`, timed on 2026-10-18), and with them the HAL (KleidiCV and carotene against
IPP) and the SIMD baseline. x86's dispatch line lists `AVX512_SKX`, compiled in but not run: the EPYC 7763 is a Zen 3
part, which has no AVX-512 (AMD's specification; the JSON does not record CPU flags). So every ratio below is core,
build and VM together: not Arm's core alone, and not KleidiCV alone, since no run here turns a HAL off. The JSON does
not say whether the x86 VM's 4 logical CPUs are 4 cores or 2 cores with two hardware threads each (an EPYC 7763 core
runs two, a Neoverse-N2 core one), so the 4-thread columns may compare 4 Arm cores with 2 x86 cores.

**Not Lambda.** These are shared GitHub VMs. Their Python is 3.13.16; the function runs Python 3.12
(`lambda_.Runtime.PYTHON_3_12` in `infra/drishyam_stack.py`). The OpenCV wheel is `abi3` (one build per platform
for every CPython from 3.7), so the Python version does not change which OpenCV runs, but the interpreter and NumPy
are each version's own builds. `uv.lock` holds two aarch64 files of the wheel (manylinux2014 and manylinux_2_28); the
Lambda package is installed for the manylinux_2_28 platform (`scripts/build_lambda.sh`), which file the runner or
the package got is not recorded, and neither is the function's build hash. Neoverse-N2 is not a Graviton core
(Graviton2 is Neoverse-N1, Graviton3 V1, Graviton4 V2), and the function has 1.7 vCPU of time, not 4 CPUs.

### Metric A on both runners

The frames and kernels of Metric A above, 30 timed repeats after 3 warm-ups; `all` is 4 threads on both. p50 ms;
x86 ÷ Arm is x86's p50 over Arm's, from the unrounded values: above 1 Arm is faster, below 1 x86 is.

| kernel | size | Arm, 1 thread | x86, 1 thread | x86 ÷ Arm | Arm, 4 threads | x86, 4 threads | x86 ÷ Arm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pyramid | 1024x768 | 0.87 | 0.94 | 1.07 | 0.25 | 0.49 | 2.00 |
| clahe_lab | 1024x768 | 9.12 | 12.4 | 1.36 | 2.77 | 5.89 | 2.12 |
| scharr_tensor | 1024x768 | 7.96 | 7.75 | 0.97 | 8.03 | 10.8 | 1.35 |
| blackhat_tophat | 1024x768 | 9.73 | 5.08 | 0.52 | 9.62 | 5.07 | 0.53 |
| canny | 1024x768 | 1.89 | 1.85 | 0.98 | 0.62 | 0.79 | 1.27 |
| components_contours | 1024x768 | 4.43 | 8.47 | 1.91 | 3.14 | 5.63 | 1.79 |
| warp_gaze | 1024x768 | 1.65 | 2.01 | 1.22 | 0.46 | 0.95 | 2.07 |
| blob | 1024x768 | 0.77 | 1.10 | 1.44 | 0.49 | 1.05 | 2.15 |
| pyramid | 2048x1536 | 3.50 | 3.67 | 1.05 | 0.91 | 1.92 | 2.10 |
| clahe_lab | 2048x1536 | 36.5 | 49.6 | 1.36 | 10.6 | 23.3 | 2.20 |
| scharr_tensor | 2048x1536 | 44.6 | 42.4 | 0.95 | 10.2 | 18.7 | 1.83 |
| blackhat_tophat | 2048x1536 | 38.6 | 20.0 | 0.52 | 13.4 | 14.4 | 1.07 |
| canny | 2048x1536 | 7.91 | 7.66 | 0.97 | 2.59 | 3.28 | 1.27 |
| components_contours | 2048x1536 | 15.9 | 32.4 | 2.04 | 10.1 | 19.1 | 1.90 |
| warp_gaze | 2048x1536 | 1.64 | 2.01 | 1.23 | 0.60 | 0.86 | 1.44 |
| blob | 2048x1536 | 1.80 | 1.22 | 0.68 | 1.07 | 1.10 | 1.03 |
| pyramid | 4000x3000 | 14.6 | 13.8 | 0.95 | 3.41 | 7.17 | 2.10 |
| clahe_lab | 4000x3000 | 143.7 | 193.6 | 1.35 | 41.6 | 90.1 | 2.16 |
| scharr_tensor | 4000x3000 | 141.9 | 122.3 | 0.86 | 47.8 | 70.6 | 1.48 |
| blackhat_tophat | 4000x3000 | 150.8 | 77.3 | 0.51 | 51.6 | 58.6 | 1.14 |
| canny | 4000x3000 | 44.5 | 29.9 | 0.67 | 11.3 | 16.5 | 1.46 |
| components_contours | 4000x3000 | 54.8 | 117.0 | 2.14 | 30.8 | 62.2 | 2.02 |
| warp_gaze | 4000x3000 | 1.65 | 2.01 | 1.22 | 0.59 | 0.86 | 1.45 |
| blob | 4000x3000 | 1.78 | 1.22 | 0.68 | 0.90 | 1.12 | 1.26 |

On one thread each runner wins 12 of the 24 kernel and size pairs, but 6 of the 24 differ by less than 6 % (5 of x86's
wins and 1 of Arm's), close enough to flip in another run (see the repeat above). x86 wins the black-hat and top-hat
pair at every size, where Arm takes about twice as long (0.51 to 0.52), Canny (0.67 to 0.98) and the structure tensor
(0.86 to 0.97) at every size, the blob from the two larger frames (0.68) and the 4000x3000 pyramid (0.95). Arm wins
contours and fitting (1.91 to 2.14), CLAHE in LAB (1.35 to 1.36), the oblique gaze warp (1.22 to 1.23), the blob from
1024x768 (1.44) and the two smaller pyramids (1.05 and 1.07). On 4 threads Arm wins 23 of 24, by 1.03 (the blob at
2048x1536) to 2.20 (CLAHE at 2048x1536); the exception is the black-hat and top-hat pair at 1024x768 (0.53), which
gains nothing from threads at that size on either runner (1.01 and 1.00 times its 1-thread speed). The structure
tensor at 1024x768 is the one kernel that gets slower on 4 threads, 0.72 times its 1-thread speed on x86 and 0.99 on
Arm, so its 1.35 is an x86 loss, not an Arm gain. Elsewhere the difference is thread scaling: from 1 to 4 threads
Arm's pyramid gets 3.5 to 4.3 times faster and x86's 1.9 times, and no x86 kernel more than 2.34 times against up to
4.37 on Arm. Arm's 4.37 (the structure tensor at 2048x1536) and 4.28 (the pyramid at 4000x3000) are above 4 on 4 CPUs,
so thread count alone does not explain them (per-core cache, perhaps; not checked). Whether x86's 4 logical CPUs are 2
cores (above) is the obvious suspect for x86's scaling and was not checked.

DNN forwards (bare `forward` of the privacy detector; the dacl10k weights are not on the runners, so no segmenter
row ran):

| forward | input | threads | Arm, p50 / p95 ms | x86, p50 / p95 ms | x86 ÷ Arm (p50) |
| --- | --- | --- | --- | --- | --- |
| NanoDet-Plus | 416x416 | 1 | 40.5 / 40.8 | 24.8 / 25.0 | 0.61 |
| NanoDet-Plus | 416x416 | 4 | 13.1 / 13.3 | 14.9 / 15.2 | 1.14 |

x86 runs this forward in 0.61 of Arm's time on one thread; on 4 threads Arm is 1.14 times faster (Arm 3.1 times its
one-thread speed, x86 1.7 times).

### Metric B on both runners

The classical segmenter, the 8 demo photos and 8 synthetic ones, one round, at 1, 2 and 4 threads (`LOOP_THREADS`),
decisions held fixed as in Metric B above. Job and step columns are p50 / p95 ms.

| threads | prefetch | jobs/s, Arm | jobs/s, x86 | job, Arm | job, x86 | step, Arm | step, x86 | CPU s per job, Arm | CPU s per job, x86 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | off | 3.972 | 3.280 | 173.2 / 415.8 | 213.4 / 518.5 | 53.0 / 59.9 | 65.3 / 75.8 | 0.252 | 0.305 |
| 1 | warp | 3.989 | 3.300 | 174.1 / 414.1 | 213.6 / 513.3 | 52.7 / 59.4 | 64.7 / 75.9 | 0.257 | 0.311 |
| 1 | warp+segment | 4.784 | 3.928 | 181.8 / 347.2 | 214.7 / 401.7 | 53.9 / 61.2 | 64.6 / 76.3 | 0.372 | 0.451 |
| 2 | off | 4.532 | 3.865 | 152.0 / 366.9 | 181.0 / 433.4 | 47.0 / 52.7 | 55.6 / 63.1 | 0.288 | 0.358 |
| 2 | warp | 4.475 | 3.827 | 154.3 / 368.7 | 182.2 / 438.7 | 47.4 / 53.4 | 55.9 / 63.5 | 0.296 | 0.369 |
| 2 | warp+segment | 5.462 | 4.120 | 158.3 / 302.7 | 204.3 / 391.7 | 48.1 / 55.2 | 62.3 / 74.6 | 0.406 | 0.577 |
| 4 | off | 4.883 | 3.837 | 139.5 / 341.6 | 179.7 / 447.7 | 43.6 / 49.7 | 56.4 / 65.0 | 0.349 | 0.553 |
| 4 | warp | 4.860 | 3.826 | 141.4 / 341.6 | 183.2 / 446.5 | 44.1 / 50.5 | 55.8 / 64.9 | 0.356 | 0.562 |
| 4 | warp+segment | 5.661 | 4.026 | 154.1 / 285.7 | 208.3 / 400.0 | 46.8 / 52.9 | 62.3 / 78.9 | 0.489 | 0.766 |

On both runners every variant made the same decisions as prefetch-off on every photo, at every thread count
(`decisions_identical`, `decisions_identical_across_threads`), in 71 steps over the 16 photos; each prefetching
variant served 19 of 71 looks from the cache (0.27) and dropped 14 of its 58 speculative looks unread, half the Mac's
two-round counts. `warp+segment` against off, with the null arm `warp` against off in brackets:

| runner, threads | jobs/s | job p50 | job p95 | step p50 | step p95 | CPU s per job |
| --- | --- | --- | --- | --- | --- | --- |
| Arm, 1 | +20.5 % (+0.4 %) | +5.0 % (+0.5 %) | -16.5 % (-0.4 %) | +1.7 % (-0.6 %) | +2.2 % (-0.8 %) | +47.6 % (+2.0 %) |
| Arm, 2 | +20.5 % (-1.3 %) | +4.1 % (+1.5 %) | -17.5 % (+0.5 %) | +2.3 % (+0.9 %) | +4.7 % (+1.3 %) | +41.0 % (+2.8 %) |
| Arm, 4 | +15.9 % (-0.5 %) | +10.5 % (+1.4 %) | -16.4 % (+0.0 %) | +7.3 % (+1.1 %) | +6.4 % (+1.6 %) | +40.1 % (+2.0 %) |
| x86, 1 | +19.7 % (+0.6 %) | +0.6 % (+0.1 %) | -22.5 % (-1.0 %) | -1.1 % (-0.9 %) | +0.7 % (+0.1 %) | +47.9 % (+2.0 %) |
| x86, 2 | +6.6 % (-1.0 %) | +12.9 % (+0.7 %) | -9.6 % (+1.2 %) | +12.1 % (+0.5 %) | +18.2 % (+0.6 %) | +61.2 % (+3.1 %) |
| x86, 4 | +4.9 % (-0.3 %) | +15.9 % (+1.9 %) | -10.7 % (-0.3 %) | +10.5 % (-1.1 %) | +21.4 % (-0.2 %) | +38.5 % (+1.6 %) |

The null arm stayed within 3.1 % everywhere, much quieter than on the Mac (up to 22 %), so here the differences
below are larger than the noise this run shows, though it is one round:

- **One thread, both runners: faster whole jobs for half again the CPU.** jobs/s +20.5 % on Arm and +19.7 % on x86,
  job p95 -16.5 % and -22.5 %, CPU per job +47.6 % and +47.9 %, steps within 2.2 %. Arm's job p50 rose 5.0 %
  (null +0.5 %). On the Mac the same variant gave jobs/s from -2 % to +15 % across runs.
- **x86 on 2 and 4 threads: a loss at the step.** step p95 +18.2 % and +21.4 %, step p50 +12.1 % and +10.5 %, job
  p50 +12.9 % and +15.9 %, for jobs/s +6.6 % and +4.9 % and CPU +61.2 % and +38.5 %.
- **Arm: the gain holds on 2 threads; on 4 the median pays.** On 2, jobs/s +20.5 %, job p95 -17.5 %, step p95
  +4.7 % (null +1.3 %), CPU +41.0 %. On 4, jobs/s +15.9 % and job p95 -16.4 %, but job p50 +10.5 % and step p50
  +7.3 % (null +1.4 % and +1.1 %).

So with a spare core, background segmentation shortens whole jobs on both runners; once OpenCV's own threads fill the
CPUs it slows the main thread's steps, sooner on x86 than on Arm (the 2-core question again; not measured).
Prefetching stays off: none of this ran on the function's 1.7 vCPU, and at 2 threads, its setting, the two runners
disagree at the step.

Arm against x86 with prefetching off: Arm ran 1.21, 1.17 and 1.27 times x86's jobs/s at 1, 2 and 4 threads, and x86
used 1.21, 1.24 and 1.58 times Arm's CPU per job. x86 gained nothing from 2 to 4 threads (3.865 to 3.837 jobs/s)
while its CPU per job rose from 0.358 to 0.553 s; Arm went from 4.532 to 4.883 jobs/s.

### Macro: where a job's time goes on both runners

`bench/macro.py` on 200 synthetic 2048x1536 images at the 30 % cap, one worker, one OpenCV thread, the default agent
on a real clock (live costs and the governor on, so a machine's speed can change what it does). ms per image, with
the share of total wall time; the first row is the stages `macro.json` labels KleidiCV-covered imgproc, which on x86
are the same OpenCV calls, with IPP as the HAL where it covers them (which of them it covers is not recorded):

| stage | Arm | x86 | x86 ÷ Arm |
| --- | --- | --- | --- |
| pyramid, saliency, warp, quality gate | 15.0 (7.8 %) | 16.3 (6.6 %) | 1.09 |
| classical segmenter | 143.0 (73.9 %) | 182.9 (74.5 %) | 1.28 |
| crack thinning | 8.2 (4.2 %) | 18.1 (7.4 %) | 2.20 |
| decode | 13.9 (7.2 %) | 13.1 (5.3 %) | 0.94 |
| Python outside the timed ops | 13.4 (6.9 %) | 15.2 (6.2 %) | 1.13 |
| whole image, p50 / p95 / p99 | 195.8 / 209.7 / 213.8 | 248.2 / 270.0 / 282.7 | 1.27 |

Arm did 5.13 images/s and x86 4.06 (Arm 1.26 times). They did the same work on all but one of the 200 images, which
took 2 steps on Arm and 3 on x86: mean steps 2.945 and 2.950, mean pixel fraction 0.2610 and 0.2614 (live costs and
each build's own arithmetic can both change a decision; which did is not separated). Of the 52 ms per image x86 needs
beyond Arm, 40 are the classical segmenter and 10 crack thinning (`measure`'s hit-or-miss thinning, 2.20 times Arm's);
x86 decodes faster (0.94). If the covered stages ran 1.8 times faster, a whole job would get 1.036 times faster on Arm
and 1.030 times on x86.

This replaces the 50-image run of 2026-10-18 in `results/linux-aarch64/macro.json` (5.55 images/s; 75 % in the
segmenter, now 74 %; the same 8 % covered and 1.04 times); README.md, docs/report.md and bench/README.md now quote
this one. 5.55 against 5.13 images/s does not compare the runner with itself: the agent changed between the two runs
(commits to `drishyam/agent.py` landed in between). On the 50 images both runs share, 11 took a different number of
steps, mean steps went from 2.76 to 2.98 and the mean time per image from 179.2 to 189.4 ms, while the median time per
image moved 0.8 % (188.2 against 189.7 ms).

### The fixed cost table

The evaluation harness charges each timed op its p50 from `results/linux-aarch64/micro.json` `job_ops` (GitHub's
arm64 runner, 2026-10-18, the same build hash). This run's aarch64 micro, kept as `micro_run39220769299.json`, does
not replace it. All 21 of its `job_ops` p50s are below the table's: by 0.1 to 4.3 %, and by 12.3 % for `downscale`
at 1024x768 (0.071 against 0.081 ms). The table stays as it is. At 2048x1536, the size of the 200-image sets, p50 ms:

| op | table in use | this run, Arm | this run, x86 | x86 ÷ Arm |
| --- | --- | --- | --- | --- |
| `pyrDown` | 3.573 | 3.420 | 3.704 | 1.08 |
| `saliency` | 3.226 | 3.167 | 2.652 | 0.84 |
| `warp` | 1.614 | 1.613 | 2.017 | 1.25 |
| `quality` | 0.899 | 0.876 | 0.720 | 0.82 |
| `segment:accurate` | 59.068 | 58.221 | 74.557 | 1.28 |
| `segment:fast` | 29.513 | 29.148 | 42.308 | 1.45 |
| `downscale` | 0.366 | 0.365 | 0.991 | 2.72 |

An x86 table would charge the classical segment op 1.28 (`accurate`) and 1.45 (`fast`) times Arm's and the
downscale baseline's `INTER_AREA` resize 2.72 times, but saliency and the quality gate 0.84 and 0.82 times.

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

- Lambda itself: no timing in this page comes from the function; the GitHub arm64 runner is a different VM, a
  different core (Neoverse-N2) and Python 3.13 against the function's 3.12.
- 2 and 4 threads on this Mac (GCD), and any number on an idle Mac: the machine was shared throughout.
- Run-to-run variation on GitHub's runners: run 39220769299 is one run per runner and one round of Metric B. The
  only repeat is the aarch64 micro, a day apart on the same build (37 of 39 ops within 3.5 %, the other two 6.5 % and
  18.1 % apart); x86 was never repeated.
- What KleidiCV itself buys on Arm: no run turns the HAL off, so Arm against x86 is core, build and VM together.
- Whether the x86 runner's 4 logical CPUs are 4 cores or 2 cores with two hardware threads each.
- Mask parity between the Arm and x86 builds: `run_all.sh` saves `parity.npz` on each runner, but the workflow does
  not upload it.
- The macro worker sweep (`WORKERS`) on GitHub: one worker only.
- The dacl10k net on Arm, FP32 or INT8: the x86 INT8 result above does not predict it.
