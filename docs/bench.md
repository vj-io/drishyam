# Benchmarks: OpenCV kernels, the agent loop, Arm against x86, Lambda threads, INT8, DNN engines

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
| This Mac | Intel Core i9-9880H, x86, 8 cores / 16 threads; macOS GCD | opencv-python-headless 5.0.0 wheel (`uv.lock`), IPP HAL | Metric A, Metric B (1 thread and all 16), dacl10k latency, INT8, DNN engines, OpenCV 5 features, 5.0 against 4.14 | `results/local-macos-x86/{kernels,loop,loop_onnx,int8,int8_repeat,opencv5_engines,opencv5_engines_repeat,opencv5_features,opencv_builds}.json` |
| GitHub `ubuntu-24.04-arm` | Neoverse-N2, 4 logical CPUs; Linux pthreads | the same version, aarch64 wheel (KleidiCV and carotene HAL) | Metric A, Metric B at 1, 2, 4 threads, micro, macro, DNN engines, OpenCV 5 features, 5.0 against 4.14 | `results/linux-aarch64/`: `micro.json` (the fixed cost table, 2026-10-18); `kernels.json`, `loop.json`, `macro.json` and `micro_run39220769299.json` from run 39220769299; `opencv5_engines.json` from run 39617514621; `opencv5_features.json` and `opencv_builds.json` from run 39665440222 |
| GitHub `ubuntu-24.04` | AMD EPYC 7763, 4 logical CPUs; Linux pthreads | the same version, x86_64 wheel (IPP HAL) | the same | `results/linux-x86_64/{micro,macro,kernels,loop}.json`, run 39220769299; `opencv5_engines.json`, run 39617514621; `opencv5_features.json` and `opencv_builds.json`, run 39665440222 |
| Lambda arm64, 3008 MB | Graviton, 1.7 vCPU of time (below) | the same version, aarch64 wheel | not measured here | the smoke runs' step p95s in `docs/aws.md` are the only Lambda timings: 67.7 to 88.0 ms with the classical engine, 0.52 to 1.14 s with v1 (live since 2026-10-21) |

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
the second thread would have at most about 0.7 of a core, and with the classical segmenter (the deployed engine until
2026-10-21) it buys no reliable gain for 40 to 60 % more CPU; it stays off. With v1 on the function it was not
measured.

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
everywhere but the stack, nothing changes. The deployed function's startup log has
`DRISHYAM_CV_THREADS=2: cv2.getNumThreads() = 2` ([aws.md](aws.md)).

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
Graviton core is faster or slower than this i9's for this net was not measured here (the weights are not on the
GitHub runner). The function settled it once v1 was deployed (2026-10-21): over the eight smoke runs before the limit below, a
v1 step's p95 was 0.56 to 1.14 s ([aws.md](aws.md)), so the 512 net does not fit 500 ms there. The v1 profile's SLA is
therefore 1500 ms, set once from those runs plus headroom, so live looks stay on the 512 net unless steps exceed
1.5 s.

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

## OpenCV 5 DNN engines for v1

    uv run python -m bench.opencv5_engines --arm local-macos-x86

OpenCV 5.0 ships two DNN engines: the classic one, whose ONNX importer and layers come from 4.x, and the new one,
which `OnnxSegmenter._load` pins with `engine=cv2.dnn.ENGINE_NEW` (the default, `ENGINE_AUTO`, tries the new engine
first and falls back to the classic one). `bench/opencv5_engines.py` reads each v1 release file
(`drishyam_dacl10k.onnx`, FP32, and `.fp16.onnx`; their sha256 is in the JSON's `files`) and a static-shape copy of
each with both engines. A static copy is the file with its input's symbolic H and W set to the size it will run at
and nothing else changed, one per size, written to the git-ignored `runs/opencv5_engines/`; `models/` and the
deployed files are untouched. For every pair that runs it times a forward at 512 and 256 (the agent's accurate and
fast sizes) on 1 and 16 threads, in 20 rounds that each run one forward of every net at that size (so a
change in load lands on all of them), with the deployed net timed in a second slot of each round as a null; and it
compares the softmax output with the deployed net (the FP32 file on `ENGINE_NEW`) and with the other engine on the
same file over 8 synthetic crops (`bench.parity.make_crops`, preprocessed as `OnnxSegmenter` does): max
|dprob| and argmax agreement. This Mac, `results/local-macos-x86/opencv5_engines.json` (1-minute load average
11.99 at the start and 9.53 at the end; `opencv5_engines_repeat.json` ran from
9.53 to 9.63):

| file | engine | size | result | 1 thread, p50 / p95 ms | 16 threads, p50 / p95 ms | max \|dprob\|, argmax agreement with the deployed net | the same, against the other engine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 (the deployed file) | `ENGINE_NEW` | 512 | runs | 567.6 / 622.5 | 148.0 / 184.5 | (reference) |  |
| FP32, the same net in a second timing slot (null) | `ENGINE_NEW` | 512 | | 576.1 / 730.6 | 145.5 / 196.1 | | |
| FP32 (the deployed file) | `ENGINE_CLASSIC` | 512 | fails to load (a) |  |  |  |  |
| FP32, static copy | `ENGINE_NEW` | 512 | runs | 560.7 / 645.5 | 145.0 / 209.8 | 0, 1.000000 | 2.1e-5, 1.000000 |
| FP32, static copy | `ENGINE_CLASSIC` | 512 | runs | 493.0 / 723.5 | 209.9 / 383.4 | 2.1e-5, 1.000000 | 2.1e-5, 1.000000 |
| FP16 | `ENGINE_NEW` | 512 | runs | 687.2 / 871.7 | 252.7 / 324.8 | 0.018, 0.999978 |  |
| FP16 | `ENGINE_CLASSIC` | 512 | fails to load (a) |  |  |  |  |
| FP16, static copy | `ENGINE_NEW` | 512 | runs | 648.4 / 822.9 | 254.5 / 295.0 | 0.018, 0.999978 |  |
| FP16, static copy | `ENGINE_CLASSIC` | 512 | loads, forward fails (b) |  |  |  |  |
| FP32 (the deployed file) | `ENGINE_NEW` | 256 | runs | 136.5 / 158.8 | 42.0 / 52.6 | (reference) |  |
| FP32, the same net in a second timing slot (null) | `ENGINE_NEW` | 256 | | 136.2 / 146.8 | 41.4 / 46.6 | | |
| FP32 (the deployed file) | `ENGINE_CLASSIC` | 256 | fails to load (a) |  |  |  |  |
| FP32, static copy | `ENGINE_NEW` | 256 | runs | 134.6 / 165.8 | 41.4 / 49.2 | 0, 1.000000 | 1.4e-5, 1.000000 |
| FP32, static copy | `ENGINE_CLASSIC` | 256 | runs | 104.5 / 129.4 | 58.5 / 79.6 | 1.4e-5, 1.000000 | 1.4e-5, 1.000000 |
| FP16 | `ENGINE_NEW` | 256 | runs | 161.0 / 194.8 | 64.5 / 74.4 | 0.037, 0.999899 |  |
| FP16 | `ENGINE_CLASSIC` | 256 | fails to load (a) |  |  |  |  |
| FP16, static copy | `ENGINE_NEW` | 256 | runs | 159.6 / 179.4 | 66.7 / 73.3 | 0.037, 0.999899 |  |
| FP16, static copy | `ENGINE_CLASSIC` | 256 | loads, forward fails (b) |  |  |  |  |

The exact errors (the `OpenCV(5.0.0) <build path>` prefix trimmed; the JSON has them whole, with OpenCV's log lines):

- (a) both v1 files on `ENGINE_CLASSIC`, at both sizes: `Node [Shape@ai.onnx]:(onnx_node!/net/decoder/Shape) parse
  error: ... onnx_importer.cpp:2591: error: (-215:Assertion failed) !isDynamicShape in function 'parseShape'`, after
  the log line `DNN/ONNX(Shape): dynamic 'zero' shapes are not supported, input input [<empty>]`, for the FP32 file
  (the error in `int8.json`'s (a) above). The FP16 file stops at `/net/decoder/Shape_2` instead, whose input is
  `/net/encoder/model/blocks.0/blocks.0.0/Add_output_0 [<empty>]`, because its nodes are in another order.
- (b) the FP16 static copy on `ENGINE_CLASSIC` loads, and its first forward fails in the stem convolution's type
  check: `layer.cpp:274: ... 'input == CV_32F || input == CV_64F || input == CV_8S || input == CV_8U || input ==
  CV_64S' where 'input' is 7 (CV_16FC1)`, logged as `[Convolution]:(onnx_node!/net/encoder/model/conv_stem/Conv):
  getMemoryShapes() throws exception`.

**Why the classic engine fails.** v1's input is `[1, 3, 'h', 'w']` on purpose: one file runs at 512 and at 256. Its
UNet decoder resizes each upsampled map to the size of its skip connection, which the PyTorch export writes as
`Shape -> Gather -> Unsqueeze -> Concat -> Resize`, five times (`files.*.graph.shape_nodes`: `/net/decoder/Shape` on
the graph input, `Shape_2` to `Shape_8` on encoder outputs). The classic importer folds every `Shape` into a constant
while it imports, records a symbolic dim as 0, and asserts that no dim is 0 (`parseShape` in `onnx_importer.cpp`; the
4.x importer has the same check, so this is not a 5.0 regression). `ENGINE_NEW` evaluates `Shape` at run time.
[bench/repro/](../bench/repro/README.md) reproduces it with a 10-node model and no weights: symbolic H and W fail on
`ENGINE_CLASSIC`, a symbolic batch alone does not, and every variant loads on `ENGINE_NEW`. That README also sets out
what is and is not known about the DeepLabV3+ parity failure on `ENGINE_NEW` (commit 95741ba), which no file in this
repository reproduces.

**The static-shape finding.** Fixing the two dims is enough: the FP32 static copy loads and runs on
`ENGINE_CLASSIC`, and its output differs from the deployed net by max |dprob| 2.1e-5 at 512 and
1.4e-5 at 256, with every argmax equal (`ENGINE_NEW` on the same copy gives the deployed net's output
exactly). The FP16 copy does not run there (b), so `ENGINE_CLASSIC` has no FP16 path for v1. Nothing is adopted: the
agent still reads the deployed file on `ENGINE_NEW`, and a static copy would have to be one file per size.

**Speed, on this machine.** Within each run the null slot's p50 was within 2 % of the deployed net's
(3 % in the repeat), and the static copy on `ENGINE_NEW` within 2 % (3 %), so a fixed shape buys
`ENGINE_NEW` nothing measurable here. `ENGINE_CLASSIC` on the static copy is faster on one thread and slower on
16: at one thread it took x0.87, x0.77 of the deployed net's p50 at 512 and 256 (x0.84, x0.74 in the repeat),
at 16 threads x1.42, x1.39 (x1.48, x1.41). The FP16 file on `ENGINE_NEW` took x1.21, x1.18, x1.71, x1.54 of the FP32 file's p50
at 512 and 256 on one thread, then on 16 (x1.21, x1.19, x1.79, x1.53 in the repeat); why FP16 is slower on this CPU was not
measured. The two runs went back
to back once the load had fallen to about 10; their absolute numbers still carry the uncertainty described at the
top of this page, and only the ratios within a run are compared here.

**What the build says.** OpenCV 5.0 also has an ONNX Runtime engine, `ENGINE_ORT`, but this wheel is built without
it: its build information reads ONNX Runtime `NO` (`onnx_runtime_in_build`), and asked for `ENGINE_ORT` it
logs that it was built with `WITH_ONNXRUNTIME=OFF` and falls back to `ENGINE_AUTO` (seen once by hand, not in the
JSON). `provenance` quotes the other build lines that decide which kernels run: `cv2_custom_hal` YES (ipp (ver 0.0.1)),
dispatch SSE4_2 AVX FP16 AVX2 AVX512_SKX on a baseline of SSE SSE2 SSE3 SSSE3 SSE4_1, parallel framework GCD.

**Arm and x86 on GitHub: run 39617514621.** The `engines` job of `.github/workflows/bench-arm64.yml` ran the same
command on GitHub's `ubuntu-24.04-arm` and `ubuntu-24.04` runners on 2026-10-21 (`-f bench=false -f engines=true` on
`release/v1.1` at a2bd2c5; the run id is the owner's, the JSON does not record it), into
`results/linux-aarch64/opencv5_engines.json` and `results/linux-x86_64/opencv5_engines.json`. It downloaded
`drishyam_dacl10k.onnx`, `.fp16.onnx` and `.sha256` from the public v1.1.0 release, checked them with `sha256sum -c`
against that file and against the hashes committed in `models/drishyam_dacl10k.provenance.json`, and uploaded only
the JSON, never the weights. The Arm runner was a Neoverse-N2, 4 logical CPUs, Linux 7.0.0-1017-azure aarch64,
`cv2_custom_hal` YES (carotene (ver 0.0.1) KleidiCV (ver 26.03)), 1-minute load average 0.34 at the start and 2.15 at
the end; the x86 one an AMD EPYC 7763 64-Core Processor, 4 logical CPUs, Linux 6.17.0-1022-azure x86_64, `cv2_custom_hal`
YES (ipp (ver 0.0.1)), load average 0.45 at the start and 2.21 at the end. Every file loaded, ran or failed on both
exactly as on the Mac, with the same errors (a) and (b). The rows that run (p50 / p95 ms, 20 interleaved rounds; 4
threads is every logical CPU of each runner):

| file | engine | size | Neoverse-N2, 1 thread | Neoverse-N2, 4 threads | EPYC 7763, 1 thread | EPYC 7763, 4 threads | max \|dprob\|, argmax agreement with the deployed net: Arm | the same: x86 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 (the deployed file) | `ENGINE_NEW` | 512 | 908.2 / 934.7 | 231.6 / 238.7 | 432.6 / 436.0 | 202.4 / 203.7 | (reference) | (reference) |
| FP32, the same net in a second timing slot (null) | `ENGINE_NEW` | 512 | 904.7 / 932.2 | 230.9 / 237.2 | 433.4 / 439.7 | 202.6 / 204.8 | | |
| FP32, static copy | `ENGINE_NEW` | 512 | 907.6 / 936.1 | 228.8 / 239.5 | 445.6 / 448.5 | 203.3 / 206.2 | 0, 1.000000 | 0, 1.000000 |
| FP32, static copy | `ENGINE_CLASSIC` | 512 | 347.5 / 391.5 | 130.6 / 145.2 | 322.1 / 326.2 | 152.7 / 156.5 | 1.6e-5, 1.000000 | 2.5e-5, 1.000000 |
| FP16 | `ENGINE_NEW` | 512 | 938.7 / 961.1 | 264.0 / 279.0 | 519.3 / 522.1 | 281.8 / 284.2 | 0.018, 0.999978 | 0.018, 0.999978 |
| FP16, static copy | `ENGINE_NEW` | 512 | 957.6 / 987.5 | 272.3 / 284.6 | 526.9 / 529.1 | 283.2 / 285.4 | 0.018, 0.999978 | 0.018, 0.999978 |
| FP32 (the deployed file) | `ENGINE_NEW` | 256 | 224.3 / 229.6 | 59.9 / 60.9 | 104.1 / 105.1 | 50.5 / 51.3 | (reference) | (reference) |
| FP32, the same net in a second timing slot (null) | `ENGINE_NEW` | 256 | 224.5 / 227.6 | 59.8 / 60.7 | 104.1 / 104.6 | 49.6 / 50.6 | | |
| FP32, static copy | `ENGINE_NEW` | 256 | 225.7 / 228.0 | 59.7 / 61.5 | 103.9 / 104.8 | 50.5 / 51.2 | 0, 1.000000 | 0, 1.000000 |
| FP32, static copy | `ENGINE_CLASSIC` | 256 | 79.5 / 90.0 | 23.4 / 24.8 | 79.0 / 80.5 | 37.5 / 38.2 | 1.7e-5, 0.999998 | 1.4e-5, 1.000000 |
| FP16 | `ENGINE_NEW` | 256 | 239.6 / 244.6 | 71.3 / 72.6 | 128.2 / 131.3 | 73.5 / 74.0 | 0.038, 0.999899 | 0.037, 0.999899 |
| FP16, static copy | `ENGINE_NEW` | 256 | 229.4 / 231.5 | 67.6 / 69.1 | 128.0 / 129.2 | 73.3 / 74.1 | 0.038, 0.999899 | 0.037, 0.999899 |

**On Arm `ENGINE_CLASSIC` runs v1 in about a third of the time.** The classic engine on the static copy took, of the
deployed net's p50 at 512 and 256 on one thread and then on 4, Arm: x0.38, x0.35, x0.56, x0.39, and x86: x0.74,
x0.76, x0.75, x0.74. The null slot was within Arm 0 %, the static copy on `ENGINE_NEW` 1 % of the deployed net's p50,
and x86 2 %, the static copy on `ENGINE_NEW` 3 %. Across the two runners the classic engine ran v1 at about the same
speed (347.5 against 322.1 ms at 512 on one thread) while `ENGINE_NEW` took 2.1 times as long on the Neoverse-N2 as on
the EPYC (908.2 against 432.6 ms): the gap is in the new engine's run on this Arm build, not in the core. 5.0's
per-layer profiler puts it in the 3x3 convolutions ("OpenCV 5 features in the pipeline", below). The classic engine
also moved one label on Arm: at 256 its argmax differed from the deployed net's at 1 of the 524,288 pixels of the 8
crops (agreement 0.999998), with max |dprob| at most 2.5e-5 on either runner. FP16 on `ENGINE_NEW` was slower than
FP32 here too, at Arm x1.03, x1.07, x1.14, x1.19 and x86 x1.20, x1.23, x1.39, x1.46 of its p50. Nothing is adopted
from this run: the agent's outputs would move (by that pixel and the 1e-5 probabilities), which "OpenCV 5 features in
the pipeline" sets out with the flag that tries it. These are GitHub VMs, one run each, not the Lambda.

## OpenCV 5 features in the pipeline

    uv run python -m bench.opencv5_features --arm local-macos-x86 --onnx <drishyam_dacl10k.onnx> <drishyam_dacl10k.fp16.onnx>

    uv run python -m bench.opencv_builds --arm local-macos-x86

What OpenCV 5.0 adds over 4.x is taken from the OpenCV wiki's "OpenCV 5" and "OpenCV 4 to 5 migration" pages (read
2026-10-21) and from the wheels themselves: `bench/opencv_builds.py` records which names each build has (`api` in
`results/local-macos-x86/opencv_builds.json`: the locked 5.0.0 wheel against opencv-python-headless 4.14.0.94, the
latest 4.x on PyPI). Of what this pipeline could use, the new DNN engine with its profiling and tracing modes, the
element types `CV_Bool`, `CV_32U` and `CV_64S`, and NumPy bool arrays as operation masks (read as `CV_Bool`) are new
in 5.0; `reduceArgMax`, `compare`, `hasNonZero`, `findContoursLinkRuns`, the `ALGO_HINT_APPROX` hint, IPP's switch and
`getPerfProfile` are in 4.14 as well. The wiki also lists, for 5.0, bilinear and bicubic warps without lookup tables
and a Pillow-compatible nearest-neighbour resize; its faster contour tracer (TRUCO) is dated 4.14 by the 5.0 wheel's
own `findContours` docstring and runs only for `RETR_LIST` without a hierarchy.

| name in `cv2` | 5.0.0 | 4.14.0 |
| --- | --- | --- |
| `dnn.ENGINE_NEW` | yes | no |
| `dnn.ENGINE_CLASSIC` | yes | no |
| `dnn.Net.setProfilingMode` | yes | no |
| `dnn.DNN_PROFILE_DETAILED` | yes | no |
| `dnn.Net.setTracingMode` | yes | no |
| `dnn.Net.getPerfProfile` | yes | yes |
| `CV_Bool` | yes | no |
| `CV_32U` | yes | no |
| `CV_64S` | yes | no |
| `reduceArgMax` | yes | yes |
| `compare` | yes | yes |
| `hasNonZero` | yes | yes |
| `findContoursLinkRuns` | yes | yes |
| `ALGO_HINT_APPROX` | yes | yes |
| `ipp.setUseIPP` | yes | yes |
| `bool mask` | yes | no |

**What each one did here** ("new" means absent from the 4.14 wheel; the figures are in the tables below and in
"OpenCV 5 DNN engines for v1"):

- `ENGINE_NEW` (new): the only engine that reads v1, the live segmenter. On 5.0 `readNet` gives it to the NanoDet
  privacy detector too, which ran in about half of 4.14's time (the builds table).
- Its profiling mode (`Net.setProfilingMode(DNN_PROFILE_DETAILED)`, new): v1's time per layer on both engines,
  below. Benchmark only.
- The engine selector (`readNetFromONNX(path, engine=...)`, new) with `ENGINE_CLASSIC` on a static-shape copy of v1:
  `OnnxSegmenter(engine="classic")`. Its traces matched here, but it was not faster here: off. On GitHub's Arm
  runner it passes both the identity and the speed test, at every thread count (below).
- `CV_Bool` (new): `CV_BOOL_MASKS`, the NumPy bool footprint as a `copyTo` mask in `segment.restrict`: identical,
  x0.83 to x0.94 of the NumPy time, but one interval reaches 1.02: off. It passes on both GitHub runners (below).
- 5.0's warp (new, bilinear without lookup tables): in use with the wheel; 4.14 took 1.6 and 2.4 times as long on
  the gaze (the builds table).
- `reduceArgMax` (OpenCV 4.6, not new): `CV_ARGMAX` in `segment.finish`: identical, x0.52 to x0.74 of NumPy's time:
  on, the one flag adopted. On a whole look the change is within this machine's noise (the last two timing rows,
  all three flags at once): `finish` is about a fifth of a classical look's time here.
- `compare` (not new): `CV_COMPARE` in `_clean`: identical, no measurable gain: off.
- The x86 wheel's IPP HAL (not new): with it off only `saliency`'s output changed, and no stage was measurably
  slower. Arm's KleidiCV HAL (in 4.x wheels too): no runtime switch, not measured (below).

**The rule, and the identity set.** A flag of `drishyam/segment.py` ships on only if, on every photo of the identity
set and with every segmenter, the agent's event stream with it on is the stream with every flag off to the byte, and
if on each unit it changes, at every thread count, its on/off interval lies wholly below 1.
`tests/test_opencv5_features.py` checks the shipped values against that rule and this machine's JSON, and runs the
same identity on four of the photos in `make test`. The identity set is every demo sample (`public/samples`: eight
synthetic bridge photos and the two real ones) and the first four images of the synth_dev split as `make eval-dev`
builds them (2048x1536, JPEG quality 92); no CODEBRIM image. Each run is the default agent (`JobConfig()`, governor
and thumbnails on) on a clock that advances 1 ms per read, so what it decides depends on the segmenter's outputs
alone, and the stream is compared whole (gaze path, causes, rationales, findings, thumbnails, work order) with only
the wall-clock fields left out, as `tests/test_prefetch.py` does. The null column is all flags off run twice.

| segmenter | photos | looks | events | null | `CV_ARGMAX` | `CV_COMPARE` | `CV_BOOL_MASKS` | all three |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical-v1 | 14 | 62 | 210 | 14 of 14 | 14 of 14 | 14 of 14 | 14 of 14 | 14 of 14 |
| v1 on `ENGINE_NEW` | 14 | 49 | 171 | 14 of 14 | 14 of 14 | 14 of 14 | 14 of 14 | 14 of 14 |

**Timing.** Each flag's unit, on the looks those runs made (every k-th, 24 at most), in 21 rounds that each run off
(every flag off), on (that flag alone, or all three) and off again (the null slot) in an order rotated by one per
round; p50 ms of a round's call over all its looks, then the on/off and null/off ratios per round as a geometric mean
with a 95 % bootstrap interval over the rounds. v1's whole look is its forward, timed with the engines below, so v1
gets the two small units only.

| flag | segmenter | unit | threads | off, p50 ms | on, p50 ms | on/off (95 % CI) | null/off (95 % CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV_ARGMAX | classical-v1 | `finish` on 24 looks | 1 | 2238.9 | 1574.6 | x0.74 (0.70-0.79) | x1.02 (0.96-1.10) |
| CV_ARGMAX | classical-v1 | `finish` on 24 looks | 16 | 1684.9 | 1120.7 | x0.68 (0.65-0.72) | x1.02 (0.99-1.06) |
| CV_ARGMAX | v1 on `ENGINE_NEW` | `finish` on 24 looks | 1 | 1585.9 | 941.5 | x0.60 (0.53-0.67) | x1.00 (0.90-1.12) |
| CV_ARGMAX | v1 on `ENGINE_NEW` | `finish` on 24 looks | 16 | 1054.5 | 558.1 | x0.52 (0.46-0.57) | x0.91 (0.84-0.97) |
| CV_COMPARE | classical-v1 | `finish` on 24 looks | 1 | 2027.0 | 1969.7 | x0.98 (0.95-1.01) | x1.00 (0.97-1.03) |
| CV_COMPARE | classical-v1 | `finish` on 24 looks | 16 | 1769.5 | 1677.5 | x0.95 (0.91-0.98) | x1.02 (0.97-1.07) |
| CV_COMPARE | classical-v1 | `restrict` on 9 looks | 1 | 385.8 | 381.9 | x1.01 (0.95-1.08) | x1.02 (0.97-1.09) |
| CV_COMPARE | classical-v1 | `restrict` on 9 looks | 16 | 400.3 | 398.5 | x1.03 (0.97-1.09) | x1.00 (0.93-1.07) |
| CV_COMPARE | v1 on `ENGINE_NEW` | `finish` on 24 looks | 1 | 1034.6 | 1046.5 | x0.97 (0.91-1.02) | x1.00 (0.90-1.09) |
| CV_COMPARE | v1 on `ENGINE_NEW` | `finish` on 24 looks | 16 | 945.8 | 929.2 | x1.00 (0.93-1.07) | x1.00 (0.94-1.07) |
| CV_COMPARE | v1 on `ENGINE_NEW` | `restrict` on 9 looks | 1 | 174.5 | 178.5 | x0.98 (0.90-1.06) | x0.99 (0.93-1.05) |
| CV_COMPARE | v1 on `ENGINE_NEW` | `restrict` on 9 looks | 16 | 168.4 | 166.8 | x0.98 (0.90-1.05) | x0.98 (0.91-1.06) |
| CV_BOOL_MASKS | classical-v1 | `restrict` on 9 looks | 1 | 396.4 | 353.9 | x0.88 (0.83-0.93) | x1.02 (0.95-1.10) |
| CV_BOOL_MASKS | classical-v1 | `restrict` on 9 looks | 16 | 366.9 | 297.7 | x0.83 (0.77-0.90) | x1.00 (0.92-1.08) |
| CV_BOOL_MASKS | v1 on `ENGINE_NEW` | `restrict` on 9 looks | 1 | 166.9 | 154.3 | x0.94 (0.86-1.02) | x0.98 (0.91-1.06) |
| CV_BOOL_MASKS | v1 on `ENGINE_NEW` | `restrict` on 9 looks | 16 | 161.4 | 147.7 | x0.88 (0.81-0.97) | x1.05 (0.97-1.14) |
| all three | classical-v1 | `segment_view` on 24 looks | 1 | 11749.7 | 11691.1 | x0.98 (0.93-1.04) | x1.01 (0.96-1.06) |
| all three | classical-v1 | `segment_view` on 24 looks | 16 | 11479.3 | 11269.1 | x0.93 (0.85-1.02) | x0.96 (0.89-1.03) |

**Where v1's time goes, layer by layer (OpenCV 5's profiler).** With `Net.setProfilingMode(DNN_PROFILE_DETAILED)`
(new in 5.0) `ENGINE_NEW`'s `getPerfProfile()` returns every layer's time, as `ENGINE_CLASSIC`'s always has. Layers
are named after their ONNX nodes, so the graph says what each one is and the two engines line up node by node. v1 at
512, 10 forwards per engine and thread count, interleaved, each layer's median; the kinds that take the most of
`ENGINE_NEW`'s time on one thread, then the totals. The run's 1-minute load average was 143.97 at the start and
22.82 at the end, so the absolute times are inflated and only the comparison within the run holds:

| layer kind (from the ONNX graph) | layers, `ENGINE_NEW` | layers, `ENGINE_CLASSIC` | 1 thread, `ENGINE_NEW`, ms | 1 thread, `ENGINE_CLASSIC`, ms | 16 threads, `ENGINE_NEW`, ms | 16 threads, `ENGINE_CLASSIC`, ms |
| --- | --- | --- | --- | --- | --- | --- |
| conv 3x3 | 12 | 12 | 1465.4 | 1555.7 | 183.8 | 269.4 |
| conv 1x1 | 46 | 46 | 158.0 | 182.8 | 33.9 | 78.4 |
| Resize | 5 | 5 | 37.2 | 50.2 | 10.5 | 11.0 |
| Mul | 9 | 9 | 32.0 | 6.9 | 7.8 | 3.8 |
| depthwise conv 3x3 | 9 | 9 | 30.0 | 35.5 | 11.1 | 9.6 |
| TransformLayout | 18 | 0 | 23.5 |  | 8.8 |  |
| depthwise conv 5x5 | 6 | 6 | 22.8 | 127.9 | 5.2 | 31.0 |
| Concat | 10 | 5 | 19.2 | 15.5 | 6.1 | 5.9 |
| every layer (the profiler's sum) | 159 | 185 | 1807.4 | 2011.7 | 348.7 | 495.0 |
| a forward, wall time (p50) | | | 1944.6 | 2041.3 | 405.6 | 579.2 |

The twelve 3x3 convolutions (the stem, the UNet decoder's ten and the segmentation head) are most of either
engine's time: 81 % of `ENGINE_NEW`'s and 77 % of `ENGINE_CLASSIC`'s on one thread. The new engine's graph lists 159
layers against the classic engine's 185, of which 57 are activations the classic engine fused into their
convolutions (listed, with no time). The new engine fuses those and the 10 residual additions into the convolutions,
keeps the decoder's shape arithmetic (`Shape`, `Gather`, `Unsqueeze`) as layers it evaluates at run time, and adds 18
layout transforms to and from its blocked layout (23.5 ms at one thread). It runs the 5x5 depthwise
convolutions in x0.18 of the classic engine's time and the squeeze-and-excitation multiplies in x4.7 its time. On
16 threads its layers sum to 348.7 against 495.0 ms, most of the difference in the 3x3 and 1x1 convolutions. On the
Neoverse-N2 the forward takes 2.6 times as long on `ENGINE_NEW` ("OpenCV 5 DNN engines for v1"); the same profile
there (below) puts that gap in the 3x3 convolutions.

**The agent on `ENGINE_CLASSIC`.** `OnnxSegmenter(engine="classic")` reads the FP32 net on `ENGINE_CLASSIC`
through a static-shape copy per input size (the FP16 net stays on `ENGINE_NEW`, the only engine that runs it); the
run puts it against `engine="new"` on the same photos. The agent's trace was identical on 14 of 14 photos, over 49
looks, all at the accurate size; across those looks max |dprob| 2.4e-5, and 0 of the 12,845,056 pixels had another
argmax. Its accurate looks (8 timed, 10 rounds, the null `ENGINE_NEW` again) took x0.92 (0.81-1.10) of
`ENGINE_NEW`'s time on one thread and x1.30 (1.29-1.32) on 16. So on this Mac it passes the identity asked of a DNN
change (every argmax equal, max |dprob| under 1e-4, the same decisions) and fails on speed: no gain this machine can
resolve on one thread, a loss on 16, the agent's default here. On Arm it passes both (below).

**OpenCV 5.0 against 4.14 on the pipeline's own stages, and the IPP HAL.** `bench/opencv_builds.py` runs each
stage on one thread in a worker process per variant: the locked 5.0 wheel (the reference), the same with
`cv2.ipp.setUseIPP(False)`, opencv-python-headless 4.14.0.94 with the same NumPy (installed by `uv` into `runs/`),
and the 5.0 wheel again (the null). 24 rounds, each running every variant once in an order rotated by one per round;
a worker times each stage 5 times and reports the median and a hash of the output. Inputs are made once by the
reference (bridge_03's pyramid, an L1 crop on its crack, the crack's mask, the detectors' 640-px copy), so an output
differs only where the stage computes differently. The ratio is the variant's time over 5.0's, per round, as a
geometric mean with a 95 % bootstrap interval; above 1, 5.0 is faster. 1-minute load average 149.01 at the start
and 150.83 at the end (`results/local-macos-x86/opencv_builds.json`):

| stage | 5.0, p50 ms | 5.0 with IPP off: time over 5.0's, output | 4.14.0: time over 5.0's, output | null: 5.0 again |
| --- | --- | --- | --- | --- |
| `decode` | 286.0 | x0.97 (0.89-1.05), same | x0.89 (0.81-0.96), same | x1.02 (0.91-1.13) |
| `pyramid` | 102.8 | x1.01 (0.91-1.13), same | x1.07 (1.01-1.14), same | x0.96 (0.89-1.03) |
| `saliency` | 50.4 | x0.99 (0.88-1.10), differs | x0.95 (0.86-1.06), same | x1.00 (0.87-1.14) |
| `gaze` | 11.1 | x1.09 (0.93-1.28), same | x1.60 (1.38-1.85), differs | x1.12 (0.97-1.28) |
| `gaze_oblique` | 6.2 | x1.12 (0.88-1.40), same | x2.43 (2.07-2.86), differs | x1.14 (0.95-1.38) |
| `quality` | 7.1 | x1.15 (0.98-1.36), same | x0.99 (0.82-1.18), same | x1.08 (0.90-1.30) |
| `enhance` | 22.3 | x1.07 (0.92-1.21), same | x1.00 (0.90-1.10), same | x0.96 (0.85-1.07) |
| `segment` | 445.2 | x1.05 (0.97-1.14), same | x0.98 (0.88-1.07), same | x1.07 (0.98-1.14) |
| `measure` | 15.8 | x1.07 (0.91-1.26), same | x1.01 (0.88-1.20), same | x1.01 (0.83-1.21) |
| `nanodet` | 306.8 | x1.22 (1.11-1.34), same | x1.95 (1.79-2.13), same | x1.10 (1.01-1.19) |
| `yunet` | 169.4 | x1.09 (0.97-1.23), same | x1.12 (0.99-1.29), same | x1.06 (0.94-1.19) |

The null slot's own interval reaches x1.38 (`gaze_oblique`), so only a ratio whose interval clears the null's for
the same stage says anything. Three do, in two findings. 5.0's warp is faster and computes differently: 4.14 took
x1.60 (1.38-1.85) of its time on the straight 512-px gaze and x2.43 (2.07-2.86) on the oblique one, and its output
differs, as the migration guide says 5.0's bilinear warp without lookup tables does. And 5.0 runs the NanoDet
privacy detector in about half the time, x1.95 (1.79-2.13) for 4.14 (`readNet` picks `ENGINE_NEW` on 5.0; 4.14 has
only its one engine), with the same boxes. Decode, pyramid, saliency, quality, enhance, the classical segmenter,
crack measurement and YuNet are within about 15 % either way. With IPP off only `saliency`'s output changed, so IPP
computes some of it, and every IPP-off interval overlaps the null's for the same stage: on this CPU no stage
measurably needs IPP. Every look is one warp, but the warp is cheap next to a segmenter look (11.1 ms here against
445.2 for the classical segmenter on the same crop), so a job barely moves.

**KleidiCV, by name only.** The aarch64 wheel's KleidiCV cannot be switched off at run time (the HAL replaces
OpenCV's calls when the wheel is built, and KleidiCV's OpenCV HAL header, `adapters/opencv/kleidicv_hal.h` on its
main branch, read 2026-10-21, has no runtime switch), so what it buys was not measured: that takes a wheel built with
`-DWITH_KLEIDICV=OFF`, which this work did not build. The aarch64 4.14 wheel has no KleidiCV (its HAL is carotene
alone), so the Arm runner's 4.14 column below includes KleidiCV's share without isolating it. By name, that header replaces these of the pipeline's calls:
`GaussianBlur`, morphology (black-hat, open, dilate, hit-or-miss), `resize`, `Sobel`, `medianBlur`, `pyrDown`,
`warpPerspective`, `remap`, `inRange` and `copyMakeBorder`, and of `cvtColor` only the gray-to-BGR, BGR-to-BGR (the
channel swap `OnnxSegmenter` does) and YUV conversions; not CLAHE, `adaptiveThreshold`, `Laplacian`,
`connectedComponentsWithStats`, `findContours`, `minAreaRect`, `fitLine`, the gray, HSV and Lab conversions or the
DNN. Whether a call takes it also depends on its type, border and kernel, which only a measurement shows; most of
its wrappers fall back to OpenCV's own code when KleidiCV declines.

**Not adopted, and not tried.**

- `CV_COMPARE` (`cv2.compare` for `_clean`'s class masks; not new in 5.0) left every trace identical but did not
  pass the rule (the timing table): it ships off, the NumPy form.
- `ENGINE_CLASSIC` for v1 (`OnnxSegmenter(engine="classic")`, or `DRISHYAM_DNN_ENGINE=classic`) ships off. It
  passed the identity asked of a DNN change on this Mac (above: every argmax equal, max |dprob| under 1e-4, every
  trace identical) but was not faster here: slower on 16 threads, the agent's default on this machine, and no
  measurable change on one. Its gain is on Arm (x0.38 of the forward on the Neoverse-N2 above), where at the agent
  level it passes both the identity and the speed test (below). It also needs the `onnx` package to write its
  static-shape copy, a development dependency the Lambda package does not install; a copy made at build time would
  remove that need (below).
- `CV_BOOL_MASKS` (a NumPy bool footprint as a `CV_Bool` `copyTo` mask in `restrict`, new in 5.0) left every trace
  identical and was faster in three of its four timings, but the fourth interval (v1, one thread) reaches 1.02, so
  under the rule it ships off. On both GitHub runners it passes (below).
- `ALGO_HINT_APPROX` (the algorithm hint of OpenCV 4.10 on) was not tried: it allows results that differ from the
  accurate path by contract, so an identity measured on this build would say nothing about the aarch64 one the
  function runs.
- TRUCO (`findContours` with `RETR_LIST` and no hierarchy) was not tried: `measure()` and the finding callouts need
  `RETR_EXTERNAL`, and the inner contours `RETR_LIST` adds where a thinned crack closes a loop would change the
  measured length.
- FP16 on the CPU: the v1 FP16 file is slower than FP32 on `ENGINE_NEW` on all three CPUs ("OpenCV 5 DNN engines for
  v1", above); `DNN_TARGET_CPU_FP16` and BF16 were not measured. `ENGINE_ORT` is not in this wheel, and G-API moved
  to opencv_contrib in 5.0.
- Batching several crops into one forward was not tried: the agent decides one look at a time, and the prefetcher's
  speculative look runs on its own segmenter instance (`drishyam/prefetch.py`).
- Two OpenCV 5.0.0 quirks turned up and are pinned by `tests/test_opencv5_features.py`:
  `readNetFromONNX(buffer, engine=ENGINE_CLASSIC)` returns an `ENGINE_NEW` net (so `segment._classic` writes its
  static copy to a temporary file), and `enableFusion(False)` leaves an `ENGINE_NEW` net empty (its forward asserts
  `!empty()`).

### On GitHub's Arm and x86 runners: run 39665440222

`.github/workflows/bench-arm64.yml` ran both benchmarks on 2026-10-21 with `-f features=true -f builds=true` (the
run id is the owner's; the JSON does not record it): `bench/opencv5_features.py --threads 1,2,4` on the release
weights, checked as above, and `bench/opencv_builds.py` with the 4.14 wheel installed on each runner, into
`results/linux-aarch64/` and `results/linux-x86_64/` (`opencv5_features.json`, `opencv_builds.json`). The features
runs: Neoverse-N2, 4 logical CPUs, 1-minute load average 0.19 at the start and 4.13 at the end; AMD EPYC 7763 64-Core
Processor, 4 logical CPUs, 1-minute load average 0.07 at the start and 2.8 at the end. These runners were quiet, so
their intervals are far narrower than the Mac's: every null ratio of the timing lay between x0.99 and x1.01. They are
GitHub VMs, one run each, not the Lambda. To run them again from a pushed branch:

```bash
gh workflow run bench-arm64.yml --repo vj-io/drishyam --ref <branch> -f bench=false -f engines=false -f features=true -f builds=true
```

**Identity on both runners.** Every variant's trace equalled the all-off trace on all 14 photos with both segmenters,
over the same 62 classical and 49 v1 looks as on the Mac: 14 of 14 in every cell of the identity table above.

**Timing on both runners** (on/off with its 95 % interval; the null columns are in the JSON):

| flag | segmenter | unit | Arm, 1 thread | Arm, 2 | Arm, 4 | x86, 1 thread | x86, 2 | x86, 4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CV_ARGMAX | classical-v1 | `finish` on 24 looks | x0.79 (0.78-0.79) | x0.74 (0.74-0.74) | x0.71 (0.70-0.71) | x0.85 (0.85-0.85) | x0.77 (0.77-0.78) | x0.77 (0.77-0.77) |
| CV_ARGMAX | v1 on `ENGINE_NEW` | `finish` on 24 looks | x0.75 (0.75-0.76) | x0.69 (0.69-0.69) | x0.64 (0.64-0.64) | x0.83 (0.83-0.83) | x0.75 (0.74-0.75) | x0.73 (0.73-0.73) |
| CV_COMPARE | classical-v1 | `finish` on 24 looks | x1.00 (1.00-1.01) | x1.01 (1.00-1.02) | x1.00 (1.00-1.01) | x1.00 (0.99-1.00) | x1.00 (0.99-1.00) | x1.00 (1.00-1.00) |
| CV_COMPARE | classical-v1 | `restrict` on 9 looks | x1.00 (1.00-1.00) | x1.01 (1.00-1.01) | x1.01 (1.00-1.01) | x1.00 (1.00-1.01) | x1.01 (1.00-1.02) | x1.02 (1.01-1.04) |
| CV_COMPARE | v1 on `ENGINE_NEW` | `finish` on 24 looks | x1.00 (1.00-1.00) | x1.00 (1.00-1.00) | x1.00 (0.99-1.00) | x1.00 (0.99-1.00) | x1.00 (1.00-1.00) | x1.00 (1.00-1.00) |
| CV_COMPARE | v1 on `ENGINE_NEW` | `restrict` on 9 looks | x1.00 (1.00-1.01) | x1.00 (0.99-1.00) | x1.00 (0.99-1.00) | x1.00 (1.00-1.01) | x1.00 (0.99-1.01) | x1.00 (0.99-1.00) |
| CV_BOOL_MASKS | classical-v1 | `restrict` on 9 looks | x0.95 (0.94-0.95) | x0.94 (0.94-0.94) | x0.93 (0.93-0.94) | x0.93 (0.92-0.94) | x0.91 (0.91-0.92) | x0.91 (0.90-0.92) |
| CV_BOOL_MASKS | v1 on `ENGINE_NEW` | `restrict` on 9 looks | x0.93 (0.93-0.93) | x0.93 (0.92-0.93) | x0.92 (0.92-0.93) | x0.93 (0.92-0.93) | x0.91 (0.91-0.92) | x0.91 (0.91-0.92) |
| all three | classical-v1 | `segment_view` on 24 looks | x0.97 (0.97-0.97) | x0.96 (0.96-0.97) | x0.96 (0.96-0.96) | x0.95 (0.94-0.95) | x0.94 (0.93-0.94) | x0.94 (0.94-0.95) |

**The rule on Arm, and on the x86 runner.** `CV_ARGMAX` passes on both (x0.64 to x0.85), `CV_COMPARE` fails on both
(no gain), and `CV_BOOL_MASKS` passes on both: x0.91 to x0.95, every interval below 1. So `CV_BOOL_MASKS`, the one
pipeline change that uses a type new in 5.0, passes on Arm, where the Lambda function runs, and on the x86 runner,
but failed on this Mac, where the load stretched its v1 one-thread interval to 1.02. It still ships off, because the
rule is applied on this machine. Shipping it is the one line `CV_BOOL_MASKS = True` in `drishyam/segment.py`, with no
new dependency: its output is a copy, identical by construction and on all 14 photos on all three machines. What it
buys is small, since `restrict` runs only on looks that cross the image's edge; with all three flags on, a whole
classical look took x0.94 to x0.97 of its time on the runners.

**`ENGINE_CLASSIC` at the agent level on both runners:**

| runner | traces identical | looks | max \|dprob\| | pixels with another argmax | look time, 1 thread | 2 threads | 4 threads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Neoverse-N2 | 14 of 14 | 49 | 1.5e-5 | 0 of 12,845,056 | x0.39 (0.39-0.40) | x0.44 (0.44-0.45) | x0.58 (0.58-0.58) |
| AMD EPYC 7763 64-Core Processor | 14 of 14 | 49 | 2.2e-5 | 0 of 12,845,056 | x0.71 (0.70-0.71) | x0.76 (0.76-0.76) | x0.76 (0.76-0.76) |

On the Neoverse-N2 the agent's accurate v1 looks on `ENGINE_CLASSIC` took x0.39 (0.39-0.40) of `ENGINE_NEW`'s time on
one thread, x0.44 (0.44-0.45) on two (the function's count) and x0.58 (0.58-0.58) on four, every trace identical,
every argmax equal, max |dprob| 1.5e-5. It passes the identity asked of a DNN change and is measurably faster on Arm
at every thread count, and on the x86 runner too; it failed on this Mac only on speed, at 16 threads. Lambda is
arm64 and runs OpenCV on 2 threads (`DRISHYAM_CV_THREADS`), so this is the change with the largest gain for the live
v1. It ships off: the rule is applied on this machine, and the Lambda package cannot make its copy. With the
governor on, faster looks would also change the live choices (fewer switches to the 256-px FP16 net under the 1500 ms
limit), which the held-out runs, governor off, do not cover.

**What shipping `ENGINE_CLASSIC` on Lambda would need.** The `onnx` package is needed only to write the static-shape
copy (`segment._classic` sets the input's symbolic H and W and saves the model); `ENGINE_CLASSIC` then reads a plain
ONNX file by path. So a copy made at build time avoids the runtime dependency: write it once in the build
environment, where `onnx` is installed (`bench/opencv5_engines.py`'s `static_copy` already writes such files, and
they loaded and ran on `ENGINE_CLASSIC` on all three CPUs), into `models/`, which `scripts/build_lambda.sh` ships
whole (`cp models/*.onnx`). It is one file of 26.8 MB, the FP32 net at 512: the fast model keeps the FP16 file on
`ENGINE_NEW`, which the stack also deploys, so no 256 copy is needed. The code would need one change, not made here:
`OnnxSegmenter` reading that file by path instead of calling `_classic`, while still hashing the original FP32 file,
since `agent.WEIGHTS` pins v1's profile to that file's sha256 and a copy hashes differently (a different hash falls
back to the classical profile). And it would need the same identity and timing on the deployed function, a Graviton,
not a Neoverse-N2. Nothing here changes the Lambda package or deploys.

**Where v1's time goes on the runners** (512, one thread, ms per layer kind; the kinds that take the most of
`ENGINE_NEW`'s time on Arm):

| layer kind | Arm, `ENGINE_NEW` | Arm, `ENGINE_CLASSIC` | x86, `ENGINE_NEW` | x86, `ENGINE_CLASSIC` |
| --- | --- | --- | --- | --- |
| conv 3x3 | 800.6 | 248.5 | 388.6 | 208.5 |
| conv 1x1 | 53.7 | 53.0 | 29.6 | 55.9 |
| depthwise conv 3x3 | 7.0 | 6.6 | 5.3 | 5.5 |
| Resize | 6.3 | 7.5 | 7.0 | 16.2 |
| depthwise conv 5x5 | 6.1 | 21.8 | 5.7 | 25.4 |
| TransformLayout | 4.5 |  | 4.3 |  |
| every layer (the profiler's sum) | 888.5 | 346.7 | 453.9 | 323.9 |
| a forward, wall time (p50) | 889.6 | 347.4 | 455.5 | 324.2 |

On the Neoverse-N2 the twelve 3x3 convolutions took 800.6 against 248.5 ms, more than the whole 541.8 ms gap between
the engines, while the 1x1 convolutions took 53.7 and 53.0 ms and the new engine was the faster at the 5x5 depthwise
ones. On the EPYC the 3x3 convolutions took 388.6 against 208.5. So the new engine's Arm gap is its 3x3 convolution;
which kernel each engine runs for it is not recorded.

**5.0 against 4.14 on the runners** (one thread, ms, and 4.14's or IPP-off's time over 5.0's):

| stage | Arm, 5.0 | Arm, 4.14: time over 5.0's, output | x86, 5.0 | x86, 4.14: time over 5.0's, output | x86, 5.0 with IPP off |
| --- | --- | --- | --- | --- | --- |
| `decode` | 32.7 | x1.01 (1.00-1.02), same | 34.3 | x1.00 (0.99-1.00), same | x1.00 (0.99-1.01), same |
| `pyramid` | 8.9 | x1.86 (1.85-1.87), same | 7.9 | x2.49 (2.48-2.49), same | x1.00 (1.00-1.00), same |
| `saliency` | 6.5 | x1.00 (1.00-1.01), same | 6.0 | x0.99 (0.98-1.00), differs | x0.99 (0.98-1.00), differs |
| `gaze` | 1.9 | x1.53 (1.53-1.54), differs | 2.2 | x0.93 (0.93-0.93), differs | x1.00 (1.00-1.00), same |
| `gaze_oblique` | 1.7 | x1.80 (1.79-1.81), differs | 2.0 | x0.84 (0.84-0.85), differs | x1.01 (1.00-1.03), same |
| `quality` | 0.9 | x1.00 (1.00-1.01), same | 0.7 | x0.99 (0.95-1.01), same | x1.00 (0.95-1.05), same |
| `enhance` | 3.1 | x1.01 (1.00-1.01), same | 4.2 | x0.96 (0.96-0.97), same | x1.00 (0.99-1.00), same |
| `segment` | 43.0 | x0.98 (0.98-0.99), same | 48.6 | x0.99 (0.98-0.99), same | x1.00 (1.00-1.01), same |
| `measure` | 1.6 | x0.99 (0.99-1.00), same | 2.9 | x1.01 (1.00-1.03), same | x1.00 (0.99-1.00), same |
| `nanodet` | 47.1 | x1.59 (1.59-1.60), same | 32.4 | x2.40 (2.39-2.40), same | x1.00 (1.00-1.01), same |
| `yunet` | 24.7 | x0.88 (0.87-0.89), same | 16.3 | x1.13 (1.12-1.14), same | x1.00 (1.00-1.00), same |

On both runners 5.0's pyramid and NanoDet forward beat 4.14's with the same output: `pyramid` x1.86 (1.85-1.87) on
Arm and x2.49 (2.48-2.49) on x86, `nanodet` x1.59 (1.59-1.60) on Arm and x2.40 (2.39-2.40) on x86. The warp depends on
the machine: 4.14 took x1.53 (1.53-1.54) and x1.80 (1.79-1.81) on Arm of 5.0's time on the straight and oblique gaze,
as on the Mac, but x0.93 (0.93-0.93) and x0.84 (0.84-0.85) on the EPYC, where 4.14's warp was the faster; its output
differs everywhere. YuNet went the other way on Arm (`yunet` x0.88 (0.87-0.89)). On Arm the comparison also mixes in
KleidiCV: the aarch64 4.14 wheel's HAL line reads YES (carotene (ver 0.0.1)), so a stage KleidiCV covers by name
(`pyrDown`, `warpPerspective`) may owe part of its Arm gain to the HAL, which this run cannot split from 5.0's own
changes; on x86 both wheels have IPP, so the pyramid's gain there is 5.0's. With IPP off on the EPYC no stage moved.
On the x86 runner `saliency`'s output hash was not the same in every 5.0 worker, the reference's included, so that
row's output column says nothing there (on Arm and on the Mac every stage hashed the same in every worker); why was
not looked into.

## What is not measured

- Lambda itself: no timing in this page comes from the function; the GitHub arm64 runner is a different VM, a
  different core (Neoverse-N2) and Python 3.13 against the function's 3.12.
- 2 and 4 threads on this Mac (GCD), and any number on an idle Mac: the machine was shared throughout.
- Run-to-run variation on GitHub's runners: run 39220769299 is one run per runner and one round of Metric B. The
  only repeat is the aarch64 micro, a day apart on the same build (37 of 39 ops within 3.5 %, the other two 6.5 % and
  18.1 % apart); x86 was never repeated.
- What KleidiCV itself buys on Arm: no run turns the HAL off, so Arm against x86 is core, build and VM together, and
  the aarch64 4.14 wheel (carotene only) against 5.0 mixes KleidiCV with 5.0's own changes.
- Whether the x86 runner's 4 logical CPUs are 4 cores or 2 cores with two hardware threads each.
- Mask parity between the Arm and x86 builds: `run_all.sh` saves `parity.npz` on each runner, but the workflow does
  not upload it.
- The macro worker sweep (`WORKERS`) on GitHub: one worker only.
- The dacl10k net on Arm in INT8: the x86 INT8 result above does not predict it. FP32 and FP16 ran on both engines
  there once (run 39617514621, "OpenCV 5 DNN engines for v1").
- "OpenCV 5 features in the pipeline" on the function itself: its two benchmarks ran on this Mac and once on each
  GitHub runner (run 39665440222), never on the Lambda's Graviton.
