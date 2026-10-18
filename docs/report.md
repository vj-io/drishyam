# Drishyam — submission report

*Virtual-PTZ inspection of bridge photos with an agent that stops looking when the next look is worth less than it
costs.* An entry in the OpenCV AI Competition 2026, powered by AWS. Live demo: https://d1xg58guluup8c.cloudfront.net · Code: https://github.com/vj-io/drishyam

Sections: 1 Problem · 2 Users · 3 How it works · 4 Architecture · 5 AWS deployment · 6 OpenCV 5 implementation ·
7 Related work · 8 Evaluation · 9 Failures · 10 Agentic evidence · 11 Responsible use and limits · 12 Team and
reproduce · Appendix: per-cap tables.

## 1. Problem

The US National Bridge Inventory lists 624,939 bridges, 41,148 of them in poor condition (FHWA, 2026 data). Each
must have a routine inspection at intervals of up to 24 months (23 CFR 650.311). One detailed drone inspection of
the St. Johns Bridge in Portland, Oregon produced 2,536 still photos (Oregon DOT and FHWA, report SPR 787), and the
dacl10k authors note that authorities often fail to keep the required inspection intervals because of staff
shortages. Every figure, with its source and the date it was checked: [impact-sources.md](impact-sources.md).

The defects that matter most (hairline cracks, early exposed rebar) are small at thumbnail scale, so a tool must
either tile every photo at full resolution or guess where to look. Drishyam's bet is that *looking is cheap when it
is a geometric warp*, and that where to look next can be read from what OpenCV just measured. This report makes no
time-saving claim: on real photos Drishyam's recall is about 0.2 (section 8).

## 2. Users

The PRD names three. An **inspection engineer** reviewing a photo backlog uploads a photo and gets findings with
the crop, the class, pixel length, width and area, and the numbered gaze step that found each one, so the evidence
comes before any score. Critical findings (exposed rebar, spalling above 40,000 px²) wait for that engineer's
approval, which the agent cannot give or downgrade, and a per-job budget in pixels, dollars and seconds sets what
the triage of a backlog may cost. **Judges** need a trace in which an OpenCV output changes the next tool call,
failure drills they can trigger, a live endpoint and honest failure cases. The **operator** who pays for the AWS
account needs caps, a budget alert and one command that stops all spending.

Not built yet: approval from an SNS email link (approval is the web review or `drishyam review`); the COOL
benchmark (section 6); the 5-minute video; delivery of alarms and budget alerts (the SNS topic's one email
subscription awaits confirmation, section 5); a budget stop action or auto-stop schedule (the stop is `scripts/aws.sh stop`); a live LLM planner (the
drills use scripted planners, section 3). Dropped: the Hugging Face Space mirror; the demo runs only on AWS.

## 3. How it works

![The agent loop: perception, decision, action and the human gate](agent-loop.svg)

Each job is a loop of camera moves over an image pyramid (L0 = full resolution, L4 = 1/16):

1. **Overview.** `pyrDown` to L4; a saliency prior at L3 from `cvtColor`, `GaussianBlur`, `Sobel`, black-hat
   `morphologyEx`, `adaptiveThreshold`, CLAHE and a rust-hue `inRange`; up to 20 ranked candidate windows.
2. **Gaze.** `warpPerspective` renders a 512² virtual-PTZ view whose pixels map back to L0 exactly: pan and zoom.
   The agent always passes yaw = pitch = 0; only the MCP tool exposes yaw and pitch, as a synthetic rotation with
   f = view size, not a calibrated rectification.
3. **Quality gate.** Laplacian sharpness and exposure; glare and blur are logged and not trusted.
4. **Segment.** Per-class probabilities, masks and an entropy score. The demo engine is classical (evidence maps,
   no weights); a dacl10k ONNX model runs on OpenCV 5 DNN `ENGINE_NEW` with the same interface.
5. **Decide.** The next look is `argmax p·s·(1+u)·a / t̂` (prior × severity × uncertainty × unseen area ÷ predicted
   ms). Escalate one level finer, centred on what it saw, when entropy > τ or the mask is below A_min; verify a new
   confident mask one level finer by mask IoU; log findings with length, width, area and orientation in pixels.
6. **Govern.** Before every step: kill switch, pixel / $ / seconds caps, p95 latency against the SLA (switch to the
   fast model, then a coarser level), warp spikes, and a $ ≥ 80 % mode that verifies only critical classes.
7. **Stop** when the best remaining expected value per ms falls below a quarter of the realised value per ms so far,
   or below a floor, the value of a look that finds nothing (ε·v0/t̂, v0 = 2), once every candidate left has been
   shown by a look as fine as its own first look (a coarse survey alone never ends a job). Under a pixel cap the cap
   usually binds first: at the 30 % cap, 125 of 200 synthetic and 183 of 200 CODEBRIM jobs end on the cap; uncapped,
   the stop rule ends 183 of 200 synthetic jobs and 197 of 200 CODEBRIM jobs (these counts are from the published
   runs, made before the floor and the seeding of unvisited candidates existed; they have not been re-run on this
   code).

Critical findings (exposed rebar; spalling above 40 000 px²) wait in `pending_review`. Only a human — the web review
or `drishyam review` — can approve or reject them; nothing the agent, the MCP tools or the planner can call reaches
that transition.

No LLM runs in the live demo. The deployed agent is the rule policy above (the stack sets no
`DRISHYAM_PLANNER_URL`), and the planner failure drills use planners scripted to time out or answer junk. The LLM
path is the MCP server: section 10 links a recorded session in which Claude chose each call from the measured
values.

## 4. Architecture

![System architecture: clients, the AWS path, the agent core, the OpenCV 5 layer and the outputs](architecture.png)

Three clients reach one agent core: the browser through CloudFront and an arm64 Lambda (section 5), and the CLI and
the MCP server locally. The core (policy, governor, optional planner behind an allow-list validator, human gate)
drives the OpenCV layer and emits an NDJSON trace and a signed work order. Every cv2 call in the figure also exists
in OpenCV 4.x except the starred `ENGINE_NEW` argument (section 6). Sources: [architecture.svg](architecture.svg)
and [architecture.mmd](architecture.mmd); the agent loop above is [agent-loop.svg](agent-loop.svg).

## 5. AWS deployment

One CDK (Python) stack in `us-east-1` (`infra/`), deployed with `scripts/aws.sh deploy`; the full page, with IAM,
limits and the cost table, is [aws.md](aws.md).

- **Entry.** CloudFront with two origins, each behind Origin Access Control: the static site from a private S3
  bucket, and `/api/*` to a Lambda Function URL with `AuthType AWS_IAM` and response streaming, so the URL cannot be
  called around CloudFront. Every `POST` carries `x-amz-content-sha256`, which the page computes.
- **AWS WAF.** Three per-IP rate rules: 10 job starts per 120 s, 30 kill or review calls per 60 s, 300 requests per
  60 s.
- **Compute.** Lambda arm64 (Graviton), Python 3.12, 3008 MB, 60 s timeout, with the Lambda Web Adapter running the
  FastAPI app unchanged. It runs the stock `opencv-python-headless` 5.0.0 wheel (section 6), not COOL. The deployed
  stack sets no reserved concurrency: the account's Lambda quota of 10 is the cap.
- **State and secrets.** A DynamoDB table holds kill flags (read before every step, so `/api/kill` reaches a job in
  another Lambda environment); Secrets Manager holds the HMAC key that signs work orders. The function's role can do
  only that plus writing its own logs and X-Ray traces, and `tests/test_infra.py` fails if it gains anything else.
- **Observability.** One EMF log line per job, a CloudWatch dashboard, 4 alarms (Errors, Throttles, Duration p95,
  AppErrors) and X-Ray tracing. The alarms and a 20 USD monthly budget publish to one SNS topic; its one email
  subscription awaits confirmation, so until it is confirmed they reach nobody.
- **Smoke test** (`scripts/aws.sh smoke`, `scripts/smoke.sh`): `smoke 2026-10-18T07:12:25Z result=pass`: a job ran
  to `awaiting_review`; KILL, sent after step 2 streamed, halted the second job after 3 steps (before step 4) with
  cause `kill-switch`; a `POST` without the body hash and a direct Function URL call both got 403; the WAF answered
  429 at the 14th empty-bodied job start (WAF evaluates rates about every 10 s, so it blocks late); and the step
  p95 was 67.7 ms on aarch64, over the 6 steps of one job.
  That run was against the deployed build of commit ca69037, the commit `/api/health` reports. An earlier run against
  the same build (`smoke 2026-10-18T07:09:20Z`, without the WAF check) also passed but measured more weakly: step p95
  83.7 ms over 6 steps, and KILL halted the second job only after 6 steps ([aws.md](aws.md) quotes both runs).
- **Cost.** Spent so far: not measured yet; the month-to-date figure from AWS Billing is still to be added. The
  list-price estimate is about 10 USD a month at 1,000 jobs a month, 8 of it the WAF ([aws.md](aws.md)). The $ shown
  in the UI and the evals is *modelled*: timed op milliseconds × 0.339 USD/h, the c8g.2xlarge rate plus the COOL
  Marketplace fee the PRD planned for. It is not the Lambda bill, and decode and the privacy blur, which run before
  the job starts, are not in it.
- **Operate.** `scripts/aws.sh stop` sets the function's reserved concurrency to 0 (nothing runs, nothing bills);
  `start` undoes it (on the live deployment, which has no reservation, run `RESERVED=none scripts/aws.sh start`);
  `destroy` deletes the stack and the site bucket.

## 6. OpenCV 5 implementation

| Stage | cv2 calls | Module |
| --- | --- | --- |
| Decode | `imdecode`, after a header sniff | `imaging.py` |
| Privacy blur | `FaceDetectorYN` (YuNet); `dnn.readNet`, `dnn.blobFromImage`, `dnn.NMSBoxes` (NanoDet-Plus); `GaussianBlur` | `privacy.py` |
| Pyramid and saliency | `pyrDown` × 4; at L3 `cvtColor`, `createCLAHE`, `GaussianBlur`, `Sobel`, `magnitude`, `morphologyEx` (black-hat), `adaptiveThreshold`, `inRange`, `connectedComponentsWithStats` | `imaging.py` |
| Gaze | `warpPerspective` (`INTER_LINEAR`, `BORDER_REFLECT_101`), `perspectiveTransform` | `imaging.py` |
| Quality gate | `Laplacian` variance; glare and dark pixel fractions | `imaging.py` |
| Segment | classical engine: `cvtColor`, `createCLAHE`, `GaussianBlur`, `medianBlur`, `morphologyEx`, `inRange`, `dilate`, `connectedComponentsWithStats`; ONNX engine: `dnn.readNetFromONNX(path, engine=cv2.dnn.ENGINE_NEW)` | `segment.py` |
| Measure and verify | `morphologyEx` (hit-or-miss thinning), `findContours`, `minAreaRect`, `arcLength`, `approxPolyDP`; cross-level mask IoU through `warpPerspective` and `dilate` | `measure.py`, `agent.py` |

Only one call is specific to OpenCV 5: the `engine=cv2.dnn.ENGINE_NEW` argument (`OnnxSegmenter._load` in `segment.py`), on the opt-in
ONNX path, which `tests/test_segment.py` exercises with a tiny ONNX model. The live demo runs the classical engine,
so it uses no OpenCV-5-only API; it runs on the stock OpenCV 5.0.0 wheel.

That wheel's aarch64 build already includes Arm's KleidiCV and carotene: the Lambda package pins the same
`opencv-python-headless` 5.0.0.93 wheel (`pyproject.toml`), whose `cv2.getBuildInformation()` on GitHub's arm64
runner reads "Custom HAL: YES (carotene (ver 0.0.1) KleidiCV (ver 26.03))" and "Baseline: NEON FP16"
(`results/linux-aarch64/micro.json`); the Lambda's own build information is not recorded. The camera moves use ops KleidiCV covers in some formats (`pyrDown`,
`resize`, `warpPerspective`), so a COOL gain would have to come on top of the KleidiCV the wheel already has.
Whether the 3-channel `BORDER_REFLECT_101` `warpPerspective` used here takes the HAL path has not been measured. There is no `filter2D` and no NumPy pixel loop on the gaze path; the classical
segmenter adds `connectedComponentsWithStats` and a NumPy softmax, and measurement adds `findContours`.

### COOL on Graviton (not run)

COOL on Graviton was not run: the Marketplace AMI could not be subscribed from our AWS account (a payment-method
restriction); the three-arm scripts are in [bench/](../bench/README.md). `bench/run_all.sh <arm>` runs the micro and
macro benchmarks and a parity save with a provenance block (OpenCV version, build-info hash, whether `cv2` loads
from `/opt/cool`), `bench/parity.py` checks that two arms produce identical masks, and the same code passes CI on
GitHub's arm64 runners. [bench/aws_runbook.md](../bench/aws_runbook.md) is the copy-paste sequence for the three
arms (COOL on c8g, stock on c8g, stock on c7i), each set to terminate itself after 3 hours, and `python -m
bench.compare_arms results/A results/B results/C` would write `docs/cool_benchmark.md` against the PRD's expected
1.2–1.8× op-level and 1.2–1.3× end-to-end range. None of it has run on AWS.

**What COOL could buy end to end, from GitHub's arm64 runner.** A 50-image macro run with the stock aarch64 wheel
on GitHub's `ubuntu-24.04-arm` runner (Neoverse-N2, one thread; a GitHub Actions VM, not the Lambda;
[results/linux-aarch64/macro.json](../results/linux-aarch64/macro.json)) puts 8 % of a job in the stages it labels
KleidiCV-covered imgproc (pyramid, saliency, warp, quality gate), 75 % in the classical segmenter (imgproc plus
NumPy), 4 % in crack thinning, 8 % in decode and 6 % in Python. So if COOL made the covered ops 1.8× faster, a whole
job would get about 1.04× faster; the x86 laptop's 8-image run
([results/local-macos-x86/macro.json](../results/local-macos-x86/macro.json)) gives the same 1.04×. The gaze path itself is cheap; most of the time is in segmentation, which is
where a COOL or DNN speed-up would have to land.

## 7. Related work

Drishyam combines old ideas; this section says which, and what it adds.

- **Image pyramids.** Burt and Adelson's Laplacian pyramid [1] made coarse-to-fine processing over a stack of
  low-pass copies of an image standard. Drishyam's L0 to L4 are a Gaussian pyramid built with `cv2.pyrDown`, and
  every look is a crop of one level.
- **Saliency.** Itti, Koch and Niebur [2] combine colour, intensity and orientation contrast across scales into one
  saliency map that orders where attention goes. Drishyam's prior is the same kind of hand-built map, with features
  chosen for concrete damage (black-hat for dark thin lines, Sobel edges, adaptive threshold, CLAHE, a rust-hue
  range), and it ranks up to 20 candidate windows.
- **A low-resolution view steering a high-resolution one.** Gould, Arfvidsson, Kaehler, Sapp, Messner, Bradski,
  Baumstarck, Chung and Ng [3] use a learned attentive interest map on a low-resolution video stream to point a
  high-resolution fovea, and follow recognised objects in the periphery, which keeps recognition real-time.
  Drishyam's overview-then-zoom is the same idea applied to one large still photo: the periphery is a coarse level
  of the same pyramid and the fovea is a `warpPerspective` crop. Its attention map is hand-built, not learned, and
  nothing is tracked.
- **Learned glimpse policies.** Mnih, Heess, Graves and Kavukcuoglu [4] train a recurrent network with reinforcement
  learning to choose a sequence of glimpses, so the computation is set by the number of glimpses rather than the
  image size. Drishyam's policy is not learned: a fixed score and a handful of rules, which keeps every choice
  explainable from logged numbers but means nothing is optimised end to end.
- **Tiling.** SAHI [5] slices a large image into overlapping tiles, runs the detector on each and merges the
  results, the standard answer to small objects in large images. Drishyam's tiling baseline re-implements SAHI's
  slicing (512 px tiles, 20 % overlap, the last row and column pulled back to the edge) and its greedy NMM merge in
  `evals/harness.py`, rather than importing the library, and runs the same grid on pyramid levels 1 and 2 for
  smaller budgets.

**What Drishyam adds** is engineering around those ideas:

1. A value-per-ms score: the next look maximises expected value per predicted millisecond, where the prediction is
   the job's own measured p95 of warp and segment time (in the published evals, a fixed table of p50s, so a re-run
   reproduces them).
2. A stop rule: the job ends when the best remaining look is worth less than a quarter of the value per ms realised
   so far, or than a floor (ε·v0/t̂, the value of a look that finds nothing) once every candidate left has been
   seen by a look as fine as its own first look.
3. Cross-scale verification: a new confident mask is looked at again one pyramid level finer and compared, class by
   class, by mask IoU through the two views' homographies. Below 0.5 for the same class with the same place seen
   as another class (IoU ≥ 0.5 between the two looks' best masks) it is a class change, otherwise a disagreement;
   the finding of that class is then `verified: false`, and a human is asked when it is critical, when no finer
   look can settle it, or when a class change involves a critical class (the finding named then waits in
   `pending_review`), with the measured IoU as the reason.
4. A governor that checks kill, caps and p95 latency before every step and writes a cause for each override, so
   every step in the trace says why it happened.
5. A human gate: critical findings wait in `pending_review`, and no agent, MCP or planner call can approve them.

**What is not new:** coarse-to-fine pyramids, saliency priors, steering a high-resolution window from a
low-resolution map, and tiling. On every set at least one simple baseline matches or beats Drishyam's recall at
similar or lower pixel cost (section 8): one downscale on the 2048×1536 set and on CODEBRIM, the L2 tiling grid at
4608×3456 and on CODEBRIM. Only at 4608×3456 does zooming beat one downscale, narrowly. So the combination has not
yet paid off in recall.

1. P. J. Burt and E. H. Adelson, "The Laplacian Pyramid as a Compact Image Code", *IEEE Transactions on
   Communications*, vol. COM-31, no. 4, pp. 532–540, April 1983.
2. L. Itti, C. Koch and E. Niebur, "A Model of Saliency-Based Visual Attention for Rapid Scene Analysis", *IEEE
   Transactions on Pattern Analysis and Machine Intelligence*, vol. 20, no. 11, pp. 1254–1259, November 1998.
3. S. Gould, J. Arfvidsson, A. Kaehler, B. Sapp, M. Messner, G. Bradski, P. Baumstarck, S. Chung and A. Y. Ng,
   "Peripheral-Foveal Vision for Real-time Object Recognition and Tracking in Video", *International Joint
   Conference on Artificial Intelligence (IJCAI)*, 2007, pp. 2115–2121.
4. V. Mnih, N. Heess, A. Graves and K. Kavukcuoglu, "Recurrent Models of Visual Attention", *Advances in Neural
   Information Processing Systems 27 (NIPS 2014)*, pp. 2204–2212.
5. F. C. Akyon, S. O. Altinuc and A. Temizel, "Slicing Aided Hyper Inference and Fine-tuning for Small Object
   Detection", *IEEE International Conference on Image Processing (ICIP)*, 2022, pp. 966–970,
   doi:10.1109/ICIP46576.2022.9897990.

## 8. Evaluation

Measured on **200 held-out labelled synthetic images** (seed 7; policy changes were tuned only on dev splits, unpublished: 60 synthetic images, seed 4242, `make eval-dev`; 30 synthetic images with no damage, seed 4243, `make eval-dev-clean`; and 100 CODEBRIM photos, seed 4244, none of them in the 200 below, `make eval-dev-real`), one segmenter and one matching rule for every policy (a ground-truth box is found by a same-class component with IoU ≥ 0.3 or containing its centre; "IoU-only" drops the centre clause). The 95 % intervals are a bootstrap over images, paired across policies. "Chance" moves each policy's own boxes (same class and size) to random places in the photo and scores them again. Precision is one-to-one: in confidence order each prediction claims at most one ground-truth box, so a defect reported again from a second look is a false box; "hits per matched box" counts the predictions that land on each box found (1 = no duplicates). All of it runs on the **fixed cost model** (`--cost-model fixed`): every op the agent times costs its p50 from the `job_ops` table in `results/linux-aarch64/micro.json` (the stock aarch64 wheel on GitHub's `ubuntu-24.04-arm` runner, Neoverse-N2, one thread) on every call, so t̂, the stop rule and the governor see the same costs on a busy machine as on an idle one, and a re-run (`make eval`, whose `MICRO_ARM` defaults to `linux-aarch64`) reproduces the numbers exactly on an x86 machine, however busy it is (on arm64 the synthetic images themselves come out slightly different: the aarch64 HAL rounds some pixel ops differently). The jobs themselves ran on the x86 laptop; only their costs are the arm64 ones. $ is that modelled time × $0.339/h (untimed Python is not charged). In this comparison the governor's model, level and gaze-size overrides are off (caps and the kill switch stay), so Drishyam segments every look with the same accurate model as the baselines; turning them on changes nothing under fixed costs at the 30 % and 100 % caps, the only caps the governor-on ablation ran (30 % only on CODEBRIM; [Ablations](#ablations-fr-21) below).

The FR-21 ablations (random gaze order, no uncertainty term, no survey look, no verify step, governor on) are under
[Ablations](#ablations-fr-21) below. The per-cap tables and per-class recall are in the
[appendix](#appendix-per-cap-tables).

### Synthetic, 2048×1536 (200 images)

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall | chance recall (IoU-only) | false boxes / image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.25 | 0.50 (0.46–0.54) | 0.43 | 0.02 (0.01) | 12.2 |
| tiling, L2 grid | 0.06 | 0.44 (0.41–0.48) | 0.40 | 0.02 (0.01) | 8.5 |
| tiling, L1 grid | 0.50 | 0.55 (0.52–0.59) | 0.48 | 0.02 (0.01) | 17.3 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.67 | 0.55 (0.51–0.59) | 0.47 | 0.03 (0.01) | 83.4 |
| **Drishyam**, cap 30 % | 0.23 | **0.47** (0.43–0.51) | 0.42 | 0.04 (0.02) | 13.9 |
| **Drishyam**, uncapped | 0.35 | **0.51** (0.48–0.55) | 0.45 | 0.03 (0.01) | 19.1 |

- At a **30 % cap** Drishyam finds **0.47** of the defects using **0.23** of the image's pixels, against 0.55 for
  exhaustive 512-px tiling with 20 % overlap, which reads 1.67× the image. That is **0.85×** tiling's recall (95 % CI
  0.79–0.91) at **0.14×** its pixels. **Verdict on the PRD hypothesis (≥ 0.9× tiling recall at ≤ 0.3× its pixels):
  not met**, though the interval reaches 0.91, so 0.9 is not ruled out either.
- **One downscale does at least as well**: against a single 1024-px downscale at about the same pixels (0.50 at
  0.25), Drishyam is 0.94× (95 % CI 0.89–1.00); the interval only just reaches 1.
- Against the other grids it is 0.84× L1 tiling (0.55 at 0.50 of the pixels, the best grid) and 1.06× L2 tiling
  (0.44 at 0.06). When a match needs IoU ≥ 0.3 (no contains-the-centre clause), it is 0.91× L0 tiling (95 % CI
  0.84–0.98).
- How predictions are boxed changes the ratio. Every number here gives each policy one box per connected component.
  With one box per class per view for every policy, the 30 % ratio to L0 tiling is 0.73×; with Drishyam's finding
  boxes against the baselines' components (the as-specified mix, which favours Drishyam) it is 1.21×
  (`agent_metrics.box_convention` in the summary).
- At a **50 % cap**: 0.51 at 0.31 of the pixels, 0.92× tiling's recall at 0.19× its pixels. Uncapped, the stop rule
  ends 183 of 200 jobs by itself, at 0.35 of the pixels with 0.94× tiling's recall.
- **Above chance**: the same boxes at random places score 0.00 to 0.04 for every policy.
- **False boxes**: precision 0.102 for Drishyam at 30 % (13.9 false boxes per image) against 0.120 for one
  downscale (12.2) and 0.021 for exhaustive tiling (83.4). Drishyam reports a defect again from each look in its
  evidence trail (1.49 predictions per matched box, against 1.03 for one downscale).
- **Hairline cracks** (87 of the 227 crack boxes are 1–2 px wide; the flag is in the ground truth): exhaustive tiling
  finds 0.25 of them, one downscale 0.10, Drishyam 0.08 at the 30 % cap and 0.13 uncapped. Only 16 boxes are under
  96² px, too few to read anything into.
- The segmenter's crack hysteresis, added with this policy (a faint crack pixel counts when it is connected to a
  confident one, as in Canny), lifted every baseline too, so it is not a Drishyam-only gain.
- Critical false-accept rate (critical prediction with no critical ground truth): Drishyam at 30 % 0.51 against
  exhaustive tiling 0.79 (the PRD asks for ≤ tiling); a single downscale (0.49) and L2 tiling (0.42) are lower still.
- Agent: 100 % of steps carry a cause; escalation rate 19 %; at the 30 % cap, 3 looks per image (p50 = p95) and
  0.23 of the pixels on average: the 125 of 200 jobs that hit the cap each use 0.27 (so p50 = p95 = 0.27), and the
  other 75 stop earlier, at 0.18 (71 on the stop rule, 4 with every candidate looked at).

The images are procedural (`evals/synth.py`): 2048×1536 weathered concrete with drawn cracks, spalls, efflorescence, exposed rebar and rust run-off, plus graffiti, formwork lines, water stains, blur and glare as distractors. The eight bundled samples in `public/samples/` come from the same generator.

![Recall against pixels processed](recall_vs_pixels.png)

(The live site's FIG. 4.1, rendered from the committed `public/results/summary.json`: bars are the 95 % intervals, dotted ticks the chance level; `scripts/render_figures.py` re-renders this image.)

Full numbers: [public/results/summary.json](../public/results/summary.json); every prediction and trace: [public/results/per_image.json.gz](../public/results/per_image.json.gz).

**Fixed costs against measured wall time.** `make eval-live` reruns the synthetic set the way the app runs, on measured wall time ([public/results/live/summary.json](../public/results/live/summary.json)), here with the machine's load average near 39 from other jobs. It was not re-run: it dates from 2026-10-17 21:37 UTC, before segmenter, agent and imaging changes that the fixed-cost runs above include (among them: straight crack components dropped, cracks drafted only if elongated, blown-out or black areas given no looks), so it differs from them in code as well as in costs. On that older code Drishyam at 30 % found 0.48 (0.45–0.52) at 0.25 of the pixels, 0.88× tiling (0.82–0.94), and the governor never switched the model but shrank the gaze and batched on 6 warp spikes. The stop rule ended 29 of 200 jobs at the 30 % cap, against 71 in the fixed-cost run; with both the code and the cost model changed, the files do not say how much of that gap each change causes. Live numbers move with the machine; the fixed ones are the comparison.

### Synthetic at CODEBRIM's size, 4608×3456 (100 images)

`make eval-large` builds 100 images with the same generator (seed 7) at 4608×3456, the size of most CODEBRIM photos, and runs the same harness, to test the claim that zooming should beat one downscale as photos grow. Published as it came out: [public/results/large/summary.json](../public/results/large/summary.json).

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall | chance recall | false boxes / image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.05 | 0.45 (0.39–0.51) | 0.40 | 0.02 | 16.0 |
| tiling, L2 grid | 0.10 | 0.53 (0.47–0.59) | 0.48 | 0.02 | 70.4 |
| tiling, L1 grid | 0.40 | 0.53 (0.47–0.59) | 0.45 | 0.02 | 529.5 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.63 | 0.32 (0.26–0.38) | 0.20 | 0.01 | 215.5 |
| **Drishyam**, cap 30 % | 0.28 | **0.51** (0.45–0.56) | 0.44 | 0.04 | 87.8 |
| **Drishyam**, uncapped | 0.46 | **0.51** (0.46–0.57) | 0.45 | 0.04 | 121.5 |

- Against one downscale, zooming now wins, narrowly: Drishyam at 30 % finds 0.51 against 0.45, 1.12× (95 % CI 1.02–1.22), where on the 2048×1536 set it was 0.94× (0.89–1.00). The downscale shrinks these images 4.5× instead of 2×.
- A coarse tiling grid still does better: L2 tiling finds 0.53 at 0.10 of the pixels, about a third of Drishyam's. Exhaustive L0 tiling falls to 0.32 (spalling 0.16, exposed rebar 0.00): the generator scales spalls with the image, so here they are about 225 to 860 px across and a 512-px full-resolution tile often holds only part of one. The headline ratio against L0 tiling (1.60×, which formally "meets" the 0.9× target) says more about L0 tiling than about Drishyam.
- Hairline cracks (39 boxes) stay nearly invisible to every policy: 0.15 for L0 tiling, 0.10 for L1, 0.03 to 0.05 for Drishyam from the 10 % cap up (0.00 at 5 %), 0.03 for one downscale. Uncapped, the stop rule ends 88 of 100 jobs by itself (the other 12 end when every candidate has been looked at).

### Real bridge photos: CODEBRIM (200 images)

The same harness, segmenter and matching rule on **200 real inspection photos** from CODEBRIM: a seeded subset (seed 7) of the 1,052 annotated originals, mostly 4608×3456 px (2 to 24 MP), holding 1,008 defect boxes with 1,685 class labels (a box showing several defects is scored once per class) and at least 175 labels in every class. `make data` downloads and checks the 8.3 GB archive and `make eval-real` runs it; the 200 file names are committed in `evals/codebrim_eval200.txt`. The dataset's licence is non-commercial and bars sharing the photos, so only the numbers and a text description of each failure are published (the failure crops stay in the git-ignored `results/codebrim/`). Two matching rules are reported side by side, because on these photos they disagree: the lenient one (IoU ≥ 0.3 or containing the centre) and IoU-only. Same fixed cost model as above.

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall (95 % CI) | chance recall (IoU-only) | boxes per image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.07 | 0.20 (0.17–0.23) | 0.08 (0.06–0.10) | 0.10 (0.02) | 184 |
| tiling, L2 grid | 0.10 | 0.21 (0.19–0.24) | 0.08 (0.06–0.10) | 0.09 (0.02) | 254 |
| tiling, L1 grid | 0.42 | 0.22 (0.19–0.25) | 0.08 (0.06–0.10) | 0.09 (0.02) | 930 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.62 | 0.21 (0.17–0.24) | 0.06 (0.04–0.07) | 0.08 (0.01) | 3,144 |
| **Drishyam**, cap 30 % | 0.29 | **0.20** (0.17–0.23) | 0.09 (0.07–0.11) | 0.12 (0.03) | 891 |
| **Drishyam**, uncapped | 0.42 | **0.22** (0.19–0.25) | 0.10 (0.07–0.12) | 0.14 (0.03) | 1,280 |

- **The PRD's hypothesis test (G1, defined on these photos) passes on paper and means little.** At a 30 % cap Drishyam reaches **0.96×** exhaustive tiling's recall (95 % CI 0.86–1.06, so not clearly above 0.9) at **0.18×** its pixels (1.54×, 1.25–1.92, when a match needs IoU ≥ 0.3), and **0.99×** a single 1024-px downscale (0.91–1.08) that uses a quarter of its pixels: the downscale does as well for far fewer pixels. Every baseline, and Drishyam from the 20 % cap up, finds only about one labelled defect in five (0.18 to 0.22).
- **Much of that lenient score is chance.** Each policy's own boxes, moved to random places, already score 0.07 to 0.14 under the lenient rule, because thousands of boxes per image (891 for Drishyam at 30 %, 3,144 for L0 tiling) contain many ground-truth centres by luck. Under IoU-only, recall is 0.04 to 0.10 and chance 0.01 to 0.03: above chance, but low for every policy.
- **On real photos the segmenter, not where the agent looks, limits recall.** The classical engine was tuned on synthetic images. Precision (one-to-one) is 0.0005 to 0.007, and 98 to 100 % of every policy's critical predictions match no critical ground truth. It finds cracks (0.30) and corrosion (0.53) and effectively misses spalling (0.00) and exposed rebar (0.01).
- **The agent's control carries over.** 100 % of steps carry a cause; 40 of 40 planner drills and 10 of 10 governor drills recovered. At the 30 % cap a job takes 17 looks (p50; 26 at p95), against 3 on the smaller synthetic images; uncapped, the stop rule ends 197 of 200 jobs, at 0.42 of the pixels.

Full numbers: [public/results/codebrim/summary.json](../public/results/codebrim/summary.json), also on the live site (FIG. 4.2). CODEBRIM: Mundt, Majumder, Murali, Panetsos and Ramesh, "Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset", CVPR 2019.

### Ablations (FR-21)

`make eval-ablations` and `make eval-ablations-real` rerun the agent with one `JobConfig` switch flipped at a time,
on the same images, segmenter, matching rule and fixed costs, and compare each variant's recall with the full
agent's by a paired bootstrap over images ([public/results/ablations.json](../public/results/ablations.json)). Ratio
to the full agent at the **30 % cap** (95 % CI):

| Variant | synthetic 2048×1536 (200) | synthetic 4608×3456 (100) | CODEBRIM (200) |
| --- | --- | --- | --- |
| full agent (recall) | 0.47 | 0.51 | 0.20 |
| random gaze order, seeds 0–2 (pooled CI only) | 0.97–1.00 | 0.85–0.94 | 0.92–0.98 |
| no uncertainty term | 1.00 (identical) | 1.00 (identical) | 1.00 (identical) |
| no survey look | 0.90 (0.84–0.97) | 0.88 (0.82–0.93) | 0.66 (0.60–0.72) |
| no verify step | 1.03 (1.00–1.07) | 1.00 (0.98–1.02) | 1.03 (0.99–1.07) |
| governor on | 1.00 (identical) | 1.00 (identical) | 1.00 (identical) |

- **The survey look earns its pixels everywhere**: without it recall drops on every set, most on the real photos
  (0.66×). Uncapped it still costs recall (0.85×, 0.80–0.90, at 2048×1536; 0.89×, 0.83–0.94, at 4608×3456).
- **The image-driven gaze order beats random order only on the larger photos.** At 4608×3456 and on CODEBRIM the
  pooled interval is below 1; on the 2048×1536 set it sits inside the 0.95–1.05 margin set in advance, so there
  random order matches the full agent. Uncapped, random order matches it on both synthetic sets.
- **Two parts do nothing measurable on these sets.** The uncertainty term `(1+u)` in the score gives identical
  recall, pixels and verify looks when it is removed, and turning the governor on gives identical results, because
  under fixed costs its latency rules never fire (its response is shown by the drills instead, section 10). The
  uncertainty result is a property of the code that ran, not of the term: then only regions already looked at had
  an entropy and only unvisited ones are scored, so every argmax read u = 0 (and severity 1). Every look now seeds
  the unvisited candidates it shows; the ablations have not been re-run on that code.
- **The verify step buys no recall**: without it recall is never measurably lower, and at the 30 % cap on
  2048×1536 it is slightly higher, so there the verify looks cost recall; uncapped at 2048×1536 it is 0.99× (0.96–1.02) at 0.31 of the pixels instead of 0.35. Its value is the cross-scale
  agreement it records for the human reviewer, not recall.
- None of the variant-against-full intervals is corrected for multiple comparisons. CODEBRIM was run at the 30 % cap
  only.

### False alarms on images with no damage (51 images)

`make eval-clean` runs the same policies on 50 synthetic images that hold only distractors (one or two of
formwork lines, water stains, graffiti and glare) plus the demo's clean sample `bridge_08`, so every box is a false alarm
([public/results/clean/summary.json](../public/results/clean/summary.json)). Per image, with one box per component,
Drishyam at the 30 % cap reports 6.35 boxes (0.02 critical), one downscale 3.00 (0.00) and exhaustive tiling 36.80
(0.04); with one box per class per view (what a Drishyam finding is), 1.04, 1.16 and 12.69. Drishyam reports some
damage on 67 % of these images and a critical finding on 1 of the 51; one downscale reports damage on 88 % and
exhaustive tiling on all of them. On `bridge_08` itself Drishyam reports nothing at any cap, one downscale 2 crack
boxes and exhaustive tiling 53.

### The trained segmenter (results pending)

These numbers call for a trained segmenter. A dacl10k-trained UNet ([training/](../training/README.md),
[models/README.md](../models/README.md)) exists: its mIoU on the dacl10k validation split over Drishyam's five defect
classes is 0.359, on the split that also chose its epoch, so no held-out number exists. Its CODEBRIM and synthetic
results are pending and not published here, and it is not the live segmenter. After `scripts/install_model.sh`, `make eval-model
DACL10K=<the dacl10k folder> CODEBRIM=data/codebrim` first runs the leakage check between dacl10k and this subset
(`evals/leakage.py`), then both comparisons with `DRISHYAM_ONNX` set. It refuses to compare unless each summary
names the segmenter it should, and publishes the model's numbers next to these (`public/results/codebrim_dacl10k/`),
leaving the classical ones in place.

The leakage check hashes each image with its lighting removed (a 255-bit pHash, in four rotations) to find
candidates, and normalised cross-correlation confirms them. Calibrated on CODEBRIM with `python -m evals.leakage
--codebrim data/codebrim`: resized, rotated, recompressed and re-exposed copies of 50 eval images sit at hash
distance ≤ 10 and correlation ≥ 0.93, and each eval image's nearest other CODEBRIM photo at ≥ 56 and ≤ 0.43, so the
thresholds 32 and 0.8 separate them; crops that drop 5 % of the frame are not found. The 9×8 dHash it replaces put
70 of the 200 eval images within 10 of 64 of an unrelated CODEBRIM photo. It has been run against dacl10k
(2026-10-18); its output is not yet published.

## 9. Failures

Eleven measured failures from the synthetic run in six categories, two per category except formwork lines, which has one (FIG. 5 on the live site; [public/results/failures.json](../public/results/failures.json)):

| Plate | Category | What happened | Rule that would catch it |
| --- | --- | --- | --- |
| ![FIG. 5.1(a)](../public/results/failures/f01.jpg) | Graffiti read as damage | s0162, drishyam@0.3: a spalling finding (confidence 0.97, step 2 at L1) lies within the bounding box of a graffiti patch and matches no ground-truth box of any class. | Chroma gate before drafting: spray paint is far more saturated than concrete or rust, so a candidate whose median saturation sits well above the surrounding concrete and outside the rust hue band is dropped as non-structural. Not implemented. |
| ![FIG. 5.1(b)](../public/results/failures/f02.jpg) | Graffiti read as damage | s0160, drishyam@0.3: a spalling finding (confidence 0.96, step 2 at L1) lies within the bounding box of a graffiti patch and matches no ground-truth box of any class. | Chroma gate before drafting: spray paint is far more saturated than concrete or rust, so a candidate whose median saturation sits well above the surrounding concrete and outside the rust hue band is dropped as non-structural. Not implemented. |
| ![FIG. 5.2(a)](../public/results/failures/f03.jpg) | Formwork line read as crack | s0147, drishyam@0.3: a crack finding (confidence 0.88, step 1 at L2) lies within the bounding box of a formwork line patch and matches no ground-truth box of any class. | Straightness test on the crack mask: a centreline that fits one straight line within about 1.5 px over a length of 20x its width is a formwork seam, not a crack, because real cracks meander (cv2.fitLine residual, minAreaRect aspect). Not implemented. |
| ![FIG. 5.3(a)](../public/results/failures/f04.jpg) | Glare caught by the quality gate | s0040, drishyam@0.3: step 3 (L0) was refused by the quality gate (glare, sharpness 20.9) because the window covers a blown-out highlight. 0 ground-truth box(es) have their centre in that window. The gate did what it should; the cost is an area the work order silently leaves unassessed. | The gate (more than 20% blown-out pixels) works. The missing rule is bookkeeping: a refused window is written to the work order as 'not assessed' with a re-shoot suggestion, so the area never reads as clean. Not implemented. |
| ![FIG. 5.3(b)](../public/results/failures/f05.jpg) | Glare caught by the quality gate | s0082, drishyam@0.3: step 3 (L0) was refused by the quality gate (glare, sharpness 19.2) because the window covers a blown-out highlight. 0 ground-truth box(es) have their centre in that window. The gate did what it should; the cost is an area the work order silently leaves unassessed. | The gate (more than 20% blown-out pixels) works. The missing rule is bookkeeping: a refused window is written to the work order as 'not assessed' with a re-shoot suggestion, so the area never reads as clean. Not implemented. |
| ![FIG. 5.4(a)](../public/results/failures/f06.jpg) | Crack hidden in blur | s0085, drishyam@1.0: a 233x460 px crack inside a blurred patch was not found; it was looked at (step 1, L2, sharpness 56.9), the gate accepted the view, and no crack mask came out of it. | A local sharpness map: where Laplacian variance falls below the gate, report the area as unreadable and ask for a re-shoot; where the gate accepted the view, scale the sharpness threshold to the expected crack width. Not implemented. |
| ![FIG. 5.4(b)](../public/results/failures/f07.jpg) | Crack hidden in blur | s0008, drishyam@1.0: a 361x385 px crack inside a blurred patch was not found; it was looked at (step 1, L2) and the quality gate refused the view (blur, sharpness 10.1). | A local sharpness map: where Laplacian variance falls below the gate, report the area as unreadable and ask for a re-shoot; where the gate accepted the view, scale the sharpness threshold to the expected crack width. Not implemented. |
| ![FIG. 5.5(a)](../public/results/failures/f08.jpg) | Crack missed under a small budget | s0032, drishyam@0.3: a 354x164 px crack box is missed under the 30% cap but found by drishyam@1.0 and tiling@L0; it was looked at (step 1, L2) but no crack mask survived. | Reserve a small share of the pixel cap for a coarse sweep of low-saliency regions, or lower the stop-rule epsilon so the job does not stop while unvisited candidates remain. Both trade pixels for recall. Not implemented. |
| ![FIG. 5.5(b)](../public/results/failures/f09.jpg) | Crack missed under a small budget | s0085, drishyam@0.3: a 267x168 px crack box is missed under the 30% cap but found by drishyam@1.0 and tiling@L0; it was looked at (step 1, L2) but no crack mask survived. | Reserve a small share of the pixel cap for a coarse sweep of low-saliency regions, or lower the stop-rule epsilon so the job does not stop while unvisited candidates remain. Both trade pixels for recall. Not implemented. |
| ![FIG. 5.6(a)](../public/results/failures/f10.jpg) | Rebar, spall and stain confusion | s0051, drishyam@1.0: ground truth is spalling, the job reported exposed_rebar there (confidence 0.93, step 4 at L0) and no spalling finding overlaps the box. | Shape and co-occurrence rule: exposed bars need rust inside a spall rim and a linear shape (minAreaRect aspect above 3); rust with no spall rim is corrosion. Where two classes clear the threshold on one region, send it to human review instead of choosing. The rust-near-spall part exists; the review-on-ambiguity step does not. |
| ![FIG. 5.6(b)](../public/results/failures/f11.jpg) | Rebar, spall and stain confusion | s0171, drishyam@1.0: ground truth is spalling, the job reported exposed_rebar there (confidence 0.93, step 4 at L0) and no spalling finding overlaps the box. | Shape and co-occurrence rule: exposed bars need rust inside a spall rim and a linear shape (minAreaRect aspect above 3); rust with no spall rim is corrosion. Where two classes clear the threshold on one region, send it to human review instead of choosing. The rust-near-spall part exists; the review-on-ambiguity step does not. |

On CODEBRIM, 8 failure cases (cracks missed under the 30 % cap that the full budget finds, and exposed rebar
reported as corrosion or spalling) are published as text only, because the licence bars sharing the photos:
[public/results/codebrim/failures_text.json](../public/results/codebrim/failures_text.json), also under FIG. 5 on
the site.

## 10. Agentic evidence

Every step's action is a logged function of an OpenCV output or a runtime
statistic, with a `cause`. A recorded trace (synthetic sample `bridge_03`, 30 % pixel cap, the published fixed arm64
costs on a virtual clock, so machine load does not change it and a re-run on x86 gives the same file:
[trace-bridge_03.ndjson](trace-bridge_03.ndjson), from `python -m evals.trace_sample`), its first four steps abridged
(the seeds without their entropy):

```json
{"step": 1, "decision": {"action": "gaze", "args": {"x": 1500, "y": 1125, "level": 3}}, "observation": {"entropy": 0.6931, "mask_area_px": 894}, "cause": "survey", "rationale": "one L3 look covers all 4 candidates before zooming; 7 looks affordable", "seeded": [{"region": "r1", "cls": "crack", "from_level": 3}, {"region": "r2", "cls": "crack", "from_level": 3}, {"region": "r3", "cls": "crack", "from_level": 3}, {"region": "r4", "cls": "crack", "from_level": 3}]}
{"step": 2, "decision": {"action": "escalate", "args": {"x": 1374, "y": 1226, "level": 2}}, "observation": {"entropy": 0.3441, "mask_area_px": 10740}, "cause": "entropy>tau", "rationale": "entropy 0.69 > 0.5 at L3", "verify": {"iou": 0.189, "class": "crack", "against_level": 3, "class_now": "crack", "iou_any": 0.189}, "seeded": [{"region": "r1", "cls": "exposed_rebar", "from_level": 2}, {"region": "r2", "cls": "spalling", "from_level": 2}, {"region": "r4", "cls": "exposed_rebar", "from_level": 2}]}
{"step": 3, "decision": {"action": "gaze", "args": {"x": 1840, "y": 1456, "level": 1}}, "observation": {"entropy": 0.4331, "mask_area_px": 28209}, "cause": "best-score", "rationale": "argmax score 4.69e-02: prior 0.69, severity 3 (exposed_rebar seen at L2), uncertainty 0.38, unseen 1.00, t_hat 61 ms; lambda 4.88e-03"}
{"step": 4, "decision": {"action": "verify", "args": {"x": 1597, "y": 1542, "level": 0}}, "observation": {"entropy": 0.3343, "mask_area_px": 57395}, "cause": "verify-new-mask", "rationale": "new spalling mask at L1; confirming at L0", "verify": {"iou": 0.0, "class": "spalling", "against_level": 1, "class_now": "corrosion", "iou_any": 0.003}}
```

The entropy OpenCV measured on the coarse survey (0.69 > τ = 0.5) changed the next tool call from "next candidate"
to "zoom one level finer, centred on the uncertain pixels", and the survey seeded all four candidates with what it
saw there (crack). That L2 look drafted an exposed-rebar finding (`f1`) and a spalling finding (`f2`), both critical,
so both wait in `pending_review`; its crack (5232 px) lay inside the spall and was recorded as spalling only, with a
review request saying so. The step's `verify` compared the L3 crack with the L2 crack (IoU 0.189): it is kept in the
step, but it judges no finding, because the only finding there holds another class. The same look re-seeded r1 and
r4 as exposed rebar, and step 3's argmax went to r1 for it: severity 3 and uncertainty 0.38 from that seed (each
term is in the rationale). Its lambda (4.88e-03) is eps × the value found per ms: the stop floor does not apply
yet, because only looks coarser than the candidates' own first look (L1) had shown them; the L1 and L0 looks that
follow see too little of the other candidates' L1 windows to seed them, and the floor never applies in this job. The L1 look extended `f1` (`steps:
[2, 3]`) and drew a new spalling mask (`f4`), so step 4 verified it one level finer. L0 saw no spalling there (IoU
0) and its best class, corrosion, overlaps the L1 spall by 0.003, so this is a disagreement about where, not a class
change; with no finer level to settle it, `f4` is `verified: false` and goes to a human as "unconfirmed: iou 0 < 0.5
between L1 and L0". At step 6 a second verify confirmed a third spalling finding (`f5`, critical) at IoU 0.691, step
7 drafted a crack, and at step 8 the pixel cap ended the job at 0.2875 of the pixels with r4 not looked at.

Failure drills, each runnable from the UI (`chaos` toggles) and covered by tests. A drill fires on a fixed step (planner timeout on step 3, p95 spike on 4, malformed planner on 6); a job that ends before that step names the drill in its `done` event's `drills_not_reached`, so a drill that tested nothing does not read as passed:

| Drill | Result |
| --- | --- |
| 20 planner timeouts + 20 malformed planner replies | 40 / 40 recovered by the rule planner (`test_20_timeouts_and_20_malformed_all_recover`) |
| Forced p95 spike on the segmenter | the next step's cause is `p95>SLA` and its model is `fast` (`test_forced_p95_spike_changes_next_steps_model`) |
| KILL during a live job | kill sent during step 3 → step 4 halted with cause `kill-switch` (`test_kill_endpoint_halts_a_running_job`, the live e2e run; on the deployed site, check 3 of `scripts/smoke.sh`, sent once step 2 has streamed: halted after 3 steps in the 07:12 smoke run and after 6 in the 07:09 one, section 5) |
| $ and pixel caps | the job halts inside one step with coverage and unvisited regions reported (`test_pixel_cap_is_never_exceeded`, `test_seconds_and_usd_caps_halt`) |
| Agent tries to approve its own critical finding | `ForbiddenTransition` (`test_agent_cannot_decide_or_downgrade`) |

The planners in the first drill are scripted to fail; no LLM runs in the live demo (section 3).

**An LLM driving the tools.** [mcp-session/transcript.md](mcp-session/transcript.md) records Claude, inside Claude
Code, driving the MCP server on `bridge_03.jpg` through a small file bridge that chooses nothing. The measurements
changed its calls: at step 6 a verify measured mask IoU 0.00 between L1 and L0, so it looked at what L0 saw and then
handed the exposed-rebar finding to a human with the disagreement as the reason; at step 18 an IoU of 0.72 confirmed
a spalling finding; at step 20 the remaining pixel budget ended the search. Its prompt states the decision rules
(entropy above 0.5 means zoom, IoU below 0.5 means disagreement; [prompt.txt](mcp-session/prompt.txt)), so the
session shows the tools carrying measured evidence to an LLM more than autonomous planning. The transcript also
notes the large spall it missed.

## 11. Responsible use and limits

- Evidence-first UI: crop, measurements and gaze step before any confidence
  number; an "AI-assisted, not an inspection" banner on every finding.
- Pixels only. No millimetres without a scale reference; no safety ratings. Criticality is a pixel-area threshold
  (spalling ≥ 40,000 px²), so it changes with camera resolution and distance.
- Faces, people and vehicles are blurred (YuNet, NanoDet) before analysis or
  display; a photo mostly filled by a person is refused.
- The planner sees a numeric digest only. Threat model and tests:
  [security.md](security.md).
- The classical engine's probabilities are a softmax over hand-built evidence scores; its confidence and entropy
  are not calibrated.
- The policy comparison holds on synthetic images, but on 200 real CODEBRIM photos the **classical** segmenter caps
  every policy at about 0.2 recall. The dacl10k model is trained but not deployed, and its results here are pending
  (section 8).
- The ablations show that two parts of the policy, the uncertainty term and the governor, change nothing measurable
  on these sets, and that image-driven gaze order beats random order only on the larger photos (section 8). The
  uncertainty term was inert by construction in that run (no unvisited candidate had an entropy); it has not been
  re-measured since candidates are seeded.
- COOL on Graviton was not run (section 6).
- The arm64 timings come from GitHub's `ubuntu-24.04-arm` runner (`results/linux-aarch64/`, Neoverse-N2, one
  thread), a GitHub Actions VM, not the deployed Lambda; on the deployed Lambda build the only numbers are the two smoke runs'
  step p95s, 67.7 and 83.7 ms, each over one job (section 5). The evals ran on the x86 laptop with those arm64 costs.
- The web demo is stateless; the signed `work_order.json` is its audit trail. There is no approval store, and
  human-approval latency is not measured.

## 12. Team and reproduce

- Vaibhav Jangid, lead engineer (agent policy, OpenCV pipeline, evaluation, AWS deployment)
- Bhavya Dhoot, design, demo and presentation (site design, video)

```bash
uv sync --all-extras && make test && make eval && make bench
```

`make eval-large`, `make eval-ablations`, `make eval-clean` and `make eval-live` rerun the other synthetic results;
`make data` then `make eval-real` and `make eval-ablations-real` the CODEBRIM ones (an 8.3 GB download). The published
fixed-cost runs use the arm64 table, the Makefile's default (`MICRO_ARM=linux-aarch64`); `MICRO_ARM=local-macos-x86`
charges the x86 laptop's table instead, and `eval-live` uses no table. `scripts/aws.sh deploy` and `scripts/aws.sh smoke` deploy and check the AWS
stack ([aws.md](aws.md)).

## Appendix: per-cap tables

Synthetic, 2048×1536, 200 images ([summary.json](../public/results/summary.json)):

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall | chance recall (IoU-only) | precision | false boxes / image | hits per matched box | $ / image (modelled) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.25 | 0.50 (0.46–0.54) | 0.43 | 0.02 (0.01) | 0.120 | 12.2 | 1.03 | 0.000017 |
| tiling, L2 grid | 0.06 | 0.44 (0.41–0.48) | 0.40 | 0.02 (0.01) | 0.148 | 8.5 | 1.04 | 0.000005 |
| tiling, L1 grid | 0.50 | 0.55 (0.52–0.59) | 0.48 | 0.02 (0.01) | 0.096 | 17.3 | 1.05 | 0.000034 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.67 | 0.55 (0.51–0.59) | 0.47 | 0.03 (0.01) | 0.021 | 83.4 | 1.08 | 0.000111 |
| **Drishyam**, cap 5 % | 0.02 | **0.00** (no 512² look fits after the overview on a 3 MP image) | 0.00 | 0.00 (0.00) | – | 0.0 | – | 0.000001 |
| **Drishyam**, cap 10 % | 0.10 | **0.30** (0.27–0.34) | 0.27 | 0.02 (0.01) | 0.134 | 6.5 | 1.05 | 0.000006 |
| **Drishyam**, cap 20 % | 0.18 | **0.42** (0.39–0.47) | 0.39 | 0.03 (0.01) | 0.117 | 10.8 | 1.46 | 0.000012 |
| **Drishyam**, cap 30 % | 0.23 | **0.47** (0.43–0.51) | 0.42 | 0.04 (0.02) | 0.102 | 13.9 | 1.49 | 0.000016 |
| **Drishyam**, cap 50 % | 0.31 | **0.51** (0.47–0.55) | 0.45 | 0.03 (0.01) | 0.089 | 17.5 | 1.63 | 0.000021 |
| **Drishyam**, cap 100 % | 0.35 | **0.51** (0.48–0.55) | 0.45 | 0.03 (0.01) | 0.083 | 19.1 | 1.68 | 0.000024 |

| per-class recall | crack | spalling | efflorescence | exposed_rebar | corrosion |
|---|---|---|---|---|---|
| Drishyam @30 % | 0.44 | 0.61 | 0.12 | 0.51 | 0.58 |
| tiling L0 | 0.47 | 0.70 | 0.25 | 0.52 | 0.75 |
| downscale | 0.48 | 0.57 | 0.22 | 0.44 | 0.66 |

Synthetic, 4608×3456, 100 images ([large/summary.json](../public/results/large/summary.json)):

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall | chance recall | false boxes / image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.05 | 0.45 (0.39–0.51) | 0.40 | 0.02 | 16.0 |
| tiling, L2 grid | 0.10 | 0.53 (0.47–0.59) | 0.48 | 0.02 | 70.4 |
| tiling, L1 grid | 0.40 | 0.53 (0.47–0.59) | 0.45 | 0.02 | 529.5 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.63 | 0.32 (0.26–0.38) | 0.20 | 0.01 | 215.5 |
| **Drishyam**, cap 10 % | 0.10 | **0.43** (0.38–0.49) | 0.38 | 0.03 | 54.2 |
| **Drishyam**, cap 30 % | 0.28 | **0.51** (0.45–0.56) | 0.44 | 0.04 | 87.8 |
| **Drishyam**, cap 100 % | 0.46 | **0.51** (0.46–0.57) | 0.45 | 0.04 | 121.5 |

CODEBRIM, 200 real photos ([codebrim/summary.json](../public/results/codebrim/summary.json)):

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall (95 % CI) | chance recall (IoU-only) | boxes per image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.07 | 0.20 (0.17–0.23) | 0.08 (0.06–0.10) | 0.10 (0.02) | 184 |
| tiling, L2 grid | 0.10 | 0.21 (0.19–0.24) | 0.08 (0.06–0.10) | 0.09 (0.02) | 254 |
| tiling, L1 grid | 0.42 | 0.22 (0.19–0.25) | 0.08 (0.06–0.10) | 0.09 (0.02) | 930 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.62 | 0.21 (0.17–0.24) | 0.06 (0.04–0.07) | 0.08 (0.01) | 3,144 |
| **Drishyam**, cap 5 % | 0.04 | **0.10** (0.07–0.14) | 0.04 (0.03–0.05) | 0.07 (0.01) | 95 |
| **Drishyam**, cap 10 % | 0.09 | **0.14** (0.11–0.17) | 0.05 (0.04–0.07) | 0.09 (0.02) | 263 |
| **Drishyam**, cap 20 % | 0.19 | **0.18** (0.15–0.21) | 0.08 (0.06–0.10) | 0.12 (0.03) | 589 |
| **Drishyam**, cap 30 % | 0.29 | **0.20** (0.17–0.23) | 0.09 (0.07–0.11) | 0.12 (0.03) | 891 |
| **Drishyam**, cap 50 % | 0.40 | **0.22** (0.19–0.25) | 0.10 (0.07–0.12) | 0.13 (0.03) | 1,242 |
| **Drishyam**, cap 100 % | 0.42 | **0.22** (0.19–0.25) | 0.10 (0.07–0.12) | 0.14 (0.03) | 1,280 |

| per-class recall | crack | spalling | efflorescence | exposed_rebar | corrosion |
|---|---|---|---|---|---|
| Drishyam @30 % | 0.30 | 0.00 | 0.05 | 0.01 | 0.53 |
| tiling L0 | 0.31 | 0.01 | 0.02 | 0.00 | 0.58 |
| downscale | 0.29 | 0.01 | 0.06 | 0.00 | 0.55 |
