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

Two parts of those stories are not built: approval from an SNS email link (approval is the web review or
`drishyam review`), and the COOL benchmark (section 6).

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
7. **Stop** when the best remaining expected value per ms falls below a quarter of the realised value per ms so far.
   Under a pixel cap the cap usually binds first: at the 30 % cap, 152 of 200 synthetic and 195 of 200 CODEBRIM
   jobs end on the cap; uncapped, the stop rule ends 182 of 200 synthetic jobs and all 200 CODEBRIM jobs.

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
  AppErrors) and X-Ray tracing. The alarms and a 20 USD monthly budget publish to one SNS topic, which has no
  subscriber yet, so today they reach nobody.
- **Smoke test** (`scripts/aws.sh smoke`, `scripts/smoke.sh`): `smoke 2026-10-17T23:12:36Z result=pass`: a job ran
  to `awaiting_review`, KILL halted the second job at step 3 with cause `kill-switch`, a `POST` without the body hash
  and a direct Function URL call both got 403, and the step p95 was 88 ms on aarch64, over the 6 steps of one job.
  That run was against a build of commit d34f74b from a tree with uncommitted changes.
- **Cost.** Spent so far: not measured yet; the month-to-date figure from AWS Billing is still to be added. The
  list-price estimate is about 10 USD a month at 1,000 jobs a month, 8 of it the WAF ([aws.md](aws.md)). The $ shown
  in the UI and the evals is *modelled*: timed op milliseconds × 0.339 USD/h, the c8g.2xlarge rate plus the COOL
  Marketplace fee the PRD planned for. It is not the Lambda bill, and decode and the privacy blur, which run before
  the job starts, are not in it.
- **Operate.** `scripts/aws.sh stop` sets the function's reserved concurrency to 0 (nothing runs, nothing bills);
  `start` undoes it; `destroy` deletes the stack and the site bucket.

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

Only one call is specific to OpenCV 5: the `engine=cv2.dnn.ENGINE_NEW` argument (`segment.py:239`), on the opt-in
ONNX path, which `tests/test_segment.py` exercises with a tiny ONNX model. The live demo runs the classical engine,
so it uses no OpenCV-5-only API; it runs on the stock OpenCV 5.0.0 wheel.

That wheel's aarch64 build already includes Arm's KleidiCV and carotene: `cv2.getBuildInformation()` in the
`opencv-python-headless` 5.0.0.93 package built for Lambda reads "Custom HAL: YES (carotene (ver 0.0.1) KleidiCV
(ver 26.03))" and "Baseline: NEON FP16". The camera moves use ops KleidiCV covers in some formats (`pyrDown`,
`resize`, `warpPerspective`). The stock aarch64 wheel the live demo runs already links KleidiCV 26.03, so a COOL gain
would have to come on top of that. Whether the 3-channel `BORDER_REFLECT_101` `warpPerspective` used here takes the
HAL path has not been measured. There is no `filter2D` and no NumPy pixel loop on the gaze path; the classical
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

**What COOL could buy end to end, from the x86 laptop.** An 8-image macro run on the local x86 laptop
([results/local-macos-x86/macro.json](../results/local-macos-x86/macro.json)) puts 9 % of a job in the stages it
labels KleidiCV-covered imgproc (pyramid, saliency, warp, quality gate), 69 % in the classical segmenter (imgproc
plus NumPy), 3 % in crack thinning, 8 % in decode and 12 % in Python. So if COOL made the covered ops 1.8× faster, a
whole job would get about 1.04× faster. The gaze path itself is cheap; most of the time is in segmentation, which is
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
   so far.
3. Cross-scale verification: a new confident mask is looked at again one pyramid level finer and compared by mask
   IoU through the two views' homographies; if the IoU stays below 0.5, the finding goes to human review with the
   measured IoU as the reason.
4. A governor that checks kill, caps and p95 latency before every step and writes a cause for each override, so
   every step in the trace says why it happened.
5. A human gate: critical findings wait in `pending_review`, and no agent, MCP or planner call can approve them.

**What is not new:** coarse-to-fine pyramids, saliency priors, steering a high-resolution window from a
low-resolution map, and tiling. On these images a single downscale and a coarse tiling grid match or beat
Drishyam's recall at equal or lower pixel cost (section 8), so the combination has not yet paid off in recall.

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

Measured on **200 held-out labelled synthetic images** (seed 7; policy changes were tuned only on a separate 60-image dev split, seed 4242, `make eval-dev`), one segmenter and one matching rule for every policy (a ground-truth box is found by a same-class component with IoU ≥ 0.3 or containing its centre; "IoU-only" drops the centre clause). The 95 % intervals are a bootstrap over images, paired across policies. "Chance" moves each policy's own boxes (same class and size) to random places in the photo and scores them again. Precision is one-to-one: in confidence order each prediction claims at most one ground-truth box, so a defect reported again from a second look is a false box; "hits per matched box" counts the predictions that land on each box found (1 = no duplicates). All of it runs on the **fixed cost model** (`--cost-model fixed`): every op the agent times costs its p50 from the `job_ops` table in `results/local-macos-x86/micro.json` on every call, so t̂, the stop rule and the governor see the same costs on a busy machine as on an idle one, and a re-run reproduces the numbers exactly. $ is that modelled time × $0.339/h (untimed Python is not charged).

The FR-21 ablations (random gaze order, no uncertainty term, no survey look, no verify step, no governor; the
switches exist in `JobConfig`) have not been run, so nothing here yet shows that the image-driven choices beat the
same budget spent without them. The per-cap tables and per-class recall are in the
[appendix](#appendix-per-cap-tables).

### Synthetic, 2048×1536 (200 images)

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall | chance recall (IoU-only) | false boxes / image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.25 | 0.50 (0.46–0.54) | 0.43 | 0.02 (0.01) | 12.2 |
| tiling, L2 grid | 0.06 | 0.44 (0.41–0.48) | 0.40 | 0.02 (0.01) | 8.5 |
| tiling, L1 grid | 0.50 | 0.55 (0.52–0.59) | 0.48 | 0.02 (0.01) | 17.3 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.67 | 0.55 (0.51–0.59) | 0.47 | 0.03 (0.01) | 83.4 |
| **Drishyam**, cap 30 % | 0.25 | **0.47** (0.44–0.51) | 0.43 | 0.02 (0.01) | 14.5 |
| **Drishyam**, uncapped | 0.46 | **0.53** (0.50–0.57) | 0.47 | 0.05 (0.02) | 24.4 |

- At a **30 % cap** Drishyam finds **0.47** of the defects using **0.25** of the image's pixels, against 0.55 for
  exhaustive 512-px tiling with 20 % overlap, which reads 1.67× the image. That is **0.86×** tiling's recall (95 % CI
  0.80–0.93) at **0.15×** its pixels. **Verdict on the PRD hypothesis (≥ 0.9× tiling recall at ≤ 0.3× its pixels):
  not met**, though the interval reaches 0.93, so 0.9 is not ruled out either.
- **One downscale does as well**: against a single 1024-px downscale at the same pixels (0.50 at 0.25), Drishyam is
  0.96× (95 % CI 0.90–1.02).
- Against the other grids it is 0.86× L1 tiling (0.55 at 0.50 of the pixels, the best grid) and 1.07× L2 tiling
  (0.44 at 0.06). When a match needs IoU ≥ 0.3 (no contains-the-centre clause), it is 0.92× L0 tiling (95 % CI
  0.85–0.99).
- How predictions are boxed changes the ratio. Every number here gives each policy one box per connected component.
  With one box per class per view for every policy, the 30 % ratio to L0 tiling is 0.75×; with Drishyam's finding
  boxes against the baselines' components (the as-specified mix, which favours Drishyam) it is 1.23×
  (`agent_metrics.box_convention` in the summary).
- At a **50 % cap**: 0.52 at 0.35 of the pixels, 0.95× tiling's recall at 0.21× its pixels. Uncapped, the stop rule
  ends 182 of 200 jobs by itself, at 0.46 of the pixels with 0.97× tiling's recall.
- **Above chance**: the same boxes at random places score 0.00 to 0.05 for every policy.
- **False boxes**: precision 0.099 for Drishyam at 30 % (14.5 false boxes per image) against 0.120 for one
  downscale (12.2) and 0.021 for exhaustive tiling (83.4). Drishyam reports a defect again from each look in its
  evidence trail (1.49 predictions per matched box, against 1.03 for one downscale).
- **Hairline cracks** (87 of the 227 crack boxes are 1–2 px wide; the flag is in the ground truth): exhaustive tiling
  finds 0.25 of them, one downscale 0.10, Drishyam 0.09 at the 30 % cap and 0.17 uncapped. Only 16 boxes are under
  96² px, too few to read anything into.
- The hysteresis segmenter shipped with this policy lifted every baseline too, so it is not a Drishyam-only gain.
- Critical false-accept rate (critical prediction with no critical ground truth): Drishyam at 30 % 0.52 against
  exhaustive tiling 0.79 (the PRD asks for ≤ tiling); a single downscale (0.49) and L2 tiling (0.42) are lower still.
- Agent: 100 % of steps carry a cause; escalation rate 17 %; at the 30 % cap, 3 looks and 0.27 of the pixels per
  image (p50 = p95).

The images are procedural (`evals/synth.py`): 2048×1536 weathered concrete with drawn cracks, spalls, efflorescence, exposed rebar and rust run-off, plus graffiti, formwork lines, water stains, blur and glare as distractors. The eight bundled samples in `public/samples/` come from the same generator.

![Recall against pixels processed](recall_vs_pixels.png)

(The live site's FIG. 4.1, rendered from the committed `public/results/summary.json`: bars are the 95 % intervals, dotted ticks the chance level.)

Full numbers: [public/results/summary.json](../public/results/summary.json); every prediction and trace: [public/results/per_image.json.gz](../public/results/per_image.json.gz).

**Fixed costs against measured wall time.** `make eval-live` reruns the synthetic set the way the app runs, on measured wall time ([public/results/live/summary.json](../public/results/live/summary.json)), here with the machine's load average near 39 from other jobs: Drishyam at 30 % found 0.48 (0.45–0.52) at 0.25 of the pixels, 0.88× tiling (0.82–0.94), and the governor never switched the model but shrank the gaze and batched on 6 warp spikes. The stop rule ended fewer jobs at the 30 % cap (29 of 200, against 47 with fixed costs), because a measured step includes the Python around each look while the fixed model charges only the timed ops. Live numbers move with the machine; the fixed ones are the comparison. On the 60-image dev split the policy was tuned on (`make eval-dev`), the 30 % ratio to tiling is 0.96× (0.86–1.08), against 0.86× on the held-out images.

### Synthetic at CODEBRIM's size, 4608×3456 (100 images)

`make eval-large` builds 100 images with the same generator (seed 7) at 4608×3456, the size of most CODEBRIM photos, and runs the same harness, to test the claim that zooming should beat one downscale as photos grow. Published as it came out: [public/results/large/summary.json](../public/results/large/summary.json).

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall | chance recall | false boxes / image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.05 | 0.45 (0.39–0.51) | 0.40 | 0.02 | 16.0 |
| tiling, L2 grid | 0.10 | 0.53 (0.47–0.59) | 0.48 | 0.02 | 70.4 |
| tiling, L1 grid | 0.40 | 0.53 (0.47–0.59) | 0.45 | 0.02 | 529.5 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.63 | 0.32 (0.26–0.38) | 0.20 | 0.01 | 215.5 |
| **Drishyam**, cap 30 % | 0.29 | **0.50** (0.45–0.55) | 0.43 | 0.03 | 95.0 |
| **Drishyam**, uncapped | 0.54 | **0.51** (0.46–0.57) | 0.45 | 0.03 | 141.9 |

- Against one downscale, zooming now wins, narrowly: Drishyam at 30 % finds 0.50 against 0.45, 1.10× (95 % CI 1.01–1.20), where on the 2048×1536 set it was 0.96× (0.90–1.02). The downscale shrinks these images 4.5× instead of 2×.
- A coarse tiling grid still does better: L2 tiling finds 0.53 at 0.10 of the pixels, a third of Drishyam's. Exhaustive L0 tiling falls to 0.32 (spalling 0.16, exposed rebar 0.00): the generator scales spalls with the image, so here they are about 225 to 860 px across and a 512-px full-resolution tile often holds only part of one. The headline ratio against L0 tiling (1.58×, which formally "meets" the 0.9× target) says more about L0 tiling than about Drishyam.
- Hairline cracks (39 boxes) stay nearly invisible to every policy: 0.15 for L0 tiling, 0.10 for L1, 0.03 to 0.05 for Drishyam, 0.03 for one downscale. Uncapped, the stop rule ends 97 of 100 jobs by itself.

### Real bridge photos: CODEBRIM (200 images)

The same harness, segmenter and matching rule on **200 real inspection photos** from CODEBRIM: a seeded subset (seed 7) of the 1,052 annotated originals, mostly 4608×3456 px (2 to 24 MP), holding 1,008 defect boxes with 1,685 class labels (a box showing several defects is scored once per class) and at least 175 labels in every class. `make data` downloads and checks the 8.3 GB archive and `make eval-real` runs it; the 200 file names are committed in `evals/codebrim_eval200.txt`. The dataset's licence is non-commercial and bars sharing the photos, so only the numbers and a text description of each failure are published (the failure crops stay in the git-ignored `results/codebrim/`). Two matching rules are reported side by side, because on these photos they disagree: the lenient one (IoU ≥ 0.3 or containing the centre) and IoU-only. Same fixed cost model as above.

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall (95 % CI) | chance recall (IoU-only) | boxes per image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.07 | 0.20 (0.17–0.23) | 0.08 (0.06–0.10) | 0.10 (0.02) | 184 |
| tiling, L2 grid | 0.10 | 0.21 (0.19–0.24) | 0.08 (0.06–0.10) | 0.09 (0.02) | 254 |
| tiling, L1 grid | 0.42 | 0.22 (0.19–0.25) | 0.08 (0.06–0.10) | 0.09 (0.02) | 930 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.62 | 0.21 (0.17–0.24) | 0.06 (0.04–0.07) | 0.08 (0.01) | 3,144 |
| **Drishyam**, cap 30 % | 0.29 | **0.20** (0.17–0.23) | 0.09 (0.07–0.11) | 0.14 (0.02) | 920 |
| **Drishyam**, uncapped | 0.44 | **0.22** (0.19–0.26) | 0.10 (0.08–0.12) | 0.14 (0.03) | 1,363 |

- **The PRD's hypothesis test (G1, defined on these photos) passes on paper and means little.** At a 30 % cap Drishyam reaches **0.97×** exhaustive tiling's recall (95 % CI 0.88–1.08, so not clearly above 0.9) at **0.18×** its pixels (1.54×, 1.26–1.92, when a match needs IoU ≥ 0.3), and 1.01× a single 1024-px downscale (0.93–1.10) that uses a quarter of its pixels. Every baseline, and Drishyam from the 20 % cap up, finds only about one labelled defect in five (0.19 to 0.22).
- **Much of that lenient score is chance.** Each policy's own boxes, moved to random places, already score 0.07 to 0.14 under the lenient rule, because thousands of boxes per image (920 for Drishyam at 30 %, 3,144 for L0 tiling) contain many ground-truth centres by luck. Under IoU-only, recall is 0.04 to 0.10 and chance 0.01 to 0.03: above chance, but low for every policy.
- **On real photos the segmenter, not where the agent looks, limits recall.** The classical engine was tuned on synthetic images. Precision (one-to-one) is 0.0005 to 0.008, and 98 to 100 % of every policy's critical predictions match no critical ground truth. It finds cracks (0.31) and corrosion (0.53) and effectively misses spalling (0.00) and exposed rebar (0.01).
- **The agent's control carries over.** 100 % of steps carry a cause; 40 of 40 planner drills and 10 of 10 governor drills recovered. With fixed costs the governor never had to step in (an earlier run on measured wall time with 8 workers saw 77 SLA breaches and 14 warp spikes). At the 30 % cap a job takes 17 looks (p50; 26 at p95), against 3 on the smaller synthetic images; uncapped, the stop rule ends all 200 jobs, at 0.44 of the pixels.

Full numbers: [public/results/codebrim/summary.json](../public/results/codebrim/summary.json), also on the live site (FIG. 4.2). CODEBRIM: Mundt, Majumder, Murali, Panetsos and Ramesh, "Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset", CVPR 2019.

### The trained segmenter (not yet evaluated)

These numbers call for a trained segmenter. A dacl10k-trained UNet ([training/](../training/README.md),
[models/README.md](../models/README.md)) exists: its validation mIoU over Drishyam's five defect classes is 0.359,
on the split that also chose its epoch, so no held-out number exists. It has not been evaluated on CODEBRIM or the
synthetic set here, and it is not the live segmenter. After `scripts/install_model.sh`, `make eval-model
DACL10K=<the dacl10k folder> CODEBRIM=data/codebrim` first runs the leakage check between dacl10k and this subset
(`evals/leakage.py`), then both comparisons with `DRISHYAM_ONNX` set. It refuses to compare unless each summary
names the segmenter it should, and publishes the model's numbers next to these (`public/results/codebrim_dacl10k/`),
leaving the classical ones in place.

The leakage check hashes each image with its lighting removed (a 255-bit pHash, in four rotations) to find
candidates, and normalised cross-correlation confirms them. Calibrated on CODEBRIM with `python -m evals.leakage
--codebrim data/codebrim`: resized, rotated, recompressed and re-exposed copies of 50 eval images sit at hash
distance ≤ 10 and correlation ≥ 0.93, and each eval image's nearest other CODEBRIM photo at ≥ 56 and ≤ 0.43, so the
thresholds 32 and 0.8 separate them; crops that drop 5 % of the frame are not found. The 9×8 dHash it replaces put
70 of the 200 eval images within 10 of 64 of an unrelated CODEBRIM photo. It has not yet been run against dacl10k.

## 9. Failures

Twelve measured failures from the synthetic run, two per category (FIG. 5 on the live site; [public/results/failures.json](../public/results/failures.json)):

| Plate | Category | What happened | Rule that would catch it |
| --- | --- | --- | --- |
| ![FIG. 5.1(a)](../public/results/failures/f01.jpg) | Graffiti read as damage | s0162, drishyam@0.3: a spalling finding (confidence 0.97, step 2 at L1) lies within the bounding box of a graffiti patch and matches no ground-truth box of any class. | Chroma gate before drafting: spray paint is far more saturated than concrete or rust, so a candidate whose median saturation sits well above the surrounding concrete and outside the rust hue band is dropped as non-structural. Not implemented. |
| ![FIG. 5.1(b)](../public/results/failures/f02.jpg) | Graffiti read as damage | s0160, drishyam@0.3: a spalling finding (confidence 0.96, step 2 at L1) lies within the bounding box of a graffiti patch and matches no ground-truth box of any class. | Chroma gate before drafting: spray paint is far more saturated than concrete or rust, so a candidate whose median saturation sits well above the surrounding concrete and outside the rust hue band is dropped as non-structural. Not implemented. |
| ![FIG. 5.2(a)](../public/results/failures/f03.jpg) | Formwork line read as crack | s0041, drishyam@1.0: a crack finding (confidence 0.94, step 4 at L0) lies within the bounding box of a formwork line patch and matches no ground-truth box of any class. | Straightness test on the crack mask: a centreline that fits one straight line within about 1.5 px over a length of 20x its width is a formwork seam, not a crack, because real cracks meander (cv2.fitLine residual, minAreaRect aspect). Not implemented. |
| ![FIG. 5.2(b)](../public/results/failures/f04.jpg) | Formwork line read as crack | s0189, drishyam@1.0: a crack finding (confidence 0.93, step 4 at L0) lies within the bounding box of a formwork line patch and matches no ground-truth box of any class. | Straightness test on the crack mask: a centreline that fits one straight line within about 1.5 px over a length of 20x its width is a formwork seam, not a crack, because real cracks meander (cv2.fitLine residual, minAreaRect aspect). Not implemented. |
| ![FIG. 5.3(a)](../public/results/failures/f05.jpg) | Glare caught by the quality gate | s0040, drishyam@0.3: step 3 (L0) was refused by the quality gate (glare, sharpness 20.9) because the window covers a blown-out highlight. 0 ground-truth box(es) have their centre in that window. The gate did what it should; the cost is an area the work order silently leaves unassessed. | The gate (more than 20% blown-out pixels) works. The missing rule is bookkeeping: a refused window is written to the work order as 'not assessed' with a re-shoot suggestion, so the area never reads as clean. Not implemented. |
| ![FIG. 5.3(b)](../public/results/failures/f06.jpg) | Glare caught by the quality gate | s0082, drishyam@0.3: step 3 (L0) was refused by the quality gate (glare, sharpness 19.2) because the window covers a blown-out highlight. 0 ground-truth box(es) have their centre in that window. The gate did what it should; the cost is an area the work order silently leaves unassessed. | The gate (more than 20% blown-out pixels) works. The missing rule is bookkeeping: a refused window is written to the work order as 'not assessed' with a re-shoot suggestion, so the area never reads as clean. Not implemented. |
| ![FIG. 5.4(a)](../public/results/failures/f07.jpg) | Crack hidden in blur | s0008, drishyam@1.0: a 361x385 px crack inside a blurred patch was not found; it was looked at (step 1, L2) and the quality gate refused the view (blur, sharpness 10.1). | A local sharpness map: where Laplacian variance falls below the gate, report the area as unreadable and ask for a re-shoot; where the gate accepted the view, scale the sharpness threshold to the expected crack width. Not implemented. |
| ![FIG. 5.4(b)](../public/results/failures/f08.jpg) | Crack hidden in blur | s0004, drishyam@1.0: a 283x366 px crack inside a blurred patch was not found; it was looked at (step 1, L2, sharpness 43.2), the gate accepted the view, and no crack mask came out of it. | A local sharpness map: where Laplacian variance falls below the gate, report the area as unreadable and ask for a re-shoot; where the gate accepted the view, scale the sharpness threshold to the expected crack width. Not implemented. |
| ![FIG. 5.5(a)](../public/results/failures/f09.jpg) | Crack missed under a small budget | s0094, drishyam@0.3: a 44x96 px crack box is missed under the 30% cap but found by drishyam@1.0 and tiling@L0; it was looked at (step 1, L2) but no crack mask survived. | Reserve a small share of the pixel cap for a coarse sweep of low-saliency regions, or lower the stop-rule epsilon so the job does not stop while unvisited candidates remain. Both trade pixels for recall. Not implemented. |
| ![FIG. 5.5(b)](../public/results/failures/f10.jpg) | Crack missed under a small budget | s0032, drishyam@0.3: a 354x164 px crack box is missed under the 30% cap but found by drishyam@1.0 and tiling@L0; it was looked at (step 1, L2) but no crack mask survived. | Reserve a small share of the pixel cap for a coarse sweep of low-saliency regions, or lower the stop-rule epsilon so the job does not stop while unvisited candidates remain. Both trade pixels for recall. Not implemented. |
| ![FIG. 5.6(a)](../public/results/failures/f11.jpg) | Rebar, spall and stain confusion | s0171, drishyam@1.0: ground truth is spalling, the job reported exposed_rebar there (confidence 0.93, step 4 at L0) and no spalling finding overlaps the box. | Shape and co-occurrence rule: exposed bars need rust inside a spall rim and a linear shape (minAreaRect aspect above 3); rust with no spall rim is corrosion. Where two classes clear the threshold on one region, send it to human review instead of choosing. The rust-near-spall part exists; the review-on-ambiguity step does not. |
| ![FIG. 5.6(b)](../public/results/failures/f12.jpg) | Rebar, spall and stain confusion | s0103, drishyam@0.3: ground truth is spalling, the job reported corrosion there (confidence 0.92, step 3 at L1) and no spalling finding overlaps the box. | Shape and co-occurrence rule: exposed bars need rust inside a spall rim and a linear shape (minAreaRect aspect above 3); rust with no spall rim is corrosion. Where two classes clear the threshold on one region, send it to human review instead of choosing. The rust-near-spall part exists; the review-on-ambiguity step does not. |

On CODEBRIM, 8 failure cases (cracks missed under the 30 % cap that the full budget finds, and exposed rebar or
spalling reported as corrosion) are published as text only, because the licence bars sharing the photos:
[public/results/codebrim/failures_text.json](../public/results/codebrim/failures_text.json), also under FIG. 5 on
the site.

## 10. Agentic evidence

Every step's action is a logged function of an OpenCV output or a runtime
statistic, with a `cause`. A recorded trace (synthetic sample `bridge_03`, 30 % pixel cap, fixed clock):

```json
{"step": 1, "decision": {"action": "gaze", "args": {"x": 1500, "y": 1125, "level": 3}}, "observation": {"entropy": 0.69, "mask_area_px": 894}, "cause": "survey", "rationale": "one L3 look covers all 3 candidates before zooming; 7 looks affordable"}
{"step": 2, "decision": {"action": "escalate", "args": {"x": 1374, "y": 1226, "level": 2}}, "observation": {"entropy": 0.34, "mask_area_px": 10740}, "cause": "entropy>tau", "rationale": "entropy 0.69 > 0.5 at L3"}
{"step": 4, "decision": {"action": "verify", "args": {"x": 1597, "y": 1542, "level": 0}}, "observation": {"entropy": 0.33, "mask_area_px": 57395}, "cause": "verify-new-mask", "rationale": "new spalling mask at L1; confirming at L0"}
```

The entropy OpenCV measured on the coarse survey (0.69 > τ = 0.5) changed the next
tool call from "next candidate" to "zoom one level finer, centred on the uncertain
pixels"; a new confident spalling mask then triggered a verification one level finer.
The resulting exposed-rebar finding carries its evidence trail (`steps: [2, 3]`) and
waits in `pending_review` for a human.

Failure drills, each runnable from the UI (`chaos` toggles) and covered by tests:

| Drill | Result |
| --- | --- |
| 20 planner timeouts + 20 malformed planner replies | 40 / 40 recovered by the rule planner (`test_20_timeouts_and_20_malformed_all_recover`) |
| Forced p95 spike on the segmenter | the next step's cause is `p95>SLA` and its model is `fast` (`test_forced_p95_spike_changes_next_steps_model`) |
| KILL during a live job | kill sent during step 3 → step 4 halted with cause `kill-switch` (`test_kill_endpoint_halts_a_running_job`, the live e2e run; on the deployed site, check 3 of `scripts/smoke.sh`) |
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
  every policy at about 0.2 recall. The dacl10k model is trained but not yet evaluated here or deployed (section 8).
- The FR-21 ablations have not been run (section 8).
- COOL on Graviton was not run (section 6).
- Every published timing table comes from one x86 laptop (`results/local-macos-x86/`); the only arm64 number is the
  smoke test's step p95 over one job (section 5).
- The web demo is stateless; the signed `work_order.json` is its audit trail. There is no approval store, and
  human-approval latency is not measured.

## 12. Team and reproduce

- Vaibhav Jangid, lead engineer (agent policy, OpenCV pipeline, evaluation, AWS deployment)
- Bhavya Dhoot, design, demo and presentation (site design, video)

```bash
uv sync --all-extras && make test && make eval && make bench
```

`make eval-large` and `make eval-live` rerun the other synthetic results; `make data` then `make eval-real` the
CODEBRIM one (an 8.3 GB download). `scripts/aws.sh deploy` and `scripts/aws.sh smoke` deploy and check the AWS
stack ([aws.md](aws.md)).

## Appendix: per-cap tables

Synthetic, 2048×1536, 200 images ([summary.json](../public/results/summary.json)):

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall | chance recall (IoU-only) | precision | false boxes / image | hits per matched box | $ / image (modelled) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.25 | 0.50 (0.46–0.54) | 0.43 | 0.02 (0.01) | 0.120 | 12.2 | 1.03 | 0.000030 |
| tiling, L2 grid | 0.06 | 0.44 (0.41–0.48) | 0.40 | 0.02 (0.01) | 0.148 | 8.5 | 1.04 | 0.000008 |
| tiling, L1 grid | 0.50 | 0.55 (0.52–0.59) | 0.48 | 0.02 (0.01) | 0.096 | 17.3 | 1.05 | 0.000060 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.67 | 0.55 (0.51–0.59) | 0.47 | 0.03 (0.01) | 0.021 | 83.4 | 1.08 | 0.000196 |
| **Drishyam**, cap 5 % | 0.02 | **0.00** (no 512² look fits after the overview on a 3 MP image) | 0.00 | 0.00 (0.00) | – | 0.0 | – | 0.000002 |
| **Drishyam**, cap 10 % | 0.10 | **0.30** (0.27–0.34) | 0.27 | 0.01 (0.00) | 0.125 | 7.0 | 1.05 | 0.000012 |
| **Drishyam**, cap 20 % | 0.18 | **0.43** (0.39–0.47) | 0.39 | 0.02 (0.01) | 0.114 | 11.2 | 1.45 | 0.000022 |
| **Drishyam**, cap 30 % | 0.25 | **0.47** (0.44–0.51) | 0.43 | 0.02 (0.01) | 0.099 | 14.5 | 1.49 | 0.000030 |
| **Drishyam**, cap 50 % | 0.35 | **0.52** (0.49–0.56) | 0.46 | 0.04 (0.02) | 0.083 | 19.4 | 1.64 | 0.000043 |
| **Drishyam**, cap 100 % | 0.46 | **0.53** (0.50–0.57) | 0.47 | 0.05 (0.02) | 0.069 | 24.4 | 1.79 | 0.000056 |

| per-class recall | crack | spalling | efflorescence | exposed_rebar | corrosion |
|---|---|---|---|---|---|
| Drishyam @30 % | 0.44 | 0.62 | 0.15 | 0.51 | 0.57 |
| tiling L0 | 0.47 | 0.70 | 0.25 | 0.52 | 0.75 |
| downscale | 0.48 | 0.57 | 0.22 | 0.44 | 0.66 |

Synthetic, 4608×3456, 100 images ([large/summary.json](../public/results/large/summary.json)):

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall | chance recall | false boxes / image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.05 | 0.45 (0.39–0.51) | 0.40 | 0.02 | 16.0 |
| tiling, L2 grid | 0.10 | 0.53 (0.47–0.59) | 0.48 | 0.02 | 70.4 |
| tiling, L1 grid | 0.40 | 0.53 (0.47–0.59) | 0.45 | 0.02 | 529.5 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.63 | 0.32 (0.26–0.38) | 0.20 | 0.01 | 215.5 |
| **Drishyam**, cap 10 % | 0.10 | **0.43** (0.38–0.50) | 0.38 | 0.03 | 63.9 |
| **Drishyam**, cap 30 % | 0.29 | **0.50** (0.45–0.55) | 0.43 | 0.03 | 95.0 |
| **Drishyam**, cap 100 % | 0.54 | **0.51** (0.46–0.57) | 0.45 | 0.03 | 141.9 |

CODEBRIM, 200 real photos ([codebrim/summary.json](../public/results/codebrim/summary.json)):

| Policy | pixels (× image) | recall (95 % CI) | IoU-only recall (95 % CI) | chance recall (IoU-only) | boxes per image |
| --- | --- | --- | --- | --- | --- |
| downscale once (1024 px) | 0.07 | 0.20 (0.17–0.23) | 0.08 (0.06–0.10) | 0.10 (0.02) | 184 |
| tiling, L2 grid | 0.10 | 0.21 (0.19–0.24) | 0.08 (0.06–0.10) | 0.09 (0.02) | 254 |
| tiling, L1 grid | 0.42 | 0.22 (0.19–0.25) | 0.08 (0.06–0.10) | 0.09 (0.02) | 930 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.62 | 0.21 (0.17–0.24) | 0.06 (0.04–0.07) | 0.08 (0.01) | 3,144 |
| **Drishyam**, cap 5 % | 0.04 | **0.11** (0.08–0.14) | 0.04 (0.03–0.05) | 0.07 (0.01) | 95 |
| **Drishyam**, cap 10 % | 0.09 | **0.13** (0.11–0.17) | 0.05 (0.04–0.07) | 0.09 (0.01) | 265 |
| **Drishyam**, cap 20 % | 0.19 | **0.19** (0.16–0.22) | 0.08 (0.06–0.10) | 0.12 (0.03) | 601 |
| **Drishyam**, cap 30 % | 0.29 | **0.20** (0.17–0.23) | 0.09 (0.07–0.11) | 0.14 (0.02) | 920 |
| **Drishyam**, cap 50 % | 0.42 | **0.22** (0.19–0.26) | 0.10 (0.08–0.12) | 0.14 (0.03) | 1,324 |
| **Drishyam**, cap 100 % | 0.44 | **0.22** (0.19–0.26) | 0.10 (0.08–0.12) | 0.14 (0.03) | 1,363 |

| per-class recall | crack | spalling | efflorescence | exposed_rebar | corrosion |
|---|---|---|---|---|---|
| Drishyam @30 % | 0.31 | 0.00 | 0.05 | 0.01 | 0.53 |
| tiling L0 | 0.31 | 0.01 | 0.02 | 0.00 | 0.58 |
| downscale | 0.29 | 0.01 | 0.06 | 0.00 | 0.55 |
