# DRISHYAM — Product Requirements Document

Oct 17, 2026 · @Vaibhav Jangid

## Summary

**Status 2026-10-18:** live demo on AWS Lambda arm64 behind CloudFront, no Hugging Face Space; COOL not run; the stock aarch64 wheel already links KleidiCV 26.03.

DRISHYAM is a virtual-PTZ inspector for bridge photos: it decides where to look, zooms until it is sure, drafts work orders a human signs, and stops looking when the next look is worth less than it costs; under a pixel cap the cap binds first (125 of 200 synthetic and 183 of 200 CODEBRIM jobs at the 30 % cap; uncapped, the stop rule ends 183 of 200 synthetic jobs). Every camera move is an OpenCV `pyrDown`/`warpPerspective`/`resize` call. The live demo runs the stock opencv-python-headless 5.0.0 wheel on Lambda arm64, whose build already includes KleidiCV 26.03 and carotene (`cv2.getBuildInformation()`: "Custom HAL: YES (carotene (ver 0.0.1) KleidiCV (ver 26.03))"). One codebase targets the Overall, Agentic Vision and Best Use of COOL awards; submission is due Oct 26, 2026 at 11:59 pm PT.

- **Headline metric:** defect recall on CODEBRIM boxes vs. fraction of full-resolution pixels processed (and $ per image), against single-downscale and exhaustive tiling. Target hypothesis: ≥0.9× tiling recall at ≤0.3× pixels; the report states the measured number, whatever it is.
- **Free build:** $0 out of pocket. Software is open source, training runs on Kaggle, the always-on demo runs on a free Hugging Face Space, and the only paid items (COOL on Graviton plus two benchmark arms, ≈$70–100) are paid from AWS Free Tier credits or the competition compute grant, behind hard spend caps.
- **Reuse first:** we write only the gaze policy, the budget governor and glue. Segmentation, tiling, agent loop, MCP, UI, HTTPS, metrics, benchmarking and IaC all come from existing projects (section 7).
- **Scope cut for 9 days:** no ALB, API Gateway, CloudFront or Fargate; a deterministic policy core with the LLM planner optional; a 200-image evaluation subset; the full dataset sweep only if time allows.

## Problem and opportunity

Bridge owners cannot keep up with inspections: the dacl10k authors write that authorities "often fail to keep up with the necessary inspection and restoration intervals due to staff shortages and budget limitations." Photo backlogs grow, and the defects that matter most (hairline cracks, early exposed rebar) are invisible at thumbnail scale, so a tool must either tile every photo at full resolution or guess where to look.

DRISHYAM's bet is that looking is cheap when it is a geometric warp, and that where to look next can be read from what OpenCV just measured: saliency, segmentation uncertainty, mask area and the live per-op latency on the box. That turns a compute problem into a budgeting problem, which is the thing a cloud customer pays for.

The competition field (4,963 participants) clusters on "detect, then an LLM explains" and recapture loops on synthetic parts, steel, PPE and debris. Nobody visible does budgeted active perception over real high-resolution imagery, makes the agent's action the COOL workload, re-plans from runtime telemetry, or reports a falsifiable recall-vs-pixels curve. Those four gaps are the product.

## Goals, success criteria and non-goals

Ship a working, measured, free-to-run agent by Oct 26; each goal below has a number the report must state.

| ID | Goal | Success criterion (measured by Oct 25) |
| --- | --- | --- |
| G1 | Find defects under a pixel budget | Recall-vs-pixels curve on 200 CODEBRIM originals for downscale-once, exhaustive tiling and DRISHYAM. Hypothesis: ≥0.9× tiling recall at ≤0.3× pixels; the measured value is reported either way |
| G2 | Prove the agent is agentic | Every step's next action is a logged function of an OpenCV output or a runtime stat; ≥1 trace per demo image shows an OpenCV value changing the next tool call; 20 injected planner timeouts, 20 rule-planner recoveries |
| G3 | Verified COOL on Graviton | Core gaze and saliency loop runs from `/opt/cool` on a c8g.2xlarge; three-arm benchmark (COOL/c8g, stock/c8g, stock/c7i) with provenance and a mask-parity check |
| G4 | Human control | 0 critical findings closed or downgraded without approval; kill switch and $ cap each halt a live job in the demo |
| G5 | Free to build and run | $0 out of pocket; AWS spend ≤$100 and fully covered by credits or the compute grant; no paid API key required to run the agent |
| G6 | Complete submission | Report, repo, pinned deps, diagrams, live endpoint, ≤5-min video, evaluation with failure cases, COOL and agentic extras, submitted by noon PT Oct 26 |

**Non-goals**

- State-of-the-art segmentation mIoU; a MobileNet-class model is enough to drive the policy
- Millimetre measurements or structural-safety ratings; outputs are in pixels unless a scale reference exists
- GPU inference, video streams, mobile apps, multi-tenant auth
- A VLM on the critical path (PaliGemma captioning stays an async experiment); Qwen2.5-VL inside OpenCV DNN is unverified and out
- ALB, API Gateway, CloudFront, Fargate, NAT gateways: none are needed and all cost money
- Any new OpenCV C++/HAL code; we compose built-in ops only
- Watching people, vehicles or traffic

## Users and user stories

Three audiences use DRISHYAM in the next three weeks: a bridge inspection engineer, the competition judges, and the solo operator who runs and pays for it.

| As a… | I want… | So that… | Accepted when… |
| --- | --- | --- | --- |
| Inspection engineer | to upload a photo and get findings with crops, pixel measurements and a numbered gaze path | I can review evidence first and decide, not trust a score | Findings appear with crop, class, confidence, px length/width and the step that found them |
| Inspection engineer | critical findings (exposed rebar, large spalling) to wait for my approval | the agent cannot close or downgrade them | Approve/reject buttons in the UI and an SNS email link; agent state blocked until a decision |
| Inspection engineer | a per-job budget in pixels, dollars and seconds | backlog triage costs what I decide | `set_budget` is honoured; job halts with partial coverage reported when the cap is hit |
| Judge (Agentic) | a trace where an OpenCV output changes the next tool call, and a failure drill | I can verify autonomy and human control | Replay shows cause fields (`entropy>τ`, `p95>SLA`); Bedrock/LLM timeout triggers the rule planner live |
| Judge (COOL) | proof the workload runs on COOL on Graviton, with a reproducible benchmark | the COOL claim is verifiable | `cv2.__file__` under `/opt/cool`, build info dumped, `bench/run_all.sh` reproduces three arms from JSON |
| Judge (Overall) | a live endpoint, a 5-minute video, and honest failure cases | I can score execution and impact | HTTPS endpoint up during declared hours; HF Space mirror always on; failure gallery ≥10 cases |
| Operator (me) | nothing to cost money unless credits cover it | the build stays free | Budgets at $50/$100 with stop action; auto-stop schedule; CloudWatch billing alarm |

## Guiding principles

Five rules settle most design arguments in advance.

1. **Reuse before build.** If a maintained open-source project does ≥70% of a capability, we adopt it and add the missing 30%. We write only the gaze policy, the budget governor and glue. Section 7 is the ledger.
2. **Free before paid.** Open-source and free tiers first; AWS only where the award requires it (COOL on Graviton), paid from credits or the grant, with hard caps and auto-stop. No paid API key is required to run the agent.
3. **Hot path = KleidiCV ops only.** `resize`, `pyrDown`, `remap`, `warpPerspective`, `GaussianBlur`, `Sobel`, `medianBlur`, `dilate`/`erode`, `cvtColor`, `convertTo`, `inRange`, `minMaxIdx`, `transpose`/`rotate`. No `filter2D`, no NumPy pixel loops (the entrant that used Gabor `filter2D` measured COOL at 0.94×).
4. **Deterministic core, optional LLM.** The policy and governor are testable code. The LLM proposes strategy changes from an allow-list through a validator and falls back to rules on any error.
5. **Report what we measured.** Pixels, not millimetres; measured speedups, not the organiser's 1.5×; a failure gallery; "first among public 2026 entries we reviewed", never "first ever".

## Requirements

P0 is what ships by Oct 26 even if everything else is cut; P1 ships if Day 3 and Day 7 gates pass; P2 is stretch or pivot material.

### Functional

| ID | Requirement | Priority | Accepted when |
| --- | --- | --- | --- |
| FR-1 | Ingest JPEG/PNG up to \~24 MP via the UI or `POST /jobs`; each job gets an id and a trace file | P0 | A 4000×3000 CODEBRIM original uploads and starts a job in under 2 s |
| FR-2 | Overview: `pyrDown` pyramid L0–L4 and a saliency prior at L3 from `cvtColor`, `GaussianBlur`, `Sobel` magnitude, `adaptiveThreshold`, `morphologyEx`, CLAHE and `inRange` (rust hue) | P0 | `overview()` returns ≤20 ranked candidate regions plus the heatmap in one call |
| FR-3 | Gaze: `gaze(x, y, zoom, yaw, pitch)` renders a rectified crop with `warpPerspective` (P1: `remap` for oblique de-warp) plus sharpness (Laplacian variance) and exposure | P0 | Crop coordinates map back to L0 pixels exactly; quality gate rejects blur/glare |
| FR-4 | Segment: dacl10k-trained ONNX model on OpenCV 5 DNN `ENGINE_NEW`, dynamic H×W; returns per-class masks, entropy and latency (P1: FP16 variant) | P0 | Loads on Graviton under `/opt/cool`; output matches PyTorch within tolerance on 20 crops |
| FR-5 | Measure: TRUCO `findContours`, skeleton + `distanceTransform` → length, width, area in px, orientation via `minAreaRect` | P0 | Width/length within 10% of hand-measured values on 10 crack crops |
| FR-6 | Verify: re-segment the region at the next finer pyramid level and return a mask-IoU agreement score | P0 | Disagreeing regions are flagged for a further gaze or human review |
| FR-7 | Policy core: next gaze = argmax expected value per cost; escalate on entropy > τ or mask area < A\_min; stop when marginal value < budget slope | P0 | Unit tests cover escalation, stop and tie-breaking; policy is deterministic for a seed |
| FR-8 | Budget governor: `set_budget(pixels, usd, seconds)`; predicted ms from live `runtime_stats`; on p95 > SLA switch to FP16, a coarser level or batching; halt at the cap with partial coverage reported | P0 | A forced p95 spike changes the next step's model or level, visible in the trace |
| FR-9 | Telemetry: per-op p50/p95, CPU%, $ so far; ≤10 CloudWatch custom metrics; full detail in the trace JSON | P0 | Dashboard shows per-op latency and $ live during a demo |
| FR-10 | Tools over MCP (FastMCP): `overview`, `gaze`, `enhance`, `segment`, `measure`, `verify`, `runtime_stats`, `set_budget`, `log_finding`, `draft_work_order`, `request_human_review` | P0 | MCP Inspector lists and calls every tool; Claude Code can drive a job |
| FR-11 | LLM planner: proposes strategy changes as JSON from an allow-list; a validator enforces list, ranges and budget; 2 s timeout → rule planner. Default model local llama.cpp (Qwen2.5-3B-Instruct Q4); Bedrock or Gemini free tier optional | P1 | 20 injected timeouts, 20 rule-planner recoveries; no API key needed to run |
| FR-12 | Findings and work orders in DynamoDB: class, severity, px measurements, crop key, gaze path, approval state | P0 | Work order renders in the UI and exports as JSON |
| FR-13 | Human approval: exposed rebar and spalling above the area threshold block until approve/reject in the UI or via an SNS email link; the agent cannot close or downgrade them | P0 | Automated test proves the agent cannot change a critical finding's state |
| FR-14 | Kill switch (global flag) and per-job $ and pixel caps halt the job within one step | P0 | Both shown live in the video |
| FR-15 | Trace per step: observation digest, OpenCV metrics, runtime stats, decision, rationale, tool call, latency, $, and a cause field; replayable | P0 | Replay reproduces the gaze path from the JSON alone |
| FR-16 | UI (Gradio): gallery and upload, live step stream, gaze replay slider, heatmap with numbered path, findings queue with approve/reject | P0 | A judge can run an image end-to-end without instructions |
| FR-17 | UI benchmark tab (static JSON + plots) and CloudWatch dashboard link | P1 | Benchmark plots load from `results/` |
| FR-18 | Batch mode: S3 upload → SQS → worker; results to DynamoDB | P1 | 50 images processed unattended within budget |
| FR-19 | Privacy blur: faces (YuNet) and people/vehicles (NanoDet) blurred before storage and display | P1 | Blur applied on a test image containing a person |
| FR-20 | Evaluation harness: recall-vs-pixels and recall-vs-$ curves for downscale-once, SAHI tiling and DRISHYAM; agent metrics; failure gallery | P0 | Curves on ≥200 images; gallery ≥10 cases in ≥5 categories |
| FR-21 | Ablations: no uncertainty term, no telemetry re-planning, LLM vs rule planner, FP16 vs FP32 | P1 | Each ablation on ≥50 images with the same seed |
| FR-22 | Benchmark: arms A/B/C (D optional); micro, macro and DNN-only workloads; provenance JSON; plots; mask parity A vs B | P0 | `bench/run_all.sh` reproduces all plots from fresh instances |
| FR-23 | Free always-on mirror: the same Gradio app on a Hugging Face Space with stock OpenCV 5 (x86) and recorded-trace replay, labelled "not COOL" | P1 | Space URL in the submission; stays up while the EC2 endpoint is stopped |
| FR-24 | IaC (CDK, Python): EC2 + EBS, DynamoDB, SQS, SNS, CloudWatch dashboard and alarms, Budgets, EventBridge Scheduler auto-stop (Budgets and auto-stop are P0 and may be set by hand on Day 1) | P1 | `cdk deploy` from a clean account reproduces the stack |
| FR-25 | Mosaic of overlapping shots with `cv2.Stitcher` to give the agent a larger canvas | P2 | Stretch only |
| FR-26 | Async PaliGemma caption via OpenCV `samples/dnn/vlm_inference.py`, labelled experimental | P2 | Stretch only; never on the critical path |
| FR-27 | COOL autotuner mode: a developer agent that tunes pipeline configs (kernel sizes, interpolation, threads, pyramid depth) against mask IoU and latency on Graviton | P2 | Pivot target if the segmenter fails the Day 3 gate |

### Non-functional

| ID | Area | Requirement |
| --- | --- | --- |
| NFR-1 | Performance | Target p95 per gaze step (warp + FP32 segment at 512×512) ≤500 ms on c8g.2xlarge with COOL; the final SLA is set from Day 3 measurements; the governor reacts within one step |
| NFR-2 | Cost | $/1k images reported per arm; default per-job cap $0.05; total AWS spend ≤$100 |
| NFR-3 | Reliability | 100% recovery from planner errors; jobs restart-safe with idempotent step ids; EC2 endpoint up during declared judging hours; HF mirror always on |
| NFR-4 | Security | Least-privilege IAM per service; no shell tool; in-image text never reaches the planner; HTTPS only; secrets in SSM Parameter Store (standard tier) |
| NFR-5 | Reproducibility | `uv` lock file, pinned AMI id, `cv2.__version__` and build-info hash, fixed seeds, `make eval`, `bench/run_all.sh` on fresh instances |
| NFR-6 | Licensing | Datasets downloaded by script and never committed (both are non-commercial); code MIT/Apache; no AGPL dependencies |
| NFR-7 | Portability | Identical code on stock OpenCV 5 (x86 or Arm); COOL is selected by activating the `/opt/cool` venv |
| NFR-8 | Observability in free tier | ≤10 custom metrics, 1 dashboard, ≤10 alarms, ≤5 GB logs per month |
| NFR-9 | Honesty | Pixel units; measured figures only; "among public 2026 entries we reviewed"; failure gallery in the report and the UI |

## Build vs. reuse matrix

Every capability below maps to an existing project; DRISHYAM's own code is the policy, the governor and the glue between rows. Licenses marked "check" must be confirmed on Day 1 before the dependency is pinned.

| Capability | Reuse (license) | We add | Why this and not a rewrite |
| --- | --- | --- | --- |
| Pyramid, virtual PTZ, saliency ops | OpenCV 5 built-ins: `pyrDown`, `warpPerspective`, `remap`, `resize`, `Sobel`, `GaussianBlur`, `adaptiveThreshold`, `morphologyEx`, CLAHE, `inRange` (Apache 2.0) | Composition, per-op timers | Ops KleidiCV covers in some formats; the stock aarch64 wheel already links KleidiCV 26.03 and carotene (`cv2.getBuildInformation()`: "Custom HAL: YES (carotene (ver 0.0.1) KleidiCV (ver 26.03))"), so a COOL gain would come on top of that, and only by calling these ops, not replacing them |
| Segmentation model | [segmentation\_models\_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) UNet/DeepLabV3+ with a timm MobileNetV3 encoder, ImageNet-pretrained (MIT); [dacl10k-toolkit](https://github.com/phiyodr/dacl10k-toolkit) loader and baseline script (check) | Fine-tune on dacl10k on Kaggle; ONNX export at opset 17 with dynamic H×W | Pretrained encoders and a proven training loop; search Hugging Face and GitHub for released dacl10k weights first and fine-tune those if any exist |
| ONNX graph cleanup | [onnxslim](https://github.com/inisis/OnnxSlim) (MIT) or onnx-simplifier | None | Removes the shape ops that make OpenCV DNN refuse a graph |
| DNN inference | OpenCV 5 DNN `ENGINE_NEW`; `samples/dnn/segmentation.py` pattern (Apache 2.0) | Pre/post-processing, FP16 variant | Keeps the model on the COOL build; no second runtime to ship |
| Exhaustive-tiling baseline and merge | [SAHI](https://github.com/obss/sahi) `slice_image` and merge/NMS (MIT) | Thin wrapper for masks | The standard slicing library makes baseline (ii) credible to judges |
| Zoom-search scheduling | [ZoomEye](https://github.com/om-ai-lab/ZoomEye) tree-search formulation (check) | Replace VLM confidences with saliency and entropy; add the cost denominator | Published algorithm, cited as prior art; the adaptation is the novelty |
| Crack skeleton and width | scikit-image `skeletonize` (BSD-3) + `cv2.distanceTransform` | A 20-line `measure()` | Standard morphology; nothing to invent |
| Agent loop, tool calling, tracing | [Strands Agents SDK](https://github.com/strands-agents/sdk-python) (Apache 2.0): MCP client and OpenTelemetry built in; model providers for Bedrock, Ollama and OpenAI-compatible servers | Allow-list validator, rule-planner fallback | AWS-native story for judges; swapping Bedrock for a local llama.cpp server is a config line |
| MCP server | [mcp Python SDK](https://github.com/modelcontextprotocol/python-sdk) FastMCP (MIT); MCP Inspector for live traces | Tool schemas | Standard; Claude Code and Kiro can drive the same tools |
| Local LLM planner | [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (MIT) with Qwen2.5-3B-Instruct GGUF Q4\_0 (Apache 2.0); Gemma 3 4B as alternate | Prompt and JSON schema | Free, CPU-only, Arm-optimised kernels; no API key; Qwen 2.5 matches OpenCV 5's own LLM support story |
| Hosted LLM (optional) | Gemini API free tier, GitHub Models, Groq free tier via LiteLLM; Bedrock only on credits | Provider config | Free tiers exist; Bedrock is a demo option, not a dependency |
| Web UI | [Gradio](https://github.com/gradio-app/gradio) mounted in FastAPI: `AnnotatedImage`, streaming generators, buttons (Apache 2.0) | Replay and approval views | No React build; the same app runs on a Hugging Face Space |
| HTTPS endpoint | [Caddy](https://github.com/caddyserver/caddy) auto-TLS (Apache 2.0) + a free DuckDNS name; Tailscale Funnel or a Cloudflare quick tunnel as fallbacks | Caddyfile | Replaces ALB, API Gateway and certificate work at $0 |
| Privacy blur | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) YuNet face (MIT) and NanoDet person/vehicle (Apache 2.0) | Blur pass | Models already validated against OpenCV DNN |
| Metrics | [supervision](https://github.com/roboflow/supervision) (MIT), torchmetrics mIoU (Apache 2.0), pycocotools-style recall@IoU | Recall-vs-budget script | Standard implementations judges recognise |
| Benchmark harness | pytest-benchmark (BSD-2), hyperfine (MIT/Apache), psutil, Plotly | Provenance block, three-arm runner | Adopt the three-arm pattern fellow entrants published and credit them |
| Telemetry | `aws-embedded-metrics` for CloudWatch EMF (Apache 2.0); Strands OpenTelemetry; Langfuse Cloud Hobby tier optional | Metric names | No custom collectors |
| IaC and auto-stop | AWS CDK Python (Apache 2.0); EventBridge Scheduler universal target `ec2:StopInstances` | One stack | Stop without a Lambda |
| Batch | S3 event → SQS → boto3 worker | Consumer loop | Always-free quotas |
| Human approval | Gradio buttons + DynamoDB + SNS email | Flow | No workflow engine for a two-state decision |
| Mosaic (P2) | `cv2.Stitcher` | None | Built-in beats a custom ALIKED + LightGlue pipeline for a stretch goal |
| VLM caption (P2) | OpenCV `samples/dnn/vlm_inference.py`, PaliGemma 2 | Async job | Experimental; CPU inference takes over a minute |
| Report and site | MkDocs Material + GitHub Pages; pandoc → PDF; Mermaid and diagrams.net | Content | Free and versioned with the repo |
| Video | OBS Studio; DaVinci Resolve (free) or Kdenlive | None | Free |
| CI | GitHub Actions, including free arm64 runners for public repos | Workflow | Arm builds and tests without touching AWS |
| Development | Laptop or GitHub Codespaces free hours; Graviton only for Arm validation and benchmarks | None | Keeps EC2 hours, and therefore credits, low |

## Free-resource plan and cost ledger

"Free" here means $0 out of pocket: open-source software, free third-party tiers, and AWS usage paid from the new-account Free Tier credits (up to $200) or the competition compute grant ($150). COOL on Graviton cannot run at literally $0 beyond its 7-day trial, so the plan keeps that spend at roughly $70 and never lets it reach a card. Free-tier limits change; the last column is the Day-1 checklist.

### Resources and their free mechanism

| Resource | Role | Free mechanism | Limit to respect | Verify Day 1 |
| --- | --- | --- | --- | --- |
| AWS Free Tier credits | Pays for EC2, EBS, IPv4 | New accounts: $100 at sign-up plus up to $100 for completing activities; 6-month expiry | Credits may not cover Marketplace software fees | Whether a Free Plan account can subscribe to a Marketplace AMI; if not, upgrade to the Paid Plan (credits carry over) and set a stop-action budget first |
| Competition compute grant | Pays for EC2 | 55 grants of $150 (one page says $200) | Application may be closed | Apply today if open |
| COOL 3.1 AMI (Graviton4) | Claimed workload | 7-day free trial on the $0.02/hr software fee (c8g.2xlarge) | Trial clock starts at subscription | Subscribe only when the instance will be used the same day; supported instance types for the AMI |
| EC2 c8g.2xlarge | Agent host, arm A | Credits or grant, $0.319/hr on-demand | Stop when idle | Quota for c8g in the chosen region |
| EC2 c8g (stock) and c7i.2xlarge | Benchmark arms B and C | Credits or grant, \~3 h each | Terminate after the run | c7i quota; `opencv-python` 5.x wheels exist for aarch64 and x86 |
| EBS gp3 \~40 GB | Eval subset, crops, traces | Credits; legacy accounts get 30 GB free for 12 months | dacl10k never lands on EC2 (training is on Kaggle) | — |
| Public IPv4 | Endpoint address | $0.005/hr from credits; legacy accounts 750 h/month free | Counted only while the instance runs | — |
| S3 ≤5 GB | Crops and trace JSON only | Legacy 12-month free tier; credits otherwise | Datasets stay on EBS, never in S3 | — |
| DynamoDB | Findings, work orders, approvals | Always free: 25 GB, 25 RCU and 25 WCU | On-demand off; provisioned at 5/5 | — |
| SQS, SNS, EventBridge Scheduler, Step Functions | Batch, approval email, auto-stop | Always free: 1M SQS requests, 1M SNS publishes and 1,000 emails, 14M Scheduler invocations, 4,000 state transitions | — | — |
| CloudWatch | Dashboard, alarms, logs | Always free: 10 custom metrics, 10 alarms, 3 dashboards, 5 GB logs | Cap custom metrics at 10; detail goes to trace JSON | — |
| AWS Budgets, SSM, CDK | Caps, run-command, IaC | Free (first two action-enabled budgets free) | — | — |
| Kaggle Notebooks | Segmenter training | 30 GPU hours/week (T4×2 or P100), 12-hour sessions | Checkpoint every epoch to Kaggle output or HF Hub | GPU quota visible in the account |
| Google Colab, Lightning AI | Backup training | Free T4 sessions; a monthly GPU-hour allowance | Sessions can disconnect | — |
| Hugging Face Hub and Spaces | Model weights; always-on mirror endpoint | Public repos free; Space CPU Basic (2 vCPU, 16 GB) free | Space sleeps after 48 h idle; wake it before judging | — |
| GitHub | Repo, CI, docs, dev | Public repo; Actions with free arm64 runners for public repos; Pages; Codespaces free hours | — | arm64 runner label available |
| LLM planner | Strategy proposals | llama.cpp locally (free); Gemini API, GitHub Models or Groq free tiers optional; Bedrock only on credits | Free API tiers are rate-limited | — |
| Langfuse Cloud | Optional trace UI | Hobby tier free | Event quota | — |
| DuckDNS, Tailscale Funnel, Cloudflare quick tunnel | HTTPS URL | Free | Quick-tunnel URLs are ephemeral; prefer DuckDNS + Caddy | — |
| CODEBRIM (Zenodo), dacl10k (GitHub/HF) | Data | Free downloads under non-commercial licenses | Never redistributed in the repo | Both downloads start today (8.3 GB for CODEBRIM) |
| OBS, DaVinci Resolve, Kdenlive, diagrams.net, MkDocs | Video, diagrams, report | Free | — | — |

### Cost ledger (estimate, us-east-1 on-demand)

| Item | Hours | Rate ($/hr) | Est. $ |
| --- | --- | --- | --- |
| c8g.2xlarge with COOL, development and validation | 36 | 0.319 + 0.02 (fee waived 7 days) | 12 |
| Benchmark arm B, c8g.2xlarge stock | 3 | 0.319 | 1 |
| Benchmark arm C, c7i.2xlarge stock | 3 | 0.357 | 1 |
| Benchmark arm D, c9g (optional) | 3 | \~0.40 (unverified) | 1 |
| Judging-window endpoint, c8g.2xlarge 10 h/day for 14 days (or a smaller c8g 24/7 if the AMI allows it) | 140 | 0.339 | 48 |
| EBS 40 GB gp3 for one month; public IPv4 while running | — | — | 5 |
| Everything else (DynamoDB, SQS, SNS, CloudWatch, Scheduler, S3 under 5 GB, Kaggle, HF, GitHub, local LLM) | — | free tier | 0 |
| **Total** |  |  | **≈68; ≤100 with contingency** |

### Spend guardrails

- AWS Budgets at $50, $100 and $150 with email alerts; the $100 budget carries a stop-instances action
- EventBridge Scheduler stops all tagged instances at 23:00 PT nightly; a cron on the instance stops it after 3 hours with no requests
- One region, no NAT gateway, no ALB, no Elastic IP left unattached, no instance left running after the video is recorded
- Credits exhausted or grant denied → the no-credit fallback below, decided at the Day 2 stand-up

### No-credit fallback ($0 cash, forfeits the COOL award)

Run the whole stack on free tiers: the Hugging Face Space (x86, stock OpenCV 5) as the live endpoint, Kaggle for training, GitHub arm64 runners for Arm tests, GitHub Pages for the report. The entry then competes for Overall and Agentic Vision only, as the brief's pivot rule already allows. An Oracle Cloud Always Free Ampere A1 instance (4 OCPU, 24 GB) is a free Arm host for the demo if capacity happens to be available; it is not Graviton and earns no COOL points.

## Agent policy specification

The policy is deterministic code that picks the next gaze by expected value per millisecond, escalates on uncertainty, and stops when the next look is not worth its cost; the LLM only proposes strategy changes through a validator.

### Scoring

For each candidate region r at the current pyramid level:

```latex
\text{score}(r) = \frac{p_r \cdot s_{c(r)} \cdot (1 + u_r) \cdot a_r}{\hat{t}(r)}
```

- p\_r: saliency prior from `overview()`, 0–1
- s\_c: class severity weight; initial defaults exposed rebar 3.0, spalling 2.0, crack 1.5, corrosion stain 1.0, efflorescence 0.5, tuned on Day 6
- u\_r: segmenter entropy at the last level that saw r (0 if unseen)
- a\_r: fraction of r not yet seen at the target level
- t̂(r): predicted milliseconds for the planned ops (warp + segment at that level and model), read from the live `runtime_stats()` p95

### Rules

0. **Survey:** when one look at a coarser level can hold every candidate window and the budget leaves at least one more look, that look comes first (cause `survey`); its follow-ups use rules 2 and 3, and candidates are then scored at their own level as below
1. **Next gaze:** argmax score(r); ties broken by larger a\_r, then by position (top-left first) for determinism
2. **Escalate** (zoom one level finer, same centre) when entropy > τ (default 0.5) or the mask area is below A\_min px (default 64), down to L0
3. **Verify** when a mask appears at a new level: re-segment and record the IoU with the previous level; IoU < 0.5 schedules one more gaze or, for a critical class, a human review
4. **Stop** when the best remaining score is below λ = ε × (value gained so far ÷ cost spent so far), ε default 0.25, or when any cap is hit
5. **Governor** runs before every step (table below) and can override the model, level or batch size; every override writes a cause into the trace

| Signal from `runtime_stats()` | Governor action | Trace cause |
| --- | --- | --- |
| p95(segment) > SLA (default 500 ms) | Switch to FP16, then to one coarser level | `p95>SLA` |
| p95(warp) > 2× its running median | Shrink gaze size; batch remaining candidates | `warp-spike` |
| $ spent ≥ 80% of the job cap | Verify only candidates with a critical class | `usd≥80%` |
| Pixels or $ at 100% of the cap, or seconds exceeded | Halt, report coverage and the unvisited candidates | `cap-hit` |
| Kill switch set | Halt immediately, state preserved for resume | `kill-switch` |

### Planner contract (LLM, optional)

The planner receives a numeric digest only (scores, entropies, p95s, $ and pixels remaining, counts by class) and must answer with this JSON; anything else is discarded and the rule planner decides.

```json
{
  "strategy": "continue | switch_model | change_level | reverify | batch_remaining | stop",
  "params": {"model": "fast | fp16 | accurate", "level": 0, "region_id": "r12"},
  "rationale": "<= 200 characters"
}
```

- Validator: strategy in the allow-list, params within ranges, no change that exceeds the budget; 2-second timeout; on any failure the rule planner's choice stands, the look keeps its own cause, and the step records the outcome in `planner_status` (`rejected` or `fallback`; the page shows it as `planner-fallback`)
- The planner never sees OCR output or any text found in images (graffiti, signs); image text is untrusted and is not a tool result
- Default model: local `llama-server` with Qwen2.5-3B-Instruct Q4\_0; Bedrock Claude or Gemini free tier selectable by environment variable for the demo

### Autonomy boundaries

| The agent may | Only a human may | Never |
| --- | --- | --- |
| Look, zoom, re-run, switch model, change level, batch, draft work orders, request review | Approve, reject, close or downgrade a critical finding (exposed rebar; spalling above the area threshold); raise a job cap | Run shell commands; call tools outside the MCP allow-list; output millimetres or safety ratings; act on text read from images |

### Trace record (one JSON line per step)

```json
{"job": "j_0142", "step": 7, "t": "2026-10-24T17:02:11Z",
 "observation": {"entropy": 0.62, "saliency_max": 0.81, "mask_area_px": 41, "sharpness": 212.4},
 "runtime": {"p50_ms": {"warp": 9, "segment": 310}, "p95_ms": {"warp": 14, "segment": 540}, "cpu_pct": 71, "usd": 0.0031},
 "decision": {"action": "gaze", "args": {"x": 2210, "y": 1480, "zoom": 2, "yaw": 0, "pitch": 0}},
 "rationale": "entropy 0.62 > 0.5 at L2; area 41 px < 64",
 "cause": "entropy>tau", "planner": "rule", "latency_ms": 352, "cost_usd": 0.00004}
```

The values above are illustrative. Traces are written to EBS and copied to S3 per job; DynamoDB keeps the summary (steps, pixels, $, findings, approval state).

## Data

Train on dacl10k, evaluate on 200 CODEBRIM originals, and keep both datasets out of the repo and out of S3.

| Dataset | Use | Size | License | Where it lives during the build |
| --- | --- | --- | --- | --- |
| [dacl10k](https://github.com/phiyodr/dacl10k-toolkit) (Flotzinger et al., WACV 2024) | Segmenter training and validation, official split | 9,920 images, >100 bridges, 19 classes (13 damage + 6 objects) | CC BY-NC 4.0 | Kaggle or Hugging Face dataset cache only; never copied to EC2 |
| [CODEBRIM](https://doi.org/10.5281/zenodo.2620293) (Mundt et al., CVPR 2019), original images | Evaluation scenes with bounding-box ground truth; the headline recall curves | 1,590 high-res images, 30 bridges, 5,354 boxes, 5 defect classes | Non-commercial | `CODEBRIM_original_images.zip` (8.3 GB) downloaded to EBS; a 200-image eval subset is the working set |
| SegCODEBRIM (optional) | Crack-mask sanity check for `measure()` | Derived from CODEBRIM | MIT | Downloaded on demand |

- **Eval subset:** 200 CODEBRIM originals chosen by a fixed seed, stratified so each of the 5 classes has ≥40 boxes; the file list `evals/codebrim_eval200.txt` is committed so anyone can rebuild it. The remaining 1,390 images are a final sweep only if Day 8 leaves time.
- **Class mapping for scoring:** dacl10k damage classes collapse to CODEBRIM's five (crack, spallation, efflorescence, exposed bars, corrosion stain); the mapping table is committed and reported, including which dacl10k classes are dropped.
- **Leakage check:** compare bridge identifiers and perceptual hashes (imagehash, BSD) between dacl10k and the eval subset; any overlap is removed from the subset and the count is reported.
- **Model weights:** our fine-tuned ONNX files are trained on dacl10k, so they inherit its CC BY-NC 4.0 licence (non-commercial); `scripts/install_model.sh` writes that into `models/README.md`, and the README's Licence section states it.
- **Redistribution:** download scripts only (`make data`), with checksums; no image is committed.

## Evaluation and success metrics

The headline exhibit is recall against pixels and against dollars, for three policies on the same 200 images; everything else supports it.

### Headline curves

- **Recall:** a CODEBRIM box counts as found when a predicted mask of the mapped class overlaps it at IoU ≥ 0.3 or contains its centre; reported per class and overall
- **Budgets:** DRISHYAM is run at pixel caps of 5, 10, 20, 30, 50 and 100% of the full-resolution image; downscale-once is one point (one 1024-px read); SAHI tiling is one point (100%) plus coarser tile grids for intermediate budgets
- **Cost axis:** $ per image = measured wall time × ($0.319 + $0.02)/3600 on arm A, so the curve is the thing a buyer pays for
- **Claim discipline:** the report states the measured recall-parity and pixel fraction; the ≥0.9× at ≤0.3× target is a hypothesis in this PRD, not a number to defend

### Agent metrics

| Metric | How measured | Target or report |
| --- | --- | --- |
| Escalation rate | Gazes that triggered a finer level ÷ gazes | Report |
| False-accept rate on critical classes | Critical findings drafted on boxes with no critical ground truth | ≤ SAHI tiling's rate |
| Steps per image, pixels per image | From traces | Report with p50 and p95 |
| Planner-failure recovery | 20 injected LLM timeouts and 20 malformed replies | 40 of 40 recovered by the rule planner |
| Governor response | 10 injected p95 spikes | 10 of 10 change the next step's model or level |
| Human-approval latency | Request → decision in DynamoDB | Report |
| Steps with a logged cause | From traces | 100% |

### Segmenter

Report dacl10k validation mIoU per class. Reference points from the dacl10k paper: a MobileNetV3 DeepLabV3+ scored 0.320 on validation and the best model (FPN + EfficientNet-B4) 0.42 on test, so anything near 0.3 is fit for purpose; the policy, not the model, is the contribution.

### Ablations (P1, ≥50 images each, same seed)

- No uncertainty term (u\_r = 0)
- No telemetry re-planning (fixed model and level)
- LLM planner vs. rule planner
- FP16 vs. FP32 segmenter

### Failure gallery (P0)

At least 10 cases across ≥5 categories: cracks thinner than the effective resolution; graffiti, formwork marks or stains read as damage; glare and blur caught by the quality gate; spalling vs. rockpocket vs. washout confusion (washouts had the lowest dacl10k test IoU, 0.121); over-confidence on steel or masonry. Each case shows the gaze path, the wrong output and the rule or threshold that would have caught it.

## COOL benchmark plan

Three arms separate the COOL gain from the hardware gain; plan for 1.2–1.3× end-to-end and report whatever the JSON says.

| Arm | Instance | OpenCV build | Purpose | Budget |
| --- | --- | --- | --- | --- |
| A | c8g.2xlarge, COOL 3.1 AMI | COOL (OpenCV 5.0-based) via `/opt/cool` venv | Claimed workload | Shared with development |
| B | c8g.2xlarge, Ubuntu 24.04 | `opencv-python-headless` 5.x pip, pinned | Isolates COOL from Graviton | 3 h |
| C | c7i.2xlarge, Ubuntu 24.04 | Same pip wheel, x86 | x86 baseline | 3 h |
| D (optional) | c9g.2xlarge, Graviton5 COOL listing | COOL G5 (OpenCV 5.1.0-dev reported by another entrant) | Forward-looking | 3 h, only if the AMI supports c9g |

**Workloads**

1. Micro: each op on the gaze and saliency path (`pyrDown`, `warpPerspective`, `remap`, `resize`, `GaussianBlur`, `Sobel`, `adaptiveThreshold`, `morphologyEx`, CLAHE, `inRange`, `cvtColor`) at 3 image sizes
2. Macro: the full DRISHYAM job on the 200-image eval subset at a fixed pixel budget, rule planner, single worker
3. DNN only: the segmenter forward pass at 512×512, FP32 and FP16, reported separately because KleidiCV mainly targets imgproc and near-parity is expected

**Metrics:** p50/p95/p99 per op and per image; images/s and gazes/s; CPU utilisation (psutil and CloudWatch); thread scaling at 1/2/4/8 workers; $/1k images = (instance $/hr + COOL fee) × hours ÷ images; an Amdahl split of time in KleidiCV-covered ops vs. DNN vs. Python.

**Reproducibility:** `bench/run_all.sh` launched through SSM on fresh instances; pinned AMI id, `cv2.__version__`, build-info hash and lock file; pytest-benchmark with warm-up, ≥30 repeats, fixed `cv2.setNumThreads` and seeds; raw JSON in `results/` with a provenance block (instance id, AMI, kernel, timestamp); one plotting script; a parity check that arms A and B produce identical masks within tolerance.

**Expectations to state up front:** op-level 1.2–1.8× on KleidiCV ops (AWS reports many imgproc functions at 1.8×; OpenCV dev notes show adaptive threshold at 3× and CLAHE at 1.56×); end-to-end 1.2–1.3× (AWS's Mars Pathfinder workflow went from 1.2 to 1.6 fps, a fellow entrant measured 1.26×); DNN-dominated jobs gain less. Hourly cost before any speedup: c8g.2xlarge $0.319 + $0.02 fee = $0.339 vs. c7i.2xlarge $0.357, about 5% cheaper, so the $/1k-images gain comes mostly from throughput.

**Design rule:** `filter2D` and NumPy pixel loops stay off the hot path; any profile showing them above 5% of time is a bug.

## Human control, safety and responsible use

A human signs every critical finding, the agent has no shell and no way to act on text it sees, and every number it reports is a pixel count it measured.

| Area | Requirement |
| --- | --- |
| Critical findings | Exposed rebar, and spalling above the area threshold, require approve/reject by a person in the UI or through the SNS email link; the agent cannot close or downgrade them (FR-13) |
| Automation bias | Evidence-first layout: crop, gaze path and measurements before any score; an "AI-assisted, not an inspection" banner on every finding |
| Kill switch and caps | Global stop flag; per-job pixel and $ caps; both demonstrated live (FR-14) |
| Audit trail | One trace line per step, replayable; findings, work orders and approvals in DynamoDB with timestamps and the approver |
| Measurement honesty | Length, width and area in pixels only; no millimetres without a scale reference; no structural-safety ratings |
| Privacy | Faces, people and vehicles blurred before storage and display (FR-19); crops retained 30 days; no imagery of people kept |
| Prompt injection | Text inside images (graffiti, signs, OCR) is never passed to the planner; the planner sees numeric digests only |
| Cloud security | Least-privilege IAM role per service: EC2 role limited to its S3 prefix, one DynamoDB table, one SQS queue, one SNS topic, CloudWatch PutMetricData; no shell tool for the agent; HTTPS only; secrets in SSM Parameter Store |
| Data bias | Report per-class and per-condition performance (wet vs. dry, lighting, camera). dacl10k is half engineering-office data and half German local-authority data from 2000–2020; CODEBRIM covers 30 bridges; the report names these geographic and material limits |
| Licensing | Non-commercial datasets downloaded by script, never redistributed; permissive code and bundled models, except the optional dacl10k-trained weights (CC BY-NC 4.0); no AGPL dependencies |
| Cost and energy | Report the dollars and CPU-hours of every evaluation and benchmark run |
| Surveillance | DRISHYAM inspects structures, not people; the UI rejects images where a person occupies more than a set fraction of the frame |
