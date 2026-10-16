# Requirement traceability

Every functional and non-functional requirement in [the PRD](PRD.md), where it
lives, the test that proves it, and its honest status. "AWS" means the code is
written and runs anywhere, but the requirement names an AWS service this
repository cannot provision without the owner's account and credits.

Status key: **done** (built and tested here) · **done, adapted** (built and
tested, on a free substitute for a paid service) · **AWS** (code or runbook
ready; needs the owner's AWS account) · **open**.

## Functional

| ID | Requirement | Where | Test | Status |
| --- | --- | --- | --- | --- |
| FR-1 | Ingest JPEG/PNG up to ~24 MP; job id + trace | `imaging.decode_image`, `web.create_job`, `cli run` | `test_agent::test_large_upload_starts_fast` (4000×3000 starts < 2 s), `test_imaging`, `test_cli` | done |
| FR-2 | Overview: pyrDown L0–L4, saliency at L3 (cvtColor, GaussianBlur, Sobel, adaptiveThreshold, morphologyEx, CLAHE, inRange); ≤20 ranked candidates + heatmap in one call | `imaging.build_pyramid`, `saliency`, `find_candidates` | `test_imaging::test_overview_finds_the_salient_spot`, `::test_pyramid_levels` | done |
| FR-3 | Gaze with warpPerspective (+yaw/pitch), sharpness and exposure; exact L0 mapping; quality gate | `imaging.gaze`, `quality` | `test_imaging::test_gaze_maps_back_to_l0_exactly[0-3]`, `::test_gaze_yaw_pitch_changes_view`, `::test_quality_gate` | done |
| FR-4 | Segment: dacl10k ONNX on OpenCV 5 DNN `ENGINE_NEW`, dynamic H×W, masks, entropy, latency | `segment.OnnxSegmenter` (+ `ClassicalSegmenter` default) ; `training/` | `test_segment::test_onnx_engine_new_dynamic_hw` (3 sizes), `::test_onnx_class_map_merges_channels`; `training/parity.py` | done (engine + export pipeline); trained dacl10k weights need a Kaggle GPU run |
| FR-5 | Measure: contours, skeleton + distanceTransform, minAreaRect → length/width/area/orientation in px | `measure.measure` (hit-or-miss thinning) | `test_measure` (25 drawn cracks within 10 %, curved crack, scale) | done |
| FR-6 | Verify: re-segment one level finer, mask-IoU agreement | `agent.look_iou`, verify steps; MCP `verify` | `test_agent::test_opencv_output_changes_the_next_tool_call`, `test_mcp` | done |
| FR-7 | Policy: argmax EV/cost, escalate on entropy > τ or area < A_min, stop on λ; deterministic | `agent.Job._score`, `run` | `test_agent::test_deterministic_for_a_fixed_clock`, `::test_opencv_output_changes_the_next_tool_call` | done |
| FR-8 | Budget governor: caps, p95 > SLA → fast model then coarser level, halt with partial coverage | `governor.check`, `Budget` | `test_planner_governor::test_p95_breach_switches_model_then_level`, `test_agent::test_forced_p95_spike_changes_next_steps_model`, `::test_pixel_cap_is_never_exceeded`, `::test_seconds_and_usd_caps_halt` | done |
| FR-9 | Telemetry: per-op p50/p95, CPU %, $ so far | `governor.RuntimeStats` → every step's `runtime` | `test_agent::test_event_order_and_every_step_has_a_cause` | done; CloudWatch EMF export is AWS |
| FR-10 | MCP tools: overview, gaze, enhance, segment, measure, verify, runtime_stats, set_budget, log_finding, draft_work_order, request_human_review | `drishyam/mcp_server.py` (+ `open_image`) | `test_mcp` (allow-list is exactly these 12, each called through the MCP protocol) | done |
| FR-11 | LLM planner: allow-list JSON, validator, 2 s timeout → rules; local llama.cpp default, no key needed | `planner.HttpPlanner`, `validate`, `consult` | `test_planner_governor::test_20_timeouts_and_20_malformed_all_recover` (40/40), `::test_validator_rejects` | done |
| FR-12 | Findings and work orders; export JSON | `findings.work_order`, `cli review`, UI download | `test_findings`, `test_cli`, e2e download | done, adapted (signed JSON instead of DynamoDB) |
| FR-13 | Critical findings block until a human approves or rejects; agent cannot change them | `findings.decide`, `Finding.agent_update`, `/api/review`, `drishyam review` | `test_findings::test_agent_cannot_decide_or_downgrade`, `test_agent::test_critical_finding_blocks_work_order_and_agent_cannot_close_it`, `test_web::test_review_flow_and_tampering`, `test_mcp` | done, adapted (signed review link flow instead of SNS email) |
| FR-14 | Kill switch and $ / pixel caps halt within one step | `governor.check`, `/api/kill`, `drishyam kill`, Ctrl-C | `test_agent::test_kill_switch_halts_within_one_step`, `test_web::test_kill_endpoint_halts_a_running_job`, `test_planner_governor::test_global_kill_file`, e2e KILL | done |
| FR-15 | One trace record per step with cause; replayable | `agent.Job.run`, `gaze_path_from_trace`, replay | `test_agent::test_replay_reproduces_gaze_path_from_trace_alone` | done |
| FR-16 | UI: gallery, upload, live steps, replay slider, heatmap + numbered path, findings queue with approve/reject | `public/` | `tests/e2e/test_ui_live.py` (real stack), `tests/e2e/test_ui_mock.py` | done, adapted (vanilla JS on Vercel instead of Gradio) |
| FR-17 | Benchmark tab from `results/` | `public/js/results.js`, `public/results/summary.json` | e2e | done |
| FR-18 | Batch S3 → SQS → worker | `drishyam run` over a folder is the local equivalent | — | AWS |
| FR-19 | Privacy blur: faces (YuNet), people/vehicles (NanoDet); people-dominant refused | `privacy.privacy_pass` | `test_web::test_people_dominant_image_refused`, `::test_privacy_blur_applies_before_analysis`; verified on an OpenCV sample photo of people | done |
| FR-20 | Evaluation: recall-vs-pixels and recall-vs-$ for downscale, tiling, Drishyam; agent metrics; failure gallery | `evals/harness.py`, `evals/failures.py` | `tests/test_evals.py` | done on synthetic-200; CODEBRIM run needs `make data` (8.3 GB) |
| FR-21 | Ablations | harness flags | — | open |
| FR-22 | Benchmark arms A/B/C, micro/macro/DNN, provenance, parity | `bench/` | `tests/test_evals.py` (parity) | done locally; arms on EC2 are AWS |
| FR-23 | Free always-on mirror, labelled "not COOL" | Vercel deployment | live e2e | done, adapted (Vercel instead of a Hugging Face Space) |
| FR-24 | IaC (CDK) | — | — | AWS |
| FR-25–27 | Stitcher mosaic, PaliGemma caption, COOL autotuner | — | — | P2, not started |

## Non-functional

| ID | Area | Where | Status |
| --- | --- | --- | --- |
| NFR-1 | p95 per gaze step ≤ 500 ms | measured per step in `runtime`; classical segmenter ~100 ms at 512² on a laptop | done (measured locally; Graviton number is AWS) |
| NFR-2 | $/1k images reported; default cap $0.05 | `governor.USD_PER_HOUR`, `bench/macro.py` | done |
| NFR-3 | 100 % planner recovery; restart-safe | `planner.consult`; `--resume` / replay | done (40/40) |
| NFR-4 | Least privilege, no shell tool, image text never reaches planner, HTTPS, secrets | [security.md](security.md) | done (IAM roles are AWS) |
| NFR-5 | Reproducibility: lock file, pinned OpenCV, seeds | `uv.lock`, `opencv-python-headless==5.0.0.93`, seeded synth + fake-clock tests | done |
| NFR-6 | Datasets never committed; permissive deps | `scripts/get_data.sh`, `.gitignore`, `models/README.md` | done |
| NFR-7 | Same code on stock OpenCV 5, x86 and Arm | CI on `ubuntu-24.04` and `ubuntu-24.04-arm` | done |
| NFR-8 | Free-tier observability | Vercel runtime logs (one line per job) | done, adapted |
| NFR-9 | Honesty: pixels only, measured numbers, failure gallery | measurements are `*_px`; UI banner; `public/results/failures.json` | done |
