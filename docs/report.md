# Drishyam — submission report

*Virtual-PTZ inspection of bridge photos with an agent that spends compute in
proportion to risk.* Live demo: https://drishyam-seven.vercel.app · Code: https://github.com/vj-io/drishyam

## 1. Problem

Bridge owners have photo backlogs they cannot inspect: the defects that matter
most (hairline cracks, early exposed rebar) are invisible at thumbnail scale, so
a tool must either tile every photo at full resolution or guess where to look.
Drishyam's bet is that *looking is cheap when it is a geometric warp*, and that
where to look next can be read from what OpenCV just measured.

## 2. What the agent does

Each job is a loop of camera moves over an image pyramid (L0 = full resolution,
L4 = 1/16):

1. **Overview.** `pyrDown` to L4; a saliency prior at L3 from `cvtColor`,
   `GaussianBlur`, `Sobel`, black-hat `morphologyEx`, `adaptiveThreshold`, CLAHE
   and a rust-hue `inRange`; up to 20 ranked candidate windows.
2. **Gaze.** `warpPerspective` renders a 512² virtual-PTZ view (pan, zoom, and
   yaw/pitch for oblique de-warp) whose pixels map back to L0 exactly.
3. **Quality gate.** Laplacian sharpness and exposure; glare and blur are logged
   and not trusted.
4. **Segment.** Per-class probabilities, masks and an entropy score. The demo
   engine is classical (evidence maps, no weights); a dacl10k ONNX model runs on
   OpenCV 5 DNN `ENGINE_NEW` with the same interface.
5. **Decide.** Escalate one level finer, centred on what it saw, when entropy
   > τ or the mask is below A_min; verify a new confident mask one level finer by
   mask IoU; log findings with length, width, area and orientation in pixels.
6. **Govern.** Before every step: kill switch, pixel / $ / seconds caps, p95
   latency against the SLA (switch to the fast model, then a coarser level),
   warp spikes, and a $ ≥ 80 % mode that verifies only critical classes.
7. **Stop** when the best remaining expected value per ms falls below a quarter
   of the realised value per ms so far.

Critical findings (exposed rebar; spalling above 40 000 px²) wait in
`pending_review`. Only a human — the web review or `drishyam review` — can
approve or reject them; nothing the agent, the MCP tools or the planner can call
reaches that transition.

## 3. Evidence that it is agentic

Every step's action is a logged function of an OpenCV output or a runtime
statistic, with a `cause`. A real trace (sample `bridge_03`, 30 % pixel cap, fixed clock):

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
| KILL during a live job on Vercel | kill sent during step 3 → step 4 halted with cause `kill-switch` (verified on the production deployment) |
| $ and pixel caps | the job halts inside one step with coverage and unvisited regions reported (`test_pixel_cap_is_never_exceeded`, `test_seconds_and_usd_caps_halt`) |
| Agent tries to approve its own critical finding | `ForbiddenTransition` (`test_agent_cannot_decide_or_downgrade`) |

## 4. Evaluation

Measured on **200 held-out labelled synthetic images** (seed 7; policy changes were tuned only on a separate 60-image dev split), one segmenter and one matching rule for every policy (a ground-truth box is found by a same-class component with IoU ≥ 0.3 or containing its centre).

| Policy | pixels processed (× image) | recall | $ / image (modelled) |
| --- | --- | --- | --- |
| downscale once (1024 px) | 0.25 | 0.50 | 0.000074 |
| tiling, L2 grid | 0.06 | 0.44 | 0.000019 |
| tiling, L1 grid | 0.50 | 0.55 | 0.000139 |
| tiling, L0 (exhaustive, 20 % overlap) | 1.67 | 0.55 | 0.000443 |
| **Drishyam**, cap 5 % | 0.02 | **0.00** (no 512² look fits after the overview on a 3 MP image) | 0.000005 |
| **Drishyam**, cap 10 % | 0.10 | **0.30** | 0.000029 |
| **Drishyam**, cap 20 % | 0.18 | **0.43** | 0.000053 |
| **Drishyam**, cap 30 % | 0.25 | **0.48** | 0.000072 |
| **Drishyam**, cap 50 % | 0.37 | **0.53** | 0.000102 |
| **Drishyam**, cap 100 % | 0.47 | **0.54** | 0.000129 |

- At a **30 % cap** Drishyam finds **0.48** of the defects using **0.25** of the image's pixels: **0.88×** exhaustive tiling's recall at **0.15×** its pixels, 0.97× a same-pixel downscale.
- At a **50 % cap**: **0.53** at 0.37 of the pixels = **0.97×** tiling's recall at **0.22×** its pixels. Measured against tiling's real pixel spend, the PRD hypothesis (≥ 0.9× recall at ≤ 0.3× pixels) holds from the 50 % cap; at the 30 % cap it is just missed (0.88×).
- Honest caveats: the best tiling grid (L1, 0.55 at 0.50) still beats Drishyam at equal pixels; the hysteresis segmenter shipped in this round lifted every baseline too (Drishyam/L0 at 30 %: 0.99× → 0.97× on dev).
- Critical false-accept rate (critical prediction with no critical ground truth): Drishyam @30 % 0.52 vs exhaustive tiling 0.79 (the PRD asks for ≤ tiling).
- Agent: 100 % of steps carry a cause; planner drills 38/40 recovered (2 malformed drills never fired because those jobs finished first); governor drills 10/10.

| per-class recall | crack | spalling | efflorescence | exposed_rebar | corrosion |
|---|---|---|---|---|---|
| Drishyam @30 % | 0.45 | 0.62 | 0.16 | 0.54 | 0.58 |
| tiling L0 | 0.47 | 0.70 | 0.25 | 0.52 | 0.75 |
| downscale | 0.48 | 0.57 | 0.22 | 0.44 | 0.66 |

Full numbers: [public/results/summary.json](../public/results/summary.json); failure gallery (12 real cases, 6 categories): [public/results/failures.json](../public/results/failures.json), also on the live site.

On 200 labelled synthetic images, Drishyam at a 30% pixel cap found 0.48 of ground-truth boxes using 0.25 of the image's pixels, versus 0.55 for 20%-overlap 512-px tiling using 1.67: 0.88x its recall (0.75x with one-box-per-class boxes, 1.25x with as-specified mixed boxes) at 0.15x its pixels, which does not meet the 0.9x-at-0.3x hypothesis. It is 0.97x a single 1024-px downscale (0.50 at 0.25 of the pixels), 0.87x L1 tiling (0.55 at 0.50, the best grid) and 1.09x L2 tiling (0.44 at 0.06), and 0.93x L0 tiling when a match needs IoU >= 0.3 (no contains-the-centre clause).

## 5. COOL on Graviton

The hot path uses only KleidiCV-accelerated ops (`pyrDown`, `warpPerspective`,
`resize`, `GaussianBlur`, `Sobel`, `morphologyEx`, `medianBlur`, `cvtColor`,
`inRange`, `dilate`/`erode`); no `filter2D`, no NumPy pixel loops on the gaze
path. `bench/run_all.sh <arm>` runs the micro, macro and DNN workloads with a
provenance block (OpenCV version, build-info hash, whether `cv2` loads from
`/opt/cool`) and `bench/parity.py` checks that two arms produce identical masks.
The three-arm run (COOL/c8g, stock/c8g, stock/c7i) needs the owner's AWS
account; [bench/README.md](../bench/README.md) is the runbook. The same code
passes CI on GitHub's free arm64 runners.

## 6. Human control, safety and responsible use

- Evidence-first UI: crop, measurements and gaze step before any confidence
  number; an "AI-assisted, not an inspection" banner on every finding.
- Pixels only. No millimetres without a scale reference; no safety ratings.
- Faces, people and vehicles are blurred (YuNet, NanoDet) before analysis or
  display; a photo mostly filled by a person is refused.
- The planner sees a numeric digest only. Threat model and tests:
  [security.md](security.md).

## 7. Cost

Spent so far: **$0.** Vercel Hobby (the live demo), GitHub Actions (CI on x86
and arm64), local CPU for evaluation. The $ in the UI is *modelled* at the
c8g.2xlarge + COOL rate ($0.339/h) so the curves read as a buyer's cost.

## 8. Limits and next steps

- Numbers above come from a **synthetic** dataset and the **classical**
  segmenter. Next: fine-tune the dacl10k model on Kaggle (`training/`), then
  `make data && make eval` on 200 CODEBRIM originals.
- The COOL three-arm benchmark and the DynamoDB / SNS / CDK stack need AWS.
- The web demo is stateless; the signed `work_order.json` is its audit trail.

## 9. Reproduce

```bash
uv sync --all-extras && make test && make eval && make bench
```
