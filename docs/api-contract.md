# Drishyam web API contract

The browser UI (`public/`) talks to the Python function (`app.py` → `drishyam/web.py`) through
four endpoints. Every response is JSON or NDJSON. All coordinates are **L0
pixels** (full-resolution image), `[x, y, w, h]` boxes have their origin at the
top-left. All measurements are pixels; there are no millimetres anywhere.

Behind CloudFront with Origin Access Control (the AWS deployment), every `POST` must also carry
`x-amz-content-sha256`: the hex SHA-256 of the exact request body. Lambda refuses unsigned payloads
through OAC with a 403. On Vercel or locally the header is ignored.

## `GET /api/health`

```json
{"ok": true, "version": "1.0.2", "opencv": "5.0.0", "build": "stock (not COOL)", "segmenter": "classical-v1",
 "segmenter_licence": "MIT (no learned weights)", "max_upload_bytes": 4400000, "max_pixels": 25000000,
 "host": "vercel", "arch": "x86_64", "commit": "7a93ec5..."}
```

`host` is `aws-lambda`, `vercel` or `local`; `arch` is `platform.machine()` (`x86_64`, `aarch64`,
`arm64`); `commit` is `VERCEL_GIT_COMMIT_SHA` or `DRISHYAM_COMMIT` (an image build argument), else `null`.
`segmenter_licence` is `CC BY-NC 4.0, trained on dacl10k` when the dacl10k-trained ONNX model is loaded.

## `POST /api/jobs`

Body: the raw image bytes (`Content-Type: image/jpeg` or `image/png`), at most
`max_upload_bytes`. The browser re-encodes larger files before upload.

Query parameters (all optional, validated server-side, out-of-range → 422):

| name | range | default | meaning |
| --- | --- | --- | --- |
| `pixel_cap` | 0.02 – 1.0 | 0.30 | fraction of full-resolution pixels the job may process |
| `usd_cap` | 0.0001 – 0.05 | 0.05 | modelled $ cap (wall time × $0.339/h) |
| `seconds_cap` | 1 – 30 | 20 | wall-clock cap |
| `chaos` | comma list of `p95_spike`, `planner_timeout`, `planner_malformed` | — | failure drills |
| `pace_ms` | 0 – 1500 | 0 | wait this long after each look, for viewing (at most 30 s of waiting per job); the job's clock excludes the wait, so `seconds_used`, `usd_used` and the caps count compute only |

With `pace_ms`, a look's `finding` and `review_request` events still arrive right after its `step`
record; the wait comes after them, and a KILL posted during the wait halts the job before its next look.

Optional header `X-Drishyam-Replay`: base64url JSON list of earlier decisions
`[["gaze", x, y, level, size], ...]` to resume a killed job (max 200 entries; `size` is the
step's `decision.args.size`, optional, 256-512, so a look the governor had shrunk replays at
the same size).

Response: `200`, `Content-Type: application/x-ndjson`, one JSON object per line,
streamed while the agent runs. Errors before the stream starts are plain JSON
`{"error": "..."}` with 400 (bad image), 413 (too large), 422 (bad params or a
people-dominant image).

Event types, in order:

```jsonc
{"type": "job", "job": "j_3f9a1c", "width": 3000, "height": 2250,
 "levels": [[3000, 2250], [1500, 1125], [750, 563], [375, 282], [188, 141]],
 "budget": {"pixel_cap_frac": 0.3, "pixels_cap": 2025000, "usd_cap": 0.05, "seconds_cap": 20},
 "privacy": {"faces_blurred": 0, "people_blurred": 0, "vehicles_blurred": 0, "boxes": [[x, y, w, h]]},
 "kill_token": "<24 url-safe chars; POST it to /api/kill to stop this job>"}

{"type": "overview", "heatmap_png_b64": "<PNG, saliency 0-255, size of level 3>",
 "candidates": [{"id": "r1", "box": [1200, 640, 768, 768], "prior": 0.81}],
 "pixels_used": 84375, "pixels_frac": 0.0125}

{"type": "step", "job": "j_3f9a1c", "step": 3, "t": "2026-10-24T17:02:11Z",
 "region": "r2",
 "decision": {"action": "gaze", "args": {"x": 2210, "y": 1480, "level": 1, "yaw": 0, "pitch": 0, "size": 512}},
 "window": [1698, 968, 1024, 1024],
 "observation": {"entropy": 0.62, "saliency_max": 0.81, "mask_area_px": 41, "sharpness": 212.4,
                 "exposure": 0.48, "classes": {"crack": 41}},
 "quality": {"ok": true, "reason": null},
 "verify": {"iou": 0.71, "against_level": 2},          // only on verify / escalate steps
 "runtime": {"p50_ms": {"warp": 9, "segment": 31}, "p95_ms": {"warp": 14, "segment": 54},
             "cpu_pct": 71, "usd": 0.0031},
 "rationale": "entropy 0.62 > 0.5 at L2",
 "cause": "entropy>tau", "planner": "rule", "model": "accurate",
 "governor": {"cause": "p95>SLA", "causes": ["p95>SLA"],              // only when the governor overrode
              "rationale": "p95(segment) 900 ms > SLA 500 ms; model -> fast"},   // a setting before this look
 "planner_status": {"status": "rejected", "reason": "strategy 'rm -rf /' not in allow-list"},
                                                        // only on a step where the planner was consulted;
                                                        // "strategy" is added when a proposal was accepted
 "latency_ms": 52, "cost_usd": 0.000005,
 "pixels_used": 870000, "pixels_frac": 0.129, "usd_used": 0.00021, "seconds_used": 2.2,
 "injected": null,                                      // or "p95_spike"
 "crop_jpeg_b64": "<160 px JPEG thumbnail of the gaze crop>"}

{"type": "finding", "finding": {"id": "f1", "cls": "crack", "critical": false, "state": "draft",
  "confidence": 0.74, "measure": {"length_px": 812.0, "width_px": 3.1, "area_px": 2530, "orientation_deg": 63.0},
  "box": [2010, 1300, 420, 380], "step": 3, "steps": [3, 4], "level": 0,
  "critical_step": null, "critical_area_px": null, "critical_rule": null,
  "verify_iou": 0.71, "verified": true, "crop_jpeg_b64": "<256 px JPEG>"}}

{"type": "review_request", "finding_id": "f2", "reason": "exposed_rebar is critical"}

{"type": "planner", "step": 5, "status": "fallback", "reason": "timeout", "proposal": null}
// status: accepted | rejected | fallback

{"type": "done", "status": "awaiting_review",           // completed | halted | awaiting_review
 "cause": "stop-rule",                                  // stop-rule | exhausted | cap-hit | kill-switch | planner:stop
 "summary": {"steps": 14, "pixels_used": 1950000, "pixels_frac": 0.289, "usd": 0.0008,
             "seconds": 8.4, "findings": 3, "critical_pending": 1},
 "coverage": 0.62, "unvisited": ["r9", "r10"],
 "work_order": { /* see below */ }, "token": "<signed work order>"}

{"type": "error", "message": "..."}
```

Severity classes: `crack`, `spalling`, `efflorescence`, `exposed_rebar`,
`corrosion`. Critical = `exposed_rebar` always, `spalling` above
`40000` px² area. Finding states: `draft`, `pending_review`, `approved`,
`rejected`. `critical_rule` says which rule made a finding critical, with the area the
`critical_step` look measured: `"exposed_rebar: always critical"` or
`"spalling 48,568 px² >= 40,000 px²"` (`null` while it is not critical).

`verify_iou` is the mask IoU of the latest cross-level check of the finding (a finer look
against the coarser look before it) and `verified` is `verify_iou >= 0.5`; both are `null`
when no check compared two looks of it. `verified: false` means unconfirmed: the looks disagreed.

A `review_request` asks a person to look; it changes no state (only a critical finding is
`pending_review`). `reason` is one of:

- `"<class> is critical"`: a new critical finding;
- `"iou 0.19 < 0.5 between L3 and L2"`: a critical finding whose finer look disagrees with the coarser one;
- `"unconfirmed: iou 0.07 < 0.5 between L1 and L0"`: a disagreement no further look will settle (no finer
  level, or the region's one extra look is spent); the finding has `verified: false`;
- `"ambiguous: crack 5232 px (>= A_min 64) lies inside spalling f2; recorded as spalling only"`: two classes
  cleared `A_min` on one spot and the agent kept the dominant one (once per finding and class).

A finding event repeats, with the same `id`, whenever another look sees the same
defect. The finding is its evidence trail: `step` is the look that found it,
`steps` every look that saw it in order, `box` the union of those looks' boxes,
`confidence` the highest, and `measure`, `level` and the crop come from the
finest look. `critical` never goes back to `false`; `critical_step` is the look that
made the finding critical and `critical_area_px` the area that look measured (both `null`
while it is not critical), since that look may be coarser than the one `measure` comes from. A look whose `window` runs past the
image (a survey wider than the image, a level-1 look at a small photo) is filled by
reflection for the segmenter, but those pixels count as background: no label, area,
entropy or box ever comes from outside the image, and every finding `box` lies inside it.

A look's `cause` is always the reason for that look itself, an OpenCV measurement or a rule.
A governor override (the model, level or gaze size changed before the look) is written to the
step's `governor` object, `{"cause": "p95>SLA", "causes": ["p95>SLA"], "rationale": "..."}` (`cause` is
the first override when several were pending), and a planner consultation to its `planner_status`;
neither replaces `cause`. The last `step` record has `decision.action` `halt` and the halt `cause`;
when the governor changed a setting on that same step before the job stopped (say `p95>SLA`
switched the model, then the stop rule ended the job), the halt record carries the same `governor`
object. Every `best-score` rationale ends with the stop threshold it beat:
`"argmax score 4.10e-03: prior 0.81, unseen 0.92, t_hat 61 ms; lambda 3.04e-03"`.

Step `decision.action` values: `gaze` (first look at a region, or the survey), `escalate`
(one level finer, same centre, because the last look was uncertain), `verify`
(one level finer to confirm a mask), `halt`. Look `cause` values:
`survey`, `best-score`, `entropy>tau`, `area<A_min`, `verify-new-mask`, `iou<0.5`,
`planner:<strategy>`, `replay`, and `random-order` (only under the `order="random"` ablation).
`governor.cause` values: `p95>SLA`, `warp-spike`, `usd≥80%`. Halt `cause` values: `stop-rule`,
`exhausted`, `cap-hit`, `kill-switch`, `planner:stop`. `planner_status.status` is `accepted`,
`rejected` or `fallback` (a timeout or a network error), as on the `planner` event. A planner's
`change_level` / `reverify` queues its own look, which carries `planner:<strategy>`; the look already
chosen on that step keeps its own cause. A `switch_model: accurate` proposal is rejected while a
`p95>SLA` downgrade stands. `planner` is `llm` on a look a planner chose or whose model or batching
it changed, else `rule`. No LLM planner is configured on the live demo: there `planner` is always
`rule`, and `planner_status` appears only in the failure drills, whose planners are scripted to fail.

## `POST /api/kill`

```json
{"job": "j_3f9a1c", "kill_token": "<from the job event>"}
```

Sets a shared flag (TTL 5 min) that the running job reads after every step; the governor then
halts before the next look with cause `kill-switch`. The flag lives in the regional Vercel Runtime
Cache, or in the DynamoDB table `DRISHYAM_KILL_TABLE` names on AWS (see [aws.md](aws.md)); neither
platform passes the browser's disconnect through to the function, so the flag is the only path.
A wrong token is accepted and kills nothing (no oracle). Returns `{"ok": true, "job": "..."}`;
400 on malformed ids.

## Work order

```json
{"id": "wo_3f9a1c", "job": "j_3f9a1c", "created": "2026-10-24T17:02:19Z",
 "status": "blocked",
 "findings": [{"id": "f1", "cls": "crack", "critical": false, "state": "draft", "confidence": 0.74,
               "measure": {"length_px": 812.0, "width_px": 3.1, "area_px": 2530, "orientation_deg": 63.0},
               "box": [2010, 1300, 420, 380], "step": 3, "steps": [3, 4], "level": 0,
               "critical_step": null, "critical_area_px": null, "critical_rule": null,
               "verify_iou": 0.71, "verified": true}],
 "gaze_path": [[2210, 1480, 1], [2210, 1480, 0]],
 "not_assessed": [{"step": 4, "box": [0, 1024, 1024, 1024], "reason": "glare", "action": "re-shoot"}],
 "approvals": [{"finding_id": "f2", "decision": "approve", "reviewer": "A. Engineer", "at": "2026-10-24T17:05:00Z"}],
 "disclaimer": "AI-assisted, not an inspection. Measurements are image pixels, not millimetres."}
```

`status` is `blocked` while any critical finding is `pending_review`, else `ready`.
`not_assessed` lists every window the quality gate refused (`reason` `blur`, `glare` or `dark`;
`box` clipped to the image), so an area the agent could not read never passes for a clean one.

### Downloaded work-order file

The file the page saves keeps the signature with the order, so the copy can be checked later:

```json
{"work_order": { /* as above */ }, "token": "<signed work order>", "alg": "HMAC-SHA256", "exp": 1793000000}
```

`token` is `<body>.<mac>`, both base64url without padding: `body` is the JSON
`{"exp": <unix seconds>, "wo": <work order>}` and `mac` its HMAC-SHA256 under the server key. `exp`
repeats the body's expiry (7 days after signing). A copy whose `work_order` was edited no longer
matches its token, and `/api/review` refuses an edited token with 403.

## `POST /api/review`

```json
{"token": "<signed work order>", "finding_id": "f2", "decision": "approve", "reviewer": "A. Engineer"}
```

`decision` is `approve` or `reject`; `reviewer` is 1–64 printable characters.
Returns `{"token": "<new signed work order>", "work_order": {...}}`.
Errors: 400 malformed, 403 bad or expired signature, 409 finding not pending.
The agent has no route to this transition: only this human endpoint (or the
`drishyam review` CLI) can approve or reject a critical finding.
