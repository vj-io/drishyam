# Drishyam web API contract

The browser UI (`public/`) talks to the Python function (`app.py` → `drishyam/web.py`) through
four endpoints. Every response is JSON or NDJSON. All coordinates are **L0
pixels** (full-resolution image), `[x, y, w, h]` boxes have their origin at the
top-left. All measurements are pixels; there are no millimetres anywhere.

## `GET /api/health`

```json
{"ok": true, "opencv": "5.0.0", "build": "stock (not COOL)", "segmenter": "classical-v1",
 "max_upload_bytes": 4400000, "max_pixels": 25000000}
```

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

Optional header `X-Drishyam-Replay`: base64url JSON list of earlier decisions
`[["gaze", x, y, level], ...]` to resume a killed job (max 200 entries).

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
 "rationale": "entropy 0.62 > 0.5 at L2; area 41 px < 64",
 "cause": "entropy>tau", "planner": "rule", "model": "accurate",
 "latency_ms": 52, "cost_usd": 0.000005,
 "pixels_used": 870000, "pixels_frac": 0.129, "usd_used": 0.00021, "seconds_used": 2.2,
 "injected": null,                                      // or "p95_spike"
 "crop_jpeg_b64": "<160 px JPEG thumbnail of the gaze crop>"}

{"type": "finding", "finding": {"id": "f1", "cls": "crack", "critical": false, "state": "draft",
  "confidence": 0.74, "measure": {"length_px": 812.0, "width_px": 3.1, "area_px": 2530, "orientation_deg": 63.0},
  "box": [2010, 1300, 420, 380], "step": 3, "level": 1, "crop_jpeg_b64": "<256 px JPEG>"}}

{"type": "review_request", "finding_id": "f2", "reason": "exposed_rebar is critical"}

{"type": "planner", "step": 5, "status": "fallback", "reason": "timeout", "proposal": null}
// status: accepted | rejected | fallback

{"type": "done", "status": "awaiting_review",           // completed | halted | awaiting_review
 "cause": "stop-rule",                                  // stop-rule | exhausted | cap-hit | kill-switch
 "summary": {"steps": 14, "pixels_used": 1950000, "pixels_frac": 0.289, "usd": 0.0008,
             "seconds": 8.4, "findings": 3, "critical_pending": 1},
 "coverage": 0.62, "unvisited": ["r9", "r10"],
 "work_order": { /* see below */ }, "token": "<signed work order>"}

{"type": "error", "message": "..."}
```

Severity classes: `crack`, `spalling`, `efflorescence`, `exposed_rebar`,
`corrosion`. Critical = `exposed_rebar` always, `spalling` above
`40000` px² area. Finding states: `draft`, `pending_review`, `approved`,
`rejected`.

Step `decision.action` values: `gaze` (first look at a region, or the survey), `escalate`
(one level finer, same centre, because the last look was uncertain), `verify`
(one level finer to confirm a mask), `halt`. Trace `cause` values:
`survey`, `best-score`, `entropy>tau`, `area<A_min`, `verify-new-mask`, `iou<0.5`,
`p95>SLA`, `warp-spike`, `usd≥80%`, `cap-hit`, `kill-switch`,
`planner-fallback`, `planner:<strategy>`, `replay`.

## `POST /api/kill`

```json
{"job": "j_3f9a1c", "kill_token": "<from the job event>"}
```

Sets a flag in the regional Vercel Runtime Cache (TTL 5 min) that the running job
reads after every step; the governor then halts before the next look with cause
`kill-switch`. A wrong token is accepted and kills nothing (no oracle). Returns
`{"ok": true, "job": "..."}`; 400 on malformed ids.

## Work order

```json
{"id": "wo_3f9a1c", "job": "j_3f9a1c", "created": "2026-10-24T17:02:19Z",
 "status": "blocked",
 "findings": [{"id": "f1", "cls": "crack", "critical": false, "state": "draft", "confidence": 0.74,
               "measure": {"length_px": 812.0, "width_px": 3.1, "area_px": 2530, "orientation_deg": 63.0},
               "box": [2010, 1300, 420, 380], "step": 3, "level": 1}],
 "gaze_path": [[2210, 1480, 1], [2210, 1480, 0]],
 "approvals": [{"finding_id": "f2", "decision": "approve", "reviewer": "A. Engineer", "at": "2026-10-24T17:05:00Z"}],
 "disclaimer": "AI-assisted, not an inspection. Measurements are image pixels, not millimetres."}
```

`status` is `blocked` while any critical finding is `pending_review`, else `ready`.

## `POST /api/review`

```json
{"token": "<signed work order>", "finding_id": "f2", "decision": "approve", "reviewer": "A. Engineer"}
```

`decision` is `approve` or `reject`; `reviewer` is 1–64 printable characters.
Returns `{"token": "<new signed work order>", "work_order": {...}}`.
Errors: 400 malformed, 403 bad or expired signature, 409 finding not pending.
The agent has no route to this transition: only this human endpoint (or the
`drishyam review` CLI) can approve or reject a critical finding.
