# Drishyam web API contract

The browser UI (`public/`) talks to the Python function (`drishyam/web.py`, on AWS Lambda) through
four endpoints. Every response is JSON or NDJSON. All coordinates are **L0
pixels** (full-resolution image), `[x, y, w, h]` boxes have their origin at the
top-left. All measurements are pixels; there are no millimetres anywhere.

Behind CloudFront with Origin Access Control (the AWS deployment), every `POST` must also carry
`x-amz-content-sha256`: the hex SHA-256 of the exact request body. Lambda refuses unsigned payloads
through OAC with a 403. Locally the header is ignored. When one of AWS WAF's per-IP rate rules blocks a
request, CloudFront answers `429` with `Content-Type: application/json` and a body such as
`{"error": "rate limited: 10 job starts per 2 min per IP; wait and retry"}`. Lambda also answers 429, with
its own body (do not count on an `error` field), when every function environment is busy (docs/aws.md, "Limits"). The
app itself never sends 429.

## `GET /api/health`

```json
{"ok": true, "version": "1.1.0", "opencv": "5.0.0", "build": "stock (not COOL)", "segmenter": "classical-v1",
 "segmenter_licence": "MIT (no learned weights)", "max_upload_bytes": 4400000, "max_pixels": 25000000,
 "host": "local", "arch": "x86_64", "commit": null,
 "cv2_build": {"hal": "YES (ipp (ver 0.0.1))", "cpu_baseline": "SSE SSE2 SSE3 SSSE3 SSE4_1",
               "cpu_dispatch": "SSE4_2 AVX FP16 AVX2 AVX512_SKX", "parallel": "GCD",
               "onnx_runtime": "NO", "dnn_engines": ["auto", "classic", "new"]}}
```

`cv2_build` is read from `cv2.getBuildInformation()` once per process: the `Custom HAL`, CPU `Baseline` and
`Dispatched code generation`, `Parallel framework` and `ONNX Runtime` lines as printed (`null` when the build prints no such line), and the
`cv2.dnn` engines the build can run: the `ENGINE_*` names `cv2.dnn` exposes, lower-cased, without `ort` unless the
`ONNX Runtime` line says `YES` (the constant exists whether or not that backend is built in). The opt-in ONNX
segmenter uses `new`. The example was measured on the macOS x86_64 wheel. The deployed
Lambda's values (an aarch64 wheel, where ARM HALs such as carotene or KleidiCV are listed on the HAL line) are what
its own `/api/health` returns; none are copied here.

`host` is `aws-lambda` or `local`; `arch` is `platform.machine()` (`x86_64`, `aarch64`,
`arm64`); `commit` is `DRISHYAM_COMMIT` (`scripts/aws.sh deploy` sets it to the git SHA it deployed), else `null`.
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
| `pace_ms` | 0 – 1500 | 0 | wait this long after each look, for viewing (at most 25 s of waiting per job); the job's clock excludes the wait, so `seconds_used`, `usd_used` and the caps count compute only |

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
 "prep_ms": {"read": 6, "decode": 84, "privacy": 60},
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
 "verify": {"iou": 0.71, "class": "crack", "against_level": 2, "class_now": "crack", "iou_any": 0.71},
                                                        // only on a look at a region an earlier look saw a class in
 "seeded": [{"region": "r4", "entropy": 0.41, "cls": "crack", "from_level": 2}],
                                                        // only when this look set unvisited candidates' uncertainty
                                                        // and class (cls null: no class there)
 "rejected": [{"cls": "crack", "cause": "straight line: formwork or joint", "box": [2232, 1250, 20, 320],
               "length_px": 318.2, "width_px": 6.56, "residual_px": 1.26}],
                                                        // only when the look dropped crack components that are
                                                        // one straight line (centreline RMS residual < 1.5 L0 px
                                                        // over >= 20x the mean width), or its crack measured as a
                                                        // blob ("not elongated: pore or bug hole": length < 8x the
                                                        // mean width, no residual_px); no finding comes from them
 "runtime": {"p50_ms": {"warp": 9, "segment": 31}, "p95_ms": {"warp": 14, "segment": 54},
             "cpu_pct": 71, "usd": 0.0031},
 "rationale": "entropy 0.62 > 0.5 at L2",
 "cause": "entropy>tau", "planner": "rule", "model": "accurate", "profile": "classical-v1",
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

{"type": "review_request", "finding_id": "f2", "kind": "critical", "reason": "exposed_rebar is critical",
 "iou": null, "level_a": null, "level_b": 1, "classes": ["exposed_rebar"]}
// kind: critical | ambiguous | disagree | class-change | unconfirmed

{"type": "planner", "step": 5, "status": "fallback", "reason": "timeout", "proposal": null}
// status: accepted | rejected | fallback

{"type": "done", "status": "awaiting_review",           // completed | halted | awaiting_review
 "cause": "stop-rule",                                  // stop-rule | exhausted | cap-hit | kill-switch | planner:stop
 "summary": {"steps": 14, "pixels_used": 1950000, "pixels_frac": 0.289, "usd": 0.0008,
             "seconds": 8.4, "findings": 3, "critical_pending": 1},
 "coverage": 0.62, "unvisited": ["r9", "r10"],
 "drills_not_reached": ["planner_malformed"],          // requested chaos drills whose fixed step (timeout 3,
                                                        // spike 4, malformed 6) the job never reached: they tested
                                                        // nothing; [] when every requested drill fired
 "work_order": { /* see below */ }, "token": "<signed work order>"}

{"type": "error", "message": "..."}
```

`prep_ms` on the `job` event is the server's time before the job's clock starts, in whole milliseconds:
`read` (receiving the upload), `decode` (`imaging.decode_image`) and `privacy` (the face, person and
vehicle blur, `privacy.privacy_pass`). None of it is in a step's `latency_ms`, in `seconds_used`,
`usd_used`, the summary's `seconds` and `usd`, or in any cap. The first job a fresh server process runs
(on Lambda, the first after a cold start) also loads the two privacy models inside `privacy`. The job's
log line repeats `prep_ms` (on Lambda as a field of its EMF record, not as a metric). The `prep_ms` in the
example is from one local run of `public/samples/bridge_03.jpg` (3000x2250) on a warm process.

Severity classes: `crack`, `spalling`, `efflorescence`, `exposed_rebar`,
`corrosion`. Critical = `exposed_rebar` always, `spalling` above
`40000` px² area, or a finding a class change put one of those on (see `class-change` below). Finding states:
`draft`, `pending_review`, `approved`, `rejected`. `critical_rule` says which rule made a finding critical, with
the area the `critical_step` look measured: `"exposed_rebar: always critical"`,
`"spalling 48,568 px² >= 40,000 px²"` or `"class change: exposed_rebar at L1 here (exposed_rebar: always
critical)"` (`null` while it is not critical).

A step's `verify` compares this look with the last coarser look at the same region that saw a class, class by
class: `class` is that coarser look's best class, `iou` the IoU of its `class` mask with this look's mask of the
**same** class (0 when the class is gone here), `class_now` this look's best class (`null`: it sees no defect) and
`iou_any` the geometric IoU of the two looks' best masks whatever their classes (equal to `iou` when the class
is the same). Masks are compared in the finer look's frame with a 2-px tolerance. There is no `verify` when no
pixel of the coarser mask lies in this look: nothing was compared, whatever this look shows elsewhere. The
check reads one of three ways: agreement (`iou >= 0.5`); a **class change** (`iou < 0.5` but `iou_any >= 0.5`:
the finer look sees the same place as another class, so it keeps that class and the coarser one is recorded in
`verify.class`); or a geometric disagreement (both below 0.5). A finer look that sees no defect at all
(`class_now` `null`, `iou` 0) is a disagreement too: at a level above L0 it spends the region's one extra look.

`verify_iou` on a finding is that same-class `iou` from the latest check of the class it holds, and `verified` is
`verify_iou >= 0.5`; both are `null` when no check compared two looks of it. Only a finding of the compared class
takes a check: the one this look drafted that class into, else the one the coarser look drafted it into (the
class is gone here, under `A_min`, or absorbed into another finding), which then repeats its finding event with
the low IoU. A finding that absorbed the compared class (rust on its bars, a crack in a spall's rim) holds another
class, so the check leaves it as it was, and no disagreement names it; the check stays in the step's `verify`.
`verified: false` means unconfirmed: the two looks of its class disagreed, or the finer one saw another class.

A `review_request` asks a person to look; the request itself changes no state (only a critical finding is
`pending_review`, and a `class-change` request that involves a critical class names a critical finding). Its
`kind` and `reason` are one of:

- `critical`, `"<class> is critical"`: a finding became critical on this look (new, or a later look measured it
  over the area threshold);
- `disagree`, `"iou 0.19 < 0.5 between L3 and L2"`: the same class's masks disagree across the two looks and the
  finding of that class is critical (it then has `verified: false`);
- `class-change`, `"class change: spalling at L1, corrosion at L0 (same place, iou 0.62)"`: the finer look sees the
  same place as another class (`iou` here is `iou_any`) and either CLASS is critical (exposed rebar at any area,
  spalling over the area threshold, its largest component measured as a finding's is, on the look that saw it)
  or either finding is. It names the critical finding, else the coarser finding, else the finer one: a coarser
  look that escalated (its mask under `A_min`) drafted nothing, and the request then names the finer look's
  finding, whose own class no check has compared (its `verify_iou` stays `null`). When the finding it names is not
  critical, the agent first makes it critical (`pending_review`, a `finding` event, `critical_rule` `"class change:
  exposed_rebar at L1 here (exposed_rebar: always critical)"`), so the work order is `blocked` and the job
  `awaiting_review` until a human approves or rejects it. A class change with nothing critical, or with no finding
  at all (both masks under `A_min`), sends no request: the step's `verify` records it and the coarser finding has
  `verified: false`;
- `unconfirmed`, `"unconfirmed: iou 0.07 < 0.5 between L1 and L0"`: a disagreement no further look will settle
  (no finer level, the region's one extra look is spent, or this look logged a critical finding, so a human
  reviews the region anyway and no look is spent on it); the finding has `verified: false`;
- `ambiguous`, `"ambiguous: crack 5232 px (>= A_min 64) is mostly inside spalling f2; recorded as spalling only"`:
  two classes cleared `A_min` on one spot and the agent kept the dominant one (once per finding and class).

The same in fields, so a client need not parse `reason`: `level_b` is the level of the look that asked; for
`disagree`, `unconfirmed` and `class-change`, `level_a` is the coarser look it was compared with, `iou` the IoU
named in the reason, and `classes` is `[class at level_a, class at level_b]` (for the two disagreements both are
the compared class); for an ambiguity `classes` is `[kept, absorbed]`, for a new critical finding `[class]`, with
`iou` and `level_a` `null`. (The MCP server's records carry `reason` only.)

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
object. Every `best-score` rationale names each term of the score and ends with the stop threshold it beat:
`"argmax score 4.10e-03: prior 0.81, severity 1.5 (crack seen at L2), uncertainty 0.62, unseen 0.92, t_hat 61 ms;
lambda 3.04e-03"`. The score of an unvisited candidate is prior x severity x (1 + uncertainty) x unseen / t_hat:
`severity` and `uncertainty` come from the finest look that showed the candidate (the latest of equals) and
passed the quality gate: its `seeded` entry, the most severe class whose area inside the candidate's first-look
window would clear `A_min` at the candidate's first-look level, and the binary damage-vs-background entropy over
that window. A look shows a candidate only when it sees at least half of that window (a look one level finer
covers a quarter of it at most, so it does not); the
rationale says `(crack seen at L2)`, `(no class seen at L2)`, or `(not seen yet)` when no look has shown it
(severity 1, uncertainty 0; `(seeding off)` under the no-seed ablation). The stop rule halts when the best score is below
lambda = max(eps x value found / ms spent, eps x v0 / t_hat), with eps 0.25 and v0 2.0: the second term is the
floor, the value of a look that finds nothing, so a photo with nothing to find can stop before its cap. The floor
holds only once every candidate still worth a look (unseen > 0) has been shown by a look at least as fine as its
own first look (L1 by default): a coarser look (the survey, L2 or L3) cannot resolve what that look would, a
hairline crack for one, so its seed orders candidates but never ends a job. A job therefore never stops on a saliency prior and a coarse view alone,
nor before it has seen anything; on a photo whose candidates lie apart the floor rarely binds and the cap ends it. A `stop-rule` halt names the binding term:
`"best score 1.2e-04 < lambda 2.0e-04 (floor: eps x v0 / t_hat)"` or `"... (eps x value/cost)"`.

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

`profile` (on every step record, the halt record included, and on the work order) names the per-segmenter
settings the job ran with, `drishyam.agent.PROFILES`, looked up by the segmenter's name (`segmenter` in
`/api/health`) and weights: a name not listed, or an ONNX file whose sha256 is not the one its profile was chosen
with (`drishyam.agent.WEIGHTS`, e.g. a retrained model installed under the same file name), gets `classical-v1`.
`min_view` is the smallest share of the photo's long side a look's window may span, which sets the level of a first
look and the finest level an `escalate` or `verify` look may go to; `absorb` is whether a crack or rust mostly inside
a spall's or bar's finding joins it (the `ambiguous` request). `classical-v1`: `min_view` 0 (first looks at L1, zoom
down to L0) and absorption on, the agent as published. `onnx:drishyam_dacl10k.onnx`, the dacl10k-trained model
(sha256 `c49416c10087...`): `min_view` 0.3 and absorption off, so no `ambiguous` request. Its finest level is the
finest whose 512-px window spans at least 0.3 of the long side: L0 up to 1706 px, L1 up to 3413 px (L0 of a 2048-px
photo spans 0.25), L2 up to 6826 px (L1 of a 4608-px photo spans 0.22, of a 3648-px one 0.28); on a 4608-px photo
first looks are at L2, after the L3 survey.

A look at the finest level whose entropy is above tau or whose mask is under `A_min` does not escalate. Its
rationale ends with what it saw and what it drafted: `"; L2 is the finest level for onnx:drishyam_dacl10k.onnx
(min_view 0.3): entropy 0.71 > 0.5, crack drafted here; no finer look"`, `"...: crack area 40 px < A_min 64, crack
not drafted; no finer look"` (a mask under `A_min` is never drafted) or `"...: entropy 0.96 > 0.5, no class; no finer
look"`. Nor does a verify look go below the finest level: a mask first drafted there is not confirmed one level
finer, so unless another look's check measures it its finding keeps `verify_iou` `null`, and the look's rationale
ends `"; new crack mask not verified: L2 is the finest level for onnx:drishyam_dacl10k.onnx (min_view 0.3), no finer
look"`. A disagreement with a coarser look there gets no extra look (`"; iou 0.31 < 0.5 between L3 and L2 not
settled: ..."`); the finding of its class, if any, gets the `unconfirmed` request. On CODEBRIM dev100 at the 30 % cap
this profile made 45 verify looks, all at L2 confirming masks the L3 survey drafted, and none after its 396 first
looks; 72 of its 661 findings have a `verify_iou`, against 302 verify looks and 417 of 926 findings with the same
model under `classical-v1` (`verify` in `results/dev/onnx_profile_sweep.json`). A verify at the same level (a
shifted window) was not built or measured.

A planner `change_level` finer than the finest level is clamped to it, and its rationale says so. Two kinds of look
are exempt from `min_view`: a replayed look (`X-Drishyam-Replay`) runs at its recorded level, and a gaze the
governor's `warp-spike` override shrank (512, 384, 288, then 256 px) does not move the finest level, so on a
3000-px photo an L1 look at 384 px spans 0.256 of the side. The dev runs had neither (no replay, governor off).
The ONNX values were chosen on the dev splits only (`results/dev/onnx_profile_sweep.json`).

Three `JobConfig` knobs from the tuning dev sweep (`results/dev/tune_sweep.json`) are off by default: measured, not
adopted. `tau_scout`, when set, is the entropy threshold of a look at L2 or coarser; `tau` (0.5) stays that of every
finer look, and an `entropy>tau` rationale names the threshold that applied (`"entropy 0.74 > 0.7 at L2"`). Unset
(every profile), `tau` applies at every level. With 0.7 a survey whose entropy is above 0.5 and at most 0.7 drafts
its classes and is checked one level finer (`verify-new-mask`) instead of escalating (`entropy>tau`) with nothing
drafted. At the 30 % cap that found x1.03 the recall on synth_dev (95 % CI 0.98-1.09, not a measurable gain) and
x1.22 on CODEBRIM dev100 (1.09-1.43): 0.153 to 0.186, where the same boxes at random positions find 0.088 and 0.099
(mean of 30 draws), IoU-only 0.066 to 0.086 against 0.019 and 0.023. It also sent 242 critical boxes to the human
gate on CODEBRIM dev100 instead of 178 (239 and 176 matching no ground-truth box), put 12.40 boxes on a clean photo
instead of 12.13 (8 more boxes, on 2 photos of 30) and sat within 0.003 of a clean dev survey's entropy (any value
in [0.692, 0.703) gives the same dev rows). The sweep's rule asked for a recall gain on synth_dev without more clean
false alarms: the gain is inside its CI and the clean count rose. The other two: `fill`, under which a
look that no longer fits under the pixel cap shrinks to what is left (a multiple of 32 px, at least 256) instead of
the job halting, with that size in `decision.args.size` and a rationale ending `"; gaze 320 px: what the pixel cap
leaves (108134 px)"` (synth_dev recall x1.03, clean boxes x1.40); and `survey_value`, which when False keeps the
survey's finds out of the stop rule's value found per ms (no measurable effect).

## `POST /api/kill`

```json
{"job": "j_3f9a1c", "kill_token": "<from the job event>"}
```

Sets a shared flag (TTL 5 min) that the running job reads after every step; the governor then
halts before the next look with cause `kill-switch`. On AWS the flag lives in the DynamoDB table
`DRISHYAM_KILL_TABLE` names (partition key `k`, a string; numeric TTL attribute `exp`, unix seconds), in
a local run in the server process; a Lambda Function URL does not pass the browser's disconnect through
to the function, so the flag is the only path.
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
 "approvals": [{"finding_id": "f2", "decision": "approve", "reviewer": "A. Engineer", "at": "2026-10-24T17:05:00Z",
                "prev_token_sha256": "<SHA-256 hex of the token this decision was applied to>"}],
 "disclaimer": "AI-assisted, not an inspection. Measurements are image pixels, not millimetres.",
 "profile": "classical-v1", "weights_sha256": null}
```

`status` is `blocked` while any critical finding is `pending_review`, else `ready`. `profile` is the job's (see the
step record) and `weights_sha256` the sha256 of the ONNX weights that made its looks (`null` for the classical
engine); a work order the MCP tools draft has neither, since there the client chose every level.
`not_assessed` lists every window the quality gate refused (`reason` `blur`, `glare` or `dark`;
`box` clipped to the image), and, with `step` 0, every blown-out or black area the overview found larger than 5 %
of the image, which gets no look at all, so an area the agent could not read never passes for a clean one.

### Downloaded work-order file

Not built yet: the page still saves the bare work order (`work_order.json`, the object above). The
planned file keeps the signature with the order, so the copy can be checked later:

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

`decision` is `approve` or `reject`; `reviewer` is 1–64 printable characters, taken as given (it is not
authenticated). Returns `{"token": "<new signed work order>", "work_order": {...}}`; the decision is appended
to `approvals` with `prev_token_sha256`, the SHA-256 hex digest of the request's `token` (the same value
as the review log's `token_in_sha256`). Nothing stops one token being signed into two conflicting decisions,
but their orders then share one `prev_token_sha256`, and a chained review's `prev_token_sha256` is the hash
of the token the previous review returned. It is a hash, not the token's MAC, so an order cannot be turned
back into the token it came from. A token has one spelling: the MAC is compared as written, so a re-padded
one, or one with `+` `/` for `-` `_`, is refused with 403 and cannot give a forked decision another hash.
(`drishyam review` edits an unsigned `work_order.json` and writes no `prev_token_sha256`.)
Errors: 400 malformed, 403 bad or expired signature, 409 finding not pending.

Each accepted review prints one log line; a refused one prints none. On AWS it is a CloudWatch EMF
record, metric `Reviews` (namespace `Drishyam`, dimension `Decision`), with `wo`, `finding`, `Decision`
and three SHA-256 hex digests: `reviewer_sha256` (of the reviewer name as stored), `token_in_sha256` (of
the request's token) and `token_out_sha256` (of the returned token). Neither the name nor a token is
logged; one review's `token_out_sha256` is the next review's `token_in_sha256`.
The agent has no route to this transition: only this human endpoint (or the
`drishyam review` CLI) can approve or reject a critical finding.
