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
segmenter uses `new`. The example was measured on the macOS x86_64 wheel, without `DRISHYAM_ONNX`; the deployed
function reports segmenter `onnx:drishyam_dacl10k.onnx` and its licence. The deployed
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
 "confirm": {"reason": "ambiguous", "net": "onnx:drishyam_dacl10k.onnx", "scout": {"entropy": 0.52,
             "classes": {"crack": 310}}, "latency_ms": 783.1},
                                                        // only with a two-model cascade, which no app path runs
                                                        // (off by default; see "Two-model cascade" below)
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
 "coverage": 0.62, "unvisited": ["r9", "r10"],           // salient area seen at any level, the survey included
 "coverage_looked": 0.41,                               // ... seen by a look other than the survey (looked at closely)
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
hairline crack for one, so its seed orders candidates but never ends a job. The stop rule therefore never ends a job on a saliency prior and a coarse view alone,
nor before it has seen anything (an overview that proposes no candidate ends the job `exhausted` with no look, and its
work order lists the whole photo as not assessed); on a photo whose candidates lie apart the floor rarely binds and the cap ends it. A `stop-rule` halt names the binding term:
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

`onnx:drishyam_dacl10k_v2a.onnx`, the retrained v2a model (sha256 `95b503c2...`), keeps v1's values (`min_view` 0.3,
absorption off), so its finest levels are v1's. The rule written down before its dev sweep asked for higher recall
than v1's values at the 30 % cap on both synth_dev and CODEBRIM dev100 (each paired 95 % CI above 1), with no more
false alarms on the clean dev split by the same standard; none of the five candidates met it
(`results/dev/onnx_v2a_profile_sweep.json`). Any level (`min_view` 0) and absorption on each lost recall on CODEBRIM
(x0.79, CI 0.70-0.88; x0.89, 0.82-0.95), and 0.15 found x1.02 (0.95-1.10) with 2.2 times the boxes.

`onnx:drishyam_dacl10k_v2b.onnx`, the retrained v2b model (an EfficientNet-B2 UNet, sha256 `47e4b687...`), keeps
v1's values too, under the same rule, also written down before its sweep (`results/dev/onnx_v2b_profile_sweep.json`).
No candidate found more than v1's values on synth_dev (each x1.00, CI 0.97-1.03 or narrower), so none could pass.
On CODEBRIM dev100 any level lost recall (x0.66, CI 0.56-0.77), as did absorption on (x0.84, 0.75-0.91), and 0.15
found x0.98 (0.90-1.06) with 2.4 times the boxes. Any level, 0.15 and 0.25 put a box on fewer clean photos (5 of
30 against 13, difference CI -0.43 to -0.10), which the rule reports but does not count. Any level and 0.15 also ran
into the 20 s time cap on 66 to 80 of the 100 CODEBRIM jobs, because v2b's cost table was timed at a higher machine
load than v2a's; on synth_dev, which settled the decision, no job went past 13.3 s.

`onnx:drishyam_dacl10k_v2c.onnx`, the retrained v2c model (v2a's MobileNetV3-Large UNet with rare classes weighted
harder in training, sha256 `24b4d561...`), keeps v1's values too, under the same rule, written down before its sweep
(`results/dev/onnx_v2c_profile_sweep.json`). No candidate found more than v1's values on synth_dev (x0.98 or x0.97,
every CI reaching 1.00 or above), so none could pass. On CODEBRIM dev100 0.15 found more (x1.09, CI 1.01-1.20) with
2.3 times the boxes, a gain on one split, which the rule does not take; any level lost recall (x0.92, 0.85-0.99), as
did absorption on (x0.88, 0.82-0.93). Any level, 0.15 and 0.25 put a box on fewer clean photos (7 of 30 against 12,
difference CI -0.30 to -0.03), which the rule reports but does not count.

Which model the live engine should run was compared on the same dev splits, each model at its own profile and on
its own cost table (`results/dev/model_choice_dev.json`, its criterion committed before any v2b row existed). No
model is best by that criterion. At the 30 % cap the three ONNX models find the same share of CODEBRIM dev100's
defects (v2b 0.480, v2a 0.480, v1 0.479; every pairwise CI spans 1, e.g. v2b against v1 x1.00, 0.93-1.09), where the
same number of boxes at random positions finds 0.24, 0.27 and 0.23 (IoU-only 0.26, 0.24 and 0.27 against 0.06,
0.07 and 0.07). On synth_dev no pair differs measurably (0.565, 0.630, 0.615). Each finds about three times what the
classical engine finds on CODEBRIM (0.153; v1 x3.13, CI 2.37-4.31). They differ in false alarms on the 30 clean
photos: v1 put a box on 1, v2a on 5, v2b on 13, and v2a and v2b each put measurably more boxes on them than v1
(+0.67 and +1.23 per photo). v1 sends more critical boxes to the human gate on CODEBRIM (7.24 per photo against
4.22 and 3.94), and its modelled ms per CODEBRIM job are 1.05 times v2a's (CI 1.03-1.08; both tables were timed in
one window). On dev only, the recommendation is v1 for the live engine, which runs it since 2026-10-21 (deployed commit
4fe8e30): it is the one model with no measurable loss against any other, at 2.00 times the classical engine's modelled ms per
CODEBRIM job on this Mac (CI 1.89-2.11). v2b's ms there are about 2.5 times v2a's, but timed in one window a v2b
look costs 0.87 to 1.04 times a v2a look: its table was timed at a higher machine load. v2c, compared later under the
same criterion (five models; the reading of "each of the others" and v2c's own rule were committed before any v2c
row), changes none of this. It finds no measurably different share on either split (CODEBRIM 0.462, synth_dev
0.600; against v1 x0.97, CI 0.88-1.06, and x0.98, 0.90-1.04), puts a box on 12 of the 30 clean photos, measurably
more boxes than v1 and v2a (+2.60 and +1.93 per photo), and has lower IoU-only recall than v1 on CODEBRIM (x0.85,
0.72-0.99); it sends 3.83 critical boxes per CODEBRIM photo to the gate. Its ms on its own table are 1.36 times
v2a's, but timed in one window a v2c look costs 0.99 times a v2a look. No held-out set was run for
this choice; the one held-out evaluation that followed, which changed nothing here, is in
`results/heldout_models.json` (report, section 8).

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

### Two-model cascade (off by default)

`drishyam.cascade` lets a fast scout segment every look and a stronger net segment again only the looks that need it.
It is off everywhere by default: a `Job` runs it only when given a confirm net (`Job(confirm=...)`), and no app path
passes one, so this API (locally and on Lambda), `drishyam run` and the MCP server never run it, whatever the
environment holds, and the Lambda stack sets no `DRISHYAM_ONNX_CONFIRM`. `DRISHYAM_ONNX` names the scout as before;
`DRISHYAM_ONNX_CONFIRM` names the confirm net (its accurate net only) for `bench.micro` and the dev evaluation.

After the scout segments a look that passed the quality gate, the confirm net segments the same crop when the scout
drafted a critical class (`reason` `critical`: a mask of at least `A_min` whose largest component is critical by the
findings rule, so exposed rebar at any area or spalling from 40,000 L0 px²), else when the scout's entropy lies in
[`tau_confirm_lo`, `tau_confirm_hi`] (`ambiguous`; `JobConfig`, 0.35 and 0.65, both ends included: not tuned, they
straddle `tau`). The look then keeps the confirm's segmentation for everything after it: escalation, drafting,
seeding and the step's `observation`. The step's `confirm` record names the reason, the net, what the scout saw
(`scout.entropy`, `scout.classes`) and the confirm's own `latency_ms`.

There is no second budget. The confirm reads the crop a second time, so a confirmed step's `pixels_used` grows by two
crops against the one pixel cap; its time runs on the job's clock, so it counts in `seconds_used`, `usd_used`, the
stop rule's spend and the step's `latency_ms`. It runs only if the job would still be inside the pixel, seconds and $
caps after it, its cost predicted as its p95 so far (before the first one, the scout's median segment); otherwise the
record has `skipped`, naming the cap (`"pixel cap: 120000 px left after this look, 262144 needed"`), no `latency_ms`,
and the look keeps the scout's segmentation. It is timed as its own op, `segment:confirm` (in `runtime`), so under the
fixed cost model each net is charged its own p50, and it always runs the confirm's accurate net: the governor's
`p95>SLA` ladder reads and moves the scout only. A work order made with a confirm net also carries `confirm_sha256`.

`bench.micro --job-ops-only` with `DRISHYAM_ONNX_CONFIRM` set adds a `segment:confirm` row per size to `job_ops` and
names the net under `job_ops.confirm`; a fixed-cost cascade refuses a table that did not time the same confirm net.
`make eval-cascade SCOUT=<onnx> CONFIRM=<onnx> [LABEL=...] [WORKERS=4]` times the pair into
`results/<arm>-<pair>/micro.json` (the pair named like the single models: `dacl10k_v2a+v2b`), then runs (at most
WORKERS processes, default 4; the numbers do not depend on it) the scout alone, the confirm net alone (each
look charged its `segment:confirm` p50, under the scout's profile, so only the net differs) and the cascade, at caps
0.3 and 1.0 on the three dev splits (synth_dev, the clean split, CODEBRIM dev100; never a held-out set), checks that
each arm ran the nets it is named for, and writes `results/dev/cascade_<pair>.json`: recall, IoU-only recall, chance
recall, boxes per photo (on the clean split, false alarms), the share of photos with any box, modelled ms per job and
recall per modelled second, each with a paired 95 % bootstrap CI against the scout alone at the same cap. Its
`readings` read every one of those CIs by the same standard, for a gain and a harm alike: a change only when the CI
excludes no change (a ratio of 1; for boxes per photo and the share with any box, a per-photo difference of 0), then
`gain` or `harm` where the direction says which (fewer false alarms on the clean split is a gain), `higher` or `lower`
where it does not (chance recall, boxes on a labelled split), else `no clear change`.

`results/dev/cascade_dacl10k_v2a.fp16+v2a.json` is a plumbing check, not a result: the v2a weights as the FP16 file
scout and the same weights in FP32 confirm, so the two nets give nearly the same masks. It shows the mechanism and its
accounting at work (confirms for both reasons, confirms skipped at the pixel and seconds caps, each net charged its own
p50), and that on this x86 Mac the FP16 file is no faster than the FP32 one (segment:accurate p50 672-805 ms against
segment:confirm 624-661 ms in the same timing run), so here the FP32 net alone is the cheaper arm.

`results/dev/cascade_dacl10k_v2a+v2b.json` is the first real comparison: v2a (FP32) scouts, v2b confirms, both under
the classical-v1 profile (`PROFILES` has none of their own yet), charged this x86 Mac's costs (v2a 665-739 ms and v2b
813-880 ms per look, timed in one run). Read by the rule above it shows no gain on any split, in recall, IoU-only
recall, false alarms, ms per job or recall per modelled second. On CODEBRIM dev100 the cascade loses recall (x0.83 at
cap 0.3, x0.85 at 1.0; CIs 0.73-0.93 and 0.76-0.94) and costs more per job (x1.12, x1.08): its confirms spend the pixel
and seconds caps, so a job takes about a third fewer looks (10.0 against 14.9 at 0.3, 16.0 against 24.2 at 1.0). v2b
alone also loses recall there (x0.89, x0.90) and is no better on synth_dev (x0.91, x0.97: no clear change); on the
clean split no arm changes the false alarms. On these splits v2b is not the stronger net, so the cascade stays off.

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

`status` is `blocked` while any critical finding is `pending_review`, else `incomplete` while `not_assessed` lists
anything (a person should look there), else `ready`; the done record's own `status` keeps its meaning. `profile` is
the job's (see the step record) and `weights_sha256` the sha256 of the ONNX weights that made its looks (`null` for
the classical engine); a work order the MCP tools draft has neither, since there the client chose every level.
`not_assessed` lists every window the quality gate refused (`reason` `blur`, `glare` or `dark`; `box` clipped to the
image), and, with `step` 0, every blown-out or black area the overview found larger than 5 % of the image, which gets
no look at all, so an area the agent could not read never passes for a clean one. With `action` `check-by-eye` it also
lists the whole photo when the overview proposed no candidate (`reason` `no candidate region was proposed`) and, after
a `kill-switch` or `cap-hit` halt, each candidate no look of its own reached (`region` names it).

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
