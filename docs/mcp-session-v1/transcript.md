# Recorded MCP sessions on the v1 engine: Claude drives Drishyam on `bridge_03.jpg` and `real_01.jpg`

Recorded 2026-10-21 on macOS (x86_64), Python 3.13.16, OpenCV 5.0.0 (DNN new engine), mcp 2.3.0, Drishyam 1.1.0 at
`45a2bc5`, default caps (30 % of pixels, $0.05, 20 s). The segmenter is v1, the live engine:
`DRISHYAM_ONNX=models/drishyam_dacl10k.onnx` (sha256 `c49416c1…`, so the server picks the `onnx:drishyam_dacl10k.onnx`
profile: min_view 0.3, absorption off, governor p95 limit 1500 ms) and `DRISHYAM_ONNX_FP16=models/drishyam_dacl10k.fp16.onnx`
(`f9b06fab…`, loaded but never used: the governor did not act in either session). The session in
[../mcp-session/](../mcp-session/transcript.md) was recorded on the classical engine (2026-10-18), before the live
site switched to v1.

The client was Claude (Opus 5.5) inside Claude Code. As before, the session went through [bridge.py](bridge.py), a
minimal MCP client that starts `uv run drishyam mcp` over stdio, forwards one `tools/call` per request file and saves
each reply and its images to disk; it chooses nothing. Its one change from [../mcp-session/bridge.py](../mcp-session/bridge.py)
is that it passes the `DRISHYAM_*` settings to the server (the MCP stdio client otherwise starts it with only
HOME, PATH and a few other variables, which would select the classical engine); `session.json` lists them. Claude read
every reply, and looked at every image the server returned, before writing the next request. `verify` returns no
image: on `bridge_03` its L1 look was judged from its segmentation only, and on `real_01` a later `enhance` returned it.

Per photo: `exchange.jsonl` holds every request and reply in order, `trace.jsonl` is the server's own trace (each `why`
is stored as the record's `rationale`), `work_order.json` is the drafted order, `crops/` holds the images the server
returned and `prompt.txt` is the task.

## The rules, and where they no longer fit v1

Both prompts are [../mcp-session/prompt.txt](../mcp-session/prompt.txt) with the photo changed and one sentence added
([bridge_03/prompt.txt](bridge_03/prompt.txt)), and for `real_01` a second ([real_01/prompt.txt](real_01/prompt.txt)):

- **One level finer stops at L1.** v1's profile allows no look spanning less than 0.3 of the long side: on these
  3000-px photos L1 (1024 px, 0.34) is the finest level and L0 (512 px, 0.17) is out, as it is for the rule agent. So
  "gaze one level finer" and `verify` work only from L2, and a mask first seen at L1 is logged unverified. The old
  session's steps "verify at L0" and "go to L0 to measure a hairline" have no v1 equivalent.
- **Crack width is not measurable on v1.** The v1 crack mask is 15-17 crop px wide at every level, so `measure`
  reports the mask's width: 30 px for the `bridge_03` crack at L1, 69 px (L2) and 34 px (L1) for the `real_01` crack,
  where the line in the crops is 2-3 crop px. The classical engine measured 5.37 px at L0 in the old session.
- **No candidate (added for `real_01`).** The prompt says what to do when `overview` proposes none: look at the heatmap
  and decide.
- **The blur gate on smooth paint (not a v1 issue, not covered by the rules).** On `real_01` every look failed the
  quality gate (blur, sharpness 5.8 to 10.7) although the crops show a crisp line on low-texture paint. Claude used
  `enhance` (CLAHE), the re-look that `log_finding`'s refusal of a gated look names. `verify` does not enhance, so its IoU could not be measured.

The thresholds that remain (entropy 0.5, IoU 0.5, spalling of 40,000 px², exposed rebar always critical, 25 calls)
are unchanged.

## `bridge_03.jpg`: how each measurement chose the next call

Job `j_0a56df`. Image links are the crops the server returned.

| # | Call | What came back | Why the next call followed |
| --- | --- | --- | --- |
| 0 | `open_image(public/samples/bridge_03.jpg)` | 3000x2250, 5 levels, pixel cap 2,025,000 | Nothing seen yet, so ask for candidates. |
| 1 | `overview` | r1 prior 0.693, r2 0.497, r3 0.461, r4 0.460 (one more than on the classical engine's run), heatmap ([img](bridge_03/crops/img-001-1.png)) | r1 has the highest prior and the brightest blob, and an L2 look there also covers r4. |
| 2 | `gaze(r1, L2)` | g1, window 2048 px, sharpness 53.5 ([img](bridge_03/crops/img-002-1.jpg)) | The crop shows a spall with three rust bars and a thin crack at the top right. Segment. |
| 3 | `segment(g1)` | spalling 7,532 px (conf 0.91), corrosion 3,908 (0.69), exposed_rebar 1,591 (0.65), crack 892 (0.90), entropy 0.16 | Entropy is low and no mask is tiny, so no zoom. The spalling mask is new: verify it at L1. |
| 4 | `verify(g1)` | g2 at L1, spalling **IoU 0.468**, kind `disagree`, best class at L1 corrosion | Below 0.5, so the two looks disagree. Nothing finer settles it on v1. See what L1 thinks. |
| 5 | `segment(g2)` | corrosion 24,817 px (conf 0.97), spalling 12,436 (0.89), no rebar, no crack | At L1 the bars read as corrosion. Log what each level showed and hand the critical ones to a human. |
| 6 | `log_finding(g2, spalling)` | f1, 41,508 px², critical by "spalling 41,508 px² >= 40,000 px²", `pending_review` | Next, the bars as L2 saw them. |
| 7 | `log_finding(g1, exposed_rebar)` | f2, critical by "exposed_rebar: always critical", `pending_review` | Then the bars as L1 saw them. |
| 8 | `log_finding(g2, corrosion)` | f3, 99,268 px², draft (the same bars as f2) | Ask for review of both critical findings, giving the measured reason. |
| 9 | `request_human_review(f1)` | recorded: "L2 and L1 masks disagree (IoU 0.468 < 0.5) and v1 has no finer level" | |
| 10 | `request_human_review(f2)` | recorded: "seen only at L2 (conf 0.65); at L1 the same bars segment as corrosion" | Next: the crack in g1's crop, at r3. |
| 11 | `gaze(r3, L1)` | g3, window 1024 px, sharpness 54.3 ([img](bridge_03/crops/img-011-1.jpg)) | The crop shows a branching hairline crack. Segment. |
| 12 | `segment(g3)` | crack 8,176 px (conf 0.95), entropy 0.22 | The same class as g1's crack mask, with low entropy. Measure it (there is no finer level to go to). |
| 13 | `measure(g3, crack)` | 1,084.6 px long, 30.15 px wide, 135.2 degrees | 30 px is the mask: the line is 2-3 px in the crop. Log it unverified, saying so. |
| 14 | `log_finding(g3, crack)` | f4, draft | Next: r2, the second blob, not yet seen. |
| 15 | `gaze(r2, L1)` | g4 ([img](bridge_03/crops/img-015-1.jpg)); an L2 window would have clamped to (1024, 1226) and its verify look would have missed the blob | A spall, no bars. Segment. |
| 16 | `segment(g4)` | spalling 14,282 px (conf 0.96), entropy 0.19 | A new mask at the finest level, so it is logged unverified. |
| 17 | `log_finding(g4, spalling)` | f5, 57,128 px², critical | |
| 18 | `request_human_review(f5)` | recorded | Check the budget before looking further. |
| 19 | `runtime_stats` | 17.1 % of the 30 % pixel cap used, p95 segment 235 ms (limit 1500 ms), governor history empty | Every candidate has been seen (r4 is inside g1's window and showed only the spall's edge). Stop looking. |
| 20 | `draft_work_order` | 5 findings, status `blocked` (three critical findings wait for a human) | End. |

The measurement that changed the plan was step 4: an IoU of 0.468 sent the large spall to a human as a disagreement
instead of a confirmed finding, and the L1 segmentation (no rebar, corrosion 0.97) became the reason given for the
rebar review. The classical session's L0 look had read the same bars as corrosion; v1 does it one level coarser.

## `real_01.jpg`: how each measurement chose the next call

Job `j_aa5556`. This is the team's own photo of a painted plaster wall. It has no ground truth.

| # | Call | What came back | Why the next call followed |
| --- | --- | --- | --- |
| 0 | `open_image(public/samples/real_01.jpg)` | 3000x2250, same caps, no people | Ask for candidates. |
| 1 | `overview` | **0 candidates** (saliency max 0.22, under the 0.5 candidate threshold), heatmap ([img](real_01/crops/img-001-1.png)) | The heatmap shows one faint line from about (480, 0) to (2440, 2240) in L0 pixels. An L2 look at its middle covers most of it. |
| 2 | `gaze(1460,1124, L2)` | g1, window [436,100,2048,2048], **gate failed: blur, sharpness 5.8** ([img](real_01/crops/img-002-1.jpg)) | The crop shows a crisp thin line on smooth paint, so this is low texture rather than blur. A refused look cannot be logged: try CLAHE. |
| 3 | `enhance(g1)` | sharpness 43.3, gate passes ([img](real_01/crops/img-003-1.jpg)) | Segment. |
| 4 | `segment(g1)` | crack 13,908 px (conf 0.96), entropy 0.18, no other class | A new mask: verify at L1. |
| 5 | `verify(g1)` | g2 at L1: `quality_ok` false, **iou null**, kind null | The finer look failed the blur gate, so no IoU. `verify` does not enhance. Enhance g2 and segment it by hand. |
| 6 | `enhance(g2)` | sharpness 9.7 to 53.5 ([img](real_01/crops/img-006-1.jpg)) | The crack crosses the crop. Segment. |
| 7 | `segment(g2)` | crack 10,897 px (conf 0.96), entropy 0.20 | Both levels show the same class, but no IoU was measured. Log the L2 mask, which spans the crack inside g1's window, and put the null IoU in the why. |
| 8 | `log_finding(g1, crack)` | f1, 3,211.6 px long, 69.29 px wide, box [480,100,1944,2048], draft | 69 px is about 17 crop px at L2. Measure at L1 to see whether the width holds. |
| 9 | `measure(g2, crack)` | 1,293.1 px long, 33.71 px wide | The width halved with the scale, so it belongs to the segmenter, not the crack (2-3 crop px at L1 by eye). The heatmap line is brightest where it runs past g1's lower edge (y 2148). |
| 10 | `gaze(2300,1900, L1)` | g3, window [1788,1226,1024,1024], gate failed (blur, 10.7) ([img](real_01/crops/img-010-1.jpg)) | The crack runs to the bottom edge. CLAHE. |
| 11 | `enhance(g3)` | sharpness 51.9 ([img](real_01/crops/img-011-1.jpg)) | Segment. |
| 12 | `segment(g3)` | crack 11,675 px (conf 0.96), entropy 0.17 | This is f1's line, continued. `log_finding` merges only boxes that overlap by more than 0.3 (here about 0.13), so logging it would list one crack twice. Not logged. (This reason is given here, not in the trace: `segment` takes no `why`.) |
| 13 | `runtime_stats` | 13.2 % of the pixels, p95 segment 452 ms | The heatmap's only structure has been looked at end to end, and the budget is not the limit. Stop looking. |
| 14 | `request_human_review(f1)` | recorded, f1 stays draft | Beyond the prompt's rules (f1 is not critical and no disagreement was measured): its check could not run and its width is wrong, so a person should know. |
| 15 | `draft_work_order` | 1 finding, status `ready`, nothing listed as not assessed | End. |

**What the segmenter flagged, and what Claude saw by eye.** v1 segmented `crack` in all three looks (conf 0.96 each,
entropy 0.17 to 0.20) and no other class anywhere. The finding's outline ([work order](real_01/work_order.json),
box [480,100,1944,2048]) follows the single line, and it does not take in the trowel marks visible in the enhanced
crops at the upper right. By eye the crops show one hairline crack, from the top edge near x 500 to the bottom edge
near x 2470, 2-3 crop px wide at L1 (about 4-6 px at full resolution), and nothing else that looks like a defect.
This photo has no ground truth, so these are readings, not a detection rate. The 69 px width and the 3,212 px length
(the skeleton of a 17-px-wide band) are the mask's measurements, not the crack's.

## What differs from the rule agent on the same photo

The rule agent was run with the same environment:
`DRISHYAM_ONNX=… DRISHYAM_ONNX_FP16=… uv run drishyam run <photo> --cost-model fixed --micro-arm local-macos-x86-dacl10k-retime`.

**`bridge_03.jpg`** (rule job `j_2d7174`):

```
step 1  gaze L3 (1500,1125)  survey  entropy 0.34  px 5.5%
step 2  verify L2 (1599,1226)  verify-new-mask  entropy 0.20  px 9.3%
step 3  gaze L1 (1840,1456)  best-score  entropy 0.17  px 13.2%
step 4  gaze L1 (784,1738)  best-score  entropy 0.19  px 17.1%
step 5  halt  stop-rule: best score 6.25e-04 < lambda 7.81e-04 (eps x value/cost)
job j_2d7174: awaiting_review (stop-rule)
findings 5  critical pending review 4
```

Its findings: f1 spalling at L2, 115,968 px², with one box spanning both spalls; f2 corrosion; f3 exposed_rebar at L2,
4,464 px², conf 0.58; f4 spalling at L1, 50,372 px²; f5 spalling at L1, 57,128 px². Four of the five are critical.

| | Rule agent (v1) | Claude over MCP (v1) | Claude over MCP (classical, 2026-10-18) |
| --- | --- | --- | --- |
| Looks (512² each) | 4: L3 survey, L2 verify, L1 at r1, L1 at r2 | 4: L2 at r1, L1 verify, L1 at r3, L1 at r2 | 7 |
| Pixels used | 17.1 % | 17.1 % | 28.75 % |
| Tool calls | none (4 looks, then a halt) | 21 | 23 |
| Findings (critical) | 5 (4) | 5 (3) | 3 (2) |
| Ground-truth boxes found (harness rule: same class, IoU >= 0.3 or contains the centre) | 3 of 4, missing the crack | 4 of 4 | 3 of 4, missing the large spall |
| Findings matching a ground-truth box one-to-one | 3 of 5 | 4 of 5 (f3 corrosion matches none) | 3 of 3 |
| Verify IoU measured | corrosion 0.884 between L3 and L2 (f2) | spalling 0.468 between L2 and L1 | 0.00 (rebar), 0.72 (spall) |

The difference is r3. The rule agent's L3 survey did not segment the crack, and its stop rule ended the job before
r3 (prior 0.46) was worth a look. Claude had seen a thin line in g1's L2 crop, whose segmentation also held a
crack mask (892 px), and spent one look on it. In the other direction, the rule agent logged the large spall twice
(f1 at L2 with a box spanning both spalls, f4 at L1), and Claude logged the rusted bars twice, once as each level
labelled them (f2 rebar, f3 corrosion).

**`real_01.jpg`** (rule job `j_712981`):

```
overview: 0 candidates
step 1  halt  exhausted: every candidate has been looked at
job j_712981: completed (exhausted)
steps 0  pixels 1.6%  $3e-06  0.028 s  looked at 0% of the salient area (the survey look not counted)
findings 0  critical pending review 0
not assessed: 1 area listed in the work order; a person should look there
```

The rule agent took no look: this run predates the zero-candidate fallback (on by default since 2026-10-21 evening,
live since 2026-10-22), with which it takes one whole-photo look at L2 that the blur gate refuses ([report, section
9](../report.md#9-failures)). Its work order is `incomplete` and lists the whole photo as not assessed ("no candidate
region was proposed", action `check-by-eye`). Claude took 3 looks (13.2 % of the pixels) where it saw a line in the
heatmap, and logged one crack. Its work order says `ready` with nothing listed as not assessed, although about a third
of the photo (32 %, outside the windows of g1 and g3) was seen only in the overview: the MCP `draft_work_order` lists
refused looks and blown-out or black areas, not areas no look reached. Here the rule agent's output is the more
conservative of the two.

## Honest limits

- **One session per photo.** Nothing here is a rate. The `bridge_03` ground truth is synthetic. `real_01` has no
  ground truth and is described by eye.
- **The session is not blind.** Before recording, the client had read `public/samples/index.json`, including the
  four ground-truth boxes for `bridge_03`, and the 2026-10-18 transcript, which names the spall it missed. Each call
  was chosen from the replies and the crops, but that prior knowledge may have shaped the choices: logging the spall
  from the disagreeing look, for one, and looking at r3. For `real_01` the brief said in advance that the overview
  proposes no candidate, and the prompt has a sentence for that case.
- **The decision rules are in the prompt** (entropy above 0.5 means zoom, IoU below 0.5 means disagreement, the
  critical thresholds, L1 as v1's finest level). So the sessions show the tools carrying measured evidence to an LLM
  more than they show autonomous planning. Where Claude went beyond the rules, the tables say so: CLAHE on the gate
  refusals, not logging g3, and the extra review request, all on `real_01`.
- **An LLM in the loop is not the deployed site's planner.** The live site runs the rule policy with no LLM. These
  sessions use the same server tools and caps on a laptop, and the client also sees the crops, which the rule agent
  does not.
- **Costs are measured differently.** The MCP sessions' seconds and latencies are this Mac's wall clock. The rule
  agent's seconds and $ come from the fixed cost table named above. The pixel counts are comparable.

## Reproduce

```bash
uv sync --extra mcp
# the v1 weights are v1.1.0 release assets, not committed: see models/README.md
DRISHYAM_ONNX=models/drishyam_dacl10k.onnx DRISHYAM_ONNX_FP16=models/drishyam_dacl10k.fp16.onnx \
  uv run python docs/mcp-session-v1/bridge.py runs/mcp-v1/bridge_03 &
# then write runs/mcp-v1/bridge_03/req-000.json, req-001.json, ... ({"tool": ..., "args": {...}}) one at a time,
# reading each rep-NNN.json (and its images) before writing the next; {"tool": "_quit"} ends the session
```
