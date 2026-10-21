# Recorded MCP session: Claude drives Drishyam on `bridge_03.jpg`

This session ran on the classical engine; two sessions with the same bridge and rules on the live v1 engine (on
`bridge_03.jpg` and the team's `real_01.jpg`) are in [../mcp-session-v1/transcript.md](../mcp-session-v1/transcript.md).

Recorded 2026-10-18 on macOS (x86_64), OpenCV 5.0.0, classical segmenter, default caps (30 % of pixels,
$0.05, 20 s). The client was Claude (Opus 5.5) inside Claude Code. `claude -p` could not run on this machine
(the account's organisation disables subscription access for it), so the session went through
[bridge.py](bridge.py): a 60-line MCP client that starts `uv run drishyam mcp` over stdio, forwards one
`tools/call` per request file and saves each reply, and the crop JPEGs, to disk. The bridge chooses nothing.
Claude read every reply, and looked at each crop image, before writing the next request.

Files: [exchange.jsonl](exchange.jsonl) holds every request and reply in order, [trace.jsonl](trace.jsonl) is
the server's own trace for job `j_90a925` (each `why` is stored as the record's `rationale`),
[work_order.json](work_order.json) is the drafted order and [crops/](crops/) holds the images the server returned.

## How each measurement chose the next call

| # | Call | What came back | Why the next call followed |
| --- | --- | --- | --- |
| 0 | `open_image(public/samples/bridge_03.jpg)` | 3000x2250, 5 pyramid levels, caps | Nothing seen yet, so ask for candidates. |
| 1 | `overview` | r1 prior 0.693, r2 0.506, r3 0.465, and the saliency heatmap ([img](crops/img-001-1.png)) | r1 has the highest prior and the brightest blob: look there first, coarse. |
| 2 | `gaze(r1, L2)` | g1, window 2048 px, sharpness 53.5 ([img](crops/img-002-1.jpg)) | The crop shows a spall with bars at crop (200,340) and a thin crack top right. Segment it. |
| 3 | `segment(g1)` | exposed_rebar 522 px (conf 0.80), entropy 0.39 | The rebar mask is small at L2 and sits off-centre: re-centre on it, one level finer. |
| 4 | `gaze(1616,1562, L1)` | g2, window 1024 px ([img](crops/img-004-1.jpg)) | Three rust bars in the spall are clear in the crop. Segment. |
| 5 | `segment(g2)` | spalling 11,676 px, exposed_rebar 3,919 px (conf 0.90), entropy 0.43 | Exposed rebar is critical, so confirm the mask one level finer before logging it. |
| 6 | `verify(g2)` | g3 at L0, **IoU 0.00**, agrees false | The two levels disagree. See what L0 thinks. |
| 7 | `segment(g3)` | corrosion 38,361 px, efflorescence 15,666 px, no rebar, no spalling | At L0 the window holds only the middle of the patch, and the rust bars read as corrosion. The L1 view is the better evidence, but a disagreement on a critical class is a human's call. |
| 8 | `log_finding(g2, exposed_rebar)` | f1, critical, `pending_review`, rule "exposed_rebar: always critical" | Hand it over with the disagreement in the reason. |
| 9 | `request_human_review(f1)` | recorded | Next: the crack seen in g1's crop, near r3. |
| 10 | `gaze(r3, L1)` | g4 ([img](crops/img-010-1.jpg)) | A branching hairline crack, about 2 px wide in the crop. Segment. |
| 11 | `segment(g4)` | crack 1,694 px (conf 0.90), entropy 0.43 | A hairline's width needs full resolution: go to L0 on it. |
| 12 | `gaze(2640,1096, L0)` | g5, window 512 px, sharpness 31.1 ([img](crops/img-012-1.jpg)) | Segment and measure. |
| 13 | `segment(g5)` | crack 3,314 px (conf 0.90), entropy 0.33 | Same class at both levels, low entropy: measure it. |
| 14 | `measure(g5, crack)` | 455.5 px long, 5.37 px wide, 124.7 degrees | Log it (not a critical class). |
| 15 | `log_finding(g5, crack)` | f2, draft | Next: r2, the second blob, not seen yet. |
| 16 | `gaze(r2, L1)` | g6 ([img](crops/img-016-1.jpg)) | A spall, no bars. Segment. |
| 17 | `segment(g6)` | spalling 12,142 px (conf 0.96), entropy 0.22 | A new mask: confirm at L0 before trusting it. |
| 18 | `verify(g6)` | g7 at L0, **IoU 0.72**, agrees true | Confirmed. Log it. |
| 19 | `log_finding(g6, spalling)` | f3, 48,568 px², critical by "spalling 48,568 px² >= 40,000 px²" | Before looking further, check what budget is left. |
| 20 | `runtime_stats` | 28.75 % of the 30 % pixel cap used; p95 segment 151 ms | One more L0 look (262,144 px) would pass the cap, so stop looking. |
| 21 | `request_human_review(f3)` | recorded | Write it up. |
| 22 | `draft_work_order` | 3 findings, status blocked (two critical findings wait for a human) | End. |

The two `verify` calls show the loop the competition's Agentic Vision rules ask for: at step 6 an OpenCV
measurement (mask IoU 0.00 between L1 and L0) changed the next actions from "log it" to "look at what L0 sees,
then hand it to a human"; at step 18 an IoU of 0.72 let the finding through. At step 20 the measured budget
ended the search.

## Against the sample's ground truth

`public/samples/index.json` lists four boxes for this synthetic photo. The session found three: the exposed
rebar (f1), the crack (f2) and the small spall (f3). It did not log the large spall around the rebar
(box 1326,1296,570,529), although g2's segmentation measured it (spalling 11,676 px): Claude logged only the
critical rebar class from that look. The rule planner in `drishyam run` logs every class over its area
threshold, so it does not make this particular miss.

## Reproduce

```bash
uv sync --extra mcp
uv run python docs/mcp-session/bridge.py runs/mcp-session &
# then write runs/mcp-session/req-000.json, req-001.json, ... ({"tool": ..., "args": {...}}) one at a time,
# reading each rep-NNN.json before writing the next; {"tool": "_quit"} ends the session
```

With a working `claude -p`, the one-command recording in [../mcp.md](../mcp.md) does the same without the bridge.
