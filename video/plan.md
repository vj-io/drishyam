# Drishyam submission video: plan

The narrated video for the OpenCV AI Competition 2026, powered by AWS (Agentic Vision entry): at most 5 minutes,
1920x1080, offline AI voice (Kokoro `af_heart` through the brag-scaled `narrate` skill), captions burned in and
also shipped as WebVTT. The look is the site's own engineering-sheet blueprint (`public/styles.css` tokens and
fonts; the brag-styles `patent-office` set): one sheet per chapter, a sheet index on the left, the sheet number
stamping in at each chapter.

## Outputs

| File | What | In git |
| --- | --- | --- |
| `public/video/drishyam.mp4` | the video (H.264 + AAC, poster baked in as frame 0) | no: `public/video/*.mp4` is ignored; it ships to the site bucket at deploy time and as a release asset |
| `public/video/drishyam.vtt` | captions, the same cues as the burned-in ones | yes |
| `public/video/poster.jpg` | the end card, settled | yes |
| `brag-output/drishyam-short.mp4` | 15-20 s 9:16 cut: the cold open and the sign-off stamp | no (brag-output/ is ignored) |

The site expects the player at `video/drishyam.mp4` with captions at `video/drishyam.vtt` (the paths the removed
`initVideo()` in `public/js/results.js` used, commit 939f973; commit 5c2fb04 removed it until the MP4 ships). Adding
the `<video>` markup to `public/index.html` is left to the commit that deploys the MP4: until the file is in
the bucket the page would request a missing file.

## Re-render: one command

```bash
~/.claude/brag-scaled-venv/bin/python video/build.py
```

It re-reads every number below from its file, re-writes the narration from `video/narration.py` (lines are
cached by text, so only changed lines are synthesised), fits the chapters to the voice, regenerates the
captions and the generated half of this file, makes the score, renders with the fast-render skill and cuts the
9:16 short. `--stills T ...` renders checking stills only; `--audit` runs the activity audit only; `--layout`
stops after the voice, the layout, the captions and this file. If `public/results/codebrim_dacl10k/summary.json`
exists at build time, the trained-segmenter line and the RESULTS PENDING stamp switch to its numbers by
themselves. Claims whose truth depends on the data are asserted in `video/narration.py`, so a re-run that changes
them stops the build instead of shipping a sentence that is no longer true.

The live footage is not re-recorded by the build (the WAF allows 10 job starts per IP per 120 s). To record it
again (4 job starts, about 3 minutes):

```bash
~/.claude/brag-scaled-venv/bin/python video/record.py            # screencast + event log -> brag-output/work/rec/, traces -> video/live/
~/.claude/brag-scaled-venv/bin/python video/record.py --health   # /api/health response -> video/live/health.json
~/.claude/brag-scaled-venv/bin/python ~/.claude/plugins/cache/brag-scaled/brag-scaled/0.8.0-scaled/skills/motion-kit/scripts/walk.py record brag-output/work/walk-plan.json --out brag-output/work/walkrec
```

## How the footage is used, honestly

- Every frame of the demo chapters is a screencast of a real job on https://d1xg58guluup8c.cloudfront.net
  (recorded 2026-10-18, live jobs `j_fdcb12` guided, `j_946907` break it, `j_93a047` KILL, `j_d4f644` Resume).
  Each run's trace is saved, token fields and embedded images removed, in `video/live/*.ndjson`; every live-run number in the voice
  or on screen comes from those files.
- The footage is retimed so each recorded event lands on the line that explains it. The window bar always shows
  the speed (`REAL TIME`, or `2.5x SPEED` and so on). Cuts (each a 0.36 s cross-dissolve): in the cold open,
  from step 4 to the first sign-off card; in the live chapter, the page's smooth scroll after "Watch it inspect a
  bridge" (it passes the hero's stat cards, which show the deployed build's numbers from before the final
  evaluation re-run) and the seconds between the end of the run and the first sign-off; in the drills, steps 1-4
  of the "Now break it" run (they repeat the guided run) and the seconds between the planner's rejection and the
  "Try KILL" click; in the 9:16 cut, from step 4 to the first reviewer field.
- The cursor is drawn from the recorder's log of real mouse moves and clicks; headless Chromium draws none.
- The reviewer name typed in the sign-off is "Demo reviewer", not a person.
- The tampered-token request was sent from the page by `video/record.py`; the inset shows its real status and
  body (`video/live/tamper.json`).
- No LLM runs in the live demo; the voice says so. The LLM chapter is the recorded Claude session in
  `docs/mcp-session/`, shown from its own files.
- CODEBRIM photos are never shown (licence); only its aggregate numbers and one text failure.
- The trained dacl10k segmenter's results are pending and not quoted.

## Not shown, and why

- The site's Benchmarks sheet: the deployed build (ca69037) still shows the numbers from before the final
  re-run; the video draws its own chart from the final `public/results/*.json` instead.
- A CloudWatch dashboard and an owner-recorded team clip: no screenshot or footage exists; the team is shown as
  name cards, as decided.
- A test pass count: `make test` was not run for this video, so no count is claimed; the reproduce sheet shows
  the commands only.
- COOL timings: COOL was not run (README.md, "COOL on Graviton (not run)").

## Known on screen

- `docs/architecture.png` is shown as committed. Its footnote cites `segment.py:239` for `cv2.dnn.ENGINE_NEW`;
  the call is now at `drishyam/segment.py:298`. The claim itself (the one OpenCV-5-only call) still holds.
- Clicking "Try KILL" clears the last job, so the page re-lays out and scrolls back to the viewer; at 5x speed the
  "How it works" sheet passes through the window for a fraction of a second. It is the real page doing it.

<!-- generated:start -->
### Requirement checklist (timestamps from this build)

| Requirement | Covered by | From | To |
| --- | --- | --- | --- |
| At most 5 minutes, 1920x1080 | open, end | 0:00.0 | 4:42.2 |
| Narrated (offline Kokoro voice af_heart) with burned-in captions; the same cues in public/video/drishyam.vtt | open, end | 0:00.0 | 4:42.2 |
| The team (name cards, no footage): Vaibhav Jangid, Bhavya Dhoot | team | 0:10.0 | 0:23.6 |
| Problem and user, with sources on screen | problem | 0:23.6 | 0:49.6 |
| /api/health on the live site: opencv 5.0.0, aarch64, aws-lambda | l1 | 0:50.1 | 0:55.0 |
| The application working: the guided run on bridge_03 ('Watch it inspect a bridge'), recorded live | l2, l3, l4, l5, l6, l7, l8 | 0:55.5 | 1:28.1 |
| Plain-words trace (the NOTE and 'What it decided and why' panels) | l3, l4, l5, l6, l7 | 0:59.1 | 1:23.2 |
| Human sign-off: reviewer name, Approve, APPROVED stamp, work order signed | l9 | 1:28.6 | 1:33.4 |
| Human control: a tampered work-order token refused (403) | l11 | 1:33.9 | 1:37.6 |
| Failure handling: 'Now break it' (slowed segmenter -> governor; malformed planner reply -> allow-list) | b0, b1, b2 | 1:38.5 | 1:53.1 |
| Human control: 'Try KILL' (KILL halts with kill-switch, Resume replays) | b3 | 1:53.6 | 1:59.1 |
| Honesty: no LLM runs in the live demo; the drill planners are scripted | b4 | 1:59.5 | 2:03.9 |
| An LLM driving the OpenCV tools over MCP: IoU 0.00 at call 6 -> human; IoU 0.72 at call 18 -> confirmed; call 20 budget | mcp | 2:04.5 | 2:31.6 |
| Architecture: docs/agent-loop.svg, public/docs/aws-simple.svg, docs/architecture.png (AWS path and OpenCV layer) | arch | 2:31.6 | 3:14.8 |
| Principal results: recall vs pixels, with one downscale, the tiling family and chance | r1, r3 | 3:15.4 | 3:33.0 |
| Larger photos (4608 px): beats one downscale, L2 tiling matches it with fewer pixels | r4 | 3:33.5 | 3:39.7 |
| Ablations (public/results/ablations.json) | r5 | 3:40.2 | 3:47.5 |
| Real photos (CODEBRIM), said plainly weak; no CODEBRIM image shown | r6 | 3:48.0 | 3:56.1 |
| False alarms (public/results/clean/summary.json) | r7 | 3:56.6 | 4:01.8 |
| Trained segmenter: 'results pending' unless public/results/codebrim_dacl10k/summary.json exists | r8 | 4:02.3 | 4:04.4 |
| Failure cases: failures.json plates and their rules; one CODEBRIM case in words only | failures | 4:05.2 | 4:21.0 |
| Limits and responsible use | limits | 4:21.0 | 4:27.6 |
| Reproduce: make test, make eval, scripts/aws.sh | repro | 4:27.6 | 4:34.8 |
| End card: live URL, repository, team | end | 4:34.8 | 4:42.2 |

### Chapters (from this build)

| Chapter | Starts | Ends | Lines |
| --- | --- | --- | --- |
| Cold open | 0:00.0 | 0:10.0 | o1, o2 |
| Team | 0:10.0 | 0:23.6 | t1, t2 |
| Problem and user | 0:23.6 | 0:49.6 | p1, p2, p3 |
| Live demo | 0:49.6 | 1:38.1 | l1, l2, l3, l4, l5, l6, l7, l8, l9, l11 |
| Failure drills | 1:38.1 | 2:04.5 | b0, b1, b2, b3, b4 |
| An LLM over MCP | 2:04.5 | 2:31.6 | m1, m2, m3, m4 |
| Architecture | 2:31.6 | 3:14.8 | a0, a1, a2, a3, a4, a5 |
| Results | 3:14.8 | 4:05.2 | r1, r3, r4, r5, r6, r7, r8 |
| Failure cases | 4:05.2 | 4:21.0 | f1, f2, f3 |
| Limits | 4:21.0 | 4:27.6 | x1 |
| Reproduce | 4:27.6 | 4:34.8 | y1 |
| End | 4:34.8 | 4:42.2 | e1 |

### Every number and URL, with its source

| What | Value | Source file | Field |
| --- | --- | --- | --- |
| synthetic photos | 200 | `public/results/summary.json` | n_images |
| recall vs exhaustive tiling (85 %) | 0.8518 | `public/results/summary.json` | headline_stats.recall_ratio_vs_tiling |
| target recall ratio | 0.9 | `public/results/summary.json` | headline_stats.target.recall_ratio |
| target met? | false | `public/results/summary.json` | headline_stats.meets_target |
| recall vs one downscale | 0.9433 | `public/results/summary.json` | headline_stats.recall_ratio_vs_downscale |
| its 95 % CI | [0.892, 1.0] | `public/results/summary.json` | headline_stats.recall_ratio_vs_downscale_ci |
| headline point: pixels, recall | [0.2344, 0.4675] | `public/results/summary.json` | policies.drishyam[pixels_frac <= 0.3][-1] (as public/js/results.js headline()) |
| exhaustive tiling: pixels, recall | [1.6667, 0.5488] | `public/results/summary.json` | policies.tiling[max pixels_frac] |
| one downscale: pixels, recall | [0.25, 0.4956] | `public/results/summary.json` | policies.downscale[0] |
| chart: every point and its chance tick | all rows | `public/results/summary.json` | policies.*[].pixels_frac, recall, recall_ci, chance_recall |
| 4608 px photos | 100 | `public/results/large/summary.json` | n_images |
| 4608 px: recall vs one downscale | 1.1176 | `public/results/large/summary.json` | headline_stats.recall_ratio_vs_downscale |
| its 95 % CI | [1.0191, 1.2208] | `public/results/large/summary.json` | headline_stats.recall_ratio_vs_downscale_ci |
| 4608 px headline point: pixels, recall | [0.2822, 0.5059] | `public/results/large/summary.json` | policies.drishyam[pixels_frac <= 0.3][-1] |
| 4608 px L2 tiling: pixels, recall | [0.0988, 0.5296] | `public/results/large/summary.json` | policies.tiling[min pixels_frac] (L2) |
| CODEBRIM photos | 200 | `public/results/codebrim/summary.json` | n_images |
| CODEBRIM recall (ours) | 0.197 | `public/results/codebrim/summary.json` | policies.drishyam[pixels_frac <= 0.3][-1].recall |
| CODEBRIM recall, every method (min, max) | [0.197, 0.2226] | `public/results/codebrim/summary.json` | policies.{downscale,tiling,drishyam head}.recall |
| CODEBRIM: its boxes at random | 0.1211 | `public/results/codebrim/summary.json` | policies.drishyam[head].chance_recall |
| 'about 1 in N' (N = round(1 / mid of the range), as the site's hero card) | 5 | `public/results/codebrim/summary.json` | derived from the range above |
| photos with no damage | 51 | `public/results/clean/summary.json` | n_images |
| share of them with any report (ours, 30 % cap) | 0.6667 | `public/results/clean/summary.json` | false_alarms['drishyam@0.3'].images_with_any |
| boxes per image (ours) | 6.35 | `public/results/clean/summary.json` | false_alarms['drishyam@0.3'].boxes_per_image |
| critical per image (ours) | 0.02 | `public/results/clean/summary.json` | false_alarms['drishyam@0.3'].critical_per_image |
| boxes per image (one downscale) | 3.0 | `public/results/clean/summary.json` | false_alarms.downscale.boxes_per_image |
| boxes per image (L0 tiling) | 36.8 | `public/results/clean/summary.json` | false_alarms['tiling@L0'].boxes_per_image |
| ablation 2048 px synthetic: no-survey / full | 0.9019 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-survey'].ratio_vs_full |
| ablation 2048 px synthetic: no-verify / full | 1.0316 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-verify'].ratio_vs_full |
| ablation 2048 px synthetic: no-uncertainty / full | 1.0 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-uncertainty'].ratio_vs_full |
| ablation 2048 px synthetic: governor-on / full | 1.0 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['governor-on'].ratio_vs_full |
| ablation 2048 px synthetic: random-order-s0 / full | 0.9747 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['random-order-s0'].ratio_vs_full |
| ablation 2048 px synthetic: random-order-s1 / full | 0.9905 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['random-order-s1'].ratio_vs_full |
| ablation 2048 px synthetic: random-order-s2 / full | 0.9968 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['random-order-s2'].ratio_vs_full |
| ablation 2048 px synthetic: random order / full, pooled 95 % CI | [0.97, 1.0] | `public/results/ablations.json` | datasets.synth200.random_order[0] |
| ablation 4608 px synthetic: no-survey / full | 0.8772 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-survey'].ratio_vs_full |
| ablation 4608 px synthetic: no-verify / full | 1.0 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-verify'].ratio_vs_full |
| ablation 4608 px synthetic: no-uncertainty / full | 1.0 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-uncertainty'].ratio_vs_full |
| ablation 4608 px synthetic: governor-on / full | 1.0 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['governor-on'].ratio_vs_full |
| ablation 4608 px synthetic: random-order-s0 / full | 0.8772 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['random-order-s0'].ratio_vs_full |
| ablation 4608 px synthetic: random-order-s1 / full | 0.9064 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['random-order-s1'].ratio_vs_full |
| ablation 4608 px synthetic: random-order-s2 / full | 0.8889 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['random-order-s2'].ratio_vs_full |
| ablation 4608 px synthetic: random order / full, pooled 95 % CI | [0.85, 0.94] | `public/results/ablations.json` | datasets.synth_large.random_order[0] |
| ablation CODEBRIM: no-survey / full | 0.6596 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-survey'].ratio_vs_full |
| ablation CODEBRIM: no-verify / full | 1.0271 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-verify'].ratio_vs_full |
| ablation CODEBRIM: no-uncertainty / full | 1.0 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-uncertainty'].ratio_vs_full |
| ablation CODEBRIM: governor-on / full | 1.0 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['governor-on'].ratio_vs_full |
| ablation CODEBRIM: random-order-s0 / full | 0.9518 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['random-order-s0'].ratio_vs_full |
| ablation CODEBRIM: random-order-s1 / full | 0.9458 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['random-order-s1'].ratio_vs_full |
| ablation CODEBRIM: random-order-s2 / full | 0.9608 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['random-order-s2'].ratio_vs_full |
| ablation CODEBRIM: random order / full, pooled 95 % CI | [0.92, 0.98] | `public/results/ablations.json` | datasets.codebrim.random_order[0] |
| failure plate s0162 | Graffiti read as damage | `public/results/failures.json` | [image=failures/f01.jpg].category, rule_that_would_catch_it |
| failure plate s0147 | Formwork line read as crack | `public/results/failures.json` | [image=failures/f03.jpg].category, rule_that_would_catch_it |
| failure plate s0040 | Glare caught by the quality gate | `public/results/failures.json` | [image=failures/f04.jpg].category, rule_that_would_catch_it |
| failure plate s0085 | Crack hidden in blur | `public/results/failures.json` | [image=failures/f06.jpg].category, rule_that_would_catch_it |
| real-photo failure (text only) | image_0001435: exposed_rebar reported as corrosion | `public/results/codebrim/failures_text.json` | [category='Rebar, spall and stain confusion'].what_happened |
| guided run: share of pixels read | 0.2487 | `video/live/guided.ndjson` | done.summary.pixels_frac |
| guided run: critical findings | 3 | `video/live/guided.ndjson` | done.summary.critical_pending |
| guided run: agent seconds | 0.712 | `video/live/guided.ndjson` | done.summary.seconds |
| step 1 entropy | 0.6931 | `video/live/guided.ndjson` | step[1].observation.entropy (rationale 'entropy 0.69 > 0.5 at L3') |
| step 4 verify IoU L1 vs L0 | 0.002 | `video/live/guided.ndjson` | review_request[level_a=1, level_b=0].iou |
| planner drill: rejected strategy | rm -rf / | `video/live/break.ndjson` | planner[status=rejected].reason |
| governor drill: segmenter p95 (ms) | 738.4 | `video/live/break.ndjson` | step[5].runtime.p95_ms['segment:accurate'] |
| governor drill: model switched to | fast | `video/live/break.ndjson` | step[5].model |
| KILL drill: halt cause | kill-switch | `video/live/kill.ndjson` | done.cause |
| Resume: steps marked replay | 5 | `video/live/resume.ndjson` | step[cause=replay] count |
| tampered token: HTTP status | 403 | `video/live/tamper.json` | status |
| tampered token: error | signature does not match | `video/live/tamper.json` | response.error |
| /api/health opencv | 5.0.0 | `video/live/health.json` | opencv |
| /api/health arch | aarch64 | `video/live/health.json` | arch |
| /api/health host | aws-lambda | `video/live/health.json` | host |
| /api/health build | stock (not COOL) | `video/live/health.json` | build |
| /api/health commit | ca690371bf3c21eb5efc61f58f9048322a41ec52 | `video/live/health.json` | commit |
| MCP: first verify call | 6 | `docs/mcp-session/exchange.jsonl` | [tool=verify][0].n |
| MCP: its IoU | 0.0 | `docs/mcp-session/exchange.jsonl` | [tool=verify][0].content[0].value.iou |
| MCP: second verify call | 18 | `docs/mcp-session/exchange.jsonl` | [tool=verify][1].n |
| MCP: its IoU | 0.719 | `docs/mcp-session/exchange.jsonl` | [tool=verify][1].content[0].value.iou |
| MCP: the budget check | 20 | `docs/mcp-session/exchange.jsonl` | [tool=runtime_stats].n |
| MCP call table: call, what came back, why | 23 rows | `docs/mcp-session/transcript.md` | 'How each measurement chose the next call' table |
| WAF: job starts | 10 | `public/docs/aws-simple.svg` | <desc> 'WAF rate limit (10 job starts per 120 s per IP)' |
| Lambda step p95 (ms), last smoke test | 67.7 | `runs/aws-smoke.log (main checkout)` | last 'INFO latency' line; run '2026-10-18T07:12:25Z' |
| stock aarch64 wheel custom HAL (GitHub arm64 runner, not Lambda) | YES (carotene (ver 0.0.1) KleidiCV (ver 26.03)) | `results/linux-aarch64/micro.json` | provenance.cv2_custom_hal |
| US bridges (FHWA NBI 2026) | 624,939 | `docs/impact-sources.md` | section 1 Figure |
| in poor condition | 41,148 | `docs/impact-sources.md` | section 1 Figure |
| routine inspection interval, months (23 CFR 650.311) | 24 | `docs/impact-sources.md` | section 2 Quote |
| photos from one drone inspection (ODOT SPR 787) | 2,536 | `docs/impact-sources.md` | section 5 Figure |
| trained segmenter | results pending | `public/results/codebrim_dacl10k/summary.json` | file absent at build time |
| live site | https://d1xg58guluup8c.cloudfront.net | `README.md` | Live demo |
| repository | https://github.com/vj-io/drishyam | `README.md / public/index.html` | GitHub link |
| health check | https://d1xg58guluup8c.cloudfront.net/api/health | `video/live/health.json` | recorded response |

### Narration and captions (from video/narration.py)

| Id | Starts | Spoken | Caption |
| --- | --- | --- | --- |
| o1 | 0:01.9 | It read 25 percent of this photo's pixels, and found three critical findings. | It read 24.9 % of this photo's pixels and found 3 critical findings. |
| o2 | 0:07.2 | It will not sign them off. A person must. | It will not sign them off. A person must. |
| t1 | 0:10.6 | Drishyam is an Agentic Vision entry to the OpenCV AI Competition, powered by AWS. | Drishyam: an Agentic Vision entry to the OpenCV AI Competition 2026, powered by AWS. |
| t2 | 0:17.7 | Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation. | Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation. |
| p1 | 0:24.1 | America has about six hundred twenty-five thousand bridges. As a rule, each gets a routine inspection at intervals of up to twenty-four months. | The US has 624,939 bridges (41,148 in poor condition); the rule: a routine inspection at intervals of up to 24 months. |
| p2 | 0:33.5 | One drone inspection of a single bridge produced more than two thousand five hundred photos. Someone has to look at every one. | One drone inspection of one bridge produced 2,536 photos. Someone has to look at every one. |
| p3 | 0:41.3 | Drishyam is for the engineer reviewing them. It zooms in only where it needs to, and hands every critical finding to a person. | Drishyam is for the engineer reviewing them. It zooms in only where it needs to, and hands every critical finding to a person. |
| l1 | 0:50.1 | The live site runs on AWS: OpenCV five, on an arm Lambda. | Live on AWS: opencv 5.0.0 · arch aarch64 · host aws-lambda |
| l2 | 0:55.5 | Watch it inspect a bridge. Faces are blurred first. | Watch it inspect a bridge. Faces are blurred first. |
| l3 | 0:59.1 | One survey look at the whole photo, at one eighth size. | One survey look at the whole photo, at one eighth size. |
| l4 | 1:02.8 | Entropy zero point six nine is above one half. Unsure, so it zooms in. | Entropy 0.69 > 0.5: unsure, so it zooms in one level. |
| l5 | 1:07.9 | Then it moves to the best region, and double-checks at full resolution. | Then it moves to the best region, and double-checks at full resolution. |
| l6 | 1:12.1 | The two looks barely overlap, zero point zero zero two. So the exposed rebar goes to a person. | The two looks barely overlap (IoU 0.002 < 0.5), so the exposed rebar goes to a person. |
| l7 | 1:19.1 | Then a second spall, and it stops when the next look would cost more than it is worth. | Then a second spall, and it stops when the next look would cost more than it is worth. |
| l8 | 1:23.7 | The agent took zero point seven seconds. The page replays it slowly. | The agent took 0.712 s; the page replays it slowly. |
| l9 | 1:28.6 | Only a person can approve. A name, Approve, and the server signs the work order. | Only a person can approve. A name, Approve, and the server signs the work order. |
| l11 | 1:33.9 | Change one character of that signature, and the server refuses it. | One character of the signature changed: HTTP 403, “signature does not match”. |
| b0 | 1:38.5 | Now, break it. | Now, break it. |
| b1 | 1:39.9 | A slowed segmenter pushes its p ninety-five to seven hundred thirty-eight milliseconds, so the governor switches to the fast model. | Slowed segmenter: p95 738 ms, so the governor switches to the fast model. |
| b2 | 1:47.7 | The planner drill sends a shell command. The allow-list rejects it, and the rule planner decides. | The planner drill sends “rm -rf /”: rejected by the allow-list; the rule planner decides. |
| b3 | 1:53.6 | KILL halts the job before its next look. Resume replays the moves so far, and carries on. | KILL: halted, cause kill-switch. Resume: 5 moves replayed, then it carries on. |
| b4 | 1:59.5 | No language model runs live. These drill planners are scripted to fail. | No language model runs live. These drill planners are scripted to fail. |
| m1 | 2:05.1 | Over MCP, Claude drove the same OpenCV tools on the same photo, one call at a time. | Recorded MCP session: Claude drives the same OpenCV tools on bridge_03, one call at a time. |
| m2 | 2:11.8 | At call six, two zoom levels overlapped by zero point zero zero. So Claude handed the rebar to a human. | Call 6: IoU 0.00 between L1 and L0, so the rebar goes to a human. |
| m3 | 2:19.2 | At call eighteen, zero point seven two confirmed a spall. At call twenty, the budget ended the search. | Call 18: IoU 0.72 confirms a spall. Call 20: the measured budget ends the search. |
| m4 | 2:26.3 | It still missed the large spall. The rules are in the prompt; the values were measured. | It still missed the large spall. The rules are in the prompt; the values were measured. |
| a0 | 2:32.2 | Every step is perception, decision and action, with a human gate at the end. | Every step is perception, decision and action, with a human gate at the end. |
| a1 | 2:37.0 | CloudFront and a rate-limiting firewall front a signed Function URL, and an arm Lambda with stock OpenCV five. | CloudFront + WAF (10 job starts per IP per 120 s) → Function URL (AWS_IAM) → Lambda arm64, stock OpenCV 5.0.0 |
| a2 | 2:45.2 | DynamoDB holds the kill flags, Secrets Manager the signing key, and CloudWatch the alarms. | DynamoDB holds the kill flags, Secrets Manager the signing key, and CloudWatch the alarms. |
| a3 | 2:51.9 | OpenCV does the looking: a pyramid, saliency, warp perspective as a virtual pan-tilt-zoom camera, a blur gate, and face blurring. | OpenCV: pyrDown pyramid · saliency (Sobel, black-hat, CLAHE) · warpPerspective virtual PTZ · Laplacian gate · YuNet + NanoDet blur |
| a4 | 3:01.0 | The live demo uses the classical segmenter. OpenCV five's new DNN engine is on the optional ONNX path. | Live: the classical segmenter. The one OpenCV-5-only call, cv2.dnn.ENGINE_NEW, is on the optional ONNX path. |
| a5 | 3:09.1 | On Lambda, a step took sixty-eight milliseconds at p ninety-five. COOL was not run. | Lambda step p95: 67.7 ms (last smoke test). COOL: not run, the Marketplace AMI could not be subscribed. |
| r1 | 3:15.4 | On two hundred synthetic photos, under a thirty percent pixel cap, it finds eighty-five percent of exhaustive tiling's recall, at twenty-three percent of the pixels. The ninety percent target was missed. | 200 synthetic photos, 30 % cap: 85 % of exhaustive tiling's recall at 23 % of the pixels. Target 90 %: missed. |
| r3 | 3:28.7 | And one downscaled pass does about as well: zero point nine four times. | One downscale does about as well: 0.94× (95 % CI 0.89–1.00). |
| r4 | 3:33.5 | On larger photos it does beat one downscale, but quarter-size tiling matches it with fewer pixels. | 4608 px: 1.12× one downscale (CI 1.02–1.22); L2 tiling 0.53 at 0.10 of the pixels, ours 0.51 at 0.28. |
| r5 | 3:40.2 | In the ablations, dropping the survey look hurts on every set. A random gaze order does as well on the smaller photos. | In the ablations, dropping the survey look hurts on every set. A random gaze order does as well on the smaller photos. |
| r6 | 3:48.0 | On two hundred real CODEBRIM photos, every method finds about one defect in five. The classical segmenter is the bottleneck. | CODEBRIM, 200 real photos: every method finds about 1 in 5 defects; its boxes at random score 0.12. |
| r7 | 3:56.6 | And on photos with no damage, it still reports something sixty-seven percent of the time. | 51 photos with no damage: something reported on 67 % of them (6.35 boxes per image). |
| r8 | 4:02.3 | Results for a trained segmenter are pending. | Trained segmenter: results pending. |
| f1 | 4:05.8 | The failures stay in the open: graffiti read as spalling, a formwork line as a crack, glare, and blur. | The failures stay in the open: graffiti read as spalling, a formwork line as a crack, glare, and blur. |
| f2 | 4:12.5 | Each names the rule that would catch it. None of them is built yet. | Each names the rule that would catch it. None of them is built yet. |
| f3 | 4:16.3 | On a real photo, exposed rebar was reported as corrosion. | CODEBRIM image_0001435 (text only): ground truth exposed rebar, reported corrosion. |
| x1 | 4:21.5 | Pixels, not millimetres. AI-assisted, not an inspection. Only a person signs. | Pixels, not millimetres. AI-assisted, not an inspection. Only a person signs. |
| y1 | 4:28.1 | Make test, make eval, and one script that deploys, tests and destroys the AWS stack. | Make test, make eval, and one script that deploys, tests and destroys the AWS stack. |
| e1 | 4:35.2 | Drishyam. It looks where it matters, and asks a human first. | Drishyam. It looks where it matters, and asks a human first. |
<!-- generated:end -->
