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
| `public/video/drishyam.vtt` | captions, the same cues as the burned-in ones (written by the same build as the MP4) | yes |
| `public/video/poster.jpg` | the end card, settled | yes |
| `brag-output/drishyam-short.mp4` | 15-20 s 9:16 cut: the cold open, ending on the settled APPROVED stamp | no (brag-output/ is ignored) |
| `brag-output/standin/film/` | `build.py --standin`: the film with every pending piece on (stand-ins): `drishyam-standin.mp4`, `drishyam-standin.vtt`, `poster-standin.jpg`, `plan-tables.md` | no |

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
captions and the generated half of this file, makes the score, renders with the fast-render skill into
`brag-output/work/` and then moves the MP4, the captions and the poster into `public/video/` together (so the
VTT always comes from the same layout as the MP4; a stopped render leaves `public/video/` as it was), and cuts
the 9:16 short. `--stills T ...` renders checking stills only; `--audit` runs the activity audit only; `--layout`
stops after the voice, the layout and this file. The build stops if the film would run over 300 s.

```bash
~/.claude/brag-scaled-venv/bin/python video/build.py --check-template   # every pending piece on, at full length: must fit 300 s
~/.claude/brag-scaled-venv/bin/python video/build.py --make-standin     # stand-in real-photo traces -> brag-output/standin/
~/.claude/brag-scaled-venv/bin/python video/build.py --standin          # the whole film with the stand-ins -> brag-output/standin/film/
```

`--standin` renders the film with the trained-segmenter stand-in (below) and the stand-in real-photo run, into
`brag-output/standin/film/` (MP4, captions, poster, `plan-tables.md`); it never touches `public/video/` or this file,
so the committed poster and captions are never from a stand-in build. A stand-in film is marked so it cannot pass for
the real one: a red strip "STAND-IN DATA, NOT FOR PUBLICATION" across the top of every frame, `-standin` in every file
name, and the same words in the MP4's metadata (`comment`). `infra/drishyam_stack.py` refuses to deploy while any MP4
under `public/` carries the mark or the name, so a renamed copy is refused too. Do not share a stand-in MP4.

## Private inputs, and the build without them

Two chapters read files that are not in the repository (`private-assets/`, git-ignored), from
`$DRISHYAM_PRIVATE_DIR` (default `/Users/mac/Documents/Projects/opencv/private-assets`). The build reads them in
place: the page gets the photos as `file://` URLs and the portrait scripts through the generated, git-ignored
`brag-output/work/private.js`; nothing is copied into a tracked path. Both team members agreed that their photos
and portraits appear in the public film.

| Input | Used by | Without it |
| --- | --- | --- |
| `team/photo-1.jpg`, `team/photo-2.jpg`, `team/team.json` (photo-1 Vaibhav Jangid, photo-2 Bhavya Dhoot; the names must match README.md's Team, in order) | Team | the two name cards |
| `team/out/portraits.py`, `portraits-data.js`, `portraits-chapter.js` (the OpenCV line portraits; `team/out/README.md`) | Team | the two name cards |
| `real-photos/clean/<image>.jpg` (the team's wall photos, metadata stripped) | Real photo | no real-photo chapter (the film is 10 s shorter) |

Anyone else gets a complete film: name cards, no real-photo chapter (`DRISHYAM_PRIVATE_DIR=/nonexistent` shows it).

**Team chapter (photo to drawing).** Each person in turn: the real photo, framed on its plate and held with the
camera settled on it for about a second; then the Drishyam gaze window (white frame, amber corners,
`LOOK n · L k · 1:2^k` in the corner away from that person's callouts, dimensions in the photo's own pixels) flies in
and takes five looks, coarse to fine (outline, face and hair, eyes, nose and mouth, collar), then a shading pass
(tagged `SHADING · L k`: it is traced at a coarser level than the looks before it); inside each the photo dissolves
while its lines draw on behind an amber nib, until the shading pass has replaced the whole photo with the drawing; the
name and the role type into the title block. The camera glides to the second plate (the first plate's text dims to
40 % while it is at the edge of the view), holds the second photo the same way (`hold`, 1 s; `t2` waits 0.95 s more
for it) and does the same. As `t2` starts both plates are framed, and each title block lights as the voice says the
name; the parts key and the tagline "Drawn by OpenCV 5, coarse to fine" close it. `t1` carries both transformations
(about 2.3 s of looks each, `pace` 0.75), `t2` the names; `film.js` throws if both plates are not framed before the
first name is said. The strip says how the drawings were made: "our photos, drawn by OpenCV 5: Canny, findContours,
approxPolyDP", and the rail note says what was not automatic: the feature regions (the crop and the boxes for eyes,
brows, nose, mouth and collar in `portraits.py`) were placed by hand; every line is traced by OpenCV at the level the
tag names. The build checks that `portraits.py` makes those three calls and says OpenCV 5. The photo layer, the
per-look dissolve, `pace`, `hold` and the voice timing (`said1At`, `said2At`) are opt-in options of
`portraits-chapter.js` (its header lists them; `portraits.html` uses none, so it renders as before; the file before
these options is kept beside it as `portraits-chapter.before-photo.js`).

**Real-photo chapter.** It appears when `video/live/real.ndjson` (the NDJSON stream of one Drishyam job on one of the
team's wall photos, embedded images removed) and `video/live/real.json` exist, and the photo is in the private folder:

```json
{"image": "IMG_2332", "engine": "classical-v1", "profile": "fixed costs (linux-aarch64), 30 % pixel cap",
 "seen": {"say": "the hairline crack", "classes": ["crack"], "by": "the team, by eye: not ground truth",
          "paths": [[[465, 15], [660, 375], [735, 600], [795, 720], [1020, 750], [1140, 818], [1275, 833], [1395, 930],
                     [1560, 1050], [1770, 1155], [2025, 1350], [2145, 1590], [2280, 1800], [2340, 1965], [2430, 2250]]]}}
```

`engine` is `classical-v1` or `onnx:<model>` (checked against the trace's cost table: a `-dacl10k` table means ONNX);
`seen` is optional and is the team's reading of the photo, not ground truth (the paths are the crack as traced by eye,
in full-resolution pixels; the ones above are IMG_2332's). The photo is labelled "REAL PHOTO (ours): painted plaster
wall, no ground truth"; the chapter shows the overview (the photo as its pyramid level, a scan line, the candidate
regions), each look as the gaze window, each finding's box and class, why it stopped, and the cards ENGINE, LOOKS,
PIXELS READ, REPORTED and BY EYE; the team's line draws on (thin, over a light dimming, so the crack stays visible
beside it) and a FOUND or MISSED stamp lands. A finding counts as on what the team sees when its box touches the
traced line; every other finding is "off the team's line" (never a false alarm: there is no ground truth).
The line `q1` is built from the trace in `video/narration.py` and asserted against it (looks, findings and pixels
against `done.summary`, "saw nothing worth a look" only with no candidate region), in one of three forms. The share of
pixels is said as the PIXELS READ card prints it: one decimal under 10 % ("one point six percent", "1.6 %"), whole
percent above.

- no look: "Our own photo of a painted wall, no ground truth: the classical engine read one point six percent of it,
  saw nothing worth a look, and missed the hairline crack."
- looks, nothing reported: "... took N looks at P of the pixels, reported nothing, and missed ..."
- findings: "... took N looks at P of the pixels and found (or missed) the hairline crack, with K findings off the team's
  line." (without `seen`: "... and reported a crack and spalling.")

To record the final trace (the owner, with the final engine): `uv run drishyam run
$DRISHYAM_PRIVATE_DIR/real-photos/clean/IMG_2332.jpg --cost-model fixed --out runs/real` (for the trained segmenter add
`DRISHYAM_ONNX=<model>` and `--micro-arm <its -dacl10k table>`), then copy `runs/real/<job>/trace.jsonl` to
`video/live/real.ndjson` with the `*_b64` fields removed (as `build.py --make-standin` does), write `video/live/real.json`,
run `--check-template`, then the build.

The stand-ins (`build.py --make-standin`, git-ignored under `brag-output/standin/`): the classical engine on IMG_2332
(the render's real photo: one L3 overview, no candidate region, nothing found, the crack missed) and on IMG_2330 (two
looks, a crack on the team's line and a spalling finding off it; used only by `--check-template`, as the longer line).

The trained segmenter is "results pending" until `public/results/codebrim_dacl10k/summary.json` exists. Then the
same build quotes it, and only like for like: its summary must say `onnx:`, governor off and not confounded, and
the classical CODEBRIM summary it is compared with must be charged the same machine's cost table on the same
data (the published one, or the classical re-run `evals/model_eval.py` makes in the main checkout's
`runs/model-eval/classical-codebrim/` when the published one was charged another machine's costs). The line
says governor off and same cost table, its recall against the classical engine's, its chance recall, and where
tiling still finds more; a near-duplicate count above zero is spoken, and the board shows the count either way.
Any failed check stops the build. `--check-template` tries all of this on a stand-in summary (the classical
numbers relabelled), with the real-photo chapter on each stand-in trace (or the owner's `video/live/real.*`), checks
the longest film still fits 300 s, and checks that a summary charged another machine's costs is refused. Last run:
296.3 s with the IMG_2332 stand-in, 298.5 s with the longer IMG_2330 line (cap 300). Claims whose truth depends on the data are asserted in `video/narration.py`
and `video/build.py`, so a re-run that changes them stops the build instead of shipping a sentence that is no
longer true; words and figures typed into the scene text (citations, the competition name, the limits, the
team) are checked against their files and listed below.

The captions are the spoken words. `video/narration.py` builds each line from parts that are either plain text
or a (spoken, printed) pair, so the caption can differ from the voice only in how a number or code name is
printed ("zero point nine four" / "0.94", "warp perspective" / "warpPerspective"). Exact figures, sources and
the extra detail (the HTTP 403 body, `rm -rf /`, the CI bounds) are on the boards and rail cards, not in the
captions.

The live footage is not re-recorded by the build (the WAF allows 10 job starts per IP per 120 s). To record it
again (4 job starts, about 3 minutes):

```bash
~/.claude/brag-scaled-venv/bin/python video/record.py            # screencast + event log -> brag-output/work/rec/, traces -> video/live/
~/.claude/brag-scaled-venv/bin/python video/record.py --health   # /api/health response -> video/live/health.json
~/.claude/brag-scaled-venv/bin/python ~/.claude/plugins/cache/brag-scaled/brag-scaled/0.8.0-scaled/skills/motion-kit/scripts/walk.py record brag-output/work/walk-plan.json --out brag-output/work/walkrec
```

## Length, with everything on

The committed tables below come from the plain build (portraits on, no real-photo trace yet, trained segmenter
pending). With every pending piece on (`--standin`: the trained-segmenter stand-in line, the IMG_2332 stand-in run)
the film runs 296.3 s:

| Chapter | Starts | Ends | Lines |
| --- | --- | --- | --- |
| Cold open | 0:00.0 | 0:09.0 | o1, o2 |
| Team (photo to drawing) | 0:09.0 | 0:23.3 | t1, t2 |
| Problem and user | 0:23.3 | 0:44.7 | p1, p2, p3 |
| Live demo | 0:44.7 | 1:34.9 | l1-l9, l11 |
| Failure drills | 1:34.9 | 1:56.8 | b0-b4 |
| An LLM over MCP | 1:56.8 | 2:21.4 | m1-m4 |
| Architecture | 2:21.4 | 3:02.7 | a0-a5 |
| Results (with the trained-segmenter line) | 3:02.7 | 4:05.3 | r1, r3-r8 |
| Real photo (ours) | 4:05.3 | 4:16.3 | q1 |
| Failure cases | 4:16.3 | 4:35.2 | f1-f3 |
| Limits | 4:35.2 | 4:42.8 | x1 |
| Reproduce | 4:42.8 | 4:49.3 | y1 |
| End | 4:49.3 | 4:56.3 | e1 |

To fit the real-photo chapter and the trained-segmenter line under 300 s, figures are said the short way ("every two
years" for 24 months, "twenty-five hundred", "under ninety milliseconds", "two times in three", the trained segmenter's
recall in whole percent; the exact figures stay on the boards), the ablation line says only what the table supports
("only dropping the survey look loses recall; the others lose none"), and the gaps between lines are 0.3 s (0.35 in
the team chapter and the drills, where the footage warp needs it; with the portraits `t2` also waits 0.95 s for the
second photo's hold). The held-out re-run (c36e375) changed three
claims, and the narration with them: the downscale is now slightly ahead within the noise (0.96x, CI 0.91-1.02), and
two of the four failure rules are built (the straight-line test, which still let the shown formwork line through at
a coarse look, and the glare gate's re-shoot listing); the plates say "built" and "not yet" from `failures.json`.

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
- The trained dacl10k segmenter's results are pending and not quoted (see above for when the line switches on).

## Not shown, and why

- The site's Benchmarks sheet: the deployed build (ca69037) still shows the numbers from before the final
  re-run; the video draws its own chart from the final `public/results/*.json` instead.
- A CloudWatch dashboard and an owner-recorded team clip: no screenshot or footage exists. The team is shown from
  their photos and the OpenCV portraits drawn from them (private inputs, above), or as name cards without them.
- A test pass count: `make test` was not run for this video, so no count is claimed; the reproduce sheet shows
  the commands only.
- COOL timings: COOL was not run (README.md, "COOL on Graviton (not run)").

## Known on screen

- `docs/architecture.png` is shown as committed. Its footnote cites `segment.py:239` for `cv2.dnn.ENGINE_NEW`;
  the call is now at `drishyam/segment.py:298`. The claim itself (the one OpenCV-5-only call) still holds.
- In the recorded page, the guided run's "FIG. 2.3.2 · ESCALATE · L2" label sits on the vertical "2048 px"
  dimension tag for a moment (cold open, about 0:04-0:06, and the live chapter). It is the deployed site's own
  layout; the fix belongs in the site, before the next recording.
- Clicking "Try KILL" clears the last job, so the page re-lays out and scrolls back to the viewer; at 5x speed the
  "How it works" sheet passes through the window for a fraction of a second. It is the real page doing it.

<!-- generated:start -->
### Requirement checklist (timestamps from this build)

| Requirement | Covered by | From | To |
| --- | --- | --- | --- |
| At most 5 minutes, 1920x1080 | open, end | 0:00.0 | 4:36.0 |
| Narrated (offline Kokoro voice af_heart) with burned-in captions; the same cues in public/video/drishyam.vtt | open, end | 0:00.0 | 4:36.0 |
| The team: Vaibhav Jangid, Bhavya Dhoot. Each real photo, then the window's looks turn it into OpenCV line art (Canny, findContours, approxPolyDP); name cards without the private inputs | team | 0:09.0 | 0:23.3 |
| Problem and user, with sources on screen | problem | 0:23.3 | 0:44.7 |
| /api/health on the live site: opencv 5.0.0, aarch64, aws-lambda | l1 | 0:45.2 | 0:50.1 |
| The application working: the guided run on bridge_03 ('Watch it inspect a bridge'), recorded live, with what it missed | l2, l3, l4, l5, l6, l7, l8 | 0:50.4 | 1:25.3 |
| Plain-words trace (the NOTE and 'What it decided and why' panels) | l3, l4, l5, l6, l7 | 0:55.4 | 1:21.7 |
| Human sign-off: reviewer name, Approve, APPROVED stamp, work order signed | l9 | 1:25.6 | 1:30.4 |
| Human control: a tampered work-order token refused (403) | l11 | 1:30.7 | 1:34.4 |
| Failure handling: 'Now break it' (slowed segmenter -> governor; malformed planner reply -> allow-list) | b0, b1, b2 | 1:35.3 | 1:46.0 |
| Human control: 'Try KILL' (KILL halts with kill-switch, Resume replays) | b3 | 1:46.4 | 1:51.6 |
| Honesty: no LLM runs in the live demo; the drill planners are scripted | b4 | 1:52.0 | 1:56.3 |
| An LLM driving the OpenCV tools over MCP: IoU 0.00 at call 6 -> human; IoU 0.72 at call 18 -> confirmed; call 20 budget | mcp | 1:56.8 | 2:21.4 |
| Architecture: docs/agent-loop.svg, public/docs/aws-simple.svg, docs/architecture.png (AWS path and OpenCV layer) | arch | 2:21.4 | 3:02.7 |
| Principal results: recall vs pixels, with one downscale, the tiling family and chance | r1, r3 | 3:03.2 | 3:21.6 |
| Larger photos (4608 px): beats one downscale (which reads far fewer pixels), L2 tiling matches it with fewer pixels | r4 | 3:21.9 | 3:29.7 |
| Ablations (public/results/ablations.json) | r5 | 3:30.0 | 3:38.6 |
| Real photos (CODEBRIM), said plainly weak; no CODEBRIM image shown | r6 | 3:38.9 | 3:47.2 |
| False alarms on synthetic photos with no damage (public/results/clean/summary.json) | r7 | 3:47.5 | 3:53.1 |
| Trained segmenter: 'results pending' until public/results/codebrim_dacl10k/summary.json exists | r8 | 3:53.4 | 3:55.6 |
| Failure cases: failures.json plates and their rules; one CODEBRIM case in words only | failures | 3:56.0 | 4:14.9 |
| Limits and responsible use | limits | 4:14.9 | 4:22.5 |
| Reproduce: make test, make eval, scripts/aws.sh | repro | 4:22.5 | 4:29.0 |
| End card: live URL, repository, team | end | 4:29.0 | 4:36.0 |

### Chapters (from this build)

| Chapter | Starts | Ends | Lines |
| --- | --- | --- | --- |
| Cold open | 0:00.0 | 0:09.0 | o1, o2 |
| Team | 0:09.0 | 0:23.3 | t1, t2 |
| Problem and user | 0:23.3 | 0:44.7 | p1, p2, p3 |
| Live demo | 0:44.7 | 1:34.9 | l1, l2, l3, l4, l5, l6, l7, l8, l9, l11 |
| Failure drills | 1:34.9 | 1:56.8 | b0, b1, b2, b3, b4 |
| An LLM over MCP | 1:56.8 | 2:21.4 | m1, m2, m3, m4 |
| Architecture | 2:21.4 | 3:02.7 | a0, a1, a2, a3, a4, a5 |
| Results | 3:02.7 | 3:56.0 | r1, r3, r4, r5, r6, r7, r8 |
| Failure cases | 3:56.0 | 4:14.9 | f1, f2, f3 |
| Limits | 4:14.9 | 4:22.5 | x1 |
| Reproduce | 4:22.5 | 4:29.0 | y1 |
| End | 4:29.0 | 4:36.0 | e1 |

### Every number and URL, with its source

| What | Value | Source file | Field |
| --- | --- | --- | --- |
| synthetic photos | 200 | `public/results/summary.json` | n_images |
| synthetic photo size (px) | [2048, 1536] | `public/results/summary.json` | dataset |
| pixel cap of the headline point | 0.3 | `public/results/summary.json` | headline_stats.budget |
| recall vs exhaustive tiling | 0.8706 | `public/results/summary.json` | headline_stats.recall_ratio_vs_tiling |
| its 95 % CI | [0.8109, 0.9284] | `public/results/summary.json` | headline_stats.recall_ratio_vs_tiling_ci |
| target recall ratio | 0.9 | `public/results/summary.json` | headline_stats.target.recall_ratio |
| target met? | false | `public/results/summary.json` | headline_stats.meets_target |
| recall vs one downscale | 0.9642 | `public/results/summary.json` | headline_stats.recall_ratio_vs_downscale |
| its 95 % CI | [0.9097, 1.0248] | `public/results/summary.json` | headline_stats.recall_ratio_vs_downscale_ci |
| headline point: pixels, recall | [0.261, 0.4778] | `public/results/summary.json` | policies.drishyam[pixels_frac <= 0.3][-1] (as public/js/results.js headline()) |
| exhaustive tiling: pixels, recall | [1.6667, 0.5488] | `public/results/summary.json` | policies.tiling[max pixels_frac] |
| one downscale: pixels, recall | [0.25, 0.4956] | `public/results/summary.json` | policies.downscale[0] |
| chart: every point and its chance tick | all rows | `public/results/summary.json` | policies.*[].pixels_frac, recall, recall_ci, chance_recall |
| cost table every result set is charged | arm64 (linux-aarch64) | `public/results/{,large/,codebrim/,clean/}summary.json` | cost_model.source |
| 4608 px photos | 100 | `public/results/large/summary.json` | n_images |
| 4608 px photo size (px) | [4608, 3456] | `public/results/large/summary.json` | dataset |
| 4608 px: recall vs one downscale | 1.1438 | `public/results/large/summary.json` | headline_stats.recall_ratio_vs_downscale |
| its 95 % CI | [1.0414, 1.2538] | `public/results/large/summary.json` | headline_stats.recall_ratio_vs_downscale_ci |
| 4608 px: full width / L2 width (the tiling level called quarter-size) | 4 | `public/results/large/summary.json` | dataset size, halved twice by pyrDown's rule (job.levels in video/live/guided.ndjson) |
| 4608 px headline point: pixels, recall | [0.2884, 0.5178] | `public/results/large/summary.json` | policies.drishyam[pixels_frac <= 0.3][-1] |
| 4608 px L2 tiling: pixels, recall | [0.0988, 0.5296] | `public/results/large/summary.json` | policies.tiling[min pixels_frac] (L2) |
| 4608 px one downscale: pixels, recall | [0.0494, 0.4527] | `public/results/large/summary.json` | policies.downscale[0] |
| CODEBRIM photos | 200 | `public/results/codebrim/summary.json` | n_images |
| CODEBRIM recall (ours) | 0.1941 | `public/results/codebrim/summary.json` | policies.drishyam[pixels_frac <= 0.3][-1].recall |
| CODEBRIM recall, every method (min, max) | [0.1941, 0.2226] | `public/results/codebrim/summary.json` | policies.{downscale,tiling,drishyam head}.recall |
| CODEBRIM: its boxes at random | 0.1223 | `public/results/codebrim/summary.json` | policies.drishyam[head].chance_recall |
| 'about 1 in N' (N = round(1 / mid of the range), as the site's hero card) | 5 | `public/results/codebrim/summary.json` | derived from the range above |
| synthetic photos with no damage | 51 | `public/results/clean/summary.json` | n_images (dataset: 'synthetic clean set') |
| share of them with any report (ours, 30 % cap) | 0.6667 | `public/results/clean/summary.json` | false_alarms['drishyam@0.3'].images_with_any |
| boxes per image (ours) | 7.08 | `public/results/clean/summary.json` | false_alarms['drishyam@0.3'].boxes_per_image |
| critical per image (ours) | 0.02 | `public/results/clean/summary.json` | false_alarms['drishyam@0.3'].critical_per_image |
| boxes per image (one downscale) | 3.0 | `public/results/clean/summary.json` | false_alarms.downscale.boxes_per_image |
| boxes per image (L0 tiling) | 36.8 | `public/results/clean/summary.json` | false_alarms['tiling@L0'].boxes_per_image |
| pixel cap of those numbers | 0.3 | `public/results/clean/summary.json` | false_alarms key 'drishyam@0.3' |
| ablation 2048 px synthetic: no-survey / full | 0.9071 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-survey'].ratio_vs_full |
| ablation 2048 px synthetic: no-verify / full | 1.031 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-verify'].ratio_vs_full |
| ablation 2048 px synthetic: no-uncertainty / full | 1.0093 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-uncertainty'].ratio_vs_full |
| ablation 2048 px synthetic: governor-on / full | 1.0 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['governor-on'].ratio_vs_full |
| ablation 2048 px synthetic: random-order-s0 / full | 0.9845 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['random-order-s0'].ratio_vs_full |
| ablation 2048 px synthetic: random-order-s1 / full | 0.9721 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['random-order-s1'].ratio_vs_full |
| ablation 2048 px synthetic: random-order-s2 / full | 0.9969 | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['random-order-s2'].ratio_vs_full |
| ablation 2048 px synthetic: no-survey verdict | below the full agent | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-survey'].vs_full |
| ablation 2048 px synthetic: no-verify verdict | above the full agent | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-verify'].vs_full |
| ablation 2048 px synthetic: no-uncertainty verdict | no measurable difference | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['no-uncertainty'].vs_full |
| ablation 2048 px synthetic: governor-on verdict | no measurable difference | `public/results/ablations.json` | datasets.synth200.by_cap['0.3']['governor-on'].vs_full |
| ablation 2048 px synthetic: random order / full, pooled 95 % CI | [0.97, 1.0] | `public/results/ablations.json` | datasets.synth200.random_order[0] |
| ablation 4608 px synthetic: no-survey / full | 0.8571 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-survey'].ratio_vs_full |
| ablation 4608 px synthetic: no-verify / full | 1.0057 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-verify'].ratio_vs_full |
| ablation 4608 px synthetic: no-uncertainty / full | 1.0057 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-uncertainty'].ratio_vs_full |
| ablation 4608 px synthetic: governor-on / full | 1.0 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['governor-on'].ratio_vs_full |
| ablation 4608 px synthetic: random-order-s0 / full | 0.8571 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['random-order-s0'].ratio_vs_full |
| ablation 4608 px synthetic: random-order-s1 / full | 0.8686 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['random-order-s1'].ratio_vs_full |
| ablation 4608 px synthetic: random-order-s2 / full | 0.8857 | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['random-order-s2'].ratio_vs_full |
| ablation 4608 px synthetic: no-survey verdict | below the full agent | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-survey'].vs_full |
| ablation 4608 px synthetic: no-verify verdict | no measurable difference | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-verify'].vs_full |
| ablation 4608 px synthetic: no-uncertainty verdict | no measurable difference | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['no-uncertainty'].vs_full |
| ablation 4608 px synthetic: governor-on verdict | no measurable difference | `public/results/ablations.json` | datasets.synth_large.by_cap['0.3']['governor-on'].vs_full |
| ablation 4608 px synthetic: random order / full, pooled 95 % CI | [0.82, 0.92] | `public/results/ablations.json` | datasets.synth_large.random_order[0] |
| ablation CODEBRIM: no-survey / full | 0.6636 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-survey'].ratio_vs_full |
| ablation CODEBRIM: no-verify / full | 1.0275 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-verify'].ratio_vs_full |
| ablation CODEBRIM: no-uncertainty / full | 0.9939 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-uncertainty'].ratio_vs_full |
| ablation CODEBRIM: governor-on / full | 1.0 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['governor-on'].ratio_vs_full |
| ablation CODEBRIM: random-order-s0 / full | 0.9572 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['random-order-s0'].ratio_vs_full |
| ablation CODEBRIM: random-order-s1 / full | 0.9602 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['random-order-s1'].ratio_vs_full |
| ablation CODEBRIM: random-order-s2 / full | 0.9694 | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['random-order-s2'].ratio_vs_full |
| ablation CODEBRIM: no-survey verdict | below the full agent | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-survey'].vs_full |
| ablation CODEBRIM: no-verify verdict | no measurable difference | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-verify'].vs_full |
| ablation CODEBRIM: no-uncertainty verdict | no measurable difference | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['no-uncertainty'].vs_full |
| ablation CODEBRIM: governor-on verdict | no measurable difference | `public/results/ablations.json` | datasets.codebrim.by_cap['0.3']['governor-on'].vs_full |
| ablation CODEBRIM: random order / full, pooled 95 % CI | [0.92, 1.0] | `public/results/ablations.json` | datasets.codebrim.random_order[0] |
| failure plate s0162 | Graffiti read as damage | `public/results/failures.json` | [image=failures/f01.jpg].category, rule_that_would_catch_it |
| failure plate s0139 | Formwork line read as crack | `public/results/failures.json` | [image=failures/f03.jpg].category, rule_that_would_catch_it |
| failure plate s0040 | Glare caught by the quality gate | `public/results/failures.json` | [image=failures/f05.jpg].category, rule_that_would_catch_it |
| failure plate s0085 | Crack hidden in blur | `public/results/failures.json` | [image=failures/f07.jpg].category, rule_that_would_catch_it |
| failure rules not built yet (of the four shown) | 2 | `public/results/failures.json` | rule_that_would_catch_it without 'Implemented (...)' |
| real-photo failure (text only) | image_0001435: exposed_rebar reported as corrosion | `public/results/codebrim/failures_text.json` | [category='Rebar, spall and stain confusion'].what_happened |
| guided run: share of pixels read | 0.2487 | `video/live/guided.ndjson` | done.summary.pixels_frac |
| guided run: critical findings | 3 | `video/live/guided.ndjson` | done.summary.critical_pending |
| guided run: agent seconds | 0.712 | `video/live/guided.ndjson` | done.summary.seconds |
| survey look: full width / L3 width | 8 | `video/live/guided.ndjson` | job.levels[0][0] / job.levels[3][0] |
| privacy pass: faces, people, vehicles blurred | [0, 0, 0] | `video/live/guided.ndjson` | job.privacy |
| step 2 entropy (the zoom decision) | 0.6931 | `video/live/guided.ndjson` | step[1].observation.entropy; step[2] rationale 'entropy 0.69 > 0.5 at L3' |
| entropy threshold | 0.5 | `video/live/guided.ndjson` | step[cause=entropy>tau].rationale |
| step 4 verify: mask IoU L1 vs L0 | 0.002 | `video/live/guided.ndjson` | review_request[level_a=1, level_b=0].iou |
| verify threshold | 0.5 | `video/live/guided.ndjson` | review_request reason 'unconfirmed: iou 0.002 < 0.5 between L1 and L0' |
| why the rebar goes to a person | exposed_rebar: always critical | `video/live/guided.ndjson` | finding f1.critical_rule; its first review_request precedes the verify |
| unvisited candidates at the stop | ["r3", "r4"] | `video/live/guided.ndjson` | done.unvisited |
| ground truth inside them that no finding touches | ["crack"] | `public/samples/index.json` | bridge_03 boxes inside overview.candidates[unvisited] |
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
| /api/health segmenter | classical-v1 | `video/live/health.json` | segmenter |
| MCP: first verify call | 6 | `docs/mcp-session/exchange.jsonl` | [tool=verify][0].n |
| MCP: its IoU | 0.0 | `docs/mcp-session/exchange.jsonl` | [tool=verify][0].content[0].value.iou |
| MCP: second verify call | 18 | `docs/mcp-session/exchange.jsonl` | [tool=verify][1].n |
| MCP: its IoU | 0.719 | `docs/mcp-session/exchange.jsonl` | [tool=verify][1].content[0].value.iou |
| MCP: the budget check | 20 | `docs/mcp-session/exchange.jsonl` | [tool=runtime_stats].n |
| MCP: job id | j_90a925 | `docs/mcp-session/exchange.jsonl` | [tool=open_image].content[0].value.job |
| MCP: call that returned crops/img-004-1.jpg, its level, its label | [4, 1, "three rust bars"] | `docs/mcp-session/exchange.jsonl; docs/mcp-session/transcript.md` | [content.file=crop].n, .args.level; the label is in that call's 'why' cell |
| MCP: call that returned crops/img-016-1.jpg, its level, its label | [16, 1, "a spall"] | `docs/mcp-session/exchange.jsonl; docs/mcp-session/transcript.md` | [content.file=crop].n, .args.level; the label is in that call's 'why' cell |
| MCP: the large spall it missed (box) | 1326,1296,570,529 | `docs/mcp-session/transcript.md` | '(box x,y,w,h)' after the call table |
| MCP call table: call, what came back, why | 23 rows | `docs/mcp-session/transcript.md` | 'How each measurement chose the next call' table |
| WAF: job starts | 10 | `public/docs/aws-simple.svg` | <desc> 'WAF rate limit (10 job starts per 120 s per IP)' |
| Lambda step p95 (ms), every smoke run (min, max) | [67.7, 88.0] | `runs/aws-smoke.log (main checkout)` | 'INFO latency' lines, runs 2026-10-17T23:12:36Z, 2026-10-18T07:09:20Z, 2026-10-18T07:12:25Z |
| smoke runs | 3 | `runs/aws-smoke.log (main checkout)` | count of 'INFO latency' lines |
| steps per smoke run | [6] | `runs/aws-smoke.log (main checkout)` | 'over N steps' |
| work-order signature | HMAC-SHA256 | `docs/security.md` | 'Tampered or replayed work order' row |
| stock aarch64 wheel custom HAL (GitHub arm64 runner, not Lambda) | YES (carotene (ver 0.0.1) KleidiCV (ver 26.03)) | `results/linux-aarch64/micro.json` | provenance.cv2_custom_hal |
| US bridges (FHWA NBI 2026) | 624,939 | `docs/impact-sources.md` | section 1 Figure |
| in poor condition | 41,148 | `docs/impact-sources.md` | section 1 Figure |
| routine inspection interval, months (23 CFR 650.311) | 24 | `docs/impact-sources.md` | section 2 Quote |
| photos from the largest drone inspection in the study (ODOT SPR 787) | 2,536 | `docs/impact-sources.md` | section 5 Figure |
| photos from the other bridges in that table (min, max) | [197, 401] | `docs/impact-sources.md` | section 5 Figure |
| bridges flown in the study | 6 | `docs/impact-sources.md` | section 5 Figure |
| trained segmenter | results pending | `public/results/codebrim_dacl10k/summary.json` | absent at build time |
| team portraits: the OpenCV calls in the header | ["cv2.Canny", "cv2.findContours", "cv2.approxPolyDP"] | `$DRISHYAM_PRIVATE_DIR/team/out/portraits.py` | cv2 calls in the script (OpenCV 5); private, git-ignored |
| team photos: photo-1, photo-2 | ["Vaibhav Jangid", "Bhavya Dhoot"] | `$DRISHYAM_PRIVATE_DIR/team/team.json` | photo-1.name, photo-2.name (both agreed to appear) |
| competition | OpenCV AI Competition 2026, powered by AWS | `README.md` | line 3 |
| entry | Agentic Vision | `docs/PRD.md` | awards line |
| name | Sanskrit: that which is seen | `README.md` | line 5 |
| team and roles | ["Vaibhav Jangid, lead engineer (agent policy, OpenCV pipeline, evaluation, AWS deployment)", "Bhavya Dhoot, design, demo and presentation (site design, video)"] | `README.md` | Team |
| problem-sheet citations | ["FHWA, Bridge Condition by Highway System 2026", "23 CFR 650.311", "SPR 787", "Table 4.1"] | `docs/impact-sources.md` | sections 1, 2, 5 |
| MCP client | Claude (Opus 5.5) inside Claude Code | `docs/mcp-session/transcript.md` | line 4 |
| Lambda CPU | arm64 (Graviton) | `docs/aws.md` | line 4 |
| health card: what the Lambda sits behind | behind CloudFront | `README.md` | Live demo line |
| reproduce sheet: what the CDK stack is | Lambda arm64 behind CloudFront and WAF | `infra/drishyam_stack.py` | stack description |
| cold-open title | ONE PHOTO, A FEW LOOKS, ONE HUMAN | `public/index.html` | the site's figure note |
| limits, in the site's words | ["Pixels, not millimetres", "AI-assisted, not an inspection", "Only a person can approve", "Its licence does not allow sharing the photos"] | `public/index.html` | footer note, Human sign-off, the CODEBRIM citation |
| reproduce: what make test runs | unit, API, MCP, CLI and eval tests (same as CI) | `Makefile` | 'test:' comment |
| footage recorded on | 2026-10-18 | `brag-output/work/rec/events.json` | file date |
| live site | https://d1xg58guluup8c.cloudfront.net | `README.md` | Live demo |
| repository | https://github.com/vj-io/drishyam | `README.md / public/index.html` | GitHub link |
| health check | https://d1xg58guluup8c.cloudfront.net/api/health | `video/live/health.json` | recorded response |

### Narration and captions (from video/narration.py)

| Id | Starts | Spoken | Caption |
| --- | --- | --- | --- |
| o1 | 0:01.7 | It read a quarter of this photo's pixels, and found three critical findings. | It read a quarter of this photo's pixels, and found 3 critical findings. |
| o2 | 0:06.2 | It will not sign them off. A person must. | It will not sign them off. A person must. |
| t1 | 0:09.5 | Drishyam, an Agentic Vision entry to the OpenCV AI Competition, powered by AWS. | Drishyam, an Agentic Vision entry to the OpenCV AI Competition, powered by AWS. |
| t2 | 0:17.5 | Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation. | Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation. |
| p1 | 0:23.8 | America has about six hundred twenty-five thousand bridges, each due a routine inspection at least every two years. | America has about 625,000 bridges, each due a routine inspection at least every two years. |
| p2 | 0:31.0 | In an Oregon study, the largest drone inspection produced more than twenty-five hundred photos of one bridge. | In an Oregon study, the largest drone inspection produced more than 2,500 photos of one bridge. |
| p3 | 0:37.6 | Drishyam is for the engineer reviewing them: it zooms in where needed, and hands critical findings to a person. | Drishyam is for the engineer reviewing them: it zooms in where needed, and hands critical findings to a person. |
| l1 | 0:45.2 | The live site runs on AWS: OpenCV five, on an arm Lambda. | The live site runs on AWS: OpenCV 5, on an arm Lambda. |
| l2 | 0:50.4 | Watch it inspect a bridge. A privacy pass runs first; nothing to blur here. | Watch it inspect a bridge. A privacy pass runs first; nothing to blur here. |
| l3 | 0:55.4 | One survey look at the whole photo, at one eighth size. | One survey look at the whole photo, at one eighth size. |
| l4 | 0:58.9 | Entropy zero point six nine is above one half. Unsure, so it zooms in. | Entropy 0.69 is above 0.5. Unsure, so it zooms in. |
| l5 | 1:03.8 | Then it moves to the best region, and double-checks at full resolution. | Then it moves to the best region, and double-checks at full resolution. |
| l6 | 1:07.8 | At full resolution the masks disagree: unconfirmed. Exposed rebar always goes to a person. | At full resolution the masks disagree: unconfirmed. Exposed rebar always goes to a person. |
| l7 | 1:14.4 | Then a second spall. It stops when the next look costs more than it is worth, and so it misses a crack in a region it never visited. | Then a second spall. It stops when the next look costs more than it is worth, and so it misses a crack in a region it never visited. |
| l8 | 1:22.0 | The agent took under a second; the page replays it slowly. | The agent took under a second; the page replays it slowly. |
| l9 | 1:25.6 | Only a person can approve. A name, Approve, and the server signs the work order. | Only a person can approve. A name, Approve, and the server signs the work order. |
| l11 | 1:30.7 | Change one character of that signature, and the server refuses it. | Change one character of that signature, and the server refuses it. |
| b0 | 1:35.3 | Now, break it. | Now, break it. |
| b1 | 1:36.5 | Slow the segmenter down, and the governor switches to the fast model. | Slow the segmenter down, and the governor switches to the fast model. |
| b2 | 1:40.6 | The planner drill sends a shell command. The allow-list rejects it, and the rule planner decides. | The planner drill sends a shell command. The allow-list rejects it, and the rule planner decides. |
| b3 | 1:46.4 | KILL halts the job before its next look; Resume replays the moves so far and carries on. | KILL halts the job before its next look; Resume replays the moves so far and carries on. |
| b4 | 1:52.0 | No language model runs live. These drill planners are scripted to fail. | No language model runs live. These drill planners are scripted to fail. |
| m1 | 1:57.3 | Over MCP, Claude drove the same OpenCV tools on the same photo. | Over MCP, Claude drove the same OpenCV tools on the same photo. |
| m2 | 2:02.5 | At call six, the full-resolution mask disagreed, and exposed rebar is critical: Claude handed it to a human. | At call 6, the full-resolution mask disagreed, and exposed rebar is critical: Claude handed it to a human. |
| m3 | 2:09.8 | At call eighteen, the masks agreed: a spall, confirmed. At call twenty, the budget ended the search. | At call 18, the masks agreed: a spall, confirmed. At call 20, the budget ended the search. |
| m4 | 2:16.3 | It still missed the large spall. The rules are in the prompt; the values were measured. | It still missed the large spall. The rules are in the prompt; the values were measured. |
| a0 | 2:21.9 | Every step is perception, decision and action, with a human gate at the end. | Every step is perception, decision and action, with a human gate at the end. |
| a1 | 2:26.6 | CloudFront and a rate-limiting firewall front a signed Function URL and the arm Lambda. | CloudFront and a rate-limiting firewall front a signed Function URL and the arm Lambda. |
| a2 | 2:32.5 | DynamoDB holds the kill flags, Secrets Manager the signing key, and CloudWatch the alarms. | DynamoDB holds the kill flags, Secrets Manager the signing key, and CloudWatch the alarms. |
| a3 | 2:39.0 | OpenCV does the looking: a pyramid, saliency, warp perspective as a virtual pan-tilt-zoom camera, a blur gate, and face blurring. | OpenCV does the looking: a pyramid, saliency, warpPerspective as a virtual pan-tilt-zoom camera, a blur gate, and face blurring. |
| a4 | 2:47.9 | Live, it uses the classical segmenter; OpenCV five's new DNN engine is on the optional ONNX path. | Live, it uses the classical segmenter; OpenCV 5's new DNN engine is on the optional ONNX path. |
| a5 | 2:55.4 | On Lambda, a step took under ninety milliseconds at p ninety-five, over three smoke runs. COOL was not run. | On Lambda, a step took under 90 ms at p95, over 3 smoke runs. COOL was not run. |
| r1 | 3:03.2 | On two hundred synthetic photos, it finds eighty-seven percent of exhaustive tiling's recall, at twenty-six percent of the pixels: short of the ninety percent target, though not significantly. | On 200 synthetic photos, it finds 87 % of exhaustive tiling's recall, at 26 % of the pixels: short of the 90 % target, though not significantly. |
| r3 | 3:15.4 | Against one downscaled pass it scores zero point nine six times: slightly behind, within the noise. | Against one downscaled pass it scores 0.96×: slightly behind, within the noise. |
| r4 | 3:21.9 | On larger photos it beats one downscale, which reads a sixth as many pixels; quarter-size tiling does as well with fewer. | On larger photos it beats one downscale, which reads a sixth as many pixels; quarter-size tiling does as well with fewer. |
| r5 | 3:30.0 | Ablations: only dropping the survey look loses recall; the others lose none, and a random gaze order matches it on smaller photos. | Ablations: only dropping the survey look loses recall; the others lose none, and a random gaze order matches it on smaller photos. |
| r6 | 3:38.9 | On two hundred real CODEBRIM photos, every method finds about one defect in five; its boxes at random, about one in eight. | On 200 real CODEBRIM photos, every method finds about 1 defect in 5; its boxes at random, about 1 in 8. |
| r7 | 3:47.5 | On fifty-one synthetic photos with no damage, it still reports something two times in three. | On 51 synthetic photos with no damage, it still reports something 2 times in 3. |
| r8 | 3:53.4 | Results for a trained segmenter are pending. | Results for a trained segmenter are pending. |
| f1 | 3:56.5 | Failures, in the open: graffiti read as spalling, a formwork line as a crack, glare left for a re-shoot, and blur. | Failures, in the open: graffiti read as spalling, a formwork line as a crack, glare left for a re-shoot, and blur. |
| f2 | 4:04.0 | Two of these rules are not built yet. The straight-line test is built, but it let this line through at a coarse look. | 2 of these rules are not built yet. The straight-line test is built, but it let this line through at a coarse look. |
| f3 | 4:10.6 | On a real photo, exposed rebar was reported as corrosion. | On a real photo, exposed rebar was reported as corrosion. |
| x1 | 4:15.4 | Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve. | Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve. |
| y1 | 4:23.0 | Make test, make eval, and one script to deploy, test and destroy the AWS stack. | make test, make eval, and one script to deploy, test and destroy the AWS stack. |
| e1 | 4:29.4 | Drishyam. It looks where it matters, and asks a human first. | Drishyam. It looks where it matters, and asks a human first. |
<!-- generated:end -->
