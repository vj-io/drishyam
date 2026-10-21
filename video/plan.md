# Drishyam submission video: plan

The narrated video for the OpenCV AI Competition 2026, powered by AWS (Agentic Vision entry): at most 5 minutes (this
cut runs 4:00, the owner's 3.5 to 4 minutes), 1920x1080, offline AI voice (Kokoro `af_heart` through the brag-scaled `narrate` skill), captions burned in and
also shipped as WebVTT. The look is the site's own engineering-sheet blueprint (`public/styles.css` tokens and
fonts; the brag-styles `patent-office` set): one sheet per chapter, a sheet index on the left, the sheet number
stamping in at each chapter.

## Outputs

| File | What | In git |
| --- | --- | --- |
| `public/video/drishyam.mp4` | the video (H.264 + AAC, poster baked in as frame 0) | no: `public/video/*.mp4` is ignored; it ships to the site bucket at deploy time and as a release asset |
| `public/video/drishyam.vtt` | captions, the same cues as the burned-in ones (written by the same build as the MP4) | yes |
| `public/video/poster.jpg` | the agent mid-gaze in the live run (line `l5`, step 3's look at the best region), settled | yes |
| `brag-output/drishyam-short.mp4` | 15-20 s 9:16 cut: the cold open's run, the sign-off (the typing at real time, then a jump cut to Approve), ending on the settled APPROVED stamp and the URL; labelled as the classical engine's footage of 2026-10-18 (the site now runs v1), v1's held-out result in the lower third | no (brag-output/ is ignored) |
| `brag-output/standin/film/` | `build.py --standin`: the film with the stand-in real-photo runs, if `--make-standin` made them (nothing is pending any more): `drishyam-standin.mp4`, `drishyam-standin.vtt`, `poster-standin.jpg`, `plan-tables.md` | no |

The site's player is static markup in `public/index.html`, where the removed `initVideo()` (commit 939f973) put it:
a `<figure id="video">` at the top of the Docs and links sheet, with a caption saying what the film shows, and the
hero's "Watch the 4-minute film" link to it. `<video controls preload="none" poster="video/poster.jpg">` with
`<source src="video/drishyam.mp4">` and `<track kind="captions" src="video/drishyam.vtt" srclang="en">`, not `default`
(the captions are burned in, so a default track would show them twice):
nothing of the MP4 is fetched before Play, so a page without the MP4 (CI, a local checkout) logs no failed load.
The CSP needs no change: `default-src 'self'` covers same-origin media and text tracks. `tests/e2e/test_ui_mock.py`
(`test_video_player`, and the markup in `test_static_csp_hygiene`) and `tests/e2e/test_ui_live.py` check the
poster, every caption cue loaded and showing, no MP4 request before Play, and the player inside 375 px. Until
`public/video/drishyam.mp4` is uploaded with the site, Play on the live site finds no file: the deploy must ship it.

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
~/.claude/brag-scaled-venv/bin/python video/build.py --check-template   # every piece on, at full length: must fit 300 s; another machine's costs refused
~/.claude/brag-scaled-venv/bin/python video/build.py --make-standin     # stand-in real-photo traces -> brag-output/standin/ (from before the owner's runs)
~/.claude/brag-scaled-venv/bin/python video/build.py --standin          # the whole film with the stand-ins -> brag-output/standin/film/
```

`--standin` renders the film with the stand-in real-photo runs, into
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
| `real-photos/clean/<image>.jpg` (the team's wall photos, metadata stripped: IMG_2332, IMG_2330) | Real photos | no real-photo chapter (the film is about 16 s shorter) |

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

**Real-photo chapter.** One plate per photo, the camera gliding from the first to the second between the lines. A
photo appears when its run is recorded, `video/live/real.ndjson` and `real.json` (then each
`video/live/variants/<image>/real.*`; the NDJSON stream of one Drishyam job on the team's photo, embedded images
removed), and the photo is in the private folder. `real.json` names the image, the engine (`classical-v1` or
`onnx:<model>`, checked against the trace's cost table: a `-dacl10k` table means ONNX), the profile and, optionally,
`seen`: what the team sees in the photo by eye, not ground truth (`say`, the classes, and the cracks as traced, in
full-resolution pixels). Each plate is labelled "REAL PHOTO (ours) · <image>: painted plaster wall, no ground truth";
it shows the overview (the photo as its pyramid level, a scan line, the candidate regions), each look as the gaze
window, each finding's box and class, why it stopped, and the cards ENGINE, LOOKS (with the saliency-weighted coverage
when it looked), PIXELS READ, REPORTED and BY EYE. Each traced line draws on, thin over a light dimming so the crack
stays visible beside it, amber where a finding's box of a class the team sees touches it and red where none does; a
stamp lands: MISSED or FOUND for one line, "k OF n" for several. A finding away from every traced line would be "off
the team's lines", never a false alarm (there is no ground truth).

Recorded 2026-10-20 with v1, the live engine (`DRISHYAM_ONNX=models/drishyam_dacl10k.onnx
DRISHYAM_ONNX_FP16=models/drishyam_dacl10k.fp16.onnx`, `--micro-arm local-macos-x86-dacl10k-retime`, fixed costs, 30 %
pixel cap). The lines `q1`, `q2` are built from the traces in `video/narration.py` and asserted against them:

- IMG_2332 (`video/live/real.*`, job `j_2b765d`): the L3 overview (saliency, no segmenter) proposed no candidate
  region, so no look ran and the trained segmenter never ran; 1.6 % of the pixels read, nothing flagged, although a
  hairline crack is visible by eye. "Our own wall, no ground truth: no region was proposed, so the trained segmenter
  never ran, and missed the hairline crack."
- IMG_2330 (`video/live/variants/IMG_2330/real.*`, job `j_67658a`): one look at L1 and one crack, whose box touches
  the lower of the two cracks the team traced; the upper one is outside the look; coverage 21 %. "On a second wall,
  one close look flagged a crack on the lower of two we traced, and never looked closely at most of the wall."

A new run is recorded with `uv run drishyam run $DRISHYAM_PRIVATE_DIR/real-photos/clean/<image>.jpg --cost-model fixed
--out runs/real` (for the trained segmenter add `DRISHYAM_ONNX=<model>` and `--micro-arm <its -dacl10k table>`), its
`runs/real/<job>/trace.jsonl` copied with the `*_b64` fields removed (as `build.py --make-standin` does), and a
`real.json` written; a run whose shape no line form covers stops the build at its assert. The site replays later runs
on the same two photos, re-recorded on the current code (jobs `j_5b5e8f`, IMG_2332, and `j_d3fd41`, IMG_2330), embedded
images kept, from `public/samples/real_01.ndjson` and `real_02.ndjson`; the film keeps the two jobs above (its copies in
`video/live/`).

**The trained segmenter (lines `r6` and `r8`, the headline board FIG. 4.3, and `r7`'s clean-set half).** `public/results/codebrim_dacl10k/summary.json` is
quoted only like for like: its summary must say `onnx:`, governor off and not confounded, its 30 % cap point (one point
per cap, in `by_budget`'s order) must be the held-out table's row in `results/heldout_models.json`, on the same
CODEBRIM photos (dataset sha256s), and the classical engine it is compared with is that table's `classical-v1` row,
each engine charged its own cost table timed on one machine (`local-macos-x86`; the build refuses tables from two
machines). The voice says its recall (49 %) and the paired ratio to the classical engine (2.5×, against the classical
engine's 1 in 5) in `r6`, then its chance recall (its boxes at random places find 30 %) and its critical boxes per photo
against the classical engine's (5.5×) in `r8`; the board makes v1 the headline and adds the CIs, the critical boxes
per photo, the classical engine as its comparison row (every classical method about 1 in 5, its chance), the tiling
levels with v1 that find as much (L2 and L1) and the near-duplicate count (0; spoken if above zero). `r7` adds that it put no box on any of the 51 synthetic photos with no damage. `--check-template` builds
the line from the real summary, measures the whole film (both real photos) against 300 s, and checks that the same
summary charged another machine's cost table is refused.

The live site now runs v1: `video/live/aws-smoke.log` is a verbatim copy of the main checkout's `runs/aws-smoke.log`
(every `scripts/aws.sh smoke` run's `/api/health` and step p95); the build groups the runs by the segmenter their
health reports. Its last run (commit 38fcaee) reports `onnx:drishyam_dacl10k.onnx`. `l0` says so before the footage
plays: "Recorded 18 October on the classical engine; the site now runs the trained v1"; `a4` adds "on OpenCV 5's new
DNN engine", and `a5` a step p95 under 90 ms
then (3 classical runs, 67.7-88.0 ms) against under 1.1 s now (6 v1 runs, 559.5-1067.3 ms). Claims whose truth
depends on the data are asserted in `video/narration.py` and `video/build.py`, so a re-run that changes them stops the
build instead of shipping a sentence that is no longer true; words and figures typed into the scene text
(citations, the competition name, the limits, the team) are checked against their files and listed below.

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

## Length

Nothing is pending: the committed tables below are the film as rendered, 239.9 s. With the trained segmenter's line,
both real photos and the portraits on, the narration first measured 285 s; it came to 4:00 by cutting words and
holds, not chapters:

- what moved from the voice to the boards and rail cards: the 4608 px result (zooming beats one downscale there,
  and L2 tiling matches it with fewer pixels; in `docs/report.md` section 8, no longer on the film's board), the
  formwork plate's built-but-passed straight-line test, the CODEBRIM case in words (shown, not said), COOL not run (on
  the Lambda card), that the rules were in Claude's prompt (the MCP rail note), the agent's 0.71 s (the live rail
  note; line `l8` is gone), the signed Function URL and CloudWatch (on the AWS diagram the camera reads);
- shorter wording throughout ("each inspected at least every two years", "one downscaled pass does about as well",
  "chance, one in eight");
- gaps of 0.25 s in the live, architecture and results chapters, tails of 0.4 s, the cold open's title held 1.5 s and
  the end card 2.0 s. The gaps between lines are 0.3 s elsewhere (0.35 in the team chapter and the drills, where the
  footage warp needs it; with the portraits `t2` also waits 0.95 s for the second photo's hold; `l9` waits 0.5 s for
  the cut to the sign-off; `q2` waits 1.1 s for the glide to the second photo).

Kept, because a first-time judge needs them: the agent's loop (perception, decision, action, the human gate), the
live run with its miss, the drills and KILL, the MCP session, real numbers with their caveats (short of the target,
one downscale does about as well, every classical method about 1 in 5 on real photos, the trained segmenter's chance
recall and critical-box load), both of our own wall photos with their failures, the team's photo-to-portrait chapter,
and AWS and Arm (the arm Lambda, its latency then and now).

## How the footage is used, honestly

- Every frame of the demo chapters is a screencast of a real job on https://d1xg58guluup8c.cloudfront.net
  (recorded 2026-10-18, live jobs `j_fdcb12` guided, `j_946907` break it, `j_93a047` KILL, `j_d4f644` Resume).
  Each run's trace is saved, token fields and embedded images removed, in `video/live/*.ndjson`; every live-run number in the voice
  or on screen comes from those files. The site ran the classical engine then (`video/live/health.json`: `classical-v1`,
  commit ca69037); it now runs v1 (above). Before the footage plays, `l0` says so, with the camera on the recorded
  response's own `"segmenter":"classical-v1"` and a THEN / NOW pair of cards; `o1` names the classical engine, and the
  cold open's rail card, the strips, the live rail note and a badge in every footage window's bar
  (`REC 2026-10-18 · CLASSICAL-V1`) carry the date and the engine.
- The footage is retimed so each recorded event lands on the line that explains it. The window bar always shows
  the speed (`REAL TIME`, or `2.5x SPEED` and so on). Cuts (each a 0.36 s cross-dissolve): in the cold open,
  from step 4 to the first sign-off card; in the live chapter, the page's smooth scroll after "Watch it inspect a
  bridge" (it passes the hero's stat cards, which show the deployed build's numbers from before the final
  evaluation re-run) and the seconds between the end of the run and the first sign-off; in the drills, steps 1-4
  of the "Now break it" run (they repeat the guided run) and the seconds between the planner's rejection and the
  "Try KILL" click; in the 9:16 cut, from step 4 to the first reviewer field, and from the typed name to just before
  the Approve click (both pieces at real time; the settled stamp is then held, the bar reading `LAST FRAME`).
- The cursor is drawn from the recorder's log of real mouse moves and clicks; headless Chromium draws none.
- The reviewer name typed in the sign-off is "Demo reviewer", not a person.
- The tampered-token request was sent from the page by `video/record.py`; the inset shows its real status and
  body (`video/live/tamper.json`).
- No LLM runs in the live demo; the voice says so. The LLM chapter is the recorded Claude session in
  `docs/mcp-session/`, shown from its own files.
- CODEBRIM photos are never shown (licence); only its aggregate numbers and one text failure.
- The trained dacl10k segmenter's results are quoted from `results/heldout_models.json` and its published summary,
  held out, against the classical engine on the same photos (above).

## Not shown, and why

- The site's Benchmarks sheet: the recorded build (ca69037) showed the numbers from before the final re-run; the film
  draws its own chart from the final `public/results/*.json` instead.
- A CloudWatch dashboard and an owner-recorded team clip: no screenshot or footage exists. The team is shown from
  their photos and the OpenCV portraits drawn from them (private inputs, above), or as name cards without them.
- A test pass count: `make test` was not run for this video, so no count is claimed; the reproduce sheet shows
  the commands only.
- COOL timings: COOL was not run (README.md, "COOL on Graviton (not run)").

## Known on screen

- `docs/architecture.png` is shown as committed. Its segment box says "default: classical, no weights" (the code's
  default); the deployed site opts in to v1, as `a4` says over it. Its footnote cites `segment.py:239` for `cv2.dnn.ENGINE_NEW`;
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
| At most 5 minutes, 1920x1080 | open, end | 0:00.0 | 3:59.9 |
| Narrated (offline Kokoro voice af_heart) with burned-in captions; the same cues in public/video/drishyam.vtt | open, end | 0:00.0 | 3:59.9 |
| The team: Vaibhav Jangid, Bhavya Dhoot. Each real photo, then the window's looks turn it into OpenCV line art (Canny, findContours, approxPolyDP); name cards without the private inputs | team | 0:09.4 | 0:23.5 |
| Problem and user, with sources on screen | problem | 0:23.5 | 0:39.4 |
| Disclosed before the demo plays: the footage was recorded 2026-10-18 on the classical engine; the site now runs the trained v1 | l0 | 0:39.9 | 0:45.3 |
| /api/health on the live site: opencv 5.0.0, aarch64, aws-lambda | l1 | 0:45.6 | 0:49.4 |
| The application working: the guided run on bridge_03 ('Watch it inspect a bridge'), recorded live, with what it missed | l2, l3, l4, l5, l6, l7 | 0:49.7 | 1:16.9 |
| Plain-words trace (the NOTE and 'What it decided and why' panels) | l3, l4, l5, l6, l7 | 0:53.8 | 1:16.9 |
| Human sign-off: reviewer name, Approve, APPROVED stamp, work order signed | l9 | 1:17.6 | 1:21.1 |
| Human control: a tampered work-order token refused (403) | l11 | 1:21.4 | 1:24.9 |
| Failure handling: 'Now break it' (slowed segmenter -> governor; malformed planner reply -> allow-list) | b0, b1, b2 | 1:25.7 | 1:34.1 |
| Human control: 'Try KILL' (KILL halts with kill-switch, Resume replays) | b3 | 1:34.4 | 1:38.6 |
| Honesty: no LLM runs in the live demo; the drill planners are scripted | b4 | 1:39.0 | 1:42.9 |
| An LLM driving the OpenCV tools over MCP: IoU 0.00 at call 6 -> human; IoU 0.72 at call 18 -> confirmed; call 20 budget | mcp | 1:43.3 | 1:59.7 |
| Architecture: docs/agent-loop.svg, public/docs/aws-simple.svg, docs/architecture.png (AWS path and OpenCV layer) | arch | 1:59.7 | 2:32.4 |
| The live site now runs the trained segmenter v1 on OpenCV 5 DNN ENGINE_NEW, and its Lambda step p95 then and now (video/live/aws-smoke.log) | a4, a5 | 2:20.7 | 2:32.0 |
| Principal results: recall vs pixels, with one downscale, the tiling family and chance | r1, r3 | 2:32.9 | 2:45.1 |
| Ablations (public/results/ablations.json) | r5 | 2:45.3 | 2:52.4 |
| Real photos (CODEBRIM), the headline board: trained v1, held out (results/heldout_models.json), against the classical engine (about 1 in 5, the comparison row), its chance recall and critical boxes; no CODEBRIM image shown | r6, r8 | 2:52.7 | 3:08.3 |
| False alarms on synthetic photos with no damage (public/results/clean/summary.json), the classical engine and v1 | r7 | 3:08.6 | 3:14.8 |
| Two real photos of our own (painted walls, no ground truth): the looks, the findings, what it missed (video/live/real.*, video/live/variants/IMG_2330/real.*) | real | 3:15.2 | 3:31.4 |
| Failure cases: failures.json plates and their rules; one CODEBRIM case in words only | failures | 3:31.4 | 3:42.3 |
| Limits and responsible use | limits | 3:42.3 | 3:49.2 |
| Reproduce: make test, make eval, scripts/aws.sh | repro | 3:49.2 | 3:53.8 |
| End card: live URL, repository, team | end | 3:53.8 | 3:59.9 |

### Chapters (from this build)

| Chapter | Starts | Ends | Lines |
| --- | --- | --- | --- |
| Cold open | 0:00.0 | 0:09.4 | o1, o2 |
| Team | 0:09.4 | 0:23.5 | t1, t2 |
| Problem and user | 0:23.5 | 0:39.4 | p1, p2, p3 |
| Live demo | 0:39.4 | 1:25.3 | l0, l1, l2, l3, l4, l5, l6, l7, l9, l11 |
| Failure drills | 1:25.3 | 1:43.3 | b0, b1, b2, b3, b4 |
| An LLM over MCP | 1:43.3 | 1:59.7 | m1, m2, m3, m4 |
| Architecture | 1:59.7 | 2:32.4 | a0, a1, a3, a4, a5 |
| Results | 2:32.4 | 3:15.2 | r1, r3, r5, r6, r8, r7 |
| Real photos (ours) | 3:15.2 | 3:31.4 | q1, q2 |
| Failure cases | 3:31.4 | 3:42.3 | f1, f2 |
| Limits | 3:42.3 | 3:49.2 | x1 |
| Reproduce | 3:49.2 | 3:53.8 | y1 |
| End | 3:53.8 | 3:59.9 | e1 |

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
| failure cases caught (of the four shown) | ["Glare caught by the quality gate"] | `public/results/failures.json` | category containing 'caught' |
| failure cases whose fix is not built (of the other three) | 3 | `public/results/failures.json` | rule 'Not implemented', or 'Implemented (...)' with a 'Not implemented:' part |
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
| Lambda step p95 (ms), every smoke run with classical-v1 (min, max) | [67.7, 88.0] | `video/live/aws-smoke.log` | 'INFO latency' lines of the runs whose health says classical-v1: 2026-10-17T23:12:36Z, 2026-10-18T07:09:20Z, 2026-10-18T07:12:25Z |
| smoke runs with classical-v1 | 3 | `video/live/aws-smoke.log` | count of those runs |
| steps per smoke run with classical-v1 | [6] | `video/live/aws-smoke.log` | 'over N steps' |
| Lambda step p95 (ms), every smoke run with onnx:drishyam_dacl10k.onnx (min, max) | [524.7, 1138.9] | `video/live/aws-smoke.log` | 'INFO latency' lines of the runs whose health says onnx:drishyam_dacl10k.onnx: 2026-10-20T19:32:23Z, 2026-10-20T19:35:07Z, 2026-10-20T19:49:31Z, 2026-10-20T19:50:52Z, 2026-10-20T19:53:48Z, 2026-10-20T19:54:14Z, 2026-10-20T21:12:31Z, 2026-10-21T04:21:29Z, 2026-10-21T10:51:27Z, 2026-10-21T11:15:48Z, 2026-10-21T11:38:30Z |
| smoke runs with onnx:drishyam_dacl10k.onnx | 11 | `video/live/aws-smoke.log` | count of those runs |
| steps per smoke run with onnx:drishyam_dacl10k.onnx | [4, 5, 6] | `video/live/aws-smoke.log` | 'over N steps' |
| the live site now: segmenter, commit (the last smoke run's /api/health) | {"segmenter": "onnx:drishyam_dacl10k.onnx", "commit": "f12f50f8cebf758daa047ceefa9317c3557f56bb"} | `video/live/aws-smoke.log` | PASS health of the run 2026-10-21T11:38:30Z |
| work-order signature | HMAC-SHA256 | `docs/security.md` | 'Tampered or replayed work order' row |
| stock aarch64 wheel custom HAL (GitHub arm64 runner, not Lambda) | YES (carotene (ver 0.0.1) KleidiCV (ver 26.03)) | `results/linux-aarch64/micro.json` | provenance.cv2_custom_hal |
| US bridges (FHWA NBI 2026) | 624,939 | `docs/impact-sources.md` | section 1 Figure |
| in poor condition | 41,148 | `docs/impact-sources.md` | section 1 Figure |
| routine inspection interval, months (23 CFR 650.311) | 24 | `docs/impact-sources.md` | section 2 Quote |
| photos from the largest drone inspection in the study (ODOT SPR 787) | 2,536 | `docs/impact-sources.md` | section 5 Figure |
| photos from the other bridges in that table (min, max) | [197, 401] | `docs/impact-sources.md` | section 5 Figure |
| bridges flown in the study | 6 | `docs/impact-sources.md` | section 5 Figure |
| trained segmenter: CODEBRIM photos | 200 | `public/results/codebrim_dacl10k/summary.json` | n_images |
| trained segmenter: CODEBRIM recall (30 % cap) | 0.4926 | `public/results/codebrim_dacl10k/summary.json` | policies.drishyam[cap 0.3].recall |
| its share of the pixels | 0.1422 | `public/results/codebrim_dacl10k/summary.json` | policies.drishyam[cap 0.3].pixels_frac |
| its 95 % CI | [0.4427, 0.54] | `public/results/codebrim_dacl10k/summary.json` | policies.drishyam[cap 0.3].recall_ci |
| its boxes at random | 0.2991 | `public/results/codebrim_dacl10k/summary.json` | policies.drishyam[cap 0.3].chance_recall |
| classical engine on the same photos (held out) | 0.1941 | `results/heldout_models.json` | table['codebrim@0.3']['classical-v1'].recall |
| trained / classical recall, paired 95 % CI | [2.5382, [2.1562, 3.014]] | `results/heldout_models.json` | pairs['codebrim@0.3: v1 / classical-v1'].recall |
| critical boxes per photo: trained, classical | [9.6, 1.73] | `results/heldout_models.json` | table['codebrim@0.3'][*].critical_per_image |
| trained / classical critical boxes per photo, paired 95 % CI | [5.5491, [3.5656, 8.8241]] | `results/heldout_models.json` | pairs['codebrim@0.3: v1 / classical-v1'].critical |
| synthetic photos with no damage given any box: trained, classical | [0.0, 0.6667] | `results/heldout_models.json` | table['clean@0.3'][*].images_with_any |
| tiling with the trained segmenter, L2/L1/L0: pixels, recall | [[0.1047, 0.5027], [0.4191, 0.5139], [1.6229, 0.3555]] | `public/results/codebrim_dacl10k/summary.json` | policies.tiling |
| tiling levels that find as much as the trained segmenter's agent, or more | ["L2", "L1"] | `public/results/codebrim_dacl10k/summary.json` | policies.tiling[].recall >= policies.drishyam[cap 0.3].recall |
| CODEBRIM photos with a near-duplicate in dacl10k | 0 | `public/results/codebrim_dacl10k/summary.json` | leakage.near_duplicates |
| team portraits: the OpenCV calls in the header | ["cv2.Canny", "cv2.findContours", "cv2.approxPolyDP"] | `$DRISHYAM_PRIVATE_DIR/team/out/portraits.py` | cv2 calls in the script (OpenCV 5); private, git-ignored |
| team photos: photo-1, photo-2 | ["Vaibhav Jangid", "Bhavya Dhoot"] | `$DRISHYAM_PRIVATE_DIR/team/team.json` | photo-1.name, photo-2.name (both agreed to appear) |
| real photo IMG_2332: engine | onnx:drishyam_dacl10k.onnx | `video/live/real.json` | engine |
| real photo IMG_2332: profile | fixed costs (local-macos-x86-dacl10k-retime), 30 % pixel cap | `real.json` | profile |
| real photo IMG_2332: candidate regions in the overview | 0 | `video/live/real.ndjson` | overview.candidates |
| real photo IMG_2332: looks | 0 | `video/live/real.ndjson` | done.summary.steps |
| real photo IMG_2332: share of pixels read | 0.0157 | `video/live/real.ndjson` | done.summary.pixels_frac |
| real photo IMG_2332: findings | 0 | `video/live/real.ndjson` | done.summary.findings (last state of each finding event) |
| real photo IMG_2332: why it stopped | exhausted: every candidate has been looked at | `video/live/real.ndjson` | step[action=halt].cause, rationale |
| real photo IMG_2332: coverage (share of the overview's saliency that a look saw) | 0.0 | `video/live/real.ndjson` | done.coverage |
| real photo IMG_2332: what the team sees, by eye (not ground truth) | the hairline crack (crack) | `real.json` | seen |
| real photo IMG_2332: findings on it / not on it (by eye) | [[], []] | `video/live/real.ndjson` | finding boxes touching seen.path |
| real photo IMG_2332: each traced line touched by a finding (by eye) | [false] | `video/live/real.ndjson` | seen.paths[i] inside a finding box of a seen class |
| real photo IMG_2330: engine | onnx:drishyam_dacl10k.onnx | `video/live/variants/IMG_2330/real.json` | engine |
| real photo IMG_2330: profile | fixed costs (local-macos-x86-dacl10k-retime), 30 % pixel cap | `real.json` | profile |
| real photo IMG_2330: candidate regions in the overview | 1 | `video/live/variants/IMG_2330/real.ndjson` | overview.candidates |
| real photo IMG_2330: looks | 1 | `video/live/variants/IMG_2330/real.ndjson` | done.summary.steps |
| real photo IMG_2330: share of pixels read | 0.0545 | `video/live/variants/IMG_2330/real.ndjson` | done.summary.pixels_frac |
| real photo IMG_2330: findings | 1 | `video/live/variants/IMG_2330/real.ndjson` | done.summary.findings (last state of each finding event) |
| real photo IMG_2330: why it stopped | exhausted: every candidate has been looked at | `video/live/variants/IMG_2330/real.ndjson` | step[action=halt].cause, rationale |
| real photo IMG_2330: coverage (share of the overview's saliency that a look saw) | 0.2065 | `video/live/variants/IMG_2330/real.ndjson` | done.coverage |
| real photo IMG_2330: what the team sees, by eye (not ground truth) | the hairline cracks (crack) | `real.json` | seen |
| real photo IMG_2330: findings on it / not on it (by eye) | [["f1"], []] | `video/live/variants/IMG_2330/real.ndjson` | finding boxes touching seen.path |
| real photo IMG_2330: each traced line touched by a finding (by eye) | [true, false] | `video/live/variants/IMG_2330/real.ndjson` | seen.paths[i] inside a finding box of a seen class |
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
| o1 | 0:01.5 | Its classical engine read a quarter of the pixels and flagged three critical findings. | Its classical engine read a quarter of the pixels and flagged 3 critical findings. |
| o2 | 0:06.7 | It will not sign them off. A person must. | It will not sign them off. A person must. |
| t1 | 0:09.9 | Drishyam, an Agentic Vision entry to the OpenCV AI Competition, powered by AWS. | Drishyam, an Agentic Vision entry to the OpenCV AI Competition, powered by AWS. |
| t2 | 0:17.8 | Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation. | Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation. |
| p1 | 0:24.0 | America has about six hundred twenty-five thousand bridges, inspected at least every two years. | America has about 625,000 bridges, inspected at least every two years. |
| p2 | 0:30.0 | One drone inspection took more than twenty-five hundred photos of one bridge. | One drone inspection took more than 2,500 photos of one bridge. |
| p3 | 0:34.7 | Drishyam is for the engineer reviewing them: it zooms in only where needed. | Drishyam is for the engineer reviewing them: it zooms in only where needed. |
| l0 | 0:39.9 | Recorded October eighteenth on the classical engine; the site now runs the trained v1. | Recorded 18 October on the classical engine; the site now runs the trained v1. |
| l1 | 0:45.6 | Live on AWS: OpenCV five on an arm Lambda. | Live on AWS: OpenCV 5 on an arm Lambda. |
| l2 | 0:49.7 | Watch it inspect a bridge; a privacy pass finds nothing to blur. | Watch it inspect a bridge; a privacy pass finds nothing to blur. |
| l3 | 0:53.8 | One survey look at the whole photo, at one eighth size. | One survey look at the whole photo, at one eighth size. |
| l4 | 0:57.2 | Entropy zero point six nine, above one half: unsure, so it zooms in. | Entropy 0.69, above 0.5: unsure, so it zooms in. |
| l5 | 1:02.0 | It moves to the best region and double-checks one level finer. | It moves to the best region and double-checks one level finer. |
| l6 | 1:05.7 | The masks disagree: unconfirmed. Exposed rebar always goes to a person. | The masks disagree: unconfirmed. Exposed rebar always goes to a person. |
| l7 | 1:11.1 | A second spall; then it stops, the next look not worth its cost, and misses a crack it never visited. | A second spall; then it stops, the next look not worth its cost, and misses a crack it never visited. |
| l9 | 1:17.6 | Only a person can approve; then the server signs the work order. | Only a person can approve; then the server signs the work order. |
| l11 | 1:21.4 | Change one character of the signature: the server refuses it. | Change one character of the signature: the server refuses it. |
| b0 | 1:25.7 | Now, break it. | Now, break it. |
| b1 | 1:27.0 | Slow the segmenter, and the governor switches to the fast model. | Slow the segmenter, and the governor switches to the fast model. |
| b2 | 1:30.7 | The planner drill sends a shell command; the allow-list rejects it. | The planner drill sends a shell command; the allow-list rejects it. |
| b3 | 1:34.4 | KILL halts it before the next look; Resume replays the moves and carries on. | KILL halts it before the next look; Resume replays the moves and carries on. |
| b4 | 1:39.0 | No language model runs live; these planners are scripted to fail. | No language model runs live; these planners are scripted to fail. |
| m1 | 1:43.8 | Over MCP, Claude drove the same tools on the same photo. | Over MCP, Claude drove the same tools on the same photo. |
| m2 | 1:48.0 | At call six the rebar masks disagreed, so Claude handed it to a human. | At call 6 the rebar masks disagreed, so Claude handed it to a human. |
| m3 | 1:52.5 | At call eighteen they agreed on a spall; at call twenty, the budget ended the search. | At call 18 they agreed on a spall; at call 20, the budget ended the search. |
| m4 | 1:57.7 | It still missed the large spall. | It still missed the large spall. |
| a0 | 2:00.2 | Each step is perception, decision and action, with a human gate at the end. | Each step is perception, decision and action, with a human gate at the end. |
| a1 | 2:04.8 | CloudFront and a rate-limiting firewall front the arm Lambda; DynamoDB keeps kill flags, Secrets Manager the key. | CloudFront and a rate-limiting firewall front the arm Lambda; DynamoDB keeps kill flags, Secrets Manager the key. |
| a3 | 2:12.5 | OpenCV does the looking: a pyramid, saliency, warp perspective as a pan-tilt-zoom camera, a blur gate, face blurring. | OpenCV does the looking: a pyramid, saliency, warpPerspective as a pan-tilt-zoom camera, a blur gate, face blurring. |
| a4 | 2:20.7 | The site now runs v1 on OpenCV five's new DNN engine. | The site now runs v1 on OpenCV 5's new DNN engine. |
| a5 | 2:25.8 | On Lambda a step took under ninety milliseconds at p ninety-five then, and under one point two seconds now. | On Lambda a step took under 90 ms at p95 then, and under 1.2 s now. |
| r1 | 2:32.9 | On two hundred synthetic photos it finds eighty-seven percent of full tiling's recall at twenty-six percent of the pixels, short of its ninety percent target. | On 200 synthetic photos it finds 87 % of full tiling's recall at 26 % of the pixels, short of its 90 % target. |
| r3 | 2:42.9 | One downscaled pass does about as well. | One downscaled pass does about as well. |
| r5 | 2:45.3 | Ablations: only dropping the survey look loses recall, and a random gaze order matches it on smaller photos. | Ablations: only dropping the survey look loses recall, and a random gaze order matches it on smaller photos. |
| r6 | 2:52.7 | On two hundred real CODEBRIM photos, held out, the trained v1 finds forty-nine percent: two and a half times the classical engine's one in five. | On 200 real CODEBRIM photos, held out, the trained v1 finds 49 %: two and a half times the classical engine's 1 in 5. |
| r8 | 3:02.2 | But its boxes find thirty percent at random, and a person gets five and a half times the critical boxes. | But its boxes find 30 % at random, and a person gets five and a half times the critical boxes. |
| r7 | 3:08.6 | On fifty-one undamaged synthetic photos, the classical engine flags two in three; v1, none. | On 51 undamaged synthetic photos, the classical engine flags 2 in 3; v1, none. |
| q1 | 3:15.7 | Our own wall, no ground truth: no region was proposed, so the trained segmenter never ran, and missed the hairline crack. | Our own wall, no ground truth: no region was proposed, so the trained segmenter never ran, and missed the hairline crack. |
| q2 | 3:24.0 | On a second wall, one close look flagged a crack on the lower of two we traced, and never looked closely at most of the wall. | On a second wall, one close look flagged a crack on the lower of two we traced, and never looked closely at most of the wall. |
| f1 | 3:31.9 | Failures in the open: graffiti read as spalling, a formwork line as a crack, glare, and blur. | Failures in the open: graffiti read as spalling, a formwork line as a crack, glare, and blur. |
| f2 | 3:38.1 | Only the glare gate caught its case; three fixes are not built yet. | Only the glare gate caught its case; three fixes are not built yet. |
| x1 | 3:42.8 | Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve. | Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve. |
| y1 | 3:49.7 | Make test, make eval, and one script for the AWS stack. | make test, make eval, and one script for the AWS stack. |
| e1 | 3:54.2 | Drishyam. It looks where it matters, and asks a human first. | Drishyam. It looks where it matters, and asks a human first. |
<!-- generated:end -->
