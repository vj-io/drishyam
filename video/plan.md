# Drishyam submission video: plan

The narrated video for the OpenCV AI Competition 2026, powered by AWS (Agentic Vision entry): at most 5 minutes (this
cut runs 4:24, the owner's 3.5 to 4 minutes, plus the re-recorded live chapter), 1920x1080, offline AI voice (Kokoro `af_heart` through the brag-scaled `narrate` skill), captions burned in and
also shipped as WebVTT. The look is the site's own engineering-sheet blueprint (`public/styles.css` tokens and
fonts; the brag-styles `patent-office` set): one sheet per chapter, a sheet index on the left, the sheet number
stamping in at each chapter.

## Outputs

| File | What | In git |
| --- | --- | --- |
| `public/video/drishyam.mp4` | the video (H.264 + AAC, poster baked in as frame 0) | no: `public/video/*.mp4` is ignored; it ships to the site bucket at deploy time and as a release asset |
| `public/video/drishyam.vtt` | captions, the same cues as the burned-in ones (written by the same build as the MP4) | yes |
| `public/video/poster.jpg` | the agent's confirm look on the rebar in the live run (line `l5`, step 3: the camera panel with its crop and WHY THIS LOOK, "...the confirm look did not confirm it, so a person checks it by eye"), settled | yes |
| `brag-output/drishyam-short.mp4` | 15-20 s 9:16 cut: the cold open's run, the sign-off (the typing at real time, then a jump cut to Approve), ending on the settled APPROVED stamp of the spalling and the URL; labelled "recorded 2026-10-23 · live site · v1 · v1-policy2" and "the rebar went to a person to check by eye"; its three lines are o1, "Only a person can approve a critical finding." and "Try it live: the trained v1."; v1's held-out result in the lower third, with the critical findings a person signs per photo (2.17, and 1.15 with v1-policy2; recall 0.99×) | no (brag-output/ is ignored) |
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
photo appears when its run is recorded, `video/live/walls/<image>/real.ndjson` and `real.json` (IMG_2332, then
IMG_2330; the NDJSON stream of one Drishyam job on the team's photo, token fields and embedded images removed), and the
photo is in the private folder. `real.json` names the image, the engine (`classical-v1` or
`onnx:<model>`, checked against the trace's cost table: a `-dacl10k` table means ONNX), the profile and, optionally,
`seen`: what the team sees in the photo by eye, not ground truth (`say`, the classes, and the cracks as traced, in
full-resolution pixels). Each plate is labelled "REAL PHOTO (ours) · <image>: painted plaster wall, no ground truth";
it shows the overview (the photo as its pyramid level, a scan line, the candidate regions), each look as the gaze
window (clipped to the photo), each finding's box and class, why it stopped, and the cards ENGINE, LOOKS (with the
saliency-weighted coverage when it looked, or the fallback look), PIXELS READ, REPORTED (a look the quality gate refused:
its reason, its sharpness against `SHARPNESS_MIN` in `drishyam/imaging.py`, and the mask it dropped) and BY EYE. Each traced line draws on, thin over a light dimming so the crack
stays visible beside it, amber where a finding's box of a class the team sees touches it and red where none does; a
stamp lands: MISSED or FOUND for one line, "k OF n" for several. A finding away from every traced line would be "off
the team's lines", never a false alarm (there is no ground truth).

Recorded 2026-10-23 on the live site (commit a2b8688, v1 with v1-policy2, measured costs, 30 % pixel cap): each photo
is the bundled sample (`public/samples/real_01.jpg` is IMG_2332, `real_02.jpg` is IMG_2330, byte for byte), run with
**Look**, not **Watch it on a real wall** (which replays an older recording), by `video/record.py --walls` (2 job
starts). The lines `q1`, `q2` are built from the traces in `video/narration.py` and asserted against them:

- IMG_2332 (`video/live/walls/IMG_2332/`, job `j_da1449`): the L3 overview proposed no candidate region, so the
  zero-candidate fallback took one look at L2 whose window spans the whole photo (cause `fallback-l2`). v1's mask there
  holds 9,136 crack px, but the quality gate refused the view as blur (Laplacian variance 5.1, under `SHARPNESS_MIN`
  12: the smooth paint gives it little texture), so its findings were dropped: nothing flagged, 9.9 % of the pixels read,
  and the work order lists the photo as not assessed twice (`check-by-eye`, and `re-shoot`, an instruction that is
  wrong for this sharp photo; README, report section 9). "Our own wall, no ground truth: no region was proposed, so it
  looked at the whole photo. The segmenter saw crack pixels, but the blur gate refused this sharp view, and nothing was
  flagged." The both-gates blur gate that would pass this view was measured on dev and not adopted
  (`results/dev/blur_gate.json`); the film does not mention it.
- IMG_2330 (`video/live/walls/IMG_2330/`, job `j_d2b9b0`): one look at L1 and one crack, whose box touches the lower
  of the two cracks the team traced; the upper one is outside the look; coverage 21 %. "On a second wall, one close
  look flagged a crack on the lower of two we traced, and never looked closely at most of the wall."

Each `real.json` is the older run's (what the team sees by eye) with this run's engine and profile. The older local
runs (2026-10-20, fixed costs `local-macos-x86-dacl10k-retime`: `video/live/real.*`, job `j_2b765d`, which predates
the fallback and took no look; `video/live/variants/IMG_2330/real.*`, job `j_67658a`) stay where they are: docs/report.md
cites them beside the site's replays (`public/samples/real_01.ndjson`, `real_02.ndjson`, jobs `j_5b5e8f` and
`j_d3fd41`). A run whose shape no line form covers stops the build at its assert.

**The trained segmenter (lines `r6` and `r8`, the headline board FIG. 4.3, and `r7`'s clean-set half).** `public/results/codebrim_dacl10k/summary.json` is
quoted only like for like: its summary must say `onnx:`, governor off and not confounded, its 30 % cap point (one point
per cap, in `by_budget`'s order) must be the held-out table's row in `results/heldout_models.json`, on the same
CODEBRIM photos (dataset sha256s), and the classical engine it is compared with is that table's `classical-v1` row,
each engine charged its own cost table timed on one machine (`local-macos-x86`; the build refuses tables from two
machines). The voice says its recall (49 %) and the paired ratio to the classical engine (2.5×, against the classical
engine's 1 in 5) in `r6`, then its chance recall (its boxes at random places find 30 %) and its critical boxes per photo
against the classical engine's (5.5×) in `r8`, which then says what a person signs, critical findings, not boxes
(README's glance line): "A person signs about two critical findings a photo, about one with the live policy, and recall
is unchanged" (v1-policy2's one held-out read beside v1's, `results/heldout_models.json` `"v1-policy2"`
`.read.codebrim.matrix`: `signoffs_per_image` 2.175 for v1's first policy and 1.145 for v1-policy2, read from the file;
its pairs with v1: recall 0.99×, CI 0.97–1.01; the critical boxes 8.75 per photo instead of 9.60, 0.91×, CI 0.84–0.98,
are on the board, labelled as boxes; the build asserts the read used v1's baselines, passed its dev rule's clauses,
compares with the board's v1 figure, and that its confidence floor is the one in the footage). The board makes v1 the
headline and adds the CIs, the critical boxes per photo (labelled as boxes, parts of findings), the critical findings a
person signs per photo, the live policy's line under them, the classical engine as its comparison row (every classical method about 1 in 5, its chance), the tiling
levels with v1 that find as much (L2 and L1) and the near-duplicate count (0; spoken if above zero). `r7` adds that it put no box on any of the 51 synthetic photos with no damage. `--check-template` builds
the line from the real summary, measures the whole film (both real photos) against 300 s, and checks that the same
summary charged another machine's cost table is refused.

The live site runs v1 with the v1-policy2 look policy (since 2026-10-22), and the footage is of that build (the
`/api/health` recorded with it, `video/live/health.json`: version 1.1.1, commit a2b8688). `video/live/aws-smoke.log` is a copy of
the main checkout's `runs/aws-smoke.log` (every `scripts/aws.sh smoke` run's `/api/health` and step p95) up to
2026-10-21T11:38:30Z: `docs/aws.md` and `tests/test_live_state_doc.py` quote its eleven v1 runs, so the two smoke runs
of 2026-10-22 (adf9474, c1830c4) are not in the copy yet. The build groups the runs by the segmenter their health
reports, and asserts that the footage's segmenter is the one the log's last run reports. `l0` says when and on what
before the footage plays: "Recorded 23 October on the live site: the trained v1 and its new look policy", over two
cards, RECORDED (the segmenter and commit) and LOOK POLICY (v1-policy2, its confidence floor read from the trace);
`a4` adds "on OpenCV 5's new DNN engine", and `a5` the step p95 of the classical runs (before v1) against the v1
runs, both rounded up from the log (the run counts and ranges are in the generated table below). Claims whose truth
depends on the data are asserted in `video/narration.py` and `video/build.py`, so a re-run that changes them stops the
build instead of shipping a sentence that is no longer true; words and figures typed into the scene text
(citations, the competition name, the limits, the team) are checked against their files and listed below.

The captions are the spoken words. `video/narration.py` builds each line from parts that are either plain text
or a (spoken, printed) pair, so the caption can differ from the voice only in how a number or code name is
printed ("zero point nine four" / "0.94", "warp perspective" / "warpPerspective"). Exact figures, sources and
the extra detail (the HTTP 403 body, `rm -rf /`, the CI bounds) are on the boards and rail cards, not in the
captions.

The live footage is not re-recorded by the build (the WAF allows 10 job starts per IP per 120 s). To record it
again (4 job starts, about 3 minutes; never during a `scripts/aws.sh smoke` WAF loop, which blocks the IP for minutes).
It writes the traces, `video/live/tamper.json` and `order_status.json` (the work order after the sign-off and the check
by eye: status, approvals and resolutions); `--health` writes `health.json` and the boxes of its key fields, the
segmenter's and the whole response's included; `--walls` runs the two wall photos live (2 more job starts, a couple of
minutes later):

```bash
~/.claude/brag-scaled-venv/bin/python video/record.py            # screencast + event log -> brag-output/work/rec/, traces -> video/live/
~/.claude/brag-scaled-venv/bin/python video/record.py --health   # /api/health response -> video/live/health.json
~/.claude/brag-scaled-venv/bin/python video/record.py --walls    # Look on real_01 and real_02 -> video/live/walls/<image>/
~/.claude/brag-scaled-venv/bin/python ~/.claude/plugins/cache/brag-scaled/brag-scaled/0.8.0-scaled/skills/motion-kit/scripts/walk.py record brag-output/work/walk-plan.json --out brag-output/work/walkrec
```

## Length

Nothing is pending: the committed tables below are the film as rendered, 263.7 s (247.9 s before the fix pass of
2026-10-23, 239.9 s before the live chapter was first re-recorded on 2026-10-22). The fix pass added about 16 s by
replacing stale lines, not by adding chapters: `p3` in the site hero's words (+7 s), the rebar checked by eye (`l9`
and a new `l10`, +2 s), `r8` in the unit a person signs (+3 s) and `q1`'s fallback look (+4 s). With the trained segmenter's line,
both real photos and the portraits on, the narration first measured 285 s; it came to 4:00 by cutting words and
holds, not chapters:

- what moved from the voice to the boards and rail cards: the 4608 px result (zooming beats one downscale there,
  and L2 tiling matches it with fewer pixels; in `docs/report.md` section 8, no longer on the film's board), the
  formwork plate's built-but-passed straight-line test, the CODEBRIM case in words (shown, not said), COOL not run (on
  the Lambda card), that the rules were in Claude's prompt (the MCP rail note), the agent's seconds (the live rail
  note; line `l8` is gone), the signed Function URL and CloudWatch (on the AWS diagram the camera reads);
- shorter wording throughout ("each inspected at least every two years", "one downscaled pass does about as well",
  "chance, one in eight");
- gaps of 0.25 s in the live, architecture and results chapters, tails of 0.4 s, the cold open's title held 1.5 s and
  the end card 2.0 s. The gaps between lines are 0.3 s elsewhere (0.35 in the team chapter and the drills, where the
  footage warp needs it; with the portraits `t2` also waits 0.95 s for the second photo's hold; `l9` waits 0.5 s for
  the cut to the sign-off; `q2` waits 1.1 s for the glide to the second photo).

Kept, because a first-time judge needs them: the agent's loop (perception, decision, action, the human gate), the
live run with what it left to a person (the rebar, then checked by eye on the page), the drills and KILL, the MCP session, real numbers with their caveats (short of the target,
one downscale does about as well, every classical method about 1 in 5 on real photos, the trained segmenter's chance
recall, its critical boxes and the critical findings a person signs), both of our own wall photos with their failures, the team's photo-to-portrait chapter,
and AWS and Arm (the arm Lambda, its latency then and now).

## How the footage is used, honestly

- Every frame of the demo chapters is a screencast of a real job on https://d1xg58guluup8c.cloudfront.net
  (recorded 2026-10-23 at about 03:13Z, live jobs `j_5a68d1` guided, `j_9f3611` break it, `j_a2aad4` KILL, `j_9ce0ae`
  Resume), on the build the site runs now: v1 with the v1-policy2 look policy (`video/live/health.json`: version
  1.1.1, `onnx:drishyam_dacl10k.onnx`, commit a2b8688, the deploy with the Checked by eye control). It replaced the
  footage of 2026-10-22 (ed7a080, before a person could record the check by eye; its frames are kept, git-ignored, in
  `brag-output/work/rec-2026-10-22/`), which had replaced the footage of 2026-10-18, recorded on the classical engine. Each run's trace is saved,
  token fields and embedded images removed, in `video/live/*.ndjson`; every live-run number in the voice or on screen
  comes from those files and `video/live/order_status.json`. Before the footage plays, `l0` says when and on what, with
  the camera on the recorded response's own `"segmenter"` field; `o1` says "its trained engine", and the cold open's
  rail card, the strips, the live rail note and a badge in every footage window's bar (`REC 2026-10-23 ·
  ONNX:DRISHYAM_DACL10K.ONNX`) carry the date and the engine.
- What the guided run shows, and its cost, said plainly (`l3`-`l10`, asserted in `video/build.py` and
  `video/narration.py`): the survey look finds spalling and corrosion; one level finer the spalling is critical by area,
  but the exposed rebar scores 0.58, under v1-policy2's confidence floor of 0.6, so it is held for one confirm look; that
  look, at L1 and centred on it, does not confirm it, so the rebar goes to a person to check by eye (the work order's
  `needs-a-look` entry), not as a critical finding. The rebar is in this synthetic photo's ground truth; every other
  ground-truth box is touched by a finding of its class. Three more looks find the crack, and it stops with every
  candidate looked at. A person approves the spalling, and the order then reads "needs a look" (`incomplete`). The same
  person then checks the rebar's entry by eye on the page (Checked by eye: the reviewer name, the note "exposed rebar in
  the spall", **Defect found**, because the photo shows three bars there and its ground truth has the rebar), the server
  re-signs the order, and it reads `ready` (`order_status.json`: status, the two approvals, the resolution). The build
  asserts that the one resolution is the rebar's, Defect found, and that the ground truth has the rebar there;
  `record.py` refuses to record a decision for any other entry.
- The footage is retimed so each recorded event lands on the line that explains it. The window bar always shows
  the speed (`REAL TIME`, or `2.5x SPEED` and so on). Cuts (each a 0.36 s cross-dissolve): in the cold open,
  from step 4 to the first sign-off card; in the live chapter, the page's smooth scroll after "Watch it inspect a
  bridge" (it passes the hero's stat cards), the seconds between the end of the run and the first sign-off, and the
  page's wait and scroll between the approval and the rebar's Checked by eye entry; in the drills, steps 1-4
  of the "Now break it" run (they repeat the guided run) and the seconds between the planner's rejection and the
  "Try KILL" click. Under `l5` and `l6` the footage of step 3 (the confirm look, and its plain-words reason) plays slowed,
  about 0.45x, so it can be read while it is said: the page is still between steps, and the bar says the speed; in the 9:16 cut, from step 4 to the first reviewer field, and from the typed name to just before
  the Approve click (both pieces at real time; the settled stamp is then held, the bar reading `LAST FRAME`).
- The cursor is drawn from the recorder's log of real mouse moves and clicks; headless Chromium draws none.
- The reviewer name typed in the sign-off and in the check by eye is "Demo reviewer", not a person.
- The tampered-token request was sent from the page by `video/record.py`; the inset shows its real status and
  body (`video/live/tamper.json`).
- No LLM runs in the live demo; the voice says so. The LLM chapter is the recorded Claude session in
  `docs/mcp-session/`, shown from its own files.
- CODEBRIM photos are never shown (licence); only its aggregate numbers and one text failure.
- The trained dacl10k segmenter's results are quoted from `results/heldout_models.json` and its published summary,
  held out, against the classical engine on the same photos (above).

## Not shown, and why

- The site's Benchmarks sheet: the recording does not visit it; the film draws its own chart from the final
  `public/results/*.json` instead.
- A CloudWatch dashboard and an owner-recorded team clip: no screenshot or footage exists. The team is shown from
  their photos and the OpenCV portraits drawn from them (private inputs, above), or as name cards without them.
- A test pass count: `make test` was not run for this video, so no count is claimed; the reproduce sheet shows
  the commands only.
- COOL timings: COOL was not run (README.md, "COOL on Graviton (not run)").

## Known on screen

- `docs/architecture.png` is shown as committed. Its segment box says "default: classical, no weights" (the code's
  default); the deployed site opts in to v1, as `a4` says over it. Its footnote cites `segment.py:239` for `cv2.dnn.ENGINE_NEW`;
  the call is now at `drishyam/segment.py:298`. The claim itself (the one OpenCV-5-only call) still holds.
- The site's step panel (FIG. 2.3b) has a fixed height, so at step 3 the last line of its WHAT IT SAW box is clipped
  by the box. The camera's view of that panel ends above WHAT IT SAW (`film.js`, the step-3 key), so the clipped line
  is not in the film; the fix belongs in the site.
- Clicking "Try KILL" clears the last job, so the page re-lays out and scrolls back to the viewer; at about 7x speed
  the "How it works" sheet passes through the window for a fraction of a second, and after KILL the page scrolls up to
  the Demo sheet's head, where Resume is. It is the real page doing it.

<!-- generated:start -->
### Requirement checklist (timestamps from this build)

| Requirement | Covered by | From | To |
| --- | --- | --- | --- |
| At most 5 minutes, 1920x1080 | open, end | 0:00.0 | 4:23.7 |
| Narrated (offline Kokoro voice af_heart) with burned-in captions; the same cues in public/video/drishyam.vtt | open, end | 0:00.0 | 4:23.7 |
| The team: Vaibhav Jangid, Bhavya Dhoot. Each real photo, then the window's looks turn it into OpenCV line art (Canny, findContours, approxPolyDP); name cards without the private inputs | team | 0:09.0 | 0:23.1 |
| Problem and user, with sources on screen | problem | 0:23.1 | 0:46.0 |
| Disclosed before the demo plays: the footage was recorded 2026-10-23 on the live site, the trained v1 with its v1-policy2 look policy, the build the site runs now | l0 | 0:46.5 | 0:51.8 |
| /api/health on the live site: opencv 5.0.0, aarch64, aws-lambda | l1 | 0:52.0 | 0:55.9 |
| The application working: the guided run on bridge_03 ('Watch it inspect a bridge'), recorded live, with what it sent to a person to check by eye (the rebar under the confidence floor, not confirmed) | l2, l3, l4, l5, l6, l7 | 0:56.1 | 1:24.8 |
| Plain-words trace (the NOTE and 'What it decided and why' panels) | l3, l4, l5, l6, l7 | 1:00.3 | 1:24.8 |
| Human sign-off: reviewer name, Approve, APPROVED stamp; then the rebar checked by eye on the page (Defect found, with a note), and the work order ready | l9, l10 | 1:25.5 | 1:32.9 |
| Human control: a tampered work-order token refused (403) | l11 | 1:33.2 | 1:37.2 |
| Failure handling: 'Now break it' (slowed segmenter -> governor; malformed planner reply -> allow-list) | b0, b1, b2 | 1:38.0 | 1:46.4 |
| Human control: 'Try KILL' (KILL halts with kill-switch, Resume replays, then the stop rule ends it with a region unvisited) | b3 | 1:46.7 | 1:50.9 |
| Honesty: no LLM runs in the live demo; the drill planners are scripted | b4 | 1:51.3 | 1:55.2 |
| An LLM driving the OpenCV tools over MCP: IoU 0.00 at call 6 -> human; IoU 0.72 at call 18 -> confirmed; call 20 budget | mcp | 1:55.6 | 2:12.0 |
| Architecture: docs/agent-loop.svg, public/docs/aws-simple.svg, docs/architecture.png (AWS path and OpenCV layer) | arch | 2:12.0 | 2:44.7 |
| The live site now runs the trained segmenter v1 on OpenCV 5 DNN ENGINE_NEW, and its Lambda step p95 then and now (video/live/aws-smoke.log) | a4, a5 | 2:33.0 | 2:44.3 |
| Principal results: recall vs pixels, with one downscale, the tiling family and chance | r1, r3 | 2:45.2 | 2:57.4 |
| Ablations (public/results/ablations.json) | r5 | 2:57.6 | 3:04.7 |
| Real photos (CODEBRIM), the headline board: trained v1, held out (results/heldout_models.json), against the classical engine (about 1 in 5, the comparison row), its chance recall, its critical boxes (the scorer's unit) and the critical findings a person signs, and what the live look policy v1-policy2 changes (about half the sign-offs, recall unchanged); no CODEBRIM image shown | r6, r8 | 3:05.0 | 3:27.8 |
| False alarms on synthetic photos with no damage (public/results/clean/summary.json), the classical engine and v1 | r7 | 3:28.1 | 3:34.3 |
| Two real photos of our own (painted walls, no ground truth), each run live on the site: the looks (on IMG_2332 the zero-candidate fallback look, refused as blur), the findings, what it missed (video/live/walls/*/real.*) | real | 3:34.7 | 3:55.2 |
| Failure cases: failures.json plates and their rules; one CODEBRIM case in words only | failures | 3:55.2 | 4:06.1 |
| Limits and responsible use | limits | 4:06.1 | 4:13.0 |
| Reproduce: make test, make eval, scripts/aws.sh | repro | 4:13.0 | 4:17.7 |
| End card: live URL, repository, team | end | 4:17.7 | 4:23.7 |

### Chapters (from this build)

| Chapter | Starts | Ends | Lines |
| --- | --- | --- | --- |
| Cold open | 0:00.0 | 0:09.0 | o1, o2 |
| Team | 0:09.0 | 0:23.1 | t1, t2 |
| Problem and user | 0:23.1 | 0:46.0 | p1, p2, p3 |
| Live demo | 0:46.0 | 1:37.6 | l0, l1, l2, l3, l4, l5, l6, l7, l9, l10, l11 |
| Failure drills | 1:37.6 | 1:55.6 | b0, b1, b2, b3, b4 |
| An LLM over MCP | 1:55.6 | 2:12.0 | m1, m2, m3, m4 |
| Architecture | 2:12.0 | 2:44.7 | a0, a1, a3, a4, a5 |
| Results | 2:44.7 | 3:34.7 | r1, r3, r5, r6, r8, r7 |
| Real photos (ours) | 3:34.7 | 3:55.2 | q1, q2 |
| Failure cases | 3:55.2 | 4:06.1 | f1, f2 |
| Limits | 4:06.1 | 4:13.0 | x1 |
| Reproduce | 4:13.0 | 4:17.7 | y1 |
| End | 4:17.7 | 4:23.7 | e1 |

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
| guided run: critical findings | 1 | `video/live/guided.ndjson` | done.summary.critical_pending |
| guided run: agent seconds | 4.113 | `video/live/guided.ndjson` | done.summary.seconds |
| survey look: full width / L3 width | 8 | `video/live/guided.ndjson` | job.levels[0][0] / job.levels[3][0] |
| privacy pass: faces, people, vehicles blurred | [0, 0, 0] | `video/live/guided.ndjson` | job.privacy |
| survey look: what it drafted | ["corrosion", "spalling"] | `video/live/guided.ndjson` | finding[step=1], first state |
| the critical finding: class, step, rule | ["spalling", 2, "spalling 115,968 px\u00b2 >= 40,000 px\u00b2"] | `video/live/guided.ndjson` | finding[critical].cls, critical_step, critical_rule |
| held for a confirm look: class, confidence, floor, at level | ["exposed_rebar", 0.58, 0.6, 2] | `video/live/guided.ndjson` | step[2].rationale '...exposed_rebar 0.58 under the floor 0.6: held for a confirm look at L1' |
| its confirm look: level, outcome | [1, "none of exposed_rebar confirmed"] | `video/live/guided.ndjson` | step[3] cause confirm-critical, rationale |
| sent to a person to check by eye (work order not_assessed, needs-a-look) | exposed_rebar at L2, confidence 0.58 under the floor 0.6: not a critical finding; not confirmed by its L1 confirm look | `video/live/guided.ndjson` | done.work_order.not_assessed[kind=needs-a-look] |
| looks after the confirm look, and what they drafted | [3, ["crack"]] | `video/live/guided.ndjson` | step[step > confirm, not halt]; finding.step |
| why it stopped | ["exhausted", "every candidate has been looked at", []] | `video/live/guided.ndjson` | step[action=halt].cause, rationale; done.unvisited |
| the step that stopped it | 7 | `video/live/guided.ndjson` | step[action=halt].step |
| ground truth (synthetic): boxes touched by a finding of their class / by the needs-a-look entry | [["spalling", "spalling", "crack"], ["exposed_rebar"]] | `public/samples/index.json` | bridge_03 boxes against the findings and not_assessed |
| the work order after the sign-off and the check by eye: status, findings approved | ["ready", 1] | `video/live/order_status.json` | status, approvals[finding_id] |
| the rebar checked by eye: decision, note, reviewer | ["defect-found", "exposed rebar in the spall", "Demo reviewer"] | `video/live/order_status.json` | resolutions[0] |
| planner drill: rejected strategy | rm -rf / | `video/live/break.ndjson` | planner[status=rejected].reason |
| governor drill: segmenter p95 (ms) | 2287.6 | `video/live/break.ndjson` | step[5].runtime.p95_ms['segment:accurate'] |
| governor drill: model switched to | fast | `video/live/break.ndjson` | step[5].model |
| governor drill: the segmenter's SLA (ms) | 1500 | `video/live/break.ndjson` | step[5].governor.rationale |
| KILL drill: halt cause | kill-switch | `video/live/kill.ndjson` | done.cause |
| KILL drill: looks before the halt | 3 | `video/live/kill.ndjson` | done.summary.steps |
| Resume: steps marked replay | 3 | `video/live/resume.ndjson` | step[cause=replay] count |
| Resume: why it stopped, regions left unvisited | ["stop-rule", ["g8", "g9"]] | `video/live/resume.ndjson` | done.cause, done.unvisited |
| tampered token: HTTP status | 403 | `video/live/tamper.json` | status |
| tampered token: error | signature does not match | `video/live/tamper.json` | response.error |
| /api/health opencv | 5.0.0 | `video/live/health.json` | opencv |
| /api/health arch | aarch64 | `video/live/health.json` | arch |
| /api/health host | aws-lambda | `video/live/health.json` | host |
| /api/health build | stock (not COOL) | `video/live/health.json` | build |
| /api/health commit | a2b86883638f0ab0a57ba436d869b8fda4d7c0c8 | `video/live/health.json` | commit |
| /api/health segmenter | onnx:drishyam_dacl10k.onnx | `video/live/health.json` | segmenter |
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
| v1-policy2 (live look policy): critical boxes per photo, CODEBRIM held out | 8.75 | `results/heldout_models.json` | ['v1-policy2'].read.codebrim.matrix['v1-policy2@0.3'].critical_per_image |
| v1-policy2 / v1: critical boxes per photo, paired 95 % CI | [0.9115, [0.8358, 0.9817]] | `results/heldout_models.json` | pairs['codebrim@0.3: v1-policy2 / v1'].critical |
| v1-policy2 / v1: recall, paired 95 % CI | [0.9928, [0.972, 1.0134]] | `results/heldout_models.json` | pairs['codebrim@0.3: v1-policy2 / v1'].recall |
| critical findings a person signs per photo, CODEBRIM held out: v1's first policy, v1-policy2 | [2.175, 1.145] | `results/heldout_models.json` | ['v1-policy2'].read.codebrim.matrix['v1@0.3', 'v1-policy2@0.3'].signoffs_per_image |
| team portraits: the OpenCV calls in the header | ["cv2.Canny", "cv2.findContours", "cv2.approxPolyDP"] | `$DRISHYAM_PRIVATE_DIR/team/out/portraits.py` | cv2 calls in the script (OpenCV 5); private, git-ignored |
| team photos: photo-1, photo-2 | ["Vaibhav Jangid", "Bhavya Dhoot"] | `$DRISHYAM_PRIVATE_DIR/team/team.json` | photo-1.name, photo-2.name (both agreed to appear) |
| real photo IMG_2332: engine | onnx:drishyam_dacl10k.onnx | `video/live/walls/IMG_2332/real.json` | engine |
| real photo IMG_2332: profile | live site (commit a2b8688), measured costs, 30 % pixel cap | `real.json` | profile |
| real photo IMG_2332: candidate regions in the overview | 0 | `video/live/walls/IMG_2332/real.ndjson` | overview.candidates |
| real photo IMG_2332: looks | 1 | `video/live/walls/IMG_2332/real.ndjson` | done.summary.steps |
| real photo IMG_2332: share of pixels read | 0.099 | `video/live/walls/IMG_2332/real.ndjson` | done.summary.pixels_frac |
| real photo IMG_2332: findings | 0 | `video/live/walls/IMG_2332/real.ndjson` | done.summary.findings (last state of each finding event) |
| real photo IMG_2332: why it stopped | exhausted: every candidate has been looked at | `video/live/walls/IMG_2332/real.ndjson` | step[action=halt].cause, rationale |
| real photo IMG_2332: coverage (share of the overview's saliency that a look saw) | 1 | `video/live/walls/IMG_2332/real.ndjson` | done.coverage |
| real photo IMG_2332: the work order's not-assessed entries (action: reason) | ["check-by-eye: no candidate region was proposed", "re-shoot: blur"] | `video/live/walls/IMG_2332/real.ndjson` | done.work_order.not_assessed |
| real photo IMG_2332: look 1 (fallback-l2, L2): refused by the quality gate, its sharpness, its mask's px | ["blur", 5.1, 9136] | `video/live/walls/IMG_2332/real.ndjson` | step[1].quality.reason, observation.sharpness, observation.mask_area_px |
| real photo IMG_2332: what the team sees, by eye (not ground truth) | the hairline crack (crack) | `real.json` | seen |
| real photo IMG_2332: findings on it / not on it (by eye) | [[], []] | `video/live/walls/IMG_2332/real.ndjson` | finding boxes touching seen.path |
| real photo IMG_2332: each traced line touched by a finding (by eye) | [false] | `video/live/walls/IMG_2332/real.ndjson` | seen.paths[i] inside a finding box of a seen class |
| real photo IMG_2330: engine | onnx:drishyam_dacl10k.onnx | `video/live/walls/IMG_2330/real.json` | engine |
| real photo IMG_2330: profile | live site (commit a2b8688), measured costs, 30 % pixel cap | `real.json` | profile |
| real photo IMG_2330: candidate regions in the overview | 1 | `video/live/walls/IMG_2330/real.ndjson` | overview.candidates |
| real photo IMG_2330: looks | 1 | `video/live/walls/IMG_2330/real.ndjson` | done.summary.steps |
| real photo IMG_2330: share of pixels read | 0.0545 | `video/live/walls/IMG_2330/real.ndjson` | done.summary.pixels_frac |
| real photo IMG_2330: findings | 1 | `video/live/walls/IMG_2330/real.ndjson` | done.summary.findings (last state of each finding event) |
| real photo IMG_2330: why it stopped | exhausted: every candidate has been looked at | `video/live/walls/IMG_2330/real.ndjson` | step[action=halt].cause, rationale |
| real photo IMG_2330: coverage (share of the overview's saliency that a look saw) | 0.2054 | `video/live/walls/IMG_2330/real.ndjson` | done.coverage |
| real photo IMG_2330: the work order's not-assessed entries (action: reason) | [] | `video/live/walls/IMG_2330/real.ndjson` | done.work_order.not_assessed |
| real photo IMG_2330: what the team sees, by eye (not ground truth) | the hairline cracks (crack) | `real.json` | seen |
| real photo IMG_2330: findings on it / not on it (by eye) | [["f1"], []] | `video/live/walls/IMG_2330/real.ndjson` | finding boxes touching seen.path |
| real photo IMG_2330: each traced line touched by a finding (by eye) | [true, false] | `video/live/walls/IMG_2330/real.ndjson` | seen.paths[i] inside a finding box of a seen class |
| the quality gate's blur threshold (Laplacian variance) | 12.0 | `drishyam/imaging.py` | SHARPNESS_MIN |
| competition | OpenCV AI Competition 2026, powered by AWS | `README.md` | line 3 |
| entry | Agentic Vision | `docs/PRD.md` | awards line |
| name | Sanskrit: that which is seen | `public/index.html` | the hero's name gloss |
| team and roles | ["Vaibhav Jangid, lead engineer (agent policy, OpenCV pipeline, evaluation, AWS deployment)", "Bhavya Dhoot, design, demo and presentation (site design, video)"] | `README.md` | Team |
| problem-sheet citations | ["FHWA, Bridge Condition by Highway System 2026", "23 CFR 650.311", "SPR 787", "Table 4.1"] | `docs/impact-sources.md` | sections 1, 2, 5 |
| MCP client | Claude (Opus 5.5) inside Claude Code | `docs/mcp-session/transcript.md` | line 4 |
| Lambda CPU | arm64 (Graviton) | `docs/aws.md` | line 4 |
| health card: what the Lambda sits behind | behind CloudFront | `README.md` | Live demo line |
| reproduce sheet: what the CDK stack is | Lambda arm64 behind CloudFront and WAF | `infra/drishyam_stack.py` | stack description |
| the live engine's look policy, and since when | With v1-policy2, live since 2026-10-22 | `public/index.html` | How it works, the confirm look |
| what it does, for whom (p3) | it chooses where to look, zooms in where its segmenter is unsure, drafts cracks, spalling and other defects for review, and hands every critical finding to a person | `public/index.html` | the hero's .lede |
| cold-open title | ONE PHOTO, A FEW LOOKS, ONE HUMAN | `public/index.html` | the site's figure note |
| limits, in the site's words | ["Pixels, not millimetres", "AI-assisted, not an inspection", "Only a person can approve", "Its licence does not allow sharing the photos"] | `public/index.html` | footer note, Human sign-off, the CODEBRIM citation |
| reproduce: what make test runs | unit, API, MCP, CLI and eval tests (same as CI) | `Makefile` | 'test:' comment |
| footage recorded on | 2026-10-23 | `brag-output/work/rec/events.json` | file date |
| live site | https://d1xg58guluup8c.cloudfront.net | `README.md` | Live demo |
| repository | https://github.com/vj-io/drishyam | `README.md / public/index.html` | GitHub link |
| health check | https://d1xg58guluup8c.cloudfront.net/api/health | `video/live/health.json` | recorded response |

### Narration and captions (from video/narration.py)

| Id | Starts | Spoken | Caption |
| --- | --- | --- | --- |
| o1 | 0:01.5 | Its trained engine read a quarter of the pixels and flagged one critical finding. | Its trained engine read a quarter of the pixels and flagged 1 critical finding. |
| o2 | 0:06.3 | It will not sign it off. A person must. | It will not sign it off. A person must. |
| t1 | 0:09.5 | Drishyam, an Agentic Vision entry to the OpenCV AI Competition, powered by AWS. | Drishyam, an Agentic Vision entry to the OpenCV AI Competition, powered by AWS. |
| t2 | 0:17.4 | Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation. | Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation. |
| p1 | 0:23.6 | America has about six hundred twenty-five thousand bridges, inspected at least every two years. | America has about 625,000 bridges, inspected at least every two years. |
| p2 | 0:29.5 | One drone inspection took more than twenty-five hundred photos of one bridge. | One drone inspection took more than 2,500 photos of one bridge. |
| p3 | 0:34.3 | Drishyam is for the engineer reviewing them: it chooses where to look, zooms in where its segmenter is unsure, drafts defects for review, and hands every critical finding to a person. | Drishyam is for the engineer reviewing them: it chooses where to look, zooms in where its segmenter is unsure, drafts defects for review, and hands every critical finding to a person. |
| l0 | 0:46.5 | Recorded October twenty-third on the live site: the trained v1 and its new look policy. | Recorded 23 October on the live site: the trained v1 and its new look policy. |
| l1 | 0:52.0 | Live on AWS: OpenCV five on an arm Lambda. | Live on AWS: OpenCV 5 on an arm Lambda. |
| l2 | 0:56.1 | Watch it inspect a bridge; a privacy pass finds nothing to blur. | Watch it inspect a bridge; a privacy pass finds nothing to blur. |
| l3 | 1:00.3 | One survey look at the whole photo, at one eighth size, finds spalling and corrosion. | One survey look at the whole photo, at one eighth size, finds spalling and corrosion. |
| l4 | 1:05.4 | One level finer, the spalling is big enough to be critical, but the reebar scores under the confidence floor. | One level finer, the spalling is big enough to be critical, but the rebar scores under the confidence floor. |
| l5 | 1:11.8 | It takes one confirm look, centred on the reebar, and does not confirm it. | It takes one confirm look, centred on the rebar, and does not confirm it. |
| l6 | 1:16.3 | So the reebar goes to a person to check by eye, not as a critical finding. | So the rebar goes to a person to check by eye, not as a critical finding. |
| l7 | 1:20.9 | Three more looks find a crack; with every region looked at, it stops. | Three more looks find a crack; with every region looked at, it stops. |
| l9 | 1:25.5 | Only a person can approve the spalling. | Only a person can approve the spalling. |
| l10 | 1:27.8 | A person then checks the reebar by eye and rekords a defect; now the order is ready. | A person then checks the rebar by eye and records a defect; now the order is ready. |
| l11 | 1:33.2 | Change one character of the order's signature: the server refuses it. | Change one character of the order's signature: the server refuses it. |
| b0 | 1:38.0 | Now, break it. | Now, break it. |
| b1 | 1:39.3 | Slow the segmenter, and the governor switches to the fast model. | Slow the segmenter, and the governor switches to the fast model. |
| b2 | 1:43.0 | The planner drill sends a shell command; the allow-list rejects it. | The planner drill sends a shell command; the allow-list rejects it. |
| b3 | 1:46.7 | KILL halts it before the next look; Resume replays the moves and carries on. | KILL halts it before the next look; Resume replays the moves and carries on. |
| b4 | 1:51.3 | No language model runs live; these planners are scripted to fail. | No language model runs live; these planners are scripted to fail. |
| m1 | 1:56.1 | Over MCP, Claude drove the same tools on the same photo. | Over MCP, Claude drove the same tools on the same photo. |
| m2 | 2:00.3 | At call six the rebar masks disagreed, so Claude handed it to a human. | At call 6 the rebar masks disagreed, so Claude handed it to a human. |
| m3 | 2:04.8 | At call eighteen they agreed on a spall; at call twenty, the budget ended the search. | At call 18 they agreed on a spall; at call 20, the budget ended the search. |
| m4 | 2:10.0 | It still missed the large spall. | It still missed the large spall. |
| a0 | 2:12.5 | Each step is perception, decision and action, with a human gate at the end. | Each step is perception, decision and action, with a human gate at the end. |
| a1 | 2:17.1 | CloudFront and a rate-limiting firewall front the arm Lambda; DynamoDB keeps kill flags, Secrets Manager the key. | CloudFront and a rate-limiting firewall front the arm Lambda; DynamoDB keeps kill flags, Secrets Manager the key. |
| a3 | 2:24.8 | OpenCV does the looking: a pyramid, saliency, warp perspective as a pan-tilt-zoom camera, a blur gate, face blurring. | OpenCV does the looking: a pyramid, saliency, warpPerspective as a pan-tilt-zoom camera, a blur gate, face blurring. |
| a4 | 2:33.0 | The site now runs v1 on OpenCV five's new DNN engine. | The site now runs v1 on OpenCV 5's new DNN engine. |
| a5 | 2:38.1 | On Lambda a step took under ninety milliseconds at p ninety-five then, and under one point two seconds now. | On Lambda a step took under 90 ms at p95 then, and under 1.2 s now. |
| r1 | 2:45.2 | On two hundred synthetic photos it finds eighty-seven percent of full tiling's recall at twenty-six percent of the pixels, short of its ninety percent target. | On 200 synthetic photos it finds 87 % of full tiling's recall at 26 % of the pixels, short of its 90 % target. |
| r3 | 2:55.2 | One downscaled pass does about as well. | One downscaled pass does about as well. |
| r5 | 2:57.6 | Ablations: only dropping the survey look loses recall, and a random gaze order matches it on smaller photos. | Ablations: only dropping the survey look loses recall, and a random gaze order matches it on smaller photos. |
| r6 | 3:05.0 | On two hundred real CODEBRIM photos, held out, the trained v1 finds forty-nine percent: two and a half times the classical engine's one in five. | On 200 real CODEBRIM photos, held out, the trained v1 finds 49 %: two and a half times the classical engine's 1 in 5. |
| r8 | 3:14.5 | But its boxes find thirty percent at random, and it draws five and a half times the critical boxes. A person signs about two critical findings a photo, about one with the live policy, and recall is unchanged. | But its boxes find 30 % at random, and it draws five and a half times the critical boxes. A person signs about two critical findings a photo, about one with the live policy, and recall is unchanged. |
| r7 | 3:28.1 | On fifty-one undamaged synthetic photos, the classical engine flags two in three; v1, none. | On 51 undamaged synthetic photos, the classical engine flags 2 in 3; v1, none. |
| q1 | 3:35.2 | Our own wall, no ground truth: no region was proposed, so it looked at the whole photo. The segmenter saw crack pixels, but the blur gate refused this sharp view, and nothing was flagged. | Our own wall, no ground truth: no region was proposed, so it looked at the whole photo. The segmenter saw crack pixels, but the blur gate refused this sharp view, and nothing was flagged. |
| q2 | 3:47.9 | On a second wall, one close look flagged a crack on the lower of two we traced, and never looked closely at most of the wall. | On a second wall, one close look flagged a crack on the lower of two we traced, and never looked closely at most of the wall. |
| f1 | 3:55.7 | Failures in the open: graffiti read as spalling, a formwork line as a crack, glare, and blur. | Failures in the open: graffiti read as spalling, a formwork line as a crack, glare, and blur. |
| f2 | 4:01.9 | Only the glare gate caught its case; three fixes are not built yet. | Only the glare gate caught its case; three fixes are not built yet. |
| x1 | 4:06.6 | Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve. | Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve. |
| y1 | 4:13.5 | Make test, make eval, and one script for the AWS stack. | make test, make eval, and one script for the AWS stack. |
| e1 | 4:18.1 | Drishyam. It looks where it matters, and asks a human first. | Drishyam. It looks where it matters, and asks a human first. |
<!-- generated:end -->
