# A backlog of photos: `drishyam batch`

One drone inspection produced 2,536 photos ([report.md](report.md), section 1). `drishyam run` inspects one of them.
`drishyam batch` inspects a backlog: it decides which photos to inspect first, runs the unchanged per-photo agent on
each under one total budget, and writes one signed inspection report with a single queue of critical findings for a
person. It approves nothing.

```bash
drishyam batch FOLDER [MORE FOLDERS, PHOTOS OR .txt LISTS] [--usd-cap TOTAL] [--seconds-cap TOTAL] \
    [--photo-pixel-cap 0.3] [--photo-usd-cap 0.05] [--photo-seconds-cap 20] [--cost-model live|fixed] [--out runs] \
    [--router rule|nova]
# a person reviews each critical finding in its photo's work order, as for one photo
drishyam review runs/batch-<id>/<job>/work_order.json f1 approve --by "A. Engineer"
# then the batch report is recomputed from the work orders and signed again
drishyam batch --report runs/batch-<id>
```

A folder contributes its JPEG and PNG files in name order (a drone numbers them in capture order); a `.txt` list holds
one photo path per line, relative to the list. `--usd-cap` and `--seconds-cap` are the totals for the whole batch
(no cap if omitted); the `--photo-*` caps are each photo's, as `drishyam run`'s `--pixel-cap`, `--usd-cap` and
`--seconds-cap`. `drishyam kill` (or Ctrl-C) stops a batch: between photos, and inside the running photo at its next
step. Each run leaves `runs/batch-<id>/`:

| File | What it holds |
| --- | --- |
| `batch_trace.jsonl` | One record per batch decision (`triaged`, with `--router nova` also `routed` and `cleared`, then `ordered`, `started`, `finished`, `budget-stop`, `killed`, `refreshed`), each with its `cause`, a `rationale` that cites the measured values, and the seconds spent so far |
| `<job>/trace.jsonl`, `<job>/work_order.json`, `<job>/crops/` | Each photo's own run, exactly what `drishyam run` writes |
| `inspection.json` | The report, sealed with the work-order HMAC (`findings.sign`) |
| `inspection.md` | The same, for a person: status, the human queue, everything not assessed, each photo in the order run |

## What it decides, and on what

1. **Triage.** Every photo gets the agent's own overview, as its job will compute it: decoded, people blurred
   (FR-19), then the first two events of `Job.run` (the pyramid, the L3 saliency map, candidate windows), so the
   candidates are exactly those the photo's job will look at. No new model. The score is the sum of the candidates'
   priors: before any look a region's expected value in the agent's own score is its prior (severity 1, no entropy
   yet), so the sum is the photo's expected value in the same units. A photo the decoder refuses, or one mostly filled
   by a person, is listed as not assessed with the reason.
2. **Order.** Highest score first; photos with no candidate (score 0) go last, in file order.
3. **Run.** The unchanged per-photo `Job`, one photo at a time, with the per-photo caps. Triage and every job are
   charged to one total on the job's own clock (the $ cap is the same budget in seconds, since the agent models $
   from seconds; decoding and the privacy blur are not charged, as with `drishyam run`). Each job gets at most what is
   left as its seconds cap, so the last photo can halt early on `cap-hit`: its unlooked regions are in its own work
   order and so in the batch's not-assessed list. Like a job's own cap, the total can be passed by up to one look.
   When nothing is left, the batch stops (`budget-stop`) and every photo not reached is listed as not assessed with
   that reason.
4. **Report.** Each photo's status (`ready`, `incomplete`, `blocked`, or `not assessed`), the human queue (every
   critical finding across all photos, by severity, then confidence), the not-assessed list (photos never reached or
   unusable, zero-candidate photos, the regions a halted job did not look at, glare and dark areas, quality-gate
   refusals) and the totals ($, seconds, triage seconds, pixels). The batch is `blocked` while any critical finding
   waits for a person, `incomplete` while anything is not assessed, and `ready` only when neither is left.

The human gate is the per-photo one. `drishyam batch --report DIR` verifies the seal, re-reads every photo's work
order and seals the result again. It refuses when the seal does not verify, or when a critical finding the sealed
report listed is gone, no longer critical, or left `pending_review` (or changed a decision) without a matching approval
recorded in its work order. The key is `DRISHYAM_SIGNING_KEY` (32 characters or more, as for the web app), else a
random key kept owner-only in `~/.drishyam/signing_key`, made on first use.

## Does triage help? (dev only)

`python -m evals.batch_triage` writes [results/dev/batch_triage.json](../results/dev/batch_triage.json). The backlog is
the two dev splits only: synth_dev (60 photos with damage, seed 4242) and the first 30 of synth_clean_dev (distractors,
no damage, seed 4243). The 90 photos are listed in a seeded shuffle that stands in for capture order ("file order").
Each job runs as `drishyam batch` runs it (default per-photo caps, the classical-v1 engine, the published fixed costs
of results/linux-aarch64/micro.json). A labelled critical defect (exposed rebar, or spalling whose box is at least
the critical area) counts once a critical finding of its class hits it under the harness's rule. The three arms are
triage (the batch's order, with triage's own cost charged first), file order, and random order (the mean of 1000
seeded permutations). Any order and budget is replayed from one full run per photo, and a real `drishyam batch` run of
the triage arm at the primary budget matched the replay: the same 42 photos reached, 16 critical defects queued, and
8.410 s spent.

**The rule, committed before the run (75fad94, `RULE` in evals/batch_triage.py).** The primary metric is critical
recall within a total budget of half the seconds the 90 jobs take in full. Triage counts as a gain only if the paired
95 % bootstrap CI of triage / random order lies above 1 and so does that of triage / file order. It counts as harm if
either CI lies below 1. Anything else is no measurable difference.

**Verdict: no measurable difference.** The point estimates favour triage, but both intervals include 1.

At the primary budget (8.37 s of the 16.75 s the 90 jobs take in full; triage costs 0.61 s of it):

| Arm | Critical recall (of 51) | IoU-only | Photos reached | First critical queued at | Queued criticals matching no label |
| --- | --- | --- | --- | --- | --- |
| triage | 0.314 (16) | 0.196 | 42 | 0.68 s | 2 |
| file order | 0.216 (11) | 0.157 | 46 | 0.13 s | 4 |
| random order | 0.255 (13.0) | 0.157 | 45.3 | 0.78 s | 4.5 |

Ratios of triage to each arm, with paired 95 % bootstrap CIs (1000 resamples of the photos within each split, 100
random orders per resample):

| Budget | Recall vs random | Recall vs file | First critical vs random | First critical vs file |
| --- | --- | --- | --- | --- |
| a quarter (4.19 s) | 1.08 (0.27-1.88) | 0.70 (0.14-2.40) | 0.87 (0.61-3.38) | 5.23 (3.43-30.46) |
| **half (8.37 s), primary** | **1.23 (0.64-1.64)** | **1.45 (0.53-4.00)** | 0.87 (0.58-3.44) | 5.23 (3.43-30.46) |
| three quarters (12.56 s) | 1.18 (0.98-1.33) | 1.10 (0.86-1.50) | 0.87 (0.58-3.62) | 5.23 (3.43-30.46) |

What the numbers say:

- **Triage is slower to the first critical than this file order (x5.2, CI 3.4-30).** The whole triage pass (0.61 s)
  runs before any job, and the shuffled listing happens to open on a photo that holds a critical defect. Against
  random order there is no measurable difference.
- **The per-photo agent caps every arm.** Run in full on all 90 photos, it queues 26 of the 51 labelled critical
  defects, so half the budget can reach at most about that many.
- **Diagnostics, not part of the verdict.** Clean photos score lower on average (prior sum 1.58 against 2.07 for
  photos with damage). The first half of the triage order holds 9 of the 30 clean photos, against 15 expected from a
  random half, and 33 of the 51 labelled critical defects. Run in full, those 45 photos queue 16 of the 26 critical
  defects a full run finds.

The ranking moves damaged photos forward. With 51 critical defects in 29 photos, and half of them out of the
per-photo agent's reach, this dev backlog cannot separate that from chance. The batch layer is still the part that
orchestrates the backlog, and it does not depend on the ranking: one budget, a stop with every photo it did not reach
listed, one human queue, and a sealed report.

## A live LLM router: `--router nova`

`drishyam batch --router nova` puts a live LLM, Amazon Nova Lite (`amazon.nova-lite-v1:0`) on Amazon Bedrock in
us-east-1, between the triage and the runs ([drishyam/router.py](../drishyam/router.py)). `--router rule`, the
default, is the batch above, unchanged.

1. **Map, one call per photo.** The router sends the photo's own L3 overview, the pyramid level the saliency map and
   the survey read, decoded and with people already blurred, at most 896 px on its long side (Nova rescales any image
   so one side passes 896 px, so a larger one buys no detail), and the OpenCV measurements of it: the triage's
   candidate count and priors, saliency statistics (mean, 95th percentile, share above 0.5) and the quality gate run
   on the overview. No file name or path is sent. One fixed prompt asks for one JSON object,
   `{"label": "Clear" | "Requires zoom" | "Critical", "reason": "<one line>"}`, at temperature 0 with at most 120
   output tokens. Eight calls run at once while the next photos are decoded.
2. **Validation.** The reply must be exactly that object (or the whole reply as one ```` ```json ```` block, which
   Nova Lite adds): exactly the two keys, one of the three labels, and a reason of 1 to 200 printable characters
   with no newline and none of `` <>[]`|\ ``. Anything else is `rejected`; a timeout (30 s), a throttle, an access
   error or any other exception is a `fallback`. Either way the photo gets the rule triage's treatment, label
   Requires zoom: it is run, in prior-sum order, and never cleared. Every reply is one `routed` line of
   batch_trace.jsonl with its status, the cause, the latency and its $; a rejected reply's first 200 characters are
   kept there too.
3. **Cache.** An accepted reply is cached by the photo's sha256 together with the model and the prompt
   (`DRISHYAM_ROUTER_CACHE`, default `~/.drishyam/router_cache`), so a rerun of the same photos makes no call and
   costs nothing, whatever the files are named. Failures are not cached: a rerun asks again.
4. **Reduce.** A Clear photo gets no job. It is listed under not assessed as "cleared by the router (not looked at
   closely)" with the router's reason, and the action "look at it, or run it with --router rule"; it is never
   reported as assessed or clean, and a batch with a cleared photo is never `ready`. Critical photos run first, then
   Requires zoom, each in the triage's order, under the batch budget, which is also charged the router's Bedrock $
   as seconds at the agent's $0.339 per hour (on a rerun, cached labels cost nothing and charge nothing). The human
   gate is the per-photo one, unchanged: the router only orders and clears, and no label approves anything.
5. **Kill.** The kill switch stops the map step before the next photo's triage or before the next reply is taken
   (calls not yet started are cancelled), and the reduce step as before. A batch killed during the map clears
   nothing: every photo is listed as not reached.

The router sees pixels, which the planner never does ([planner.py](../drishyam/planner.py) gets numbers only), so
text painted on a wall could steer its label, though the prompt says to ignore it. A label can only order a photo or
clear it, and a cleared photo stays in the report as not assessed. Credentials come from the default AWS chain
(`AWS_PROFILE`); a profile made by `aws login` also needs `pip install "botocore[crt]"`.

**Price.** On-demand Nova Lite in us-east-1 is $0.00006 per 1,000 input tokens and $0.00024 per 1,000 output tokens
(AWS Price List API, `USE1-NovaLite-input-tokens` and `-output-tokens`, read 2026-10-21); an image is billed as input
tokens. Measured: a 2048x1536 photo's 256x192 overview is about 830 input tokens ($0.000057 a call), a 4608x3456
photo's 576x432 overview about 2,036 ($0.00013 a call), and a reply about 30 output tokens.

### Does the router help? (dev only)

`python -m evals.router_eval` writes [results/dev/router_eval.json](../results/dev/router_eval.json). The backlog is
the dev splits only: synth_dev (60), the first 30 of synth_clean_dev and CODEBRIM dev100, 190 photos in a seeded
shuffle (file order). Each photo's job runs once with v1 (`onnx:drishyam_dacl10k.onnx`, its FP16 net as the fast
model) on the fixed costs of results/local-macos-x86-dacl10k-retime/micro.json, and any order and budget is replayed
from those runs as in the triage evaluation above; each photo is routed once through drishyam.router. The 190 jobs
take 487.1 s run in full; triage costs 7.3 s; the router's 190 calls cost $0.0163, which the batch charges as 173.5
s. A real `drishyam batch --router nova` run at the primary budget matched the replay: 71 photos reached, 209
labelled critical defects queued (its labels came from the cache, so it charged no router $, as the replay with the
router's $ not charged).

**The rule, committed before any paid router call (0d38696, `RULE` in evals/router_eval.py).** The primary metric is
critical recall within a total budget of half the seconds the 190 jobs take in full, the router arm charged its
triage and its Bedrock $. Gain: the paired 95 % bootstrap CIs of router / rule triage and of router / random order
both lie above 1. Harm: either lies below 1. Safety: the share of photos holding a labelled critical defect that the
router clears, with a Wilson 95 % CI; clearing is called safe only if its upper bound is at most 0.05.

**Verdict: harm, and clearing is not shown safe.**

At the primary budget (243.6 s, of which the router's $ is 173.5 s):

| Arm | Critical recall (of 370) | IoU-only | Photos reached | First critical queued at | Queued criticals matching no label |
| --- | --- | --- | --- | --- | --- |
| router, its $ charged | 0.181 (67) | 0.059 | 19 | 181.5 s | 27 |
| router, its $ not charged | 0.565 (209) | 0.149 | 71 | 8.0 s | 83 |
| rule triage | 0.497 (184) | 0.124 | 64 | 8.0 s | 77 |
| file order | 0.332 (123) | 0.097 | 92 | 3.0 s | 53 |
| random order | 0.331 (122.4) | 0.102 | 95.5 | 6.1 s | 49.9 |

Recall ratios of the router to each arm, with paired 95 % bootstrap CIs (1000 resamples within each split):

| Budget | vs rule triage | vs random | $ not charged, vs rule triage | $ not charged, vs random |
| --- | --- | --- | --- | --- |
| a quarter (121.8 s) | 0 (0-0) | 0 (0-0) | 1.44 (1.07-3.31) | 1.46 (0.96-2.51) |
| **half (243.6 s), primary** | **0.36 (0.13-0.67)** | **0.55 (0.21-0.88)** | 1.14 (0.98-1.44) | 1.71 (1.53-1.85) |
| three quarters (365.3 s) | 0.88 (0.67-0.94) | 1.08 (0.82-1.17) | 1.03 (0.98-1.08) | 1.27 (1.19-1.32) |

What the numbers say:

- **The router costs more than it saves.** One call is $0.000086 a photo on this backlog, 36 % of the modelled $
  of the photo's whole v1 job ($0.000241). Routing all 190 photos costs 173.5 s of the 243.6 s budget, so the router
  arm reaches 19 photos where the triage alone reaches 64; at a quarter of the budget it reaches none.
- **Its order, free, is no measurable gain over the triage.** With its $ not charged, Critical first and Clear
  skipped queues 209 critical defects against the triage's 184: 1.14 (CI 0.98-1.44).
- **It clears photos that hold critical defects.** It cleared 36 of the 190 photos: 8 of the 32 with no labelled
  defect, 19 with only non-critical ones, and 9 of the 81 holding a labelled critical defect (miss rate 0.111, Wilson
  CI 0.060-0.198; 2 of 29 on synth_dev, 7 of 52 on CODEBRIM). Those 9 hold 16 of the 370 labelled critical defects,
  9 of which the agent queues when it runs them in full.
- **It rarely says Critical.** Labels against the labels (rows: label; columns: router, Clear / Requires zoom /
  Critical): no defect 8 / 24 / 0, non-critical 19 / 58 / 0, critical 9 / 68 / 4; accuracy 0.37. All four
  Critical calls were right, all on CODEBRIM.
- **Latency and failures.** At a 10 s timeout the first pass timed out 37 of the 190 calls (19 %, most of them
  4608x3456 CODEBRIM photos; 9 timed out again on a second pass, with the machine at a load average of 100 to 500
  from other work); one such call alone took 5.9 s. Failures are not cached, so each rerun asked again; at 30 s every
  photo had an accepted reply (p50 1.2 s on the first pass, 1.7 s for the last 10, the slowest 3.0 s). No reply was
  rejected once the ```json fence was accepted.

## Not done

- No MCP tool: a read-only `batch_status` would be small, but it would change the MCP allow-list of twelve that
  [mcp.md](mcp.md) documents, and reading the key must not create a file from an MCP call.
- Photos run one at a time, and each is decoded twice (triage, then its job): a backlog does not fit in memory.
- Triage reads the overview only. Its score does not see blur (the quality gate runs on each look), nor which class a
  candidate holds; a photo whose damage the saliency map misses is reached late or not at all, and is then listed as
  not reached, never as clean.
- Non-critical review requests (ambiguous, disagree, class change, unconfirmed) stay in each photo's trace; the batch
  queue holds the critical findings, which are what blocks a work order.
- The router runs from the CLI only: no web, MCP or Lambda path calls it, and nothing in infra/ changed.
- The router reads one small overview. At 2048x1536 that is 256x192 px, where a 1-2 px crack is gone; it cannot see
  what the per-photo agent zooms for, so it can order and skip photos but not assess one.
- A call that times out is not cached and its tokens are not counted: Bedrock may still bill it, so the router's $
  in the report is a floor (CloudWatch's token counts for the model are the bill).
- The cache has no expiry or size cap (one small JSON file per photo).
- Each Clear is one model's judgement of a thumbnail; nothing re-checks a sample of the cleared photos.
