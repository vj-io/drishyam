# A backlog of photos: `drishyam batch`

One drone inspection produced 2,536 photos ([report.md](report.md), section 1). `drishyam run` inspects one of them.
`drishyam batch` inspects a backlog: it decides which photos to inspect first, runs the unchanged per-photo agent on
each under one total budget, and writes one signed inspection report with a single queue of critical findings for a
person. It approves nothing.

```bash
drishyam batch FOLDER [MORE FOLDERS, PHOTOS OR .txt LISTS] [--usd-cap TOTAL] [--seconds-cap TOTAL] \
    [--photo-pixel-cap 0.3] [--photo-usd-cap 0.05] [--photo-seconds-cap 20] [--cost-model live|fixed] [--out runs]
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
| `batch_trace.jsonl` | One record per batch decision (`triaged`, `ordered`, `started`, `finished`, `budget-stop`, `killed`, `refreshed`), each with its `cause`, a `rationale` that cites the measured values, and the seconds spent so far |
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

## Not done

- No MCP tool: a read-only `batch_status` would be small, but it would change the MCP allow-list of twelve that
  [mcp.md](mcp.md) documents, and reading the key must not create a file from an MCP call.
- Photos run one at a time, and each is decoded twice (triage, then its job): a backlog does not fit in memory.
- Triage reads the overview only. Its score does not see blur (the quality gate runs on each look), nor which class a
  candidate holds; a photo whose damage the saliency map misses is reached late or not at all, and is then listed as
  not reached, never as clean.
- Non-critical review requests (ambiguous, disagree, class change, unconfirmed) stay in each photo's trace; the batch
  queue holds the critical findings, which are what blocks a work order.
