# 2,536 photos through `drishyam batch --router nova`

A backlog the size of one real drone inspection (docs/report.md, section 1), from dev-safe photos only (evals/backlog_2536.py): 1,220 synthetic photos with damage (seed 2536) and 1,216 without (seeds 2537, 2539, 2541, 2543), 4608x3456 and labelled, plus the 100 CODEBRIM dev photos, in a seeded shuffle (seed 2536). v1 on the fixed costs of results/local-macos-x86-dacl10k-retime, default per-photo caps, a total budget of $0.50 (the agent's compute at $0.339 an hour plus the router's Bedrock $). Batch batch-19ef1b, 2026-10-21T20:26:37Z; the numbers: [summary.json](summary.json).

| | |
| --- | --- |
| Photos triaged and routed | 2,536 (2,436 Nova Lite calls, 100 from the cache) |
| Router labels | 867 Clear, 1,664 Requires zoom, 5 Critical; 8 fell back to the triage (timeout: 8), 0 rejected |
| Photos run | 587 (5 Critical, then 582 Requires zoom in triage order); 1,082 not reached; 867 cleared, not looked at; 0 unusable |
| Critical findings queued for a person | 684 (684 pending review); they hit 607 of the 1,568 labelled critical defects |
| Photos with a labelled critical defect | 746: 294 run, 40 cleared by the router (miss rate 0.054, Wilson CI 0.040-0.072) |
| Router labels against the labels | accuracy 0.43; cleared 675 of 1,218 photos with no labelled defect and 152 of 572 with only non-critical ones |
| $ | Bedrock $0.316 (4,942,210 input, 79,452 output tokens); compute $0.184 modelled; total $0.500 of $0.50 |
| Time | 72 min on the wall; the router's $ charged as 3,352 s (63% of the 5,310 s budget); triage 168 s |
| Router latency | p50 1.35 s, p95 4.70 s, slowest 30.0 s |
| Status | blocked: 684 critical findings wait for `drishyam review`; 1,949 photos not assessed |
