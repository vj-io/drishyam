# Drishyam over MCP, and the command line

`drishyam mcp` serves the inspector to an LLM client over stdio. The server is `drishyam/mcp_server.py`
(mcp 2.x, `mcp.server.mcpserver.MCPServer`); install it with `pip install 'drishyam[mcp]'`.

```bash
claude mcp add drishyam -- drishyam mcp          # Claude Code
npx @modelcontextprotocol/inspector drishyam mcp  # MCP Inspector: lists and calls every tool
```

## Tools (a fixed allow-list of twelve)

| Tool | What it does |
| --- | --- |
| `open_image(path)` | Starts a job on a JPEG/PNG under an allowed image root; people are blurred first. Returns the job id and caps. |
| `overview(job)` | At most 20 ranked candidate boxes (L0 pixels) and a saliency heatmap PNG. |
| `gaze(job, x, y, level, yaw=0, pitch=0)` | Virtual-PTZ look at an L0 pixel at pyramid level 0-3. Returns a gaze id, window, quality gate and a JPEG of the crop (at most 512 px). |
| `enhance(job, gaze_id)` | CLAHE re-look. Replaces that gaze's crop; segment again afterwards. |
| `segment(job, gaze_id, model)` | Per-class areas, confidence, entropy, latency. `model` is `accurate` or `fast`. |
| `measure(job, gaze_id, cls)` | Length, width, area, orientation of one class, in L0 pixels. |
| `verify(job, gaze_id)` | Looks one level finer at the same centre, segments, returns the mask IoU against the coarser look (below 0.5 is disagreement). |
| `runtime_stats(job)` | p50/p95 per op, CPU %, $ so far, budget against caps, what the governor changed. |
| `set_budget(job, pixel_cap, usd_cap, seconds_cap)` | Lowers caps (0.02-1.0, 0.0001-0.05, 1-30). Raising one is refused. |
| `log_finding(job, gaze_id, cls)` | Records a class that the last segmentation shows. Critical ones become `pending_review`. |
| `request_human_review(job, finding_id, reason)` | Records the request in the trace. Changes no state. |
| `draft_work_order(job)` | Writes `runs/<job>/work_order.json` and `crops/` and returns the order. |

There is no shell, no file-write and no approve/reject tool, and no OCR: tool results hold numbers, class
names and ids, so text painted on a bridge cannot reach the model. `open_image` resolves the real path and
refuses anything outside the allowed roots (`..`, absolute paths, symlinks that lead out), and anything that
is not a JPEG or PNG by magic bytes or is over 25 MP.

## What the human controls

- **Approval.** Only `drishyam review WORK_ORDER FINDING_ID approve|reject --by NAME` moves a critical finding
  out of `pending_review`. A later `draft_work_order` keeps decisions already recorded in the file.
- **Caps.** The starting caps are the agent defaults (30 % of pixels, $0.05, 20 s of tool compute), or what you
  set in `DRISHYAM_PIXEL_CAP`, `DRISHYAM_USD_CAP`, `DRISHYAM_SECONDS_CAP` before starting the server. The model can
  only lower them.
- **Kill switch.** `drishyam kill` creates the flag file (`DRISHYAM_KILL_FILE`, default `~/.drishyam/KILL`);
  `DRISHYAM_KILL_SWITCH=1` also works. Every tool call, `open_image` included, is refused while it is set, so an
  LLM-driven job stops within one call. `drishyam unkill` clears it. After a cap is hit, the looking tools are
  refused but `runtime_stats`, `set_budget`, `log_finding`, `request_human_review` and `draft_work_order` still
  work, so a capped job can be written up.
- **Where things live.** `DRISHYAM_IMAGE_ROOT` (default `./data` and `./public/samples`; separate several
  with `:`) and `DRISHYAM_RUNS_DIR` (default `./runs`).

The governor runs before every call. When p95 segment latency passes the SLA it downgrades the model, and a
`segment` reply says so under `governor`; a slower warp shrinks later looks, and at 80 % of the $ cap `verify`
only accepts gazes with a critical finding.

## Trace

Every call appends one line to `runs/<job>/trace.jsonl`. Looks (`gaze`, `verify`) are `step` records with the
shape in `api-contract.md` (`decision.args` holds the centre actually used, `window`, `quality`, `runtime`,
`cause`, `pixels_used`). Other calls are `tool` records, refusals are `step` records with action `halt`, and
failures are `error` records. `drishyam replay runs/<job>/trace.jsonl` prints the gaze path and checks it
against `work_order.json`. Calls rejected by schema validation (a level of 9, say) never reach the server code,
so they leave no line.

## Command line

```text
drishyam run IMAGE [--pixel-cap 0.3] [--usd-cap 0.05] [--seconds-cap 20] [--chaos p95_spike,...]
                   [--out runs] [--resume TRACE]
drishyam review WORK_ORDER FINDING_ID approve|reject --by NAME
drishyam replay TRACE
drishyam kill | unkill
drishyam mcp
drishyam serve [--port 8000]
```

`run` streams one line per event to stderr and writes `runs/<job>/{trace.jsonl, work_order.json, crops/}`. Ctrl-C
halts within one step and keeps the state (exit code 130); `--resume` re-executes the decisions of an earlier
trace (a trace cut off mid-line is fine) and continues in a new run folder.
