# Drishyam over MCP, and the command line

`drishyam mcp` serves the inspector to an LLM client over stdio. The server is `drishyam/mcp_server.py`
(mcp 2.x, `mcp.server.mcpserver.MCPServer`). From a clone of the repo, `uv sync --extra mcp`, then, for Claude Code:

```bash
claude mcp add drishyam -- uv run --directory /abs/path/to/drishyam drishyam mcp
```

or for the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv run --directory /abs/path/to/drishyam drishyam mcp
```

## Tools (a fixed allow-list of twelve)

| Tool | What it does |
| --- | --- |
| `open_image(path)` | Starts a job on a JPEG/PNG under an allowed image root; people are blurred first. Returns the job id and caps. |
| `overview(job)` | At most 20 ranked candidate boxes (L0 pixels) and a saliency heatmap PNG. |
| `gaze(job, x, y, level, yaw=0, pitch=0, why="")` | Virtual-PTZ look at an L0 pixel at pyramid level 0-3. Returns a gaze id, window, quality gate and a JPEG of the crop (at most 512 px). |
| `enhance(job, gaze_id, why="")` | CLAHE re-look. Replaces that gaze's crop; segment again afterwards. |
| `segment(job, gaze_id, model)` | Per-class areas, confidence, entropy, latency. `model` is `accurate` or `fast`. Crack components that are one straight line (formwork, joints) are dropped and listed under `rejected`, as in the agent's step records. |
| `measure(job, gaze_id, cls)` | Length, width, area, orientation of one class, in L0 pixels. |
| `verify(job, gaze_id, why="")` | Looks one level finer at the same centre, segments, returns the mask IoU against the coarser look (below 0.5 is disagreement). |
| `runtime_stats(job)` | p50/p95 per op, CPU %, $ so far, budget against caps, what the governor changed. |
| `set_budget(job, pixel_cap, usd_cap, seconds_cap, why="")` | Lowers caps (0.02-1.0, 0.0001-0.05, 1-30). Raising one is refused. |
| `log_finding(job, gaze_id, cls, why="")` | Records a class that the last segmentation shows. Critical ones become `pending_review`. |
| `request_human_review(job, finding_id, reason)` | Records the request in the trace. Changes no state. |
| `draft_work_order(job)` | Writes `runs/<job>/work_order.json` and `crops/` and returns the order. |

`why` (optional, at most 200 printable characters) is the client's reason for the call. It is stored as the
trace record's `rationale`, and the server's instructions ask the client to cite the measured value behind it
(say `entropy 0.71 > 0.5 on g2 at L2`), so the trace shows which measurement each call answered.

There is no shell, no file-write and no approve/reject tool, and no OCR. `overview`, `gaze` and `enhance`
return images (heatmap, crop) so a vision-capable client can look; text painted on a bridge is therefore
visible to that client and must be treated as untrusted data. Whatever it reads, the model can only call
the tools above, none of which approves, writes files or runs commands. `open_image` resolves the real path and
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
failures are `error` records. `rationale` is the client's `why`; when the governor acted before a call (it changed a
setting, or a cap was hit), the record carries `governor: {cause, rationale}` and the reply `governor: {cause, note}`. The work order's `not_assessed` lists every gaze
whose quality gate failed and that no `enhance` repaired. `drishyam replay runs/<job>/trace.jsonl` prints the gaze path and checks it
against `work_order.json`. Calls rejected by schema validation (a level of 9, say) never reach the server code,
so they leave no line.

## The optional LLM planner (`drishyam run` and the web API)

Over MCP the client model is the planner. `drishyam run` and the web API instead run the rule planner and can
consult an LLM every fifth step through an allow-list validator with a 2 s timeout (`drishyam/planner.py`;
threat model in [security.md](security.md)). It is off unless the server environment sets:

| Variable | Meaning |
| --- | --- |
| `DRISHYAM_PLANNER_URL` | An OpenAI-compatible base URL (`.../v1`): `https://`, or plain `http://` to `127.0.0.1`, `::1` or `localhost` |
| `DRISHYAM_PLANNER_MODEL` | Model name sent to it (default `qwen2.5-3b-instruct`) |
| `DRISHYAM_PLANNER_KEY` | Bearer token, if the endpoint needs one |

Each consultation is a `planner` event, and the step records `planner_status`. No LLM planner runs in the live
demo: what it shows is the planner interface and the failure drills (`planner_timeout`, `planner_malformed`),
whose planners are scripted to time out or answer junk.

## Recording a session

The command that records Claude Code driving these tools on a sample photo, from the repository root after
`uv sync --extra mcp` ([mcp-session/mcp.json](mcp-session/mcp.json) starts `uv run drishyam mcp`;
[mcp-session/prompt.txt](mcp-session/prompt.txt) asks for the coarse-to-fine flow, a `why` that cites the
measured value on every call, and `request_human_review` for each critical finding and each disagreement):

```bash
mkdir -p runs
claude -p "$(cat docs/mcp-session/prompt.txt)" --mcp-config docs/mcp-session/mcp.json --strict-mcp-config --tools "" --allowedTools mcp__drishyam --output-format stream-json --verbose > runs/mcp-session.jsonl
```

`--tools ""` leaves the model no built-in tool (no shell, no file access), only the twelve above. The job's own
trace lands in `runs/<job>/trace.jsonl`.

**Recorded session.** [mcp-session/transcript.md](mcp-session/transcript.md) is a real session on
`bridge_03.jpg`: Claude chose each of 23 calls after reading the previous reply and looking at the crops, and
two `verify` IoUs (0.00, then 0.72) decided whether a finding went to a human or through. `claude -p` was not
available on the recording machine, so it ran through [mcp-session/bridge.py](mcp-session/bridge.py), a minimal
stdio MCP client that only forwards calls.

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
