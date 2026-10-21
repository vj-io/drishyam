#!/usr/bin/env bash
# Smoke test of a deployed Drishyam (docs/aws.md, "Smoke"). Needs curl, jq and shasum.
#
#   scripts/smoke.sh https://dxxxxxxxxxxxx.cloudfront.net
#   FUNCTION_URL=https://xxxx.lambda-url.us-east-1.on.aws scripts/smoke.sh <url>    adds: raw Function URL is refused
#   E2E=1 scripts/smoke.sh <url>         adds: tests/e2e/test_ui_live.py against <url> (Playwright in .venv, see README)
#   WAF_LOOP=1 scripts/smoke.sh <url>    adds: job starts from this IP are blocked after about 10 in 120 s
#                                        (that IP cannot start jobs for up to two minutes after)
#   SMOKE_LOG=runs/smoke.log scripts/smoke.sh <url>    appends every result line and the summary line to the file
#   CODE_SHA256=<digest>                 put in the summary as the package digest (scripts/aws.sh smoke sets it)
#
# Checks, one line each: /api/health twice (time to first byte of the first request, cold if the function
# was idle or just deployed, and of the warm second); a job on public/samples/bridge_03.jpg reaches its
# done event, and the p95 of its steps' latency_ms; a KILL sent once step 2 has streamed halts the job with
# cause kill-switch; behind CloudFront, a POST without x-amz-content-sha256 is refused with 403 (Origin
# Access Control). The last line sums it up with timestamp, commit and package digest. Exits 1 if any
# check fails.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${1:?usage: scripts/smoke.sh <base url>}
URL=${URL%/}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
FAILED=0 LINES="" TTFB1=n/a TTFB2=n/a P95=n/a
say() { echo "$1"; LINES+="$1"$'\n'; }
pass() { say "PASS $1: $2"; }
fail() { say "FAIL $1: $2"; FAILED=1; }
finish() {
  local commit
  commit=$(jq -r '.commit // "unreported"' "$TMP/health.json" 2>/dev/null) || commit=unreported
  say "smoke $(date -u +%FT%TZ) $URL commit=$commit package=${CODE_SHA256:-unreported} ttfb_first=$TTFB1 ttfb_warm=$TTFB2 step_p95_ms=$P95 result=$([ $FAILED = 0 ] && echo pass || echo FAIL)"
  [ -z "${SMOKE_LOG:-}" ] || printf '%s' "$LINES" >>"$SMOKE_LOG"
  exit $FAILED
}
sha() { shasum -a 256 | cut -d' ' -f1; }
post() {  # post <path> <content-type> <body file> [curl args...]: body hash header included, as OAC requires
  local path=$1 type=$2 body=$3; shift 3
  curl -sS -H "Content-Type: $type" -H "x-amz-content-sha256: $(sha <"$body")" --data-binary @"$body" "$@" "$URL$path"
}

health() { curl -sS --max-time 60 -D "$TMP/headers" -o "$TMP/health.json" -w '%{time_starttransfer}s' "$URL/api/health"; }
TTFB1=$(health) || { TTFB1=n/a; fail health "no answer from $URL"; finish; }
TTFB2=$(health) || TTFB2=n/a
if jq -e '.ok == true' "$TMP/health.json" >/dev/null 2>&1; then
  pass health "$(jq -c '{version, opencv, build, segmenter, host, arch, commit}' "$TMP/health.json"), ttfb first $TTFB1 (cold if the function was idle or just deployed), warm $TTFB2"
else
  fail health "$(head -c 300 "$TMP/health.json")"; finish
fi
CLOUDFRONT=$(grep -qi '^x-amz-cf-id:' "$TMP/headers" && echo 1 || echo 0)

# A job holds its Lambda environment while it streams, so a request sent meanwhile starts another one. The browser
# does this at every job start (public/js/api.js), so a later KILL finds that environment warm. warm FILE: once FILE
# (a job's stream) is not empty, GET /api/health.
warm() { for _ in $(seq 600); do [ -s "$1" ] && break; sleep 0.1; done; curl -sS --max-time 60 -o /dev/null "$URL/api/health" || true; }

IMG=public/samples/bridge_03.jpg
warm "$TMP/job.ndjson" & warm_pid=$!
code=$(post "/api/jobs?pixel_cap=0.3" image/jpeg "$IMG" -N --max-time 90 -o "$TMP/job.ndjson" \
  -w '%{http_code} %{time_starttransfer} %{time_total}') || true
wait "$warm_pid" || true
done_ev=$(grep '"type":"done"' "$TMP/job.ndjson" || true)
if [ "${code%% *}" = 200 ] && [ -n "$done_ev" ]; then
  read -r _ t1 tt <<<"$code"
  pass job "$(jq -r '"\(.status) cause=\(.cause) steps=\(.summary.steps) seconds=\(.summary.seconds)"' <<<"$done_ev"), ttfb ${t1}s, total ${tt}s"
else
  fail job "HTTP ${code%% *}: $(head -c 300 "$TMP/job.ndjson")"
fi
# Per-step latency (warp plus segmentation, no pacing) on this host, nearest-rank p95 over the job's steps.
read -r n P95 < <(jq -rs '[.[] | select(.type == "step") | .latency_ms | numbers] | sort
  | "\(length) \(if length > 0 then .[(length * 0.95 | ceil) - 1] else "n/a" end)"' "$TMP/job.ndjson" 2>/dev/null) ||
  { n=0 P95=n/a; }
if [ "$n" -gt 0 ]; then
  say "INFO latency: p95 ${P95} ms per step over $n steps, $(jq -r '"\(.arch) OpenCV \(.opencv) \(.build)"' "$TMP/health.json")"
else
  say "SKIP latency: no step event carries latency_ms"
fi

# KILL: start a paced job (pace_ms is ignored by a server without it), wait for step 2, send the kill token.
post "/api/jobs?pixel_cap=0.3&pace_ms=1000" image/jpeg "$IMG" -N --max-time 90 -o "$TMP/kill.ndjson" &
job_pid=$!
warm "$TMP/kill.ndjson" & warm_pid=$!
for _ in $(seq 600); do
  n=$(grep -c '"type":"step"' "$TMP/kill.ndjson" 2>/dev/null) || true
  [ "${n:-0}" -ge 2 ] && break
  kill -0 "$job_pid" 2>/dev/null || break
  sleep 0.1
done
wait "$warm_pid" || true
head -n 1 "$TMP/kill.ndjson" | jq -c '{job, kill_token}' >"$TMP/kill.json" 2>/dev/null || echo '{}' >"$TMP/kill.json"
kill_code=$(post /api/kill application/json "$TMP/kill.json" -o /dev/null -w '%{http_code}' --max-time 30) || true
wait "$job_pid" || true
halted=$(grep '"type":"done"' "$TMP/kill.ndjson" | jq -r '"\(.status) cause=\(.cause) steps=\(.summary.steps)"' || true)
if [ "$kill_code" = 200 ] && [[ $halted == "halted cause=kill-switch"* ]]; then
  pass kill "$halted"
else
  fail kill "POST /api/kill -> $kill_code; job ended '${halted:-without a done event}'"
fi

if [ "$CLOUDFRONT" = 1 ]; then
  printf '{"job":"j_000000","kill_token":"AAAAAAAAAAAAAAAAAAAAAAAA"}' >"$TMP/k.json"
  code=$(curl -sS -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' --data-binary @"$TMP/k.json" \
    --max-time 30 "$URL/api/kill") || true
  [ "$code" = 403 ] && pass oac-unsigned-post 403 || fail oac-unsigned-post "expected 403, got $code"
else
  say "SKIP oac-unsigned-post: $URL is not served by CloudFront"
fi

if [ -n "${FUNCTION_URL:-}" ]; then
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 "${FUNCTION_URL%/}/api/health") || true
  [ "$code" = 403 ] && pass function-url-direct 403 || fail function-url-direct "expected 403, got $code"
fi

if [ "${E2E:-}" = 1 ]; then  # before the WAF loop, which would block its job starts
  if out=$(.venv/bin/python tests/e2e/test_ui_live.py "$URL" 2>&1); then
    pass e2e "tests/e2e/test_ui_live.py: $(grep -c '^ok' <<<"$out") browser flows ok"
  else
    fail e2e "tests/e2e/test_ui_live.py: $(tail -n 3 <<<"$out" | tr '\n' ' ')"
  fi
fi

if [ "${WAF_LOOP:-}" = 1 ]; then  # empty bodies: the app answers 400 at once, so no job runs; WAF still counts them
  : >"$TMP/empty"
  codes=""
  for i in $(seq 40); do
    c=$(post /api/jobs text/plain "$TMP/empty" -o "$TMP/waf.json" -w '%{http_code}' --max-time 30) || true
    codes="$codes $c"
    # The stack's WAF block response (an older deploy answered 403). Lambda's own throttle is also a 429,
    # so only the WAF's JSON body counts.
    waf=0; [ "$c" = 429 ] && jq -e '.error | startswith("rate limited")' "$TMP/waf.json" >/dev/null 2>&1 && waf=1
    [ $waf = 1 ] && break
    sleep 1.5
  done
  # The job starts above count too, so the block can come a few requests early; never on the first.
  if [ $waf = 1 ] && [ "$i" -gt 1 ]; then pass waf "blocked at request $i:$codes"
  else fail waf "codes:$codes"; fi
fi

finish
