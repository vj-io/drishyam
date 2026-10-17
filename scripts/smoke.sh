#!/usr/bin/env bash
# Smoke test of a deployed Drishyam (docs/aws.md, "Smoke"). Needs curl, jq and shasum; stores nothing.
#
#   scripts/smoke.sh https://dxxxxxxxxxxxx.cloudfront.net
#   FUNCTION_URL=https://xxxx.lambda-url.us-east-1.on.aws scripts/smoke.sh <url>    adds: raw Function URL is refused
#   WAF_LOOP=1 scripts/smoke.sh <url>    adds: job starts from this IP are blocked after about 12 in 60 s
#                                        (that IP cannot start jobs for about a minute after)
#
# Checks, one line each: /api/health; a job on public/samples/bridge_03.jpg reaches its done event; a KILL
# sent once step 2 has streamed halts the job with cause kill-switch; behind CloudFront, a POST without
# x-amz-content-sha256 is refused with 403 (Origin Access Control). Exits 1 if any check fails.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${1:?usage: scripts/smoke.sh <base url>}
URL=${URL%/}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
FAILED=0
pass() { echo "PASS $1: $2"; }
fail() { echo "FAIL $1: $2"; FAILED=1; }
sha() { shasum -a 256 | cut -d' ' -f1; }
post() {  # post <path> <content-type> <body file> [curl args...]: body hash header included, as OAC requires
  local path=$1 type=$2 body=$3; shift 3
  curl -sS -H "Content-Type: $type" -H "x-amz-content-sha256: $(sha <"$body")" --data-binary @"$body" "$@" "$URL$path"
}

health=$(curl -sS --max-time 60 -D "$TMP/headers" -w '\n%{time_starttransfer}' "$URL/api/health") ||
  { fail health "no answer from $URL"; exit 1; }
ttfb=${health##*$'\n'}
health=${health%$'\n'*}
if jq -e '.ok == true' <<<"$health" >/dev/null 2>&1; then
  pass health "$(jq -c '{version, opencv, build, segmenter, host, machine, commit}' <<<"$health"), ttfb ${ttfb}s"
else
  fail health "$health"; exit 1
fi
CLOUDFRONT=$(grep -qi '^x-amz-cf-id:' "$TMP/headers" && echo 1 || echo 0)

IMG=public/samples/bridge_03.jpg
code=$(post "/api/jobs?pixel_cap=0.3" image/jpeg "$IMG" -N --max-time 90 -o "$TMP/job.ndjson" \
  -w '%{http_code} %{time_starttransfer} %{time_total}') || true
done_ev=$(grep '"type":"done"' "$TMP/job.ndjson" || true)
if [ "${code%% *}" = 200 ] && [ -n "$done_ev" ]; then
  read -r _ t1 tt <<<"$code"
  pass job "$(jq -r '"\(.status) cause=\(.cause) steps=\(.summary.steps) seconds=\(.summary.seconds)"' <<<"$done_ev"), ttfb ${t1}s, total ${tt}s"
else
  fail job "HTTP ${code%% *}: $(head -c 300 "$TMP/job.ndjson")"
fi

# KILL: start a paced job (pace_ms is ignored by a server without it), wait for step 2, send the kill token.
post "/api/jobs?pixel_cap=0.3&pace_ms=1000" image/jpeg "$IMG" -N --max-time 90 -o "$TMP/kill.ndjson" &
job_pid=$!
for _ in $(seq 600); do
  n=$(grep -c '"type":"step"' "$TMP/kill.ndjson" 2>/dev/null) || true
  [ "${n:-0}" -ge 2 ] && break
  kill -0 "$job_pid" 2>/dev/null || break
  sleep 0.1
done
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
  echo "SKIP oac-unsigned-post: $URL is not served by CloudFront"
fi

if [ -n "${FUNCTION_URL:-}" ]; then
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 "${FUNCTION_URL%/}/api/health") || true
  [ "$code" = 403 ] && pass function-url-direct 403 || fail function-url-direct "expected 403, got $code"
fi

if [ "${WAF_LOOP:-}" = 1 ]; then  # empty bodies: the app answers 400 at once, so no job runs; WAF still counts them
  : >"$TMP/empty"
  codes=""
  for i in $(seq 40); do
    c=$(post /api/jobs text/plain "$TMP/empty" -o /dev/null -w '%{http_code}' --max-time 30) || true
    codes="$codes $c"
    [ "$c" = 403 ] && break
    sleep 1.5
  done
  if [ "$c" = 403 ] && [ "$i" -gt 5 ]; then pass waf "blocked at request $i:$codes"
  else fail waf "codes:$codes"; fi
fi

echo "smoke $(date -u +%FT%TZ) $URL commit=$(jq -r '.commit // "unreported"' <<<"$health") result=$([ $FAILED = 0 ] && echo pass || echo FAIL)"
exit $FAILED
