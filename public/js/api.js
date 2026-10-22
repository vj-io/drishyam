// The only module that talks to the network. ?mock=1 swaps the server for fixtures/job.ndjson; a `trace` replays a saved job.
export const MOCK = new URLSearchParams(location.search).get('mock') === '1';

const abortError = () => new DOMException('Aborted', 'AbortError');
const sleep = (ms, signal) => new Promise((res, rej) => {
  const t = setTimeout(res, ms);
  signal?.addEventListener('abort', () => { clearTimeout(t); rej(abortError()); }, { once: true });
});

/** Parsed JSON, or null for any 404 / network / parse failure. */
export async function getJSON(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

export const health = () => (MOCK
  ? Promise.resolve({ ok: true })  // the page says it is a recorded run; no API is called
  : getJSON('/api/health'));

const b64url = (s) => btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

/** Hex SHA-256 of the exact body bytes: CloudFront OAC in front of a Lambda Function URL requires it on every POST. */
async function sha256(body) {
  const bytes = typeof body === 'string' ? new TextEncoder().encode(body) : await body.arrayBuffer();
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((b) => b.toString(16).padStart(2, '0')).join('');
}

const post = async (url, body, headers, extra = {}) => fetch(url, {
  method: 'POST', body, ...extra, headers: { ...headers, 'x-amz-content-sha256': await sha256(body) },
});

async function* ndjson(body) {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) yield JSON.parse(line);
    }
  }
  if ((buf += dec.decode()).trim()) yield JSON.parse(buf);
}

// A 429 is a WAF per-IP rate rule (infra/drishyam_stack.py; its body's error starts "rate limited"), or Lambda refusing
// because every environment is busy or the demo is stopped (any other body): docs/api-contract.md.
const BUSY = 'The demo is busy: no server is free right now. Wait a few seconds and retry.';
const JOB_LIMIT = 'Too many jobs from this address: 10 starts per 2 minutes. Wait a minute and retry.';
const throttled = (body, perIp) => (String(body?.error).startsWith('rate limited') ? perIp : BUSY);

/** Async generator of job events. `replay` is a list of [action, x, y, level] to resume from; `params` may carry pace_ms;
    `trace` (a same-origin .ndjson path) replays that recorded job instead of asking the server. */
export async function* streamJob({ blob, params, replay, signal, trace }) {
  if (MOCK || trace) return yield* mockStream(trace ?? 'fixtures/job.ndjson', replay?.length ?? 0, signal);
  const headers = { 'Content-Type': blob.type || 'image/jpeg' };
  if (replay?.length) headers['X-Drishyam-Replay'] = b64url(JSON.stringify(replay.slice(0, 200)));
  const res = await post(`/api/jobs?${new URLSearchParams(params)}`, blob, headers, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw Object.assign(new Error(res.status === 429 ? throttled(body, JOB_LIMIT) : `${res.status}: ${body.error ?? 'the server refused the job'}`),
      { status: res.status });
  }
  // This job now holds one Lambda environment until it ends, so KILL lands in another. Start that one now: a cold start
  // (0.86 to 6.27 s to the first byte over the v1 smoke runs) after KILL can let a short job finish before the kill
  // flag is read (docs/aws.md, Smoke, check 3).
  fetch('/api/health', { cache: 'no-store' }).catch(() => {});
  yield* ndjson(res.body);
}

const REVIEW_ERRORS = {
  400: 'The review request was malformed.',
  403: 'The signature is bad or expired. Run the job again.',
  409: 'This finding is no longer pending review.',
  429: 'Too many requests from this address. Wait a minute and retry.',
};
const CHECKED = 'This area was already checked by eye.';  // a 409 on a check by eye

/** A decision ({finding_id, decision: approve|reject}) or a check by eye ({not_assessed, decision: nothing-found|
    defect-found, note}). A `recorded` job (a saved trace) was signed by another server, so the page settles it itself. */
export async function postReview(body, recorded = false) {
  if (MOCK || recorded) return mockReview(body);
  const res = await post('/api/review', JSON.stringify(body), { 'Content-Type': 'application/json' });
  const data = await res.json().catch(() => ({}));
  const said = body.not_assessed != null && res.status === 409 ? CHECKED : REVIEW_ERRORS[res.status];
  if (!res.ok) throw new Error(`${res.status}: ${res.status === 429 ? throttled(data, REVIEW_ERRORS[429]) : said ?? data.error ?? 'review failed'}`);
  return data;
}

/** Tell the server to halt this job before its next look (the aborted stream alone cannot reach it). */
export async function postKill(job, killToken) {
  if (MOCK || !job || !killToken) return;
  try {
    await post('/api/kill', JSON.stringify({ job, kill_token: killToken }), { 'Content-Type': 'application/json' }, { keepalive: true });
  } catch { /* the stream is already aborted client-side; the server stops at its cap regardless */ }
}

// ---- mock mode -------------------------------------------------------------
let mockOrder = null;
let mockTokens = 1;

async function* mockStream(url, replayCount, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url} not found`);
  for (const line of (await res.text()).split('\n')) {
    if (!line.trim()) continue;
    let ev = JSON.parse(line);
    const replayed = ev.type === 'step' && ev.step <= replayCount;
    if (replayed) ev = { ...ev, cause: 'replay' };
    await sleep(replayed ? 40 : 250, signal);
    if (ev.type === 'done') mockOrder = structuredClone(ev.work_order);
    yield ev;
  }
}

async function mockReview({ finding_id, not_assessed: index, decision, reviewer, note = '' }) {
  await sleep(350);
  const at = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
  if (index != null) {  // checked by eye, as drishyam/findings.py resolve_entry records it
    const a = mockOrder?.not_assessed?.[index];
    if (!a || a.resolution) throw new Error(`409: ${CHECKED}`);
    a.resolution = { decision, reviewer, at, note };
    mockOrder.approvals.push({ not_assessed: index, decision, reviewer, at });
  } else {
    const f = mockOrder?.findings.find((x) => x.id === finding_id);
    if (!f || f.state !== 'pending_review') throw new Error(`409: ${REVIEW_ERRORS[409]}`);
    f.state = decision === 'approve' ? 'approved' : 'rejected';
    mockOrder.approvals.push({ finding_id, decision, reviewer, at });
  }
  mockOrder.status = mockOrder.findings.some((x) => x.critical && x.state === 'pending_review') ? 'blocked'  // order_status
    : (mockOrder.not_assessed ?? []).some((a) => !a.resolution) ? 'incomplete' : 'ready';
  return { token: `mock.signed.work-order.${++mockTokens}`, work_order: structuredClone(mockOrder) };
}
