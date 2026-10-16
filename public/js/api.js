// The only module that talks to the network. ?mock=1 swaps the server for fixtures/job.ndjson.
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
  ? Promise.resolve({ ok: true, opencv: '5.0.0', build: 'mock fixture, no server', segmenter: 'fixture' })
  : getJSON('/api/health'));

const b64url = (s) => btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

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

/** Async generator of job events. `replay` is a list of [action, x, y, level] to resume from. */
export async function* streamJob({ blob, params, replay, signal }) {
  if (MOCK) return yield* mockStream(replay?.length ?? 0, signal);
  const headers = { 'Content-Type': blob.type || 'image/jpeg' };
  if (replay?.length) headers['X-Drishyam-Replay'] = b64url(JSON.stringify(replay.slice(0, 200)));
  const res = await fetch(`/api/jobs?${new URLSearchParams(params)}`, { method: 'POST', headers, body: blob, signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(`${res.status}: ${body.error ?? 'the server refused the job'}`);
  }
  yield* ndjson(res.body);
}

const REVIEW_ERRORS = {
  400: 'The review request was malformed.',
  403: 'The signature is bad or expired. Run the job again.',
  409: 'This finding is no longer pending review.',
};

export async function postReview(body) {
  if (MOCK) return mockReview(body);
  const res = await fetch('/api/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`${res.status}: ${REVIEW_ERRORS[res.status] ?? data.error ?? 'review failed'}`);
  return data;
}

/** Tell the server to halt this job before its next look (the aborted stream alone cannot reach it). */
export async function postKill(job, killToken) {
  if (MOCK || !job || !killToken) return;
  try {
    await fetch('/api/kill', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job, kill_token: killToken }), keepalive: true });
  } catch { /* the stream is already aborted client-side; the server stops at its cap regardless */ }
}

// ---- mock mode -------------------------------------------------------------
let mockOrder = null;
let mockTokens = 1;

async function* mockStream(replayCount, signal) {
  const res = await fetch('fixtures/job.ndjson', { signal });
  if (!res.ok) throw new Error('fixtures/job.ndjson not found');
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

async function mockReview({ finding_id, decision, reviewer }) {
  await sleep(350);
  const f = mockOrder?.findings.find((x) => x.id === finding_id);
  if (!f || f.state !== 'pending_review') throw new Error(`409: ${REVIEW_ERRORS[409]}`);
  f.state = decision === 'approve' ? 'approved' : 'rejected';
  mockOrder.approvals.push({ finding_id, decision, reviewer, at: new Date().toISOString().replace(/\.\d+Z$/, 'Z') });
  mockOrder.status = mockOrder.findings.some((x) => x.critical && x.state === 'pending_review') ? 'blocked' : 'ready';
  return { token: `mock.signed.work-order.${++mockTokens}`, work_order: structuredClone(mockOrder) };
}
