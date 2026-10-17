// Drishyam UI entry: wires the controls, runs a job (live or ?mock=1), and feeds every event to the panels.
import { $, $$, h, num } from './js/dom.js';
import { MOCK, streamJob, postReview, postKill, getJSON, health } from './js/api.js';
import { prepareImage } from './js/prep.js';
import { Viewport, pointOf } from './js/viewport.js';
import { Budget, Trace, renderStepDetail, isGovernor, plannerFell } from './js/panels.js';
import { Findings } from './js/findings.js';
import { countUp, popup, reduced, replay, sound } from './js/fx.js';
import { initResults } from './js/results.js';

const el = {
  pill: $('#job-pill'), text: $('#job-text'), count: $('#job-steps'), kill: $('#kill'), resume: $('#resume'), sound: $('#sound'),
  look: $('#look'), file: $('#file'), drop: $('#drop'), prep: $('#prep'), gallery: $('#gallery'), galleryEmpty: $('#gallery-empty'),
  slider: $('#replay'), sliderOut: $('#replay-out'), play: $('#play'), heat: $('#heat'), detail: $('#step-detail'), done: $('#done'),
  tbSheet: $('#tb-sheet'), tbScale: $('#tb-scale'), tbChecked: $('#tb-checked'),
};

const S = { image: null, ctl: null, mode: 'idle', job: null, overview: null, steps: [], done: null, token: null, order: null, ui: null, timer: 0, rev: 0 };

const viewport = new Viewport($('#stage'), (k) => goto(k));
const budget = new Budget($('.win--budget'));
const trace = new Trace($('#trace'), $('#trace-empty'), (n) => goto(S.steps.findIndex((s) => s.step === n) + 1));
const findings = new Findings($('#findings'), $('#findings-empty'), review);

const GOVERNOR_TEXT = {
  'p95>SLA': (s) => `p95(segment) > SLA → switched to the ${s.model ?? 'fast'} model`,
  'warp-spike': () => 'warp p95 spike → shrinking the gaze window',
  'usd≥80%': () => '$ at 80% of the cap → verifying critical candidates only',
  'cap-hit': () => 'a cap was hit → halting, coverage reported',
  'kill-switch': () => 'kill switch → halted, state preserved',
  'planner-fallback': () => 'planner failed → the rule planner took over',
};

// ---- taskbar / mode ---------------------------------------------------------
const displayStatus = () => (S.done?.status === 'awaiting_review' && S.order?.status === 'ready' ? 'completed' : S.done?.status);

function setMode(mode, text) {
  const onKill = document.activeElement === el.kill;  // hiding KILL would drop keyboard focus on the page body
  S.mode = mode;
  $('#taskbar').dataset.mode = mode;
  const label = { idle: 'ready', running: 'looking', killed: 'killed', error: 'error', done: displayStatus() }[mode];
  el.pill.className = `pill pill--${label}`;
  el.pill.textContent = label.replace('_', ' ');
  el.text.textContent = text ?? ({
    idle: MOCK ? 'Mock mode: a recorded job streams from fixtures. Hit LOOK.' : 'Pick a photo, set a budget, hit LOOK.',
    running: 'Looking. Every move is chosen from what OpenCV just measured.',
  }[mode] ?? '');
  el.kill.hidden = mode !== 'running';
  el.resume.hidden = !(mode === 'killed' || (mode === 'error' && S.steps.length > 0));
  el.look.disabled = mode === 'running' || !S.image;
  if (mode !== 'running') el.count.textContent = '';
  if (onKill && mode !== 'running') (el.resume.hidden ? el.look : el.resume).focus({ preventScroll: true });
}

// ---- stepping / replay --------------------------------------------------------
function goto(k, animate = false) {
  k = Math.max(0, Math.min(k, S.steps.length));
  const step = S.steps[k - 1];
  viewport.setCursor(k, animate);
  el.slider.value = k;
  el.sliderOut.textContent = `step ${k} / ${S.steps.length}`;
  trace.mark(step?.step);
  el.tbSheet.textContent = `${k} of ${S.steps.length}`;
  const level = S.steps.slice(0, k).reverse().find((s) => num(s.decision?.args?.level))?.decision.args.level;  // a halt keeps the last look's scale
  el.tbScale.textContent = num(level) ? `L${level} · 1:${2 ** level}` : 'overview';
  renderStepDetail(el.detail, step);
  budget.show(step
    ? { pixels: step.pixels_frac ?? 0, usd: step.usd_used ?? 0, seconds: step.seconds_used ?? 0 }
    : { pixels: S.overview?.pixels_frac ?? 0, usd: 0, seconds: 0 });
  budget.setOps(step);
}

function stopPlay() {
  clearInterval(S.timer);
  S.timer = 0;
  el.play.textContent = 'Play';
}

function togglePlay() {
  if (S.timer) return stopPlay();
  if (viewport.cursor >= S.steps.length) goto(0);
  el.play.textContent = 'Pause';
  S.timer = setInterval(() => (viewport.cursor >= S.steps.length ? stopPlay() : goto(viewport.cursor + 1, true)), 800);
}

function clearJob() {
  stopPlay();
  Object.assign(S, { job: null, overview: null, done: null, token: null, order: null, ui: null, rev: 0 });
  el.tbChecked.textContent = 'human reviewer';
  viewport.reset(null);
  S.steps = viewport.steps;
  trace.clear();
  findings.clear();
  el.slider.max = 0;
  el.slider.disabled = el.play.disabled = true;
  el.done.replaceChildren(h('p', { class: 'note' }, 'No job has finished yet. Status, coverage and the work order download appear here.'));
  goto(0);
}

// ---- stream events ------------------------------------------------------------
function handle(ev) {
  switch (ev.type) {
    case 'job':
      clearJob();
      S.job = ev;
      viewport.reset(ev);
      S.steps = viewport.steps;
      budget.setCaps(ev.budget);
      goto(0);
      break;
    case 'overview':
      S.overview = ev;
      viewport.setOverview(ev);
      goto(0);
      break;
    case 'step':
      addStep(ev);
      break;
    case 'finding':
      findings.add(ev.finding);
      break;
    case 'planner':
      trace.addPlanner(ev);
      break;
    case 'done':
      finish(ev);
      break;
    case 'error':
      setMode('error', ev.message ?? 'The job failed.');
      break;
    default:
  }
}

function addStep(step) {
  const follow = viewport.cursor === S.steps.length;
  S.steps.push(step);
  trace.addStep(step);
  el.slider.max = S.steps.length;
  el.slider.disabled = el.play.disabled = false;
  el.count.textContent = `step ${step.step}`;
  if (follow) goto(S.steps.length, true);
  else el.sliderOut.textContent = `step ${viewport.cursor} / ${S.steps.length}`;
  // a halt record may carry the governor's last override; a failed planner consultation is in planner_status
  for (const cause of [step.governor?.cause, step.cause, plannerFell(step) && 'planner-fallback']) {
    if (isGovernor(cause)) popup(++S.rev, cause, GOVERNOR_TEXT[cause](step));
  }
  sound.scratch();
}

// ---- work order / done state --------------------------------------------------
function tile(label) {
  const v = h('b', { class: 'tile__v' }, '0');
  return [h('div', { class: 'tile' }, v, h('span', { class: 'tile__k' }, label)), v];
}

function buildDone() {
  const { summary: sm, coverage, unvisited, cause } = S.done;
  const pill = h('span', { class: 'pill' });
  const orderState = h('span', { class: 'note' });
  const specs = [
    ['looks', sm.steps, {}], ['of the pixels', sm.pixels_frac * 100, { decimals: 1, suffix: ' %' }],
    ['$ modelled', sm.usd, { decimals: 4, prefix: '$' }], ['seconds', sm.seconds, { decimals: 1 }],
    ['findings', sm.findings, {}], ['critical pending', sm.critical_pending, {}],
  ].map(([label, value, opts]) => ({ value, opts, pair: tile(label) }));
  const cover = h('div', { class: 'meter__fill' });
  const coverNum = h('b', { class: 'meter__val' }, '0 %');
  const privacy = S.job?.privacy;
  const download = h('button', { class: 'btn', type: 'button' }, 'Download work_order.json');
  download.addEventListener('click', downloadOrder);
  el.done.replaceChildren(h('div', { class: 'stack' },
    h('div', { class: 'done__head' }, pill, h('span', { class: 'note' }, `stopped by ${cause}`)),
    orderState,
    h('div', { class: 'tiles' }, specs.map((s) => s.pair[0])),
    h('div', { class: 'meter' },
      h('div', { class: 'meter__head' }, h('span', {}, 'saliency coverage'), h('span', {}, coverNum)),
      h('div', { class: 'meter__track', role: 'progressbar', 'aria-label': 'Saliency-weighted coverage', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': Math.round(coverage * 100) }, cover)),
    h('div', {}, h('h4', { class: 'label' }, 'unvisited regions'),
      unvisited?.length ? h('p', { class: 'chips' }, unvisited.map((r) => h('span', { class: 'chip chip--look' }, r))) : h('p', { class: 'note' }, 'none: every candidate was visited')),
    privacy ? h('p', { class: 'note' }, `privacy: ${privacy.faces_blurred ?? 0} faces and ${privacy.people_blurred ?? 0} people blurred`) : null,
    download));
  S.ui = { pill, orderState, pending: specs[5].pair[1] };
  specs.forEach((s) => countUp(s.pair[1], s.value, s.opts));
  requestAnimationFrame(() => { cover.style.width = `${coverage * 100}%`; });
  countUp(coverNum, coverage * 100, { decimals: 0, suffix: ' %' });
  updateDone();
}

function updateDone() {
  const pending = S.order.findings.filter((f) => f.critical && f.state === 'pending_review').length;
  const status = displayStatus();
  const label = status.replace('_', ' ');
  S.ui.pill.className = `pill pill--stamp pill--${status}`;
  if (S.ui.pill.textContent !== label) {
    S.ui.pill.textContent = label;
    replay(S.ui.pill, 'is-stamped');
  }
  const signers = [...new Set((S.order.approvals ?? []).map((a) => a.reviewer).filter(Boolean))];
  el.tbChecked.textContent = signers.length ? signers.join(', ') : 'human reviewer';
  S.ui.orderState.textContent = `work order ${S.order.id} is ${S.order.status}${pending ? `: ${pending} critical finding(s) wait for a human` : ''}`;
  countUp(S.ui.pending, pending);
  setMode('done', el.text.textContent);
}

function downloadOrder() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(S.order, null, 2)], { type: 'application/json' }));
  const a = h('a', { href: url, download: 'work_order.json' });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function finish(ev) {
  S.done = ev;
  S.token = ev.token;
  S.order = ev.work_order;
  const sm = ev.summary;
  setMode('done', `${sm.steps} looks · ${(sm.pixels_frac * 100).toFixed(1)} % of pixels · ${sm.seconds} s · saliency coverage ${Math.round(ev.coverage * 100)} %`);
  findings.setReady();
  buildDone();
  replay(el.tbSheet, 'is-stamped');  // the sheet number stamps in: this run is on file
  sound.stamp();
}

async function review(id, decision, reviewer) {
  const res = await postReview({ token: S.token, finding_id: id, decision, reviewer });
  S.token = res.token;
  S.order = res.work_order;
  findings.sync(S.order);
  updateDone();
}

// ---- running a job -------------------------------------------------------------
function params() {
  const p = {
    pixel_cap: +(Number($('#cap-px').value) / 100).toFixed(2),
    usd_cap: $('input[name=usd]:checked').value,
    seconds_cap: $('#cap-sec').value,
  };
  const chaos = $$('input[name=chaos]:checked').map((i) => i.value).join(',');
  if (chaos) p.chaos = chaos;
  return p;
}

async function run(replay = null) {
  if (!S.image || S.mode === 'running') return;
  S.ctl = new AbortController();
  const refocus = [el.look, el.resume].includes(document.activeElement);  // both are disabled or hidden while running
  setMode('running');
  if (refocus) el.kill.focus({ preventScroll: true });
  if (matchMedia('(max-width: 960px)').matches) $('.win--view').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
  try {
    for await (const ev of streamJob({ blob: S.image.blob, params: params(), replay, signal: S.ctl.signal })) handle(ev);
    if (S.mode === 'running') setMode('killed', `The stream ended at step ${S.steps.length} before the job finished. Resume to continue.`);
  } catch (err) {
    if (err.name !== 'AbortError') setMode('error', err.message);
  }
}

function kill() {
  postKill(S.job?.job, S.job?.kill_token);
  S.ctl.abort();
  setMode('killed', `Killed at step ${S.steps.length} — state preserved. Resume re-sends the photo and replays your ${S.steps.length} moves.`);
}

function resume() {
  const moves = S.steps.filter((s) => ['gaze', 'escalate', 'verify'].includes(s.decision?.action) && pointOf(s))
    .map((s) => [s.decision.action, s.decision.args.x, s.decision.args.y, s.decision.args.level, s.decision.args.size]
      .filter((v) => v !== undefined));
  run(moves.slice(0, 200));
}

// ---- image picking ---------------------------------------------------------------
async function useImage(blob, label) {
  if (S.mode === 'running') {
    el.prep.textContent = 'A job is running. KILL it first to change the photo.';
    return;
  }
  el.prep.textContent = 'Reading the photo…';
  try {
    const p = await prepareImage(blob);
    if (S.image) URL.revokeObjectURL(S.image.url);
    S.image = { blob: p.blob, url: URL.createObjectURL(blob) };
    clearJob();
    viewport.setImage(S.image.url);
    el.prep.textContent = [label, `${p.width}×${p.height}`, `${(p.blob.size / 1e6).toFixed(1)} MB`, p.note].filter(Boolean).join(' · ');
    setMode('idle', S.mode === 'idle' ? 'Photo ready. Check the budget, then hit LOOK.' : undefined);
  } catch {
    el.prep.textContent = 'Could not read that image. Use a JPEG or PNG.';
  }
}

function pickFile(file) {
  if (!file) return;
  if (!/^image\/(jpeg|png)$/.test(file.type)) {
    el.prep.textContent = 'Only JPEG or PNG files, please.';
    return;
  }
  useImage(file, file.name);
}

async function loadSamples() {
  const list = await getJSON('samples/index.json');
  const items = (Array.isArray(list) ? list : []).filter((s) => s && /^[\w.-]+$/.test(s.file ?? ''));
  el.galleryEmpty.hidden = items.length > 0;
  el.gallery.replaceChildren(...items.map((s) => {
    const thumb = /^\w[\w./-]*$/.test(s.thumb ?? '') && !s.thumb.includes('..') ? s.thumb : s.file;
    const btn = h('button', { class: 'thumb', type: 'button', 'aria-pressed': 'false' },
      h('img', { src: `samples/${thumb}`, alt: '', loading: 'lazy', width: 96, height: 72 }),
      h('span', { class: 'thumb__t' }, String(s.title ?? s.file)),
      s.note ? h('span', { class: 'thumb__n' }, String(s.note)) : null,
      s.synthetic ? h('span', { class: 'tag' }, 'synthetic') : null);
    btn.addEventListener('click', async () => {
      if (S.mode === 'running') { el.prep.textContent = 'A job is running. KILL it first to change the photo.'; return; }
      el.look.disabled = true;  // the old photo must not run while the new one loads
      el.prep.textContent = `Loading ${String(s.title ?? s.file)}…`;
      const res = await fetch(`samples/${s.file}`).catch(() => null);
      if (!res?.ok) { el.prep.textContent = 'That sample could not be loaded.'; el.look.disabled = !S.image; return; }
      for (const b of $$('.thumb', el.gallery)) b.setAttribute('aria-pressed', String(b === btn));
      useImage(await res.blob(), String(s.title ?? s.file));
    });
    return h('li', {}, btn);
  }));
}

// ---- wiring ------------------------------------------------------------------------
el.look.addEventListener('click', () => run());
el.kill.addEventListener('click', kill);
el.resume.addEventListener('click', resume);
el.play.addEventListener('click', togglePlay);
el.heat.addEventListener('change', () => viewport.setHeat(el.heat.checked));
el.slider.addEventListener('input', () => {
  stopPlay();
  goto(Number(el.slider.value));
});
el.sound.addEventListener('click', () => {
  const on = sound.toggle();
  el.sound.setAttribute('aria-pressed', String(on));
  el.sound.textContent = `Sound: ${on ? 'on' : 'off'}`;
  if (on) sound.click();
});
el.file.addEventListener('change', () => {
  pickFile(el.file.files[0]);
  el.file.value = '';
});
el.drop.addEventListener('dragover', (e) => { e.preventDefault(); el.drop.classList.add('is-over'); });
el.drop.addEventListener('dragleave', () => el.drop.classList.remove('is-over'));
el.drop.addEventListener('drop', (e) => { e.preventDefault(); el.drop.classList.remove('is-over'); pickFile(e.dataTransfer.files[0]); });
for (const type of ['dragover', 'drop']) window.addEventListener(type, (e) => e.preventDefault());
$('#cap-px').addEventListener('input', (e) => { $('#cap-px-out').textContent = `${e.target.value} %`; });
$('#cap-sec').addEventListener('input', (e) => { $('#cap-sec-out').textContent = `${e.target.value} s`; });

// Phones see only the photo and its numbered balloons (the parts list below carries the callout text).
const artMq = matchMedia('(max-width: 560px)');
const fitArt = () => $('.art__svg').setAttribute('viewBox', artMq.matches ? '0 0 492 372' : '0 0 600 430');
artMq.onchange = fitArt;
fitArt();

// The sticky bar's real height: scroll padding and focus scrolling keep controls clear of it.
new ResizeObserver(() => document.documentElement.style.setProperty('--bar', `${$('#taskbar').offsetHeight}px`)).observe($('#taskbar'));

const pops = new IntersectionObserver((entries) => {
  let i = 0;
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.style.setProperty('--d', `${i++ * 90}ms`);
    e.target.classList.add('is-in');
    pops.unobserve(e.target);
  }
}, { threshold: 0.06 });
$$('.pop').forEach((w) => pops.observe(w));
document.documentElement.classList.add('js');  // only now may CSS hide not-yet-scrolled windows (a failed script leaves them visible)

clearJob();
setMode('idle');
loadSamples();
initResults();
health().then((hl) => {
  $('#api-status').textContent = hl?.ok
    ? `API online · OpenCV ${hl.opencv} · ${hl.build} · segmenter ${hl.segmenter}`
    : 'The API is not reachable from this page. Add ?mock=1 to the address to watch a recorded job.';
});
if (MOCK) {
  fetch('fixtures/mock.jpg').then((r) => r.blob()).then((b) => useImage(b, 'mock.jpg (synthetic)'));
}
