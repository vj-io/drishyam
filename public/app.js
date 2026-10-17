// Drishyam UI entry: wires the controls, runs a job (live or ?mock=1) and presents its events at reading speed.
import { $, $$, h, num, dataUrl } from './js/dom.js';
import { MOCK, streamJob, postReview, postKill, getJSON, health } from './js/api.js';
import { prepareImage } from './js/prep.js';
import { Viewport, pointOf, tileBoxes } from './js/viewport.js';
import { Budget, Trace, renderStepDetail, isGovernor, plannerFell, explain, haltText, kindOf, pad, px } from './js/panels.js';
import { Findings, ruleOf } from './js/findings.js';
import { countUp, popup, reduced, replay, sound } from './js/fx.js';
import { initResults } from './js/results.js';

const PACE = 5000;  // ms each decision stays on screen: long enough to read its NOTE
const HOLD = PACE + 3000;  // the first critical finding's NOTE outlasts every decision: the hand-off to a person
const GUIDE = 'bridge_03.jpg';  // exposed rebar: the default run ends with critical findings for a human
const GUIDE_PACE_MS = 1000;  // asks the server to wait this long between looks (pace_ms); its timings exclude the wait
const NOTE = 'Press Run and the agent’s reasoning appears here, one decision at a time, in plain words.';

const el = {
  pill: $('#job-pill'), text: $('#job-text'), count: $('#job-steps'), kill: $('#kill'), resume: $('#resume'), sound: $('#sound'),
  look: $('#look'), runBar: $('#run-bar'), skip: $('#skip'), guided: $('#guided'), recorded: $('#recorded'),
  file: $('#file'), drop: $('#drop'), prep: $('#prep'), gallery: $('#gallery'), galleryEmpty: $('#gallery-empty'),
  slider: $('#replay'), sliderOut: $('#replay-out'), play: $('#play'), heat: $('#heat'), tiles: $('#tiles'), tilesNote: $('#tiles-note'),
  detail: $('#step-detail'), done: $('#done'), recap: $('#recap'), narrator: $('#narrator'), narratorSub: $('#narrator-sub'),
  camImg: $('#cam-img'), pip: $('#pip'), camCap: $('#cam-cap'), stages: $('#stages'), say1: $('#say-1'), say4: $('#say-4'),
  states: [1, 2, 3, 4].map((i) => $(`#state-${i}`)),
  tbSheet: $('#tb-sheet'), tbScale: $('#tb-scale'), tbChecked: $('#tb-checked'),
};

const S = {
  image: null, sample: null, ready: null, ctl: null, mode: 'idle', opts: {}, job: null, overview: null, steps: [], done: null,
  token: null, order: null, ui: null, timer: 0, rev: 0, events: [], planners: new Map(), arrived: null, note: null, critSeen: false, killTimer: 0,
};
// The presentation queue: stream events wait here and are shown one decision at a time.
const Q = { items: [], timer: 0, skip: false };

const viewport = new Viewport($('#stage'), (k) => goto(k));
const budget = new Budget($('#budget'));
const trace = new Trace($('#trace'), $('#trace-empty'), (n) => goto(S.steps.findIndex((s) => s.step === n) + 1));
const findings = new Findings($('#findings'), $('#findings-empty'), review, showFinding, levelsOf);

const GOVERNOR_TEXT = {
  'p95>SLA': (s) => `The segmenter got slow, so the governor switched to the ${s.model ?? 'fast'} model.`,
  'warp-spike': () => 'Cutting out views got slow, so the governor shrank the view.',
  'usd≥80%': () => '80 % of the $ cap is spent, so it now double-checks critical damage only.',
  'cap-hit': () => 'A budget cap was hit, so it stopped and reported what it did not reach.',
  'kill-switch': () => 'KILL: it halted before the next look; its state is kept.',
  'planner-fallback': () => 'The planner failed, so the built-in rule planner took over.',
};
const STOPPED = {
  'stop-rule': 'the next look was worth less than its cost', exhausted: 'every candidate region was looked at',
  'cap-hit': 'budget spent', 'kill-switch': 'you pressed KILL', 'planner:stop': 'the planner proposed stopping',
};

// ---- the four steps: Waiting / Running / Done, each with a sentence in plain words -------------------
const pendingCount = () => (S.order?.findings ?? []).filter((f) => f.critical && f.state === 'pending_review').length;

function syncSteps() {
  const run = S.mode === 'running' ? 'running' : S.done ? 'done' : 'waiting';
  const pending = pendingCount();
  const states = [S.image ? 'done' : 'waiting', run, run, S.done && !pending ? 'done' : 'waiting'];
  states.forEach((state, i) => {
    el.states[i].dataset.state = state;
    el.states[i].textContent = { waiting: 'Waiting', running: 'Running', done: 'Done' }[state];
  });
  const critical = (S.order?.findings ?? []).some((f) => f.critical);
  el.say4.textContent = !S.done
    ? (S.mode === 'running' ? 'Findings appear here as the agent makes them. Anything critical waits for your sign-off when the run ends.'
      : 'Nothing to sign yet. When a run ends, anything critical waits here for a person. The agent cannot approve it.')
    : pending ? `Waiting for your sign-off (${pending}): type your name on each card marked NEEDS A HUMAN, then Approve or Reject. The agent cannot do this for you.`
      : critical ? 'Signed: ready to send. Download the signed work order below.'
        : 'Nothing critical was found, so nothing needs your sign-off. The work order is ready.';
}

// ---- taskbar / mode -----------------------------------------------------------------------------------
function doneKey() {
  const d = S.done;
  if (d?.status === 'halted') return d.cause === 'cap-hit' ? 'budget' : 'halted';
  if (d?.status === 'awaiting_review' && S.order?.status === 'ready') return 'signed';
  return d?.status ?? 'completed';
}
const PILL = {
  idle: 'ready', running: 'running', killed: 'killed', error: 'error',
  awaiting_review: 'needs sign-off', signed: 'signed', completed: 'done', halted: 'halted', budget: 'budget spent',
};

function setMode(mode, text) {
  const onBar = [el.kill, el.skip].includes(document.activeElement);  // hiding them would drop keyboard focus on the page body
  S.mode = mode;
  $('#taskbar').dataset.mode = mode;
  const key = mode === 'done' ? doneKey() : mode;
  el.pill.className = `pill pill--${key}`;
  el.pill.textContent = PILL[key] ?? key;
  el.text.textContent = text ?? ({
    idle: MOCK ? 'Recorded run: a saved job replays from fixtures, no server. Press Run.'
      : S.image ? 'Photo ready. Press Run, or Watch it inspect a bridge for a guided run.' : 'Pick a photo, then press Run.',
    running: 'Sending the photo to the agent…',
  }[mode] ?? '');
  const running = mode === 'running';
  el.kill.hidden = !running;
  el.skip.hidden = !running || Q.skip;
  el.runBar.hidden = running;
  el.resume.hidden = !(mode === 'killed' || (mode === 'error' && S.steps.length > 0) || (mode === 'done' && S.done?.cause === 'kill-switch'));
  el.look.disabled = el.runBar.disabled = running || !S.image;
  el.guided.disabled = running;
  if (!running) el.count.textContent = '';
  if (onBar && !running) (el.resume.hidden ? el.look : el.resume).focus({ preventScroll: true });
  syncSteps();
}

function narrate(text, sub = '') {
  el.narrator.textContent = text;
  el.narratorSub.textContent = sub;
}

const overviewLevel = () => {
  const i = S.job?.levels?.findIndex(([w, hh]) => w * hh === S.overview?.pixels_used) ?? -1;
  return i >= 0 ? i : null;
};

function overviewText() {
  const n = S.overview.candidates?.length ?? 0;
  const top = (S.job?.levels?.length ?? 1) - 1;
  return `It shrank the photo into a pyramid (L0 is full size, each level up halves it, down to L${top}), `
    + `mapped where damage looks likely (the heatmap) and marked ${n} region${n === 1 ? '' : 's'} worth a closer look (dashed boxes).`;
}

/** The camera panel: the newest look's crop as the agent saw it, or the stages before the first look. */
function showCam(k) {
  const look = S.steps.slice(0, k).reverse().find((s) => s.crop_jpeg_b64);
  const url = look ? dataUrl('image/jpeg', look.crop_jpeg_b64) : null;
  el.camImg.hidden = el.pip.hidden = !url;
  el.stages.hidden = Boolean(url);
  if (!url) {
    el.camCap.textContent = S.job ? 'Before the first look' : 'No look yet. Press Run.';
    return;
  }
  el.camImg.src = el.pip.src = url;
  el.camImg.alt = `What the agent saw at step ${look.step}`;
  const L = look.decision.args.level;
  const ov = overviewLevel();
  const zoom = num(ov) ? 2 ** (ov - L) : null;
  el.camCap.textContent = [`step ${look.step}`, `L${L} · 1:${2 ** L}`,
    zoom > 1 ? `${zoom}× closer than the overview` : zoom === 1 ? 'the overview’s scale' : zoom ? `${1 / zoom}× wider than the overview` : null,
    look !== S.steps[k - 1] ? 'the last look before this step' : null].filter(Boolean).join(' · ');
}

function stage(name, text) {
  const li = $(`[data-stage=${name}]`, el.stages);
  li.classList.add('is-on');
  if (text) li.textContent = text;
}

// ---- stepping / replay --------------------------------------------------------------------------------
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
  const planner = step && S.planners.get(step.step);
  renderStepDetail(el.detail, step, planner);
  budget.show(step
    ? { pixels: step.pixels_frac ?? 0, usd: step.usd_used ?? 0, seconds: step.seconds_used ?? 0 }
    : { pixels: S.overview?.pixels_frac ?? 0, usd: 0, seconds: 0 });
  budget.setOps(step);
  showCam(k);
  if (step) {  // the NOTE keeps to the main sentence; the rest goes on its secondary line
    const [main, ...more] = explain(step, planner);
    narrate(main, more.length ? more.join(' ') : step.rationale ? `recorded: ${step.rationale}` : '');
  }
  else if (S.overview) narrate(overviewText());
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
  S.timer = setInterval(() => (viewport.cursor >= S.steps.length ? stopPlay() : goto(viewport.cursor + 1, true)), 1200);  // at least the 950 ms draw
}

function clearJob() {
  stopPlay();
  Object.assign(S, { job: null, overview: null, done: null, token: null, order: null, ui: null, rev: 0, planners: new Map(), critSeen: false });
  el.tbChecked.textContent = 'human reviewer';
  viewport.reset(null);
  S.steps = viewport.steps;
  trace.clear();
  findings.clear();
  el.slider.max = 0;
  el.slider.disabled = el.play.disabled = true;
  el.tiles.checked = false;
  el.tiles.disabled = true;
  el.tilesNote.hidden = true;
  el.recap.hidden = true;
  for (const li of el.stages.children) {
    li.classList.remove('is-on');
    li.textContent = li.dataset.text;
  }
  el.done.replaceChildren(h('p', { class: 'note' }, 'No job has finished yet. Its status, coverage and the signed work order appear here.'));
  narrate(NOTE);
  goto(0);
}

function levelsOf(region) {
  const levels = S.steps.filter((s) => s.region === region && num(s.decision?.args?.level)).map((s) => `L${s.decision.args.level}`);
  return levels.length ? `looked at ${levels.filter((l, i) => l !== levels[i - 1]).join(' → ')}` : '';
}

/** Which earlier measurement caused this step: the look it follows up, or the overview. */
function causeFrom(s) {
  if (['escalate', 'verify'].includes(s.decision?.action) && s.cause !== 'replay' && !String(s.cause).startsWith('planner:')) {
    const prev = S.steps.slice(0, -1).reverse().find((p) => p.region === s.region && p.decision?.action !== 'halt');
    return prev ? pad(prev.step) : null;
  }
  return s.cause === 'survey' || /^(?:.*; )?argmax/.test(s.rationale ?? '') ? 'the overview' : null;
}

function scrollToViewer() {
  bars();  // the job bar has just changed height (running), and the scroll margin is computed from it now
  $('.win--view').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
}

function showBox(box) {
  viewport.highlight(box);
  scrollToViewer();
}

function showFinding(f) {
  const i = S.steps.findIndex((s) => s.step === f.step);
  if (i >= 0) goto(i + 1);
  showBox(f.box);
}

// ---- stream events, presented one decision at a time ---------------------------------------------------
/** Present one event; returns how long it stays on screen before the next one (ms). */
function handle(ev) {
  switch (ev.type) {
    case 'job': {
      clearJob();
      S.job = ev;
      viewport.reset(ev);
      S.steps = viewport.steps;
      budget.setCaps(ev.budget);
      goto(0);
      const pv = ev.privacy ?? {};
      stage('photo', `photo received · ${ev.width} × ${ev.height} px`);
      stage('privacy', `privacy blur · ${pv.faces_blurred ?? 0} faces, ${pv.people_blurred ?? 0} people`);
      showCam(0);
      narrate(`Photo received: ${ev.width} × ${ev.height} px. Before any look, faces and people are blurred (${pv.faces_blurred ?? 0} faces and ${pv.people_blurred ?? 0} people here).`);
      if (!S.arrived) {
        el.text.textContent = S.opts.killHint ? `${MOCK ? 'Recorded run' : 'Live'}: each look is shown the moment it arrives. Press KILL in this bar while it is looking.`
          : MOCK ? 'Recorded run: replaying a saved job from fixtures (no server), at reading speed.'
            : ev.pace_ms > 0 ? `Paced for viewing: the agent waits ${ev.pace_ms / 1000} s between looks; timings exclude the wait.`
              : 'Running live on the server; each decision is shown at reading speed.';
      }
      return 900;
    }
    case 'overview': {
      S.overview = ev;
      viewport.setOverview(ev);
      const n = ev.candidates?.length ?? 0;
      stage('pyramid', `pyramid L0–L${(S.job?.levels?.length ?? 1) - 1}`);
      stage('saliency', 'saliency map (the heatmap)');
      stage('regions', `${n} candidate region${n === 1 ? '' : 's'}`);
      goto(0);
      return 2400;
    }
    case 'step':
      return addStep(ev);
    case 'finding':
      return addFinding(ev.finding);
    case 'review_request': {
      findings.addReason(ev.finding_id, ev.reason);
      if (!String(ev.reason).endsWith(' is critical')) {  // that one is said by the finding's own line
        const f = findings.cards.get(ev.finding_id)?.finding;
        trace.addNote('review', S.steps.at(-1)?.step ?? 0, ev.finding_id, `${f?.cls ?? 'finding'} ${ev.finding_id} → human`,
          /^iou /.test(ev.reason) ? `${ev.reason}: two looks disagree on a critical finding, so a person must check it` : ev.reason,
          () => findings.focus(ev.finding_id));
      }
      return 0;
    }
    case 'planner':
      S.planners.set(ev.step, ev);
      trace.addPlanner(ev);
      return 0;
    case 'done':
      finish(ev);
      return 0;
    case 'error':
      setMode('error', ev.message ?? 'The job failed.');
      return 0;
    case 'end':  // the stream closed without a done record
      if (S.mode === 'running') setMode('killed', `The stream ended at step ${S.steps.length} before the job finished. Resume to continue.`);
      return 0;
    default:
      return 0;
  }
}

function addStep(step) {
  const follow = viewport.cursor === S.steps.length;
  S.steps.push(step);
  trace.addStep(step, causeFrom(step));
  findings.refresh();
  el.slider.max = S.steps.length;
  el.slider.disabled = el.play.disabled = false;
  el.count.textContent = `step ${step.step}`;
  if (follow) goto(S.steps.length, !Q.skip);
  else el.sliderOut.textContent = `step ${viewport.cursor} / ${S.steps.length}`;
  // a halt record may carry the governor's last override; a failed planner consultation is in planner_status
  for (const cause of [step.governor?.cause, step.cause, plannerFell(step) && 'planner-fallback']) {
    if (isGovernor(cause)) popup(++S.rev, cause, GOVERNOR_TEXT[cause](step));
  }
  sound.scratch();
  return step.decision?.action === 'halt' ? 1500 : PACE;
}

function addFinding(f) {
  if (!findings.add(f)) return 0;  // a re-measure of a known finding updates its card only
  const at = S.steps.at(-1)?.step ?? f.step;
  trace.addNote('finding', at, f.id,
    `${f.cls} · ${px(f.measure?.area_px)} px²${f.critical ? ' → human sign-off requested' : ' · draft, no sign-off needed'}`,
    f.critical ? ruleOf(f) : null, () => findings.focus(f.id));
  if (!f.critical || S.critSeen) return 0;
  S.critSeen = true;
  narrate(`Critical: ${f.cls.replace('_', ' ')}, found at step ${at}. The agent cannot approve this; only you can, in step 4 once the run ends.`,
    ruleOf(f) ? `rule: ${ruleOf(f)}` : '');
  return HOLD;
}

function pump() {
  if (Q.timer) return;
  while (Q.items.length) {
    const wait = handle(Q.items.shift());
    if (wait && !Q.skip) {
      Q.timer = setTimeout(() => {
        Q.timer = 0;
        pump();
      }, wait);
      return;
    }
  }
}

/** Show everything that has arrived at once, and anything still to come as it arrives. */
function skip() {
  Q.skip = true;
  clearTimeout(Q.timer);
  Q.timer = 0;
  if (document.activeElement === el.skip) el.kill.focus({ preventScroll: true });
  el.skip.hidden = true;
  pump();
}

function arrive(ev) {
  if (ev.type === 'done') {
    S.arrived = ev;
    if (Q.timer || Q.items.length) {
      const n = S.events.filter((e) => e.type === 'step').length;
      el.text.textContent = `Replay: ${MOCK ? 'the recorded agent' : 'the agent'} took ${ev.summary.seconds} s for its ${n} decisions`  // a phone shows only the start
        + `${S.job?.pace_ms > 0 ? ' (not counting the paced waits)' : ''}; shown at reading speed. Skip to end shows the result now.`;
    }
  }
  if (!['end', 'error'].includes(ev.type)) S.events.push(ev);
  Q.items.push(ev);
  if (ev.type === 'error') skip();  // say what went wrong now, after what had arrived
  else pump();
}

// ---- work order / done state ---------------------------------------------------------------------------
function tile(label) {
  const v = h('b', { class: 'tile__v' }, '0');
  return [h('div', { class: 'tile' }, v, h('span', { class: 'tile__k' }, label)), v];
}

function save(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function tokenExp(token) {
  try {
    return JSON.parse(atob(String(token).split('.')[0].replace(/-/g, '+').replace(/_/g, '/'))).exp ?? null;
  } catch {
    return null;
  }
}

function buildDone() {
  const { summary: sm, coverage, unvisited, cause } = S.done;
  const pill = h('span', { class: 'pill' });
  const orderState = h('span', { class: 'note' });
  const specs = [
    ['looks', sm.steps, {}], ['of the pixels', sm.pixels_frac * 100, { decimals: 1, suffix: ' %' }],
    ['$ modelled', sm.usd, { decimals: 6, prefix: '$' }], ['seconds of agent time', sm.seconds, { decimals: 1 }],
    ['findings', sm.findings, {}], ['critical, waiting for you', sm.critical_pending, {}],
  ].map(([label, value, opts]) => ({ value, opts, pair: tile(label) }));
  const cover = h('div', { class: 'meter__fill' });
  const coverNum = h('b', { class: 'meter__val' }, '0 %');
  const privacy = S.job?.privacy;
  const order = h('button', { class: 'btn', type: 'button' }, 'Download work_order.json');
  order.addEventListener('click', () => save('work_order.json',
    JSON.stringify({ work_order: S.order, token: S.token, alg: 'HMAC-SHA256', exp: tokenExp(S.token) }, null, 2), 'application/json'));
  const log = h('button', { class: 'btn', type: 'button' }, 'Download trace.ndjson');
  log.addEventListener('click', () => save('trace.ndjson', `${S.events.map((e) => JSON.stringify(e)).join('\n')}\n`, 'application/x-ndjson'));
  const box = (id) => S.overview?.candidates?.find((c) => c.id === id)?.box;
  el.done.replaceChildren(h('div', { class: 'stack' },
    h('div', { class: 'done__head' }, pill, h('span', { class: 'note' }, `stopped because ${STOPPED[cause] ?? cause} (${cause})`)),
    orderState,
    h('div', { class: 'tiles' }, specs.map((s) => s.pair[0])),
    h('div', { class: 'meter' },
      h('div', { class: 'meter__head' }, h('span', {}, 'salient area seen at any level (incl. survey)'), h('span', {}, coverNum)),
      h('div', { class: 'meter__track', role: 'progressbar', 'aria-label': 'Saliency-weighted coverage at any level', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': Math.round(coverage * 100) }, cover)),
    h('div', {}, h('h4', { class: 'label' }, 'not zoomed into'),
      unvisited?.length ? h('p', { class: 'chips' }, unvisited.map((r) => {
        const b = h('button', { class: 'chip chip--look chip--btn', type: 'button' }, `${r}: check it yourself`);
        b.addEventListener('click', () => showBox(box(r)));
        return b;
      })) : h('p', { class: 'note' }, 'none: every candidate region was looked at')),
    privacy ? h('p', { class: 'note' }, `privacy: ${privacy.faces_blurred ?? 0} faces and ${privacy.people_blurred ?? 0} people blurred`) : null,
    h('div', { class: 'done__dl' }, order, log)));
  S.ui = { pill, orderState, pending: specs[5].pair[1] };
  specs.forEach((s) => countUp(s.pair[1], s.value, s.opts));
  requestAnimationFrame(() => { cover.style.width = `${coverage * 100}%`; });
  countUp(coverNum, coverage * 100, { decimals: 0, suffix: ' %' });
  updateDone();
}

function updateDone() {
  const pending = pendingCount();
  const key = doneKey();
  S.ui.pill.className = `pill pill--stamp pill--${key}`;
  if (S.ui.pill.textContent !== PILL[key]) {
    S.ui.pill.textContent = PILL[key];
    replay(S.ui.pill, 'is-stamped');
  }
  const signers = [...new Set((S.order.approvals ?? []).map((a) => a.reviewer).filter(Boolean))];
  el.tbChecked.textContent = signers.length ? signers.join(', ') : 'human reviewer';
  const critical = S.order.findings.some((f) => f.critical);
  S.ui.orderState.textContent = pending
    ? `Waiting for your sign-off (${pending}): work order ${S.order.id} is ${S.order.status} until a person approves or rejects each critical finding.`
    : critical ? `Signed: ready to send. Work order ${S.order.id} is ${S.order.status}.`
      : `Work order ${S.order.id} is ${S.order.status}: nothing critical needed a sign-off.`;
  countUp(S.ui.pending, pending);
  setMode('done', el.text.textContent);
}

/** "What you just saw": the run in four honest numbers, and two ways to try it harder. */
function recap() {
  const kinds = { vision: 0, governor: 0, planner: 0, replay: 0 };
  for (const s of S.steps) kinds[kindOf(s.cause)] += 1;
  const sm = S.done.summary;
  const crit = S.order.findings.filter((f) => f.critical).length;
  const why = [[kinds.vision, 'from what OpenCV measured in the photo'], [kinds.governor, 'from the governor’s timing and budget checks'],
    [kinds.planner, 'from the planner drill (the rule planner’s choice stood)'], [kinds.replay, 'replayed from before KILL']]
    .filter(([n]) => n).map(([n, t]) => `${n} ${t}`);
  const breakIt = h('button', { class: 'btn btn--amber btn--sm', type: 'button' }, 'Now break it');
  breakIt.addEventListener('click', () => run({ pace: GUIDE_PACE_MS, chaos: 'planner_malformed,p95_spike' }));
  const tryKill = h('button', { class: 'btn btn--red btn--sm', type: 'button' }, 'Try KILL');
  tryKill.addEventListener('click', () => run({ pace: GUIDE_PACE_MS, chaos: 'planner_timeout', killHint: true }));
  const sign = h('button', { class: 'btn btn--amber btn--sm', type: 'button', id: 'sign-off' }, 'Sign them off');
  sign.addEventListener('click', () => findings.focus(S.order.findings.find((f) => f.critical && f.state === 'pending_review')?.id, true));
  el.recap.replaceChildren(
    h('h4', { class: 'recap__t' }, 'What you just saw'),
    h('ul', { class: 'recap__list' },
      h('li', {}, h('b', {}, String(S.steps.length)), ` decisions, every one with a recorded cause: ${why.join(', ')}.`),
      h('li', {}, h('b', {}, String(crit)), ` critical finding${crit === 1 ? ' was' : 's were'} sent to a person: the agent is not allowed to approve ${crit === 1 ? 'it' : 'them'}.`),
      h('li', {}, h('b', {}, `${Math.round(S.done.coverage * 100)} %`), ' of the salient area was seen at some level (the first survey look counts).'),
      h('li', {}, h('b', {}, `$${sm.usd.toFixed(6)}`), ` modelled for ${sm.seconds} s of agent time (wall time × $0.339/h, not billed).`)),
    h('div', { class: 'recap__next' },
      pendingCount() ? h('p', {}, sign, ' takes you to the first card that needs a person: type your name, then Approve or Reject.') : null,
      h('p', {}, breakIt, ' reruns it with a planner that sends a shell command and a segmenter slowed on purpose.'),
      h('p', {}, tryKill, ' starts a run shown live, whose planner stalls 2 s at step 3, so you can halt it with KILL in the bar above.')));
  el.recap.hidden = false;
}

function finish(ev) {
  clearTimeout(S.killTimer);
  S.done = ev;
  S.token = ev.token;
  S.order = ev.work_order;
  const sm = ev.summary;
  const n = sm.critical_pending;
  setMode('done', S.note ?? (ev.cause === 'kill-switch'
    ? `Halted by KILL at step ${sm.steps}: the trace shows kill-switch and the state is kept. Resume carries on.`
    : `Done in ${sm.seconds} s of agent time: ${sm.steps} looks, ${(sm.pixels_frac * 100).toFixed(1)} % of the pixels.`));
  narrate(n ? `${n} critical finding${n === 1 ? ' needs' : 's need'} you. Sign ${n === 1 ? 'it' : 'them'} off in step 4: the agent cannot.`
    : 'Nothing critical was found, so nothing needs your sign-off.', haltText(ev.cause));
  findings.setReady();
  findings.refresh();
  el.tiles.disabled = false;
  buildDone();
  recap();
  replay(el.tbSheet, 'is-stamped');  // the sheet number stamps in: this run is on file
  sound.stamp();
  if (S.opts.guided) {  // end on what the run showed, in view; its Sign them off button hands over to the human
    el.recap.scrollIntoView({ block: 'start', behavior: reduced() ? 'auto' : 'smooth' });
    $('#sign-off')?.focus({ preventScroll: true });
  }
}

async function review(id, decision, reviewer) {
  const res = await postReview({ token: S.token, finding_id: id, decision, reviewer });
  S.token = res.token;
  S.order = res.work_order;
  findings.sync(S.order);
  updateDone();
}

// ---- running a job -----------------------------------------------------------------------------------
function params(opts) {
  const p = {
    pixel_cap: +(Number($('#cap-px').value) / 100).toFixed(2),
    usd_cap: $('input[name=usd]:checked').value,
    seconds_cap: $('#cap-sec').value,
  };
  const chaos = opts.chaos ?? $$('input[name=chaos]:checked').map((i) => i.value).join(',');
  if (chaos) p.chaos = chaos;
  if (opts.pace) p.pace_ms = opts.pace;
  return p;
}

/** opts: { replay, pace (ms the server waits between looks), chaos, guided, killHint }. */
async function run(opts = {}) {
  if (!S.image || S.mode === 'running') return;
  S.opts = opts;
  S.ctl = new AbortController();
  Object.assign(S, { events: [], arrived: null, note: null });
  clearTimeout(Q.timer);
  Object.assign(Q, { items: [], timer: 0, skip: Boolean(opts.killHint) });  // a KILL drill is shown live: never behind the job it asks to stop
  const refocus = [el.look, el.runBar, el.resume, el.guided].includes(document.activeElement);  // all hidden or disabled while running
  setMode('running', MOCK ? 'Recorded run: replaying a saved job (no server).'
    : opts.replay ? 'Resuming: re-sending the photo and replaying the moves so far…' : undefined);
  if (refocus) el.skip.focus({ preventScroll: true });  // the safe next key press: show the result now
  scrollToViewer();
  try {
    for await (const ev of streamJob({ blob: S.image.blob, params: params(opts), replay: opts.replay, signal: S.ctl.signal })) arrive(ev);
    arrive({ type: 'end' });
  } catch (err) {
    if (err.name === 'AbortError') return;
    const offline = err instanceof TypeError;  // fetch could not reach the server at all
    el.recorded.hidden = MOCK || !offline;
    arrive({ type: 'error', message: offline ? 'Could not reach the API. Check your connection, or watch a recorded run instead.' : `The job failed: ${err.message}` });
  }
}

/** Abort the stream here, show what arrived, and say so. */
function stop(why) {
  S.ctl.abort();
  skip();
  const n = S.steps.length;
  setMode('killed', `Killed at step ${n}${why} — state preserved. Resume re-sends the photo and replays your ${n} moves.`);
}

function kill() {
  if (S.arrived) {  // the server finished before the click: there is nothing left to stop
    S.note = `The job had already finished on the server (the agent took ${S.arrived.summary.seconds} s) before KILL could reach it; here is its result.`;
    return skip();
  }
  if (MOCK || !S.job) return stop('');  // a recording (or a job not started yet) stops here
  postKill(S.job.job, S.job.kill_token);  // the server halts before its next look and sends the kill-switch record
  skip();
  el.text.textContent = 'KILL sent: the agent halts before its next look…';
  clearTimeout(S.killTimer);
  S.killTimer = setTimeout(() => {
    if (S.mode === 'running' && !S.arrived) stop(': the server did not confirm in time, so the page closed the stream');
  }, 6000);
}

function resume() {
  const moves = S.steps.filter((s) => ['gaze', 'escalate', 'verify'].includes(s.decision?.action) && pointOf(s))
    .map((s) => [s.decision.action, s.decision.args.x, s.decision.args.y, s.decision.args.level, s.decision.args.size]
      .filter((v) => v !== undefined));
  run({ pace: S.opts.pace, chaos: S.opts.chaos, replay: moves.slice(0, 200) });
}

async function guided() {
  if (S.mode === 'running') return;
  await S.ready;
  const btn = !MOCK && $(`.thumb[data-file="${GUIDE}"]`, el.gallery);
  if (btn && S.sample !== GUIDE) await choose(btn);
  run({ pace: GUIDE_PACE_MS, guided: true });
}

// ---- image picking -----------------------------------------------------------------------------------
async function useImage(blob, label, file = null) {
  if (S.mode === 'running') {
    el.prep.textContent = 'A job is running. KILL it first to change the photo.';
    return;
  }
  el.prep.textContent = 'Reading the photo…';
  try {
    const p = await prepareImage(blob);
    if (S.image) URL.revokeObjectURL(S.image.url);
    S.image = { blob: p.blob, url: URL.createObjectURL(blob) };
    S.sample = file;
    clearJob();
    viewport.setImage(S.image.url);
    el.prep.textContent = [label, `${p.width}×${p.height}`, `${(p.blob.size / 1e6).toFixed(1)} MB`, p.note].filter(Boolean).join(' · ');
    el.say1.textContent = `Ready: ${label} (${p.width} × ${p.height} px). Pick another sample, or use your own photo.`;
    for (const b of $$('.thumb', el.gallery)) b.setAttribute('aria-pressed', String(b.dataset.file === file));
    setMode('idle');
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

async function choose(btn) {
  const s = btn.sample;
  if (S.mode === 'running') { el.prep.textContent = 'A job is running. KILL it first to change the photo.'; return; }
  el.look.disabled = el.runBar.disabled = true;  // the old photo must not run while the new one loads
  el.prep.textContent = `Loading ${String(s.title ?? s.file)}…`;
  const res = await fetch(`samples/${s.file}`).catch(() => null);
  if (!res?.ok) {
    el.prep.textContent = 'That sample could not be loaded.';
    el.look.disabled = el.runBar.disabled = !S.image;
    return;
  }
  await useImage(await res.blob(), String(s.title ?? s.file), s.file);
}

/** Builds the gallery; resolves to the sample to preselect (the guided one when it is there). */
async function loadSamples() {
  const list = await getJSON('samples/index.json');
  const items = (Array.isArray(list) ? list : []).filter((s) => s && /^[\w.-]+$/.test(s.file ?? ''));
  el.galleryEmpty.hidden = items.length > 0;
  const btns = items.map((s) => {
    const thumb = /^\w[\w./-]*$/.test(s.thumb ?? '') && !s.thumb.includes('..') ? s.thumb : s.file;
    const btn = h('button', { class: 'thumb', type: 'button', 'aria-pressed': 'false', 'data-file': s.file },
      h('img', { src: `samples/${thumb}`, alt: '', loading: 'lazy', width: 96, height: 72 }),
      h('span', { class: 'thumb__t' }, String(s.title ?? s.file)),
      s.note ? h('span', { class: 'thumb__n' }, String(s.note)) : null,
      s.synthetic ? h('span', { class: 'tag' }, 'synthetic') : null);
    btn.sample = s;
    btn.addEventListener('click', () => choose(btn));
    return btn;
  });
  el.gallery.replaceChildren(...btns.map((b) => h('li', {}, b)));
  if (!items.length) el.say1.textContent = 'Drop your own JPEG or PNG to start.';
  return btns.find((b) => b.dataset.file === GUIDE) ?? btns[0];
}

// ---- wiring ------------------------------------------------------------------------------------------
el.look.addEventListener('click', () => run());
el.runBar.addEventListener('click', () => run());
el.guided.addEventListener('click', guided);
el.skip.addEventListener('click', skip);
el.kill.addEventListener('click', kill);
el.resume.addEventListener('click', resume);
el.play.addEventListener('click', togglePlay);
el.heat.addEventListener('change', () => viewport.setHeat(el.heat.checked));
el.tiles.addEventListener('change', () => {
  viewport.setTiles(el.tiles.checked);
  el.tilesNote.hidden = !el.tiles.checked;
  if (S.job) el.tilesNote.textContent = `${tileBoxes(S.job.width, S.job.height).length} tiles of 512 px at full resolution for a full tiling scan, vs ${S.done?.summary.steps ?? S.steps.length} agent looks.`;
});
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
for (const li of el.stages.children) li.dataset.text = li.textContent;

// The nav marks the section in the middle of the screen; the phone menu closes once a link is picked.
const NAV = { demo: '#demo', how: '#how', benchmarks: '#benchmarks', failures: '#benchmarks', docs: '#docs' };
const navLinks = $$('.nav__links a, .nav__drop a');
const navSpy = new IntersectionObserver((entries) => {
  for (const e of entries.filter((x) => x.isIntersecting)) {
    for (const a of navLinks) {
      if (a.getAttribute('href') === NAV[e.target.id]) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    }
  }
}, { rootMargin: '-45% 0px -50% 0px' });
$$('#top, main > section, #docs').forEach((s) => navSpy.observe(s));
$('#nav-menu').addEventListener('click', (e) => { if (e.target.closest('a')) $('#nav-menu').open = false; });

// Phones see only the photo and its numbered balloons (the parts list below carries the callout text).
const artMq = matchMedia('(max-width: 560px)');
const fitArt = () => $('.art__svg').setAttribute('viewBox', artMq.matches ? '0 0 492 372' : '0 0 600 430');
artMq.onchange = fitArt;
fitArt();

// The sticky bars' real heights: scroll padding and focus scrolling keep controls clear of them.
function bars() {
  document.documentElement.style.setProperty('--nav', `${$('#nav').offsetHeight}px`);
  document.documentElement.style.setProperty('--bar', `${$('#taskbar').offsetHeight}px`);
}
new ResizeObserver(bars).observe($('#taskbar'));

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
S.ready = loadSamples().then((first) => (MOCK
  ? fetch('fixtures/mock.jpg').then((r) => r.blob()).then((b) => useImage(b, 'mock.jpg (recorded job)'))
  : first && choose(first)));
initResults();
health().then((hl) => {
  const arch = hl?.arch ?? hl?.machine;
  const where = [hl?.host && `host ${hl.host}`, arch, hl?.commit && `commit ${String(hl.commit).slice(0, 7)}`].filter(Boolean);
  $('#api-status').textContent = hl?.ok
    ? ['API online', `OpenCV ${hl.opencv}`, hl.build, `segmenter ${hl.segmenter}`, ...where].join(' · ')
    : 'The API is not reachable from this page. Watch a recorded run instead: it replays a saved job.';
  if (hl?.ok && hl.host) $('#live-where').textContent = `Live demo: OpenCV ${hl.opencv}, ${hl.build}, on ${hl.host}${arch ? ` (${arch})` : ''}.`;
  el.recorded.hidden = MOCK || Boolean(hl?.ok);
});
