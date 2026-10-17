// Budget ruler meters, trace schedule lines, the step detail card, and the plain-words narration of every decision.
import { $, $$, h, num } from './dom.js';
import { countUp } from './fx.js';

const GOVERNOR = new Set(['p95>SLA', 'warp-spike', 'usd≥80%', 'cap-hit', 'kill-switch', 'planner-fallback']);
export const isGovernor = (cause) => GOVERNOR.has(cause);
// A planner that timed out or was refused leaves the rule's look standing; the step keeps its own cause.
export const plannerFell = (s) => ['rejected', 'fallback'].includes(s.planner_status?.status);

const CHIP = {
  'best-score': 'look', 'entropy>tau': 'unsure', 'area<A_min': 'unsure', 'iou<0.5': 'unsure',
  'verify-new-mask': 'verify', replay: 'plain', survey: 'look',
};
const chip = (cause) => h('span', { class: `chip chip--${GOVERNOR.has(cause) ? 'gov' : CHIP[cause] ?? 'plain'}` }, cause);

const ms = (v) => (num(v) ? `${v}` : '–');
const pair = (rt, op) => `${ms(rt?.p50_ms?.[op])} / ${ms(rt?.p95_ms?.[op])} ms`;
export const pad = (n) => `#${String(n).padStart(2, '0')}`;
export const px = (v) => (num(v) ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : '–');

// ---- plain words -------------------------------------------------------------------------------
const size = (L) => (L ? `1/${2 ** L} size` : 'full resolution');
const name = (cls) => String(cls).replace('_', ' ');

/** What kind of evidence a cause is: the photo itself (OpenCV), the governor, the planner drill, or a replay. */
export const kindOf = (cause) => (cause === 'replay' ? 'replay'
  : cause === 'planner-fallback' || String(cause).startsWith('planner:') ? 'planner'
    : GOVERNOR.has(cause) ? 'governor' : 'vision');

const HALT = {
  'stop-rule': 'The next look would be worth less than it costs, so it stops here.',
  exhausted: 'Every candidate region has been looked at, so it stops.',
  'cap-hit': 'A budget cap was reached (budget spent), so it stops and reports what it did not reach.',
  'kill-switch': 'You pressed KILL, so it stopped before its next look. Its state is kept: Resume carries on from here.',
  'planner:stop': 'The planner proposed stopping and the validator accepted, so it stops.',
};
export const haltText = (cause) => HALT[cause] ?? `It stopped (${cause}).`;

function governorText(cause, why) {
  if (cause === 'p95>SLA') {
    const m = why.match(/p95\(segment\) (\d+) ms > SLA (\d+) ms/);
    return `The segmenter got slow${m ? ` (p95 ${m[1]} ms, over the ${m[2]} ms limit)` : ''}, so the governor (the budget-and-speed watchdog) switched to the fast model.`;
  }
  if (cause === 'warp-spike') return 'Cutting out views got slow, so the governor shrank the view and batches the remaining regions.';
  if (cause === 'usd≥80%') return '80 % of the $ budget is spent, so the governor now allows double-checks only on critical damage.';
  return '';
}

function plannerText(ev) {
  if (ev?.status === 'fallback') {
    return `The planner ${ev.reason === 'timeout' ? 'did not answer in time' : `failed (${ev.reason})`}, so the built-in rule planner's choice stands.`;
  }
  if (ev?.status === 'rejected') return `The planner proposed something not allowed (${ev.reason}), so it was rejected and the rule planner's choice stands.`;
  return '';
}

/** The look itself, from the rationale the agent recorded (its cause can be overridden by a governor or planner note). */
function lookText(s) {
  const why = s.rationale ?? '';
  const L = s.decision?.args?.level;
  const now = num(L) ? ` (now L${L}, ${size(L)})` : '';
  let m;
  if (s.cause === 'replay') return `Replaying look ${s.step} from before KILL; nothing new is decided.`;
  if ((m = why.match(/one L(\d) look covers all (\d+) candidates/))) return `First it looks at the whole photo at ${size(+m[1])} (L${m[1]}) to see all ${m[2]} candidate regions at once, before zooming in anywhere.`;
  if ((m = why.match(/entropy ([\d.]+) > ([\d.]+) at L\d/))) return `It was unsure what it saw (entropy ${m[1]} > ${m[2]}), so it zooms in 2× on the same spot${now}.`;
  if ((m = why.match(/(\w+) area (\d+) px < (\d+) at L\d/))) return `The ${name(m[1])} it found is tiny (${m[2]} px < ${m[3]} px), so it zooms in 2× to measure it properly${now}.`;
  if ((m = why.match(/iou ([\d.]+) < 0\.5 between L(\d) and L(\d)/))) return `Its L${m[2]} and L${m[3]} looks disagree (overlap ${m[1]} < 0.5), so it takes one more, finer look${now}.`;
  if ((m = why.match(/new (\w+) mask at L(\d); confirming at L(\d)/))) return `A new ${name(m[1])} mask appeared at L${m[2]}, so it double-checks it one level finer (L${m[3]}) before trusting it.`;
  if ((m = why.match(/argmax score [^:]+: prior ([\d.]+), unseen ([\d.]+), t_hat (\d+) ms/))) {
    return `Region ${s.region} is now the best value for the time: damage looks likely there (prior ${m[1]}), ${Math.round(+m[2] * 100)} % of it is unseen, and a look costs about ${m[3]} ms. So it looks there next${now}.`;
  }
  if (String(s.cause).startsWith('planner:')) return `The planner asked to ${s.cause.slice(8).replace('_', ' ')}, and the validator allowed it${now}.`;
  return why || 'No reason was recorded.';
}

/** One step in plain English, as its sentences: governor or planner notes first (the main one when there is one),
    then the look (or the halt), then the quality gate. */
export function explain(s, planner) {
  const why = s.rationale ?? '';
  const halt = s.decision?.action === 'halt';
  const gov = s.governor?.cause ?? (isGovernor(s.cause) && !['cap-hit', 'kill-switch', 'planner-fallback'].includes(s.cause) ? s.cause : null);
  const quality = why.match(/quality gate: (.+)$/);
  return [
    gov ? governorText(gov, s.governor?.rationale ?? why) : '',
    plannerText(planner ?? s.planner_status),
    halt ? haltText(s.cause) : lookText(s),
    quality ? `The view failed the quality check (${quality[1]}), so it does not trust this look.` : '',
  ].filter(Boolean);
}

/** The rationale as text nodes, its deciding number (the one before a comparison, else the first) in bold. */
function marked(text) {
  const m = text.match(/(-?\d[\d.]*(?:e[-+]?\d+)?)(?= (?:ms |px )?[<>])/) ?? text.match(/-?\d[\d.]*(?:e[-+]?\d+)?/);
  if (!m) return [text];
  return [text.slice(0, m.index), h('b', { class: 't-key' }, m[0]), text.slice(m.index + m[0].length)];
}

/** What a look measured, in one line. */
function sawText(s) {
  const o = s.observation;
  if (!o) return null;
  const top = Object.entries(o.classes ?? {}).sort((x, y) => y[1] - x[1])[0];
  return [
    num(o.entropy) ? `entropy ${o.entropy.toFixed(2)}` : null,
    num(s.verify?.iou) ? `iou ${s.verify.iou.toFixed(2)} vs L${s.verify.against_level}` : null,
    top ? `${top[0]} ${px(top[1])} px` : 'no damage mask',
    s.quality?.ok === false ? `quality: ${s.quality.reason ?? 'rejected'}` : null,
  ].filter(Boolean).join(' · ');
}

export class Budget {
  constructor(root) {
    this.rows = Object.fromEntries($$('[data-meter]', root).map((m) => [m.dataset.meter, m]));
    this.ops = root;
    this.caps = null;
  }

  setCaps(b) {
    this.caps = {
      pixels: b.pixel_cap_frac, usd: b.usd_cap, seconds: b.seconds_cap,
    };
    const text = { pixels: `of cap ${+(b.pixel_cap_frac * 100).toFixed(1)} % of image`, usd: `of cap $${b.usd_cap}`, seconds: `of cap ${b.seconds_cap} s` };
    for (const [k, row] of Object.entries(this.rows)) $('.meter__cap', row).textContent = text[k];
  }

  /** `used` = { pixels (fraction of image), usd, seconds }. */
  show(used) {
    const fmt = {
      pixels: [used.pixels * 100, { decimals: 1, suffix: ' %' }],
      usd: [used.usd, { decimals: 6, prefix: '$' }],
      seconds: [used.seconds, { decimals: 1, suffix: ' s' }],
    };
    for (const [k, row] of Object.entries(this.rows)) {
      const share = this.caps?.[k] ? Math.min(100, (used[k] / this.caps[k]) * 100) : 0;
      countUp($('.meter__val', row), ...fmt[k]);
      $('.meter__fill', row).style.width = `${share}%`;
      $('.meter__track', row).setAttribute('aria-valuenow', Math.round(share));
      row.classList.toggle('meter--hot', share >= 80);
    }
  }

  setOps(step) {
    const rt = step?.runtime;
    $('[data-op=warp]', this.ops).textContent = rt ? pair(rt, 'warp') : '—';
    $('[data-op=segment]', this.ops).textContent = rt ? pair(rt, 'segment') : '—';
    $('[data-op=model]', this.ops).textContent = step?.model ?? '—';
  }
}

export class Trace {
  constructor(list, empty, onPick) {
    this.list = list;
    this.empty = empty;
    this.onPick = onPick;
  }

  clear() {
    this.list.replaceChildren();
    this.empty.hidden = false;
  }

  push(li) {
    const stick = this.list.scrollHeight - this.list.scrollTop - this.list.clientHeight < 40;
    this.empty.hidden = true;
    this.list.append(li);
    if (stick) this.list.scrollTop = this.list.scrollHeight;
  }

  /** `from` names what caused the step: an earlier step ("#01") or "the overview". */
  addStep(s, from) {
    const a = s.decision?.args ?? {};
    const parts = [
      h('span', { class: 't-num' }, pad(s.step)),
      h('span', { class: `t-act t-act--${s.decision?.action}` }, s.decision?.action ?? 'step'),
    ];
    if (num(a.level)) parts.push(h('span', {}, `L${a.level}`));
    if (num(a.x) && num(a.y)) parts.push(h('span', {}, `(${a.x},${a.y})`));
    if (s.region) parts.push(h('span', {}, s.region));
    if (s.cause) parts.push(chip(s.cause));
    if (s.governor?.cause) parts.push(chip(s.governor.cause));  // an override made on the step that then halted
    if (plannerFell(s)) parts.push(chip('planner-fallback'));
    if (s.quality?.ok === false) parts.push(h('span', { class: 'chip chip--gov' }, `quality: ${s.quality.reason ?? 'rejected'}`));
    if (s.injected) parts.push(h('span', { class: 'chip chip--inject' }, `injected ${s.injected}`));
    if (num(s.latency_ms)) parts.push(h('span', { class: 't-ms' }, `${s.latency_ms} ms`));
    const saw = sawText(s);
    const btn = h('button', { type: 'button', class: 'trace__btn' },
      h('span', { class: 'trace__head' }, parts),
      s.rationale ? h('span', { class: 'trace__why' }, h('i', {}, 'because'), from ? ` (from ${from})` : '', ': ', marked(s.rationale)) : null,
      saw ? h('span', { class: 'trace__saw' }, h('i', {}, 'saw'), ': ', saw) : null);
    btn.addEventListener('click', () => this.onPick(s.step));
    this.push(h('li', { class: 'trace__line', 'data-step': s.step }, btn));
  }

  /** A finding or a review request: what the agent handed on, and why. Clicking it opens the finding's card. */
  addNote(kind, stepNo, id, text, because, onOpen) {
    const btn = h('button', { type: 'button', class: 'trace__btn' },
      h('span', { class: 'trace__head' }, h('span', { class: 't-num' }, pad(stepNo)), h('span', { class: `t-act t-act--${kind}` }, kind), h('span', {}, text)),
      because ? h('span', { class: 'trace__why' }, h('i', {}, 'because'), ': ', because) : null);
    btn.addEventListener('click', onOpen);
    this.push(h('li', { class: `trace__line trace__line--${kind}`, 'data-finding': id }, btn));
  }

  addPlanner(ev) {
    this.push(h('li', { class: 'trace__line trace__line--planner' },
      h('span', { class: 't-num' }, pad(ev.step)), h('span', { class: 't-act t-act--planner' }, 'planner'),
      `${ev.status}${ev.reason ? ` (${ev.reason})` : ''}`,
      h('span', { class: 'trace__why' }, plannerText(ev) || 'The planner\'s plan passed the allow-list.')));
  }

  mark(stepNo) {
    for (const li of $$('.trace__line[data-step]', this.list)) {
      const on = Number(li.dataset.step) === stepNo;
      li.classList.toggle('is-current', on);
      if (on) li.firstElementChild.setAttribute('aria-current', 'step');
      else li.firstElementChild.removeAttribute('aria-current');
    }
  }
}

const row = (k, v) => (v == null || v === '' ? null : h('div', {}, h('dt', {}, k), h('dd', {}, v)));

export function renderStepDetail(root, step, planner) {
  if (!step) {
    root.replaceChildren(h('p', { class: 'note' }, 'No step picked yet. Scrub the replay slider, click a numbered circle on the photo or click a trace line.'));
    return;
  }
  const a = step.decision?.args ?? {};
  const o = step.observation;
  const rt = step.runtime;
  const saw = sawText(step);
  const classes = o?.classes && Object.keys(o.classes).length ? Object.entries(o.classes).map(([k, v]) => `${k} ${px(v)} px`).join(', ') : 'none';
  root.replaceChildren(...[
    h('p', { class: 'detail__title' }, `step ${step.step} · ${step.decision?.action ?? 'step'}`, step.cause ? ' ' : null, step.cause ? chip(step.cause) : null),
    h('h4', { class: 'detail__h' }, 'Why this look'),
    h('p', { class: 'detail__why' }, explain(step, planner).join(' ')),
    saw ? h('h4', { class: 'detail__h' }, 'What it saw') : null,
    saw ? h('p', { class: 'detail__why' }, saw) : null,
    h('details', { class: 'detail__more' }, h('summary', {}, 'All numbers for this step'),
      h('p', { class: 'detail__raw' }, 'recorded: ', step.rationale ?? 'no rationale recorded'),
      step.governor?.cause ? h('p', { class: 'detail__raw' }, `governor: ${step.governor.cause}: ${step.governor.rationale ?? ''}`) : null,
      h('dl', { class: 'kv' },
        row('entropy (unsure, 0-1)', num(o?.entropy) ? o.entropy.toFixed(2) : null),
        row('mask area', num(o?.mask_area_px) ? `${px(o.mask_area_px)} px` : null),
        row('classes', o ? classes : null),
        row('verify IoU (overlap)', num(step.verify?.iou) ? `${step.verify.iou.toFixed(2)} vs L${step.verify.against_level}` : null),
        row('sharpness / exposure', o ? `${o.sharpness ?? '–'} / ${o.exposure ?? '–'}` : null),
        row('quality', step.quality ? (step.quality.ok ? 'ok' : `rejected: ${step.quality.reason}`) : null),
        row('region', step.region),
        row('level', num(a.level) ? `L${a.level} (${size(a.level)})` : null),
        row('centre (L0 px)', num(a.x) ? `${a.x}, ${a.y}` : null),
        row('model / planner', step.model ? `${step.model} / ${step.planner ?? '–'}` : null),
        row('warp p50/p95', rt ? pair(rt, 'warp') : null),
        row('segment p50/p95', rt ? pair(rt, 'segment') : null),
        row('latency / cost', num(step.latency_ms) ? `${step.latency_ms} ms / $${step.cost_usd ?? '–'}` : null))),
  ].filter(Boolean));
}
