// budget.sys meters, trace.log lines and the step detail card.
import { $, $$, h, dataUrl, num } from './dom.js';
import { countUp } from './fx.js';

const GOVERNOR = new Set(['p95>SLA', 'warp-spike', 'usd≥80%', 'cap-hit', 'kill-switch', 'planner-fallback']);
export const isGovernor = (cause) => GOVERNOR.has(cause);

const CHIP = {
  'best-score': 'look', 'entropy>tau': 'unsure', 'area<A_min': 'unsure', 'iou<0.5': 'unsure',
  'verify-new-mask': 'verify', replay: 'chrome',
};
const chip = (cause) => h('span', { class: `chip chip--${GOVERNOR.has(cause) ? 'gov' : CHIP[cause] ?? 'chrome'}` }, cause);

const ms = (v) => (num(v) ? `${v}` : '–');
const pair = (rt, op) => `${ms(rt?.p50_ms?.[op])} / ${ms(rt?.p95_ms?.[op])} ms`;

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
      usd: [used.usd, { decimals: 5, prefix: '$' }],
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

  addStep(s) {
    const a = s.decision?.args ?? {};
    const parts = [
      h('span', { class: 't-num' }, `#${String(s.step).padStart(2, '0')}`),
      h('span', { class: `t-act t-act--${s.decision?.action}` }, s.decision?.action ?? 'step'),
    ];
    if (num(a.level)) parts.push(h('span', {}, `L${a.level}`));
    if (num(a.x) && num(a.y)) parts.push(h('span', {}, `(${a.x},${a.y})`));
    if (s.region) parts.push(h('span', {}, s.region));
    if (num(s.observation?.entropy)) parts.push('·', h('span', {}, 'entropy ', h('b', {}, s.observation.entropy.toFixed(2))));
    if (num(s.verify?.iou)) parts.push('·', h('span', {}, 'iou ', h('b', {}, s.verify.iou.toFixed(2))));
    if (s.cause) parts.push('·', chip(s.cause));
    if (s.quality?.ok === false) parts.push(h('span', { class: 'chip chip--gov' }, `quality: ${s.quality.reason ?? 'rejected'}`));
    if (s.injected) parts.push(h('span', { class: 'chip chip--inject' }, `injected ${s.injected}`));
    if (num(s.latency_ms)) parts.push('·', h('span', {}, `${s.latency_ms} ms`));
    const btn = h('button', { type: 'button', class: 'trace__btn', title: s.rationale ?? '' }, ...parts);
    btn.addEventListener('click', () => this.onPick(s.step));
    this.push(h('li', { class: 'trace__line', 'data-step': s.step }, btn));
  }

  addPlanner(ev) {
    this.push(h('li', { class: 'trace__line trace__line--planner' },
      h('span', { class: 't-num' }, `#${String(ev.step).padStart(2, '0')}`), h('span', { class: 't-act t-act--planner' }, 'planner'),
      `${ev.status}${ev.reason ? ` (${ev.reason})` : ''}`));
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

export function renderStepDetail(root, step) {
  if (!step) {
    root.replaceChildren(h('p', { class: 'note' }, 'Scrub the replay slider or click a numbered dot to inspect a step.'));
    return;
  }
  const a = step.decision?.args ?? {};
  const o = step.observation;
  const rt = step.runtime;
  const crop = dataUrl('image/jpeg', step.crop_jpeg_b64);
  const classes = o?.classes && Object.keys(o.classes).length ? Object.entries(o.classes).map(([k, v]) => `${k} ${v}`).join(', ') : 'none';
  root.replaceChildren(
    h('div', { class: 'detail__head' },
      crop ? h('img', { class: 'detail__crop', src: crop, alt: `Gaze crop for step ${step.step}`, width: 80, height: 80 }) : null,
      h('div', {},
        h('p', { class: 'detail__title' }, `step ${step.step} · ${step.decision?.action ?? 'step'}`, step.cause ? ' ' : null, step.cause ? chip(step.cause) : null),
        h('p', { class: 'detail__why' }, step.rationale ?? 'no rationale recorded'))),
    h('dl', { class: 'kv' },
      row('region', step.region),
      row('level', num(a.level) ? `L${a.level}` : null),
      row('centre (L0 px)', num(a.x) ? `${a.x}, ${a.y}` : null),
      row('window (L0 px)', step.window?.length === 4 ? `${step.window[0]}, ${step.window[1]} · ${step.window[2]}×${step.window[3]}` : null),
      row('entropy', num(o?.entropy) ? o.entropy.toFixed(2) : null),
      row('saliency max', num(o?.saliency_max) ? o.saliency_max.toFixed(2) : null),
      row('mask area', num(o?.mask_area_px) ? `${o.mask_area_px} px` : null),
      row('classes', o ? classes : null),
      row('sharpness / exposure', o ? `${o.sharpness ?? '–'} / ${o.exposure ?? '–'}` : null),
      row('quality', step.quality ? (step.quality.ok ? 'ok' : `rejected: ${step.quality.reason}`) : null),
      row('verify IoU', num(step.verify?.iou) ? `${step.verify.iou.toFixed(2)} vs L${step.verify.against_level}` : null),
      row('model / planner', step.model ? `${step.model} / ${step.planner ?? '–'}` : null),
      row('warp p50/p95', rt ? pair(rt, 'warp') : null),
      row('segment p50/p95', rt ? pair(rt, 'segment') : null),
      row('latency / cost', num(step.latency_ms) ? `${step.latency_ms} ms / $${step.cost_usd ?? '–'}` : null)));
}
