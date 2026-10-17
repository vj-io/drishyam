// Benchmarks (a hand-drawn blueprint line chart + tables) and the failure plates. Both degrade to a "pending" state.
import { $, h, svg, num } from './dom.js';
import { getJSON } from './api.js';
import { countUp } from './fx.js';

const SERIES = [
  ['downscale', 'downscale once', 'downscale', 'rect'],
  ['tiling', 'exhaustive tiling', 'tiling', 'diamond'],
  ['drishyam', 'Drishyam', 'drishyam', 'circle'],
];

const pending = (what, file, by = 'the evaluation harness') => h('div', { class: 'pending' },
  h('p', { class: 'pending__title' }, `${what} pending`),
  h('p', { class: 'note' }, `${file} has not been published yet. Nothing is faked here: when ${by} writes it, this window fills in by itself.`),
  h('div', { class: 'meter__track is-indeterminate', 'aria-hidden': 'true' }, h('div', { class: 'meter__fill' })));

/** The point the hero quotes: Drishyam's best recall within 30 % of the pixels. */
const headline = (policies) => points(policies?.drishyam).filter((p) => p.pixels_frac <= 0.3).pop();

const points = (list) => (Array.isArray(list) ? list.filter((p) => num(p?.pixels_frac) && num(p?.recall)).sort((a, b) => a.pixels_frac - b.pixels_frac) : []);

function marker(kind, x, y, cls) {
  const a = { class: `mk mk--${cls}` };
  if (kind === 'rect') return svg('rect', { ...a, x: x - 5.5, y: y - 5.5, width: 11, height: 11 });
  if (kind === 'diamond') return svg('polygon', { ...a, points: `${x},${y - 7.5} ${x + 7.5},${y} ${x},${y + 7.5} ${x - 7.5},${y}` });
  return svg('circle', { ...a, cx: x, cy: y, r: 6 });
}

const NARROW = matchMedia('(max-width: 640px)');

function chart(policies) {
  // Phones get a smaller sheet, so the lettering is drawn at a legible size instead of shrinking with the chart.
  const [W, H, L, R, T, B] = NARROW.matches ? [360, 300, 54, 14, 14, 50] : [640, 380, 58, 20, 18, 58];
  const PW = W - L - R;
  // Overlapping tiles can process more than 100% of the pixels. Then the x axis is broken after 75% (a drafting
  // break mark), so the range where the policies actually differ gets most of the width.
  const xmax = Math.max(1, ...SERIES.flatMap(([key]) => points(policies[key]).map((p) => p.pixels_frac)));
  const BRK = 0.75;
  const broken = xmax > 1;
  const X = (v) => (!broken ? L + (v / xmax) * PW
    : v <= BRK ? L + (v / BRK) * PW * 0.8 : L + PW * 0.9 + ((v - BRK) / (xmax - BRK)) * PW * 0.1);
  const Y = (v) => T + (1 - v) * (H - T - B);
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': `Line chart of recall against the fraction of full-resolution pixels processed, for three policies${broken ? `; the pixel axis is broken between 75% and ${Math.round(xmax * 100)}%` : ''}. A data table follows.` });
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    root.append(
      svg('line', { class: 'chart__grid', x1: L, x2: W - R, y1: Y(v), y2: Y(v) }),
      svg('text', { class: 'chart__tick', x: L - 8, y: Y(v) + 4, 'text-anchor': 'end' }, v.toFixed(1)));
  }
  [0, 0.25, 0.5, 0.75, xmax].forEach((v, i, all) => root.append(
    svg('line', { class: 'chart__grid', x1: X(v), x2: X(v), y1: T, y2: H - B }),
    svg('text', { class: 'chart__tick', x: X(v), y: H - B + 18, 'text-anchor': i === all.length - 1 ? 'end' : 'middle' }, `${Math.round(v * 100)}%`)));
  const axisY = H - B;
  const gap = [L + PW * 0.84, L + PW * 0.87];  // clear of the 75% label
  root.append(
    ...(broken
      ? [svg('line', { class: 'chart__axis', x1: L, x2: gap[0], y1: axisY, y2: axisY }),
        svg('line', { class: 'chart__axis', x1: gap[1], x2: W - R, y1: axisY, y2: axisY }),
        svg('path', { class: 'chart__axis', d: gap.map((x) => `M${x - 3} ${axisY + 6} L${x + 3} ${axisY - 6}`).join(' ') })]
      : [svg('line', { class: 'chart__axis', x1: L, x2: W - R, y1: axisY, y2: axisY })]),
    svg('line', { class: 'chart__axis', x1: L, x2: L, y1: T, y2: H - B }),
    svg('text', { class: 'chart__title', x: (L + W - R) / 2, y: H - 10, 'text-anchor': 'middle' }, NARROW.matches ? 'full-res pixels processed' : 'fraction of full-resolution pixels processed'),
    svg('text', { class: 'chart__title', transform: `translate(${NARROW.matches ? 12 : 14} ${(T + H - B) / 2}) rotate(-90)`, 'text-anchor': 'middle' }, 'defect recall'));
  const plot = svg('g', { class: 'chart__plot' });
  root.append(plot);
  for (const [key, name, cls, kind] of SERIES) {
    const pts = points(policies[key]);
    if (pts.length > 1) {
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.pixels_frac)} ${Y(p.recall)}`).join(' ');
      plot.append(svg('path', { class: 'chart__halo', d }), svg('path', { class: `chart__line chart__line--${cls}`, d }));
    }
    for (const p of pts) {
      const m = marker(kind, X(p.pixels_frac), Y(p.recall), cls);
      m.append(svg('title', {}, `${name}: ${(p.pixels_frac * 100).toFixed(1)}% of pixels, recall ${p.recall.toFixed(2)}${num(p.usd) ? `, $${p.usd}/image` : ''}`));
      plot.append(m);
    }
  }
  const hp = headline(policies);
  if (hp) root.append(callout(X(hp.pixels_frac), Y(hp.recall), [`recall ${hp.recall.toFixed(2)}`, `at ${Math.round(hp.pixels_frac * 100)}% of pixels`], W, H - B));
  return root;
}

/** Numbered leader from a data point to a two-line note, kept inside the plot. */
function callout(x, y, lines, W, bottom) {
  const right = x < W * 0.6;
  const s = right ? 1 : -1;
  const cx = x + s * 44;
  const cy = Math.min(bottom - 30, y + 48);
  const tx = cx + s * 18;
  const nx = right ? tx + 6 : tx - 6;
  return svg('g', { class: 'chart__callout', 'aria-hidden': 'true' },
    svg('path', { class: 'chart__lead', d: `M${x} ${y} L${cx - s * 10 * 0.7} ${cy - 10 * 0.7}` }),
    svg('circle', { class: 'chart__num', cx, cy, r: 10 }),
    svg('text', { class: 'chart__numt', x: cx, y: cy + 4, 'text-anchor': 'middle' }, '1'),
    svg('path', { class: 'chart__lead', d: `M${cx + s * 10} ${cy} H${tx}` }),
    svg('text', { class: 'chart__co', x: nx, y: cy + 4, 'text-anchor': right ? 'start' : 'end' },
      lines.map((t, i) => svg('tspan', { x: nx, dy: i ? '1.3em' : 0 }, t))));
}

const legend = () => h('ul', { class: 'legend' }, SERIES.map(([, name, cls]) => h('li', {}, h('i', { class: `sw sw--${cls}` }), name)));

function dataTable(policies) {
  const rows = SERIES.flatMap(([key, name]) => points(policies[key]).map((p) => h('tr', {},
    h('th', { scope: 'row' }, name), h('td', {}, `${(p.pixels_frac * 100).toFixed(1)}%`), h('td', {}, p.recall.toFixed(2)), h('td', {}, num(p.usd) ? `$${p.usd}` : '–'))));
  return h('details', { class: 'data' }, h('summary', {}, 'Chart data as a table'),
    h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Chart data' },
      h('table', { class: 'tbl tbl--tight' }, h('thead', {}, h('tr', {}, ['policy', 'pixels', 'recall', '$ / image'].map((t) => h('th', { scope: 'col' }, t)))), h('tbody', {}, rows))));
}

function microTable(micro) {
  const ops = micro.filter((m) => m && m.op);
  const sizes = [...new Set(ops.map((m) => String(m.size ?? '')))];
  const one = sizes.length === 1;  // every op ran on one input size: say it once, in the caption
  const rows = ops.map((m) => h('tr', {},
    h('th', { scope: 'row' }, String(m.op)), one ? null : h('td', {}, String(m.size ?? '')),
    h('td', { class: 'n' }, String(m.p50_ms ?? '')), h('td', { class: 'n' }, String(m.p95_ms ?? ''))));
  return h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Micro-benchmarks' },
    h('table', { class: 'tbl tbl--tight' }, h('caption', {}, `micro-benchmarks (ms per call${one && sizes[0] ? `, ${sizes[0]}` : ''})`),
      h('thead', {}, h('tr', {}, (one ? ['op', 'p50', 'p95'] : ['op', 'size', 'p50', 'p95']).map((t) => h('th', { scope: 'col', class: t[0] === 'p' ? 'n' : null }, t)))), h('tbody', {}, rows)));
}

function heroChips(sum) {
  const drishyam = headline(sum.policies);
  // Exhaustive tiling (the most pixels) is the PRD's reference point for the headline ratio.
  const tiling = points(sum.policies?.tiling).pop();
  const fill = (id, value, opts, sub) => {
    const el = $(`#${id}`);
    countUp($('.stat__v', el), value, opts);
    $('.stat__s', el).textContent = sub;
    el.classList.add('is-measured');
  };
  if (drishyam && tiling?.recall > 0) {
    fill('stat-recall', drishyam.recall / tiling.recall, { decimals: 2, suffix: '× tiling' }, `recall at ${(drishyam.pixels_frac * 100).toFixed(0)}% of the pixels, synthetic set`);
  }
  if (drishyam && num(drishyam.usd)) {
    fill('stat-cost', drishyam.usd, { decimals: 6, prefix: '$' }, num(tiling?.usd) ? `per image, vs $${tiling.usd} for tiling` : 'per image, modelled');
  }
  // Agentic evidence, not speed: injected planner and governor faults the agent recovered from.
  const am = sum.agent_metrics ?? {};
  const fired = (am.planner_recovery?.timeout?.fired ?? 0) + (am.planner_recovery?.malformed?.fired ?? 0) + (am.governor_response?.fired ?? 0);
  const ok = (am.planner_recovery?.recovered_total ?? 0) + (am.governor_response?.recovered ?? 0);
  if (fired > 0) fill('stat-ops', ok, { decimals: 0, suffix: ` / ${fired}` }, `fired faults recovered; ${num(am.steps_with_cause_pct) ? Math.round(am.steps_with_cause_pct) : '–'} % of steps carry a cause`);
}

function renderBench(sum, id, file, note) {
  const root = $(`#${id}`);
  if (!sum?.policies) return root.replaceChildren(pending('benchmarks', file));
  const prov = Object.entries(sum.provenance ?? {});
  const plot = h('div', {}, chart(sum.policies), legend(), dataTable(sum.policies));
  NARROW.addEventListener('change', () => plot.firstElementChild.replaceWith(chart(sum.policies)));
  root.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'dataset' }, h('span', { class: 'dataset__k' }, 'dataset'), h('b', {}, String(sum.dataset ?? 'unlabelled'))),
    note?.(sum) ?? null,
    h('div', { class: 'bench' },
      plot,
      h('div', { class: 'stack' },
        Array.isArray(sum.micro) && sum.micro.length ? microTable(sum.micro) : null,
        prov.length ? h('div', { class: 'prov' }, h('h4', { class: 'label' }, 'provenance'),
          h('dl', { class: 'kv' }, prov.map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, String(v)))))) : null))));
  return true;
}

/** What the real-photo chart says on its own: every policy's recall is capped by the segmenter there. */
function realNote(sum) {
  const best = Math.max(...SERIES.flatMap(([key]) => points(sum.policies[key]).map((p) => p.recall)));
  const strict = Object.values(sum.agent_metrics?.box_convention?.recall_iou_only?.components ?? {}).filter(num);
  if (!Number.isFinite(best)) return null;
  return h('p', { class: 'note' }, `The best recall of any policy on these photos is ${best.toFixed(2)}`,
    strict.length ? ` (${Math.max(...strict).toFixed(2)} when a match needs IoU ≥ 0.3)` : '',
    '. Here the classical segmenter, tuned on synthetic images, limits recall more than where the agent looks; a trained model is the next step.');
}

const times = (v) => (num(v) ? `${v}×` : '–');
const span = (r) => (Array.isArray(r) && r.length === 2 && r.every(num) ? `${r[0]}–${r[1]}×` : null);
const sig = (v, prefix = '') => (num(v) ? `${prefix}${+v.toPrecision(3)}` : '–');
const raw = (v) => (num(v) ? String(v) : '–');

/** FIG. 4.3 and the footer banner, from results/cool.json (the --summary file of `python -m bench.compare_arms`). */
function renderCool(cool) {
  const root = $('#bench-cool');
  const arms = Object.entries(cool?.arms ?? {}).filter(([, a]) => a && typeof a === 'object');
  const e2e = cool?.end_to_end ?? {};
  if (!arms.length || !num(e2e.cool_gain)) return root.replaceChildren(pending('COOL benchmark', 'results/cool.json', 'bench.compare_arms (after the AWS run)'));
  const op = cool.op_level_geomean ?? {};
  const ex = cool.expected ?? {};
  const par = cool.parity_a_vs_b ?? {};
  const gains = [  // [label, measured, the PRD's expected range]
    ['COOL gain, whole job', times(e2e.cool_gain), span(ex.end_to_end)],
    ['COOL gain, single ops (geomean)', times(op.cool_gain), span(ex.op_level)],
    ['hardware gain, whole job', times(e2e.hardware_gain)],
    ['hardware gain, single ops (geomean)', times(op.hardware_gain)],
    ['A over C, whole job', times(e2e.a_over_c)],
    [`mask pixel agreement, A vs B${num(par.crops) ? ` (${par.crops} crops)` : ''}`, raw(par.pixel_agreement)],
    ['worst crop agreement, A vs B', raw(par.worst_crop_agreement)],
  ];
  const cols = ['instance', 'cv2', 'images/s', '$ / h', '$ / 1k images'];
  const rows = arms.map(([r, a]) => h('tr', {},
    h('th', { scope: 'row' }, a.what ? `${r}: ${a.what}` : r),
    [String(a.instance_type ?? '–'), `${String(a.cv2 ?? '–')}${a.cv2_from_opt_cool ? ', /opt/cool' : ''}`,
      sig(a.images_per_s), sig(a.usd_per_hour, '$'), sig(a.usd_per_1k_images, '$')]
      .map((v, i) => h('td', { 'data-label': cols[i], class: i > 1 ? 'n' : null }, v))));
  root.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'dataset' }, h('span', { class: 'dataset__k' }, 'measured'), h('b', {}, `COOL gain ${times(e2e.cool_gain)} on a whole job`)),
    h('div', { class: 'bench' },
      h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Three arms' },
        h('table', { class: 'tbl tbl--tight tbl--rows tbl--pairs' }, h('caption', {}, 'three arms, whole job, one worker'),
          h('thead', {}, h('tr', {}, ['arm', ...cols].map((t, i) => h('th', { scope: 'col', class: i > 2 ? 'n' : null }, t)))), h('tbody', {}, rows))),
      h('div', { class: 'prov' }, h('h4', { class: 'label' }, 'speed-ups and mask parity'),
        h('dl', { class: 'kv kv--nums' }, gains.map(([k, v, x]) => h('div', {},
          h('dt', {}, k, x ? h('span', { class: 'kv__sub' }, `PRD expects ${x}`) : null), h('dd', {}, v))))))));
  $('#cool-status').textContent = `The COOL/Graviton run has been measured (FIG. 4.3): a whole job ran at ${times(e2e.cool_gain)} the speed of the stock wheel on the same instance type${num(e2e.a_over_c) ? `, and at ${e2e.a_over_c}× the speed of the stock wheel on x86` : ''}.`;
}

function failureImage(path) {
  const ok = typeof path === 'string' && /^\w[\w./-]*$/.test(path) && !path.includes('..');
  if (!ok) return null;
  const img = h('img', { class: 'fail__img', src: `results/${path}`, alt: '', loading: 'lazy' });
  img.addEventListener('error', () => img.replaceWith(h('div', { class: 'fail__img fail__img--none' }, 'image missing')), { once: true });
  return img;
}

/** One plate per failure category; each case on it is a sub-figure (a), (b), ... under one shared rule. */
function renderFailures(list) {
  const root = $('#fail');
  const items = Array.isArray(list) ? list.filter((f) => f && f.category) : [];
  if (!items.length) return root.replaceChildren(pending('failure gallery', 'results/failures.json'));
  const plates = [...Map.groupBy(items, (f) => String(f.category))];
  const fig = root.closest('.win')?.querySelector('.win__fig');
  if (fig) fig.textContent = plates.length > 1 ? `FIG. 5.1–5.${plates.length}` : 'FIG. 5.1';
  root.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'note' }, 'Where it was wrong, and the rule that would have caught it. Kept in the open on purpose.'),
    h('div', { class: 'fails' }, plates.map(([category, cases], i) => h('article', { class: 'fail' },
      h('div', { class: 'fail__fig' }, `FIG. 5.${i + 1}`),
      h('h4', { class: 'fail__t' }, category),
      h('div', { class: 'fail__cases' }, cases.map((f, j) => {
        const img = failureImage(f.image);
        return h('figure', { class: 'fail__case' },
          img ? h('div', { class: 'fail__frame' }, img) : null,
          h('figcaption', {}, h('b', { class: 'fail__sub' }, `(${String.fromCharCode(97 + j)})`), ' ', String(f.what_happened ?? '')));
      })),
      [...new Set(cases.map((f) => String(f.rule_that_would_catch_it ?? '')))].map((rule) => h('p', { class: 'fail__rule' }, h('b', {}, 'Rule that would catch it'), rule)))))));
}

export async function initResults() {
  const [sum, real, cool, fails] = await Promise.all(
    ['results/summary.json', 'results/codebrim/summary.json', 'results/cool.json', 'results/failures.json'].map(getJSON));
  if (renderBench(sum, 'bench', 'results/summary.json')) heroChips(sum);
  renderBench(real, 'bench-real', 'results/codebrim/summary.json', realNote);
  renderCool(cool);
  renderFailures(fails);
}
