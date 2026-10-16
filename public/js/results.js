// Benchmarks (hand-written SVG chart + tables) and the failure gallery. Both degrade to a "pending" state.
import { $, h, svg, num } from './dom.js';
import { getJSON } from './api.js';
import { countUp } from './fx.js';

const SERIES = [
  ['downscale', 'downscale once', 'cobalt', 'rect'],
  ['tiling', 'exhaustive tiling', 'pink', 'diamond'],
  ['drishyam', 'Drishyam', 'lime', 'circle'],
];

const pending = (what, file) => h('div', { class: 'pending' },
  h('p', { class: 'pending__title' }, `${what} pending`),
  h('p', { class: 'note' }, `${file} has not been published yet. Nothing is faked here: when the evaluation harness writes it, this window fills in by itself.`),
  h('div', { class: 'meter__track is-indeterminate', 'aria-hidden': 'true' }, h('div', { class: 'meter__fill' })));

const points = (list) => (Array.isArray(list) ? list.filter((p) => num(p?.pixels_frac) && num(p?.recall)).sort((a, b) => a.pixels_frac - b.pixels_frac) : []);

function marker(kind, x, y, cls) {
  const a = { class: `mk mk--${cls}` };
  if (kind === 'rect') return svg('rect', { ...a, x: x - 5.5, y: y - 5.5, width: 11, height: 11 });
  if (kind === 'diamond') return svg('polygon', { ...a, points: `${x},${y - 7.5} ${x + 7.5},${y} ${x},${y + 7.5} ${x - 7.5},${y}` });
  return svg('circle', { ...a, cx: x, cy: y, r: 6 });
}

function chart(policies) {
  const W = 640, H = 380, L = 58, R = 20, T = 18, B = 58;
  const X = (v) => L + v * (W - L - R);
  const Y = (v) => T + (1 - v) * (H - T - B);
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': 'Line chart of recall against the fraction of full-resolution pixels processed, for three policies. A data table follows.' });
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    root.append(
      svg('line', { class: 'chart__grid', x1: L, x2: W - R, y1: Y(v), y2: Y(v) }),
      svg('line', { class: 'chart__grid', x1: X(v), x2: X(v), y1: T, y2: H - B }),
      svg('text', { class: 'chart__tick', x: L - 8, y: Y(v) + 4, 'text-anchor': 'end' }, v.toFixed(1)),
      svg('text', { class: 'chart__tick', x: X(v), y: H - B + 18, 'text-anchor': 'middle' }, `${Math.round(v * 100)}%`));
  }
  root.append(
    svg('line', { class: 'chart__axis', x1: L, x2: W - R, y1: H - B, y2: H - B }),
    svg('line', { class: 'chart__axis', x1: L, x2: L, y1: T, y2: H - B }),
    svg('text', { class: 'chart__title', x: (L + W - R) / 2, y: H - 10, 'text-anchor': 'middle' }, 'fraction of full-resolution pixels processed'),
    svg('text', { class: 'chart__title', transform: `translate(14 ${(T + H - B) / 2}) rotate(-90)`, 'text-anchor': 'middle' }, 'defect recall'));
  for (const [key, name, cls, kind] of SERIES) {
    const pts = points(policies[key]);
    if (pts.length > 1) {
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.pixels_frac)} ${Y(p.recall)}`).join(' ');
      root.append(svg('path', { class: 'chart__halo', d }), svg('path', { class: `chart__line chart__line--${cls}`, d }));
    }
    for (const p of pts) {
      const m = marker(kind, X(p.pixels_frac), Y(p.recall), cls);
      m.append(svg('title', {}, `${name}: ${(p.pixels_frac * 100).toFixed(1)}% of pixels, recall ${p.recall.toFixed(2)}${num(p.usd) ? `, $${p.usd}/image` : ''}`));
      root.append(m);
    }
  }
  return root;
}

const legend = () => h('ul', { class: 'legend' }, SERIES.map(([, name, cls]) => h('li', {}, h('i', { class: `sw sw--${cls}` }), name)));

function dataTable(policies) {
  const rows = SERIES.flatMap(([key, name]) => points(policies[key]).map((p) => h('tr', {},
    h('th', { scope: 'row' }, name), h('td', {}, `${(p.pixels_frac * 100).toFixed(1)}%`), h('td', {}, p.recall.toFixed(2)), h('td', {}, num(p.usd) ? `$${p.usd}` : '–'))));
  return h('details', { class: 'data' }, h('summary', {}, 'Chart data as a table'),
    h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Chart data' },
      h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, ['policy', 'pixels', 'recall', '$ / image'].map((t) => h('th', { scope: 'col' }, t)))), h('tbody', {}, rows))));
}

function microTable(micro) {
  const rows = micro.filter((m) => m && m.op).map((m) => h('tr', {},
    h('th', { scope: 'row' }, String(m.op)), h('td', {}, String(m.size ?? '')), h('td', { class: 'n' }, String(m.p50_ms ?? '')), h('td', { class: 'n' }, String(m.p95_ms ?? ''))));
  return h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Micro-benchmarks' },
    h('table', { class: 'tbl' }, h('caption', {}, 'micro-benchmarks (ms per call)'),
      h('thead', {}, h('tr', {}, ['op', 'size', 'p50', 'p95'].map((t, i) => h('th', { scope: 'col', class: i > 1 ? 'n' : null }, t)))), h('tbody', {}, rows)));
}

function heroChips(sum) {
  const drishyam = points(sum.policies?.drishyam).filter((p) => p.pixels_frac <= 0.3).pop();
  const tiling = points(sum.policies?.tiling).reduce((best, p) => (!best || p.recall > best.recall ? p : best), null);
  const fill = (id, value, opts, sub) => {
    const el = $(`#${id}`);
    countUp($('.stat__v', el), value, opts);
    $('.stat__s', el).textContent = sub;
  };
  if (drishyam && tiling?.recall > 0) {
    fill('stat-recall', drishyam.recall / tiling.recall, { decimals: 2, suffix: '× tiling' }, `recall at ${(drishyam.pixels_frac * 100).toFixed(0)}% of the pixels`);
  }
  if (drishyam && num(drishyam.usd)) {
    fill('stat-cost', drishyam.usd, { decimals: 4, prefix: '$' }, num(tiling?.usd) ? `per image, vs $${tiling.usd} for tiling` : 'per image, modelled');
  }
  const slow = (sum.micro ?? []).filter((m) => num(m?.p95_ms)).reduce((a, m) => (!a || m.p95_ms > a.p95_ms ? m : a), null);
  if (slow) fill('stat-ops', slow.p95_ms, { decimals: 1, suffix: ' ms' }, `${slow.op} ${slow.size ?? ''} p95`.trim());
}

function renderBench(sum) {
  const root = $('#bench');
  if (!sum?.policies) return root.replaceChildren(pending('benchmarks', 'results/summary.json'));
  const prov = Object.entries(sum.provenance ?? {});
  root.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'dataset' }, h('span', { class: 'dataset__k' }, 'dataset'), h('b', {}, String(sum.dataset ?? 'unlabelled'))),
    h('div', { class: 'bench' },
      h('div', {}, chart(sum.policies), legend(), dataTable(sum.policies)),
      h('div', { class: 'stack' },
        Array.isArray(sum.micro) && sum.micro.length ? microTable(sum.micro) : null,
        prov.length ? h('div', { class: 'prov' }, h('h4', { class: 'label' }, 'provenance'),
          h('dl', { class: 'kv' }, prov.map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, String(v)))))) : null))));
  heroChips(sum);
}

function failureImage(path) {
  const ok = typeof path === 'string' && /^\w[\w./-]*$/.test(path) && !path.includes('..');
  if (!ok) return null;
  const img = h('img', { class: 'fail__img', src: `results/${path}`, alt: '', loading: 'lazy' });
  img.addEventListener('error', () => img.replaceWith(h('div', { class: 'fail__img fail__img--none' }, 'image missing')), { once: true });
  return img;
}

function renderFailures(list) {
  const root = $('#fail');
  const items = Array.isArray(list) ? list.filter((f) => f && f.category) : [];
  if (!items.length) return root.replaceChildren(pending('failure gallery', 'results/failures.json'));
  root.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'note' }, 'Where it was wrong, and the rule that would have caught it. Kept in the open on purpose.'),
    h('div', { class: 'fails' }, items.map((f) => h('article', { class: 'fail' },
      failureImage(f.image),
      h('span', { class: 'chip chip--unsure' }, String(f.category)),
      h('p', {}, String(f.what_happened ?? '')),
      h('p', { class: 'fail__rule' }, h('b', {}, 'Rule that would catch it: '), String(f.rule_that_would_catch_it ?? '')))))));
}

export async function initResults() {
  const [sum, fails] = await Promise.all([getJSON('results/summary.json'), getJSON('results/failures.json')]);
  renderBench(sum);
  renderFailures(fails);
}
