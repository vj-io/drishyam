// Benchmarks (a hand-drawn blueprint line chart + tables) and the failure plates. Each degrades to a "pending" or empty state.
import { $, h, svg, num } from './dom.js';
import { getJSON } from './api.js';
import { countUp } from './fx.js';
import { size } from './panels.js';

const SERIES = [
  ['downscale', 'downscale once', 'downscale', 'rect'],
  ['tiling', 'tiling (L2, L1, L0 grids)', 'tiling', 'diamond'],
  ['drishyam', 'Drishyam', 'drishyam', 'circle'],
];
// v1's Drishyam curve on the same images (evals.model_eval --publish): the segmenter the live demo runs since 2026-10-21,
// with its first look policy (v1-policy2, live since 2026-10-22, is #policy2-note under FIG. 4.2).
const MODEL = ['model', 'Drishyam, v1 (first look policy)', 'model', 'circle'];
// evals.harness.BUDGETS: the pixel caps of a summary's policies.drishyam points, in order
const CAPS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0];

const pending = (what, file, by = 'the evaluation harness') => h('div', { class: 'pending' },
  h('p', { class: 'pending__title' }, `${what} pending`),
  h('p', { class: 'note' }, `${file} has not been published yet. Nothing is faked here: when ${by} writes it, this window fills in by itself.`),
  h('div', { class: 'meter__track is-indeterminate', 'aria-hidden': 'true' }, h('div', { class: 'meter__fill' })));

/** The point the hero quotes: Drishyam's best recall within 30 % of the pixels. */
const headline = (policies) => points(policies?.drishyam).filter((p) => p.pixels_frac <= 0.3).pop();

const points = (list) => (Array.isArray(list) ? list.filter((p) => num(p?.pixels_frac) && num(p?.recall)).sort((a, b) => a.pixels_frac - b.pixels_frac) : []);

/** [lo, hi] when `r` is a two-number interval, else null. */
const interval = (r) => (Array.isArray(r) && r.length === 2 && r.every(num) ? r : null);
const dec = (v, d = 2) => (num(v) ? v.toFixed(d) : '–');
const pct = (v) => `${Math.round(v * 100)} %`;

/** Drishyam's point at the summary's headline pixel cap (30 %), picked by cap: v1 never reads 30 % of the pixels. */
const atCap = (s) => s?.policies?.drishyam?.[CAPS.indexOf(s?.headline_stats?.budget)];
/** At that cap: critical boxes per photo (exposed rebar, or spalling of 40,000 px² or more), and recall of one class. */
const critical = (s) => s?.agent_metrics?.false_accept_critical?.[`drishyam@${s?.headline_stats?.budget}`]?.critical_predictions / s?.n_images;
const classRecall = (s, cls) => s?.per_class?.[`drishyam@${s?.headline_stats?.budget}`]?.[cls];

/** v1 held out, against the classical engine on the same photos (`cl`), with every number stated before the runs
 * beside the favourable one: chance, IoU-only, the critical boxes a person must check, and on the synthetic photos
 * (`v1s`, `cls`) the classes it misses. Null without v1's point. */
function v1Facts(v1, cl, v1s, cls) {
  const p = atCap(v1);
  const c = atCap(cl);
  if (!p || !num(p.recall)) return null;
  const ci = interval(p.recall_ci);
  const out = [`recall ${dec(p.recall)}${ci ? ` (95 % CI ${dec(ci[0])}–${dec(ci[1])})` : ''} at the ${pct(v1.headline_stats.budget)} pixel cap${num(c?.recall) ? `, against the classical engine’s ${dec(c.recall)}` : ''}.`];
  if (num(p.chance_recall)) out.push(` Its boxes at random places score ${dec(p.chance_recall)}${num(p.recall_iou_only) && num(p.chance_recall_iou_only) ? `, and with IoU-only matching it finds ${dec(p.recall_iou_only)} against ${dec(p.chance_recall_iou_only)} by chance` : ''}.`);
  if (num(critical(v1))) out.push(` It sends ${dec(critical(v1))} critical boxes per photo to a person${num(critical(cl)) ? `, against the classical engine’s ${dec(critical(cl))}` : ''}.`);
  const [er, ef] = ['exposed_rebar', 'efflorescence'].map((k) => classRecall(v1s, k));
  if (num(er) && num(ef)) {
    const [cer, cef] = ['exposed_rebar', 'efflorescence'].map((k) => classRecall(cls, k));
    out.push(` On the synthetic photos it finds ${dec(er)} of the exposed rebar and ${dec(ef)} of the efflorescence${num(cer) && num(cef) ? `, where the classical engine finds ${dec(cer)} and ${dec(cef)}` : ''}.`);
  }
  out.push(' Measured with the governor off, the 512-px FP32 net on every look; live, the v1 engine keeps that net unless a step takes over 1.5 s.');
  return out.join('');
}

function marker(kind, x, y, cls) {
  const a = { class: `mk mk--${cls}` };
  if (kind === 'rect') return svg('rect', { ...a, x: x - 5.5, y: y - 5.5, width: 11, height: 11 });
  if (kind === 'diamond') return svg('polygon', { ...a, points: `${x},${y - 7.5} ${x + 7.5},${y} ${x},${y + 7.5} ${x - 7.5},${y}` });
  return svg('circle', { ...a, cx: x, cy: y, r: 6 });
}

const NARROW = matchMedia('(max-width: 640px)');

function chart(policies, series = SERIES, iou = false) {
  // Phones get a smaller sheet, so the lettering is drawn at a legible size instead of shrinking with the chart.
  const [W, H, L, R, T, B] = NARROW.matches ? [360, 300, 54, 14, 14, 50] : [640, 380, 58, 20, 18, 58];
  const PW = W - L - R;
  // Overlapping tiles can process more than 100% of the pixels. Then the x axis is broken after 75% (a drafting
  // break mark), so the range where the policies actually differ gets most of the width.
  const xmax = Math.max(1, ...series.flatMap(([key]) => points(policies[key]).map((p) => p.pixels_frac)));
  const BRK = 0.75;
  const broken = xmax > 1;
  const X = (v) => (!broken ? L + (v / xmax) * PW
    : v <= BRK ? L + (v / BRK) * PW * 0.8 : L + PW * 0.9 + ((v - BRK) / (xmax - BRK)) * PW * 0.1);
  const Y = (v) => T + (1 - v) * (H - T - B);
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': `Line chart of recall against the fraction of full-resolution pixels processed, for ${series.length} policies${broken ? `; the pixel axis is broken between 75% and ${Math.round(xmax * 100)}%` : ''}. A data table follows.` });
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
  const lines = svg('g');  // every line under every marker, so a later series' halo never hides an earlier point
  plot.append(lines);
  root.append(plot);
  for (const [key, name, cls, kind] of series) {
    const pts = points(policies[key]);
    if (pts.length > 1) {
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.pixels_frac)} ${Y(p.recall)}`).join(' ');
      lines.append(svg('path', { class: 'chart__halo', d }), svg('path', { class: `chart__line chart__line--${cls}`, d }));
    }
    for (const p of pts) {
      const x = X(p.pixels_frac);
      const ci = interval(p.recall_ci);
      // 95 % bootstrap interval as a capped bar, and the policy's chance level as a dotted tick under it
      if (ci) plot.append(svg('path', { class: `mk mk--${cls}`, opacity: 0.55, d: `M${x} ${Y(ci[0])}V${Y(ci[1])}M${x - 4} ${Y(ci[0])}h8M${x - 4} ${Y(ci[1])}h8` }));
      if (num(p.chance_recall)) plot.append(svg('path', { class: `mk mk--${cls}`, 'stroke-dasharray': '2 3', d: `M${x - 9} ${Y(p.chance_recall)}h18` }));
      // IoU-only recall (no contains-the-centre clause) as a cross, where the two matching rules are co-primary
      if (iou && num(p.recall_iou_only)) {
        const y = Y(p.recall_iou_only);
        plot.append(svg('path', { class: `mk mk--${cls}`, d: `M${x - 5} ${y - 5}l10 10M${x - 5} ${y + 5}l10 -10` },
          svg('title', {}, `${name}, IoU-only: recall ${p.recall_iou_only.toFixed(2)}`)));
      }
      const m = marker(kind, x, Y(p.recall), cls);
      m.append(svg('title', {}, `${name}: ${(p.pixels_frac * 100).toFixed(1)}% of pixels, recall ${p.recall.toFixed(2)}${ci ? ` (95% CI ${ci[0].toFixed(2)}–${ci[1].toFixed(2)})` : ''}${num(p.chance_recall) ? `, chance ${p.chance_recall.toFixed(2)}` : ''}${num(p.usd) ? `, $${p.usd}/image` : ''}`));
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

const legend = (series) => h('ul', { class: 'legend' }, series.map(([, name, cls]) => h('li', {}, h('i', { class: `sw sw--${cls}` }), name)));

/** What the bars, dotted ticks and crosses mean, when the file has them. */
const marks = (policies, series, iou) => (series.some(([key]) => points(policies[key]).some((p) => interval(p.recall_ci)))
  ? h('p', { class: 'note' }, 'Bars: 95 % interval (bootstrap over images). Dotted ticks: chance, the same policy’s boxes moved to random places in the photo.',
    iou ? ' Crosses: recall when a match needs IoU ≥ 0.3 (no contains-the-centre clause).' : '')
  : null);

function dataTable(policies, series) {
  const ci = (r) => (interval(r) ? `${r[0].toFixed(2)}–${r[1].toFixed(2)}` : '–');
  const rows = series.flatMap(([key, name]) => points(policies[key]).map((p) => h('tr', {},
    h('th', { scope: 'row' }, name),
    [`${(p.pixels_frac * 100).toFixed(1)}%`, p.recall.toFixed(2), ci(p.recall_ci), dec(p.recall_iou_only), dec(p.chance_recall),
      dec(p.precision, 3), dec(p.fp_per_image, 1), dec(p.boxes_per_image, 1), dec(p.hits_per_matched_gt), num(p.usd) ? `$${p.usd}` : '–'].map((v) => h('td', {}, v)))));
  const cols = ['policy', 'pixels', 'recall', '95 % CI', 'IoU-only recall', 'chance recall', 'precision', 'false boxes / image', 'boxes / image', 'hits per matched box', '$ / image'];
  return h('details', { class: 'data' }, h('summary', {}, 'Chart data as a table'),
    h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Chart data' },
      h('table', { class: 'tbl tbl--tight' }, h('thead', {}, h('tr', {}, cols.map((t) => h('th', { scope: 'col' }, t)))), h('tbody', {}, rows))));
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

/** How Drishyam compares with one downscale, from the paired 95 % interval of the recall ratio. */
function versusDownscale(hs) {
  const ci = interval(hs?.recall_ratio_vs_downscale_ci);
  if (!ci) return '';
  return ci[0] > 1 ? '; it beats one downscale' : ci[1] < 1 ? '; one downscale does better' : '; one downscale does as well';
}

/** Hero chips, each a measured fact with its caveat in 25 words or fewer, every pixel figure a share of the image's own
 * pixels. #stat-real, the first chip, holds v1 on CODEBRIM (`v1.real`, its summary) when published, else `real` (the
 * classical runs); the other three are the classical engine's. Returns the long versions, for FIG. 4.1's notes. */
function heroChips(sum, real, v1) {
  const long = [];
  // Exhaustive tiling (the most pixels) is the PRD's reference point for the headline ratio.
  const tiling = points(sum.policies?.tiling).pop();
  const hs = sum.headline_stats ?? {};
  const fill = (id, key, value, opts, sub) => {
    const el = $(`#${id}`);
    if (!el) return;  // a slot the page does not have (yet)
    if (!$('.stat__v', el)) {
      el.classList.add('stat');
      el.replaceChildren(h('span', { class: 'stat__k' }), h('i', { class: 'stat__dim', 'aria-hidden': 'true' }), h('b', { class: 'stat__v' }), h('span', { class: 'stat__s' }));
    }
    if (key) $('.stat__k', el).textContent = key;
    countUp($('.stat__v', el), value, opts);
    $('.stat__s', el).textContent = sub;
    el.classList.add('is-measured');
  };
  const drishyam = headline(sum.policies);
  if (drishyam && tiling?.recall > 0) {
    const ratio = num(hs.recall_ratio_vs_tiling) ? hs.recall_ratio_vs_tiling : drishyam.recall / tiling.recall;
    const target = hs.target?.recall_ratio;
    const ds = interval(hs.recall_ratio_vs_downscale_ci);
    const vsDown = !ds ? '' : ds[0] > 1 ? ' It beats one downscaled pass' : ds[1] < 1 ? ' One downscaled pass does better' : ' One downscaled pass does about as well';
    const key = `Classical engine, synthetic test set${num(sum.n_images) ? ` (${sum.n_images} photos)` : ''}`;
    const met = num(target) ? `${pct(target)}: ${ratio >= target ? 'met' : 'missed'}.` : '';
    fill('stat-recall', key, Math.round(100 * ratio), { suffix: ' %' },
      `of what checking every tile finds, at ${pct(drishyam.pixels_frac)} of the pixels (tiling ${pct(tiling.pixels_frac)}).`
      + `${met ? ` Target ${met}` : ''}${vsDown ? `${vsDown.replace('downscaled pass', 'downscale')}.` : ''}`);
    long.push(`${key}: Drishyam finds ${Math.round(100 * ratio)} % of what checking every tile finds, reading ${pct(drishyam.pixels_frac)} of the image's pixels${num(hs.budget) ? ` under the ${pct(hs.budget)} pixel cap` : ''} (tiling reads ${pct(tiling.pixels_frac)}${tiling.pixels_frac > 1 ? ': its tiles overlap' : ''}).`
      + `${met ? ` Target was ${met}` : ''}`
      + `${vsDown ? `${vsDown} (${dec(hs.recall_ratio_vs_downscale)}×, 95 % CI ${dec(ds[0])}–${dec(ds[1])}).` : ''}`);
  }
  // v1's chip, first and two lines long: its interval, chance and the classical engine's recall (its costs and misses
  // are in FIG. 4.2's note)
  const v1p = atCap(v1?.real);
  const v1ci = interval(v1p?.recall_ci);
  const rd = headline(real?.policies);
  if (num(v1p?.recall)) {
    fill('stat-real', `Live engine (v1), real bridge photos (CODEBRIM${num(v1.real.n_images) ? `, ${v1.real.n_images}` : ''})`,
      v1p.recall, { decimals: 2, prefix: 'recall ' },
      `at the ${pct(v1.real.headline_stats.budget)} pixel cap${v1ci ? ` (CI ${dec(v1ci[0])}–${dec(v1ci[1])})` : ''}`
      + `${num(v1p.chance_recall) ? `; chance ${dec(v1p.chance_recall)}` : ''}`
      + `${num(atCap(real)?.recall) ? `, classical engine ${dec(atCap(real).recall)}` : ''}.`);
  } else if (rd) {
    const others = [...points(real.policies.downscale), ...points(real.policies.tiling)].map((p) => p.recall);
    const [lo, hi] = [Math.min(rd.recall, ...others), Math.max(rd.recall, ...others)];
    const mid = (lo + hi) / 2;
    const plain = hi - lo <= 0.05 && mid > 0;  // every policy within 0.05 of the others: one plain number for all of them
    const u = (v) => (plain && v > 0 ? `about 1 in ${Math.round(1 / v)}` : dec(v));  // and the card keeps that one unit
    const chance = num(rd.chance_recall) ? ` Its boxes at random places ${plain ? 'find' : 'score'} ${u(rd.chance_recall)}.` : '';
    const strict = num(rd.recall_iou_only) && num(rd.chance_recall_iou_only)
      ? ` With strict IoU matching it ${plain ? 'finds' : 'is'} ${u(rd.recall_iou_only)} against ${u(rd.chance_recall_iou_only)} by chance.` : '';
    const why = ' The classical segmenter is the bottleneck, not where it looks.';
    const key = `Real bridge photos (CODEBRIM${num(real.n_images) ? `, ${real.n_images}` : ''})`;
    if (plain) fill('stat-real', key, Math.round(1 / mid), { prefix: 'about 1 in ' }, `defects found by every method, ours included.${chance}${strict}${why}`);
    else fill('stat-real', key, rd.recall, { decimals: 2, prefix: 'recall ' }, `ours; the other methods ${dec(Math.min(...others))}–${dec(Math.max(...others))}.${chance}${strict}${why}`);
  }
  // Two caps, each named: at a 100 % pixel cap (none below the whole image) the stop rule (value per ms) ends the job
  // by itself; at the headline's cap, the cap binds first. by_budget is keyed by the pixel cap ("1.0", "0.3").
  const am = sum.agent_metrics ?? {};
  const uncapped = points(sum.policies?.drishyam).pop();
  const total = (halts) => Object.values(halts).reduce((a, b) => a + b, 0);
  const halts = am.by_budget?.['1.0']?.halt_causes;
  const capped = am.by_budget?.[String(hs.budget)]?.halt_causes;
  // every halt cause is counted, so the numbers add up: "exhausted" is a job that had looked at every candidate
  const other = (h) => Object.entries(h).filter(([k, n]) => n && !['stop-rule', 'cap-hit'].includes(k))
    .map(([k, n]) => `${n} ${k === 'exhausted' ? 'had looked at every candidate' : `stopped on ${k}`}`);
  if (uncapped && halts) {
    const stops = halts['stop-rule'] ?? 0;
    const rest = [halts['cap-hit'] && `${halts['cap-hit']} hit a time, $ or step cap`, ...other(halts)].filter(Boolean);
    const at = capped && [`a cap ended ${capped['cap-hit'] ?? 0} of ${total(capped)} first`, `the stop rule ${capped['stop-rule'] ?? 0}`, ...other(capped)];
    fill('stat-cost', `Classical engine, no pixel cap (100 %): ${stops * 2 >= total(halts) ? 'stops on its own' : 'the stop rule'}`, stops, { suffix: ` of ${total(halts)} jobs` },
      `ended by the stop rule, when the next look was worth less than it cost: ${pct(uncapped.pixels_frac)} of the pixels on average (FIG. 4.1).`);
    long.push(`Classical engine, no pixel cap (100 %): the stop rule ended ${stops} of ${total(halts)} jobs when the next look was worth less than it cost, at ${pct(uncapped.pixels_frac)} of the image's pixels on average`
      + `${rest.length ? `; the other ${rest.join(', ')}` : ''}.`
      + `${at ? ` At the ${pct(hs.budget)} pixel cap of the headline: ${at.slice(0, -1).join(', ')}, and ${at.at(-1)}.` : ''}`);
  }
  // Agentic evidence: injected planner and governor faults, and how many the agent recovered from.
  const fired = (am.planner_recovery?.timeout?.fired ?? 0) + (am.planner_recovery?.malformed?.fired ?? 0) + (am.governor_response?.fired ?? 0);
  const ok = (am.planner_recovery?.recovered_total ?? 0) + (am.governor_response?.recovered ?? 0);
  if (fired > 0) {
    fill('stat-ops', 'Classical engine, failure drills', ok, { suffix: ` / ${fired} recovered` },
      'planner timeouts, malformed replies and slow-segmenter faults, injected on purpose');
  }
  return long.map((t) => h('p', { class: 'note' }, t));
}

/** The machine was busy: its 1-minute load average was at or above its logical CPU count. */
const busy = (prov) => {
  const cpus = Number(/(\d+) logical CPUs/.exec(String(prov?.machine ?? ''))?.[1]);
  return Array.isArray(prov?.load_avg) && cpus > 0 && prov.load_avg[0] >= cpus;
};

/** Where a summary's jobs ran, in plain words. The published runs ran on a busy x86 laptop, charged the fixed costs
 * (and showing the micro-benchmarks) of GitHub's arm64 runner; neither is Graviton. */
function place(prov) {
  const machine = String(prov?.machine ?? '');
  if (!/^Darwin .*x86_64/.test(machine)) return machine;
  const cpu = String(prov.cpu ?? '').replace(/\((R|TM)\)| Core| CPU.*$/g, '');  // "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz" -> "Intel i9-9880H"
  const arm = /micro arm linux-aarch64/.test(machine) ? '; costs and micro-benchmarks from GitHub’s arm64 runner' : '';
  return `local x86 laptop${cpu ? ` (${cpu})` : ''}${busy(prov) ? ', under load' : ''}${arm}`;
}

/** `model` (optional): {sum, text, policy2}, v1's summary on the same images, drawn dashed, what its note says, and where
 * v1-policy2's own read is (the live look policy: #policy2-note under FIG. 4.2).
 * `extra` (optional) adds lines to the folded provenance block. */
function renderBench(sum, id, file, note, model, iou = false, extra = null) {
  const root = $(`#${id}`);
  if (!sum?.policies) return root.replaceChildren(pending('benchmarks', file));
  const prov = Object.entries(sum.provenance ?? {});
  const micro = Array.isArray(sum.micro) && sum.micro.length;
  const where = place(sum.provenance);
  const withModel = points(model?.sum?.policies?.drishyam).length > 0;
  const series = withModel ? [...SERIES, MODEL] : SERIES;
  const policies = withModel ? { ...sum.policies, model: model.sum.policies.drishyam } : sum.policies;
  const plot = h('div', {}, chart(policies, series, iou), legend(series), dataTable(policies, series));
  NARROW.addEventListener('change', () => plot.firstElementChild.replaceWith(chart(policies, series, iou)));
  root.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'dataset' }, h('span', { class: 'dataset__k' }, 'dataset'), h('b', {}, String(sum.dataset ?? 'unlabelled'))),
    h('div', { class: 'bench' },
      plot,
      h('div', { class: 'stack' },  // beside the chart: what its lines and marks mean, then the notes
        withModel ? h('p', { class: 'note' }, `Dashed red: Drishyam with v1, the dacl10k-trained segmenter the live demo runs since 2026-10-21, on the same images, with its first look policy; the live policy since 2026-10-22, v1-policy2, is summarised ${model.policy2}. The other lines use the classical engine.${model.text ? ` v1 held out: ${model.text}` : ''}`) : null,
        marks(policies, series, iou),
        note?.(sum) ?? null,
        micro || prov.length ? h('details', { class: 'data' },
          h('summary', {}, `${micro ? 'Micro-benchmarks and provenance' : 'Provenance'}${where ? `: ${where}` : ''}`),
          h('div', { class: 'stack' },
            micro ? microTable(sum.micro) : null,
            prov.length ? h('div', { class: 'prov' }, h('h4', { class: 'label' }, 'provenance'),
              h('dl', { class: 'kv' }, prov.map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, String(v))))), extra) : null)) : null))));
  return true;
}

/** What the real-photo chart says on its own: every policy's recall is capped by the segmenter there. */
function realNote(sum) {
  const best = Math.max(...SERIES.flatMap(([key]) => points(sum.policies[key]).map((p) => p.recall)));
  const strict = Object.values(sum.agent_metrics?.box_convention?.recall_iou_only?.components ?? {}).filter(num);
  if (!Number.isFinite(best)) return null;
  const all = SERIES.flatMap(([key]) => points(sum.policies[key]));
  const top = (f) => Math.max(...all.map((p) => p[f]).filter(num));
  const chance = top('chance_recall');
  return h('p', { class: 'note' }, `With the classical segmenter, the best recall of any policy on these photos is ${best.toFixed(2)}`,
    strict.length ? ` (${Math.max(...strict).toFixed(2)} when a match needs IoU ≥ 0.3, the stricter rule, reported alongside)` : '',
    Number.isFinite(chance) ? `; each policy’s own boxes at random places already score up to ${chance.toFixed(2)} (${top('chance_recall_iou_only').toFixed(2)} with IoU only), so much of the lenient score is chance` : '',
    '. Here that segmenter, tuned on synthetic images, limits recall more than where the agent looks.');
}

/** FIG. 4.1's footnotes: the CODEBRIM-sized synthetic set and the run on measured wall time, when published. */
function synthNotes(large, live) {
  const at = (s) => s?.policies && headline(s.policies);
  const out = [];
  const ld = at(large);
  const ls = points(large?.policies?.downscale)[0];
  const lt = points(large?.policies?.tiling).pop();
  if (ld && ls && lt) {
    const cmp = versusDownscale(large.headline_stats);
    const best = points(large.policies.tiling).reduce((a, b) => (b.recall > a.recall ? b : a));
    out.push(h('p', { class: 'note' }, `At CODEBRIM’s size, on ${String(large.dataset ?? 'the large synthetic set')}: Drishyam found ${ld.recall.toFixed(2)} at ${pct(ld.pixels_frac)} of the pixels, `
      + `one downscale ${ls.recall.toFixed(2)} at ${pct(ls.pixels_frac)}, exhaustive tiling ${lt.recall.toFixed(2)}`
      + `${best !== lt ? ` and the best tiling grid ${best.recall.toFixed(2)} at ${pct(best.pixels_frac)}` : ''}${cmp}${cmp ? ' (by the 95 % interval of the ratio)' : ''}.`));
  }
  const vd = at(live);
  const gov = live?.agent_metrics?.governor_activity_in_eval ?? {};
  if (vd && num(gov['p95>SLA'])) {
    const acts = [gov['p95>SLA'] > 0 && `switched the model ${gov['p95>SLA']} times`,
      gov['warp-spike'] > 0 && `shrank the gaze on ${gov['warp-spike']} warp spikes`].filter(Boolean);  // a count of 0 is not an event
    out.push(h('p', { class: 'note' }, 'On measured wall time instead of fixed costs, '
      + `Drishyam found ${vd.recall.toFixed(2)} at ${pct(vd.pixels_frac)} of the pixels${acts.length ? `; the governor ${acts.join(' and ')}` : ''}`
      + '. Live timings move with the machine, so the chart uses fixed costs.'));
  }
  return out;
}

/** FIG. 4.1's folded provenance: how busy the machine was during the measured-wall-time run. */
function liveLoad(live) {
  return busy(live?.provenance) && headline(live.policies)
    ? h('p', { class: 'note' }, `Measured-wall-time run: machine load ${live.provenance.load_avg[0]} (the laptop was busy; the fixed-cost runs are the fair comparison).`)
    : null;
}

function failureImage(path) {
  const ok = typeof path === 'string' && /^\w[\w./-]*$/.test(path) && !path.includes('..');
  if (!ok) return null;
  const img = h('img', { class: 'fail__img', src: `results/${path}`, alt: '', loading: 'lazy' });
  img.addEventListener('error', () => img.replaceWith(h('div', { class: 'fail__img fail__img--none' }, 'image missing')), { once: true });
  return img;
}

/** The failure data's raw ids in the site's words: drishyam@0.3, tiling@L0 (the level words of the step detail), exposed_rebar. */
const inWords = (t) => String(t ?? '')
  .replace(/\bdrishyam@(\d*\.?\d+)/g, (_, c) => `Drishyam at a ${pct(Number(c))} pixel cap`)
  .replace(/\btiling@L(\d)\b/g, (_, L) => `tiling at ${size(Number(L))}`)
  .replace(/\b(?:exposed_rebar|not_assessed)\b/g, (m) => m.replaceAll('_', ' '));

const rules = (cases) => [...new Set(cases.map((f) => String(f.rule_that_would_catch_it ?? '')))]
  .map((rule) => h('p', { class: 'fail__rule' }, h('b', {}, 'Rule that would catch it'), inWords(rule)));

/** One plate per failure category; each case on it is a sub-figure (a), (b), ... under one shared rule. Real-photo
 * failures (CODEBRIM, text only: its licence bars sharing the photos) follow as one more plate. */
function renderFailures(list, realList) {
  const root = $('#fail');
  const items = Array.isArray(list) ? list.filter((f) => f && f.category) : [];
  const real = Array.isArray(realList) ? realList.filter((f) => f && f.what_happened) : [];
  if (!items.length) return root.replaceChildren(pending('failure gallery', 'results/failures.json'), real.length ? realFailures(real, 1) : '');
  const plates = [...Map.groupBy(items, (f) => String(f.category))];
  const n = plates.length + (real.length ? 1 : 0);
  const fig = root.closest('.win')?.querySelector('.win__fig');
  if (fig) fig.textContent = n > 1 ? `FIG. 5.1–5.${n}` : 'FIG. 5.1';
  root.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'note' }, 'Where it was wrong, and the rule that would have caught it. Kept in the open on purpose. '
      + 'These plates come from the classical engine’s published run on the 200 held-out synthetic images (2026-10-18, agent of '
      + 'commit 832299b), not from v1, the engine the live demo runs.'),
    h('div', { class: 'fails' }, plates.map(([category, cases], i) => h('article', { class: 'fail' },
      h('div', { class: 'fail__fig' }, `FIG. 5.${i + 1}`),
      h('h4', { class: 'fail__t' }, category),
      h('div', { class: 'fail__cases' }, cases.map((f, j) => {
        const img = failureImage(f.image);
        return h('figure', { class: 'fail__case' },
          img ? h('div', { class: 'fail__frame' }, img) : null,
          h('figcaption', {}, h('b', { class: 'fail__sub' }, `(${String.fromCharCode(97 + j)})`), ' ', inWords(f.what_happened)));
      })),
      rules(cases)))),
    real.length ? realFailures(real, n) : null));
}

/** FIG. 5.n: the CODEBRIM failures as text tables (the licence bars sharing the photos, not describing them). */
function realFailures(real, n) {
  const table = (label, cols, rows) => h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': label },
    h('table', { class: 'tbl tbl--tight tbl--rows' }, h('caption', {}, label),
      h('thead', {}, h('tr', {}, cols.map((t) => h('th', { scope: 'col' }, t)))),
      h('tbody', {}, rows.map(([head, ...cells]) => h('tr', {}, h('th', { scope: 'row' }, head),
        cells.map((v, i) => h('td', { 'data-label': cols[i + 1] }, v)))))));
  const byRule = [...Map.groupBy(real, (f) => String(f.category ?? ''))];
  return h('details', { class: 'data' },
    h('summary', {}, `FIG. 5.${n} · Real-photo failures (described; photos not redistributable)`),
    h('div', { class: 'stack' },
      h('p', { class: 'note' }, `${real.length} cases from the classical engine’s CODEBRIM evaluation (the same published runs, not v1), written out because its licence does not allow sharing the photos.`),
      table('Real-photo failures', ['photo', 'class', 'what happened'],
        real.map((f) => [String(f.id ?? ''), inWords(f.class ?? '–'), inWords(f.what_happened)])),
      table('Rules that would catch them', ['failure', 'rule that would catch it'],
        byRule.map(([category, cases]) => [category, inWords(cases[0].rule_that_would_catch_it)]))));
}

export async function initResults() {
  // v1's held-out summaries on the same images are drawn as the dashed model line. COOL was not run, so FIG. 4.3 is
  // static text.
  const [sum, real, fails, realFails, large, live, v1synth, v1real] = await Promise.all(
    ['results/summary.json', 'results/codebrim/summary.json', 'results/failures.json',
      'results/codebrim/failures_text.json', 'results/large/summary.json', 'results/live/summary.json',
      'results/synth_dacl10k/summary.json', 'results/codebrim_dacl10k/summary.json'].map(getJSON));
  const v1 = { synth: v1synth, real: v1real };
  const long = sum?.policies ? heroChips(sum, real, v1) : [];  // the hero cards' long versions are FIG. 4.1's first notes
  renderBench(sum, 'bench', 'results/summary.json', () => [...long, ...synthNotes(large, live)],
    { sum: v1synth, text: v1Facts(v1synth, sum, v1synth, sum), policy2: 'under FIG. 4.2' }, false, liveLoad(live));
  renderBench(real, 'bench-real', 'results/codebrim/summary.json', realNote,
    { sum: v1real, text: v1Facts(v1real, real, v1synth, sum), policy2: 'in the note below' }, true);
  renderFailures(fails, realFails);
}
