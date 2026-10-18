// Benchmarks (a hand-drawn blueprint line chart + tables) and the failure plates. Each degrades to a "pending" or empty state.
import { $, h, svg, num } from './dom.js';
import { getJSON } from './api.js';
import { countUp } from './fx.js';

const SERIES = [
  ['downscale', 'downscale once', 'downscale', 'rect'],
  ['tiling', 'tiling (L2, L1, L0 grids)', 'tiling', 'diamond'],
  ['drishyam', 'Drishyam', 'drishyam', 'circle'],
];
// The trained segmenter's Drishyam curve (evals.model_eval --publish), drawn only once its summary is published.
const MODEL = ['model', 'Drishyam, dacl10k model', 'model', 'circle'];

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

/** Hero chips, each a measured fact with its caveat, every pixel figure a share of the image's own pixels.
 * `real` (CODEBRIM) fills #stat-real when the page has that slot. */
function heroChips(sum, real) {
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
    fill('stat-recall', `Synthetic test set${num(sum.n_images) ? ` (${sum.n_images} photos)` : ''}`, Math.round(100 * ratio), { suffix: ' %' },
      `of what checking every tile finds, reading ${pct(drishyam.pixels_frac)} of the image's pixels${num(hs.budget) ? ` under the ${pct(hs.budget)} pixel cap` : ''} (tiling reads ${pct(tiling.pixels_frac)}${tiling.pixels_frac > 1 ? ': its tiles overlap' : ''}).`
      + `${num(target) ? ` Target was ${pct(target)}: ${ratio >= target ? 'met' : 'missed'}.` : ''}`
      + `${vsDown ? `${vsDown} (${dec(hs.recall_ratio_vs_downscale)}×, 95 % CI ${dec(ds[0])}–${dec(ds[1])}).` : ''}`);
  }
  const rd = headline(real?.policies);
  if (rd) {
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
  if (uncapped && halts) {
    const stops = halts['stop-rule'] ?? 0;
    fill('stat-cost', `${stops * 2 >= total(halts) ? 'Stops on its own' : 'Stop rule'} at the 100 % pixel cap`, stops, { suffix: ` of ${total(halts)} jobs` },
      `with the pixel cap at 100 % (the whole image), the stop rule ended them when the next look was worth less than it cost, at ${pct(uncapped.pixels_frac)} of the image's pixels on average`
      + `${capped ? `. At the ${pct(hs.budget)} pixel cap of the headline, the cap ends ${capped['cap-hit'] ?? 0} of ${total(capped)} first and the stop rule ${capped['stop-rule'] ?? 0}` : ''}.`);
  }
  // Agentic evidence: injected planner and governor faults, and how many the agent recovered from.
  const fired = (am.planner_recovery?.timeout?.fired ?? 0) + (am.planner_recovery?.malformed?.fired ?? 0) + (am.governor_response?.fired ?? 0);
  const ok = (am.planner_recovery?.recovered_total ?? 0) + (am.governor_response?.recovered ?? 0);
  if (fired > 0) {
    fill('stat-ops', 'Failure drills', ok, { suffix: ` / ${fired} recovered` },
      'planner timeouts, malformed replies and slow-segmenter faults, injected on purpose');
  }
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

/** `extra` (optional) adds lines to the folded provenance block. */
function renderBench(sum, id, file, note, model, iou = false, extra = null) {
  const root = $(`#${id}`);
  if (!sum?.policies) return root.replaceChildren(pending('benchmarks', file));
  const prov = Object.entries(sum.provenance ?? {});
  const micro = Array.isArray(sum.micro) && sum.micro.length;
  const where = place(sum.provenance);
  const withModel = points(model?.policies?.drishyam).length > 0;
  const series = withModel ? [...SERIES, MODEL] : SERIES;
  const policies = withModel ? { ...sum.policies, model: model.policies.drishyam } : sum.policies;
  const plot = h('div', {}, chart(policies, series, iou), legend(series), marks(policies, series, iou), dataTable(policies, series),
    withModel ? h('p', { class: 'note' }, `Dashed red: Drishyam with the trained segmenter (${String(model.segmenter ?? 'ONNX')}) on the same images; the other lines use the classical one.`) : null);
  NARROW.addEventListener('change', () => plot.firstElementChild.replaceWith(chart(policies, series, iou)));
  root.replaceChildren(h('div', { class: 'stack' },
    h('p', { class: 'dataset' }, h('span', { class: 'dataset__k' }, 'dataset'), h('b', {}, String(sum.dataset ?? 'unlabelled'))),
    h('div', { class: 'bench' },
      plot,
      h('div', { class: 'stack' },
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
  return h('p', { class: 'note' }, `The best recall of any policy on these photos is ${best.toFixed(2)}`,
    strict.length ? ` (${Math.max(...strict).toFixed(2)} when a match needs IoU ≥ 0.3, the stricter rule, reported alongside)` : '',
    Number.isFinite(chance) ? `; each policy’s own boxes at random places already score up to ${chance.toFixed(2)} (${top('chance_recall_iou_only').toFixed(2)} with IoU only), so much of the lenient score is chance` : '',
    '. Here the classical segmenter, tuned on synthetic images, limits recall more than where the agent looks; a trained model is the next step.');
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

const rules = (cases) => [...new Set(cases.map((f) => String(f.rule_that_would_catch_it ?? '')))]
  .map((rule) => h('p', { class: 'fail__rule' }, h('b', {}, 'Rule that would catch it'), rule));

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
      h('p', { class: 'note' }, `${real.length} cases from the CODEBRIM evaluation, written out because its licence does not allow sharing the photos.`),
      table('Real-photo failures', ['photo', 'class', 'what happened'],
        real.map((f) => [String(f.id ?? ''), String(f.class ?? '–'), String(f.what_happened)])),
      table('Rules that would catch them', ['failure', 'rule that would catch it'],
        byRule.map(([category, cases]) => [category, String(cases[0].rule_that_would_catch_it ?? '')]))));
}

export async function initResults() {
  // The trained segmenter's summaries (results/synth_dacl10k, results/codebrim_dacl10k) are not published yet; pass
  // them as renderBench's `model` once they are. COOL was not run, so FIG. 4.3 is static text.
  const [sum, real, fails, realFails, large, live] = await Promise.all(
    ['results/summary.json', 'results/codebrim/summary.json', 'results/failures.json',
      'results/codebrim/failures_text.json', 'results/large/summary.json', 'results/live/summary.json'].map(getJSON));
  if (renderBench(sum, 'bench', 'results/summary.json', () => synthNotes(large, live), null, false, liveLoad(live))) heroChips(sum, real);
  renderBench(real, 'bench-real', 'results/codebrim/summary.json', realNote, null, true);
  renderFailures(fails, realFails);
}
