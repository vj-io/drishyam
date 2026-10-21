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
const INJECTED = { p95_spike: 'slow segmenter' };  // a chaos drill the agent records on its step
const chip = (cause) => h('span', { class: `chip chip--${GOVERNOR.has(cause) ? 'gov' : CHIP[cause] ?? 'plain'}` }, cause);

const ms = (v) => (num(v) ? `${v}` : '–');
const pair = (rt, op) => `${ms(rt?.p50_ms?.[op])} / ${ms(rt?.p95_ms?.[op])} ms`;
export const pad = (n) => `#${String(n).padStart(2, '0')}`;
export const px = (v) => (num(v) ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : '–');

// ---- plain words -------------------------------------------------------------------------------
const size = (L) => (L ? `1/${2 ** L} size` : 'full resolution');
const scale = (L) => `1:${2 ** L}`;
const pct = (v) => `${Math.round(Number(v) * 100)}\u00a0%`;  // a no-break space: "19 %" never splits across lines
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const dec = (v) => Number(v).toPrecision(2);  // 3.98e-04 -> 0.00040: no scientific notation on screen
/** A damage class in words ("exposed rebar"), and as a label ("Exposed rebar"). */
export const words = (cls) => String(cls).replaceAll('_', ' ');
export const label = (cls) => cap(words(cls));

/** What kind of evidence a cause is: the photo itself (OpenCV), KILL, the governor, the planner drill, or a replay. */
export const kindOf = (cause) => (cause === 'replay' ? 'replay' : cause === 'kill-switch' ? 'kill'
  : cause === 'planner-fallback' || String(cause).startsWith('planner:') ? 'planner'
    : GOVERNOR.has(cause) ? 'governor' : 'vision');

const HALT = {
  'stop-rule': 'The next look would be worth less than it costs, so it stops here.',
  exhausted: 'Every candidate region has been looked at, so it stops.',
  'cap-hit': 'A budget cap was reached (budget spent), so it stops and reports what it did not reach.',
  'kill-switch': 'You pressed KILL, so it stopped before its next look. Its state is kept: Resume carries on from here.',
  'planner:stop': 'The planner proposed stopping and the validator accepted, so it stops.',
};
// A kill-switch halt as the job's first record: only the operator-wide switch can do that (a page's KILL is read after a look).
const KILLED_FIRST = 'The kill switch was already set, so it stopped before its first look. Nothing was looked at, so there is nothing to resume.';
/** The stop rule's bar was its floor (ε × v0 ÷ t̂) rather than ε × value so far ÷ cost so far: "... (floor: eps x v0 / t_hat)". */
const floorBound = (rationale) => /\(floor[:)]/.test(String(rationale ?? ''));
const FLOOR = ' The bar it fell under was the floor, the least any look must promise: ε × v0 ÷ t̂, where v0 is the value of a look'
  + ' that finds nothing. It was higher than the usual bar, ε × value found so far ÷ cost so far.';
/** Why a job stopped; `rationale` (the halt record's) says which term of the stop rule bound; `looked`: whether any look
    came before the halt (a halt record numbered 1 had none). */
export const haltText = (cause, rationale, looked = true) => (cause === 'kill-switch' && !looked ? KILLED_FIRST
  : (HALT[cause] ?? `It stopped (${cause}).`) + (cause === 'stop-rule' && floorBound(rationale) ? FLOOR : ''));

/** Where a candidate's severity came from, as the agent records it after the number: "spalling seen at L2", "no class
    seen at L2", "not seen yet", "seeding off" (an ablation), or a bare class. Unknown text is kept as recorded. */
function severitySource(t) {
  let m;
  if ((m = t.match(/^(\w+) seen at L(\d)$/)) && m[1] !== 'no') return `${words(m[1])}, seen at ${scale(m[2])}`;
  if ((m = t.match(/^no class seen at L(\d)$/))) return `the default: no class seen there at ${scale(m[1])}`;
  if (t === 'not seen yet') return 'the default: no look has shown it yet';
  if (t === 'seeding off') return 'the default: looks do not set it in this run';
  return /^\w+$/.test(t) ? words(t) : t;
}

/** The terms an argmax (or random-pick) rationale names: "prior 0.45, severity 2 (spalling seen at L2), uncertainty 0.31,
    unseen 0.62, t_hat 70 ms". Each is read on its own, so a clause that leaves one out still reads. */
function terms(t) {
  const sev = t.match(/severity ([\d.]+)(?: \(([^)]*)\))?/);
  return {
    prior: t.match(/prior ([\d.]+)/)?.[1], unseen: t.match(/unseen ([\d.]+)/)?.[1], t: t.match(/t_hat ([\d.]+) ?ms/)?.[1],
    u: t.match(/(?:uncertainty|entropy) ([\d.]+)/)?.[1], sev: sev?.[1], sevFrom: sev?.[2] ? severitySource(sev[2]) : null,
  };
}
const severity = (x) => (x.sev ? `severity ${x.sev}${x.sevFrom ? ` (${x.sevFrom})` : ''}` : '');

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
  if ((m = why.match(/(\w+) area (\d+) px < (\d+) at L\d/))) return `The ${words(m[1])} it found is tiny (${m[2]} px < ${m[3]} px), so it zooms in 2× to measure it properly${now}.`;
  if ((m = why.match(/iou ([\d.]+) < 0\.5 between L(\d) and L(\d)/))) return `Its L${m[2]} and L${m[3]} looks disagree (overlap ${m[1]} < 0.5), so it takes one more, finer look${now}.`;
  if ((m = why.match(/new (\w+) mask at L(\d); confirming at L(\d)/))) return `A new ${words(m[1])} mask appeared at L${m[2]}, so it double-checks it one level finer (L${m[3]}) before trusting it.`;
  const x = (m = why.match(/argmax score [^:]+: ([^;]+)/)) && terms(m[1]);
  if (x?.prior && x.unseen && x.t) {
    const why2 = [`damage looks likely there (prior ${x.prior})`, x.sev && `it weighs ${severity(x)}`, x.u && `it is unsure there (uncertainty ${x.u})`,
      `${Math.round(+x.unseen * 100)} % of it is unseen`].filter(Boolean).join(', ');
    return `Region ${s.region} is now the best value for the time: ${why2}, and a look costs about ${x.t} ms. So it looks there next${now}.`;
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
    halt ? haltText(s.cause, why, s.step > 1) : lookText(s),
    quality ? `The view failed the quality check (${quality[1]}), so it does not trust this look.` : '',
  ].filter(Boolean);
}

// A recorded rationale is "; "-joined clauses (governor notes, the look, the quality gate). Each clause in plain
// words for the trace line: '' drops it, and one starting "so"/"and"/"but" carries on the sentence before it.
const CLAUSES = [
  [/^one L(\d) look covers all (\d+) candidates before zooming$/, (m) => `the whole photo first, at ${size(+m[1])}: all ${m[2]} candidate regions in one look`],
  [/^(\d+) looks affordable$/, (m) => `the budget has room for ${m[1]} looks`],
  [/^(argmax|random pick[^,]*,) score [^:]+: (.+)$/, (m) => {
    const x = terms(m[2]);
    if (!(x.prior && x.unseen && x.t)) return m[0];  // a clause this page cannot read is kept as recorded
    return [`${m[1] === 'argmax' ? 'best value for the time' : 'a random pick'}: damage ${+x.prior >= 0.5 ? 'looks likely' : 'is possible'} (${x.prior})`,
      severity(x), x.u && `unsure ${x.u}`, +x.unseen >= 0.995 ? 'not seen yet' : `${pct(x.unseen)} not seen yet`, `a look costs about ${x.t} ms`].filter(Boolean).join(', ');
  }],
  [/^lambda /, () => ''],  // the stop line: a halt says it in words
  [/^entropy ([\d.]+) > ([\d.]+) at L\d$/, (m) => `unsure what it saw (${m[1]}, over ${m[2]}), so it zooms in 2×`],
  [/^(\w+) area (\d+) px < (\d+) at L\d$/, (m) => `the ${words(m[1])} is tiny (${m[2]} px, under ${m[3]} px), so it zooms in 2× to measure it`],
  [/^iou ([\d.e+-]+) < [\d.]+ between L(\d) and L(\d)$/, (m) => `the ${scale(m[2])} and ${scale(m[3])} looks disagree (${pct(m[1])} overlap), so it takes one more, finer look`],
  [/^new (\w+) mask at L(\d)$/, (m) => `a new ${words(m[1])} at ${scale(m[2])}`],
  [/^confirming at L(\d)$/, (m) => `so it double-checks at ${scale(m[1])}`],
  [/^new (\w+) mask not verified: L(\d) is the finest level for \S+ \(min_view ([\d.]+)\), no finer look$/, (m) => `a new ${words(m[1])}, `
    + `not double-checked: ${size(+m[2])} is the finest this engine looks (every look spans at least ${pct(m[3])} of the photo’s long side), so there is no finer look to confirm it`],
  [/^L(\d) is the finest level for \S+ \(min_view ([\d.]+)\): ([^,]+), (.+)$/, (m) => {  // agent.py: a verify the finest level rules out
    let x;
    const saw = (x = m[3].match(/^entropy ([\d.]+) > ([\d.]+)$/)) ? `it was unsure what it saw (${x[1]}, over ${x[2]})`
      : (x = m[3].match(/^(\w+) area (\d+) px < A_min (\d+)$/)) ? `the ${words(x[1])} is tiny (${x[2]} px, under ${x[3]} px)` : m[3];
    const kept = (x = m[4].match(/^(.+) drafted here$/)) ? `it drafts ${words(x[1])} here`
      : (x = m[4].match(/^(\w+) not drafted$/)) ? `the ${words(x[1])} is not drafted` : m[4] === 'no class' ? 'nothing is drafted' : m[4];
    return `${size(+m[1])} is the finest this engine looks (every look spans at least ${pct(m[2])} of the photo’s long side): ${saw}, ${kept}`;
  }],
  [/^no finer look$/, () => 'and there is no finer look to confirm it'],
  [/^iou ([\d.e+-]+) < [\d.]+ between L(\d) and L(\d) not settled: L(\d) is the finest level for \S+ \(min_view ([\d.]+)\), no finer look$/,
    (m) => `the ${scale(m[2])} and ${scale(m[3])} looks disagree (${pct(m[1])} overlap), and it cannot settle it: ${size(+m[4])} is `
      + `the finest this engine looks (every look spans at least ${pct(m[5])} of the photo’s long side), so there is no finer look`],
  [/^p95\(segment\) (\d+) ms > SLA (\d+) ms$/, (m) => `the segmenter got slow (${m[1]} ms, limit ${m[2]} ms)`],
  [/^model -> (\w+)$/, (m) => `so it switches to the ${m[1]} model`],
  [/^one level coarser/, () => 'so it looks one level coarser'],
  [/^p95\(warp\) ([\d.]+) ms > 2x median$/, (m) => `cutting out views got slow (${m[1]} ms, over twice the usual)`],
  [/^gaze (\d+) px, batching$/, (m) => `so views shrink to ${m[1]} px and the rest are batched`],
  [/^80% of the \$ cap spent$/, () => '80 % of the $ budget is spent'],
  [/^verifying critical classes only$/, () => 'so only critical damage is double-checked'],
  [/^quality gate: (\w+)$/, (m) => `the view failed the quality check (${m[1]}), so it does not trust this look`],
  [/^every candidate has been looked at$/, () => 'every candidate region has been looked at'],
  [/^best score (\S+) < lambda (\S+)/, (m) => (floorBound(m.input)
    ? `the next look would be worth less than it costs: its score ${dec(m[1])} is under the floor ${dec(m[2])} (the least any look must promise, ε × v0 ÷ t̂)`
    : 'the next look would be worth less than it costs')],
  [/^kill switch set$/, (m, s) => (s.step > 1 ? 'you pressed KILL' : 'the kill switch was already set before its first look')],
  [/^state preserved for resume$/, (m, s) => (s.step > 1 ? 'its state is kept for Resume' : 'but nothing was looked at, so there is nothing to resume')],
  [/^next look needs (\d+) px$/, (m) => `the next look needs ${px(+m[1])} px`],
  [/^(-?\d+) px left of the cap$/, (m) => `but only ${px(+m[1])} px of the pixel budget are left`],
  [/^\$[\d.]+ spent of \$[\d.]+$/, () => 'the $ budget is spent'],
  [/^[\d.]+ s used of [\d.]+ s$/, () => 'the time budget is used up'],
  [/^max_steps (\d+) reached$/, (m) => `the limit of ${m[1]} steps is reached`],
  [/^planner proposed stop$/, () => 'the planner proposed stopping'],
  [/^validator accepted$/, () => 'and the validator accepted'],
];

/** A step's reason in one or two plain sentences, built from its recorded rationale (an unknown clause is kept as recorded). */
export function brief(s) {
  if (s.cause === 'replay') return 'Replayed from before KILL; nothing new is decided.';
  if (String(s.cause).startsWith('planner:')) return `The planner asked to ${words(s.cause.slice(8))}, and the validator allowed it.`;
  const text = String(s.rationale ?? '').split('; ').reduce((out, clause) => {
    const rule = CLAUSES.find(([re]) => re.test(clause));
    const said = rule ? rule[1](clause.match(rule[0]), s) : clause;
    if (!said) return out;
    return !out ? cap(said) : /^(so|and|but) /.test(said) ? `${out}, ${said}` : `${out}. ${cap(said)}`;
  }, '');
  return text ? `${text}.` : 'No reason was recorded.';
}

/** A review request's kind: the server's `kind` (critical, ambiguous, disagree, class-change, unconfirmed), else read from
    its reason (a trace recorded before the field). */
export function reviewKind(ev) {
  if (ev.kind) return String(ev.kind);
  const r = String(ev.reason ?? '');
  return r.endsWith(' is critical') ? 'critical' : r.startsWith('ambiguous:') ? 'ambiguous' : r.startsWith('unconfirmed:') ? 'unconfirmed'
    : r.startsWith('class change:') ? 'class-change' : r.startsWith('iou ') ? 'disagree' : 'review';
}

/** A review request's reason in plain words (an unknown one is kept as recorded). */
export function reviewText(reason) {
  let m;
  if ((m = reason.match(/^(unconfirmed: )?iou ([\d.e+-]+) < [\d.]+ between L(\d) and L(\d)$/))) {
    return `the ${scale(m[3])} and ${scale(m[4])} zoom levels disagree (${pct(m[2])} overlap)${m[1] ? ', and no finer look is left to settle it' : ''}`;
  }
  // "class change: spalling at L1, corrosion at L0 (same place, iou 0.62)": the closer look is the lower level
  if ((m = reason.match(/^class change: (\w+) at L(\d), (\w+) at L(\d)(?: \(same place, iou ([\d.]+)\))?/))) {
    const [near, far] = +m[2] < +m[4] ? [m[1], m[3]] : [m[3], m[1]];
    return `the closer look saw ${words(near)} where the coarser one saw ${words(far)}: a person decides`
      + `${m[5] ? ` (same place: the two looks' masks overlap ${pct(m[5])})` : ''}`;
  }
  if ((m = reason.match(/^ambiguous: (\w+) (\d+) px .* inside (\w+) \w+; recorded as/))) {
    return `it also looks like ${words(m[1])} (${px(+m[2])} px, mostly inside it), but only ${words(m[3])} was recorded`;
  }
  return (m = reason.match(/^(\w+) is critical$/)) ? `${words(m[1])} is critical` : reason;
}

/** The cross-level check of a look, from its `verify` record {iou (same class), class, against_level, class_now, iou_any}. */
function verifyText(v) {
  if (!v) return null;
  const then = scale(v.against_level);
  const agree = num(v.iou) && v.iou >= 0.5;  // the compared class still overlaps: no class change, whatever the best class now is
  if (!agree && v.class && v.class_now && v.class_now !== v.class && num(v.iou_any) && v.iou_any >= 0.5) {  // its own class may be gone (no iou)
    return `same place, different class: ${words(v.class_now)} now, ${words(v.class)} in the ${then} look (${pct(v.iou_any)} overlap)`;
  }
  return num(v.iou) ? `${v.class ? `its ${words(v.class)} mask: ` : ''}${pct(v.iou)} overlap with the ${then} look` : null;
}

/** A step's `seeded` list, the entries that carry a region and an uncertainty. */
const seededOf = (s) => (Array.isArray(s.seeded) ? s.seeded : []).filter((c) => c?.region && num(c.entropy));

/** The unvisited candidates this look set the uncertainty and class of ({region, entropy, cls, from_level}), in words. */
export function seededText(s) {
  const list = seededOf(s);
  if (!list.length) return null;
  const one = (c) => `${c.region} (${c.cls ? `${words(c.cls)} seen there` : 'no class seen there'}, uncertainty ${c.entropy.toFixed(2)})`;
  return `It also set what the agent expects in ${list.length} region${list.length === 1 ? '' : 's'} it has not visited: `
    + `${list.map(one).join(', ')}. Their next scores use these numbers.`;
}

/** What a look measured, in one line. */
function sawText(s) {
  const o = s.observation;
  if (!o) return null;
  const best = s.verify?.class_now;
  const top = best && num(o.classes?.[best]) ? [best, o.classes[best]] : Object.entries(o.classes ?? {}).sort((x, y) => y[1] - x[1])[0];
  return [
    top ? `${words(top[0])} ${px(top[1])} px` : 'no damage',
    num(o.entropy) ? `uncertainty ${o.entropy.toFixed(2)}` : null,
    verifyText(s.verify),
    s.quality?.ok === false ? `failed the quality check (${s.quality.reason ?? 'rejected'})` : null,
  ].filter(Boolean).join(' · ');
}

/** The raw record behind a plain-words line, folded away. */
const numbers = (title, ...raw) => h('details', { class: 'detail__more' }, h('summary', {}, title),
  raw.filter(Boolean).map((t) => h('p', { class: 'detail__raw' }, t)));

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
    if (s.injected) parts.push(h('span', { class: 'chip chip--inject' }, `injected: ${INJECTED[s.injected] ?? words(s.injected)}`));
    if (num(s.latency_ms)) parts.push(h('span', { class: 't-ms' }, `${s.latency_ms} ms`));
    const saw = sawText(s);
    const seeded = seededOf(s);
    const btn = h('button', { type: 'button', class: 'trace__btn' },
      h('span', { class: 'trace__head' }, parts),
      s.rationale ? h('span', { class: 'trace__why' }, h('i', {}, 'because'), from ? ` (from ${from})` : '', ': ', brief(s)) : null,
      saw ? h('span', { class: 'trace__saw' }, h('i', {}, 'saw'), ': ', saw) : null,
      seeded.length ? h('span', { class: 'trace__saw trace__seeded' }, h('i', {}, 'set for later'), ': ',
        seeded.map((c) => `${c.region}: ${c.cls ? `${words(c.cls)} seen there` : 'no class seen there'}, unsure ${c.entropy.toFixed(2)}`).join('; ')) : null);
    btn.addEventListener('click', () => this.onPick(s.step));
    this.push(h('li', { class: 'trace__line', 'data-step': s.step }, btn,
      s.rationale ? numbers('All numbers for this step', `recorded: ${s.rationale}`) : null));
  }

  /** A finding or a review request: what the agent handed on, and why (`raw`: the reason as recorded, folded away;
      `tag`: a chip, such as the review's kind). Clicking it opens the finding's card. */
  addNote(kind, stepNo, id, text, because, onOpen, raw = null, tag = null) {
    const btn = h('button', { type: 'button', class: 'trace__btn' },
      h('span', { class: 'trace__head' }, h('span', { class: 't-num' }, pad(stepNo)), h('span', { class: `t-act t-act--${kind}` }, kind),
        tag ? h('span', { class: 'chip chip--unsure' }, tag) : null, h('span', {}, text)),
      because ? h('span', { class: 'trace__why' }, h('i', {}, 'because'), ': ', because) : null);
    btn.addEventListener('click', onOpen);
    this.push(h('li', { class: `trace__line trace__line--${kind}`, 'data-finding': id }, btn,
      raw ? numbers('All numbers', `recorded: ${raw}`) : null));
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
  const seeded = seededText(step);
  const v = step.verify;
  const classes = o?.classes && Object.keys(o.classes).length ? Object.entries(o.classes).map(([k, n]) => `${k} ${px(n)} px`).join(', ') : 'none';
  root.replaceChildren(...[
    h('p', { class: 'detail__title' }, `step ${step.step} · ${step.decision?.action ?? 'step'}`, step.cause ? ' ' : null, step.cause ? chip(step.cause) : null),
    h('h4', { class: 'detail__h' }, step.decision?.action === 'halt' ? 'Why it stopped' : 'Why this look'),
    h('p', { class: 'detail__why' }, explain(step, planner).join(' ')),
    saw ? h('h4', { class: 'detail__h' }, 'What it saw') : null,
    saw ? h('p', { class: 'detail__why' }, saw) : null,
    seeded ? h('h4', { class: 'detail__h' }, 'What it set for later') : null,
    seeded ? h('p', { class: 'detail__why' }, seeded) : null,
    h('details', { class: 'detail__more' }, h('summary', {}, 'All numbers for this step'),
      h('p', { class: 'detail__raw' }, 'recorded: ', step.rationale ?? 'no rationale recorded'),
      step.governor?.cause ? h('p', { class: 'detail__raw' }, `governor: ${step.governor.cause}: ${step.governor.rationale ?? ''}`) : null,
      h('dl', { class: 'kv' },
        row('entropy (unsure, 0-1)', num(o?.entropy) ? o.entropy.toFixed(2) : null),
        row('mask area', num(o?.mask_area_px) ? `${px(o.mask_area_px)} px` : null),
        row('classes', o ? classes : null),
        row('verify IoU (overlap)', num(v?.iou) ? `${v.iou.toFixed(2)}${v.class ? `: the ${words(v.class)} mask vs L${v.against_level}`
          + `${o && !o.classes?.[v.class] ? `, ${words(v.class)} gone at this look` : ''}` : ` vs L${v.against_level}`}` : null),
        row('best class then → now', v?.class && v.class_now !== undefined ? `${words(v.class)} at L${v.against_level} → ${v.class_now ? words(v.class_now) : 'no defect'} now` : null),
        row('best masks overlap (any class)', num(v?.iou_any) ? v.iou_any.toFixed(2) : null),
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
