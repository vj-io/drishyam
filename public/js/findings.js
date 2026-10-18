// Finding cards, drawn as detail callouts and grouped by the region they were found in. Evidence first (crop, px
// measurements), confidence last; critical ones wait for a human, whose decision lands on the card as a rubber stamp.
import { h, dataUrl, num } from './dom.js';
import { countUp, replay, sound } from './fx.js';
import { label, words } from './panels.js';

const PRINTABLE = /^[^\x00-\x1f\x7f]{1,64}$/;
const STATE = { draft: 'draft', pending_review: 'needs review', approved: 'approved', rejected: 'rejected' };
const ISO_UTC = /^(\d{4}-\d\d-\d\d)T(\d\d:\d\d)(:\d\d(\.\d+)?)?Z$/;
const when = (at) => String(at ?? '').replace(ISO_UTC, '$1 $2 UTC');  // 2026-10-17T06:36:31Z -> 2026-10-17 06:36 UTC
const px = (v, d = 0) => (num(v) ? `${v.toLocaleString('en-US', { maximumFractionDigits: d })}` : '–');
// Until the server sends `critical_rule`, the two rules drishyam/findings.py enforces, in words.
const RULE = { exposed_rebar: 'exposed rebar is always critical', spalling: 'spalling of 40,000 px² or more is critical' };
/** The rule that made a finding critical, in words: the server's `critical_rule` ("exposed_rebar: always critical",
    "spalling 52,000 px² >= 40,000 px²") rephrased, or the rule above. */
export function ruleOf(f) {
  const r = f.critical_rule ?? '';
  let m;
  if ((m = r.match(/^(\w+): always critical$/))) return `${words(m[1])} is always critical`;
  if ((m = r.match(/^(\w+) ([\d,]+) px² >= ([\d,]+) px²$/))) return `${words(m[1])} of ${m[2]} px² is critical (${m[3]} px² or more)`;
  return r || (RULE[f.cls] ?? null);
}
const regionName = (r) => (r === 's0' ? 'Whole photo (survey look)' : `Region ${r}`);

export class Findings {
  /** `onShow(f)` points the photo at a finding; `history(region)` says which levels that region was looked at. */
  constructor(list, empty, onReview, onShow, history) {
    Object.assign(this, { list, empty, onReview, onShow, history });
    this.ready = false;
    this.cards = new Map();
    this.groups = new Map();
    this.reasons = new Map();
  }

  clear() {
    this.list.replaceChildren();
    this.cards.clear();
    this.groups.clear();
    this.reasons.clear();
    this.empty.hidden = false;
    this.ready = false;
  }

  group(region) {
    let g = this.groups.get(region);
    if (!g) {
      const hist = h('span', { class: 'fgroup__hist' });
      const cards = h('div', { class: 'cards' });
      const sec = h('section', { class: 'fgroup', 'data-region': region },
        h('header', { class: 'fgroup__head' }, h('h4', { class: 'fgroup__t' }, regionName(region)), hist), cards);
      g = { sec, cards, hist };
      this.groups.set(region, g);
      this.list.append(sec);
    }
    g.hist.textContent = this.history(region);
    return g;
  }

  /** Re-read every region's level history (after a new step). */
  refresh() {
    for (const [region, g] of this.groups) g.hist.textContent = this.history(region);
  }

  /** A finding event may repeat for the same id (re-measured at a finer level): update that card in place. */
  add(f) {
    const known = this.cards.get(f.id);
    if (known) return this.update(known, f);
    const parts = this.parts(f);
    const card = h('article', { 'data-id': f.id, tabindex: '-1' }, parts.head, parts.body, parts.conf);
    card.parts = parts;
    this.restyle(card, f);
    if (f.critical) this.addReview(card, f);
    this.cards.set(f.id, card);
    this.empty.hidden = true;
    this.group(f.region ?? '–').cards.append(card);
    this.sync1(card);
    sound.click();
    return true;
  }

  update(card, f) {
    const old = card.parts;
    const parts = this.parts(f);
    old.head.replaceWith(parts.head);
    old.body.replaceWith(parts.body);
    old.conf.replaceWith(parts.conf);
    card.parts = parts;
    this.restyle(card, f);
    this.sync1(card);
    replay(card, 'is-revised');
    return false;
  }

  restyle(card, f) {
    card.finding = f;
    card.className = `finding finding--${f.cls}${f.critical ? ' finding--critical' : ''}`;
  }

  /** Why a review request was raised for finding `id`, as recorded and in plain words (shown on its card above the sign-off). */
  addReason(id, reason, plain = reason) {
    const list = this.reasons.get(id) ?? new Map();
    list.set(reason, plain);
    this.reasons.set(id, list);
    const card = this.cards.get(id);
    if (card?.review) card.review.why.replaceChildren(...this.why(card.finding));
  }

  why(f) {
    const extra = [...(this.reasons.get(f.id) ?? [])].filter(([r]) => !r.endsWith(' is critical')).map(([, plain]) => plain);
    return [
      h('b', {}, 'Why you: '),
      [ruleOf(f), ...extra].filter(Boolean).join('; '),
      num(f.verify_iou) ? `. The latest check across zoom levels overlapped ${Math.round(f.verify_iou * 100)}\u00a0%` : '',
      f.unconfirmed ? '. Not confirmed at a finer level' : '',
      '.',
    ];
  }

  /** The finding's length as a dimension line under its crop; the number counts up to the measured value. */
  dim(m) {
    if (!num(m.length_px)) return null;
    const v = h('span', { class: 'dim__v' });
    countUp(v, m.length_px, { decimals: 1, suffix: ' px' });
    return h('div', { class: 'dim finding__dim', 'aria-hidden': 'true' }, v);
  }

  parts(f) {
    const m = f.measure ?? {};
    const crop = dataUrl('image/jpeg', f.crop_jpeg_b64);
    const state = h('span', { class: 'chip chip--state' }, STATE[f.state] ?? f.state);
    const refined = Array.isArray(f.steps) && f.steps.length > 1 ? `, refined at ${f.steps.slice(1).join(', ')}` : '';
    // Criticality may come from a coarser look than the one `measure` shows: say which, and with what area.
    const trigger = f.critical && num(f.critical_step)
      ? h('div', { class: 'finding__trigger' }, h('dt', {}, 'critical at'), h('dd', {}, `step ${f.critical_step}, ${px(f.critical_area_px)} px²`))
      : null;
    const show = h('button', { class: 'btn btn--sm finding__show', type: 'button' }, 'Show on photo');
    show.addEventListener('click', () => this.onShow(f));
    return {
      state,
      head: h('header', { class: 'finding__head' },
        h('span', { class: `chip chip--cls chip--${f.cls}` }, label(f.cls)),
        h('span', { class: 'finding__where' }, `found at step ${f.step}${refined}, L${f.level}`),
        state),
      body: h('div', { class: 'finding__body' },
        h('figure', { class: 'finding__fig' },
          crop ? h('img', { class: 'finding__crop', src: crop, alt: `Crop of the ${words(f.cls)} finding`, width: 128, height: 128 }) : h('div', { class: 'finding__crop finding__crop--none' }, 'no crop'),
          this.dim(m)),
        h('dl', { class: 'kv kv--px' },
          h('div', {}, h('dt', {}, 'length'), h('dd', {}, `${px(m.length_px, 1)} px`)),
          h('div', {}, h('dt', {}, 'width'), h('dd', {}, `${px(m.width_px, 1)} px`)),
          h('div', {}, h('dt', {}, 'area'), h('dd', {}, `${px(m.area_px)} px²`)),
          h('div', {}, h('dt', {}, 'orientation'), h('dd', {}, `${px(m.orientation_deg, 0)}°`)),
          trigger)),
      conf: h('div', { class: 'finding__foot' },
        h('p', { class: 'finding__conf' }, 'confidence ', h('b', {}, num(f.confidence) ? `${Math.round(f.confidence * 100)} %` : '–')),
        show),
    };
  }

  addReview(card, f) {
    const name = h('input', { class: 'input', type: 'text', maxlength: '64', autocomplete: 'name', id: `rev-${f.id}`, placeholder: 'Your name' });
    name.addEventListener('input', () => { err.textContent = ''; });  // err is declared just below; the listener runs later
    const approve = h('button', { class: 'btn btn--amber btn--sm', type: 'button' }, 'Approve');
    const reject = h('button', { class: 'btn btn--red btn--sm', type: 'button' }, 'Reject');
    const err = h('p', { class: 'finding__err', role: 'alert' });
    const hint = h('p', { class: 'note' }, 'Waiting for the job to finish before a decision can be signed.');
    const why = h('p', { class: 'review__why' }, ...this.why(f));
    const box = h('div', { class: 'review' },
      h('span', { class: 'badge' }, 'NEEDS A HUMAN'),
      why,
      h('p', { class: 'review__only' }, 'The agent cannot approve this. Only you can.'),
      h('label', { class: 'review__label', for: name.id }, 'Reviewer'),
      name,
      h('div', { class: 'review__btns' }, approve, reject),
      hint, err);
    const submit = async (decision) => {
      const who = name.value.trim();
      err.textContent = '';
      if (!PRINTABLE.test(who)) {
        err.textContent = 'Enter a reviewer name (1 to 64 printable characters).';
        name.focus();
        return;
      }
      card.busy = true;
      this.sync1(card);
      try {
        await this.onReview(f.id, decision, who);
      } catch (e) {
        err.textContent = e.message;
      }
      card.busy = false;
      this.sync1(card);
    };
    approve.addEventListener('click', () => submit('approve'));
    reject.addEventListener('click', () => submit('reject'));
    card.review = { box, approve, reject, name, hint, why };
    card.append(box);
  }

  sync1(card) {
    const f = card.finding;
    card.parts.state.textContent = STATE[f.state] ?? f.state;
    card.dataset.state = f.state;
    const r = card.review;
    if (!r) return;
    const open = f.state === 'pending_review';
    r.approve.disabled = r.reject.disabled = !(open && this.ready) || card.busy;
    r.name.disabled = !open || card.busy;
    r.hint.textContent = open ? 'Waiting for the job to finish before a decision can be signed.' : card.note ?? f.state;
    r.hint.hidden = open && this.ready;
    r.box.classList.toggle('review--done', !open);
  }

  /** Scroll to a finding's card and flash it; `edit` also puts the cursor in its reviewer field. */
  focus(id, edit = false) {
    const card = this.cards.get(id);
    if (!card) return;
    const field = edit && card.review && !card.review.name.disabled ? card.review.name : card;
    card.scrollIntoView({ block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    field.focus({ preventScroll: true });
    replay(card, 'is-flash');
  }

  /** A human's decision lands on the card as a rubber stamp (decorative: the state chip and note carry the text). */
  stamp(card, state, approval) {
    if (state !== 'approved' && state !== 'rejected') return;
    card.querySelector('.stamp')?.remove();
    card.append(h('span', { class: `stamp stamp--${state}`, 'aria-hidden': 'true' }, state,
      approval ? h('small', {}, `${approval.reviewer} · ${String(approval.at ?? '').slice(0, 10)}`) : null));
    sound.stamp();
  }

  /** Called when the `done` event arrives: reviews may now be signed with its token. */
  setReady() {
    this.ready = true;
    for (const c of this.cards.values()) this.sync1(c);
  }

  /** Apply the states from a (new) signed work order. */
  sync(order) {
    for (const wf of order.findings ?? []) {
      const card = this.cards.get(wf.id);
      if (!card) continue;
      const before = card.finding.state;
      card.finding.state = wf.state;
      this.sync1(card);
      if (before !== wf.state && card.review) {
        const a = (order.approvals ?? []).find((x) => x.finding_id === wf.id);
        card.note = a ? `${wf.state} by ${a.reviewer} at ${when(a.at)}` : wf.state;
        this.sync1(card);
        this.stamp(card, wf.state, a);
      }
    }
  }
}
