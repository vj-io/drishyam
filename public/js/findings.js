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
    this.showAsk(card);
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
    if (f.critical && !card.review) this.addReview(card, f);  // a finer look measured it past the critical rule: a person must sign it
    this.showAsk(card);
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
    else if (card) this.showAsk(card);
  }

  extra(f) {
    return [...(this.reasons.get(f.id) ?? [])].filter(([r]) => !r.endsWith(' is critical')).map(([, plain]) => plain);
  }

  /** A draft the agent flagged for a person (an ambiguity, a disagreement no look will settle): said on its card. A
      critical card says it in its sign-off box instead; a draft has no sign-off, so the work order does not wait for it. */
  showAsk(card) {
    const extra = card.review ? [] : this.extra(card.finding);
    card.ask?.remove();
    card.ask = extra.length ? h('p', { class: 'finding__ask' }, h('b', {}, 'Flagged for a person to look at: '), `${extra.join('; ')}. `,
      'No sign-off is needed for a draft, so the work order does not wait for it.') : null;
    if (card.ask) card.insertBefore(card.ask, card.parts.conf);
  }

  why(f) {
    const extra = this.extra(f);
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

// ---- checked by eye: the work order's not_assessed entries, each resolved by a person ----------------------------
const UNREAD = { glare: 'Too much glare to read', blur: 'Too blurred to read', dark: 'Too dark to read' };
const DECIDED = { 'nothing-found': 'nothing found', 'defect-found': 'defect found' };

/** One not_assessed entry in words: what the agent could not settle there, and why. */
function areaText(a) {
  const reason = String(a.reason ?? '');
  if (a.kind === 'needs-a-look') {  // agent.py, v1-policy2: a critical mask held under the confidence floor
    const [held, ...why] = reason.split('; ');
    const floor = held.match(/under the floor ([\d.]+)/)?.[1];
    return `${label(a.cls)}, confidence ${num(a.confidence) ? Math.round(a.confidence * 100) : '–'} %, under the floor `
      + `${floor ?? '–'}: too uncertain to draft as critical${why.length ? `; ${why.join('; ')}` : ''}.`;
  }
  if (a.region) return `Region ${a.region}: ${reason}.`;
  if (reason === 'no candidate region was proposed') return 'Whole photo: no region stood out, so nothing was looked at closely.';
  return `${UNREAD[reason] ?? reason}${a.step ? ` at step ${a.step}` : ' in the overview'}.`;
}

/** Step 4's "Not assessed" group: every area the work order lists as not assessed, each with a small Checked by eye
    control (the cards' reviewer field, Nothing found / Defect found, an optional note). A defect found by eye is a
    human finding: who, when and the note. Nothing here turns it into a confidence or a measurement. */
export class Checks {
  /** `onCheck(index, decision, reviewer, note)` signs a resolution; `onShow(entry)` points the photo at its box. */
  constructor(root, onCheck, onShow) {
    Object.assign(this, { root, onCheck, onShow });
    this.items = [];
  }

  clear() {
    this.root.replaceChildren();
    this.items = [];
  }

  /** Called when the job's done event arrives: one item per not_assessed entry. */
  show(order) {
    const na = order.not_assessed ?? [];
    this.clear();
    if (!na.length) return;
    this.items = na.map((a, i) => this.item(a, i));
    this.count = h('span', { class: 'fgroup__hist' });
    this.root.append(h('section', { class: 'fgroup' },
      h('header', { class: 'fgroup__head' }, h('h4', { class: 'fgroup__t' }, 'Not assessed: check by eye'), this.count),
      h('div', { class: 'cards' }, this.items.map((x) => x.el))));
    this.sync(order);
  }

  item(a, i) {
    const name = h('input', { class: 'input', type: 'text', maxlength: '64', autocomplete: 'name', id: `chk-${i}`, placeholder: 'Your name' });
    const note = h('input', { class: 'input', type: 'text', maxlength: '200', id: `chk-note-${i}`, placeholder: 'What you saw' });
    const none = h('button', { class: 'btn btn--amber btn--sm', type: 'button' }, 'Nothing found');
    const found = h('button', { class: 'btn btn--red btn--sm', type: 'button' }, 'Defect found');
    const err = h('p', { class: 'finding__err', role: 'alert' });
    name.addEventListener('input', () => { err.textContent = ''; });
    const show = h('button', { class: 'btn btn--sm finding__show', type: 'button' }, 'Show on photo');
    show.addEventListener('click', () => this.onShow(a));
    const box = h('div', { class: 'review' },
      h('b', { class: 'review__t' }, 'Checked by eye'),
      h('p', { class: 'review__why' }, 'The agent could not assess this area, so the work order is not ready until a person looks. Look at it, then say what you saw.'),
      h('label', { class: 'review__label', for: name.id }, 'Reviewer'), name,
      h('label', { class: 'review__label', for: note.id }, 'Note (optional)'), note,
      h('div', { class: 'review__btns' }, none, found), err);
    const el = h('article', { class: 'finding check', 'data-index': String(i), tabindex: '-1' },
      h('header', { class: 'finding__head' },
        h('span', { class: 'chip chip--state' }, a.kind === 'needs-a-look' ? 'needs a look' : 'not assessed'),
        h('span', { class: 'finding__where' }, a.step ? `step ${a.step}` : 'overview'), show),
      h('p', { class: 'check__what' }, areaText(a)),
      box);
    const item = { el, box, name, note, none, found, err, done: false };
    const submit = async (decision) => {
      const who = name.value.trim();
      err.textContent = '';
      if (!PRINTABLE.test(who)) {
        err.textContent = 'Enter a reviewer name (1 to 64 printable characters).';
        name.focus();
        return;
      }
      none.disabled = found.disabled = name.disabled = note.disabled = true;
      try {
        await this.onCheck(i, decision, who, note.value.trim());
      } catch (e) {
        err.textContent = e.message;
      }
      if (!item.done) none.disabled = found.disabled = name.disabled = note.disabled = false;
    };
    none.addEventListener('click', () => submit('nothing-found'));
    found.addEventListener('click', () => submit('defect-found'));
    return item;
  }

  /** Apply the resolutions of a (new) signed work order: a resolved entry shows who checked it, when, and the note. */
  sync(order) {
    const na = order.not_assessed ?? [];
    na.forEach((a, i) => {
      const item = this.items[i];
      const r = a.resolution;
      if (!item || !r || item.done) return;
      item.done = true;
      item.el.dataset.state = r.decision;
      item.el.querySelector('.chip--state').textContent = 'checked by eye';
      item.box.classList.add('review--checked');
      item.box.replaceChildren(h('b', { class: 'review__t' }, 'Checked by eye'),
        h('p', { class: 'check__done' }, h('b', {}, DECIDED[r.decision] ?? r.decision), ` by ${r.reviewer} at ${when(r.at)}.`),
        r.note ? h('p', { class: 'check__done' }, `Note: ${r.note}`) : null);
      sound.stamp();
    });
    const left = na.filter((a) => !a.resolution).length;
    if (this.count) this.count.textContent = left ? `${left} of ${na.length} still to check` : na.length === 1 ? 'checked' : `all ${na.length} checked`;
  }
}
