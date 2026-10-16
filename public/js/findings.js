// findings.queue cards. Evidence first (crop, px measurements), confidence last; critical ones wait for a human.
import { h, dataUrl, num } from './dom.js';
import { burstAt, sound } from './fx.js';

const PRINTABLE = /^[^\x00-\x1f\x7f]{1,64}$/;
const STATE = { draft: 'draft', pending_review: 'needs review', approved: 'approved', rejected: 'rejected' };
const px = (v, d = 0) => (num(v) ? `${v.toLocaleString('en-US', { maximumFractionDigits: d })}` : '–');

export class Findings {
  constructor(list, empty, onReview) {
    this.list = list;
    this.empty = empty;
    this.onReview = onReview;
    this.ready = false;
    this.cards = new Map();
  }

  clear() {
    this.list.replaceChildren();
    this.cards.clear();
    this.empty.hidden = false;
    this.ready = false;
  }

  /** A finding event may repeat for the same id (re-measured at a finer level): update that card in place. */
  add(f) {
    const known = this.cards.get(f.id);
    if (known) return this.update(known, f);
    const parts = this.parts(f);
    const card = h('article', { 'data-id': f.id }, parts.head, parts.body, parts.conf);
    card.parts = parts;
    this.restyle(card, f);
    if (f.critical) this.addReview(card, f);
    this.cards.set(f.id, card);
    this.empty.hidden = true;
    this.list.append(card);
    this.sync1(card);
    burstAt(card, 14);
    sound.pop();
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
    burstAt(card, 6);
  }

  restyle(card, f) {
    card.finding = f;
    card.className = `finding finding--${f.cls}${f.critical ? ' finding--critical' : ''}`;
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
    return {
      state,
      head: h('header', { class: 'finding__head' },
        h('span', { class: `chip chip--cls chip--${f.cls}` }, f.cls),
        h('span', { class: 'finding__where' }, `found at step ${f.step}${refined}, L${f.level}`),
        state),
      body: h('div', { class: 'finding__body' },
        crop ? h('img', { class: 'finding__crop', src: crop, alt: `Crop of the ${f.cls} finding`, width: 128, height: 128 }) : h('div', { class: 'finding__crop finding__crop--none' }, 'no crop'),
        h('dl', { class: 'kv kv--px' },
          h('div', {}, h('dt', {}, 'length'), h('dd', {}, `${px(m.length_px, 1)} px`)),
          h('div', {}, h('dt', {}, 'width'), h('dd', {}, `${px(m.width_px, 1)} px`)),
          h('div', {}, h('dt', {}, 'area'), h('dd', {}, `${px(m.area_px)} px²`)),
          h('div', {}, h('dt', {}, 'orientation'), h('dd', {}, `${px(m.orientation_deg, 0)}°`)),
          trigger)),
      conf: h('p', { class: 'finding__conf' }, 'confidence ', h('b', {}, num(f.confidence) ? `${Math.round(f.confidence * 100)} %` : '–')),
    };
  }

  addReview(card, f) {
    const name = h('input', { class: 'input', type: 'text', maxlength: '64', autocomplete: 'name', id: `rev-${f.id}`, placeholder: 'Your name' });
    const approve = h('button', { class: 'btn btn--lime btn--sm', type: 'button' }, 'Approve');
    const reject = h('button', { class: 'btn btn--pink btn--sm', type: 'button' }, 'Reject');
    const err = h('p', { class: 'finding__err', role: 'alert' });
    const hint = h('p', { class: 'note' }, 'Waiting for the job to finish before a decision can be signed.');
    const box = h('div', { class: 'review' },
      h('span', { class: 'badge' }, 'NEEDS A HUMAN'),
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
    card.review = { box, approve, reject, name, hint };
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
        card.note = a ? `${wf.state} by ${a.reviewer} at ${a.at}` : wf.state;
        this.sync1(card);
        burstAt(card, wf.state === 'approved' ? 22 : 10);
        sound.pop();
      }
    }
  }
}
