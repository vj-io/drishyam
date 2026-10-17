// Drafting-room feedback: count-ups, re-stamped marks, governor revision notes, optional pencil and ruler sounds.
import { h } from './dom.js';

export const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Count a number up to `to`. Each call supersedes earlier ones on the same element (the latest value always wins) and a decrease snaps. */
export function countUp(el, to, { decimals = 0, prefix = '', suffix = '', ms = 650 } = {}) {
  const run = (el._run = (el._run ?? 0) + 1);
  const from = el._v ?? 0;
  const show = (v) => {
    if (el._run !== run) return;
    el.textContent = prefix + v.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
  };
  el._v = to;
  el.dataset.value = to;
  cancelAnimationFrame(el._raf);
  if (reduced() || to <= from) return show(to);
  const t0 = performance.now();
  const tick = (now) => {
    const k = Math.min(1, (now - t0) / ms);
    show(k < 1 ? from + (to - from) * (1 - (1 - k) ** 3) : to);
    if (k < 1 && el._run === run) el._raf = requestAnimationFrame(tick);
  };
  el._raf = requestAnimationFrame(tick);
}

/** Restart a one-shot CSS animation class: a stamp landing again, a dimension line re-extending. */
export function replay(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
}

/** A governor override, filed as numbered revision note `n` that closes itself. */
export function popup(n, cause, message) {
  const host = document.getElementById('toasts');
  while (host.children.length >= 4) host.firstElementChild.remove();
  const close = h('button', { class: 'win__x', type: 'button', 'aria-label': 'Close notification' }, 'X');
  const el = h('div', { class: 'win win--toast' },
    h('div', { class: 'win__bar' }, h('span', { class: 'rev', 'aria-hidden': 'true' }, String(n)), h('span', { class: 'win__title' }, `Revision ${n}`), close),
    h('div', { class: 'win__body' }, h('span', { class: 'chip chip--gov' }, cause), h('p', {}, message)));
  const bye = () => {
    el.classList.add('is-closing');
    setTimeout(() => el.remove(), 260);
  };
  close.addEventListener('click', bye);
  setTimeout(bye, 4800);
  host.append(el);
}

let ctx = null;
let grain = null;
export const sound = {
  on: false,
  toggle() {
    this.on = !this.on;
    if (this.on) {
      ctx ??= new (window.AudioContext || window.webkitAudioContext)();
      ctx.resume();
    }
    return this.on;
  },
  /** Pencil on paper: band-passed noise whose centre wanders like a stroke. */
  scratch(len = 0.2) {
    if (!this.on || !ctx) return;
    if (!grain) {
      grain = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
      const d = grain.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    const t = ctx.currentTime;
    const src = ctx.createBufferSource();
    src.buffer = grain;
    const band = ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.Q.value = 1.4;
    band.frequency.setValueAtTime(2400, t);
    band.frequency.linearRampToValueAtTime(4300, t + len * 0.45);
    band.frequency.linearRampToValueAtTime(2900, t + len);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.07, t + 0.025);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + len);
    src.connect(band).connect(gain).connect(ctx.destination);
    src.start(t, Math.random() * 0.5, len + 0.05);
  },
  /** A ruler set down: a bright tick over a short wooden knock. */
  click() {
    this.tone(2600, 0, 0.025, 0.05, 'square');
    this.tone(420, 0, 0.07, 0.08, 'triangle');
  },
  /** A rubber stamp: a low thump with a little paper noise. */
  stamp() {
    this.tone(120, 0, 0.18, 0.16, 'sine');
    this.scratch(0.07);
  },
  tone(freq, at, len, peak, type) {
    if (!this.on || !ctx) return;
    const t = ctx.currentTime + at;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(peak, t + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + len);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + len + 0.02);
  },
};
