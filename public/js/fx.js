// Dopamine: count-up numbers, sparkle bursts, confetti, governor pop-ups, optional WebAudio blips.
import { h } from './dom.js';

const PALETTE = ['var(--pink)', 'var(--lime)', 'var(--cobalt)', '#fff', 'var(--chrome-2)'];
const rand = (a, b) => a + Math.random() * (b - a);

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

function spawn(cls, vars, x, y) {
  const layer = document.getElementById('fx');
  const el = h('span', { class: cls });
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
  for (const [k, v] of Object.entries(vars)) el.style.setProperty(k, v);
  el.addEventListener('animationend', () => el.remove(), { once: true });
  layer.append(el);
}

/** Four-point sparkles flying out of a viewport coordinate. */
export function burst(x, y, n = 12) {
  if (reduced()) return;
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 + rand(-0.2, 0.2);
    const d = rand(36, 96);
    spawn('burst', {
      '--dx': `${Math.cos(a) * d}px`, '--dy': `${Math.sin(a) * d}px`, '--s': rand(0.7, 1.5),
      '--c': PALETTE[i % PALETTE.length], '--r': `${rand(-90, 90)}deg`,
    }, x, y);
  }
}

export function burstAt(el, n) {
  const r = el.getBoundingClientRect();
  burst(r.left + r.width / 2, r.top + Math.min(r.height / 2, 80), n);
}

export function confetti(n = 80) {
  if (reduced()) return;
  for (let i = 0; i < n; i++) {
    spawn('confetti', {
      '--x': `${rand(-30, 30)}vw`, '--r': `${rand(360, 1080)}deg`, '--d': `${rand(1.6, 3.2)}s`,
      '--c': PALETTE[i % PALETTE.length], '--w': `${rand(6, 12)}px`,
    }, rand(10, window.innerWidth - 10), -16);
  }
}

/** Little desktop pop-up window that closes itself. */
export function popup(title, cause, message) {
  const host = document.getElementById('toasts');
  while (host.children.length >= 4) host.firstElementChild.remove();
  const close = h('button', { class: 'win__x', type: 'button', 'aria-label': 'Close notification' }, 'X');
  const el = h('div', { class: 'win win--toast' },
    h('div', { class: 'win__bar' }, h('i', { class: 'win__orb' }), h('span', { class: 'win__title' }, title), close),
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
  pop() { this.notes([[740, 0], [1110, 0.06]], 0.1); },
  chime() { this.notes([[1047, 0], [1319, 0.14], [1568, 0.28], [2093, 0.42]], 0.32); },
  notes(list, len) {
    if (!this.on || !ctx) return;
    for (const [freq, at] of list) {
      const t = ctx.currentTime + at;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.09, t + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + len);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t);
      osc.stop(t + len + 0.02);
    }
  },
};
