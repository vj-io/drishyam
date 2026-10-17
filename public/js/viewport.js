// The photo canvas, drawn like a sheet: fitted photo, saliency heatmap, candidate boxes, privacy blur, the
// current gaze window with dimension arrows in L0 px, and the gaze path as numbered leaders with a FIG. callout.
import { dataUrl, num } from './dom.js';
import { reduced } from './fx.js';

const C = { deep: '#082A55', line: '#E8F1FF', line2: '#B8CDEC', line3: '#8EAEDB', amber: '#FFC857', ink: '#2E2100', sky: '#9CC8FF' };
const RING = { gaze: C.line, escalate: C.sky, verify: C.amber };
const MONO = '"IBM Plex Mono", ui-monospace, monospace';
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => 1 - (1 - t) ** 3;
const back = (t) => 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2;
const phase = (t, a, b) => Math.min(1, Math.max(0, (t - a) / (b - a)));

export const pointOf = (step) => {
  const a = step.decision?.args;
  return num(a?.x) && num(a?.y) ? { x: a.x, y: a.y } : null;
};

const RAMP = [[0, [15, 59, 115]], [0.5, [156, 200, 255]], [1, [255, 200, 87]]];
function ramp(t) {
  const [[t0, c0], [t1, c1]] = t < 0.5 ? [RAMP[0], RAMP[1]] : [RAMP[1], RAMP[2]];
  const k = (t - t0) / (t1 - t0);
  return c0.map((v, i) => Math.round(lerp(v, c1[i], k)));
}

/** Saliency PNG (0-255) -> RGBA canvas using a blueprint > sky > amber ramp. */
async function colourise(b64) {
  const url = dataUrl('image/png', b64);
  if (!url) return null;
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < d.data.length; i += 4) {
      const t = d.data[i] / 255;
      [d.data[i], d.data[i + 1], d.data[i + 2]] = ramp(t);
      d.data[i + 3] = Math.round(255 * Math.min(1, t * 1.5) * 0.85);
    }
    g.putImageData(d, 0, 0);
    return c;
  } catch {
    return null;
  }
}

function arrow(g, x, y, ang) {
  g.beginPath();
  g.moveTo(x, y);
  g.lineTo(x - 9 * Math.cos(ang - 0.42), y - 9 * Math.sin(ang - 0.42));
  g.lineTo(x - 9 * Math.cos(ang + 0.42), y - 9 * Math.sin(ang + 0.42));
  g.closePath();
  g.fill();
}

/** Text in a filled box, centred on (x, y). */
function label(g, text, x, y, { bg = C.deep, fg = C.line, edge = C.line, font = `700 11.5px ${MONO}` } = {}) {
  g.font = font;
  const w = g.measureText(text).width + 12;
  g.fillStyle = bg;
  g.fillRect(x - w / 2, y - 9, w, 18);
  if (edge) {
    g.strokeStyle = edge;
    g.lineWidth = 1.5;
    g.strokeRect(x - w / 2, y - 9, w, 18);
  }
  g.fillStyle = fg;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, x, y + 0.5);
}

export class Viewport {
  constructor(canvas, onPick) {
    this.c = canvas;
    this.g = canvas.getContext('2d');
    this.onPick = onPick;
    this.img = null;
    this.job = null;
    this.heat = null;
    this.mask = null;
    this.showHeat = true;
    this.cands = [];
    this.steps = [];
    this.cursor = 0;
    this.anim = null;
    this.raf = 0;
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener('pointerdown', (e) => this.pick(e));
    document.fonts?.ready.then(() => this.draw());
  }

  dims() {
    if (this.job) return { W: this.job.width, H: this.job.height };
    return this.img ? { W: this.img.naturalWidth, H: this.img.naturalHeight } : { W: 4, H: 3 };
  }

  setImage(url) {
    const img = new Image();
    img.onload = () => {
      this.img = img;
      this.rebuildPrivacy();
      this.draw();
    };
    img.src = url;
  }

  reset(job) {
    Object.assign(this, { job, heat: null, cands: [], steps: [], cursor: 0, anim: null });
    this.rebuildPrivacy();
    this.draw();
  }

  async setOverview(ov) {
    this.cands = Array.isArray(ov.candidates) ? ov.candidates : [];
    this.draw();
    this.heat = await colourise(ov.heatmap_png_b64);
    this.draw();
  }

  setHeat(on) {
    this.showHeat = on;
    this.draw();
  }

  /** Show the first k steps; `animate` draws the newest look: path, leader, callout and window dimensions. */
  setCursor(k, animate = false) {
    this.cursor = k;
    const step = this.steps[k - 1];
    if (animate && step && (pointOf(step) || step.window?.length === 4) && !reduced()) {
      this.anim = { k, t0: performance.now(), dur: 950 };
      this.raf ||= requestAnimationFrame(this.frame);
    } else {
      this.anim = null;
    }
    this.c.setAttribute('aria-label', step ? `Photo viewport, step ${k} of ${this.steps.length}: ${step.decision?.action ?? 'step'}${pointOf(step) ? ` at (${step.decision.args.x}, ${step.decision.args.y}) level ${step.decision.args.level}` : ''}` : 'Photo viewport with no steps shown');
    this.draw();
  }

  frame = (now) => {
    this.raf = 0;
    const a = this.anim;
    if (a && now - a.t0 < a.dur) {
      this.draw(now);
      this.raf = requestAnimationFrame(this.frame);
    } else {
      this.anim = null;
      this.draw();
    }
  };

  /** 0..1 through the newest step's drawing; 1 when nothing is animating. */
  progress(now) {
    const a = this.anim;
    return a && a.k === this.cursor ? Math.min(1, Math.max(0, (now - a.t0) / a.dur)) : 1;
  }

  resize() {
    const r = this.c.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.cw = r.width;
    this.ch = r.height;
    this.c.width = Math.round(r.width * this.dpr);
    this.c.height = Math.round(r.height * this.dpr);
    this.rebuildPrivacy();
    this.draw();
  }

  fit() {
    const { W, H } = this.dims();
    const pad = 12;
    const s = Math.min((this.cw - pad * 2) / W, (this.ch - pad * 2) / H);
    return { x: (this.cw - W * s) / 2, y: (this.ch - H * s) / 2, s, W, H };
  }

  /** Faces / people boxes come from the server, but the photo is the local file: blur them here. */
  rebuildPrivacy() {
    this.mask = null;
    const boxes = (this.job?.privacy?.boxes ?? []).filter((b) => Array.isArray(b) && b.length === 4 && b.every(num));
    if (!this.img || !this.cw || !boxes.length) return;
    const f = this.fit();
    const w = Math.round(f.W * f.s);
    const h = Math.round(f.H * f.s);
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const g = c.getContext('2d');
    for (const [x, y, bw, bh] of boxes) {
      g.save();
      g.beginPath();
      g.rect(x * f.s, y * f.s, bw * f.s, bh * f.s);
      g.clip();
      if ('filter' in g) {
        g.filter = 'blur(24px)';
        g.drawImage(this.img, 0, 0, w, h);
      } else {
        g.fillStyle = '#8EAEDB';
        g.fillRect(0, 0, w, h);
      }
      g.restore();
    }
    this.mask = c;
  }

  grid(g) {
    g.fillStyle = C.deep;
    g.fillRect(0, 0, this.cw, this.ch);
    g.lineWidth = 1;
    for (const [step, alpha] of [[16, 0.06], [80, 0.14]]) {
      g.strokeStyle = `rgba(232,241,255,${alpha})`;
      g.beginPath();
      for (let x = 0.5; x < this.cw; x += step) { g.moveTo(x, 0); g.lineTo(x, this.ch); }
      for (let y = 0.5; y < this.ch; y += step) { g.moveTo(0, y); g.lineTo(this.cw, y); }
      g.stroke();
    }
  }

  draw(now = performance.now()) {
    if (!this.cw) return;
    const { g } = this;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.grid(g);
    if (!this.img) {
      label(g, 'NO PHOTO YET · PICK ONE ON THE LEFT', this.cw / 2, this.ch / 2, { font: `500 13px ${MONO}`, edge: C.line3 });
      return;
    }
    const f = this.fit();
    const X = (x) => f.x + x * f.s;
    const Y = (y) => f.y + y * f.s;
    g.drawImage(this.img, f.x, f.y, f.W * f.s, f.H * f.s);
    if (this.mask) g.drawImage(this.mask, f.x, f.y);
    if (this.showHeat && this.heat) {
      g.globalAlpha = 0.7;
      g.drawImage(this.heat, f.x, f.y, f.W * f.s, f.H * f.s);
      g.globalAlpha = 1;
    }
    g.strokeStyle = C.line;
    g.lineWidth = 2.5;
    g.strokeRect(f.x - 1.25, f.y - 1.25, f.W * f.s + 2.5, f.H * f.s + 2.5);

    const t = this.progress(now);
    const shown = this.steps.slice(0, this.cursor);
    this.drawCandidates(g, X, Y, f, shown);
    const current = shown[shown.length - 1];
    const win = current?.window?.length === 4 ? current : null;
    if (win) this.drawWindow(g, X, Y, f, win);
    const co = this.drawPath(g, X, Y, shown, t);
    if (win) this.drawDims(g, X, Y, win, t);  // measured values and the callout stay readable over the path
    if (co) this.callout(g, ...co);
  }

  drawCandidates(g, X, Y, f, shown) {
    const seen = new Set(shown.map((s) => s.region));
    g.save();
    for (const c of this.cands) {
      if (!Array.isArray(c.box) || !c.box.every(num)) continue;
      const [bx, by, bw, bh] = c.box;
      const col = seen.has(c.id) ? C.amber : C.line2;
      g.setLineDash([8, 5]);
      g.lineWidth = 2;
      g.strokeStyle = col;
      g.strokeRect(X(bx), Y(by), bw * f.s, bh * f.s);
      g.setLineDash([]);
      g.font = `700 11px ${MONO}`;
      const text = String(c.id);
      const w = g.measureText(text).width + 10;
      g.fillStyle = C.deep;
      g.fillRect(X(bx), Y(by), w, 17);
      g.fillStyle = col;
      g.textAlign = 'left';
      g.textBaseline = 'middle';
      g.fillText(text, X(bx) + 5, Y(by) + 9);
    }
    g.restore();
  }

  drawWindow(g, X, Y, f, step) {
    const [wx, wy, ww, wh] = step.window;
    const x = X(wx), y = Y(wy), w = ww * f.s, h = wh * f.s;
    g.save();
    g.save();
    g.beginPath();
    g.rect(f.x, f.y, f.W * f.s, f.H * f.s);
    g.clip();
    g.fillStyle = 'rgba(8,42,85,.5)';
    g.beginPath();
    g.rect(f.x, f.y, f.W * f.s, f.H * f.s);
    g.rect(x, y, w, h);
    g.fill('evenodd');
    g.restore();
    g.strokeStyle = C.line;
    g.lineWidth = 2.5;
    g.strokeRect(x, y, w, h);
    const k = Math.min(20, w / 4, h / 4);
    g.strokeStyle = C.amber;
    g.lineWidth = 3;
    g.beginPath();
    for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
      g.moveTo(cx + dx * k, cy + dy * 5);
      g.lineTo(cx + dx * 5, cy + dy * 5);
      g.lineTo(cx + dx * 5, cy + dy * k);
    }
    g.stroke();
    g.restore();
  }

  /** Dimension arrows that extend to the window's L0 size while the number counts up, plus the level tag. */
  drawDims(g, X, Y, step, t) {
    const [wx, wy, ww, wh] = step.window;
    const x = X(wx), y = Y(wy), w = ww * (X(1) - X(0)), h = wh * (Y(1) - Y(0));
    g.save();
    const d = ease(phase(t, 0.12, 0.85));
    this.dimH(g, x, x + w, y, ww, d);
    this.dimV(g, y, y + h, x, wh, d);
    const level = step.decision?.args?.level;
    if (num(level)) {
      const text = `L${level} · 1:${2 ** level}`;
      g.font = `700 11.5px ${MONO}`;
      const tw = g.measureText(text).width + 12;
      const tx = Math.min(Math.max(x + 10, 26), this.cw - tw - 4);
      const ty = Math.min(Math.max(y + 12, 34), this.ch - 22);
      label(g, text, tx + tw / 2, ty + 9, { bg: C.amber, fg: C.ink, edge: null });
    }
    g.restore();
  }

  /** Horizontal dimension above the window (or just inside it when the window runs off the top). */
  dimH(g, x0, x1, edge, value, d) {
    const yy = edge - 16 >= 10 ? edge - 16 : Math.min(this.ch - 10, Math.max(10, edge + 22));
    const mid = (x0 + x1) / 2;
    const a = mid - ((x1 - x0) / 2) * d;
    const b = mid + ((x1 - x0) / 2) * d;
    const va = Math.max(a, 4);
    const vb = Math.min(b, this.cw - 4);
    if (vb <= va) return;
    g.setLineDash([4, 3]);
    g.lineWidth = 1.5;
    g.strokeStyle = C.line3;
    g.beginPath();
    for (const x of [a, b]) if (x === Math.min(Math.max(x, 4), this.cw - 4) && Math.abs(yy - edge) > 2) { g.moveTo(x, edge); g.lineTo(x, yy + Math.sign(yy - edge) * 6); }
    g.stroke();
    g.setLineDash([]);
    g.strokeStyle = C.line;
    g.fillStyle = C.line;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(va, yy);
    g.lineTo(vb, yy);
    g.stroke();
    if (a >= 4) arrow(g, a, yy, Math.PI);
    if (b <= this.cw - 4) arrow(g, b, yy, 0);
    label(g, `${Math.round(value * d)} px`, (va + vb) / 2, yy);
  }

  /** Vertical dimension left of the window (or just inside it when the window runs off the left). */
  dimV(g, y0, y1, edge, value, d) {
    const xx = edge - 16 >= 10 ? edge - 16 : Math.min(this.cw - 10, Math.max(10, edge + 22));
    const mid = (y0 + y1) / 2;
    const a = mid - ((y1 - y0) / 2) * d;
    const b = mid + ((y1 - y0) / 2) * d;
    const va = Math.max(a, 4);
    const vb = Math.min(b, this.ch - 4);
    if (vb <= va) return;
    g.setLineDash([4, 3]);
    g.lineWidth = 1.5;
    g.strokeStyle = C.line3;
    g.beginPath();
    for (const y of [a, b]) if (y === Math.min(Math.max(y, 4), this.ch - 4) && Math.abs(xx - edge) > 2) { g.moveTo(edge, y); g.lineTo(xx + Math.sign(xx - edge) * 6, y); }
    g.stroke();
    g.setLineDash([]);
    g.strokeStyle = C.line;
    g.fillStyle = C.line;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(xx, va);
    g.lineTo(xx, vb);
    g.stroke();
    if (a >= 4) arrow(g, xx, a, -Math.PI / 2);
    if (b <= this.ch - 4) arrow(g, xx, b, Math.PI / 2);
    g.save();
    g.translate(xx, (va + vb) / 2);
    g.rotate(-Math.PI / 2);
    label(g, `${Math.round(value * d)} px`, 0, 0);
    g.restore();
  }

  /** Gaze point, and where its numbered circle sits at the end of a leader. Leaders turn by step so close looks stay legible. */
  layout(pts, X, Y) {
    const DIRS = [[1, -1], [-1, -1], [1, 1], [-1, 1]];
    const base = [];
    return pts.map((s, i) => {
      const px = X(pointOf(s).x);
      const py = Y(pointOf(s).y);
      const dup = base.filter(([bx, by]) => Math.hypot(bx - px, by - py) < 8).length;
      base.push([px, py]);
      const len = Math.round(Math.min(30, Math.max(20, this.cw * 0.035))) + dup * 14;
      let [dx, dy] = DIRS[(i + dup) % 4];
      if (px + dx * len > this.cw - 16 || px + dx * len < 16) dx = -dx;
      if (py + dy * len > this.ch - 16 || py + dy * len < 16) dy = -dy;
      return { px, py, cx: px + dx * len, cy: py + dy * len, dx, dy };
    });
  }

  drawPath(g, X, Y, shown, t) {
    const pts = shown.filter(pointOf);
    if (!pts.length) return;
    const at = this.layout(pts, X, Y);
    const tt = pts[pts.length - 1] === shown[shown.length - 1] ? t : 1;
    const last = at.length - 1;
    g.save();
    g.lineJoin = 'round';
    g.lineCap = 'round';
    // the gaze path, pencilled from look to look; the newest segment draws itself
    g.strokeStyle = C.amber;
    g.lineWidth = 2.5;
    g.beginPath();
    at.forEach((p, i) => {
      if (!i) return g.moveTo(p.px, p.py);
      const k = i === last ? ease(phase(tt, 0, 0.35)) : 1;
      g.lineTo(lerp(at[i - 1].px, p.px, k), lerp(at[i - 1].py, p.py, k));
    });
    g.stroke();
    at.forEach((p, i) => {
      const newest = i === last;
      const show = newest ? phase(tt, 0.3, 0.42) : 1;
      const lead = newest ? ease(phase(tt, 0.35, 0.6)) : 1;
      const snap = newest ? phase(tt, 0.55, 0.78) : 1;
      if (show > 0) this.cross(g, p.px, p.py, show, newest);
      if (lead > 0) {
        const r = Math.hypot(p.cx - p.px, p.cy - p.py);
        const k = Math.max(0, (r - 13) / r) * lead;
        g.strokeStyle = newest ? C.amber : C.line2;
        g.lineWidth = 2;
        g.beginPath();
        g.moveTo(p.px, p.py);
        g.lineTo(lerp(p.px, p.cx, k), lerp(p.py, p.cy, k));
        g.stroke();
      }
      if (snap > 0) this.circle(g, p.cx, p.cy, pts[i], newest, back(snap));
    });
    g.restore();
    const k = ease(phase(tt, 0.72, 0.98));
    return k > 0 ? [at[last], pts[last], k] : null;
  }

  cross(g, x, y, k, current) {
    g.strokeStyle = current ? C.amber : C.line;
    g.lineWidth = 2;
    g.beginPath();
    g.arc(x, y, 4.5 * k, 0, Math.PI * 2);
    g.moveTo(x - 9 * k, y);
    g.lineTo(x + 9 * k, y);
    g.moveTo(x, y - 9 * k);
    g.lineTo(x, y + 9 * k);
    g.stroke();
  }

  circle(g, x, y, step, current, scale) {
    const r = (this.cw < 500 ? 10 : 12) * scale;
    const ring = RING[step.decision?.action] ?? C.line;
    g.fillStyle = current ? C.amber : C.deep;
    g.strokeStyle = current ? C.ink : ring;
    g.lineWidth = 2.5;
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    if (current) {
      g.setLineDash([3, 3]);
      g.strokeStyle = C.amber;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(x, y, r + 5, 0, Math.PI * 2);
      g.stroke();
      g.setLineDash([]);
    }
    g.fillStyle = current ? C.ink : ring;
    g.font = `700 ${Math.round((this.cw < 500 ? 11 : 12) * Math.min(scale, 1.3))}px ${MONO}`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(String(step.step), x, y + 1);
  }

  /** "FIG. n · ACTION · Ln" snapped onto a shoulder line off the newest numbered circle. */
  callout(g, p, step, k) {
    const a = step.decision?.args ?? {};
    const text = `FIG. ${step.step} · ${String(step.decision?.action ?? 'step').toUpperCase()}${num(a.level) ? ` · L${a.level}` : ''}`;
    g.font = `700 11.5px ${MONO}`;
    const w = g.measureText(text).width + 14;
    let dir = 1;  // to the right: the window's own dimensions sit above and to the left
    if (dir > 0 && p.cx + 18 + w > this.cw - 4) dir = -1;
    if (dir < 0 && p.cx - 18 - w < 4) dir = 1;
    const x0 = p.cx + dir * 18;
    const x1 = x0 + dir * w * k;
    const y = p.cy;
    g.strokeStyle = C.amber;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(p.cx + dir * 12, y);
    g.lineTo(x1, y);
    g.stroke();
    g.globalAlpha = k;
    const left = dir > 0 ? x0 : x0 - w;
    g.fillStyle = 'rgba(8,42,85,.88)';
    g.fillRect(left, y - 21, w, 18);
    g.fillStyle = C.amber;
    g.textAlign = 'left';
    g.textBaseline = 'middle';
    g.fillText(text, left + 7 - dir * 6 * (1 - k), y - 11.5);
    g.globalAlpha = 1;
  }

  pick(e) {
    const r = this.c.getBoundingClientRect();
    const f = this.fit();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    const shown = this.steps.slice(0, this.cursor).filter(pointOf);
    const at = this.layout(shown, (x) => f.x + x * f.s, (y) => f.y + y * f.s);
    let best = -1;
    let bestD = 18;
    at.forEach((p, i) => {
      const d = Math.min(Math.hypot(p.cx - px, p.cy - py), Math.hypot(p.px - px, p.py - py));
      if (d <= bestD) [best, bestD] = [i, d];
    });
    if (best >= 0) this.onPick(this.steps.indexOf(shown[best]) + 1);
  }
}
