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
const around = (x, y, r) => ({ x: x - r, y: y - r, w: 2 * r, h: 2 * r });
const hit = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
const overlap = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));

/** The 512 px, 20 %-overlap L0 tiles exhaustive tiling would read (evals/harness.py slice_boxes, after SAHI). */
export function tileBoxes(W, H, size = 512, overlap = 0.2) {
  const ov = Math.trunc(overlap * size);
  const starts = (n) => {
    const out = [];
    for (let lo = 0, hi = 0; hi < n; lo = hi - ov) {
      hi = lo + size;
      out.push(Math.max(0, Math.min(n, hi) - size));
    }
    return out;
  };
  return starts(H).flatMap((y) => starts(W).map((x) => [x, y, Math.min(size, W), Math.min(size, H)]));
}

/** A look window [x, y, w, h] in L0 px, cut to the photo: a coarse look can reach past its edges. */
const clip = ([x, y, w, h], W, H) => {
  const x0 = Math.max(0, x), y0 = Math.max(0, y);
  return [x0, y0, Math.max(0, Math.min(W, x + w) - x0), Math.max(0, Math.min(H, y + h) - y0)];
};

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

/** Text in a filled box, centred on (x, y). Returns the box. */
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
  return { x: x - w / 2, y: y - 9, w, h: 18 };
}

export class Viewport {
  constructor(canvas, onPick) {
    this.c = canvas;
    this.g = canvas.getContext('2d');
    this.onPick = onPick;
    this.fig = canvas.closest('.win')?.querySelector('.win__fig')?.textContent.trim() ?? 'FIG.';  // gaze step n is FIG. 2.3.n
    this.occ = [];  // what is drawn so far (balloons, crosses, labels), for the labels placed after it to avoid
    this.img = null;
    this.job = null;
    this.heat = null;
    this.mask = null;
    this.showHeat = true;
    this.showTiles = false;
    this.focusBox = null;  // a finding or candidate the viewer asked to see: [x, y, w, h] in L0 px
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
    Object.assign(this, { job, heat: null, cands: [], steps: [], cursor: 0, anim: null, focusBox: null, showTiles: false });
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

  setTiles(on) {
    this.showTiles = on;
    this.draw();
  }

  /** Outline one L0 box (a finding, or a region it never zoomed into) until the next step is picked. */
  highlight(box) {
    this.focusBox = Array.isArray(box) && box.length === 4 && box.every(num) ? box : null;
    this.draw();
  }

  /** Show the first k steps; `animate` draws the newest look: path, leader, callout and window dimensions. */
  setCursor(k, animate = false) {
    this.cursor = k;
    this.focusBox = null;
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
    this.narrow = r.width < 500;  // phones: no FIG. callout (the step detail below carries it), one dimension only
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
      label(g, 'NO PHOTO YET · PICK ONE FIRST', this.cw / 2, this.ch / 2, { font: `500 13px ${MONO}`, edge: C.line3 });
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
    const seen = new Set(shown.map((s) => s.region));
    const boxes = this.cands.filter((c) => Array.isArray(c.box) && c.box.length === 4 && c.box.every(num));
    this.occ = [];
    if (this.showTiles) this.drawTiles(g, X, Y, f);
    this.drawCandidates(g, X, Y, f, boxes, seen);
    const current = shown[shown.length - 1];
    let win = current?.window?.length === 4 && current.window.every(num) ? clip(current.window, f.W, f.H) : null;
    const prev = shown.slice(0, -1).reverse().find((s) => s.window?.length === 4 && s.window.every(num));
    if (win && prev && t < 1) {  // the look window slides from the last look to this one
      const from = clip(prev.window, f.W, f.H);
      const k = ease(phase(t, 0, 0.4));
      win = win.map((v, i) => lerp(from[i], v, k));
    }
    if (win) this.drawWindow(g, X, Y, f, win);
    const co = this.drawPath(g, X, Y, shown, t);
    if (win) this.drawDims(g, X, Y, win, current.decision?.args?.level, t);  // measured values and the callout stay readable over the path
    if (co && !this.narrow) this.callout(g, ...co);
    this.drawTags(g, X, Y, f, boxes, seen);
    if (this.focusBox) this.drawFocus(g, X, Y, f, this.focusBox);
  }

  /** The tiles a full tiling scan would read, faint, so the agent's few looks can be compared with them. */
  drawTiles(g, X, Y, f) {
    g.save();
    g.strokeStyle = 'rgba(232,241,255,.42)';
    g.lineWidth = 1;
    for (const [x, y, w, h] of tileBoxes(f.W, f.H)) g.strokeRect(X(x) + 0.5, Y(y) + 0.5, w * f.s - 1, h * f.s - 1);
    g.restore();
  }

  drawFocus(g, X, Y, f, [x, y, w, h]) {
    g.save();
    g.strokeStyle = C.amber;
    g.lineWidth = 4;
    g.setLineDash([]);
    g.strokeRect(X(x) - 3, Y(y) - 3, w * f.s + 6, h * f.s + 6);
    g.restore();
    label(g, 'HERE', Math.min(Math.max(X(x) + 24, 28), this.cw - 28), Math.max(Y(y) - 14, 12), { bg: C.amber, fg: C.ink, edge: null });
  }

  /** Candidate regions as dashed boxes, stopped at the photo's frame like any line on a drawing. */
  drawCandidates(g, X, Y, f, boxes, seen) {
    g.save();
    g.beginPath();
    g.rect(f.x, f.y, f.W * f.s, f.H * f.s);
    g.clip();
    g.setLineDash([8, 5]);
    g.lineWidth = 2;
    for (const c of boxes) {
      const [bx, by, bw, bh] = c.box;
      g.strokeStyle = seen.has(c.id) ? C.amber : C.line2;
      g.strokeRect(X(bx), Y(by), bw * f.s, bh * f.s);
    }
    g.restore();
  }

  /** Region tags go on last, in the first corner of their box (inside the photo) that covers no balloon or label; none fits, no tag. */
  drawTags(g, X, Y, f, boxes, seen) {
    g.save();
    g.font = `700 11px ${MONO}`;
    g.textAlign = 'left';
    g.textBaseline = 'middle';
    const fx1 = f.x + f.W * f.s;
    const fy1 = f.y + f.H * f.s;
    for (const c of boxes) {
      const [bx, by, bw, bh] = c.box;
      const text = String(c.id);
      const w = g.measureText(text).width + 10;
      const h = 17;
      const l = Math.max(X(bx), f.x);
      const t = Math.max(Y(by), f.y);
      const r = Math.min(X(bx + bw), fx1) - w;
      const b = Math.min(Y(by + bh), fy1) - h;
      const spot = [[l, t], [r, t], [l, b], [r, b]].map(([x, y]) => ({ x, y, w, h })).find((q) => !this.occ.some((o) => hit(q, o)));
      if (!spot) continue;
      this.occ.push(spot);
      g.fillStyle = C.deep;
      g.fillRect(spot.x, spot.y, w, h);
      g.fillStyle = seen.has(c.id) ? C.amber : C.line2;
      g.fillText(text, spot.x + 5, spot.y + 9);
    }
    g.restore();
  }

  drawWindow(g, X, Y, f, [wx, wy, ww, wh]) {
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
  drawDims(g, X, Y, [wx, wy, ww, wh], level, t) {
    const x = X(wx), y = Y(wy), w = ww * (X(1) - X(0)), h = wh * (Y(1) - Y(0));
    g.save();
    const d = ease(phase(t, 0.12, 0.85));
    this.dimH(g, x, x + w, y, y + h, ww, d);
    if (!this.narrow) this.dimV(g, y, y + h, x, x + w, wh, d);
    if (num(level)) {
      const text = `L${level} · 1:${2 ** level}`;
      g.font = `700 11.5px ${MONO}`;
      const tw = g.measureText(text).width + 12;
      const at = (tx, ty) => ({ x: Math.min(Math.max(tx, 4), this.cw - tw - 4), y: Math.min(Math.max(ty, 4), this.ch - 22), w: tw, h: 18 });
      // inside the window's top-left corner, else below it, inside its foot or beside it: whichever covers least
      const spot = this.least([at(Math.max(x + 10, 26), Math.max(y + 12, 34)), at(x, y + h + 6), at(x + 10, y + h - 30), at(x + w + 6, y), at(x - tw - 6, y)]);
      this.occ.push(label(g, text, spot.x + tw / 2, spot.y + 9, { bg: C.amber, fg: C.ink, edge: null }));
    }
    g.restore();
  }

  /** The candidate box (first choice first) that covers least of what is already placed: balloons, crosses, labels. */
  least(boxes) {
    const cost = (b) => this.occ.reduce((n, o) => n + overlap(b, o), 0);
    return boxes.reduce((best, b) => (cost(b) < cost(best) ? b : best));
  }

  /**
   * Which side of the window a dimension goes on (`sides`: first choice first) and how far its label slides from
   * mid-span, so the label clears every balloon and label already placed. As on paper, a label that fits nowhere
   * between the arrows goes just outside one of them. Reserves the final label box.
   */
  place(lo, hi, size, sides, box, limit) {
    const mid = (lo + hi) / 2;
    const inside = [0, -1, 1, -2, 2].map((k) => Math.min(Math.max(mid + k * (size + 10), lo + size / 2), Math.max(lo + size / 2, hi - size / 2)));
    const spots = [...inside, lo - size / 2 - 12, hi + size / 2 + 12].map((v) => Math.min(Math.max(v, size / 2 + 4), limit - size / 2 - 4));
    const best = this.least(sides.flatMap((side) => spots.map((c) => Object.assign(box(side, c), { side, c }))));
    this.occ.push(best);
    return [best.side, best.c - mid];
  }

  /** Horizontal dimension above the window, else below it (or just inside it when the window runs off the top). */
  dimH(g, x0, x1, top, bottom, value, d) {
    g.font = `700 11.5px ${MONO}`;
    const fw = g.measureText(`${Math.round(value)} px`).width + 12;  // the final label, so nothing hops while it counts up
    const sides = top - 16 >= 10 ? [top - 16, ...(bottom + 16 <= this.ch - 10 ? [bottom + 16] : [])] : [Math.min(this.ch - 10, Math.max(10, top + 22))];
    const [yy, off] = this.place(Math.max(x0, 4), Math.min(x1, this.cw - 4), fw, sides, (row, c) => ({ x: c - fw / 2, y: row - 9, w: fw, h: 18 }), this.cw);
    const edge = yy > bottom ? bottom : top;
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
    const lx = (va + vb) / 2 + off * d;  // the line runs on to a label set outside the arrows
    g.beginPath();
    g.moveTo(Math.min(va, lx), yy);
    g.lineTo(Math.max(vb, lx), yy);
    g.stroke();
    if (a >= 4) arrow(g, a, yy, Math.PI);
    if (b <= this.cw - 4) arrow(g, b, yy, 0);
    label(g, `${Math.round(value * d)} px`, lx, yy);
  }

  /** Vertical dimension left of the window, else right of it (or just inside it when the window runs off the left). */
  dimV(g, y0, y1, left, right, value, d) {
    g.font = `700 11.5px ${MONO}`;
    const fw = g.measureText(`${Math.round(value)} px`).width + 12;
    const sides = left - 16 >= 10 ? [left - 16, ...(right + 16 <= this.cw - 10 ? [right + 16] : [])] : [Math.min(this.cw - 10, Math.max(10, left + 22))];
    const [xx, off] = this.place(Math.max(y0, 4), Math.min(y1, this.ch - 4), fw, sides, (col, c) => ({ x: col - 9, y: c - fw / 2, w: 18, h: fw }), this.ch);
    const edge = xx > right ? right : left;
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
    const ly = (va + vb) / 2 + off * d;
    g.beginPath();
    g.moveTo(xx, Math.min(va, ly));
    g.lineTo(xx, Math.max(vb, ly));
    g.stroke();
    if (a >= 4) arrow(g, xx, a, -Math.PI / 2);
    if (b <= this.ch - 4) arrow(g, xx, b, Math.PI / 2);
    g.save();
    g.translate(xx, ly);
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
    for (const p of at) this.occ.push(around(p.cx, p.cy, 17), around(p.px, p.py, 9));
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

  /** "FIG. 2.3.n · ACTION · Ln" snapped onto a shoulder line off the newest numbered circle. */
  callout(g, p, step, k) {
    const a = step.decision?.args ?? {};
    const text = `${this.fig}.${step.step} · ${String(step.decision?.action ?? 'step').toUpperCase()}${num(a.level) ? ` · L${a.level}` : ''}`;
    g.font = `700 11.5px ${MONO}`;
    const w = g.measureText(text).width + 14;
    const y = p.cy;
    // right of the balloon and above its shoulder first (the window's dimensions sit above and to the left), then
    // the other side, then below the shoulder: whichever spot inside the canvas covers least
    const box = ([dir, dy]) => ({ x: dir > 0 ? p.cx + 18 : p.cx - 18 - w, y: y + dy, w, h: 18 });
    const fits = (o) => box(o).x >= 4 && box(o).x + w <= this.cw - 4;
    const opts = [[1, -21], [-1, -21], [1, 3], [-1, 3]];
    const [dir, dy] = this.least((opts.some(fits) ? opts.filter(fits) : opts).map((o) => Object.assign(box(o), { o }))).o;
    const { x: left } = box([dir, dy]);
    this.occ.push(box([dir, dy]));
    g.strokeStyle = C.amber;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(p.cx + dir * 12, y);
    g.lineTo(p.cx + dir * (18 + w * k), y);
    g.stroke();
    g.globalAlpha = k;
    g.fillStyle = 'rgba(8,42,85,.88)';
    g.fillRect(left, y + dy, w, 18);
    g.fillStyle = C.amber;
    g.textAlign = 'left';
    g.textBaseline = 'middle';
    g.fillText(text, left + 7 - dir * 6 * (1 - k), y + dy + 9.5);
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
