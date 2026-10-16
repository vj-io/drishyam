// The photo canvas: fitted image, saliency heatmap, candidate boxes, privacy blur, gaze window,
// numbered gaze path and the glossy orb that bounces to each new gaze point.
import { dataUrl, num } from './dom.js';
import { reduced } from './fx.js';

const C = { cobalt: '#2440FF', deep: '#14248F', pink: '#FF4FD8', lime: '#B6FF3B', ice: '#BFE6FF', ink: '#0B1450' };
const DOT = { gaze: [C.pink, C.ink], escalate: [C.cobalt, '#fff'], verify: [C.lime, C.ink] };
const MONO = '12px "JetBrains Mono", ui-monospace, monospace';
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

export const pointOf = (step) => {
  const a = step.decision?.args;
  return num(a?.x) && num(a?.y) ? { x: a.x, y: a.y } : null;
};

const RAMP = [[0, [36, 64, 255]], [0.5, [255, 79, 216]], [1, [182, 255, 59]]];
function ramp(t) {
  const [[t0, c0], [t1, c1]] = t < 0.5 ? [RAMP[0], RAMP[1]] : [RAMP[1], RAMP[2]];
  const k = (t - t0) / (t1 - t0);
  return c0.map((v, i) => Math.round(lerp(v, c1[i], k)));
}

/** Saliency PNG (0-255) -> RGBA canvas using a cobalt > pink > lime ramp. */
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

  /** Show the first k steps; `animate` bounces the orb to the newest point. */
  setCursor(k, animate = false) {
    this.cursor = k;
    const step = this.steps[k - 1];
    if (animate && step && pointOf(step) && !reduced()) {
      const from = this.steps.slice(0, k - 1).reverse().find(pointOf);
      this.anim = { to: pointOf(step), from: from ? pointOf(from) : null, t0: performance.now(), k };
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
    this.draw(now);
    if (a && now - a.t0 < a.dur + 260) this.raf = requestAnimationFrame(this.frame);
    else this.anim = null;
  };

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
        g.fillStyle = '#8FA3B8';
        g.fillRect(0, 0, w, h);
      }
      g.restore();
    }
    this.mask = c;
  }

  draw(now = performance.now()) {
    if (!this.cw) return;
    const { g } = this;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.fillStyle = C.ink;
    g.fillRect(0, 0, this.cw, this.ch);
    g.fillStyle = 'rgba(191,230,255,.13)';
    for (let x = 10; x < this.cw; x += 20) for (let y = 10; y < this.ch; y += 20) g.fillRect(x, y, 2, 2);
    if (!this.img) {
      g.fillStyle = C.ice;
      g.font = '14px "JetBrains Mono", monospace';
      g.textAlign = 'center';
      g.fillText('no photo yet · pick one on the left', this.cw / 2, this.ch / 2);
      return;
    }
    const f = this.fit();
    const X = (x) => f.x + x * f.s;
    const Y = (y) => f.y + y * f.s;
    g.drawImage(this.img, f.x, f.y, f.W * f.s, f.H * f.s);
    if (this.mask) g.drawImage(this.mask, f.x, f.y);
    if (this.showHeat && this.heat) {
      g.globalAlpha = 0.72;
      g.drawImage(this.heat, f.x, f.y, f.W * f.s, f.H * f.s);
      g.globalAlpha = 1;
    }
    g.strokeStyle = '#fff';
    g.lineWidth = 2;
    g.strokeRect(f.x - 1, f.y - 1, f.W * f.s + 2, f.H * f.s + 2);

    const shown = this.steps.slice(0, this.cursor);
    this.drawCandidates(g, X, Y, f, shown);
    const current = shown[shown.length - 1];
    if (current?.window?.length === 4) this.drawWindow(g, X, Y, f, current);
    this.drawPath(g, X, Y, shown, now);
  }

  drawCandidates(g, X, Y, f, shown) {
    const seen = new Set(shown.map((s) => s.region));
    g.save();
    g.font = MONO;
    g.textAlign = 'left';
    g.textBaseline = 'middle';
    for (const c of this.cands) {
      if (!Array.isArray(c.box) || !c.box.every(num)) continue;
      const [bx, by, bw, bh] = c.box;
      const col = seen.has(c.id) ? C.lime : C.ice;
      g.setLineDash([7, 5]);
      g.lineWidth = 1.6;
      g.strokeStyle = col;
      g.strokeRect(X(bx), Y(by), bw * f.s, bh * f.s);
      g.setLineDash([]);
      const label = String(c.id);
      const w = g.measureText(label).width + 10;
      g.fillStyle = 'rgba(11,20,80,.85)';
      g.fillRect(X(bx), Y(by), w, 17);
      g.fillStyle = col;
      g.fillText(label, X(bx) + 5, Y(by) + 9);
    }
    g.restore();
  }

  drawWindow(g, X, Y, f, step) {
    const [wx, wy, ww, wh] = step.window;
    const x = X(wx), y = Y(wy), w = ww * f.s, h = wh * f.s;
    g.save();
    g.fillStyle = 'rgba(11,20,80,.38)';
    g.beginPath();
    g.rect(f.x, f.y, f.W * f.s, f.H * f.s);
    g.rect(x, y, w, h);
    g.fill('evenodd');
    const grad = g.createLinearGradient(x, y, x + w, y + h);
    grad.addColorStop(0, '#F4F7FA');
    grad.addColorStop(0.5, '#8FA3B8');
    grad.addColorStop(1, '#F4F7FA');
    g.strokeStyle = grad;
    g.lineWidth = 5;
    g.strokeRect(x, y, w, h);
    g.strokeStyle = C.cobalt;
    g.lineWidth = 1.5;
    g.strokeRect(x + 3.5, y + 3.5, w - 7, h - 7);
    const level = step.decision?.args?.level;
    if (num(level)) {
      g.font = `bold ${MONO}`;
      g.textBaseline = 'middle';
      const ly = y - 22 > f.y ? y - 22 : y + 5;
      g.fillStyle = C.cobalt;
      g.beginPath();
      g.roundRect(x, ly, 34, 18, 9);
      g.fill();
      g.strokeStyle = '#fff';
      g.lineWidth = 1.5;
      g.stroke();
      g.fillStyle = '#fff';
      g.textAlign = 'center';
      g.fillText(`L${level}`, x + 17, ly + 9.5);
    }
    g.restore();
  }

  /** Canvas positions for each step with a point; dots that land on the same spot fan out so every number stays visible. */
  layout(pts, X, Y) {
    const base = [];
    return pts.map((s) => {
      const x = X(pointOf(s).x);
      const y = Y(pointOf(s).y);
      const dup = base.filter(([bx, by]) => Math.hypot(bx - x, by - y) < 14).length;
      base.push([x, y]);
      return dup ? [x + Math.cos(dup * 2.4) * 15, y + Math.sin(dup * 2.4) * 15] : [x, y];
    });
  }

  drawPath(g, X, Y, shown, now) {
    const pts = shown.filter(pointOf);
    const at = this.layout(pts, X, Y);
    const a = this.anim;
    let flight = null;
    if (a && pts.length && pts[pts.length - 1] === this.steps[a.k - 1]) {
      const to = at[at.length - 1];
      const from = at.length > 1 ? at[at.length - 2] : [to[0], to[1] - 120];
      const dist = Math.hypot(to[0] - from[0], to[1] - from[1]);
      a.dur = Math.min(850, Math.max(430, 330 + dist * 0.8));
      const t = Math.min(1, Math.max(0, (now - a.t0) / a.dur));
      flight = { t, dist, x: lerp(from[0], to[0], ease(t)), y: lerp(from[1], to[1], ease(t)), landed: t >= 1 };
    }
    const inFlight = flight && !flight.landed;
    const line = inFlight ? at.slice(0, -1) : at.slice();
    if (inFlight && at.length > 1) line.push([flight.x, flight.y]);
    if (line.length > 1) {
      g.save();
      g.lineJoin = 'round';
      g.lineCap = 'round';
      for (const [w, col] of [[7, C.deep], [3.2, C.pink]]) {
        g.lineWidth = w;
        g.strokeStyle = col;
        g.beginPath();
        line.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
        g.stroke();
      }
      g.restore();
    }
    const dots = inFlight ? pts.slice(0, -1) : pts;
    const popT = flight?.landed ? Math.min(1, (now - a.t0 - a.dur) / 220) : 1;
    dots.forEach((s, i) => {
      const last = i === dots.length - 1;
      this.dot(g, at[i][0], at[i][1], s, last, last && flight?.landed ? 1 + 0.45 * (1 - popT) : 1);
    });
    if (inFlight) this.orb(g, flight.x, flight.y, 4 * Math.min(90, Math.max(28, flight.dist * 0.3)) * flight.t * (1 - flight.t));
  }

  dot(g, x, y, step, current, scale = 1) {
    const [fill, text] = DOT[step.decision?.action] ?? DOT.gaze;
    const r = (current ? 13 : 11) * scale;
    g.save();
    if (current) {
      g.strokeStyle = '#fff';
      g.lineWidth = 3;
      g.beginPath();
      g.arc(x, y, r + 5, 0, Math.PI * 2);
      g.stroke();
    }
    const grad = g.createRadialGradient(x - r * 0.35, y - r * 0.4, 1, x, y, r);
    grad.addColorStop(0, '#fff');
    grad.addColorStop(0.28, fill);
    grad.addColorStop(1, fill);
    g.fillStyle = grad;
    g.strokeStyle = C.deep;
    g.lineWidth = 2;
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    g.fillStyle = text;
    g.font = `bold ${MONO}`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(String(step.step), x, y + 1);
    g.restore();
  }

  orb(g, x, y, hop) {
    g.save();
    g.fillStyle = `rgba(11,20,80,${0.35 - Math.min(0.25, hop / 400)})`;
    g.beginPath();
    g.ellipse(x, y + 4, 14 - hop / 14, 5, 0, 0, Math.PI * 2);
    g.fill();
    const oy = y - 14 - hop;
    const grad = g.createRadialGradient(x - 5, oy - 6, 1, x, oy, 16);
    grad.addColorStop(0, '#fff');
    grad.addColorStop(0.3, '#FFB4F0');
    grad.addColorStop(1, '#D61FB0');
    g.fillStyle = grad;
    g.strokeStyle = C.deep;
    g.lineWidth = 2;
    g.beginPath();
    g.arc(x, oy, 14, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    g.restore();
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
    at.forEach(([x, y], i) => {
      const d = Math.hypot(x - px, y - py);
      if (d <= bestD) [best, bestD] = [i, d];
    });
    if (best >= 0) this.onPick(this.steps.indexOf(shown[best]) + 1);
  }
}
