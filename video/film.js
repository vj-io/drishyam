// Drishyam submission film. Every frame is a pure function of t (window.render). The words and numbers come from
// window.FILM (brag-output/work/film-data.js, written by video/build.py from the result files); the live footage is
// the screencast video/record.py took of https://d1xg58guluup8c.cloudfront.net. Helpers lifted from motion-kit.
'use strict';
const F = window.FILM, D = F.D, SHORT = window.MODE === 'short';
const $ = (s, r = document) => r.querySelector(s);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const seg = (t, a, b) => clamp((t - a) / (b - a));
const eo = (x) => 1 - Math.pow(1 - x, 3);
const eio = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const spring = (x) => (x <= 0 ? 0 : 1 - Math.exp(-x * 8) * Math.cos(x * 20));
const bump = (x) => (x <= 0 || x >= 1 ? 0 : Math.sin(x * Math.PI));
const lerp = (a, b, p) => a + (b - a) * p;
const fmt = (v, d = 2) => Number(v).toFixed(d);
const pc = (v) => `${Math.round(v * 100)} %`;
function el(tag, cls, host, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (host) host.append(e); return e; }
function svgEl(tag, attrs, host) { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v); if (host) host.append(e); return e; }
const css = (e, o) => Object.assign(e.style, o);
const REC = '../brag-output/work/rec/';
const SFX = [];
const cue = (t, type, x) => SFX.push({ t: +t.toFixed(3), type, ...(x != null ? { x } : {}) });
const LINE = Object.fromEntries(F.lines.map((l) => [l.id, l]));
const at = (id) => LINE[id].t, end = (id) => LINE[id].t + LINE[id].dur;
const stage = $('#stage');
const loads = [];
function img(src, host, cls) { const i = el('img', cls, host); i.src = src; loads.push(i.decode().catch(() => {})); return i; }

// ------------------------------------------------------------------ footage: warp, frames, camera, cursor
const FR = F.frames, FT = FR.map((f) => f.t);
function frameAt(f) { let lo = 0, hi = FT.length - 1; if (f <= FT[0]) return 0; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (FT[m] <= f) lo = m; else hi = m - 1; } return lo; }
function segments(w) {   // [t0, t1, f0, f1] pieces, with the cuts between them
  const P = w.pts, out = [];
  for (let i = 0; i < P.length - 1; i++) {
    const a = P[i]; if (a.cut) continue;
    let j = i + 1, cutp = null; if (P[j].cut) { cutp = P[j]; j++; }
    const b = P[j];
    if (!cutp) { out.push({ t0: a.t, t1: b.t, f0: a.f, f1: b.f }); continue; }
    const da = cutp.a - a.f, db = b.f - cutp.b, tc = a.t + (b.t - a.t) * da / (da + db);
    out.push({ t0: a.t, t1: tc, f0: a.f, f1: cutp.a }, { t0: tc, t1: b.t, f0: cutp.b, f1: b.f, cutIn: true });
  }
  out.forEach((s) => { s.v = (s.f1 - s.f0) / (s.t1 - s.t0); });
  return out;
}
function warpAt(S, t) {
  let s = S[0]; for (const x of S) if (t >= x.t0) s = x;
  const f = s.f0 + (clamp(t, s.t0, s.t1) - s.t0) * s.v;
  return { f, s };
}
function filmOf(S, f) { for (const s of S) if (f >= Math.min(s.f0, s.f1) - 1e-6 && f <= Math.max(s.f0, s.f1) + 1e-6) return s.t0 + (f - s.f0) / s.v; return null; }

const MOVES = F.events.filter((e) => e.kind === 'move'), CLICKS = F.events.filter((e) => e.kind === 'click');
function cursorAt(f) {
  let m = null; for (const x of MOVES) if (x.t0 <= f) m = x;
  if (!m) return MOVES.length ? [MOVES[0].x0, MOVES[0].y0] : [900, 600];
  const p = eio(seg(f, m.t0, m.t));
  return [lerp(m.x0, m.x, p), lerp(m.y0, m.y, p)];
}

class Footage {
  // a browser window in the sheet's look; screen w x h; camera keyframes {f, c: [cx, cy, w]} in page CSS px
  constructor(host, w, h, url, warp, keys, opts = {}) {
    this.win = el('div', 'win', host); css(this.win, { width: w + 'px', height: h + 'px', zIndex: 1 });
    const bar = el('div', 'bar', this.win); el('i', '', bar); el('i', '', bar); el('i', '', bar);
    this.url = el('div', 'url', bar, url); this.tag = el('div', 'tag', bar, '');
    this.screen = el('div', 'screen', this.win); this.sw = w - 6; this.sh = h - 49 - 3;
    this.cam = el('div', 'cam', this.screen); css(this.cam, { width: '1280px', height: '1200px' });
    this.A = el('img', '', this.cam); this.B = el('img', '', this.cam);
    this.cur = svgEl('svg', { class: 'cur', viewBox: '-3 -3 34 46' }, this.screen);
    svgEl('path', { d: 'M0 0 L0 34 L9 26 L16 41 L22 38 L16 24 L28 24 Z', fill: '#0a0a0a', stroke: '#fff', 'stroke-width': 3, 'stroke-linejoin': 'round' }, this.cur);
    this.ring = el('div', 'ring', this.screen);
    this.S = segments(warp); this.w = warp; this.keys = keys.map((k) => ({ ...k, T: k.T ?? filmOf(this.S, k.f) })).filter((k) => k.T != null).sort((a, b) => a.T - b.T);
    this.overlays = el('div', 'abs', this.screen); css(this.overlays, { inset: 0 });
    this.opts = opts;
  }
  camAt(t) {
    const K = this.keys; let k = 0; while (k < K.length - 1 && t >= K[k + 1].T) k++;
    let c = K[k].c.slice();
    if (k < K.length - 1) {
      const n = K[k + 1], mv = Math.min(1.1, (n.T - K[k].T) * 0.7), p = eio(seg(t, n.T - mv, n.T));
      const drift = 1 - 0.07 * seg(t, K[k].T, n.T - mv);       // a held view keeps pushing in while it is read
      c = [lerp(c[0], n.c[0], p), lerp(c[1], n.c[1], p), lerp(c[2] * drift, n.c[2], p)];
    }
    const z = this.sw / c[2], h = this.sh / z;
    c[0] = clamp(c[0], c[2] / 2, 1280 - c[2] / 2); c[1] = clamp(c[1], h / 2, 1200 - h / 2);
    return [c[0], c[1], z];
  }
  toScreen(x, y, c) { return [this.sw / 2 + (x - c[0]) * c[2], this.sh / 2 + (y - c[1]) * c[2]]; }
  async draw(t) {
    const { f, s } = warpAt(this.S, t);
    const c = this.camAt(t); this.c = c; this.f = f;
    css(this.cam, { transform: `translate(${(this.sw / 2 - c[0] * c[2]).toFixed(2)}px,${(this.sh / 2 - c[1] * c[2]).toFixed(2)}px) scale(${c[2].toFixed(4)})` });
    // a cut is a 0.36 s cross-dissolve from the last frame before it to the footage after it
    let a = frameAt(f), b = null, o = 0;
    const cutS = this.S.find((x) => x.cutIn && Math.abs(t - x.t0) < 0.18);
    if (cutS) { const prev = this.S[this.S.indexOf(cutS) - 1]; o = seg(t, cutS.t0 - 0.18, cutS.t0 + 0.18); b = frameAt(Math.max(f, cutS.f0)); a = frameAt(prev.f1); }
    const want = [[this.A, a, 1], [this.B, b, o]];
    const waits = [];
    for (const [im, idx, op] of want) {
      if (idx == null || op <= 0) { im.style.opacity = 0; continue; }
      const src = REC + FR[idx].f;
      if (im.dataset.src !== src) { im.dataset.src = src; im.src = src; waits.push(im.decode().catch(() => {})); }
      im.style.opacity = op;
    }
    // speed, as shown: the footage is retimed to the voice between recorded events
    const v = Math.abs(s.v);
    this.tag.textContent = Math.abs(v - 1) < 0.15 ? 'REAL TIME' : `${v.toFixed(1)}× SPEED`;
    this.tag.className = 'tag' + (Math.abs(v - 1) < 0.15 ? ' rt' : '');
    // cursor
    const [px, py] = cursorAt(f), [sx, sy] = this.toScreen(px, py, c);
    const clk = CLICKS.filter((e) => e.t <= f + 0.05 && f - e.t < 0.6 * Math.max(1, v)).pop();
    const pr = clk ? bump(seg(f - clk.t, -0.05, 0.16 * Math.max(1, v))) : 0;
    const vis = this.opts.cursor !== false && sx > -40 && sx < this.sw + 40 && sy > -40 && sy < this.sh + 40;
    css(this.cur, { opacity: vis ? 1 : 0, transform: `translate(${sx.toFixed(1)}px,${sy.toFixed(1)}px) scale(${(1 - 0.16 * pr).toFixed(3)})` });
    const rp = clk ? seg(f - clk.t, 0, 0.5 * Math.max(1, v)) : 1;
    css(this.ring, { opacity: rp > 0 && rp < 1 && vis ? 1 - rp : 0, transform: `translate(${sx.toFixed(1)}px,${sy.toFixed(1)}px) scale(${(0.3 + eo(rp) * 1.2).toFixed(3)})` });
    await Promise.all(waits);
  }
  clicksSfx() {   // a click sound for each recorded click the film shows
    for (const e of CLICKS) { const T = filmOf(this.S, e.t); if (T != null && T >= this.w.t0 && T <= this.w.t1) cue(T, 'click'); }
  }
}

// a highlight box or callout in a footage window, tied to film times, placed in page coordinates
function railcall(n, big, label, t0, t1) { CALLS.push({ n, big, label, t0, t1 }); cue(t0, 'pop', typeof n === 'number' ? n : 3); }

// ------------------------------------------------------------------ the sheet: frame, strip, rail, captions
const ORDER = F.order, SC = F.scenes;
const TITLES = { open: 'Cold open', team: 'Team', problem: 'Problem and user', live: 'Live demo on AWS', break: 'Failure drills, human control', mcp: 'An LLM over MCP',
  arch: 'Architecture', results: 'Principal results', failures: 'Failure cases', limits: 'Limits', repro: 'Reproduce', end: 'Links' };
const LIT = D.lit, ENTRY = `${LIT.competition} · ${LIT.award} entry`;
const STRIPB = { open: 'live recording, synthetic sample photo bridge_03', team: ENTRY,
  problem: 'sources: docs/impact-sources.md', live: `recorded on ${D.urls.live.replace('https://', '')}, ${LIT.recorded}`, break: 'live: drills injected on purpose',
  mcp: 'docs/mcp-session: a recorded Claude session', arch: 'docs/architecture.png · docs/agent-loop.svg · public/docs/aws-simple.svg',
  results: 'public/results/*.json, final classical-engine run', failures: 'public/results/failures.json', limits: 'pixels, not millimetres', repro: D.urls.repo.replace('https://', ''), end: '' };

let strip, stripA, stripB, stripNo, rail, chs = [], railNote, railCall, capBox, capSpan;
const CALLS = [];
function buildFrame() {
  el('div', 'sheetframe', stage);
  for (const [l, tp] of [[20, 20], [1874, 20], [20, 1034], [1874, 1034]]) {
    const s = svgEl('svg', { class: 'cross', viewBox: '0 0 26 26', style: `left:${l}px;top:${tp}px` }, stage);
    svgEl('path', { d: 'M13 0v26M0 13h26', stroke: '#E8F1FF', 'stroke-width': 2 }, s); svgEl('circle', { cx: 13, cy: 13, r: 6, fill: 'none', stroke: '#E8F1FF', 'stroke-width': 2 }, s);
  }
  strip = el('div', '', stage); strip.id = 'strip';
  stripA = el('span', 'a', strip); stripB = el('span', 'b', strip); stripNo = el('span', 'no', strip);
  rail = el('div', '', stage); rail.id = 'rail'; el('div', 't', rail, 'SHEET INDEX');
  ORDER.forEach((id, k) => { const c = el('div', 'ch', rail, `<i>${k + 1}</i><span>${TITLES[id]}</span>`); chs.push(c); });
  railNote = el('div', '', rail); railNote.id = 'railnote';
  railCall = el('div', '', rail); railCall.id = 'railcall';
  capBox = el('div', '', stage); capBox.id = 'cap'; capSpan = el('span', '', capBox);
}
function sceneAt(t) { let k = 0; ORDER.forEach((id, i) => { if (t >= SC[id].t0) k = i; }); return k; }
function drawFrame(t) {
  const k = sceneAt(t), id = ORDER[k], s = SC[id];
  stripA.textContent = `DRISHYAM // ${TITLES[id]}`; stripB.textContent = STRIPB[id] || '';
  const p = seg(t, s.t0, s.t0 + 0.35);
  stripNo.innerHTML = `SHEET <b style="display:inline-block;transform:scale(${(1.7 - 0.7 * eo(p)).toFixed(3)});opacity:${(0.3 + 0.7 * p).toFixed(2)}">${k + 1}</b> OF ${ORDER.length}`;
  chs.forEach((c, i) => { c.className = 'ch' + (i < k ? ' done' : i === k ? ' on' : ''); c.style.transform = `scale(${(1 + 0.1 * bump(seg(t, SC[ORDER[i]].t0, SC[ORDER[i]].t0 + 0.4))).toFixed(3)})`; });
  const call = CALLS.find((c) => t >= c.t0 && t < c.t1);
  if (call) { const p = seg(t, call.t0, call.t0 + 0.3), q = 1 - seg(t, call.t1 - 0.25, call.t1);
    railCall.innerHTML = `<div class="n">${call.n}</div><div class="big">${call.big}</div><div class="lb">${call.label}</div>`;
    css(railCall, { opacity: Math.min(p, q), transform: `scale(${(0.85 + 0.15 * spring(seg(t, call.t0, call.t0 + 0.6))).toFixed(3)})` }); railNote.style.opacity = 0; }
  else { railCall.style.opacity = 0; railNote.style.opacity = 1; }
  railNote.innerHTML = RAILNOTE[id] ? RAILNOTE[id](t) : '';
  const c = F.cues.find((x) => t >= x.t - 0.05 && t <= x.end + 0.25);
  capSpan.textContent = c ? c.text : ''; capBox.style.opacity = c ? 1 : 0;
}
const RAILNOTE = {
  open: () => `Synthetic sample photo <b>bridge_03</b>, live job <b>${D.guided.job}</b>.`,
  live: () => `Live job <b>${D.guided.job}</b> on ${D.health.host}, ${D.health.arch}, OpenCV ${D.health.opencv}. Retimed to the voice: the speed is shown in the window bar.`,
  break: () => 'Faults injected on purpose: a malformed planner reply, a slowed segmenter, a planner that stalls.',
  mcp: () => 'Calls, values and reasons are copied from <b>exchange.jsonl</b>.',
  results: () => 'Classical segmenter, governor off, fixed per-call costs from the arm64 benchmark.',
  failures: () => 'CODEBRIM photos are never shown: its licence bars sharing them.',
};

// ------------------------------------------------------------------ scenes
const SCN = {};
function scene(id, full) { const e = el('div', 'scene' + (full ? ' full' : ''), stage); SCN[id] = { e, draws: [] }; return SCN[id]; }
function visible(id, t) {   // the sheet slides up out of the plan chest, then the next one rises in
  const s = SC[id], e = SCN[id].e, pin = s.t0 === 0 ? 1 : eo(seg(t, s.t0, s.t0 + 0.45)), pout = eio(seg(t, s.t1 - 0.32, s.t1));
  const on = t >= s.t0 - 0.01 && t < s.t1 + 0.01;
  css(e, { opacity: on ? Math.min(pin, 1 - pout) : 0, transform: `translateY(${((1 - pin) * 46 - pout * 40).toFixed(1)}px)`, display: on ? 'block' : 'none' });
  return on;
}
const KF = (f, cx, cy, w) => ({ f, c: [cx, cy, w] });

// 1. cold open
function buildOpen() {
  const S = scene('open'), s = SC.open, ev = F.ev;
  const fw = new Footage(S.e, 1530, 836, D.urls.live.replace('https://', ''), F.warp.open, [
    KF(ev['step:1'] - 0.2, 360, 520, 760), KF(ev['step:3'], 340, 560, 640), KF(ev['step:4'] + 1.4, 330, 600, 560),
    KF(ev['click:Sign them off'] + 0.9, 340, 700, 760), KF(ev['click:Sign them off'] + 2.6, 330, 690, 700)]);
  const big = el('div', 'title', S.e, `<div class="h1" style="font-size:120px">${LIT.title.split(', ').slice(0, 2).join(' · ')}<br>${LIT.title.split(', ')[2]}</div><div class="lab" style="margin-top:26px;font-size:26px">the site’s own figure note, public/index.html</div>`);
  css(big, { top: '230px' });
  const veil = el('div', 'abs', S.e); css(veil, { inset: 0, background: 'var(--chest)' });
  S.draws.push(async (t) => {
    const u = t - s.t0, p = seg(u, 1.3, 1.75);
    css(big, { opacity: 1 - p, transform: `scale(${(1.25 - 0.25 * eo(seg(u, 0, 0.5)) - 0.05 * p).toFixed(3)})` }); css(veil, { opacity: 1 - p, zIndex: 3 }); big.style.zIndex = 4;
    await fw.draw(Math.max(t, F.warp.open.t0));
  });
  cue(s.t0 + 0.05, 'slam'); fw.clicksSfx();
  railcall(1, (D.guided.pixels_frac * 100).toFixed(1) + ' %', `of the pixels read; ${D.guided.critical} critical findings`, at('o1') + 0.6, end('o2') + 0.3);
}

// a camera over a board: keyframes [film t, cx, cy, width], eased moves, a push-in while a view is held
function boardView(host, VW = 1530, VH = 836) {
  const view = el('div', 'abs', host); css(view, { left: 0, top: 0, width: VW + 'px', height: VH + 'px', overflow: 'hidden' });
  const board = el('div', 'abs', view); css(board, { left: 0, top: 0, width: '6800px', height: '3400px', transformOrigin: '0 0' });
  return { view, board, VW, VH };
}
function camBoard(B, K, push = 0.08) {
  return (t) => {
    let k = 0; while (k < K.length - 1 && t >= K[k + 1][0]) k++;
    let c = K[k].slice(1);
    if (k < K.length - 1) {
      const n = K[k + 1], mv = Math.min(1.25, (n[0] - K[k][0]) * 0.6), p = eio(seg(t, n[0] - mv, n[0]));
      const far = Math.hypot(n[1] - c[0], n[2] - c[1]) > 1500, out = far ? 1 + 0.9 * bump(p) : 1;      // a long flight pulls back first
      const hold = 1 - push * seg(t, K[k][0], n[0] - mv);
      c = [lerp(c[0], n[1], p), lerp(c[1], n[2], p), lerp(c[2] * hold, n[3], p) * out];
    } else c[2] *= 1 - push * seg(t, K[k][0], K[k][0] + 6);
    const z = B.VW / c[2];
    css(B.board, { transform: `translate(${(B.VW / 2 - c[0] * z).toFixed(1)}px,${(B.VH / 2 - c[1] * z).toFixed(1)}px) scale(${z.toFixed(4)})` });
  };
}
const typed = (text, p) => text.slice(0, Math.round(text.length * clamp(p)));

// 2. team: two title blocks, drawn by a pen, the camera on the one being read
function buildTeam() {
  const S = scene('team'), s = SC.team, B = boardView(S.e), H = B.board;
  el('div', 'abs', H, `<span class="fig">FIG. 0</span> <span class="lab" style="margin-left:14px;font-size:24px">${ENTRY}</span>`).style.cssText = 'left:0;top:40px';
  const cards = LIT.team.map((x) => { const [, name, role, did] = x.match(/^([^,]+), (.+) \((.+)\)$/); return [role.toUpperCase(), name, did.split(', ')]; });   // README.md's Team lines
  const T = [s.t0 + 0.25, at('t2') + LINE.t2.dur * 0.45];
  cards.forEach(([role, name, did], i) => {
    const x = i * 900, y = 150, w = 740, h = 560, box = el('div', 'abs', H); css(box, { left: x + 'px', top: y + 'px', width: w + 'px', height: h + 'px' });
    const sv = svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, class: 'abs' }, box);
    const r = svgEl('rect', { x: 3, y: 3, width: w - 6, height: h - 6, fill: 'rgba(11,50,100,.92)', stroke: '#E8F1FF', 'stroke-width': 3, pathLength: 1, 'stroke-dasharray': 1 }, sv);
    const rows = [92, 300].map((yy) => svgEl('path', { d: `M3 ${yy}H${w - 3}`, stroke: '#B8CDEC', 'stroke-width': 2.5, pathLength: 1, 'stroke-dasharray': 1 }, sv));
    const dim = svgEl('path', { d: `M30 ${h - 34}H${w - 30}M30 ${h - 46}V${h - 22}M${w - 30} ${h - 46}V${h - 22}`, stroke: '#FFC857', 'stroke-width': 2.5, fill: 'none', pathLength: 1, 'stroke-dasharray': 1 }, sv);
    const a = el('div', 'abs lab', box, 'ROLE'); css(a, { left: '30px', top: '24px', fontSize: '20px', color: 'var(--line-3)' });
    const b = el('div', 'abs', box, ''); css(b, { left: '30px', top: '50px', font: '700 28px/1 var(--fm)', letterSpacing: '.12em', color: 'var(--amber)' });
    const c = el('div', 'abs lab', box, 'NAME'); css(c, { left: '30px', top: '116px', fontSize: '20px', color: 'var(--line-3)' });
    const n = el('div', 'abs h2', box, ''); css(n, { left: '30px', top: '160px', fontSize: '92px', whiteSpace: 'nowrap' });
    const d = el('div', 'abs lab', box, 'WORK ON DRISHYAM'); css(d, { left: '30px', top: '324px', fontSize: '20px', color: 'var(--line-3)' });
    const items = did.map((it, j) => { const e = el('div', 'abs mono', box, `<span class="amber">▸</span> ${it}`); css(e, { left: '30px', top: 362 + j * 40 + 'px', fontSize: '30px' }); return e; });
    const pen = svgEl('svg', { width: 40, height: 40, viewBox: '0 0 40 40', class: 'abs' }, box);
    svgEl('path', { d: 'M4 36 L10 22 L30 2 L38 10 L18 30 Z', fill: '#FFC857', stroke: '#2E2100', 'stroke-width': 2 }, pen);
    S.draws.push((t) => {
      const p = eio(seg(t, T[i], T[i] + 1.1));
      r.setAttribute('stroke-dashoffset', 1 - p); rows.forEach((q, j) => q.setAttribute('stroke-dashoffset', 1 - eio(seg(t, T[i] + 0.6 + j * 0.2, T[i] + 1.2 + j * 0.2))));
      dim.setAttribute('stroke-dashoffset', 1 - eio(seg(t, T[i] + 2.2, T[i] + 3.0)));
      const per = 2 * (w + h), dp = p * per; let px, py;
      if (dp < w) { px = dp; py = 0; } else if (dp < w + h) { px = w; py = dp - w; } else if (dp < 2 * w + h) { px = w - (dp - w - h); py = h; } else { px = 0; py = h - (dp - 2 * w - h); }
      css(pen, { left: px - 4 + 'px', top: py - 36 + 'px', opacity: p > 0 && p < 1 ? 1 : 0 });
      b.textContent = typed(role, seg(t, T[i] + 0.5, T[i] + 1.1)); n.textContent = typed(name, seg(t, T[i] + 0.8, T[i] + 1.6));
      [[a, 0.4], [c, 0.7], [d, 1.4]].forEach(([e, k]) => { e.style.opacity = seg(t, T[i] + k, T[i] + k + 0.3); });
      items.forEach((e, j) => { const q = eo(seg(t, T[i] + 1.6 + j * 0.35, T[i] + 1.9 + j * 0.35)); css(e, { opacity: q, transform: `translateX(${((1 - q) * -24).toFixed(1)}px)` }); });
      box.style.opacity = t >= T[i] - 0.05 ? 1 : 0;
    });
    cue(T[i], 'scribble', 1.1); cue(T[i] + 0.8, 'type', 0.8); did.forEach((_, j) => cue(T[i] + 1.6 + j * 0.35, 'tick', j + 2 * i));
  });
  const cam = camBoard(B, [[s.t0, 380, 430, 1180], [T[1] - 0.3, 1330, 430, 1180], [end('t2') - 0.2, 820, 430, 1800]]);
  S.draws.unshift(cam);
}

// 3. problem and user: three figures counting up as they are said, then the backlog, square by square
function buildProblem() {
  const S = scene('problem'), s = SC.problem, I = D.impact, B = boardView(S.e), H = B.board;
  const col = (x, w) => { const e = el('div', 'abs', H); css(e, { left: x + 'px', top: '40px', width: w + 'px' }); return e; };
  const c1 = col(0, 560), c2 = col(600, 400), c3 = col(1040, 490);
  c1.innerHTML = `<div class="lab">US bridges</div><div class="h1 amber" style="font-size:132px;margin-top:14px" data-k="b">0</div>
    <div class="mono" style="margin-top:10px"><span data-k="p">0</span> in poor condition</div><div class="src" style="margin-top:12px">${LIT.cites[0]}</div>`;
  c2.innerHTML = `<div class="lab">Routine inspection</div><div class="h1 amber" style="font-size:132px;margin-top:14px">≤ <span data-k="m">0</span></div>
    <div class="mono" style="margin-top:10px">months apart</div><div class="src" style="margin-top:12px">${LIT.cites[1]}</div>`;
  c3.innerHTML = `<div class="lab">Largest of ${I.n_bridges} drone inspections</div><div class="h1 amber" style="font-size:132px;margin-top:14px" data-k="h">0</div>
    <div class="mono" style="margin-top:10px">photos, one bridge (others: ${I.others[0]}–${I.others[1]})</div><div class="src" style="margin-top:12px">ODOT/FHWA ${LIT.cites[2]}, ${LIT.cites[3]}</div>`;
  const q = (k) => H.querySelector(`[data-k="${k}"]`);
  const cv = el('canvas', 'abs', H); cv.width = 1530; cv.height = 300; css(cv, { left: '0px', top: '420px' });
  const g = cv.getContext('2d');
  const user = el('div', 'abs', H, `<svg width="1530" height="40" class="abs" style="top:0"><path class="dimline" d="M2 20H1528M2 6V34M1528 6V34"/></svg>
    <div class="lab" style="position:absolute;top:52px;left:0">THE USER</div><div class="mono" style="position:absolute;top:84px;left:0;font-size:34px">an inspection engineer working through a backlog of bridge photos</div>`);
  css(user, { left: '0px', top: '720px', width: '1530px', height: '120px' });
  const nb = +I.bridges.replace(/,/g, ''), np = +I.poor.replace(/,/g, ''), nm = +I.months, nh = +I.photos.replace(/,/g, '');
  const T1 = at('p1') + 0.2, T2 = at('p1') + LINE.p1.dur * 0.55, T3 = at('p2') + 0.2, T4 = at('p3'), D3 = LINE.p2.dur + 0.6;
  S.draws.push((t) => {
    const p1 = eo(seg(t, T1, T1 + 2.2)), p2 = eo(seg(t, T2, T2 + 1.2)), p3 = seg(t, T3, T3 + D3);
    q('b').textContent = Math.round(nb * p1).toLocaleString('en-US'); q('p').textContent = Math.round(np * p1).toLocaleString('en-US');
    q('m').textContent = Math.round(nm * p2); q('h').textContent = Math.round(nh * p3).toLocaleString('en-US');
    c1.style.opacity = seg(t, s.t0, s.t0 + 0.3); c2.style.opacity = seg(t, T2 - 0.3, T2); c3.style.opacity = seg(t, T3 - 0.3, T3);
    g.clearRect(0, 0, 1530, 300); const n = Math.round(nh * p3), cols = 127, sz = 10, gap = 2;   // one small square per photo
    g.fillStyle = 'rgba(232,241,255,.12)'; if (t > T3 - 0.3) for (let i = 0; i < nh; i++) g.fillRect((i % cols) * (sz + gap), Math.floor(i / cols) * (sz + gap) + 30, sz, sz);
    g.fillStyle = '#FFC857'; for (let i = 0; i < n; i++) g.fillRect((i % cols) * (sz + gap), Math.floor(i / cols) * (sz + gap) + 30, sz, sz);
    const u = eo(seg(t, T4, T4 + 0.6)); css(user, { opacity: u, transform: `translateX(${((1 - u) * -30).toFixed(1)}px)` });
  });
  S.draws.unshift(camBoard(B, [[s.t0, 765, 418, 1700], [T1 - 0.2, 490, 260, 1100], [T2 - 0.2, 490, 250, 1100], [T3 - 0.2, 1090, 300, 1080],
    [T3 + D3 * 0.55, 765, 440, 1760], [T4 + 0.2, 765, 450, 1760], [end('p3') - 0.6, 765, 460, 1740]]));   // full shots: the figures stay whole over the waffle
  cue(T1, 'count', 2.2); cue(T2, 'count', 1.2); cue(T3, 'count', D3); cue(T4, 'switch');
}

// 4. the live demo: the health check, then the guided run and the sign-off
function buildLive() {
  const S = scene('live'), s = SC.live, ev = F.ev, wk = F.walk.steps[0];
  // a. /api/health, as the browser shows it (walk.py), and the three values it reports, drawn out on leader lines
  const hw = el('div', 'abs', S.e); css(hw, { inset: 0 });
  const hwin = el('div', 'abs', hw); css(hwin, { left: '0px', top: '0px', width: '1530px', height: '330px' });
  const H = new Footage(hwin, 1530, 330, D.urls.health.replace('https://', ''), { t0: s.t0, t1: s.t1, pts: [{ t: s.t0, f: 0 }, { t: s.t1, f: 1 }] }, [{ T: s.t0, c: [640, 360, 1280] }], { cursor: false });
  H.A.src = '../brag-output/work/walkrec/' + wk.shots[0]; loads.push(H.A.decode().catch(() => {})); css(H.A, { width: '1280px', height: '720px' });
  H.draw = function (t) {
    const w = lerp(1180, 1000, eio(seg(t, s.t0 + 0.3, s.t0 + 1.4))) * (1 - 0.14 * eio(seg(t, s.t0 + 1.4, F.warp.live.t0)));
    const z = this.sw / w, c = [w / 2, this.sh / z / 2, z]; this.c = c;
    css(this.cam, { transform: `translate(${(this.sw / 2 - c[0] * z).toFixed(2)}px,${(this.sh / 2 - c[1] * z).toFixed(2)}px) scale(${z.toFixed(4)})` });
    this.tag.textContent = 'GET /api/health'; this.tag.className = 'tag'; this.A.style.opacity = 1; this.B.style.opacity = 0; css(this.cur, { opacity: 0 });
  };
  const keys = [['"opencv":"5.0.0"', 'opencv', D.health.opencv, `build: ${D.health.build}`], ['"host":"aws-lambda"', 'host', D.health.host, 'behind CloudFront'], ['"arch":"aarch64"', 'arch', D.health.arch, LIT.graviton]];
  const lead = svgEl('svg', { width: 1530, height: 836, class: 'abs', style: 'left:0;top:0;z-index:2' }, hw);
  const JSON_BOTTOM = Math.max(...Object.values(F.health_boxes).map(([r]) => r[1] + r[3]));   // the last line of the response
  const hls = keys.map(([k, name, val, sub], i) => {
    const e = el('div', 'hl', H.overlays), r = F.health_boxes[k][0], T = at('l1') + 0.7 + i * 0.75;
    css(e, { border: '0', boxShadow: 'none', background: 'rgba(255,200,87,.42)' });
    const box = el('div', 'box amb', hw, `<div class="lab">${name}</div><div class="h1 amber" style="font-size:66px;margin:14px 0 12px;white-space:nowrap">${val}</div><div class="mono" style="font-size:24px">${sub}</div>`);
    css(box, { left: 30 + i * 490 + 'px', top: '440px', width: '470px', height: '250px', padding: '22px 26px' });
    const path = svgEl('path', { fill: 'none', stroke: '#FFC857', 'stroke-width': 3.5, pathLength: 1, 'stroke-dasharray': 1 }, lead);
    const dot = svgEl('circle', { r: 6, fill: '#FFC857' }, lead);
    return { e, r, T, box, path, dot, i };
  });
  const foot = el('div', 'abs src', hw, `commit ${D.health.commit.slice(0, 7)} · the response is saved in video/live/health.json`); css(foot, { left: '0px', top: '720px', fontSize: '22px' });
  hls.forEach((h) => cue(h.T, 'tick', 4 + h.i));
  // b. the guided run on the live site
  const fw2 = el('div', 'abs', S.e); css(fw2, { inset: 0 });
  const cw = ev['click:Watch it inspect a bridge'];
  const fw = new Footage(fw2, 1530, 836, D.urls.live.replace('https://', ''), F.warp.live, [
    KF(cw - 1.9, 560, 470, 1150), KF(cw + 0.15, 330, 560, 900),
    KF(ev['step:1'] - 0.6, 380, 560, 1060), KF(ev['step:2'] + 0.4, 340, 560, 900),
    KF(ev['step:3'] + 0.3, 900, 380, 760), KF(ev['step:4'] + 0.3, 930, 700, 760),
    KF(ev['step:5'] + 0.3, 420, 560, 1000), KF(ev['done:guided'] + 0.3, 560, 330, 1080),
    KF(ev['click:Sign them off'], 640, 560, 1240), KF(ev['click:Sign them off'] + 2.6, 330, 720, 840),
    KF(ev['click:Approve#0'] + 0.6, 330, 760, 720), KF(ev['approved:1'] - 1.6, 330, 560, 760), KF(ev['approved:2'] + 0.4, 640, 470, 1240), KF(ev['approved:2'] + 2.2, 940, 420, 900)]);
  railcall(2, `${fmt(D.guided.entropy1)} &gt; ${D.guided.tau}`, 'entropy at L3: unsure, so it zooms in', at('l4') + 0.3, end('l4') + 0.3);
  railcall(4, `IoU ${fmt(D.guided.iou, 3)}`, `mask IoU, L1 vs L0, under ${D.guided.iou_min}: unconfirmed. Exposed rebar always goes to a person`, at('l6') + 0.3, end('l6') + 0.3);
  railcall('×', `${D.guided.missed.join(', ')} missed`, `the stop left regions ${D.guided.unvisited.join(' and ')} unvisited; the ${D.guided.missed[0]} in ${D.guided.unvisited[0]} was never seen`, at('l7') + LINE.l7.dur * 0.4, end('l7') + 0.3);
  railcall('✓', 'a person', 'approves each critical finding; the server signs', at('l9') + 0.3, end('l9') + 0.3);
  railcall('!', `HTTP ${D.tamper.status}`, D.tamper.error, at('l11') + 1.4, end('l11') + 0.6);
  // c. the tampered token, answered by the server
  const term = el('div', 'term', S.e); css(term, { right: '28px', bottom: '28px', width: '980px', zIndex: 9 });
  const L1 = `<span class="p">POST</span> /api/review   token: the work order's, one character`, L2 = `     of its ${D.aws.hmac} signature changed`,
    L3 = `<span class="no">HTTP ${D.tamper.status}</span>  {"error": "${D.tamper.error}"}`;
  S.draws.push(async (t) => {
    const x = seg(t, F.warp.live.t0 - 0.2, F.warp.live.t0 + 0.25);
    hw.style.opacity = 1 - x; fw2.style.opacity = x;
    if (t < F.warp.live.t0 + 0.3) { H.draw(t); hls.forEach((h) => { const p = seg(t, h.T, h.T + 0.3); const [sx, sy] = H.toScreen(h.r[0] - 3, h.r[1], H.c);
      const bw = (h.r[2] + 6) * H.c[2], bh = h.r[3] * H.c[2], below = H.toScreen(0, JSON_BOTTOM + 4, H.c)[1];
      css(h.e, { opacity: p, left: sx + 'px', top: sy + 'px', width: bw + 'px', height: bh + 'px', transform: `scale(${(1.15 - 0.15 * eo(p)).toFixed(3)})` });
      const x0 = 3 + sx + bw / 2, y0 = 49 + below, x1 = 30 + h.i * 490 + 235, y1 = 440, q = eio(seg(t, h.T + 0.15, h.T + 0.6));
      h.path.setAttribute('d', `M${x0.toFixed(1)} ${y0.toFixed(1)} V${((y0 + y1) / 2).toFixed(1)} H${x1} V${y1}`); h.path.setAttribute('stroke-dashoffset', 1 - q);
      h.dot.setAttribute('cx', x0); h.dot.setAttribute('cy', y0); h.dot.setAttribute('opacity', p);
      const bq = seg(t, h.T + 0.5, h.T + 0.8); css(h.box, { opacity: bq, transform: `translateY(${((1 - eo(bq)) * 24).toFixed(1)}px)` }); });
      foot.style.opacity = seg(t, hls[2].T + 0.8, hls[2].T + 1.1); }
    if (t >= F.warp.live.t0 - 0.25) await fw.draw(t);
    const T0 = at('l11') - 0.1, p = eo(seg(t, T0, T0 + 0.4)), ty = (u) => Math.floor(seg(t, T0 + 0.3 + u, T0 + 0.9 + u) * 200);
    term.style.opacity = p; term.style.transform = `translateY(${((1 - p) * 40).toFixed(1)}px)`;
    const full = (h) => h.replace(/<[^>]+>/g, '');
    term.innerHTML = [L1, L2].map((l, i) => (ty(i * 0.4) >= full(l).length ? l : full(l).slice(0, ty(i * 0.4)))).join('\n') + (t > T0 + 1.6 ? '\n' + L3 : '\n<span class="caret"></span>');
  });
  cue(at('l11') + 0.2, 'type', 1.0); cue(at('l11') + 1.6, 'alert'); fw.clicksSfx();
  for (const k of ['approved:0', 'approved:1', 'approved:2']) { const T = filmOf(fw.S, ev[k] + 0.2); if (T) cue(T, 'stamp'); }
  for (const e of F.events.filter((e) => e.kind === 'type')) { const T = filmOf(fw.S, e.t0); if (T) cue(T, 'type', 0.5); }
}

// 5. failure drills and human control
function buildBreak() {
  const S = scene('break'), ev = F.ev;
  const fw = new Footage(S.e, 1530, 836, D.urls.live.replace('https://', ''), F.warp.break, [
    KF(ev['click:Now break it'] - 1.2, 400, 640, 900), KF(ev['click:Now break it'] + 0.1, 330, 640, 760),
    KF(ev['step-break:5'] - 1.2, 520, 520, 1200), KF(ev['step-break:5'] + 0.4, 330, 760, 760),
    KF(ev['step-break:6'] + 0.3, 900, 330, 760), KF(ev['click:Try KILL'] - 0.8, 400, 640, 900),
    KF(ev['click:KILL'] - 0.9, 900, 140, 900), KF(ev['done:kill'] + 0.6, 760, 300, 1100),
    KF(ev['click:Resume'] + 0.1, 760, 300, 1060), KF(ev['step-resume:2'], 930, 560, 820), KF(ev['step-resume:3'] + 1.2, 380, 560, 900)]);
  railcall(1, `p95 ${Math.round(D.break.p95)} ms`, `segmenter slowed on purpose: the governor picks the ${D.break.model} model`, at('b1') + 0.4, end('b1') + 0.3);
  railcall(2, `“${D.break.planner_bad}”`, 'the planner drill’s reply: rejected by the allow-list', at('b2') + 0.3, end('b2') + 0.3);
  railcall(3, D.kill.cause, 'halted before the next look; Resume replays', at('b3') + 0.3, end('b3') + 0.4);
  S.draws.push(async (t) => { await fw.draw(t); });
  fw.clicksSfx(); cue(at('b2') + 0.6, 'glitch'); cue(at('b3') + 0.5 * LINE.b3.dur, 'thud');
}

// 6. an LLM over MCP: the recorded call table, one call at a time, and the crops Claude looked at
function buildMcp() {
  const S = scene('mcp'), s = SC.mcp, m = D.mcp, B = boardView(S.e), H = B.board;
  el('div', 'abs lab', H, `docs/mcp-session · ${LIT.client} → drishyam mcp over stdio · job ${m.job}`).style.cssText = 'left:0;top:0;font-size:21px';
  const crops = [['crops/img-004-1.jpg', 'three rust bars'], ['crops/img-016-1.jpg', 'a spall']].map(([f, lab]) => ['../docs/mcp-session/' + f, `call ${m.crops[f][0]} · gaze L${m.crops[f][1]}: ${lab}`]);
  const cx = crops.map(([src, lab], i) => { const b = el('div', 'abs', H); css(b, { left: '0px', top: 52 + i * 394 + 'px', width: '340px', height: '380px' });
    const im = img(src, b); css(im, { position: 'absolute', left: 0, top: 0, width: '340px', height: '340px', border: '3px solid var(--line)' });
    el('div', 'src', b, lab).style.cssText = 'position:absolute;left:0;top:350px;width:340px;color:var(--line-2)'; return b; });
  railcall(m.call_a, `IoU ${fmt(m.iou_a)}`, `mask IoU, L1 vs L0 at call ${m.call_a}: they disagree; exposed rebar is always critical`, at('m2') + 0.3, end('m2') + 0.3);
  railcall(m.call_b, `IoU ${fmt(m.iou_b)}`, `at call ${m.call_b}: the spall is confirmed`, at('m3') + 0.3, at('m3') + LINE.m3.dur * 0.55);
  const box = el('div', 'abs', H); css(box, { left: '380px', top: '52px', width: '1150px', height: '650px', overflow: 'hidden', border: '3px solid var(--line)', background: 'rgba(8,42,85,.95)' });
  const tbl = el('table', 't', box); css(tbl, { position: 'absolute', left: 0, top: 0, width: '1144px', fontSize: '23px' });
  tbl.innerHTML = '<tr><th>#</th><th>call</th><th>what came back</th></tr>' + m.rows.map((r) => `<tr data-n="${r.n}"><td class="n">${r.n}</td><td>${r.call}</td><td style="white-space:normal">${r.got}</td></tr>`).join('');
  const why = el('div', 'box amb', H); css(why, { left: '380px', top: '714px', width: '1150px', minHeight: '104px', padding: '12px 20px', font: '400 21px/1.38 var(--fm)', whiteSpace: 'normal' });
  const R = (n) => m.rows.find((r) => r.n === n);
  // m1: the 23 calls stream in, one at a time; then the camera goes back to the three that decided something
  const T0 = at('m1') + 0.3, T1 = end('m1') + 0.2, N = m.rows.length, IA = m.rows.findIndex((r) => r.n === m.call_a);
  const beats = [{ n: m.call_a, T: at('m2') + 0.2, cls: 'bad', why: R(m.call_a + 1).why },
    { n: m.call_b, T: at('m3') + 0.2, cls: 'good', why: R(m.call_b).why }, { n: m.call_c, T: at('m3') + LINE.m3.dur * 0.55, cls: 'good', why: R(m.call_c).why },
    { n: m.rows[N - 1].n, T: at('m4') + 0.2, why: `Missed: the large spall around the rebar (box ${m.missed_box}). Logged: the rebar, the crack and the small spall (transcript.md).` }];
  for (let k = 0; k <= IA; k++) cue(T0 + (T1 - T0) * k / (IA + 1), 'tick', k % 9);
  beats.forEach((b) => cue(b.T, 'switch'));
  const rows = () => [...tbl.querySelectorAll('tr[data-n]')];
  S.draws.push((t) => {
    const rs = rows(); let i, cls = '', w;
    if (t < beats[0].T - 0.4) {                                        // streaming
      const k = clamp(Math.floor(seg(t, T0, T1) * (IA + 1)), 0, IA);
      rs.forEach((r, j) => { r.style.opacity = j <= k ? 1 : 0.08; });
      i = k; w = R(m.rows[k].n).why;
      const y = rs[k].offsetTop; tbl.style.transform = `translateY(${(-clamp(y - 470, 0, tbl.offsetHeight - 644)).toFixed(1)}px)`;
    } else {
      let b = beats[0]; for (const x of beats) if (t >= x.T) b = x;
      const prev = beats[Math.max(0, beats.indexOf(b) - 1)];
      i = m.rows.findIndex((r) => r.n === b.n); const ip = b === beats[0] ? IA : m.rows.findIndex((r) => r.n === prev.n);
      const p = eio(seg(t, b.T - 0.5, b.T + 0.3)), y = lerp(rs[ip].offsetTop, rs[i].offsetTop, p), shown = lerp(ip, i, p);
      rs.forEach((r, j) => { r.style.opacity = j <= shown + 0.5 ? 1 : 0.08; });
      tbl.style.transform = `translateY(${(-clamp(y - 250, 0, tbl.offsetHeight - 644)).toFixed(1)}px)`;
      cls = b.cls || ''; w = b.why;
    }
    rs.forEach((r, k) => { const on = k === i; r.style.background = on ? (cls === 'bad' ? 'rgba(255,154,138,.25)' : 'rgba(255,200,87,.22)') : 'transparent'; r.style.outline = on ? `3px solid ${cls === 'bad' ? 'var(--red)' : 'var(--amber)'}` : 'none'; });
    why.innerHTML = `<b class="amber">why the next call followed</b> · ${w || ''}`;
    cx.forEach((c, k) => { const qq = k === 0 ? seg(t, at('m2') - 0.2, at('m2') + 0.3) * (1 - seg(t, at('m3') - 0.2, at('m3'))) : seg(t, at('m3') - 0.2, at('m3') + 0.3);
      css(c, { opacity: 0.35 + 0.65 * qq, transform: `scale(${(0.96 + 0.04 * qq).toFixed(3)})` }); });
  });
  S.draws.unshift(camBoard(B, [[s.t0, 765, 418, 1720], [T0 + 0.5, 765, 418, 1700], [at('m2') - 0.1, 765, 410, 1690], [at('m3') - 0.1, 765, 430, 1690], [at('m4'), 765, 418, 1720]], 0.04));
}

// 7. architecture: one board, a camera flying across the three diagrams
function buildArch() {
  const S = scene('arch'), s = SC.arch;
  const view = el('div', 'abs', S.e); css(view, { inset: 0, overflow: 'hidden', border: '3px solid var(--line)', background: '#0b2e5c' });
  const board = el('div', 'abs', view); css(board, { left: 0, top: 0, transformOrigin: '0 0' });
  const A = img('../docs/architecture.png', board); css(A, { position: 'absolute', left: '0px', top: '0px', width: '2680px', height: '3280px' });
  // far enough apart that each diagram's whole view (its height / 0.92 for the push) shows none of the others
  const LX = 4700, WX = 8650;
  const L = img('../docs/agent-loop.svg', board); css(L, { position: 'absolute', left: LX + 'px', top: '0px', width: '2680px', height: '2472px' });
  const W = img('../public/docs/aws-simple.svg', board); css(W, { position: 'absolute', left: WX + 'px', top: '0px', width: '900px', height: '1650px', background: '#0F3B73' });
  // the Lambda numbers go on the rail, beside the diagram, never over its text
  railcall('★', `${fmt(D.aws.p95[0], 1)}–${fmt(D.aws.p95[1], 1)} ms`, `step p95 on Lambda: ${D.aws.smoke_runs} smoke runs, ${D.aws.smoke_steps.join('/')} steps each. COOL on Graviton: not run`, at('a5') + 0.5, end('a5') + 0.4);
  const VW = 1524, VH = 830;
  const a0 = at('a0'), d = (id, f) => at(id) + LINE[id].dur * f;
  const whole = (h) => h * VW / VH / 0.92;   // the width that shows a diagram of height h whole, even at the end of the push
  // crops: each view's edges, at the start of the hold and after the 8 % push, fall in text-free bands of the diagram
  // each diagram is held whole for a beat first (the agent loop, the AWS path, the system sheet), then the camera reads it box by box
  const K = [[a0 - 0.6, LX + 1340, 1236, whole(2472)], [d('a0', 0.5), LX + 1340, 932, 2846], [d('a0', 0.8), LX + 1340, 2430, 2938],
    [at('a1'), WX + 450, 825, whole(1650)], [d('a1', 0.5), WX + 450, 360, 1150], [d('a1', 0.78), WX + 450, 1122, 1790],   // the AWS path whole, then boxes 1-2 and 3-7 with their titles whole
    [at('a2'), 1340, 1640, whole(3280)], [d('a2', 0.45), 1500, 850, 950], [d('a2', 0.75), 1500, 1110, 950],
    [at('a3') + 0.1, 1975, 2427, 1300], [d('a3', 0.5), 734, 2427, 1320],   // OpenCV layer: pyramid and saliency, then the virtual PTZ, blur gate and privacy pass
    [at('a4') + 0.1, 1668, 2601, 620], [d('a4', 0.55), 1668, 2610, 600], [at('a5') + 0.1, 600, 1230, 1120], [d('a5', 0.55), 600, 1225, 1060]];   // a4 above the full-width footnote, a5 the Function URL and Lambda boxes whole
  const cam = camBoard({ board, VW, VH }, K, 0.08);
  S.draws.push(cam);
  K.forEach((k, i) => i && cue(k[0] - 0.6, 'whoosh'));
}

// 8. principal results: chart, then the tables and cards on one board
function chartSvg(host, P, w, h) {
  const L = 100, R = 30, T = 30, B = 80, PW = w - L - R, PH = h - T - B, xmax = Math.max(...Object.values(P).flat().map((p) => p.pixels_frac)), BRK = 0.75;
  const X = (v) => (v <= BRK ? L + (v / BRK) * PW * 0.82 : L + PW * 0.9 + ((v - BRK) / (xmax - BRK)) * PW * 0.1), Y = (v) => T + (1 - v / 0.7) * PH;
  const sv = svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}` }, host);
  const parts = [];
  for (let i = 0; i <= 7; i++) { const v = i / 10; svgEl('line', { x1: L, x2: w - R, y1: Y(v), y2: Y(v), stroke: 'rgba(232,241,255,.14)', 'stroke-width': 1.5 }, sv);
    const tx = svgEl('text', { x: L - 14, y: Y(v) + 8, 'text-anchor': 'end', fill: '#B8CDEC', 'font-size': 22 }, sv); tx.textContent = v.toFixed(1); }
  [0, 0.25, 0.5, 0.75, xmax].forEach((v) => { const tx = svgEl('text', { x: X(v), y: h - B + 32, 'text-anchor': 'middle', fill: '#B8CDEC', 'font-size': 22 }, sv); tx.textContent = `${Math.round(v * 100)}%`; });
  const gx = [L + PW * 0.85, L + PW * 0.875];
  const axis = svgEl('path', { d: `M${L} ${T}V${h - B}H${gx[0]}M${gx[1]} ${h - B}H${w - R}M${gx[0] - 5} ${h - B + 9}l10 -18M${gx[1] - 5} ${h - B + 9}l10 -18`, stroke: '#E8F1FF', 'stroke-width': 3, fill: 'none', pathLength: 1, 'stroke-dasharray': 1 }, sv);
  const xl = svgEl('text', { x: L + PW / 2, y: h - 14, 'text-anchor': 'middle', fill: '#E8F1FF', 'font-size': 23 }, sv); xl.textContent = 'fraction of full-resolution pixels processed';
  const yl = svgEl('text', { transform: `translate(26 ${T + PH / 2}) rotate(-90)`, 'text-anchor': 'middle', fill: '#E8F1FF', 'font-size': 23 }, sv); yl.textContent = 'defect recall';
  const col = { downscale: '#9CC8FF', tiling: '#E8F1FF', drishyam: '#FFC857' };
  const ser = {};
  for (const k of ['downscale', 'tiling', 'drishyam']) {
    const pts = P[k]; const g = svgEl('g', {}, sv); ser[k] = { g, pts: [] };
    if (pts.length > 1) ser[k].line = svgEl('path', { d: pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.pixels_frac)} ${Y(p.recall)}`).join(' '), stroke: col[k], 'stroke-width': 4, fill: 'none', pathLength: 1, 'stroke-dasharray': 1 }, g);
    for (const p of pts) {
      const x = X(p.pixels_frac), y = Y(p.recall), q = svgEl('g', {}, g);
      if (p.recall_ci) svgEl('path', { d: `M${x} ${Y(p.recall_ci[0])}V${Y(p.recall_ci[1])}M${x - 6} ${Y(p.recall_ci[0])}h12M${x - 6} ${Y(p.recall_ci[1])}h12`, stroke: col[k], 'stroke-width': 2.5, opacity: 0.7 }, q);
      if (p.chance_recall != null) svgEl('path', { d: `M${x - 13} ${Y(p.chance_recall)}h26`, stroke: col[k], 'stroke-width': 3, 'stroke-dasharray': '3 4' }, q);
      if (k === 'downscale') svgEl('rect', { x: x - 9, y: y - 9, width: 18, height: 18, fill: '#082A55', stroke: col[k], 'stroke-width': 3.5 }, q);
      else if (k === 'tiling') svgEl('polygon', { points: `${x},${y - 12} ${x + 12},${y} ${x},${y + 12} ${x - 12},${y}`, fill: '#082A55', stroke: col[k], 'stroke-width': 3.5 }, q);
      else svgEl('circle', { cx: x, cy: y, r: 9, fill: '#082A55', stroke: col[k], 'stroke-width': 3.5 }, q);
      ser[k].pts.push({ q, x, y, p });
    }
  }
  return { sv, X, Y, axis, ser };
}
function buildResults() {
  const S = scene('results'), s = SC.results, sy = D.syn;
  const view = el('div', 'abs', S.e); css(view, { inset: 0, overflow: 'hidden' });
  const board = el('div', 'abs', view); css(board, { left: 0, top: 0, width: '3700px', height: '2300px', transformOrigin: '0 0' });
  const P2X = 2100, P3Y = 1300;   // panels far enough apart that a zoomed view never shows a neighbour's cut edge
  // P1: the chart
  const p1 = el('div', 'abs', board); css(p1, { left: '0px', top: '0px', width: '1530px', height: '836px' });
  el('div', 'abs', p1, `<span class="fig">FIG. 4</span><span class="lab" style="margin-left:16px;font-size:24px">${sy.n} labelled synthetic photos, ${sy.size[0]} × ${sy.size[1]} px · public/results/summary.json</span>`).style.cssText = 'left:0;top:0';
  const ch = el('div', 'abs', p1); css(ch, { left: '0px', top: '56px' });
  const C = chartSvg(ch, sy.policies, 1100, 770);
  const legend = el('div', 'abs', p1, `<div class="mono" style="font-size:22px;line-height:1.6"><span style="color:#9CC8FF">■</span> one downscale<br><span>◆</span> tiling (L2, L1, L0)<br><span class="amber">●</span> Drishyam, by pixel cap<br><span class="src" style="font-size:18px">bars: 95 % CI · dashes: chance, its boxes at random</span></div>`);
  css(legend, { left: '1130px', top: '56px', width: '400px' });
  const h = sy.head, tl = sy.tiling, ds = sy.ds;
  const card = el('div', 'box amb', p1, `<div class="lab amber">${pc(sy.cap)} pixel cap</div><div class="h2 amber" style="font-size:72px;margin:6px 0">${pc(sy.ratio)}</div>
    <div class="mono" style="font-size:21px;line-height:1.35">of exhaustive tiling’s recall (95 % CI ${pc(sy.ratio_ci[0])}–${pc(sy.ratio_ci[1])}), at ${pc(h.pixels_frac)} of the pixels (tiling: ${pc(tl.pixels_frac)})</div>
    <div class="mono" style="font-size:21px;line-height:1.35;margin-top:6px">target ${pc(sy.target)}: <b style="color:var(--red)">missed</b>${sy.ratio_ci[1] >= sy.target ? ' (point estimate)' : ''}</div>`);
  css(card, { left: '1130px', top: '250px', width: '400px', padding: '14px 18px' });
  const card2 = el('div', 'box', p1, `<div class="lab">ours / one downscale</div><div class="h2" style="font-size:56px;margin:4px 0">${fmt(sy.ds_ratio)}×</div><div class="mono" style="font-size:21px;line-height:1.35">95 % CI ${fmt(sy.ds_ci[0])}–${fmt(sy.ds_ci[1])}: one downscale (${fmt(ds.recall)} at ${pc(ds.pixels_frac)}) does as well or slightly better</div>`);
  css(card2, { left: '1130px', top: '600px', width: '400px', padding: '12px 18px' });
  // P2: 4608 px and the ablations
  const lg = D.large, ab = D.abl, p2 = el('div', 'abs', board); css(p2, { left: P2X + 'px', top: '0px', width: '1530px', height: '836px' });
  const vr = (v) => `<td class="n ${v < 0.95 ? 'bad' : v > 1.05 ? 'good' : ''}">${fmt(v)}</td>`;
  const dsl = Object.keys(ab);
  p2.innerHTML = `<div class="lab" style="font-size:24px">${lg.n} synthetic photos, ${lg.size[0]} × ${lg.size[1]} px</div><div class="src">public/results/large/summary.json</div>
    <table class="t" style="margin-top:12px"><tr><th>policy</th><th>pixels</th><th>recall</th></tr>
    <tr><td class="amber">Drishyam, ${pc(sy.cap)} cap</td><td class="n">${fmt(lg.head.pixels_frac)}</td><td class="n">${fmt(lg.head.recall)}</td></tr>
    <tr><td>one downscale</td><td class="n">${fmt(lg.ds.pixels_frac)}</td><td class="n">${fmt(lg.ds.recall)}</td></tr>
    <tr><td>tiling at L2 (quarter size)</td><td class="n">${fmt(lg.l2.pixels_frac)}</td><td class="n">${fmt(lg.l2.recall)}</td></tr>
    <tr><td colspan="3">ours / one downscale: <b class="amber">${fmt(lg.ds_ratio)}×</b> (95 % CI ${fmt(lg.ds_ci[0])}–${fmt(lg.ds_ci[1])})</td></tr></table>
    <div class="lab" style="font-size:24px;margin-top:200px">Ablations, ${pc(sy.cap)} cap: recall / full agent</div><div class="src">public/results/ablations.json</div>
    <table class="t" style="margin-top:12px"><tr><th>variant</th>${dsl.map((d) => `<th>${ab[d].label}</th>`).join('')}</tr>
    ${[['no-survey', 'no survey look'], ['no-verify', 'no verify'], ['no-uncertainty', 'no uncertainty term'], ['governor-on', 'governor on']].map(([k, n]) => `<tr><td>${n}</td>${dsl.map((d) => vr(ab[d][k])).join('')}</tr>`).join('')}
    <tr><td>random gaze order (95 % CI)</td>${dsl.map((d) => `<td class="n ${ab[d].random_hi < 1 ? 'bad' : ''}">${fmt(ab[d].random_lo)}–${fmt(ab[d].random_hi)}</td>`).join('')}</tr></table>`;
  // P3: CODEBRIM, false alarms, the trained segmenter
  const cb = D.codebrim, cl = D.clean, p3 = el('div', 'abs', board); css(p3, { left: '0px', top: P3Y + 'px', width: '1800px', height: '836px' });
  p3.innerHTML = `<div class="box" style="left:0;top:0;width:740px;height:560px;padding:26px 30px">
      <div class="lab">${cb.n} real photos · CODEBRIM</div><div class="h1 amber" style="font-size:100px;margin:12px 0;white-space:nowrap">about 1 in ${cb.one_in}</div>
      <div class="mono" style="font-size:26px">defects found by every method, ours included (recall ${fmt(cb.range[0])}–${fmt(cb.range[1])}). Drishyam’s boxes placed at random would find ${fmt(cb.chance)}.</div>
      <div class="mono amber" style="font-size:26px;margin-top:16px">The classical segmenter is the bottleneck.</div>
      <div class="src" style="margin-top:16px">public/results/codebrim/summary.json · the licence bars showing the photos</div></div>
    <div class="box" style="left:1000px;top:0;width:740px;height:560px;padding:26px 30px">
      <div class="lab">${cl.n} synthetic photos with no damage</div><div class="h1 amber" style="font-size:120px;margin:12px 0">${pc(cl.any)}</div>
      <div class="mono" style="font-size:26px">of them got a report from Drishyam (${pc(cl.cap)} cap)</div>
      <table class="t" style="margin-top:20px"><tr><th>boxes per image</th><th></th></tr><tr><td>Drishyam</td><td class="n">${fmt(cl.boxes)}</td></tr><tr><td>one downscale</td><td class="n">${fmt(cl.ds_boxes)}</td></tr><tr><td>L0 tiling</td><td class="n">${fmt(cl.tl_boxes)}</td></tr></table>
      <div class="src" style="margin-top:14px">public/results/clean/summary.json · false_alarms</div></div>
    ${D.model ? `<div id="pend" class="box amb" style="left:0;top:600px;width:1740px;padding:16px 26px"><div class="lab amber">trained segmenter (dacl10k) on the same ${cb.n} CODEBRIM photos · governor off · both charged the ${D.model.arm} cost table</div>
      <div class="mono" style="font-size:26px;margin-top:10px">recall ${fmt(D.model.recall)} (95 % CI ${fmt(D.model.ci[0])}–${fmt(D.model.ci[1])}) vs ${fmt(D.model.classical)} classical · its boxes at random ${fmt(D.model.chance)} · ${D.model.tiling_wins.length ? 'tiling at ' + D.model.tiling_wins.join(', ') + ' still finds more' : 'no tiling level finds more'} · near-duplicates in its training set: ${D.model.near_dups}</div></div>`
      : '<div id="pend" class="stamp" style="left:330px;top:620px;font-size:46px">trained segmenter: results pending</div>'}`;
  const VW = 1530, VH = 836;
  // the whole chart while it draws and the 85 % card is read; the card alone; the whole chart again for the 0.94x card;
  // each table framed so its first column and heading stay inside the view under the 8 % push
  // the ablation rows as r5 says them: survey; uncertainty and governor; verify; random order (a camera stop and a row highlight each)
  const r5 = LINE.r5.say, AB = [['survey', [6]], ['uncertainty', [8, 9]], ['verify', [7]], ['random', [10]]].map(([w, rows]) => [at('r5') + LINE.r5.dur * r5.indexOf(w) / r5.length, rows]);
  const AX = P2X - 20 + 0.46 * 1400;
  const K = [[s.t0, 765, 418, 1660], [at('r1') + LINE.r1.dur * 0.72, 1330, 446, 760], [at('r3') + 0.1, 765, 418, 1660],
    [at('r4') - 0.2, P2X - 20 + 0.46 * 1300, 150, 1300], [at('r5') - 0.1, AX, 705, 1400], ...AB.slice(1).map(([T], i) => [T - 0.2, AX, 712 + 6 * i, 1400]),
    [at('r6') - 0.2, 380, P3Y + 300, 1150], [at('r7') - 0.1, 1370, P3Y + 300, 1150], [at('r8') - 0.1, 870, P3Y + 420, 1960]];
  const cam = camBoard({ board, VW, VH }, K, 0.08);
  const pend = $('#pend', p3);
  S.draws.push((t) => {
    cam(t);
    // the chart draws itself: axes, then each series point by point, then the callouts snap in
    const T0 = s.t0 + 0.4; C.axis.setAttribute('stroke-dashoffset', 1 - eio(seg(t, T0, T0 + 0.9)));
    let n = 0; for (const key of ['downscale', 'tiling', 'drishyam']) { const sr = C.ser[key];
      sr.pts.forEach((q, i) => { const tt = T0 + 1.0 + (n++) * 0.28; const p = seg(t, tt, tt + 0.3); q.q.setAttribute('opacity', p); q.q.setAttribute('transform', `translate(${q.x} ${q.y}) scale(${(0.4 + 0.6 * spring(p)).toFixed(3)}) translate(${-q.x} ${-q.y})`); });
      if (sr.line) sr.line.setAttribute('stroke-dashoffset', 1 - eio(seg(t, T0 + 1.0 + (n - sr.pts.length) * 0.28, T0 + 1.0 + n * 0.28))); }
    const cp = seg(t, at('r1') + LINE.r1.dur * 0.35, at('r1') + LINE.r1.dur * 0.35 + 0.4), cp2 = seg(t, at('r3'), at('r3') + 0.4);
    card2.style.top = card.offsetTop + card.offsetHeight + 14 + 'px';   // stacked under the 85 % card, never over it
    css(card, { opacity: cp, transform: `scale(${(0.9 + 0.1 * spring(cp)).toFixed(3)})` }); css(card2, { opacity: cp2, transform: `scale(${(0.9 + 0.1 * spring(cp2)).toFixed(3)})` });
    const sp = seg(t, at('r8') + 0.2, at('r8') + 0.45); css(pend, { opacity: sp, transform: `${D.model ? '' : 'rotate(-6deg) '}scale(${((D.model ? 1.1 : 1.6) - (D.model ? 0.1 : 0.6) * eo(sp)).toFixed(3)})` });
    const hot = t < end('r5') + 0.3 ? (AB.filter(([T]) => t >= T - 0.1).pop() || [0, []])[1] : [];
    [...p2.querySelectorAll('tr')].forEach((r, i) => { const T = i < 5 ? at('r4') + 0.3 + i * 0.25 : at('r5') + 0.2 + (i - 5) * 0.3; r.style.opacity = 0.15 + 0.85 * seg(t, T, T + 0.25);
      css(r, { background: hot.includes(i) ? 'rgba(255,200,87,.16)' : 'transparent', outline: hot.includes(i) ? '3px solid var(--amber)' : 'none' }); });
  });
  let n = 0; for (const key of ['downscale', 'tiling', 'drishyam']) for (const _ of sy.policies[key]) cue(s.t0 + 1.4 + (n++) * 0.28, 'pop', n);
  cue(s.t0 + 0.4, 'scribble', 0.9); cue(at('r8') + 0.25, 'stamp'); [at('r4'), at('r6'), at('r7')].forEach((T) => cue(T - 0.9, 'whoosh'));
}

// 9. failure cases: the plates from failures.json, and one CODEBRIM case in words only
function buildFailures() {
  const S = scene('failures'), s = SC.failures, B = boardView(S.e), H = B.board;
  const cells = D.fails.map((f, i) => {
    const b = el('div', 'box', H); css(b, { left: (i % 2) * 900 + 'px', top: Math.floor(i / 2) * 400 + 'px', width: '755px', height: '322px', overflow: 'hidden' });
    const im = img(f.img, b); css(im, { position: 'absolute', left: '0px', top: '0px', height: '316px', width: '330px', objectFit: 'cover', borderRight: '3px solid var(--line)' });
    el('div', 'abs', b, `<div class="lab" style="color:var(--line-3)">FIG. 5.${i + 1} · ${f.id}</div><div class="h2 amber" style="font-size:40px;margin:10px 0 12px;white-space:normal">${f.cat}</div>
      <div class="mono" style="font-size:22px;line-height:1.35;white-space:normal"><span style="color:var(--line-2)">rule that would catch it:</span> ${f.ruleShort}</div>
      <div class="mono" style="font-size:22px;color:var(--red);margin-top:8px">not implemented</div>`).style.cssText = 'left:352px;top:18px;width:385px';
    return b;
  });
  const rf = D.real_fail, txt = el('div', 'box amb', H, `<span class="lab amber">CODEBRIM ${rf.id}, in words only</span><span class="mono" style="margin-left:18px;font-size:26px">ground truth <b>${rf.gt}</b> · reported <b>${rf.got}</b></span>`);
  css(txt, { left: '0px', top: '800px', width: '1655px', height: '96px', padding: '28px 26px' });
  const say = LINE.f1.say, T = ['graffiti', 'formwork', 'glare', 'blur'].map((w, i) => at('f1') + (i ? LINE.f1.dur * say.indexOf(w) / say.length : 0.2));
  S.draws.push((t) => {
    cells.forEach((c, i) => { const p = seg(t, T[i] - 0.5, T[i] - 0.15), cur = t < at('f2') && i === T.filter((x) => t >= x - 0.3).length - 1;
      css(c, { opacity: p, transform: `translateY(${((1 - eo(p)) * 30).toFixed(1)}px)`, borderColor: cur ? 'var(--amber)' : 'var(--line)' }); });
    const q = seg(t, at('f3') - 0.1, at('f3') + 0.3); css(txt, { opacity: q, transform: `translateX(${((1 - eo(q)) * -40).toFixed(1)}px)` });
  });
  S.draws.unshift(camBoard(B, [[s.t0, 828, 448, 1800], [T[0] - 0.2, 380, 165, 900], [T[1] - 0.2, 1277, 165, 900], [T[2] - 0.2, 380, 561, 900], [T[3] - 0.2, 1277, 561, 900],
    [at('f2') + 0.2, 828, 440, 1800], [at('f3') - 0.1, 828, 452, 1800]]));
  T.forEach((x) => cue(x - 0.3, 'drop')); cue(at('f3'), 'switch');
}

// 10. limits: the site's own wording (footer note, Human sign-off, the CODEBRIM citation), ticked off as they are said
function buildLimits() {
  const S = scene('limits'), s = SC.limits, B = boardView(S.e), H = B.board;
  el('div', 'abs', H, '<span class="fig">LIMITS</span><span class="src" style="margin-left:16px">in the site’s words (public/index.html: footer note, Human sign-off, the CODEBRIM citation)</span>').style.cssText = 'left:60px;top:40px';
  const notes = LIT.limits.map((n, i) => (i === 3 ? 'CODEBRIM: ' + n[0].toLowerCase() + n.slice(1) : n));   // checked against public/index.html by build.py
  const L = at('x1'), d = LINE.x1.dur;
  const rows = notes.map((n, i) => { const r = el('div', 'abs', H, `<span style="display:inline-grid;place-items:center;width:64px;height:64px;border:4px solid var(--amber);border-radius:50%;color:var(--amber);font:700 30px/1 var(--fm);margin-right:30px">${i + 1}</span><span class="h2" style="font-size:46px;white-space:nowrap"></span>`);
    css(r, { left: '60px', top: 150 + i * 150 + 'px', display: 'flex', alignItems: 'center' }); return r; });
  const T = [L + 0.1, L + d * 0.28, L + d * 0.56, L + d * 0.72];
  S.draws.push((t) => rows.forEach((r, i) => { const p = eo(seg(t, T[i], T[i] + 0.35)); css(r, { opacity: p, transform: `translateX(${((1 - p) * -40).toFixed(1)}px)` });
    r.lastChild.textContent = typed(notes[i], seg(t, T[i], T[i] + 0.7)); }));
  S.draws.unshift(camBoard(B, [[s.t0, 765, 418, 1720], ...T.map((x, i) => [x - 0.1, 765, 380 + i * 24, 1700])]));   // full width: every line stays whole under the push
  T.forEach((x, i) => cue(x, 'tick', i + 2));
}

// 11. reproduce: the three commands, typed
function buildRepro() {
  const S = scene('repro'), s = SC.repro, B = boardView(S.e), H = B.board;
  const term = el('div', 'term', H); css(term, { left: '0px', top: '60px', width: '1530px', height: '560px', fontSize: '34px', lineHeight: '1.75' });
  const cmds = [['uv sync --all-extras && make test', LIT.test_note], ['make eval', `${D.syn.n} synthetic images → public/results`], ['scripts/aws.sh deploy | smoke | destroy', 'the CDK stack: CloudFront, WAF, Lambda arm64, ...']];
  const repo = el('div', 'abs mono', H, `<span class="lab">source</span> ${D.urls.repo}`); css(repo, { left: '0px', top: '680px', fontSize: '38px' });
  const L = at('y1') - 0.2, d = LINE.y1.dur + 0.4, Ts = [L, L + d * 0.32, L + d * 0.6];
  S.draws.push((t) => {
    term.innerHTML = cmds.map(([c, note], i) => { const T = Ts[i], n = Math.floor(seg(t, T, T + d * 0.24) * c.length);
      return t < T ? '' : `<span class="p">$</span> ${c.slice(0, n)}${n < c.length ? '<span class="caret"></span>' : `\n  <span class="ok"># ${note}</span>`}`; }).filter(Boolean).join('\n');
    repo.style.opacity = seg(t, L + d * 0.85, L + d * 0.85 + 0.3);
  });
  S.draws.unshift(camBoard(B, [[s.t0, 765, 418, 1660], [Ts[0], 640, 200, 1360], [Ts[1], 640, 330, 1360], [Ts[2], 640, 420, 1360], [L + d * 0.85, 765, 418, 1660]]));
  Ts.forEach((T) => cue(T, 'type', d * 0.24));
}

// 12. end card
function buildEnd() {
  const S = scene('end'), s = SC.end;
  const e = el('div', 'abs', S.e, `<div class="h1" style="font-size:190px">DRISHYAM</div><div class="lab" style="font-size:26px;margin-top:6px">${LIT.sanskrit}</div>
    <div class="h2 amber" style="font-size:60px;margin-top:40px">Look where it matters. Ask a human first.</div>
    <div class="mono" style="font-size:40px;margin-top:46px">${D.urls.live.replace('https://', '')}</div><div class="mono" style="font-size:40px;margin-top:8px">${D.urls.repo.replace('https://', '')}</div>
    <div class="lab" style="font-size:26px;margin-top:44px;color:var(--line)">${LIT.team.map((x) => x.split(',')[0]).join(' · ')}</div><div class="lab" style="font-size:22px;margin-top:10px">${ENTRY}</div>`);
  css(e, { left: '0px', top: '40px', width: '1530px' });
  const kids = [...e.children];
  S.draws.push((t) => kids.forEach((k, i) => { const p = eo(seg(t, s.t0 + 0.15 + i * 0.22, s.t0 + 0.5 + i * 0.22)); css(k, { opacity: p, transform: `translateY(${((1 - p) * 24).toFixed(1)}px)` }); }));
  cue(s.t0 + 0.3, 'final');
}

// ------------------------------------------------------------------ the 9:16 cut
function buildShort() {
  document.body.classList.add('short'); css(document.documentElement, { width: '1080px', height: '1920px' }); css(document.body, { width: '1080px', height: '1920px' });
  const Sh = window.SHORT, ev = F.ev, dur = Math.max(15.5, Sh.dur);
  // it ends on the settled APPROVED stamp: the last recorded frame before the page scrolls on to the next finding
  const SETTLED = Math.max(...F.frames.filter((x) => x.t <= ev['approved:0'] + 1.5 && x.sy === F.frames.find((y) => y.t >= ev['approved:0']).sy).map((x) => x.t)) - ev['approved:0'];
  el('div', 'sheetframe', stage);
  const top = el('div', 'abs', stage, `<div class="h1" style="font-size:118px">DRISHYAM</div><div class="lab" style="font-size:26px;margin-top:10px">one photo · a few looks · one human</div>`);
  css(top, { left: '108px', top: '150px', width: '864px' });
  const l0 = Sh.lines[0], l1 = Sh.lines[1];
  const warp = { t0: 0, t1: dur, pts: [{ t: 0.2, f: ev['step:1'] - 0.3 }, { t: l0.t + l0.dur + 0.3, f: ev['step:4'] + 1.0 }, { cut: true, a: ev['step:4'] + 1.1, b: ev['click:Reviewer'] - 0.8 }, { t: dur, f: ev['approved:0'] + SETTLED }] };
  const host = el('div', 'abs', stage); css(host, { left: '60px', top: '420px', width: '960px', height: '1000px' });
  const fw = new Footage(host, 960, 1000, D.urls.live.replace('https://', ''), warp, [KF(ev['step:1'] - 0.3, 340, 560, 620), KF(ev['step:4'] + 0.6, 330, 600, 520), KF(ev['click:Reviewer'] - 0.6, 330, 700, 640), KF(ev['approved:0'] + 0.5, 340, 640, 660)]);
  const cap = el('div', 'abs', stage); css(cap, { left: '108px', top: '1450px', width: '864px', textAlign: 'center' }); const cs = el('span', '', cap);
  css(cs, { display: 'inline-block', padding: '12px 22px', background: 'rgba(4,20,44,.94)', border: '2.5px solid var(--line-3)', borderLeft: '8px solid var(--amber)', font: '500 36px/1.3 var(--fm)', color: 'var(--line)' });
  const url = el('div', 'abs lab', stage, D.urls.live.replace('https://', '')); css(url, { left: '108px', top: '1440px', width: '864px', textAlign: 'center', fontSize: '28px', color: 'var(--amber)' });
  const caps = [[l0.t, l0.t + l0.dur, LINE.o1.cap], [l1.t, l1.t + l1.dur, l1.text]];   // the captions of the two lines the short speaks
  window.DURATION = dur;
  window.render = async (t) => {
    const c = caps.find(([a, b]) => t >= a - 0.05 && t <= b + 0.3); cs.textContent = c ? c[2] : ''; cap.style.opacity = c ? 1 : 0;
    url.style.opacity = seg(t, dur - 2.6, dur - 2.2); url.style.transform = `translateY(${(1 - seg(t, dur - 2.6, dur - 2.2)) * 20}px)`;
    host.style.opacity = seg(t, 0, 0.3); await fw.draw(t);
  };
  fw.clicksSfx(); const T = filmOf(fw.S, ev['approved:0'] + 0.2); if (T) cue(T, 'stamp'); cue(dur - 2.4, 'final');
  window.SFX = SFX.sort((a, b) => a.t - b.t);
  window.ready = Promise.all([document.fonts.load('700 60px "Archivo Narrow"'), document.fonts.load('500 30px "IBM Plex Mono"'), document.fonts.load('700 30px "IBM Plex Mono"'), document.fonts.ready, ...loads]).then(() => window.render(0));
}

// ------------------------------------------------------------------ assemble
if (SHORT) buildShort();
else {
  buildFrame();
  buildOpen(); buildTeam(); buildProblem(); buildLive(); buildBreak(); buildMcp(); buildArch(); buildResults(); buildFailures(); buildLimits(); buildRepro(); buildEnd();
  ORDER.forEach((id, i) => { if (i) { cue(SC[id].t0, 'whoosh'); cue(SC[id].t0 + 0.2, 'switch'); } });
  window.DURATION = F.dur;
  window.SFX = SFX.sort((a, b) => a.t - b.t);
  window.MARKS = Object.fromEntries(ORDER.map((id) => [id, SC[id].t0]));
  window.render = async (t) => {
    drawFrame(t);
    for (const id of ORDER) if (visible(id, t)) for (const d of SCN[id].draws) await d(t);
  };
  window.ready = Promise.all([document.fonts.load('700 60px "Archivo Narrow"'), document.fonts.load('500 30px "Archivo Narrow"'), document.fonts.load('400 30px "IBM Plex Mono"'),
    document.fonts.load('500 30px "IBM Plex Mono"'), document.fonts.load('700 30px "IBM Plex Mono"'), document.fonts.ready, ...loads]).then(() => window.render(0));
}
