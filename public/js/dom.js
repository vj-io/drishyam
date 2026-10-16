// Tiny DOM helpers. Everything is built with createElement and text nodes, never with markup strings.
const SVG_NS = 'http://www.w3.org/2000/svg';

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  el.append(...kids.flat().filter((k) => k != null && k !== false));
  return el;
}

export function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  el.append(...kids.flat().filter((k) => k != null));
  return el;
}

const B64 = /^[A-Za-z0-9+/=]+$/;

/** data: URL for a server-supplied base64 payload, or null when it is not plain base64. */
export function dataUrl(mime, b64) {
  return typeof b64 === 'string' && b64.length > 0 && B64.test(b64) ? `data:${mime};base64,${b64}` : null;
}

export const num = (v) => typeof v === 'number' && Number.isFinite(v);
