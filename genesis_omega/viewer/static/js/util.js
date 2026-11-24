// Small shared helpers: DOM, formatting, epistemic tags, colours, sphere coordinates.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function fmt(v, d = 2) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v !== "number") return String(v);
  if (!Number.isFinite(v)) return "—";
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  const a = Math.abs(v);
  if (a >= 1e6 || (a < 1e-3 && a > 0)) return v.toExponential(2);
  return v.toFixed(d);
}

export function pct(v) {
  return Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—";
}

export const row = (k, v) => `<div class="row"><span>${esc(k)}</span><span>${v}</span></div>`;
export const bar = (f) => `<div class="bar"><i style="width:${Math.round(Math.max(0, Math.min(1, f)) * 100)}%"></i></div>`;

export function tagEpi(e) {
  return e ? `<span class="tag t-${esc(e)}" title="epistemic status">${esc(e)}</span>` : "";
}

export function tagVal(v) {
  if (!v) return "";
  const k = String(v).split(" ")[0];
  return `<span class="tag v-${esc(k)}" title="validation status">${esc(v)}</span>`;
}

export function tagStatus(s) {
  const k = String(s).split(" ")[0];
  return `<span class="tag s-${esc(k)}">${esc(s)}</span>`;
}

// ------------------------------------------------------------------ colours
/** Approximate sRGB of a blackbody (same fit as the backend), components in 0..1. */
export function blackbody(teff) {
  const t = Math.max(1000, Math.min(40000, teff)) / 100;
  let r, g, b;
  if (t <= 66) {
    r = 255;
    g = 99.4708025861 * Math.log(t) - 161.1195681661;
    b = t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  } else {
    r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
    g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
    b = 255;
  }
  return [r, g, b].map((c) => Math.min(255, Math.max(0, c)) / 255);
}

/** Deterministic, well-separated colour for an integer id (materials, unknown species). */
export function idColor(n) {
  const h = ((n * 0.6180339887) % 1 + 1) % 1;
  return hsl(h, 0.55, 0.6);
}

export function hsl(h, s, l) {
  const f = (n) => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

export const css = (c) => `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;

const RAMPS = {
  thermal: [[0.05, 0.1, 0.45], [0.2, 0.55, 0.95], [0.95, 0.95, 0.95], [0.98, 0.6, 0.2], [0.75, 0.08, 0.08]],
  life: [[0.06, 0.05, 0.04], [0.25, 0.22, 0.1], [0.15, 0.5, 0.12], [0.55, 0.85, 0.25], [0.95, 0.95, 0.5]],
  water: [[0.5, 0.36, 0.2], [0.75, 0.68, 0.45], [0.4, 0.75, 0.6], [0.15, 0.5, 0.85], [0.08, 0.2, 0.6]],
  magma: [[0.0, 0.0, 0.02], [0.3, 0.05, 0.4], [0.75, 0.2, 0.4], [0.98, 0.55, 0.3], [0.99, 0.95, 0.7]],
  ice: [[0.02, 0.04, 0.08], [0.1, 0.3, 0.45], [0.35, 0.7, 0.8], [0.8, 0.95, 1.0], [1, 1, 1]],
};

/** Sample a named colour ramp at f ∈ [0,1]. */
export function ramp(name, f) {
  const r = RAMPS[name] || RAMPS.magma;
  const x = Math.max(0, Math.min(1, f)) * (r.length - 1);
  const i = Math.min(r.length - 2, Math.floor(x));
  const t = x - i;
  return [0, 1, 2].map((k) => r[i][k] * (1 - t) + r[i + 1][k] * t);
}

// ------------------------------------------------------------------ geometry
// Simulation coordinates (x, y, z with z = north) map to three.js (x, z, −y) with y up.
export const DEG = Math.PI / 180;

export function latLonToXYZ(lat, lon, r, out) {
  const c = Math.cos(lat);
  out[0] = r * c * Math.cos(lon);
  out[1] = r * Math.sin(lat);
  out[2] = -r * c * Math.sin(lon);
  return out;
}

export function xyzToLatLon(x, y, z) {
  const r = Math.hypot(x, y, z) || 1;
  return [Math.asin(Math.max(-1, Math.min(1, y / r))), Math.atan2(-z, x)];
}

export function angularDistance(lat1, lon1, lat2, lon2) {
  const c = Math.sin(lat1) * Math.sin(lat2) + Math.cos(lat1) * Math.cos(lat2) * Math.cos(lon1 - lon2);
  return Math.acos(Math.max(-1, Math.min(1, c)));
}

/** Keplerian position (AU) of a planet with the backend's elements at galaxy time t (years). */
export function orbitPosition(p, t) {
  const a = p.a, e = p.e, per = Math.max(p.period, 1e-6);
  const m = ((p.m0 + (2 * Math.PI * t) / per) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
  let E = m;
  for (let k = 0; k < 8; k++) E -= (E - e * Math.sin(E) - m) / (1 - e * Math.cos(E));
  const x = a * (Math.cos(E) - e);
  const y = a * Math.sqrt(1 - e * e) * Math.sin(E);
  const w = p.omega;
  const xr = x * Math.cos(w) - y * Math.sin(w);
  const yr = x * Math.sin(w) + y * Math.cos(w);
  return [xr, yr, xr * p.incl];
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
