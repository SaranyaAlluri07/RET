// Small shared helpers: formatting, time, storage, DOM.

/** HTML-escape a value for safe interpolation into template strings. */
export function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export const DASH = '—';

/** Fixed-decimals number, or an em dash when missing. */
export function num(v, d = 1) { return isNum(v) ? v.toFixed(d) : DASH; }
/** Per-unit (0..1) value as percent of rated. */
export function pctPu(v, d = 1) { return isNum(v) ? (v * 100).toFixed(d) : DASH; }
/** Fraction as percent string with unit. */
export function pct(v, d = 0) { return isNum(v) ? `${(v * 100).toFixed(d)} %` : DASH; }
/** "n/N (p %)" - every percentage is shown with its counts. */
export function frac(n, N, d = 0) {
  if (!isNum(n) || !isNum(N) || N === 0) return DASH;
  return `${n}/${N} (${((n / N) * 100).toFixed(d)} %)`;
}
export function signed(v, d = 1) { return isNum(v) ? `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(d)}` : DASH; }

// ---------------------------------------------------------------- time
// Timestamps in the API are ISO-8601 strings; we interpret them consistently in local time
// (same as the ECharts time axis), so labels and chart positions always agree.
export function ms(t) { if (t == null) return NaN; return typeof t === 'number' ? t : Date.parse(t); }
const p2 = (n) => String(n).padStart(2, '0');
export function fmtTime(t, { seconds = false, date = true } = {}) {
  const v = ms(t); if (!Number.isFinite(v)) return DASH;
  const d = new Date(v);
  const hm = `${p2(d.getHours())}:${p2(d.getMinutes())}${seconds ? ':' + p2(d.getSeconds()) : ''}`;
  return date ? `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${hm}` : hm;
}
export function fmtDate(t) { const v = ms(t); if (!Number.isFinite(v)) return DASH; const d = new Date(v); return `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())}`; }
/** Duration in hours -> "3 d 4 h" / "5.2 h" / "40 min". */
export function fmtHours(h) {
  if (!isNum(h)) return DASH;
  const a = Math.abs(h), s = h < 0 ? '−' : '';
  if (a < 1) return `${s}${Math.round(a * 60)} min`;
  if (a < 48) return `${s}${a.toFixed(1)} h`;
  const d = Math.floor(a / 24); return `${s}${d} d ${Math.round(a - d * 24)} h`;
}
export function hoursBetween(a, b) { return (ms(b) - ms(a)) / 3.6e6; }

// ---------------------------------------------------------------- storage (never throws)
export function lsGet(key, fallback = null) {
  try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
}
export function lsSet(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } }

// ---------------------------------------------------------------- DOM
export function h(html) { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; }
export function $(sel, root = document) { return root.querySelector(sel); }
export function $$(sel, root = document) { return [...root.querySelectorAll(sel)]; }

export function toast(msg, kind = '') {
  const box = document.getElementById('toasts'); if (!box) return;
  const el = h(`<div class="toast ${kind}">${esc(msg)}</div>`);
  box.appendChild(el); setTimeout(() => el.remove(), 4200);
}

// ---------------------------------------------------------------- domain labels
const COMPONENT_LABELS = {
  generator_bearing: 'Generator bearing', gearbox: 'Gearbox', hydraulic: 'Hydraulic group', transformer: 'Transformer',
  converter: 'Converter', performance: 'Power performance', generator: 'Generator', stator: 'Generator stator',
  pitch: 'Pitch system', yaw: 'Yaw system', main_bearing: 'Main bearing',
};
export function compLabel(name) {
  if (name == null || name === '') return DASH;
  if (COMPONENT_LABELS[name]) return COMPONENT_LABELS[name];
  const s = String(name).replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}
export function turbineLabel(id, meta) {
  const a = meta?.assets?.find((x) => String(x.id) === String(id));
  return a?.label ?? `T-${id}`;
}
