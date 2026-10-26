// Status vocabulary + badge renderers. Every status carries an icon and a text label (never colour alone).
import { icon } from './icons.js';
import { esc, isNum, DASH } from '../util.js';

export const STATES = ['normal', 'watch', 'warning', 'fault', 'unknown', 'offline', 'tripped'];
export const STATE_LABEL = {
  normal: 'Normal', watch: 'Watch', warning: 'Warning', fault: 'Fault', unknown: 'Unknown',
  offline: 'Offline', tripped: 'Tripped', nodata: 'No data',
};
export const STATE_HELP = {
  normal: 'All monitored statistics within limits',
  watch: 'EWMA above warning limit, persistence not yet met',
  warning: 'Deviation persisted beyond the persistence window',
  fault: 'Sustained deviation: CUSUM decision + persistence',
  unknown: 'Data quality insufficient to judge',
  offline: 'Turbine stopped / non-production state',
  tripped: 'Protection trip (simulation)',
  nodata: 'Not part of the current data source',
};
/** Severity rank for sorting (higher = worse). */
export const STATE_RANK = { tripped: 6, fault: 5, warning: 4, unknown: 3, watch: 2, offline: 1, normal: 0, nodata: -1 };

export function stateBadge(st, { lg = false } = {}) {
  const s = STATE_LABEL[st] ? st : 'unknown';
  return `<span class="badge st-${s}${lg ? ' lg' : ''}" title="${esc(STATE_HELP[s])}">${icon('st_' + s)}${STATE_LABEL[s]}</span>`;
}

export function caseStatusBadge(s) {
  const lab = { open: 'Open', acknowledged: 'Acknowledged', closed: 'Closed' }[s] || s;
  return `<span class="badge case-${esc(s)}">${esc(lab)}</span>`;
}

export function qualityBadge(q) {
  const s = typeof q === 'string' ? q : q?.state || 'ok';
  const issues = (q && q.issues) || [];
  if (s === 'nodata') return `<span class="badge st-nodata">${icon('st_nodata')}No data</span>`;
  const lab = { ok: 'OK', degraded: 'Degraded', bad: 'Bad' }[s] || s;
  const ic = s === 'ok' ? 'check' : 'alert';
  return `<span class="badge q-${esc(s)}" title="${esc(issues.join('; ') || 'No data-quality issues')}">${icon(ic)}${esc(lab)}</span>`;
}

/** Health 0..100 as a small bar; colour by band, number always shown. */
export function healthBar(v) {
  if (!isNum(v)) return `<span class="muted">${DASH}</span>`;
  const c = v >= 85 ? 'var(--st-normal)' : v >= 65 ? 'var(--st-watch)' : v >= 40 ? 'var(--st-warning)' : 'var(--st-fault)';
  return `<span class="health" title="Health index ${v.toFixed(0)} / 100"><span class="bar"><i style="width:${Math.max(2, v)}%;background:${c}"></i></span><b>${v.toFixed(0)}</b></span>`;
}

export function meter(frac01, color = 'var(--accent)') {
  const w = isNum(frac01) ? Math.max(0, Math.min(1, frac01)) * 100 : 0;
  return `<span class="meter" style="display:block"><i style="width:${w}%;background:${color}"></i></span>`;
}

export function yesNo(v, yes = 'Yes', no = 'No') {
  if (v === true) return `<span class="badge yes">${icon('check')}${yes}</span>`;
  if (v === false) return `<span class="badge no">${icon('x')}${no}</span>`;
  return `<span class="muted">${DASH}</span>`;
}

/** CSS colour token for a state (used by charts and 3D via getComputedStyle). */
export const stateVar = (st) => `--st-${STATE_LABEL[st] ? (st === 'nodata' ? 'nodata' : st) : 'unknown'}`;
