// Data-source adapter: gives views one mode-agnostic interface over
//   - SIMULATION: FleetState / TurbineSeries / Case[] from /api/sim/*
//   - REPLAY: one precomputed ReplayFile (real SCADA, one asset) played client-side with a cursor.
// Views never branch on raw API shapes; they call fleet(), series(id), cases(), now().
import { state } from './state.js';
import { api } from './api.js';
import { ms, lsGet, lsSet, isNum } from './util.js';

const ALARM = new Set(['warning', 'fault']);

// ---------------------------------------------------------------- replay preparation
/** Precompute arrays used for cursor-based snapshots (energy prefix sums, alarm onsets). */
export function prepareReplay(r) {
  const n = r.t.length;
  const tms = r.t.map(ms);
  const dts = []; for (let i = 1; i < n; i++) dts.push(tms[i] - tms[i - 1]);
  dts.sort((a, b) => a - b);
  const stepH = (dts[dts.length >> 1] || 3.6e6) / 3.6e6;
  const eAct = new Float64Array(n), eExp = new Float64Array(n), eLost = new Float64Array(n);
  let a = 0, e = 0, l = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.max(0, r.power[i] ?? 0), x = Math.max(0, r.expected[i] ?? 0);
    a += p * stepH; e += x * stepH; l += Math.max(0, x - p) * stepH;
    eAct[i] = a; eExp[i] = e; eLost[i] = l;
  }
  const onsets = [];
  for (let i = 0; i < n; i++) if (ALARM.has(r.state[i]) && !ALARM.has(r.state[i - 1])) onsets.push(i);
  const firstAlarm = (r.cases || []).map((c) => ms(c.alarm_time)).filter(Number.isFinite).sort((x, y) => x - y)[0]
    ?? (onsets.length ? tms[onsets[0]] : null);
  r._prep = { tms, stepH, eAct, eExp, eLost, onsets, firstAlarm };
  return r;
}

/** Index of the last row with t <= time (binary search). */
export function indexAt(tms, time) {
  let lo = 0, hi = tms.length - 1;
  if (time <= tms[0]) return 0;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (tms[mid] <= time) lo = mid; else hi = mid - 1; }
  return lo;
}

// ---------------------------------------------------------------- current time
export function now() {
  if (state.mode === 'replay') {
    const r = state.replay; return r ? r._prep.tms[Math.min(state.cursor, r.t.length - 1)] : NaN;
  }
  return state.fleet ? ms(state.fleet.t) : NaN;
}

// ---------------------------------------------------------------- fleet snapshot
const qState = (q) => (q == null ? 'ok' : typeof q === 'string' ? q : q.state || 'ok');
const qIssues = (q) => (q && typeof q === 'object' && Array.isArray(q.issues) ? q.issues : []);

function replaySnapshot(r, i, limits) {
  const comps = Object.entries(r.components || {}).map(([name, c]) => {
    const ew = c.ewma?.[i] ?? null;
    return { name, score: c.score?.[i] ?? null, ewma: ew, cusum: c.cusum?.[i] ?? null,
      state: !isNum(ew) ? 'unknown' : ew > (limits.ewma_fault ?? 3) ? 'fault' : ew > (limits.ewma_warn ?? 2) ? 'warning'
        : ew > 0.6 * (limits.ewma_warn ?? 2) ? 'watch' : 'normal' };
  });
  const top = comps.filter((c) => isNum(c.ewma)).sort((a, b) => b.ewma - a.ewma)[0];
  const st = r.state[i];
  const power = r.power[i];
  const q = r.quality?.[i];
  return {
    id: r.asset, label: null, t: r.t[i], source: 'replay', state: st, health: r.health?.[i] ?? null,
    power, expected: r.expected[i], expected_lo: r.expected_lo?.[i] ?? null, expected_hi: r.expected_hi?.[i] ?? null,
    wind: r.wind?.[i] ?? null, wind_dir: r.wind_dir?.[i] ?? null, nacelle_dir: r.nacelle_dir?.[i] ?? r.wind_dir?.[i] ?? null,
    yaw_error: null, rotor_rpm: r.rotor_rpm?.[i] ?? (isNum(power) && power > 0.005 ? 9 + 7 * Math.min(1, power * 1.4) : 0),
    pitch: r.pitch?.[i] ?? null, ambient: r.ambient?.[i] ?? null,
    top_component: top && top.ewma > 0.6 * (limits.ewma_warn ?? 2) && st !== 'offline' ? top.name : null,
    components: comps, quality: { state: qState(q), issues: qIssues(q) }, injected: [],
    _rpmEstimated: !r.rotor_rpm,
  };
}

/** FleetState for the current mode. In replay only the replayed asset has data; others are 'nodata'. */
export function fleet() {
  if (state.mode !== 'replay') return state.fleet;
  const r = state.replay; if (!r) return null;
  const i = Math.min(state.cursor, r.t.length - 1);
  const limits = r.limits || {};
  const snap = replaySnapshot(r, i, limits);
  const assets = state.meta?.assets || [{ id: r.asset }];
  const turbines = assets.map((a) => (String(a.id) === String(r.asset)
    ? { ...snap, label: a.label }
    : { id: a.id, label: a.label, t: r.t[i], source: 'replay', state: 'nodata', nodata: true, health: null, power: null,
      expected: null, components: [], quality: { state: 'nodata', issues: ['not part of this replay dataset'] }, injected: [] }));
  const P = r._prep, t = P.tms[i];
  const onsets24 = P.onsets.filter((k) => k <= i && P.tms[k] > t - 24 * 3.6e6).length;
  return {
    t: r.t[i], source: 'replay', weather: null, turbines,
    farm: { power: snap.power, expected: snap.expected, energy_actual: P.eAct[i], energy_expected: P.eExp[i], lost_energy: P.eLost[i] },
    open_cases: cases().filter((c) => c.status !== 'closed').length, alarms_24h: onsets24,
    _replayAssets: 1,
  };
}

// ---------------------------------------------------------------- cases (+ local analyst overrides)
const CASE_KEY = 'windtwin.caseLocal';
export function caseLocal(c) { return (lsGet(CASE_KEY, {}) || {})[`${c.source}:${c.id}`] || {}; }
export function setCaseLocal(c, patch) {
  const all = lsGet(CASE_KEY, {}) || {}; const k = `${c.source}:${c.id}`;
  all[k] = { ...(all[k] || {}), ...patch, at: new Date().toISOString() }; lsSet(CASE_KEY, all);
}

/** Cases visible at the current time, with local acknowledge/feedback merged in. */
export function cases() {
  let list;
  if (state.mode === 'replay') {
    const r = state.replay; if (!r) return [];
    const t = now();
    list = (r.cases || []).filter((c) => ms(c.alarm_time) <= t).map((c) => {
      const cleared = c.cleared_time && ms(c.cleared_time) <= t;
      return { ...c, source: c.source || 'replay', cleared_time: cleared ? c.cleared_time : null,
        status: cleared ? 'closed' : c.status === 'closed' ? 'open' : c.status };
    });
  } else list = (state.simCases || []).map((c) => ({ ...c }));
  for (const c of list) {
    const loc = caseLocal(c);
    c.feedback = loc.feedback || null;
    if (loc.status && c.status !== 'closed') c.status = loc.status;
  }
  const rank = { open: 0, acknowledged: 1, closed: 2 };
  return list.sort((a, b) => (rank[a.status] - rank[b.status]) || (ms(b.alarm_time) - ms(a.alarm_time)));
}

// ---------------------------------------------------------------- per-turbine series
let simCache = { key: '', p: null };

/** Normalised series for charts: {id, t(ms[]), power, signals, components, state, quality, health, limits, end, xMin, xMax, event, firstAlarm}. */
export async function series(id) {
  if (state.mode === 'replay') {
    const r = state.replay; if (!r || String(r.asset) !== String(id)) return null;
    const P = r._prep, end = Math.min(state.cursor, r.t.length - 1);
    const sig = { ...(r.signals || {}) };
    return {
      id: r.asset, source: 'replay', t: P.tms, end, xMin: P.tms[0], xMax: P.tms[P.tms.length - 1],
      power: { label: 'Active power', unit: 'pu', observed: r.power, expected: r.expected, lo: r.expected_lo, hi: r.expected_hi },
      signals: sig, components: r.components || {}, state: r.state, quality: (r.quality || []).map(qState), health: r.health || null,
      limits: r.limits || {}, event: r.event_start ? { start: ms(r.event_start), end: ms(r.event_end) } : null,
      firstAlarm: P.firstAlarm, wind: r.wind,
    };
  }
  const key = `${id}|${state.fleet?.t}`;
  if (simCache.key !== key) { simCache = { key, p: api.simTurbine(id, 48) }; }
  let s;
  try { s = await simCache.p; } catch (e) { simCache = { key: '', p: null }; throw e; }
  const tms = s.t.map(ms);
  const sig = { ...(s.signals || {}) };
  const power = sig.power || null; delete sig.power;
  return {
    id: s.id, source: 'simulation', t: tms, end: tms.length - 1, xMin: tms[0], xMax: tms[tms.length - 1],
    power, signals: sig, components: s.components || {}, state: s.state || [], quality: (s.quality || []).map(qState),
    health: s.health || null, limits: s.limits || {}, event: null, firstAlarm: null, wind: s.wind || null,
  };
}

export function dataSourceLabel() {
  const d = state.meta?.dataset;
  if (state.mode === 'replay') {
    const id = state.replay?.id ?? state.replayId ?? '?';
    return `real SCADA (${d?.name?.split(' - ')[0] || 'CARE to Compare'} v${d?.version ?? '?'}, dataset ${id})`;
  }
  return 'synthetic data';
}
