// Turbine investigation (core screen): header (state, health), evidence behind the current alert,
// and one aligned timeline: state strip, power, component score/EWMA/CUSUM with limits, per-signal
// observed vs expected with prediction band, and a data-quality strip. All charts share zoom + tooltip.
import { state } from '../state.js';
import { fleet, cases, series, now } from '../source.js';

import { mount, connect, disposeWithin, signalOption, statOption, stripOption, markers, tok } from '../components/charts.js';
import { stateBadge, qualityBadge, healthBar, meter } from '../components/badges.js';
import { openCase, evidenceTable, hypothesisText, isUnknownCase } from '../components/casedetail.js';
import { icon } from '../components/icons.js';
import { esc, num, pctPu, isNum, compLabel, fmtTime, fmtHours, hoursBetween, DASH, signed } from '../util.js';

const GROUP = 'turbine-investigation';
let root = null, token = 0, showAll = false, focus = null, lastSeriesKey = '';

function turbineId(param) {
  const f = fleet();
  const valid = (id) => f?.turbines.some((t) => String(t.id) === String(id) && !t.nodata);
  if (param != null && valid(param)) return param;
  if (state.selected != null && valid(state.selected)) return state.selected;
  return f?.turbines.find((t) => !t.nodata)?.id ?? null;
}

/** Active (non-closed) case for this turbine, most recent alarm first. */
const activeCase = (id) => cases().find((c) => String(c.turbine) === String(id) && c.status !== 'closed');

/** Current band-normalised deviation per signal (used when there is no active case). */
function currentDeviations(s) {
  const out = [];
  const zBand = s.limits?.z_band || 1.645;
  for (const [key, sig] of Object.entries(s.signals)) {
    const i = s.end; const o = sig.observed?.[i], e = sig.expected?.[i];
    if (!isNum(o) || !isNum(e)) continue;
    const half = isNum(sig.hi?.[i]) && isNum(sig.lo?.[i]) ? (sig.hi[i] - sig.lo[i]) / 2 : null;
    out.push({ signal: key, label: sig.label, observed: o, expected: e, z: half ? ((o - e) / half) * zBand : null });
  }
  const tot = out.reduce((a, r) => a + Math.max(0, r.z ?? 0) ** 2, 0) || 1;
  out.forEach((r) => { r.contribution = Math.max(0, r.z ?? 0) ** 2 / tot; });
  return out.sort((a, b) => Math.abs(b.z ?? 0) - Math.abs(a.z ?? 0));
}

function header(t, f) {
  const tabs = f.turbines.map((x) => `<button class="${String(x.id) === String(t.id) ? 'on' : ''}" data-tid="${esc(x.id)}" ${x.nodata ? 'disabled title="Not part of this replay"' : ''}>
      <span class="dot" style="background:var(--st-${x.nodata ? 'nodata' : x.state})"></span>${esc(x.label)}</button>`).join('');
  const out = isNum(t.expected_lo) && isNum(t.power) && (t.power < t.expected_lo || t.power > t.expected_hi);
  return `<div class="tabs" role="tablist" aria-label="Turbine">${tabs}</div>
    <div class="turbine-head">
      <div class="name">${esc(t.label)}</div>${stateBadge(t.state, { lg: true })}
      <div class="facts">
        <div class="fact"><small>Health</small><b>${healthBar(t.health)}</b></div>
        <div class="fact"><small>Power vs expected</small><b>${pctPu(t.power)} <small>/ ${pctPu(t.expected)} % rated</small>
          ${isNum(t.power) && isNum(t.expected) ? `<small class="${out ? 'delta-bad' : ''}">${signed((t.power - t.expected) * 100, 1)} pp${out ? ' · outside band' : ''}</small>` : ''}</b></div>
        <div class="fact"><small>Wind</small><b>${isNum(t.wind) ? t.wind.toFixed(1) : DASH} <small>m/s${isNum(t.wind_dir) ? ` from ${Math.round(t.wind_dir)}°` : ''}</small></b></div>
        <div class="fact"><small>Rotor / pitch</small><b>${isNum(t.rotor_rpm) ? t.rotor_rpm.toFixed(1) : DASH} <small>rpm${t._rpmEstimated ? ' (est.)' : ''}</small> ${isNum(t.pitch) ? `${t.pitch.toFixed(1)} <small>°</small>` : ''}</b></div>
        <div class="fact"><small>Top component</small><b>${t.top_component ? esc(compLabel(t.top_component)) : `<span class="muted">${DASH}</span>`}</b></div>
        <div class="fact"><small>Data quality</small><b>${qualityBadge(t.quality)}</b></div>
      </div>
      <div style="margin-left:auto" class="small muted">${t.source === 'replay' ? 'replayed' : 'simulated'} time<br><b class="tnum" style="color:var(--text)">${fmtTime(t.t)}</b></div>
    </div>`;
}

function evidencePanel(t, s) {
  const c = activeCase(t.id);
  const rated = state.rated || 2.0;
  const activeInj = state.mode === 'simulation'
    ? state.injections?.find((i) => String(i.turbine) === String(t.id) && i.active)
    : null;
  const injHtml = activeInj ? `
    <div class="inj-context-banner mb-2">
      ${icon('sim')}
      <div style="flex:1">
        <div><b>Injected Scenario: ${esc(activeInj.name)}</b> <span class="badge outline" style="margin-left:4px">${esc(activeInj.severity)}</span></div>
        <div class="xsmall muted" style="margin-top:2px">
          Onset: ${fmtTime(activeInj.start)} · ${activeInj.detected_time ? `Detected: ${fmtTime(activeInj.detected_time)} (${activeInj.delay_h}h delay)` : 'Evaluating detector response…'}
          · Ground truth loss: <b>${activeInj.lost_energy_pu_h} pu·h</b> (≈ ${(activeInj.lost_energy_pu_h * rated).toFixed(2)} MWh)
        </div>
      </div>
    </div>` : '';
  const replayHtml = state.mode === 'replay' && state.replay ? `
    <div class="replay-context-banner mb-2">
      ${icon('replay')}
      <div style="flex:1">
        <div><b>CARE v6 Replay · Dataset ${esc(state.replay.id)}</b> <span class="badge outline" style="margin-left:4px">${esc(state.replay.label)}</span></div>
        <div class="xsmall muted" style="margin-top:2px">
          Logged event: <b>${esc(state.replay.description || 'Normal operation')}</b> · Window: ${fmtTime(state.replay.event_start)} → ${fmtTime(state.replay.event_end)}
        </div>
      </div>
    </div>` : '';

  if (c) {
    const unknown = isUnknownCase(c);
    return `<div class="panel-h"><h2>Why this alert</h2><button class="btn sm" data-open-case="${esc(c.id)}">Case ${esc(c.id)} ${icon('arrowRight')}</button></div>
      <div class="panel-b stack" style="gap:12px">
        ${injHtml || replayHtml}
        <div><div class="row">${stateBadge(c.state)}<span class="small muted">alarm ${fmtTime(c.alarm_time)}</span></div>
          <div class="hypo mt-2 ${unknown ? 'unknown' : ''}">${esc(hypothesisText(c))}</div></div>
        <dl class="kv">
          <dt>Deviation start</dt><dd>${fmtTime(c.deviation_start)}</dd>
          <dt>Elapsed</dt><dd>${fmtHours(hoursBetween(c.deviation_start, now()))}</dd>
          <dt>Severity</dt><dd><span class="row" style="flex-wrap:nowrap">${meter(c.severity, 'var(--st-warning)')} ${num(c.severity, 2)}</span></dd>
          <dt>Confidence</dt><dd>${isNum(c.confidence) ? Math.round(c.confidence * 100) + ' %' : `${DASH} <span class="muted">not calibrated</span>`}</dd>
        </dl>
        <div><div class="section-label">Signals ranked by contribution (at alarm)</div>${evidenceTable(c.evidence)}</div>
      </div>`;
  }
  const dev = s ? currentDeviations(s) : [];
  return `<div class="panel-h"><h2>Why this state</h2><span class="hint">no active case</span></div>
    <div class="panel-b stack" style="gap:12px">
      ${injHtml || replayHtml}
      <div class="small text-2">${t.state === 'unknown' ? 'Data quality is insufficient to judge this turbine; no component hypothesis is made.'
        : t.state === 'watch' ? 'A statistic is above its watch limit but persistence is not met yet - no alarm raised.'
          : t.state === 'offline' ? 'Turbine is in a non-production state; deviations are not evaluated.' : 'All monitored statistics are within limits.'}</div>
      <div><div class="section-label">Current deviations (band-normalised z)</div>${evidenceTable(dev.slice(0, 6))}</div>
    </div>`;
}

function componentPanel(t) {
  const comps = t.components || [];
  if (!comps.length) return '';
  return `<div class="panel-h"><h2>Components</h2><span class="hint">EWMA / CUSUM now</span></div>
    <div class="panel-b flush"><table class="tbl dense"><thead><tr><th>Component</th><th>State</th><th class="num">EWMA</th><th class="num">CUSUM</th></tr></thead><tbody>
    ${[...comps].sort((a, b) => (b.ewma ?? 0) - (a.ewma ?? 0)).map((c) => `<tr class="clickable" data-comp="${esc(c.name)}"><td>${esc(compLabel(c.name))}</td><td>${stateBadge(c.state)}</td>
      <td class="num">${num(c.ewma, 2)}</td><td class="num">${num(c.cusum, 1)}</td></tr>`).join('')}</tbody></table></div>`;
}

function chartBlocks(s, sigKeys) {
  const block = (id, title, unit, cls = '') => `<div class="chart-block"><div class="chart-title"><span>${title}</span><span class="u">${unit}</span></div><div class="chart ${cls}" id="${id}"></div></div>`;
  return block('c-state', 'Alarm state', 'normal · watch · warning · fault · unknown (hatched) · offline', 'strip')
    + (s.power ? block('c-power', 'Active power — observed vs expected', '% of rated · band = prediction interval', 'sm') : '')
    + block('c-score', 'Component score (normalised residual)', '|z|', 'xs')
    + block('c-ewma', 'EWMA of score', 'limits: watch/warning · fault', 'sm')
    + block('c-cusum', 'CUSUM of score', 'decision limit h', 'xs')
    + sigKeys.map((k) => block(`c-sig-${k}`, esc(s.signals[k].label), esc(s.signals[k].unit || ''), 'xs')).join('')
    + block('c-quality', 'Data quality', 'ok · degraded · bad/missing', 'strip" style="height:76px');
}

function legendHtml(s) {
  const t = tok();
  const names = Object.keys(s.components);
  return `<div class="legend-inline"><span><i style="background:var(--chart-observed)"></i>Observed</span><span><i class="dash"></i>Expected</span><span><i class="band"></i>Prediction band</span>
    ${s.event ? '<span><i class="sq" style="background:var(--chart-event)"></i>Logged event window</span>' : ''}
    ${isNum(s.firstAlarm) ? '<span><i style="background:var(--st-warning)"></i>Detector first alarm (dashed)</span>' : ''}</div>
    <div class="chips mt-2" id="comp-chips">${names.map((n, i) => `<button class="chip ${focus === n ? 'on' : ''}" data-focus="${esc(n)}"><i class="sw" style="background:${t.series[i % 8]}"></i>${esc(compLabel(n))}</button>`).join('')}
    <span class="xsmall muted" style="align-self:center">Click to focus a component · shift+wheel or slider to zoom</span></div>`;
}

function drawCharts(s, sigKeys) {
  const t = tok();
  const curT = s.t[s.end];
  const fa = isNum(s.firstAlarm) && s.firstAlarm <= curT ? s.firstAlarm : null;
  const mk = markers(t, { cursor: s.source === 'replay' ? curT : null, event: s.event, firstAlarm: fa });
  const set = (id, opt) => { const el = root.querySelector('#' + id); if (el) mount(el, GROUP).setOption(opt, { replaceMerge: ['series'] }); };
  set('c-state', stripOption(s, s.state, { kind: 'state', mk }));
  if (s.power) set('c-power', signalOption(s, s.power, { mk }));
  const names = Object.keys(s.components);
  const lines = (stat) => names.map((n, i) => ({ name: compLabel(n), data: s.components[n][stat], color: t.series[i % 8],
    dim: focus != null && focus !== n, width: focus === n ? 2.2 : 1.5 }));
  const L = s.limits || {};
  set('c-score', statOption(s, lines('score'), [], { mk }));
  set('c-ewma', statOption(s, lines('ewma'), [
    ...(isNum(L.ewma_warn) ? [{ value: L.ewma_warn, label: 'warn', color: t.st.warning }] : []),
    ...(isNum(L.ewma_fault) ? [{ value: L.ewma_fault, label: 'fault', color: t.st.fault }] : [])], { mk }));
  set('c-cusum', statOption(s, lines('cusum'), isNum(L.cusum_h) ? [{ value: L.cusum_h, label: 'h', color: t.st.fault }] : [], { mk, fmtDigits: 1 }));
  for (const k of sigKeys) set(`c-sig-${k}`, signalOption(s, s.signals[k], { mk }));
  set('c-quality', stripOption(s, s.quality.length ? s.quality : s.t.map(() => 'ok'), { kind: 'quality', mk, showX: true, slider: true }));
  connect(GROUP);
}

/** Signals to chart: ordered by case evidence, else by current deviation; 4 unless "show all". */
function pickSignals(s, t) {
  const c = activeCase(t.id);
  const order = c?.evidence?.map((e) => e.signal).filter((k) => s.signals[k]) || [];
  for (const d of currentDeviations(s)) if (!order.includes(d.signal)) order.push(d.signal);
  for (const k of Object.keys(s.signals)) if (!order.includes(k)) order.push(k);
  return showAll ? order : order.slice(0, 4);
}

async function render(param) {
  if (!root) return;
  const my = ++token;
  const f = fleet();
  const id = turbineId(param);
  if (!f || id == null) { root.innerHTML = '<div class="notice">No turbine data available in this mode.</div>'; return; }
  if (String(state.selected) !== String(id)) state.selected = id;
  const t = f.turbines.find((x) => String(x.id) === String(id));
  let s = null, err = null;
  try { s = await series(id); } catch (e) { err = e; }
  if (my !== token || !root) return;
  if (focus == null || (s && !s.components[focus])) focus = activeCase(id)?.component ?? t.top_component ?? null;

  const sigKeys = s ? pickSignals(s, t) : [];
  const key = `${id}|${sigKeys.join(',')}|${state.mode}|${focus}`;
  root.querySelector('#t-head').innerHTML = header(t, f);
  root.querySelector('#t-evidence').innerHTML = evidencePanel(t, s);
  root.querySelector('#t-comps').innerHTML = componentPanel(t);
  wireHeader();
  const charts = root.querySelector('#t-charts');
  if (!s) { disposeWithin(charts); lastSeriesKey = ''; charts.innerHTML = `<div class="error-box">Series unavailable${err ? ': ' + esc(err.message) : ''}</div>`; return; }
  if (key !== lastSeriesKey) {         // layout changed: rebuild chart containers
    disposeWithin(charts);
    charts.innerHTML = `<div class="panel-h"><h2>Timeline</h2><div class="row"><span class="hint">${s.source === 'replay' ? 'data up to the replay cursor' : 'last 48 h'} · ${s.t.length} rows</span>
      <button class="btn sm" id="t-all">${showAll ? 'Top 4 signals' : `All ${Object.keys(s.signals).length} signals`}</button></div></div>
      <div class="panel-b">${legendHtml(s)}<div class="mt-3">${chartBlocks(s, sigKeys)}</div></div>`;
    charts.querySelector('#t-all').addEventListener('click', () => { showAll = !showAll; render(param); });
    charts.querySelectorAll('[data-focus]').forEach((b) => b.addEventListener('click', () => { focus = focus === b.dataset.focus ? null : b.dataset.focus; render(param); }));
    lastSeriesKey = key;
  }
  drawCharts(s, sigKeys);
}

function wireHeader() {
  root.querySelectorAll('[data-tid]').forEach((b) => b.addEventListener('click', () => {
    focus = null; state.selected = b.dataset.tid; location.hash = `#/turbine/${encodeURIComponent(b.dataset.tid)}`;
  }));
  root.querySelector('[data-open-case]')?.addEventListener('click', (e) => {
    const c = cases().find((x) => x.id === e.currentTarget.dataset.openCase); if (c) openCase(c, { onChange: () => render(currentParam) });
  });
  root.querySelectorAll('[data-comp]').forEach((tr) => tr.addEventListener('click', () => { focus = tr.dataset.comp; render(currentParam); }));
}

let currentParam = null;
export default {
  mount(el, { param }) {
    root = el; currentParam = param; lastSeriesKey = ''; focus = null;
    el.innerHTML = `
      <div class="view-head"><div><h1>Turbine investigation</h1><div class="sub">Evidence, expected-vs-observed signals and alarm statistics on one aligned timeline</div></div></div>
      <section class="panel" id="t-head"></section>
      <div class="grid g-side-main mt-4" style="align-items:start">
        <div class="stack"><section class="panel" id="t-evidence"></section><section class="panel" id="t-comps"></section></div>
        <section class="panel" id="t-charts"><div class="loading">Loading series…</div></section>
      </div>`;
    render(param);
  },
  update(reason) {
    if (reason === 'select') { if (String(state.selected) !== String(currentParam)) { currentParam = state.selected; lastSeriesKey = ''; } render(currentParam); }
    else if (reason === 'data' || reason === 'state') render(currentParam);
  },
  unmount() { token++; root = null; },
};
