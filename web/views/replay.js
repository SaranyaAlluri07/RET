// Historical replay: choose a CARE dataset, scrub a timeline with the logged event window and the detector's
// first alarm, read the lead time (with the Farm A logbook caveat), and play the replay client-side.
import { state } from '../state.js';
import { loadDatasets, loadReplay, setMode, setCursor, play } from '../actions.js';
import { renderTable } from '../components/table.js';
import { mount, tok, stripOption, markers, GRID, tooltipBase } from '../components/charts.js';
import { stateBadge } from '../components/badges.js';
import { icon } from '../components/icons.js';
import { esc, fmtTime, fmtHours, isNum, ms, turbineLabel, DASH } from '../util.js';

let root = null;

async function choose(id) {
  if (state.mode !== 'replay') { state.replayId = id; state.replay = null; await setMode('replay'); }
  else await loadReplay(id);
}

function datasetsTable(ds) {
  const curId = state.mode === 'replay' ? state.replay?.id : null;
  renderTable(root.querySelector('#r-datasets'), {
    rows: ds, rowKey: (d) => d.id, selected: curId, onRowClick: (d) => choose(d.id), dense: true,
    columns: [
      { key: 'id', label: 'ID', align: 'num' },
      { key: 'asset', label: 'Asset', render: (d) => esc(turbineLabel(d.asset, state.meta)) },
      { key: 'label', label: 'Label', render: (d) => (d.label === 'anomaly' ? `<span class="badge st-warning">${icon('st_warning')}Anomaly</span>` : `<span class="badge st-normal">${icon('st_normal')}Normal</span>`) },
      { key: 'description', label: 'Description', render: (d) => esc(d.description || DASH) },
      { key: 'event_start', label: 'Logged event / window', render: (d) => `<span class="tnum">${fmtTime(d.event_start)}</span><span class="sub">→ ${fmtTime(d.event_end)}</span>` },
      { key: 'n_rows', label: 'Rows', align: 'num', render: (d) => (isNum(d.n_rows) ? d.n_rows.toLocaleString() : DASH) },
    ],
  });
}

function leadTime(r) {
  const fa = r._prep.firstAlarm, ev = ms(r.event_start);
  if (r.label === 'normal') return { text: 'n/a (normal dataset)', note: r._prep.onsets.length ? `${r._prep.onsets.length} alarm onset(s) in a normal dataset count as false alarms.` : 'No alarms raised in this normal dataset.' };
  if (!isNum(fa)) return { text: 'Not detected', note: 'The detector raised no warning/fault in this replay window.' };
  const h = (ev - fa) / 3.6e6;
  return { text: h >= 0 ? `${fmtHours(h)} before logged start` : `${fmtHours(-h)} after logged start`, note: '' };
}

function overviewOption(r) {
  const t = tok();
  const P = r._prep;
  const s = { t: P.tms, end: r.t.length - 1, xMin: P.tms[0], xMax: P.tms[P.tms.length - 1] };
  const cur = P.tms[state.cursor];
  const mk = markers(t, { cursor: cur, event: { start: ms(r.event_start), end: ms(r.event_end) }, firstAlarm: P.firstAlarm, showLabels: true });
  const health = (r.health || []).map((v, i) => [P.tms[i], v]);
  const strip = stripOption(s, r.state, { kind: 'state' });
  return {
    animation: false, textStyle: { fontFamily: t.font },
    grid: [{ ...GRID, top: 18, height: 110 }, { ...GRID, top: 134, height: 22 }],
    xAxis: [
      { type: 'time', gridIndex: 0, min: s.xMin, max: s.xMax, axisLabel: { show: false }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, splitLine: { show: false } },
      { type: 'time', gridIndex: 1, min: s.xMin, max: s.xMax, axisLabel: { color: t.muted, fontSize: 10, hideOverlap: true }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { show: false } },
    ],
    yAxis: [
      { type: 'value', gridIndex: 0, min: 0, max: 100, splitNumber: 2, axisLabel: { color: t.muted, fontSize: 10 }, splitLine: { lineStyle: { color: t.grid } } },
      { type: 'value', gridIndex: 1, min: 0, max: 1, show: false },
    ],
    tooltip: { ...tooltipBase(t), trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: t.muted } },
      formatter: (ps) => { const p = (Array.isArray(ps) ? ps : [ps]).find((x) => x.seriesName === 'Health'); if (!p) return '';
        const i = p.dataIndex; return `<div style="color:${t.muted};font-size:11px">${fmtTime(P.tms[i])}</div>Health <b>${p.value[1] ?? DASH}</b> · ${esc(r.state[i])}<br><span style="color:${t.muted}">click to move the cursor</span>`; } },
    series: [
      { name: 'Health', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: health, symbol: 'none', lineStyle: { color: t.observed, width: 1.6 }, ...mk },
      { ...strip.series[0], xAxisIndex: 1, yAxisIndex: 1, tooltip: { show: false } },
    ],
  };
}

function renderReplay() {
  const box = root.querySelector('#r-detail');
  if (state.mode !== 'replay') {
    box.innerHTML = `<div class="notice">You are in <b>Simulation</b> mode. Select a dataset above (or <button class="btn sm" id="r-switch">${icon('replay')}switch to Replay</button>) to replay real SCADA from CARE to Compare. Replay and simulation data are never mixed.</div>`;
    box.querySelector('#r-switch').addEventListener('click', () => setMode('replay'));
    return;
  }
  const r = state.replay; if (!r) { box.innerHTML = '<div class="loading">Loading replay…</div>'; return; }
  const lt = leadTime(r);
  const i = state.cursor, P = r._prep;
  box.innerHTML = `
    <section class="panel">
      <div class="panel-h"><div class="row"><h2>Dataset ${esc(r.id)} · ${esc(turbineLabel(r.asset, state.meta))}</h2>
        ${r.label === 'anomaly' ? `<span class="badge st-warning">${icon('st_warning')}Anomaly</span>` : `<span class="badge st-normal">${icon('st_normal')}Normal</span>`}
        <span class="text-2">${esc(r.description || '')}</span></div>
        <div class="row"><button class="btn sm" id="r-play">${icon('play')}Play</button><a class="btn sm primary" href="#/turbine/${encodeURIComponent(r.asset)}">Investigate at cursor ${icon('arrowRight')}</a></div></div>
      <div class="panel-b">
        <div class="grid g-4" style="margin-bottom:12px">
          <div class="fact"><small>Logged event</small><b class="small">${fmtTime(r.event_start)} → ${fmtTime(r.event_end)}</b></div>
          <div class="fact"><small>Detector first alarm</small><b class="small">${isNum(P.firstAlarm) ? fmtTime(P.firstAlarm) : 'none'}</b></div>
          <div class="fact"><small>Lead time vs logged start</small><b>${esc(lt.text)}</b></div>
          <div class="fact"><small>Cursor</small><b class="small tnum">${fmtTime(P.tms[i])} ${stateBadge(r.state[i])}</b></div>
        </div>
        <div class="chart-title"><span>Health index and alarm state over the replay window</span><span class="u">click the chart or drag the scrubber to move the cursor</span></div>
        <div class="chart" id="r-overview" style="height:184px"></div>
        <input type="range" id="r-scrub" min="0" max="${r.t.length - 1}" value="${i}" aria-label="Replay cursor" class="mt-2">
        <div class="row between xsmall muted"><span>${fmtTime(r.t[0])}</span><span>row ${i + 1} / ${r.t.length} · step ${fmtHours(P.stepH)}</span><span>${fmtTime(r.t[r.t.length - 1])}</span></div>
        <div class="caveat mt-3">${icon('alert')}<span><b>Lead-time caveat.</b> Wind Farm A logbook times are <i>fault start stamps</i>, not anomaly onsets; the true onset may be earlier or later. Lead time is measured to that stamp and is not a guarantee of warning time. ${esc(lt.note)}</span></div>
        <div class="xsmall muted mt-2">Pipeline ${esc(r.pipeline_version || DASH)} · precomputed replay file · ${state.mock ? 'MOCK fixture standing in for real SCADA - not real data' : 'data shown is real SCADA'} (per-unit power).</div>
      </div>
    </section>`;
  const c = mount(box.querySelector('#r-overview'));
  c.setOption(overviewOption(r));
  c.getZr().on('click', (e) => {
    const p = c.convertFromPixel({ gridIndex: 0 }, [e.offsetX, e.offsetY]);
    if (p && isNum(p[0])) { let k = 0; while (k < P.tms.length - 1 && P.tms[k + 1] <= p[0]) k++; setCursor(k); }
  });
  box.querySelector('#r-scrub').addEventListener('input', (e) => setCursor(Number(e.target.value)));
  box.querySelector('#r-play').addEventListener('click', play);
}

function updateCursor() {
  const r = state.replay; if (!r || !root || state.mode !== 'replay') return;
  const el = root.querySelector('#r-overview'); if (!el) return renderReplay();
  const scrub = root.querySelector('#r-scrub'); if (scrub && Number(scrub.value) !== state.cursor) scrub.value = state.cursor;
  const c = window.echarts.getInstanceByDom(el); if (c) c.setOption(overviewOption(r));
  const facts = root.querySelectorAll('#r-detail .fact b');
  if (facts[3]) facts[3].innerHTML = `${fmtTime(r._prep.tms[state.cursor])} ${stateBadge(r.state[state.cursor])}`;
}

export default {
  async mount(el) {
    root = el;
    el.innerHTML = `
      <div class="view-head"><div><h1>Historical replay</h1><div class="sub">Real SCADA from CARE to Compare (Wind Farm A), replayed through the precomputed detector output</div></div></div>
      <div id="r-detail"></div>
      <section class="panel mt-4"><div class="panel-h"><h2>CARE datasets</h2><span class="hint">click a row to load its replay</span></div>
        <div class="panel-b flush" id="r-datasets" style="max-height:420px;overflow:auto"><div class="loading">Loading datasets…</div></div></section>`;
    renderReplay();
    try { const ds = await loadDatasets(); if (root === el) datasetsTable(ds); }
    catch (e) { el.querySelector('#r-datasets').innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
  },
  update(reason) { if (reason === 'data') updateCursor(); },
  unmount() { root = null; },
};
