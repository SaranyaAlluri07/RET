import { state } from '../state.js';
import { api } from '../api.js';
import { setMode, simInject, simRepair, simReset, launchGuidedScenario } from '../actions.js';
import { renderTable } from '../components/table.js';
import { stateBadge, yesNo } from '../components/badges.js';
import { icon } from '../components/icons.js';
import { esc, num, isNum, compLabel, fmtTime, fmtHours, turbineLabel, toast, DASH, frac } from '../util.js';

let root = null, scenarios = null, selScenario = null;
const CATS = [
  ['component', 'Component faults', 'st_fault'],
  ['sensor', 'Sensor / data faults', 'st_unknown'],
  ['performance', 'Performance faults', 'eval'],
];

function library() {
  return CATS.map(([cat, title, ic]) => {
    const list = scenarios.filter((s) => s.category === cat); if (!list.length) return '';
    return `<div class="cat-h">${icon(ic)}${title} <span class="muted">(${list.length})</span></div><div class="scen-grid">${list.map((s) => `
      <button class="scen ${s.id === selScenario ? 'on' : ''}" data-scen="${esc(s.id)}"><b>${esc(s.name)}</b><p>${esc(s.description || '')}</p>
        <span class="meta">${s.component ? 'component: ' + esc(compLabel(s.component)) : 'no component - expect "unknown"'}</span></button>`).join('')}</div>`;
  }).join('');
}

function form() {
  const sc = scenarios.find((s) => s.id === selScenario) || scenarios[0];
  const sev = sc?.params?.severity?.length ? sc.params.severity : ['mild', 'moderate', 'severe'];
  const ramps = sc?.params?.ramp_h?.length ? sc.params.ramp_h : [0, 6, 24];
  const ts = state.fleet?.turbines || [];
  return `
    <div class="guided-card panel-b mb-3" style="border-radius:var(--radius-md);display:flex;align-items:center;flex-wrap:wrap;gap:8px">
      <span class="xsmall" style="font-weight:var(--fw-bold);color:var(--mode-sim);text-transform:uppercase;letter-spacing:.05em">${icon('sim')} 1-Click Guided:</span>
      <button class="btn sm primary-tinted" data-guided-sim="gearbox" title="Turbine T-10: Gearbox bearing thermal excursion">Gearbox Overheating (T-10)</button>
      <button class="btn sm primary-tinted" data-guided-sim="hydraulic" title="Turbine T-00: Hydraulic oil overheating">Hydraulic Fault (T-00)</button>
      <button class="btn sm primary-tinted" data-guided-sim="sensor" title="Turbine T-21: Stuck telemetry sensor">Sensor Anomaly (T-21)</button>
    </div>
    <div class="form-grid">
      <label class="field">Turbine<select id="f-turbine">${ts.map((t) => `<option value="${esc(t.id)}" ${String(t.id) === String(state.selected) ? 'selected' : ''}>${esc(t.label)}</option>`).join('')}</select></label>
      <label class="field" style="grid-column:span 2">Scenario<select id="f-scen">${scenarios.map((s) => `<option value="${esc(s.id)}" ${s.id === sc?.id ? 'selected' : ''}>${esc(s.name)} (${s.category})</option>`).join('')}</select></label>
      <label class="field">Severity<select id="f-sev">${sev.map((v, i) => `<option value="${esc(v)}" ${i === Math.min(1, sev.length - 1) ? 'selected' : ''}>${esc(v)}</option>`).join('')}</select></label>
      <label class="field">Ramp (h)<select id="f-ramp">${ramps.map((v, i) => `<option value="${v}" ${i === Math.min(1, ramps.length - 1) ? 'selected' : ''}>${v} h</option>`).join('')}</select></label>
      <label class="field">Duration (h)<input id="f-dur" type="number" min="1" max="720" step="1" value="48"></label>
      <label class="field">Seed (optional)<input id="f-seed" type="number" min="0" step="1" placeholder="random"></label>
      <div style="grid-column:span 2;display:flex;align-items:flex-end;gap:8px;justify-content:flex-end">
        <button class="btn" id="f-reset" title="Reset the simulation (uses the seed if given)">${icon('reset')}Reset simulation</button>
        <button class="btn primary" id="f-inject">${icon('sim')}Inject fault</button></div>
    </div>`;
}

function wire() {
  root.querySelectorAll('[data-scen]').forEach((b) => b.addEventListener('click', () => { selScenario = b.dataset.scen; render(); }));
  root.querySelector('#f-scen')?.addEventListener('change', (e) => { selScenario = e.target.value; render(); });
  root.querySelector('#f-inject')?.addEventListener('click', async (e) => {
    const btn = e.currentTarget; btn.disabled = true;
    const seed = root.querySelector('#f-seed').value;
    const body = { turbine: Number(root.querySelector('#f-turbine').value), scenario: root.querySelector('#f-scen').value,
      severity: root.querySelector('#f-sev').value, ramp_h: Number(root.querySelector('#f-ramp').value),
      duration_h: Number(root.querySelector('#f-dur').value) || 24, ...(seed !== '' ? { seed: Number(seed) } : {}) };
    try { const inj = await simInject(body); toast(`Injected ${inj?.name || body.scenario} into ${turbineLabel(body.turbine, state.meta)}. Press play to let it develop.`); }
    catch (err) { toast(`Inject failed: ${err.message}`, 'err'); } finally { btn.disabled = false; }
  });
  root.querySelector('#f-reset')?.addEventListener('click', async () => {
    const seed = root.querySelector('#f-seed').value;
    try { await simReset(seed !== '' ? { seed: Number(seed) } : {}); toast('Simulation reset.'); } catch (err) { toast(`Reset failed: ${err.message}`, 'err'); }
  });
  root.querySelectorAll('[data-guided-sim]').forEach((b) => b.addEventListener('click', async () => {
    const type = b.dataset.guidedSim;
    if (type === 'gearbox') {
      await launchGuidedScenario({ scenario: 'gearbox_bearing', turbine: 10, severity: 'moderate', ramp_h: 6, duration_h: 48 });
    } else if (type === 'hydraulic') {
      await launchGuidedScenario({ scenario: 'hydraulic_overheating', turbine: 0, severity: 'severe', ramp_h: 2, duration_h: 48 });
    } else if (type === 'sensor') {
      await launchGuidedScenario({ scenario: 'sensor_stuck', turbine: 21, severity: 'moderate', ramp_h: 0, duration_h: 48 });
    }
  }));
}

function injectionsTable() {
  const el = root.querySelector('#s-inj'); if (!el) return;
  renderTable(el, {
    rows: state.injections || [], rowKey: (i) => i.id, dense: true, empty: 'No injections yet',
    columns: [
      { key: 'id', label: 'ID', render: (i) => `<span class="mono nowrap">${esc(i.id)}</span>` },
      { key: 'turbine', label: 'Turbine', render: (i) => esc(turbineLabel(i.turbine, state.meta)) },
      { key: 'name', label: 'Scenario', render: (i) => `${esc(i.name || i.scenario)}<span class="sub">${esc(i.category || '')}</span>` },
      { key: 'severity', label: 'Severity', render: (i) => esc(i.severity ?? DASH) },
      { key: 'ramp_h', label: 'Ramp', align: 'num', render: (i) => `${num(i.ramp_h, 0)} h` },
      { key: 'start', label: 'Start → end', render: (i) => `<span class="tnum nowrap">${fmtTime(i.start)}</span><span class="sub nowrap">→ ${fmtTime(i.end)}</span>` },
      { key: 'active', label: 'Status', render: (i) => (i.active ? '<span class="badge case-open">Active</span>' : '<span class="badge case-closed">Ended</span>') },
      { key: 'act', label: '', sortable: false, render: (i) => (i.active ? `<button class="btn sm" data-repair="${esc(i.id)}">${icon('wrench')}Repair</button>` : '') },
    ],
  });
  el.querySelectorAll('[data-repair]').forEach((b) => b.addEventListener('click', async () => {
    b.disabled = true;
    try { await simRepair(b.dataset.repair); toast(`Repaired ${b.dataset.repair}.`); } catch (e) { toast(`Repair failed: ${e.message}`, 'err'); b.disabled = false; }
  }));
}

function oracle() {
  const inj = state.injections || [];
  const el = root.querySelector('#s-oracle'); if (!el) return;
  const detected = inj.filter((i) => i.detected_time);
  const delays = detected.map((i) => i.delay_h).filter(isNum).sort((a, b) => a - b);
  const med = delays.length ? delays[(delays.length - 1) >> 1] : null;
  const judged = inj.filter((i) => i.correct != null);
  el.querySelector('#o-sum').innerHTML = `
    <div class="fact"><small>Injections detected</small><b>${frac(detected.length, inj.length)}</b></div>
    <div class="fact"><small>Median detection delay</small><b>${fmtHours(med)}</b></div>
    <div class="fact"><small>Diagnosis correct</small><b>${frac(judged.filter((i) => i.correct).length, judged.length)}</b></div>
    <div class="fact"><small>Ground truth lost energy</small><b>${num(inj.reduce((a, i) => a + (i.lost_energy_pu_h || 0), 0), 2)} <small>pu·h</small></b></div>`;
  renderTable(el.querySelector('#o-table'), {
    rows: inj, rowKey: (i) => i.id, dense: true, empty: 'Inject a fault to see ground truth vs detection',
    columns: [
      { key: 'id', label: 'Injection', render: (i) => `<span class="mono nowrap">${esc(i.id)}</span>` },
      { key: 'turbine', label: 'Turbine', render: (i) => esc(turbineLabel(i.turbine, state.meta)) },
      { key: 'name', label: 'Truth: scenario', render: (i) => esc(i.name || i.scenario) },
      { key: 'component', label: 'Truth: component', render: (i) => (i.component ? esc(compLabel(i.component)) : '<span class="muted">none (sensor) → expect unknown</span>') },
      { key: 'detected_time', label: 'Detected', render: (i) => (i.detected_time ? `<span class="tnum">${fmtTime(i.detected_time)}</span>` : (i.active ? '<span class="muted">not yet</span>' : '<span class="delta-bad">missed</span>')) },
      { key: 'delay_h', label: 'Delay', align: 'num', render: (i) => fmtHours(i.delay_h) },
      { key: 'detected_component', label: 'Diagnosed', render: (i) => (i.detected_time ? (i.detected_component ? esc(compLabel(i.detected_component)) : stateBadge('unknown')) : DASH) },
      { key: 'correct', label: 'Correct?', render: (i) => yesNo(i.correct) },
      { key: 'lost_energy_pu_h', label: 'Lost energy', align: 'num', render: (i) => `${num(i.lost_energy_pu_h, 2)} pu·h` },
    ],
  });
}

function render() {
  if (!root) return;
  if (state.mode !== 'simulation') {
    root.querySelector('#s-body').innerHTML = `<div class="notice">The simulation lab works on <b>synthetic data</b> only. You are currently in <b>Replay</b> mode (real SCADA).
      <button class="btn sm" id="s-switch">${icon('sim')}Switch to Simulation</button></div>`;
    root.querySelector('#s-switch').addEventListener('click', () => setMode('simulation'));
    return;
  }
  if (!scenarios) { root.querySelector('#s-body').innerHTML = '<div class="loading">Loading scenarios…</div>'; return; }
  if (!selScenario) selScenario = scenarios[0]?.id;
  root.querySelector('#s-body').innerHTML = `
    <div class="grid g-2" style="align-items:start">
      <section class="panel"><div class="panel-h"><h2>Scenario library</h2><span class="hint">${scenarios.length} reproducible, seeded scenarios</span></div>
        <div class="panel-b">${library()}</div></section>
      <div class="stack">
        <section class="panel"><div class="panel-h"><h2>Inject</h2><span class="hint">at simulated time ${fmtTime(state.fleet?.t)}</span></div><div class="panel-b">${form()}</div></section>
        <section class="panel"><div class="panel-h"><h2>Injections</h2><span class="hint">repair ends an injection early</span></div><div class="panel-b flush" id="s-inj"></div></section>
      </div>
    </div>
    <section class="oracle mt-4" id="s-oracle">
      <div class="panel-h"><div class="row"><span class="oracle-tag">TEST ORACLE</span><h2>Ground truth vs detection</h2></div>
        <span class="hint">Not visible in operator views. Synthetic results are not evidence of real-data accuracy.</span></div>
      <div class="panel-b"><div class="grid g-4" id="o-sum" style="margin-bottom:12px"></div><div id="o-table"></div></div>
    </section>`;
  wire(); injectionsTable(); oracle();
}

export default {
  async mount(el) {
    root = el;
    el.innerHTML = `<div class="view-head"><div><h1>Simulation lab</h1><div class="sub">Inject reproducible synthetic faults into simulated turbines and check how the detector responds</div></div></div><div id="s-body"></div>`;
    render();
    try { scenarios = scenarios || await api.simScenarios(); } catch (e) { toast(`Could not load scenarios: ${e.message}`, 'err'); scenarios = []; }
    if (root === el) render();
  },
  update(reason) {
    if (reason !== 'data' || !root || state.mode !== 'simulation') return;
    injectionsTable(); oracle();
  },
  unmount() { root = null; },
};
