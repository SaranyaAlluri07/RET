import { state } from '../state.js';
import { fleet, cases, now } from '../source.js';
import { selectTurbine, launchGuidedScenario, simReset } from '../actions.js';
import { createFarm } from '../components/farm3d.js';
import { renderTable } from '../components/table.js';
import { stateBadge, qualityBadge, healthBar, STATE_RANK } from '../components/badges.js';
import { openCase, hypothesisText } from '../components/casedetail.js';
import { icon } from '../components/icons.js';
import { esc, num, pctPu, isNum, compLabel, fmtTime, fmtHours, hoursBetween, DASH, signed } from '../util.js';

let farm = null, root = null;

function kpi(label, value, unit, sub, title = '') {
  return `<div class="kpi" title="${esc(title)}"><div class="k-label">${label}</div><div class="k-value tnum">${value}${unit ? `<small>${unit}</small>` : ''}</div><div class="k-sub">${sub}</div></div>`;
}

function renderKpis(f) {
  const rated = state.rated;
  const live = f.turbines.filter((t) => !t.nodata);
  const N = live.length || 1;
  const fp = f.farm?.power, fe = f.farm?.expected;
  const ea = f.farm?.energy_actual, ee = f.farm?.energy_expected, lost = f.farm?.lost_energy;
  const cs = cases().filter((c) => c.status !== 'closed');
  const unk = cs.filter((c) => c.state === 'unknown').length;
  const scope = f._replayAssets ? `${f._replayAssets} of ${f.turbines.length} turbines in this replay` : `${N} turbines`;
  const mw = (pu) => (isNum(pu) ? (pu * rated).toFixed(2) : DASH);
  const mwh = (puh) => (isNum(puh) ? (puh * rated).toFixed(1) : DASH);
  const since = state.mode === 'replay' ? 'since replay window start' : 'since simulation start';
  return `<div class="kpis">
    ${kpi('Farm power', isNum(fp) ? ((fp / N) * 100).toFixed(1) : DASH, '% rated', `<span class="illus">≈ ${mw(fp)} MW illustrative</span> · ${scope}`, 'Sum of per-unit power over turbines with data, as % of their combined rating')}
    ${kpi('Expected power', isNum(fe) ? ((fe / N) * 100).toFixed(1) : DASH, '% rated',
      `${isNum(fp) && isNum(fe) ? `<span class="${fp < fe - 0.02 * N ? 'delta-bad' : ''}">${signed(((fp - fe) / N) * 100, 1)} pp vs expected</span>` : DASH}`, 'Normal-behaviour model expectation for current wind')}
    ${kpi('Energy actual / expected', `${mwh(ea)} / ${mwh(ee)}`, 'MWh', `<span class="illus">illustrative</span> · ${num(ea, 1)} / ${num(ee, 1)} pu·h ${since}`)}
    ${kpi('Lost energy', mwh(lost), 'MWh', `<span class="illus">illustrative</span> · ${isNum(lost) && ee ? ((lost / ee) * 100).toFixed(1) + ' % of expected' : DASH}`, 'Expected minus actual where positive. Not all loss is a fault: curtailment and wind effects also reduce output.')}
    ${kpi('Open cases', String(cs.length), '', `${unk} with unknown diagnosis`)}
    ${kpi('Alarms · 24 h', isNum(f.alarms_24h) ? String(f.alarms_24h) : DASH, '', `warning/fault onsets, ${state.mode === 'replay' ? 'replayed' : 'simulated'} time`)}
  </div>`;
}

function fleetColumns() {
  const openBy = new Map(); for (const c of cases()) if (c.status !== 'closed' && !openBy.has(String(c.turbine))) openBy.set(String(c.turbine), c);
  return [
    { key: 'label', label: 'Turbine', render: (t) => `<b>${esc(t.label)}</b>`, sort: (t) => t.label },
    { key: 'state', label: 'State', render: (t) => stateBadge(t.nodata ? 'nodata' : t.state), sort: (t) => -(STATE_RANK[t.nodata ? 'nodata' : t.state] ?? 0) },
    { key: 'health', label: 'Health', render: (t) => healthBar(t.health), sort: (t) => t.health },
    { key: 'power', label: 'Power', align: 'num', title: 'Observed power, % of rated', render: (t) => (t.nodata ? DASH : `${pctPu(t.power)} %`) },
    { key: 'expected', label: 'Expected', align: 'num', title: 'Expected power from the normal-behaviour model, with band', render: (t) => (t.nodata ? DASH
      : `${pctPu(t.expected)} %<span class="sub">${isNum(t.expected_lo) ? `${pctPu(t.expected_lo, 0)}–${pctPu(t.expected_hi, 0)} %` : ''}</span>`) },
    { key: 'delta', label: 'Δ', align: 'num', title: 'Observed minus expected, percentage points', sort: (t) => (isNum(t.power) && isNum(t.expected) ? t.power - t.expected : null),
      render: (t) => { if (!isNum(t.power) || !isNum(t.expected)) return DASH; const d = (t.power - t.expected) * 100;
        const out = isNum(t.expected_lo) && (t.power < t.expected_lo || t.power > t.expected_hi);
        return `<span class="${out ? 'delta-bad' : ''}" title="${out ? 'outside expected band' : 'within expected band'}">${signed(d, 1)} pp${out ? ' ' + icon('alert') : ''}</span>`; } },
    { key: 'wind', label: 'Wind', align: 'num', render: (t) => (isNum(t.wind) ? `${t.wind.toFixed(1)} m/s` : DASH) },
    { key: 'top_component', label: 'Top component', render: (t) => (t.top_component ? esc(compLabel(t.top_component)) : `<span class="muted">${DASH}</span>`) },
    { key: 'quality', label: 'Data quality', render: (t) => qualityBadge(t.quality), sort: (t) => t.quality?.state },
    { key: 'case', label: 'Case', sortable: false, render: (t) => { const c = openBy.get(String(t.id)); return c ? `<a href="#/cases" data-case="${esc(c.id)}">${esc(c.id)}</a>` : `<span class="muted">${DASH}</span>`; } },
    { key: 'go', label: '', sortable: false, render: (t) => (t.nodata ? '' : `<a class="btn sm ghost" href="#/turbine/${encodeURIComponent(t.id)}" title="Open investigation">${icon('arrowRight')}</a>`) },
  ];
}

function renderCases(el) {
  const list = cases().filter((c) => c.status !== 'closed');
  if (!list.length) { el.innerHTML = '<div class="empty">No open cases at this time.</div>'; return; }
  const t = now();
  el.innerHTML = `<ul class="case-list">${list.map((c) => `<li data-id="${esc(c.id)}" tabindex="0">
    ${stateBadge(c.state)}<span class="c-title ellipsis" title="${esc(hypothesisText(c))}">${esc(hypothesisText(c))}</span><span class="badge outline">${esc(c.id)}</span>
    <span class="c-meta">${esc(state.meta?.assets?.find((a) => String(a.id) === String(c.turbine))?.label ?? c.turbine)} · alarm ${fmtTime(c.alarm_time)} · ${fmtHours(hoursBetween(c.deviation_start, t))} since deviation · ${esc(c.status)}</span></li>`).join('')}</ul>`;
  el.querySelectorAll('li').forEach((li) => {
    const open = () => openCase(list.find((c) => c.id === li.dataset.id), { onChange: () => render() });
    li.addEventListener('click', open); li.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(); });
  });
}

function render() {
  if (!root) return;
  const f = fleet();
  if (!f) { root.querySelector('#kpis').innerHTML = '<div class="notice">No data loaded for this mode yet.</div>'; return; }
  root.querySelector('#kpis').innerHTML = renderKpis(f);
  const sel = f.turbines.find((t) => String(t.id) === String(state.selected));
  root.querySelector('#farm-sel').innerHTML = sel && !sel.nodata
    ? `${stateBadge(sel.state)} <b>${esc(sel.label)}</b> <a class="btn sm" href="#/turbine/${encodeURIComponent(sel.id)}">Investigate ${icon('arrowRight')}</a>` : '<span class="hint">Click a turbine to select</span>';
  const live = f.turbines.filter((t) => !t.nodata);
  const windDir = f.weather?.wind_dir ?? live.find((t) => isNum(t.wind_dir))?.wind_dir ?? null;
  const ws = live.filter((t) => isNum(t.wind)); const windSpeed = ws.length ? ws.reduce((a, t) => a + t.wind, 0) / ws.length : null;
  farm?.update(f.turbines, { selected: state.selected, windDir, windSpeed });
  renderTable(root.querySelector('#fleet-table'), {
    columns: fleetColumns(), rows: f.turbines, selected: state.selected, onRowClick: (t) => selectTurbine(t.id),
    rowClass: (t) => (t.nodata ? 'dim' : ''),
  });
  root.querySelectorAll('#fleet-table [data-case]').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault(); const c = cases().find((x) => x.id === a.dataset.case); if (c) openCase(c, { onChange: render });
  }));
  renderCases(root.querySelector('#open-cases'));
  root.querySelector('#fleet-note').textContent = state.mode === 'replay'
    ? 'Replay datasets cover one asset each; other turbines show "No data". Times are replayed SCADA timestamps.'
    : 'Synthetic turbines driven by models trained on real data. Ground truth of injected faults is only shown in Sim lab › Test oracle.';

  renderGuidedPanel();
}

function renderGuidedPanel() {
  const guidedEl = root.querySelector('#guided-panel');
  if (!guidedEl) return;
  if (state.mode !== 'simulation') {
    guidedEl.innerHTML = '';
    return;
  }
  const activeInj = state.injections?.find((i) => i.active);
  guidedEl.innerHTML = `
    <div class="panel guided-card mt-3">
      <div class="panel-h">
        <div>
          <h2 style="display:flex;align-items:center;gap:8px">
            ${icon('sim')} Guided Simulation Mode
            <span class="badge outline" style="font-size:11px;font-weight:normal;color:var(--mode-sim)">One-Click Demos</span>
          </h2>
          <p class="xsmall muted" style="margin:2px 0 0">Run reproducible fault scenarios with automated injection, detector response, and energy tracking.</p>
        </div>
        <div class="row" style="margin-left:auto;gap:8px">
          <button class="btn sm" id="guided-reset" title="Reset simulation to fresh baseline">${icon('reset')} Reset</button>
          <a class="btn sm ghost" href="#/sim">Sim Lab ${icon('arrowRight')}</a>
        </div>
      </div>
      <div class="panel-b" style="display:flex;flex-wrap:wrap;gap:10px;align-items:center">
        <button class="btn sm primary-tinted" data-guided="gearbox" title="Turbine T-10: Gearbox bearing thermal excursion">
          ${icon('wrench')} <b>Gearbox Overheating</b> <small style="opacity:0.8">(T-10 · 6h ramp)</small>
        </button>
        <button class="btn sm primary-tinted" data-guided="hydraulic" title="Turbine T-00: Hydraulic oil cooling loss">
          ${icon('alert')} <b>Hydraulic Fault</b> <small style="opacity:0.8">(T-00 · severe)</small>
        </button>
        <button class="btn sm primary-tinted" data-guided="sensor" title="Turbine T-21: Stuck telemetry sensor">
          ${icon('st_unknown')} <b>Sensor Anomaly</b> <small style="opacity:0.8">(T-21 · stuck)</small>
        </button>
        ${activeInj ? `
          <div class="inj-status-chip ml-auto">
            <span class="dot pulse" style="background:var(--st-warning)"></span>
            <span>Active: <b>${esc(activeInj.name)}</b> on T-${esc(String(activeInj.turbine).padStart(2, '0'))}</span>
            <a class="btn xs" href="#/turbine/${encodeURIComponent(activeInj.turbine)}">Investigate ${icon('arrowRight')}</a>
          </div>` : ''}
      </div>
    </div>`;

  guidedEl.querySelector('#guided-reset')?.addEventListener('click', async () => {
    await simReset();
    render();
  });
  guidedEl.querySelectorAll('[data-guided]').forEach((b) => b.addEventListener('click', async () => {
    const type = b.dataset.guided;
    if (type === 'gearbox') {
      await launchGuidedScenario({ scenario: 'gearbox_bearing', turbine: 10, severity: 'moderate', ramp_h: 6, duration_h: 48 });
    } else if (type === 'hydraulic') {
      await launchGuidedScenario({ scenario: 'hydraulic_overheating', turbine: 0, severity: 'severe', ramp_h: 2, duration_h: 48 });
    } else if (type === 'sensor') {
      await launchGuidedScenario({ scenario: 'sensor_stuck', turbine: 21, severity: 'moderate', ramp_h: 0, duration_h: 48 });
    }
    render();
  }));
}

export default {
  mount(el) {
    root = el;
    el.innerHTML = `
      <div class="view-head"><div><h1>Fleet overview</h1><div class="sub">${esc(state.meta?.dataset?.name || '')} · ${state.meta?.assets?.length ?? 0} turbines</div></div></div>
      <div id="kpis"></div>
      <div id="guided-panel"></div>
      <div class="grid g-main-side mt-4">
        <section class="panel"><div class="panel-h"><h2>Farm</h2><div class="row" id="farm-sel"></div></div>
          <div class="farm-wrap" id="farm3d"></div></section>
        <section class="panel"><div class="panel-h"><h2>Open cases</h2><a class="hint" href="#/cases">All cases ${icon('arrowRight')}</a></div>
          <div class="panel-b flush" id="open-cases" style="overflow:auto;max-height:430px"></div></section>
      </div>
      <section class="panel mt-4"><div class="panel-h"><h2>Turbines</h2><span class="hint" id="fleet-note"></span></div>
        <div class="panel-b flush" id="fleet-table"></div></section>`;
    farm = createFarm(el.querySelector('#farm3d'), { assets: state.meta?.assets || [], onSelect: (id) => selectTurbine(id) });
    render();
  },
  update(reason) { if (['data', 'select', 'rated', 'state'].includes(reason)) render(); },
  unmount() { farm?.dispose(); farm = null; root = null; },
};
