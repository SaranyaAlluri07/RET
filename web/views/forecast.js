// Forecast: ML vs physics metrics, power curve with prediction band, interval coverage by regime, audit checklist.
import { api } from '../api.js';
import { mount, powerCurveOption, coverageOption } from '../components/charts.js';
import { icon } from '../components/icons.js';
import { esc, num, isNum, DASH } from '../util.js';

let root = null;

function metric(label, ml, ph, fmt, better) {
  const win = isNum(ml) && isNum(ph) ? (better === 'low' ? ml < ph : ml > ph) : null;
  return `<div class="kpi"><div class="k-label">${label}</div><div class="metric-pair"><span class="v">${fmt(ml)}</span><span class="vs">physics ${fmt(ph)}</span></div>
    <div class="k-sub">${win == null ? '' : win ? '<span class="delta-good">ML better</span>' : '<span class="delta-bad">physics better</span>'}</div></div>`;
}

function render(f) {
  const ml = f.metrics?.ml || {}, ph = f.metrics?.physics || {};
  const pc = (v) => (isNum(v) ? `${(v * 100).toFixed(2)} %` : DASH);
  const nominal = f.intervals?.nominal ?? 0.8;
  const reg = f.intervals?.coverage_by_regime || [];
  const checks = f.audit?.leakage_checks || [];
  root.innerHTML = `
    <div class="view-head"><div><h1>Power forecast</h1><div class="sub">Expected power is the basis of the performance evidence. Evaluated separately from fault detection on held-out data.</div></div></div>
    <div class="kpis" style="grid-template-columns:repeat(4,minmax(0,1fr))">
      ${metric('MAE · % of rated', ml.mae, ph.mae, pc, 'low')}
      ${metric('RMSE · % of rated', ml.rmse, ph.rmse, pc, 'low')}
      ${metric('R²', ml.r2, ph.r2, (v) => num(v, 4), 'high')}
      <div class="kpi"><div class="k-label">Test rows</div><div class="k-value tnum">${isNum(ml.n) ? ml.n.toLocaleString() : DASH}</div><div class="k-sub">unseen prediction periods</div></div>
    </div>
    <div class="grid g-main-side mt-4" style="align-items:start">
      <section class="panel"><div class="panel-h"><h2>Power curve with ${Math.round(nominal * 100)} % prediction interval</h2><span class="hint">${esc(f.intervals?.method || '')}</span></div>
        <div class="panel-b"><div class="legend-inline"><span><i style="background:var(--chart-observed)"></i>Median forecast</span><span><i class="band"></i>Prediction interval</span></div>
        <div class="chart xl" id="f-curve"></div></div></section>
      <section class="panel"><div class="panel-h"><h2>Audit checklist</h2></div><div class="panel-b">
        <div class="small text-2" style="margin-bottom:8px"><b>Split:</b> ${esc(f.audit?.split || DASH)}</div>
        <ul class="check-list">${checks.map((c) => { const o = typeof c === 'string' ? { check: c, passed: null } : c;
          return `<li><span class="${o.passed ? 'ok' : 'nok'}">${icon(o.passed ? 'check' : o.passed === false ? 'alert' : 'question')}</span><span>${esc(o.check)}<small>${esc(o.detail || (o.passed ? 'passed' : o.passed === false ? 'not passed' : ''))}</small></span></li>`; }).join('')}</ul>
      </div></section>
    </div>
    <section class="panel mt-4"><div class="panel-h"><h2>Interval coverage by wind regime</h2><span class="hint">achieved vs nominal ${Math.round(nominal * 100)} % · amber = off by more than 3 pp</span></div>
      <div class="panel-b"><div class="chart lg" id="f-cov"></div></div></section>`;
  mount(root.querySelector('#f-curve')).setOption(powerCurveOption(f.power_curve || [], { nominal }), true);
  mount(root.querySelector('#f-cov')).setOption(coverageOption(reg, nominal), true);
}

export default {
  async mount(el) {
    root = el; el.innerHTML = '<div class="loading">Loading forecast…</div>';
    try { const f = await api.forecast(); if (root === el) render(f); } catch (e) { el.innerHTML = `<div class="error-box">Could not load forecast: ${esc(e.message)}</div>`; }
  },
  unmount() { root = null; },
};
