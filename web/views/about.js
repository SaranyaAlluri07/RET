// About & data: dataset citation and version, units disclaimer, modes, pipeline version, method summary.
import { state } from '../state.js';
import { isMock, mockInfo } from '../api.js';
import { esc } from '../util.js';

export default {
  mount(el) {
    const m = state.meta || {}, d = m.dataset || {};
    el.innerHTML = `
      <div class="view-head"><div><h1>About & data</h1><div class="sub">What the data is, what the numbers mean, and how alarms are produced</div></div></div>
      <div class="grid g-2" style="align-items:start">
        <section class="panel"><div class="panel-h"><h2>Dataset</h2></div><div class="panel-b prose">
          <dl class="kv"><dt>Name</dt><dd>${esc(d.name || '—')}</dd><dt>Version</dt><dd>v${esc(d.version || '—')}</dd>
            <dt>Zenodo record</dt><dd>${esc(d.record || '—')}</dd><dt>DOI</dt><dd>${d.doi ? `<a href="https://doi.org/${esc(d.doi)}" target="_blank" rel="noopener">${esc(d.doi)}</a>` : '—'}</dd>
            <dt>Assets</dt><dd>${(m.assets || []).map((a) => esc(a.label)).join(', ')}</dd><dt>Pipeline</dt><dd class="mono">${esc(m.pipeline_version || '—')}</dd></dl>
          <h3>Citation</h3>
          <div class="cite">Gück, C., Roelofs, C. M. A., Faulstich, S. CARE to Compare: A real-world benchmark dataset for early fault detection in wind turbine data. Zenodo, version ${esc(d.version || '?')}. ${d.doi ? 'https://doi.org/' + esc(d.doi) : ''}</div>
          <p class="mt-2 small">Only per-timestamp average signals are used; the dataset documentation warns that some min/max/std channels contain implausible values and pitch angles can wrap.</p>
        </div></section>
        <section class="panel"><div class="panel-h"><h2>Units and modes</h2></div><div class="panel-b prose">
          <ul>
            <li><b>Power is per-unit of rated</b> (the dataset is normalised 0–1). The UI shows % of rated. MW / MWh values use the rated power you enter in the top bar (${esc(state.rated)} MW) and are <b>illustrative</b>, not measured farm capacity.</li>
            <li><b>Replay</b> plays precomputed detector output over real SCADA from one CARE dataset (one asset). Timestamps are replayed, not current.</li>
            <li><b>Simulation</b> runs synthetic turbines driven by models trained on real data, with reproducible fault injection. Synthetic detections are never evidence of real-data accuracy.</li>
            <li>There is no live-telemetry connection; nothing in WindTwin is labelled "live".</li>
            ${isMock() ? `<li><b>Mock data:</b> ${esc(mockInfo())}; this session runs on placeholder fixtures (web/mock). Numbers are not results.</li>` : ''}
          </ul></div></section>
        <section class="panel" style="grid-column:1/-1"><div class="panel-h"><h2>Method summary</h2></div><div class="panel-b prose" style="max-width:none">
          <ol>
            <li><b>Data quality and operating state</b> — missing, flat-lined, out-of-range and implausible values; non-production states are <i>offline</i>, insufficient quality is <i>unknown</i>.</li>
            <li><b>Normal-behaviour models</b> per target signal (component temperatures, power) with leakage-safe inputs, trained on normal-status training rows only.</li>
            <li><b>Regime-aware residual normalisation</b> — z = (observed − expected) / σ, with σ estimated per operating regime on validation data.</li>
            <li><b>Temporal alarm logic</b> — EWMA and CUSUM per component, persistence, hysteresis and cooldown: normal → watch → warning → fault.</li>
            <li><b>Evidence-based diagnosis</b> — components ranked by residual evidence; an explicit <i>Unknown — insufficient evidence</i> outcome; confidence only shown when calibrated.</li>
            <li><b>Evaluation</b> — CARE score and sub-scores, event recall, false alarms per turbine-month, delay median/p90, component-level results with counts; hyper-parameters chosen leave-one-turbine-out.</li>
          </ol></div></section>
      </div>`;
  },
};
