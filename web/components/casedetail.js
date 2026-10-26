// Case detail drawer: hypothesis, ranked evidence, timeline, analyst feedback (stored locally).
// Operator view: simulation ground truth (case.truth) is intentionally NOT shown here - see Simulation lab > Test oracle.
import { openDrawer, closeDrawer } from './drawer.js';
import { stateBadge, caseStatusBadge, meter } from './badges.js';
import { icon } from './icons.js';
import { esc, num, fmtTime, fmtHours, hoursBetween, compLabel, turbineLabel, isNum, DASH } from '../util.js';
import { state } from '../state.js';
import { now, setCaseLocal, caseLocal } from '../source.js';

export const FEEDBACK = [
  { id: 'true_alarm', label: 'True alarm', icon: 'check' },
  { id: 'false_alarm', label: 'False alarm', icon: 'x' },
  { id: 'unclear', label: 'Unclear', icon: 'question' },
];
export const feedbackLabel = (f) => FEEDBACK.find((x) => x.id === f)?.label || DASH;

export const isUnknownCase = (c) => c.state === 'unknown' || !c.component || /^unknown/i.test(c.hypothesis || '');
export function hypothesisText(c) {
  if (isUnknownCase(c) && !c.hypothesis) return 'Unknown — insufficient evidence';
  return (c.hypothesis || DASH).replace(/^Unknown\s*-\s*/i, 'Unknown — ');
}

export function evidenceTable(ev = []) {
  if (!ev.length) return '<div class="empty">No evidence recorded</div>';
  const max = Math.max(...ev.map((e) => e.contribution || 0), 1e-9);
  return `<div class="tbl-wrap"><table class="tbl dense"><thead><tr><th>Signal</th><th class="num" title="Observed">Obs.</th><th class="num" title="Expected">Exp.</th><th class="num" title="Regime-normalised residual">z</th><th style="width:92px">Contrib.</th></tr></thead><tbody>
    ${ev.map((e) => `<tr><td>${esc(e.label || e.signal)}</td><td class="num">${num(e.observed, 1)}</td><td class="num">${num(e.expected, 1)}</td>
      <td class="num" style="${Math.abs(e.z) >= 3 ? 'color:var(--st-warning);font-weight:600' : ''}">${isNum(e.z) ? (e.z > 0 ? '+' : '') + e.z.toFixed(2) : DASH}</td>
      <td><span class="contrib">${meter((e.contribution || 0) / max)}<span>${isNum(e.contribution) ? Math.round(e.contribution * 100) + ' %' : DASH}</span></span></td></tr>`).join('')}
  </tbody></table></div>`;
}

/** Open the drawer for a case. onChange is called after acknowledge / feedback. */
export function openCase(c, { onChange } = {}) {
  const t = now();
  const elapsedEnd = c.cleared_time || t;
  const rated = state.rated ?? state.meta?.rated_mw_default ?? 2;
  const unknown = isUnknownCase(c);
  const render = () => {
    const loc = caseLocal(c);
    const fb = loc.feedback || null;
    const status = c.status === 'closed' ? 'closed' : loc.status || c.status;
    return { fb, status };
  };
  const { status } = render();
  const body = `
    <div>
      <div class="section-label">Leading hypothesis</div>
      <div class="hypo ${unknown ? 'unknown' : ''}">${unknown ? icon('st_unknown') + ' ' : ''}${esc(hypothesisText(c))}</div>
      ${unknown ? '<div class="small muted mt-2">The evidence does not single out a component (or data quality is insufficient). WindTwin reports this explicitly instead of forcing a fault class.</div>' : ''}
    </div>
    <dl class="kv">
      <dt>Turbine</dt><dd>${esc(turbineLabel(c.turbine, state.meta))}</dd>
      <dt>Component</dt><dd>${esc(c.component ? compLabel(c.component) : 'Not identified')}</dd>
      <dt>Severity</dt><dd><span class="row" style="flex-wrap:nowrap">${meter(c.severity, 'var(--st-warning)')}<span>${num(c.severity, 2)}</span></span></dd>
      <dt>Confidence</dt><dd>${isNum(c.confidence) ? `${Math.round(c.confidence * 100)} % <span class="muted">(calibrated)</span>` : `${DASH} <span class="muted">not calibrated</span>`}</dd>
      <dt>Elapsed since deviation</dt><dd>${fmtHours(hoursBetween(c.deviation_start, elapsedEnd))}</dd>
      <dt>Deviation → alarm</dt><dd>${fmtHours(hoursBetween(c.deviation_start, c.alarm_time))}</dd>
      <dt>Lost energy</dt><dd>${num(c.lost_energy_pu_h, 2)} pu·h <span class="muted">≈ ${isNum(c.lost_energy_pu_h) ? (c.lost_energy_pu_h * rated).toFixed(2) : DASH} MWh illustrative @ ${rated} MW</span></dd>
      <dt>Source</dt><dd>${c.source === 'replay' ? 'Replay · real SCADA' : 'Simulation · synthetic data'}</dd>
    </dl>
    <div><div class="section-label">Evidence · signals behind the alert (at alarm time)</div>${evidenceTable(c.evidence)}</div>
    <div><div class="section-label">Timeline</div><ul class="tl">
      <li><i style="border-color:var(--st-watch)"></i><b>Deviation start</b><span>${fmtTime(c.deviation_start)}</span></li>
      <li><i style="border-color:var(--st-${c.state === 'fault' ? 'fault' : c.state === 'unknown' ? 'unknown' : 'warning'})"></i><b>Alarm raised (${esc(c.state)})</b><span>${fmtTime(c.alarm_time)}</span></li>
      ${c.cleared_time ? `<li><i style="border-color:var(--st-normal)"></i><b>Cleared</b><span>${fmtTime(c.cleared_time)}</span></li>`
        : `<li><i style="border-color:var(--muted)"></i><b>Still active</b><span>as of ${fmtTime(t)}</span></li>`}
    </ul></div>`;
  const footer = `<span class="small muted">Analyst feedback:</span>
    ${FEEDBACK.map((f) => `<button class="btn sm" data-fb="${f.id}">${icon(f.icon)}${f.label}</button>`).join('')}
    <span style="flex:1"></span>
    <button class="btn sm" data-ack ${status !== 'open' ? 'disabled' : ''}>Acknowledge</button>
    <a class="btn sm primary" href="#/turbine/${encodeURIComponent(c.turbine)}" data-inv>Investigate ${icon('arrowRight')}</a>`;
  const d = openDrawer({
    title: `<b class="nowrap">Case ${esc(c.id)}</b>${stateBadge(c.state)}<span data-status>${caseStatusBadge(status)}</span>`,
    body, footer,
  });
  const paint = () => {
    const { fb, status: s } = render();
    d.querySelectorAll('[data-fb]').forEach((b) => b.classList.toggle('pressed', b.dataset.fb === fb));
    d.querySelector('[data-status]').innerHTML = caseStatusBadge(s);
    d.querySelector('[data-ack]').disabled = s !== 'open';
  };
  d.querySelectorAll('[data-fb]').forEach((b) => b.addEventListener('click', () => {
    const cur = caseLocal(c).feedback;
    setCaseLocal(c, { feedback: cur === b.dataset.fb ? null : b.dataset.fb }); paint(); onChange?.();
  }));
  d.querySelector('[data-ack]').addEventListener('click', () => { setCaseLocal(c, { status: 'acknowledged' }); paint(); onChange?.(); });
  d.querySelector('[data-inv]').addEventListener('click', () => { state.selected = c.turbine; closeDrawer(); });
  paint();
}
