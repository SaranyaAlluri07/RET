// Cases inbox: filterable table of alarm cases with a detail drawer and local analyst feedback.
import { state } from '../state.js';
import { cases, now } from '../source.js';
import { renderTable } from '../components/table.js';
import { stateBadge, caseStatusBadge, meter, STATE_RANK } from '../components/badges.js';
import { openCase, hypothesisText, isUnknownCase, feedbackLabel } from '../components/casedetail.js';
import { esc, num, isNum, compLabel, fmtTime, fmtHours, hoursBetween, turbineLabel, DASH } from '../util.js';

let root = null, filter = 'active';

function render() {
  if (!root) return;
  const all = cases();
  const counts = { active: all.filter((c) => c.status !== 'closed').length, open: 0, acknowledged: 0, closed: 0, all: all.length };
  for (const c of all) counts[c.status] = (counts[c.status] || 0) + 1;
  const rows = all.filter((c) => (filter === 'all' ? true : filter === 'active' ? c.status !== 'closed' : c.status === filter));
  const fb = all.filter((c) => c.feedback);
  root.querySelector('#c-filter').innerHTML = [['active', 'Active'], ['open', 'Open'], ['acknowledged', 'Acknowledged'], ['closed', 'Closed'], ['all', 'All']]
    .map(([k, l]) => `<button class="${filter === k ? 'on' : ''}" data-f="${k}">${l} <span class="muted tnum">${counts[k] || 0}</span></button>`).join('');
  root.querySelectorAll('[data-f]').forEach((b) => b.addEventListener('click', () => { filter = b.dataset.f; render(); }));
  root.querySelector('#c-fb').textContent = fb.length
    ? `Analyst feedback recorded on ${fb.length} case(s): ${['true_alarm', 'false_alarm', 'unclear'].map((k) => `${fb.filter((c) => c.feedback === k).length} ${feedbackLabel(k).toLowerCase()}`).join(' · ')} (stored in this browser)`
    : 'No analyst feedback recorded yet (stored in this browser only).';
  const t = now();
  renderTable(root.querySelector('#c-table'), {
    rows, rowKey: (c) => c.id, onRowClick: (c) => openCase(c, { onChange: render }), empty: 'No cases in this view',
    sortKey: null,
    columns: [
      { key: 'id', label: 'Case', render: (c) => `<b>${esc(c.id)}</b>` },
      { key: 'turbine', label: 'Turbine', render: (c) => esc(turbineLabel(c.turbine, state.meta)) },
      { key: 'status', label: 'Status', render: (c) => caseStatusBadge(c.status) },
      { key: 'state', label: 'State', render: (c) => stateBadge(c.state), sort: (c) => -(STATE_RANK[c.state] ?? 0) },
      { key: 'component', label: 'Component', render: (c) => (c.component ? esc(compLabel(c.component)) : '<span class="muted">Not identified</span>') },
      { key: 'hypothesis', label: 'Hypothesis', render: (c) => `<span class="${isUnknownCase(c) ? 'hypo-unknown' : ''}" style="${isUnknownCase(c) ? 'color:var(--st-unknown)' : ''}">${esc(hypothesisText(c))}</span>` },
      { key: 'deviation_start', label: 'Deviation start', render: (c) => `<span class="tnum">${fmtTime(c.deviation_start)}</span>` },
      { key: 'alarm_time', label: 'Alarm', render: (c) => `<span class="tnum">${fmtTime(c.alarm_time)}</span>` },
      { key: 'elapsed', label: 'Elapsed', align: 'num', title: 'Time since deviation start (until cleared)', sort: (c) => hoursBetween(c.deviation_start, c.cleared_time || t),
        render: (c) => fmtHours(hoursBetween(c.deviation_start, c.cleared_time || t)) },
      { key: 'severity', label: 'Severity', sort: (c) => c.severity, render: (c) => `<span class="row" style="flex-wrap:nowrap;gap:6px"><span style="width:48px">${meter(c.severity, 'var(--st-warning)')}</span><span class="tnum">${num(c.severity, 2)}</span></span>` },
      { key: 'confidence', label: 'Confidence', align: 'num', title: '— when the diagnosis is not calibrated', render: (c) => (isNum(c.confidence) ? `${Math.round(c.confidence * 100)} %` : `<span class="muted" title="not calibrated">${DASH}</span>`) },
      { key: 'feedback', label: 'Feedback', render: (c) => (c.feedback ? `<span class="badge neutral">${esc(feedbackLabel(c.feedback))}</span>` : `<span class="muted">${DASH}</span>`) },
    ],
  });
}

export default {
  mount(el) {
    root = el;
    el.innerHTML = `
      <div class="view-head"><div><h1>Cases</h1><div class="sub">${state.mode === 'replay' ? 'Cases raised so far in the replay (up to the replay cursor)' : 'Cases raised by the detector on simulated turbines'} · confidence shown only when calibrated</div></div>
        <div class="seg" id="c-filter" role="group" aria-label="Filter by status"></div></div>
      <section class="panel"><div class="panel-b flush" id="c-table"></div><div class="panel-f" id="c-fb"></div></section>`;
    render();
  },
  update(reason) { if (reason === 'data' || reason === 'state') render(); },
  unmount() { root = null; },
};
