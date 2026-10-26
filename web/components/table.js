// Sortable data table. Columns: {key, label, align:'num'|'center', render(row)->html, sort(row)->value, title}
import { esc } from '../util.js';

export function renderTable(container, { columns, rows, rowKey = (r) => r.id, onRowClick, selected, rowClass, dense = false,
  sortKey = null, sortDir = 1, empty = 'No rows' }) {
  const st = container._tbl || (container._tbl = { sortKey, sortDir });
  const draw = () => {
    let data = rows.slice();
    const col = columns.find((c) => c.key === st.sortKey);
    if (col) {
      const f = col.sort || ((r) => r[col.key]);
      data.sort((a, b) => {
        const x = f(a), y = f(b);
        if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
        return (x < y ? -1 : x > y ? 1 : 0) * st.sortDir;
      });
    }
    const head = columns.map((c) => {
      const sortable = c.sortable !== false;
      const arrow = st.sortKey === c.key ? `<span class="arrow">${st.sortDir > 0 ? '▲' : '▼'}</span>` : '';
      return `<th class="${c.align || ''} ${sortable ? 'sortable' : ''}" data-key="${esc(c.key)}" ${c.title ? `title="${esc(c.title)}"` : ''}
        ${sortable ? `aria-sort="${st.sortKey === c.key ? (st.sortDir > 0 ? 'ascending' : 'descending') : 'none'}"` : ''}>${c.label}${arrow}</th>`;
    }).join('');
    const body = data.length ? data.map((r) => {
      const k = rowKey(r);
      const cls = [onRowClick ? 'clickable' : '', selected != null && String(selected) === String(k) ? 'selected' : '', rowClass ? rowClass(r) : ''].join(' ');
      return `<tr class="${cls}" data-key="${esc(k)}" ${onRowClick ? 'tabindex="0"' : ''}>${columns.map((c) =>
        `<td class="${c.align || ''}">${c.render ? c.render(r) : esc(r[c.key] ?? '—')}</td>`).join('')}</tr>`;
    }).join('') : `<tr><td colspan="${columns.length}" class="empty">${esc(empty)}</td></tr>`;
    container.innerHTML = `<div class="tbl-wrap"><table class="tbl ${dense ? 'dense' : ''}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    container.querySelectorAll('th.sortable').forEach((th) => th.addEventListener('click', () => {
      const k = th.dataset.key; if (st.sortKey === k) st.sortDir *= -1; else { st.sortKey = k; st.sortDir = 1; } draw();
    }));
    if (onRowClick) {
      const byKey = new Map(rows.map((r) => [String(rowKey(r)), r]));
      container.querySelectorAll('tbody tr[data-key]').forEach((tr) => {
        const go = (e) => { if (e.target.closest('button, a, input, select')) return; onRowClick(byKey.get(tr.dataset.key), e); };
        tr.addEventListener('click', go);
        tr.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(e); });
      });
    }
  };
  draw();
}
