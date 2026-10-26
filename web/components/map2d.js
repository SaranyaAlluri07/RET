// 2D SVG farm map: fallback when WebGL is unavailable. Same interface as the 3D view: update(), setTheme(), dispose().
import { STATE_LABEL } from './badges.js';
import { esc, isNum } from '../util.js';

export function createMap2D(container, { assets, onSelect, note }) {
  const xs = assets.map((a) => a.x), ys = assets.map((a) => a.y);
  const pad = 160;
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad, minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
  const W = maxX - minX, H = maxY - minY;
  const sx = (x) => x - minX, sy = (y) => maxY - y; // north up

  const wrap = document.createElement('div');
  wrap.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center';
  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%" role="img" aria-label="Wind farm map">
    <defs><pattern id="m2d-hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="10" height="10" fill="var(--st-unknown-bg)"/><line x1="0" y1="0" x2="0" y2="10" stroke="var(--st-unknown)" stroke-width="3"/></pattern></defs>
    <g class="gridlines" stroke="var(--chart-grid)">${Array.from({ length: 12 }, (_, i) => `<line x1="${(i * W) / 11}" y1="0" x2="${(i * W) / 11}" y2="${H}"/>`).join('')}</g>
    <g class="wind" transform="translate(${W - 70},70)"><circle r="44" fill="var(--surface-1)" stroke="var(--border-strong)"/><text y="-30" text-anchor="middle" font-size="16" fill="var(--muted)">N</text>
      <g class="arr"><path d="M0 -26 L9 -8 L3 -8 L3 24 L-3 24 L-3 -8 L-9 -8 Z" fill="var(--accent)"/></g></g>
    ${assets.map((a) => `<g class="t" data-id="${esc(a.id)}" transform="translate(${sx(a.x)},${sy(a.y)})" style="cursor:pointer">
      <circle class="sel" r="58" fill="none" stroke="var(--accent)" stroke-width="5" stroke-dasharray="10 6" visibility="hidden"/>
      <circle class="ring" r="44" stroke-width="7"/>
      <g class="rotor"><path d="M0 0 L0 -36 M0 0 L31 18 M0 0 L-31 18" stroke="var(--text)" stroke-width="5" stroke-linecap="round"/></g>
      <circle r="6" fill="var(--text)"/>
      <text class="lab" y="74" text-anchor="middle" font-size="22" font-weight="600" fill="var(--text)">${esc(a.label)}</text>
      <text class="sub" y="98" text-anchor="middle" font-size="18" fill="var(--text-2)"></text></g>`).join('')}
  </svg>${note ? `<div class="farm-fallback-note">${esc(note)}</div>` : ''}`;
  container.appendChild(wrap);
  wrap.querySelectorAll('g.t').forEach((g) => g.addEventListener('click', () => {
    const a = assets.find((x) => String(x.id) === g.dataset.id); onSelect?.(a.id);
  }));
  const ang = new Map(); let last = performance.now(), raf = 0, rpm = new Map();
  const tick = (now) => {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    wrap.querySelectorAll('g.t').forEach((g) => {
      const id = g.dataset.id; const a = (ang.get(id) || 0) + (rpm.get(id) || 0) * 6 * dt; ang.set(id, a);
      g.querySelector('.rotor').setAttribute('transform', `rotate(${a})`);
    });
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  return {
    kind: '2d',
    update(turbines, { selected, windDir, windSpeed } = {}) {
      for (const s of turbines || []) {
        const g = wrap.querySelector(`g.t[data-id="${CSS.escape(String(s.id))}"]`); if (!g) continue;
        const st = s.nodata ? 'nodata' : s.state || 'unknown';
        const ring = g.querySelector('.ring');
        ring.setAttribute('stroke', `var(--st-${st})`);
        ring.setAttribute('fill', st === 'unknown' ? 'url(#m2d-hatch)' : `var(--st-${st}-bg, transparent)`);
        ring.setAttribute('stroke-dasharray', st === 'unknown' || st === 'nodata' ? '8 6' : '');
        g.querySelector('.sel').setAttribute('visibility', String(s.id) === String(selected) ? 'visible' : 'hidden');
        g.querySelector('.sub').textContent = `${STATE_LABEL[st] || st} · ${isNum(s.power) ? (s.power * 100).toFixed(0) + ' %' : '—'}`;
        rpm.set(String(s.id), s.nodata ? 0 : s.rotor_rpm || 0);
      }
      const arr = wrap.querySelector('.wind .arr');
      arr.style.display = isNum(windDir) ? '' : 'none';
      if (isNum(windDir)) arr.setAttribute('transform', `rotate(${(windDir + 180) % 360})`);
      wrap.querySelector('.wind').setAttribute('aria-label', isNum(windDir) ? `wind ${windSpeed?.toFixed?.(1)} m/s from ${Math.round(windDir)} degrees` : 'wind direction n/a');
    },
    setTheme() { /* SVG uses CSS variables directly */ },
    dispose() { cancelAnimationFrame(raf); container.innerHTML = ''; },
  };
}
