// ECharts helpers: theme tokens from CSS, chart lifecycle (resize, dispose, connect), and option builders
// for the chart forms WindTwin uses. Conventions: thin 1.5-2px lines, recessive grid, one y-axis per chart,
// units in the chart title, observed = accent, expected = dashed neutral + grey prediction band,
// status colours only for state, categorical palette in fixed order for components.
import { fmtTime, isNum, esc } from '../util.js';
import { STATE_LABEL } from './badges.js';

const E = () => window.echarts;
const charts = new Set();

// ---------------------------------------------------------------- tokens
export function tok() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    font: v('--font-sans'), text: v('--text'), text2: v('--text-2'), muted: v('--muted'), grid: v('--chart-grid'),
    axis: v('--chart-axis'), surface: v('--surface-1'), surface3: v('--surface-3'), border: v('--border-strong'),
    accent: v('--accent'), observed: v('--chart-observed'), expected: v('--chart-expected'), band: v('--chart-band'),
    event: v('--chart-event'), fault: v('--st-fault'),
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => v(`--series-${i}`)),
    seq: [1, 2, 3, 4, 5].map((i) => v(`--seq-${i}`)),
    st: Object.fromEntries(['normal', 'watch', 'warning', 'fault', 'unknown', 'offline', 'tripped', 'nodata']
      .map((s) => [s, v(`--st-${s}`)])),
    q: { ok: v('--surface-3'), degraded: v('--st-warning'), bad: v('--st-fault'), nodata: v('--st-nodata') },
  };
}

// ---------------------------------------------------------------- lifecycle
const ro = new ResizeObserver((entries) => {
  for (const e of entries) { const c = E().getInstanceByDom(e.target); if (c && !c.isDisposed()) c.resize(); }
});

/** Create (or reuse) a chart on an element. */
export function mount(el, group) {
  let c = E().getInstanceByDom(el);
  if (!c) { c = E().init(el, null, { renderer: 'canvas' }); ro.observe(el); charts.add(c); }
  if (group) c.group = group;
  return c;
}
/** Dispose all charts inside a container (call on view unmount / re-render). */
export function disposeWithin(root) {
  for (const c of [...charts]) {
    const dom = c.isDisposed() ? null : c.getDom();
    if (!dom || root.contains(dom)) { if (dom) ro.unobserve(dom); if (!c.isDisposed()) c.dispose(); charts.delete(c); }
  }
}
export function connect(group) { E().connect(group); }

// ---------------------------------------------------------------- shared pieces
export function tooltipBase(t) {
  return {
    backgroundColor: t.surface3, borderColor: t.border, borderWidth: 1, padding: [6, 9],
    textStyle: { color: t.text, fontSize: 12, fontFamily: t.font }, extraCssText: 'box-shadow:0 4px 14px rgba(0,0,0,.25);border-radius:6px;',
  };
}
/** Axis tooltip that hides helper series (names starting with '_') and formats values. */
function axisTooltip(t, fmt) {
  return {
    ...tooltipBase(t), trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: t.muted, width: 1 } },
    formatter: (ps) => {
      const arr = (Array.isArray(ps) ? ps : [ps]).filter((p) => !String(p.seriesName).startsWith('_'));
      if (!arr.length) return '';
      const head = `<div style="color:${t.muted};font-size:11px;margin-bottom:3px">${fmtTime(arr[0].axisValue)}</div>`;
      return head + arr.map((p) => {
        const v = Array.isArray(p.value) ? p.value[1] : p.value;
        return `<div style="display:flex;gap:10px;justify-content:space-between"><span>${p.marker}${esc(p.seriesName)}</span><b style="font-variant-numeric:tabular-nums">${fmt(v)}</b></div>`;
      }).join('');
    },
  };
}
function timeAxis(t, { min, max, showLabels = true }) {
  return {
    type: 'time', min, max,
    axisLabel: { show: showLabels, color: t.muted, fontSize: 10, hideOverlap: true },
    axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, splitLine: { show: false },
    axisPointer: { label: { show: false } },
  };
}
function valueAxis(t, extra = {}) {
  return {
    type: 'value', scale: true, splitNumber: 3,
    axisLabel: { color: t.muted, fontSize: 10, hideOverlap: true }, axisLine: { show: false }, axisTick: { show: false },
    splitLine: { lineStyle: { color: t.grid } }, ...extra,
  };
}
export const GRID = { left: 48, right: 14 };
const zoomInside = () => [{ type: 'inside', xAxisIndex: 0, zoomOnMouseWheel: 'shift', moveOnMouseWheel: false, moveOnMouseMove: true, filterMode: 'none' }];

/** [t, v] pairs up to (and including) index `end`. */
export function pairs(tms, arr, end = tms.length - 1, k = 1) {
  const out = new Array(Math.min(end + 1, tms.length));
  for (let i = 0; i < out.length; i++) { const v = arr?.[i]; out[i] = [tms[i], isNum(v) ? v * k : null]; }
  return out;
}
function bandPairs(tms, lo, hi, end, k) {
  const a = [], b = [];
  for (let i = 0; i <= Math.min(end, tms.length - 1); i++) {
    const l = lo?.[i], h = hi?.[i];
    const ok = isNum(l) && isNum(h);
    a.push([tms[i], ok ? l * k : null]); b.push([tms[i], ok ? (h - l) * k : null]);
  }
  return [a, b];
}

/** Time markers: replay cursor, logged event window, detector first alarm. */
export function markers(t, { cursor, event, firstAlarm, showLabels = false } = {}) {
  const lines = [];
  if (isNum(cursor)) lines.push({ xAxis: cursor, lineStyle: { color: t.text, width: 1, type: 'solid', opacity: .6 }, label: { show: false } });
  if (isNum(firstAlarm)) lines.push({ xAxis: firstAlarm, lineStyle: { color: t.st.warning, width: 1.5, type: 'dashed' },
    label: { show: showLabels, formatter: 'first alarm', color: t.st.warning, fontSize: 10, position: 'insideEndTop' } });
  const out = {};
  if (lines.length) out.markLine = { silent: true, symbol: 'none', animation: false, data: lines };
  if (event && isNum(event.start)) {
    out.markArea = { silent: true, itemStyle: { color: t.event }, data: [[{ xAxis: event.start, name: showLabels ? 'logged event' : '' }, { xAxis: event.end }]],
      label: { show: showLabels, color: t.fault, fontSize: 10, position: 'insideTopLeft' } };
  }
  return out;
}

// ---------------------------------------------------------------- observed vs expected with band
/**
 * sig: {label, unit, observed, expected, lo?, hi?}; s: normalised series (t, end, xMin, xMax).
 * Per-unit power is shown as % of rated.
 */
export function signalOption(s, sig, { mk = {}, showX = false, height } = {}) {
  const t = tok();
  const k = sig.unit === 'pu' ? 100 : 1;
  const unit = sig.unit === 'pu' ? '% rated' : sig.unit || '';
  const fmt = (v) => (isNum(v) ? `${v.toFixed(1)} ${unit}` : '—');
  const ser = [];
  if (sig.lo && sig.hi) {
    const [a, b] = bandPairs(s.t, sig.lo, sig.hi, s.end, k);
    ser.push({ name: '_lo', type: 'line', data: a, stack: 'band', symbol: 'none', lineStyle: { opacity: 0 }, silent: true, connectNulls: false });
    ser.push({ name: '_band', type: 'line', data: b, stack: 'band', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: t.band }, silent: true });
  }
  ser.push({ name: 'Expected', type: 'line', data: pairs(s.t, sig.expected, s.end, k), symbol: 'none', lineStyle: { color: t.expected, width: 1.5, type: [4, 3] }, itemStyle: { color: t.expected } });
  ser.push({ name: 'Observed', type: 'line', data: pairs(s.t, sig.observed, s.end, k), symbol: 'none', lineStyle: { color: t.observed, width: 1.8 }, itemStyle: { color: t.observed }, ...mk });
  return {
    animation: false, textStyle: { fontFamily: t.font },
    grid: { ...GRID, top: 8, bottom: showX ? 22 : 6, height },
    xAxis: timeAxis(t, { min: s.xMin, max: s.xMax, showLabels: showX }),
    yAxis: valueAxis(t), tooltip: axisTooltip(t, fmt), dataZoom: zoomInside(), series: ser,
  };
}

// ---------------------------------------------------------------- component statistics (score / EWMA / CUSUM)
/** lines: [{name, data(array aligned with s.t), color}], limits: [{value, label, color}] */
export function statOption(s, lines, limits = [], { mk = {}, showX = false, fmtDigits = 2 } = {}) {
  const t = tok();
  const fmt = (v) => (isNum(v) ? v.toFixed(fmtDigits) : '—');
  const ser = lines.map((l, i) => ({
    name: l.name, type: 'line', data: pairs(s.t, l.data, s.end), symbol: 'none',
    lineStyle: { color: l.color, width: l.width || 1.6, opacity: l.dim ? 0.35 : 1 }, itemStyle: { color: l.color },
    ...(i === 0 ? mk : {}),
  }));
  if (limits.length) {
    ser.push({
      name: '_limits', type: 'line', data: [], silent: true,
      markLine: { silent: true, symbol: 'none', animation: false, data: limits.map((L) => ({
        yAxis: L.value, lineStyle: { color: L.color || t.muted, type: 'dashed', width: 1 },
        label: { formatter: `${L.label} ${L.value}`, color: L.color || t.muted, fontSize: 10, position: 'insideEndTop' } })) },
    });
  }
  const maxLimit = Math.max(0, ...limits.map((l) => l.value));
  return {
    animation: false, textStyle: { fontFamily: t.font },
    grid: { ...GRID, top: 8, bottom: showX ? 22 : 6 },
    xAxis: timeAxis(t, { min: s.xMin, max: s.xMax, showLabels: showX }),
    yAxis: valueAxis(t, { scale: false, min: 0, max: (v) => { const m = Math.max(v.max, maxLimit * 1.15); return m > 5 ? Math.ceil(m) : Math.ceil(m * 10) / 10; } }),
    tooltip: axisTooltip(t, fmt), dataZoom: zoomInside(), series: ser,
  };
}

// ---------------------------------------------------------------- categorical strip (state / data quality)
let hatchCache = { key: '', canvas: null };
function hatch(color, bg) {
  const key = color + bg;
  if (hatchCache.key !== key) {
    const c = document.createElement('canvas'); c.width = c.height = 8;
    const g = c.getContext('2d'); g.fillStyle = bg; g.fillRect(0, 0, 8, 8);
    g.strokeStyle = color; g.lineWidth = 2; g.beginPath(); g.moveTo(-2, 10); g.lineTo(10, -2); g.moveTo(6, 10); g.lineTo(10, 6); g.moveTo(-2, 2); g.lineTo(2, -2); g.stroke();
    hatchCache = { key, canvas: c };
  }
  return { image: hatchCache.canvas, repeat: 'repeat' };
}
/** Run-length segments [[t0, t1, category], ...] up to index end. */
export function segments(tms, cats, end = tms.length - 1) {
  const out = []; const n = Math.min(end, tms.length - 1);
  const step = tms.length > 1 ? tms[1] - tms[0] : 6e5;
  let s = 0;
  for (let i = 1; i <= n + 1; i++) {
    if (i > n || cats[i] !== cats[s]) { out.push([tms[s], i <= n ? tms[i] : tms[n] + step, cats[s] ?? 'nodata']); s = i; }
  }
  return out;
}
/** kind: 'state' | 'quality' */
export function stripOption(s, cats, { kind = 'state', mk = {}, showX = false, slider = false } = {}) {
  const t = tok();
  const colorFor = (c) => {
    if (kind === 'quality') return t.q[c] || t.q.nodata;
    if (c === 'unknown') return hatch(t.st.unknown, t.surface);
    return t.st[c] || t.st.nodata;
  };
  const names = kind === 'quality' ? { ok: 'Data OK', degraded: 'Degraded', bad: 'Bad / missing', nodata: 'No data' } : STATE_LABEL;
  // custom-series values must be numeric: encode the category as an index into `keys`
  const segs = segments(s.t, cats, s.end);
  const keys = [...new Set(segs.map((g) => g[2]))];
  const data = segs.map((g) => [g[0], g[1], keys.indexOf(g[2])]);
  return {
    animation: false, textStyle: { fontFamily: t.font },
    grid: { ...GRID, top: 2, bottom: (showX ? 20 : 2) + (slider ? 30 : 0) },
    xAxis: timeAxis(t, { min: s.xMin, max: s.xMax, showLabels: showX }),
    yAxis: { type: 'value', min: 0, max: 1, show: false },
    tooltip: { ...tooltipBase(t), trigger: 'item',
      formatter: (p) => `<b>${esc(names[keys[p.value[2]]] || keys[p.value[2]])}</b><br><span style="color:${t.muted}">${fmtTime(p.value[0])} → ${fmtTime(p.value[1])}</span>` },
    dataZoom: slider ? [...zoomInside(), { type: 'slider', xAxisIndex: 0, height: 18, bottom: 4, filterMode: 'none', borderColor: t.border,
      fillerColor: 'rgba(127,127,127,.12)', textStyle: { color: t.muted, fontSize: 10 }, labelFormatter: (v) => fmtTime(v),
      dataBackground: { lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 } }, handleStyle: { color: t.surface3, borderColor: t.border } }] : zoomInside(),
    series: [{
      type: 'custom', name: kind, data, encode: { x: [0, 1] }, clip: true,
      renderItem: (params, api) => {
        const a = api.coord([api.value(0), 0]), b = api.coord([api.value(1), 0]);
        const cs = params.coordSys;
        const x0 = Math.max(cs.x, a[0]), x1 = Math.min(cs.x + cs.width, b[0]);
        if (x1 <= x0) return null;
        return { type: 'rect', shape: { x: x0, y: cs.y, width: Math.max(1, x1 - x0), height: cs.height },
          style: { fill: colorFor(keys[api.value(2)]) } };
      },
      ...mk,
    }],
  };
}

// ---------------------------------------------------------------- generic bars (method comparison small multiples)
/** cats: labels; values: numbers; highlight: index set; counts: optional label strings per bar */
export function barOption({ cats, values, highlight = new Set(), fmt = (v) => v, labels, max, baselineIdx = -1 }) {
  const t = tok();
  return {
    animation: false, textStyle: { fontFamily: t.font },
    grid: { left: 8, right: 54, top: 4, bottom: 4, containLabel: true },
    xAxis: { type: 'value', max, show: false },
    yAxis: { type: 'category', data: cats, inverse: true, axisLabel: { color: t.text2, fontSize: 11, width: 150, overflow: 'truncate' }, axisLine: { show: false }, axisTick: { show: false } },
    tooltip: { ...tooltipBase(t), trigger: 'item', formatter: (p) => `${esc(p.name)}<br><b>${labels ? esc(labels[p.dataIndex]) : fmt(p.value)}</b>` },
    series: [{
      type: 'bar', barWidth: 12, data: values.map((v, i) => ({ value: v, itemStyle: {
        color: highlight.has(i) ? t.accent : i === baselineIdx ? t.expected : t.axis, borderRadius: [0, 3, 3, 0] } })),
      label: { show: true, position: 'right', color: t.text2, fontSize: 11, formatter: (p) => (labels ? labels[p.dataIndex] : fmt(p.value)) },
    }],
  };
}

// ---------------------------------------------------------------- confusion matrix heatmap
export function heatmapOption({ rows, cols, matrix }) {
  const t = tok();
  const data = []; let mx = 1;
  matrix.forEach((r, i) => r.forEach((v, j) => { data.push([j, i, v]); mx = Math.max(mx, v); }));
  return {
    animation: false, textStyle: { fontFamily: t.font },
    grid: { left: 8, right: 8, top: 8, bottom: 28, containLabel: true },
    xAxis: { type: 'category', data: cols, position: 'bottom', name: 'Diagnosed', nameLocation: 'middle', nameGap: 42, nameTextStyle: { color: t.muted, fontSize: 11 },
      axisLabel: { color: t.text2, fontSize: 10, interval: 0, rotate: 20 }, axisLine: { show: false }, axisTick: { show: false }, splitArea: { show: false } },
    yAxis: { type: 'category', data: rows, inverse: true, name: 'Logged fault', nameTextStyle: { color: t.muted, fontSize: 11, align: 'right' },
      axisLabel: { color: t.text2, fontSize: 11 }, axisLine: { show: false }, axisTick: { show: false } },
    visualMap: { show: false, min: 0, max: mx, inRange: { color: [t.seq[0], t.seq[1], t.seq[2], t.seq[3], t.seq[4]] } },
    tooltip: { ...tooltipBase(t), trigger: 'item', formatter: (p) => `Logged: <b>${esc(rows[p.value[1]])}</b><br>Diagnosed: <b>${esc(cols[p.value[0]])}</b><br>Count: <b>${p.value[2]}</b>` },
    series: [{ type: 'heatmap', data, itemStyle: { borderColor: t.surface, borderWidth: 2, borderRadius: 3 },
      label: { show: true, fontSize: 12, fontWeight: 600, formatter: (p) => p.value[2],
        color: t.text, textBorderColor: t.surface, textBorderWidth: 2 } }],
  };
}

// ---------------------------------------------------------------- power curve with prediction band
export function powerCurveOption(curve, { nominal } = {}) {
  const t = tok();
  const xs = curve.map((p) => p.wind);
  const lo = curve.map((p) => [p.wind, p.p_lo * 100]);
  const band = curve.map((p) => [p.wind, (p.p_hi - p.p_lo) * 100]);
  return {
    animation: false, textStyle: { fontFamily: t.font },
    grid: { left: 48, right: 16, top: 12, bottom: 36 },
    xAxis: { type: 'value', min: 0, max: Math.max(...xs), name: 'Wind speed (m/s)', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: t.muted, fontSize: 11 },
      axisLabel: { color: t.muted, fontSize: 10 }, axisLine: { lineStyle: { color: t.axis } }, splitLine: { show: false } },
    yAxis: valueAxis(t, { scale: false, min: 0, max: 110, axisLabel: { color: t.muted, fontSize: 10, formatter: '{value} %' } }),
    tooltip: { ...tooltipBase(t), trigger: 'axis', formatter: (ps) => {
      const i = ps[0].dataIndex, p = curve[i];
      return `<div style="color:${t.muted};font-size:11px">${p.wind.toFixed(1)} m/s</div>Median <b>${(p.p_med * 100).toFixed(1)} %</b><br>${nominal ? `${Math.round(nominal * 100)} % interval` : 'Interval'} <b>${(p.p_lo * 100).toFixed(1)}–${(p.p_hi * 100).toFixed(1)} %</b>`;
    } },
    series: [
      { name: '_lo', type: 'line', data: lo, stack: 'b', symbol: 'none', lineStyle: { opacity: 0 }, silent: true },
      { name: 'Prediction interval', type: 'line', data: band, stack: 'b', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: t.band }, silent: true },
      { name: 'Median forecast', type: 'line', data: curve.map((p) => [p.wind, p.p_med * 100]), symbol: 'none', lineStyle: { color: t.observed, width: 2 } },
    ],
  };
}

// ---------------------------------------------------------------- interval coverage per regime
export function coverageOption(rows, nominal) {
  const t = tok();
  return {
    animation: false, textStyle: { fontFamily: t.font },
    grid: { left: 8, right: 16, top: 22, bottom: 8, containLabel: true },
    xAxis: { type: 'category', data: rows.map((r) => r.regime), axisLabel: { color: t.text2, fontSize: 10, interval: 0, width: 90, overflow: 'break' }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false } },
    yAxis: valueAxis(t, { scale: false, min: (v) => Math.max(0, Math.floor(Math.min(v.min, nominal * 100) / 10) * 10 - 10), max: 100,
      axisLabel: { color: t.muted, fontSize: 10, formatter: '{value} %' } }),
    tooltip: { ...tooltipBase(t), trigger: 'item', formatter: (p) => { const r = rows[p.dataIndex];
      return `<b>${esc(r.regime)}</b>${isNum(r.wind_lo) ? ` (${r.wind_lo}–${r.wind_hi} m/s)` : ''}<br>Achieved <b>${(r.coverage * 100).toFixed(1)} %</b> vs nominal ${(nominal * 100).toFixed(0)} %<br>n = ${r.n?.toLocaleString?.() ?? '—'}`; } },
    series: [{
      type: 'bar', barMaxWidth: 34, data: rows.map((r) => ({ value: r.coverage * 100,
        itemStyle: { color: Math.abs(r.coverage - nominal) <= 0.03 ? t.accent : t.st.warning, borderRadius: [3, 3, 0, 0] } })),
      label: { show: true, position: 'top', color: t.text2, fontSize: 10, formatter: (p) => `${p.value.toFixed(1)} %` },
      markLine: { silent: true, symbol: 'none', data: [{ yAxis: nominal * 100 }], lineStyle: { color: t.text, type: 'dashed', width: 1 },
        label: { formatter: `nominal ${(nominal * 100).toFixed(0)} %`, color: t.text2, fontSize: 10, position: 'insideEndTop' } },
    }],
  };
}
