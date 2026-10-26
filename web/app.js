// WindTwin SPA entry: app shell (nav rail, top bar, data-source banner), hash router, theme, playback controls.
import { state, setState, subscribe } from './state.js';
import { api, init as apiInit, isMock, mockInfo } from './api.js';
import { now, dataSourceLabel } from './source.js';
import { SPEEDS, setMode, ensureModeData, togglePlay, stepOnce, setSpeed, simReset } from './actions.js';
import { icon } from './components/icons.js';
import { disposeWithin } from './components/charts.js';
import { closeDrawer } from './components/drawer.js';
import { esc, fmtTime, isNum, toast } from './util.js';

import fleet from './views/fleet.js';
import turbine from './views/turbine.js';
import cases from './views/cases.js';
import replay from './views/replay.js';
import simlab from './views/simlab.js';
import evaluation from './views/evaluation.js';
import forecast from './views/forecast.js';
import about from './views/about.js';

const ROUTES = [
  { id: 'fleet', label: 'Fleet', icon: 'fleet', view: fleet },
  { id: 'turbine', label: 'Investigate', icon: 'turbine', view: turbine },
  { id: 'cases', label: 'Cases', icon: 'cases', view: cases },
  { sep: true },
  { id: 'replay', label: 'Replay', icon: 'replay', view: replay },
  { id: 'sim', label: 'Sim lab', icon: 'sim', view: simlab },
  { sep: true },
  { id: 'eval', label: 'Evaluation', icon: 'eval', view: evaluation },
  { id: 'forecast', label: 'Forecast', icon: 'forecast', view: forecast },
  { id: 'about', label: 'About & data', icon: 'about', view: about },
];

const $main = document.getElementById('main');
let current = null; // {route, view, el, param}

// ---------------------------------------------------------------- shell
function renderRail() {
  document.getElementById('rail').innerHTML = `<div class="logo" title="WindTwin">${icon('logo')}</div>` +
    ROUTES.map((r) => (r.sep ? '<div class="rail-sep"></div>'
      : `<a href="#/${r.id}" data-route="${r.id}">${icon(r.icon)}<span>${r.label}</span></a>`)).join('') +
    `<div class="rail-foot">${esc(state.meta?.pipeline_version ? 'pipeline ' + state.meta.pipeline_version : '')}</div>`;
}
function markRail() {
  document.querySelectorAll('#rail a').forEach((a) => {
    const on = a.dataset.route === current?.route.id;
    a.classList.toggle('active', on); if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
}

function renderTopbar() {
  const rated = state.rated ?? state.meta?.rated_mw_default ?? 2;
  document.getElementById('topbar').innerHTML = `
    <div class="brand"><b>WindTwin</b><span>Condition monitoring · ${esc(state.meta?.dataset?.name?.split(' - ')[1] || 'Wind Farm A')}</span></div>
    <div class="seg" role="group" aria-label="Data source mode">
      <button class="replay ${state.mode === 'replay' ? 'on' : ''}" data-mode="replay" title="Historical replay of real SCADA (CARE to Compare)">${icon('replay')}Replay</button>
      <button class="simulation ${state.mode === 'simulation' ? 'on' : ''}" data-mode="simulation" title="Synthetic simulation with fault injection">${icon('sim')}Simulation</button>
    </div>
    <div class="spacer"></div>
    <label class="field-inline" title="Power in the data is per-unit of rated. MW / MWh figures use this user-entered rating and are illustrative only.">
      Rated power <input id="rated" type="number" min="0.1" max="20" step="0.1" value="${rated}" aria-label="Rated power per turbine (MW, illustrative)"> MW
      <span class="tag-illustrative">illustrative</span></label>
    <button class="btn ghost icon" id="theme" aria-label="Toggle light/dark theme" title="Toggle theme">${icon(state.theme === 'light' ? 'moon' : 'sun')}</button>`;
  document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  document.getElementById('rated').addEventListener('change', (e) => {
    const v = parseFloat(e.target.value); if (isNum(v) && v > 0) setState({ rated: v }, 'rated');
  });
  document.getElementById('theme').addEventListener('click', toggleTheme);
}

function renderBanner() {
  const b = document.getElementById('banner');
  const m = state.mode;
  b.className = `banner ${m}`;
  const speeds = SPEEDS[m];
  b.innerHTML = `
    <span class="mode-label">${m === 'replay' ? 'REPLAY' : 'SIMULATION'}</span>
    <span class="source" title="${esc(dataSourceLabel())}">${esc(dataSourceLabel())}${state.mock ? " · placeholder fixture" : ""}${m === 'replay' && state.replay ? ` · ${esc(state.replay.description || '')}` : ''}</span>
    <span class="clock"><small>${m === 'replay' ? 'replayed time' : 'simulated time'}</small><b id="clock">—</b></span>
    <span class="spacer"></span>
    ${state.mock ? `<span class="chip-mock" title="${esc(mockInfo())}: UI is running on placeholder fixtures in web/mock - numbers are not results">MOCK DATA</span>` : ''}
    <div class="playback" role="group" aria-label="Playback">
      ${m === 'simulation' ? `<button class="btn sm icon ghost" id="pb-reset" title="Reset simulation to healthy baseline" aria-label="Reset simulation">${icon('reset')}</button>` : ''}
      ${m === 'replay' ? `<button class="btn sm icon ghost" id="pb-back" title="Step back" aria-label="Step back">${icon('back')}</button>` : ''}
      <button class="btn sm icon ${state.playing ? 'pressed' : ''}" id="pb-play" title="${state.playing ? 'Pause' : 'Play'}" aria-label="${state.playing ? 'Pause' : 'Play'}">${icon(state.playing ? 'pause' : 'play')}</button>
      <button class="btn sm icon ghost" id="pb-step" title="Step forward one ${m === 'replay' ? 'row' : '10-min step'}" aria-label="Step forward">${icon('step')}</button>
      <select id="pb-speed" aria-label="Playback speed" title="${m === 'replay' ? 'rows' : '10-min steps'} per tick">
        ${speeds.map((s) => `<option value="${s}" ${s === state.speed ? 'selected' : ''}>×${s}</option>`).join('')}</select>
    </div>`;
  b.querySelector('#pb-reset')?.addEventListener('click', async () => {
    try { await simReset(); toast('Simulation reset to healthy baseline.'); } catch (e) { toast(`Reset failed: ${e.message}`, 'err'); }
  });
  b.querySelector('#pb-play').addEventListener('click', togglePlay);
  b.querySelector('#pb-step').addEventListener('click', () => stepOnce(1));
  b.querySelector('#pb-back')?.addEventListener('click', () => stepOnce(-1));
  b.querySelector('#pb-speed').addEventListener('change', (e) => setSpeed(e.target.value));
  updateClock();
}
function updateClock() {
  const el = document.getElementById('clock'); if (!el) return;
  const t = now();
  el.textContent = Number.isFinite(t) ? fmtTime(t) : '—';
  if (state.mode === 'replay' && state.replay) el.title = `row ${state.cursor + 1} of ${state.replay.t.length}`;
}

// ---------------------------------------------------------------- theme
function toggleTheme() {
  const t = state.theme === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', t);
  try { localStorage.setItem('windtwin.theme', t); } catch { /* ignore */ }
  setState({ theme: t }, 'theme');
}

// ---------------------------------------------------------------- router
function parseHash() {
  const [id, ...rest] = (location.hash.replace(/^#\/?/, '') || 'fleet').split('/');
  const route = ROUTES.find((r) => r.id === id) || ROUTES[0];
  return { route, param: rest.length ? decodeURIComponent(rest.join('/')) : null };
}
function mountRoute() {
  const { route, param } = parseHash();
  closeDrawer();
  if (current) { try { current.view.unmount?.(); } catch (e) { console.error(e); } disposeWithin($main); }
  $main.innerHTML = '';
  const el = document.createElement('div'); el.className = 'view'; $main.appendChild(el);
  current = { route, view: route.view, el, param };
  document.title = `${route.label} · WindTwin`;
  markRail();
  try { route.view.mount(el, { param }); } catch (e) { console.error(e); el.innerHTML = `<div class="error-box">View failed to render: ${esc(e.message)}</div>`; }
  $main.scrollTop = 0;
}
let remountPending = false;
function scheduleRemount() {
  if (remountPending) return; remountPending = true;
  setTimeout(() => { remountPending = false; remount(); }, 0);
}
function remount() { // theme switch: rebuild charts / 3D with new colours, keep route
  if (!current) return;
  const top = $main.scrollTop; mountRoute(); $main.scrollTop = top;
}

// ---------------------------------------------------------------- state wiring
subscribe((reason) => {
  if (reason === 'data' || reason === 'replay') updateClock();
  if (reason === 'mode' || reason === 'playback' || reason === 'replay') renderBanner();
  if (reason === 'mode') renderTopbar();
  if (reason === 'theme') renderTopbar();
  // a new data source (mode switch / replay dataset) or theme rebuilds the view; debounced
  if (reason === 'theme' || reason === 'mode' || reason === 'replay') { scheduleRemount(); return; }
  if (current?.view.update) {
    try { current.view.update(reason); } catch (e) { console.error(e); }
  }
});
window.addEventListener('hashchange', mountRoute);

// ---------------------------------------------------------------- boot
async function boot() {
  await apiInit();
  state.mock = isMock();
  try { state.meta = await api.meta(); } catch (e) {
    $main.innerHTML = `<div class="view"><div class="error-box">Could not load /api/meta: ${esc(e.message)}</div></div>`; return;
  }
  if (state.rated == null) state.rated = state.meta.rated_mw_default ?? 2;
  if (!['replay', 'simulation'].includes(state.mode)) state.mode = 'simulation';
  renderRail(); renderTopbar(); renderBanner();
  await ensureModeData();
  renderBanner();
  mountRoute();
}
boot();
