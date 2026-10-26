// Application actions: data loading per mode, mode switching, playback (replay cursor / simulation steps).
import { state, setState, emit } from './state.js';
import { api } from './api.js';
import { prepareReplay } from './source.js';
import { toast } from './util.js';

export const SPEEDS = { replay: [1, 3, 6, 12, 24], simulation: [1, 3, 6, 18, 36] };
const TICK_MS = { replay: 450, simulation: 1000 };
let timer = null, inflight = false;

// ---------------------------------------------------------------- simulation
export async function refreshSim({ fleet = true } = {}) {
  const [f, cases, inj] = await Promise.all([
    fleet ? api.simState() : Promise.resolve(state.fleet), api.simCases(), api.simInjections().catch(() => [])]);
  setState({ fleet: f, simCases: cases, injections: inj }, 'data');
  if (state.selected == null && f?.turbines?.length) state.selected = pickDefault(f.turbines);
}
function pickDefault(ts) {
  const rank = { tripped: 6, fault: 5, warning: 4, unknown: 3, watch: 2, offline: 1, normal: 0 };
  return [...ts].sort((a, b) => (rank[b.state] ?? 0) - (rank[a.state] ?? 0))[0].id;
}
export async function simStep(steps) {
  const f = await api.simStep(steps);
  state.fleet = f;
  await refreshSim({ fleet: false });
}
export async function simInject(body) {
  const r = await api.simInject(body);
  state.fleet = r.state;
  await refreshSim({ fleet: false });
  return r.injection;
}
export async function simRepair(id) { state.fleet = await api.simRepair(id); await refreshSim({ fleet: false }); }
export async function simReset(body) { state.fleet = await api.simReset(body); await refreshSim({ fleet: false }); }

export async function launchGuidedScenario({
  scenario = 'gearbox_bearing',
  turbine = 10,
  severity = 'moderate',
  ramp_h = 6,
  duration_h = 48,
  seed = 42,
} = {}) {
  pause();
  toast('Resetting simulation with deterministic seed…');
  await simReset({ seed });
  toast(`Injecting ${scenario} on turbine ${turbine}…`);
  const inj = await simInject({ turbine, scenario, severity, ramp_h, duration_h, seed });
  state.selected = turbine;
  play();
  toast(`Guided scenario active: ${inj?.name || scenario}. Simulation playing!`);
  return inj;
}

// ---------------------------------------------------------------- replay
export async function loadDatasets() {
  if (!state.datasets) state.datasets = await api.datasets();
  return state.datasets;
}
export async function loadReplay(id, { cursor = null } = {}) {
  pause();
  const r = prepareReplay(await api.replay(id));
  if (r._mock_substituted_for != null && String(r._mock_substituted_for) !== String(r.id)) {
    toast(`Mock mode: no replay fixture for dataset ${r._mock_substituted_for}; showing dataset ${r.id}.`);
  }
  // start just before the prediction window / a few days before the logged event if possible
  let c = cursor;
  if (c == null) {
    const ev = Date.parse(r.event_start);
    const target = Number.isFinite(ev) ? ev - 6 * 86400e3 : r._prep.tms[0];
    c = Math.max(0, r._prep.tms.findIndex((x) => x >= target));
  }
  setState({ replay: r, replayId: r.id, cursor: c, selected: r.asset }, 'replay');
}
export function setCursor(i) {
  const r = state.replay; if (!r) return;
  setState({ cursor: Math.max(0, Math.min(r.t.length - 1, i)) }, 'data');
}

// ---------------------------------------------------------------- mode
export async function ensureModeData() {
  try {
    if (state.mode === 'replay') {
      if (!state.replay) {
        const ds = await loadDatasets();
        const id = state.replayId ?? (ds.find((d) => d.label === 'anomaly') || ds[0])?.id ?? 0;
        await loadReplay(id);
      }
    } else if (!state.fleet) {
      await refreshSim();
    }
  } catch (e) {
    console.error(e); toast(`Could not load ${state.mode} data: ${e.message}`, 'err');
  }
}
export async function setMode(mode) {
  if (mode === state.mode) return;
  pause();
  setState({ mode, selected: null, speed: SPEEDS[mode][0] }, 'mode');
  await ensureModeData();
  if (mode === 'simulation' && state.fleet) state.selected = pickDefault(state.fleet.turbines);
  if (mode === 'replay' && state.replay) state.selected = state.replay.asset;
  emit('mode');
}

// ---------------------------------------------------------------- playback
export function play() {
  if (state.playing) return;
  setState({ playing: true }, 'playback');
  timer = setInterval(tick, TICK_MS[state.mode]);
}
export function pause() {
  if (timer) clearInterval(timer); timer = null;
  if (state.playing) setState({ playing: false }, 'playback');
}
export function togglePlay() { (state.playing ? pause : play)(); }
export function setSpeed(s) { setState({ speed: Number(s) }, 'playback'); }

async function tick() {
  if (state.mode === 'replay') {
    const r = state.replay; if (!r) return pause();
    if (state.cursor >= r.t.length - 1) { pause(); toast('End of replay window reached.'); return; }
    setCursor(state.cursor + state.speed);
    return;
  }
  if (inflight) return;
  inflight = true;
  try { await simStep(state.speed); } catch (e) { pause(); toast(`Simulation step failed: ${e.message}`, 'err'); } finally { inflight = false; }
}
export async function stepOnce(dir = 1) {
  if (state.mode === 'replay') return setCursor(state.cursor + dir);
  if (dir < 0) return; // the simulation cannot step backwards
  try { await simStep(1); } catch (e) { toast(`Simulation step failed: ${e.message}`, 'err'); }
}

export function selectTurbine(id) { setState({ selected: id }, 'select'); }
