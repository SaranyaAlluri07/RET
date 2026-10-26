// Global application state + a minimal pub/sub. Views subscribe to change reasons.
//   reasons: 'data' (time advanced / new fleet state), 'mode', 'theme', 'rated', 'select', 'replay', 'playback'
import { lsGet, lsSet } from './util.js';

export const state = {
  meta: null,
  mock: false,
  // 'replay' | 'simulation' - never 'live'. ?mode=replay|simulation in the URL overrides the stored choice.
  mode: ['replay', 'simulation'].includes(new URLSearchParams(location.search).get('mode'))
    ? new URLSearchParams(location.search).get('mode') : lsGet('windtwin.mode', 'simulation'),
  theme: document.documentElement.getAttribute('data-theme') || 'dark',
  rated: lsGet('windtwin.rated', null),          // illustrative MW per turbine (user-entered)
  selected: null,                                // selected turbine id
  playing: false,
  speed: 1,                                      // steps per tick
  // simulation
  fleet: null, simCases: [], injections: [], scenarios: null,
  // replay
  datasets: null, replayId: lsGet('windtwin.replayId', null), replay: null, cursor: 0,
};

const subs = new Set();
export function subscribe(fn) { subs.add(fn); return () => subs.delete(fn); }
export function emit(reason) { for (const fn of [...subs]) { try { fn(reason); } catch (e) { console.error(e); } } }
export function setState(patch, reason = 'state') {
  Object.assign(state, patch);
  if ('mode' in patch) lsSet('windtwin.mode', state.mode);
  if ('rated' in patch) lsSet('windtwin.rated', state.rated);
  if ('replayId' in patch) lsSet('windtwin.replayId', state.replayId);
  emit(reason);
}
