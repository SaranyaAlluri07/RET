// REST client for the WindTwin backend (docs/ARCHITECTURE.md), with a mock fallback.
//
// Fallback rule: on startup we probe GET /api/meta. If the probe fails (network error, timeout,
// non-2xx, or a non-JSON body - e.g. a static file server without the backend) or the URL contains
// ?mock=1, the whole session uses the fixtures in ./mock/ and a mocked simulation engine.
// After a successful probe, real API errors are surfaced as errors (never silently mocked).
import { MockSim } from './api-mock.js';

const params = new URLSearchParams(location.search);
const FORCE_MOCK = params.get('mock') === '1' || params.get('mock') === 'true';
const TIMEOUT_MS = 4000;

let mode = FORCE_MOCK ? 'mock' : 'unknown'; // 'api' | 'mock' | 'unknown'
let mockReason = FORCE_MOCK ? '?mock=1 in URL' : '';
let mockSim = null;
const fixtureCache = new Map();

export class ApiError extends Error {
  constructor(msg, status) { super(msg); this.status = status; }
}

async function request(path, { method = 'GET', body } = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`/api${path}`, {
      method, signal: ctl.signal,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      let detail = ''; try { detail = (await res.json()).detail || ''; } catch { /* ignore */ }
      throw new ApiError(`${method} /api${path} -> ${res.status} ${detail}`.trim(), res.status);
    }
    return await res.json();
  } finally { clearTimeout(timer); }
}

/** Load a fixture from ./mock/ (cached; returns a deep copy so callers may mutate). */
async function fixture(name) {
  if (!fixtureCache.has(name)) {
    fixtureCache.set(name, fetch(`./mock/${name}`).then((r) => {
      if (!r.ok) throw new ApiError(`mock fixture ${name} missing`, r.status);
      return r.json();
    }));
  }
  try { return structuredClone(await fixtureCache.get(name)); }
  catch (e) { fixtureCache.delete(name); throw e; }
}
async function fixtureOr(name, fallbackName) {
  try { return await fixture(name); } catch { return fixture(fallbackName); }
}

/** Decide api vs mock once. Safe to call repeatedly. */
export async function init() {
  if (mode !== 'unknown') return mode;
  try {
    await request('/meta');
    mode = 'api';
  } catch (e) {
    mode = 'mock';
    mockReason = e.name === 'AbortError' ? 'API timeout' : 'API unreachable';
  }
  return mode;
}
export const isMock = () => mode === 'mock';
export const mockInfo = () => mockReason;

async function sim() {
  if (!mockSim) {
    const [state, cases, injections, scenarios] = await Promise.all([
      fixture('sim_state.json'), fixture('sim_cases.json'), fixture('sim_injections.json'), fixture('sim_scenarios.json')]);
    mockSim = new MockSim({ state, cases, injections, scenarios });
  }
  return mockSim;
}

/** Route a call to the real API or to fixtures. */
async function call(path, mockFn, opts) {
  if (mode === 'unknown') await init();
  return mode === 'api' ? request(path, opts) : mockFn();
}

export const api = {
  meta: () => call('/meta', () => fixture('meta.json')),
  benchmark: () => call('/benchmark', () => fixture('benchmark.json')),
  benchmarkMethod: (m) => call(`/benchmark/${encodeURIComponent(m)}`,
    () => fixtureOr(`benchmark_${m}.json`, 'benchmark_nbm_ewma_cusum.json')),
  forecast: () => call('/forecast', () => fixture('forecast.json')),
  datasets: () => call('/datasets', () => fixture('datasets.json')),
  // Only dataset 0 has a replay fixture; other ids fall back to it and are flagged.
  replay: (id) => call(`/replay/${encodeURIComponent(id)}`, async () => {
    try { return await fixture(`replay_${id}.json`); } catch {
      const r = await fixture('replay_0.json'); r._mock_substituted_for = id; return r;
    }
  }),
  simState: () => call('/sim/state', async () => (await sim()).getState()),
  simReset: (body = {}) => call('/sim/reset', async () => (await sim()).reset(body), { method: 'POST', body }),
  simStep: (steps = 1) => call('/sim/step', async () => (await sim()).step(steps), { method: 'POST', body: { steps } }),
  simInject: (body) => call('/sim/inject', async () => (await sim()).inject(body), { method: 'POST', body }),
  simRepair: (injection_id) => call('/sim/repair', async () => (await sim()).repair(injection_id), { method: 'POST', body: { injection_id } }),
  simScenarios: () => call('/sim/scenarios', () => fixture('sim_scenarios.json')),
  simTurbine: (id, hours = 48) => call(`/sim/turbine/${encodeURIComponent(id)}?hours=${hours}`,
    async () => (await sim()).turbineSeries(id, await fixtureOr(`sim_turbine_${id}.json`, 'sim_turbine_0.json'))),
  simCases: () => call('/sim/cases', async () => (await sim()).getCases()),
  simInjections: () => call('/sim/injections', async () => (await sim()).getInjections()),
};
