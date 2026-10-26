// Tiny in-browser stand-in for the simulation engine, used only when the API is unreachable or ?mock=1.
// It is deliberately crude (placeholder physics, scripted detection) - it exists so the UI can be
// developed and demonstrated offline. Nothing it produces is a result.

const STEP_MS = 10 * 60 * 1000;
const iso = (v) => { const d = new Date(v); const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:00`; };
const curve = (w) => (w < 3 || w > 25 ? 0 : w >= 12.5 ? 1 : Math.min(1, ((w - 3) / 9.5) ** 2.6));

export class MockSim {
  constructor({ state, cases, injections, scenarios }) {
    this.initial = structuredClone({ state, cases, injections });
    this.scenarios = scenarios;
    this._load(state, cases, injections);
  }
  _load(state, cases, injections) {
    this.state = state; this.cases = cases; this.injections = injections;
    this.seq = 10; this.rng = mulberry32(1);
    this.offsets = Object.fromEntries(state.turbines.map((t) => [t.id, (t.wind ?? 8) - (state.weather?.wind ?? 8)]));
  }
  getState() { return structuredClone(this.state); }
  getCases() { return structuredClone(this.cases); }
  getInjections() { return structuredClone(this.injections); }

  reset({ seed } = {}) {
    const s = structuredClone(this.initial);
    this._load(s.state, s.cases, s.injections);
    if (seed != null) this.rng = mulberry32(Number(seed) || 1);
    return this.getState();
  }

  step(steps = 1) {
    const st = this.state; const n = Math.max(1, Math.min(144, Number(steps) || 1));
    for (let k = 0; k < n; k++) {
      const t = Date.parse(st.t) + STEP_MS; st.t = iso(t);
      const w = st.weather; w.wind = clamp((w.wind ?? 9) + 0.04 * (9.5 - w.wind) + this.gauss() * 0.35, 1, 20);
      w.wind_dir = (w.wind_dir + this.gauss() * 1.2 + 360) % 360;
      let fp = 0, fe = 0;
      for (const tb of st.turbines) {
        tb.t = st.t;
        tb.wind = round(Math.max(0.3, w.wind + this.offsets[tb.id] + this.gauss() * 0.25), 2);
        tb.wind_dir = round(w.wind_dir, 1);
        tb.nacelle_dir = round((tb.nacelle_dir + 0.25 * angDiff(tb.wind_dir, tb.nacelle_dir) + 360) % 360, 1);
        tb.yaw_error = round(angDiff(tb.nacelle_dir, tb.wind_dir), 1);
        const exp = curve(tb.wind);
        const perfLoss = this.injections.some((i) => i.active && i.turbine === tb.id && i.category === 'performance') ? 0.12 : 0;
        tb.expected = round(exp, 3); tb.expected_lo = round(Math.max(0, exp - 0.07), 3); tb.expected_hi = round(Math.min(1.05, exp + 0.07), 3);
        tb.power = tb.state === 'tripped' ? 0 : round(Math.max(0, exp * (1 - perfLoss) + this.gauss() * 0.02 * (exp > 0)), 3);
        tb.rotor_rpm = round(tb.power <= 0 ? 0 : 9 + 7 * Math.min(1, tb.power * 1.4), 2);
        tb.pitch = round(tb.power < 0.95 ? 0.5 : 2 + (tb.wind - 12.5) * 1.6, 1);
        if (exp < 0.005 && tb.state === 'normal') tb.state = 'offline';
        else if (exp >= 0.005 && tb.state === 'offline') tb.state = 'normal';
        fp += tb.power; fe += exp;
      }
      st.farm.power = round(fp, 3); st.farm.expected = round(fe, 3);
      st.farm.energy_actual = round(st.farm.energy_actual + fp / 6, 3);
      st.farm.energy_expected = round(st.farm.energy_expected + fe / 6, 3);
      st.farm.lost_energy = round(Math.max(0, st.farm.energy_expected - st.farm.energy_actual), 3);
      this._detect(t);
    }
    st.open_cases = this.cases.filter((c) => c.status !== 'closed').length;
    return this.getState();
  }

  /** Scripted "detection": an injection is flagged after a delay proportional to its ramp. */
  _detect(t) {
    for (const inj of this.injections) {
      if (!inj.active) continue;
      const el = (t - Date.parse(inj.start)) / 3.6e6;
      if (Date.parse(inj.end) <= t) { inj.active = false; this._clear(inj, t); continue; }
      if (inj.detected_time || el < Math.max(1, (inj.ramp_h || 0) * 0.5 + 0.5)) continue;
      const sensor = inj.category === 'sensor';
      inj.detected_time = iso(t); inj.delay_h = round(el, 2);
      inj.detected_component = sensor ? null : inj.component; inj.correct = true;
      const tb = this.state.turbines.find((x) => x.id === inj.turbine);
      if (tb) {
        tb.state = sensor ? 'unknown' : 'warning'; tb.top_component = sensor ? null : inj.component;
        tb.health = sensor ? tb.health : Math.min(tb.health, 72);
        if (sensor) tb.quality = { state: 'degraded', issues: ['channel flat-lined (mock)'] };
      }
      this.cases.unshift({
        id: `S-${++this.seq}`, turbine: inj.turbine, source: 'simulation', status: 'open',
        state: sensor ? 'unknown' : 'warning', component: sensor ? null : inj.component,
        hypothesis: sensor ? 'Unknown - insufficient evidence (data quality)' : `${inj.name} suspected`,
        deviation_start: iso(Date.parse(inj.start) + 0.5 * 3.6e6), alarm_time: iso(t), cleared_time: null,
        severity: { mild: 0.3, moderate: 0.55, severe: 0.8 }[inj.severity] ?? 0.5,
        confidence: sensor ? null : 0.6, lost_energy_pu_h: 0,
        evidence: [{ signal: 'mock', label: 'Mock evidence (placeholder)', observed: null, expected: null, z: 3.1, contribution: 1 }],
        truth: { injection_id: inj.id, scenario: inj.scenario, delay_h: inj.delay_h },
      });
      this.state.alarms_24h += 1;
    }
  }
  _clear(inj, t) {
    const tb = this.state.turbines.find((x) => x.id === inj.turbine);
    if (tb) {
      tb.injected = (tb.injected || []).filter((s) => s !== inj.scenario);
      if (!this.injections.some((i) => i.active && i.turbine === tb.id)) {
        tb.state = 'normal'; tb.top_component = null; tb.health = 96; tb.quality = { state: 'ok', issues: [] };
      }
    }
    for (const c of this.cases) if (c.truth?.injection_id === inj.id && !c.cleared_time) { c.cleared_time = iso(t); c.status = 'closed'; }
  }

  inject({ turbine, scenario, severity = 'moderate', ramp_h = 6, duration_h = 24, seed } = {}) {
    const sc = this.scenarios.find((s) => s.id === scenario);
    if (!sc) throw new Error(`unknown scenario ${scenario}`);
    const t = Date.parse(this.state.t);
    const inj = {
      id: `inj-${++this.seq}`, turbine: Number(turbine), scenario, name: sc.name, category: sc.category, component: sc.component,
      severity, ramp_h: Number(ramp_h), duration_h: Number(duration_h), seed: seed ?? null,
      start: iso(t), end: iso(t + Number(duration_h) * 3.6e6), active: true,
      detected_time: null, delay_h: null, detected_component: null, correct: null, lost_energy_pu_h: 0,
    };
    this.injections.unshift(inj);
    const tb = this.state.turbines.find((x) => x.id === inj.turbine);
    if (tb) tb.injected = [...(tb.injected || []), scenario];
    return { injection: structuredClone(inj), state: this.getState() };
  }

  repair(id) {
    const inj = this.injections.find((i) => i.id === id);
    if (inj && inj.active) { const t = Date.parse(this.state.t); inj.active = false; inj.end = iso(t); this._clear(inj, t); }
    this.state.open_cases = this.cases.filter((c) => c.status !== 'closed').length;
    return this.getState();
  }

  /** Shift a fixture series so that it ends at the current simulated time. */
  turbineSeries(id, series) {
    const shift = Date.parse(this.state.t) - Date.parse(series.t[series.t.length - 1]);
    if (shift) series.t = series.t.map((x) => iso(Date.parse(x) + shift));
    series.id = Number(id);
    return series;
  }

  gauss() { const u = this.rng() || 1e-9, v = this.rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
}

function mulberry32(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const round = (v, d) => Math.round(v * 10 ** d) / 10 ** d;
const angDiff = (a, b) => ((a - b + 540) % 360) - 180;
