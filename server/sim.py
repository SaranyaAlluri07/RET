"""Synthetic simulation session: physics (server/physics.py) -> injected scenarios (windtwin.scenarios)
-> the same Monitor that runs on the real-data replay.

Ground truth of every injection is kept apart from the operator outputs: it is only reported in
/api/sim/injections (test oracle) and in Case.truth.
"""
import numpy as np
import pandas as pd

import copy

from windtwin import config as C, scenarios, data, quality, residuals, alarms
from windtwin.monitor import Monitor, load_bundle, score_frame, LAMBDAS
from server import physics as ph

ASSETS = {0: (-520, -160), 10: (-260, 120), 11: (0, -120), 13: (260, 140), 21: (520, -110)}
KEEP = 30 * 24 * C.STEPS_PER_HOUR          # rows kept per turbine (30 days)
WARMUP = C.WARMUP_STEPS
SEVERITIES = ("mild", "moderate", "severe")


class FarmSim:
    def __init__(self, seed=7, weather="normal"):
        self.gens = {a: ph.load_generator(a, C.DATASETS_DIR / f"{C.SIM_DATASETS[a]}.csv") for a in ASSETS}
        self.bundles = {a: load_bundle(C.SIM_DATASETS[a]) for a in ASSETS}
        self.reset(seed, weather)

    # ------------------------------------------------------------------ session
    def reset(self, seed=None, weather=None):
        self.seed = 7 if seed is None else int(seed)
        self.weather = ph.FarmWeather(self.gens, seed=self.seed, scenario=weather or "normal")
        self.phys = {a: ph.PhysicalTurbine(g, seed=self.seed * 100 + a) for a, g in self.gens.items()}
        for a, t in self.phys.items():
            t.power_fn = self.bundles[a].models[C.POWER].predict
        self.mon = {a: Monitor(b, params=b.params_global, turbine=a, source="simulation")
                    for a, b in self.bundles.items()}
        self.rows = {a: [] for a in ASSETS}
        self.case_list = {}
        self.injections = []
        self.energy = {"actual": 0.0, "expected": 0.0, "lost": 0.0}
        self.next_id = 1
        # warm-up: history for rolling features / flatline checks, no alarm logic
        w = self.weather.step(WARMUP)
        for a in ASSETS:
            self.mon[a].prime(self.phys[a].step(w[a]))
        self._calibrate()
        self.step(6)
        return self.state()

    def _commission(self, a, hist, calib):
        """Twin commissioning on the simulated plant: re-estimate the residual normalization and EWMA scales
        on healthy simulated days (as build_bundle does on the real validation split). The generator models
        signals one by one, so their cross-correlations differ from the real turbine's; the real-data
        bundles used by replay and the benchmark are not changed."""
        b = copy.deepcopy(self.bundles[a])
        df = data.prepare(pd.concat([hist, calib], ignore_index=True))
        sc = score_frame(b, df)
        lo = len(hist)
        R, pw, am, ok = sc["R"][lo:], sc["power"][lo:], sc["ambient"][lo:], sc["valid"][lo:]
        b.normalizer = residuals.RegimeNormalizer(b.targets).fit(R, pw, am, ok)
        Z = b.normalizer.transform(R, pw, am)
        Z[~ok] = np.nan
        b.scales = {lam: alarms.ewma_scale(Z, lam) for lam in LAMBDAS}
        return b

    def _calibrate(self, days=2):
        """Healthy run: remove each raw signal's median offset vs the twin's expectation, so the synthetic
        plant and the twin agree before any injection (generator and NBM are different regressions)."""
        w = self.weather.step(days * 24 * C.STEPS_PER_HOUR)
        for a in ASSETS:
            df = self.phys[a].step(w[a])
            rows, _ = self.mon[a].update(df)
            # power: per-wind-bin gain so the synthetic power curve matches the twin's expectation
            pr = [(r["wind"], r["power"], r["signals"][C.POWER][1]) for r in rows if C.POWER in r["signals"]
                  and r["operating_state"] == "production" and None not in (r["wind"], r["power"], r["signals"][C.POWER][1])]
            if len(pr) > 50:
                wv, po, pe = map(np.array, zip(*pr))
                bins = np.arange(0, 26, 1.0)
                g = [np.median(pe[(wv >= b) & (wv < b + 1)] / np.clip(po[(wv >= b) & (wv < b + 1)], 1e-3, None))
                     if ((wv >= b) & (wv < b + 1) & (po > 0.02)).sum() >= 8 else np.nan for b in bins]
                g = pd.Series(g).clip(0.7, 1.3).interpolate(limit_direction="both").fillna(1.0).to_numpy()
                self.phys[a].p_gain = (bins + 0.5, g)
            for t in self.bundles[a].targets:
                if t == C.POWER:
                    continue
                raw = C.PITCH_RAW if t == "pitch" else t
                if raw not in self.phys[a].m["gen"] and t != "pitch":
                    continue
                d = np.array([r["signals"][t][0] - r["signals"][t][1] for r in rows
                              if t in r["signals"] and None not in r["signals"][t][:2]
                              and r["operating_state"] == "production"])
                if len(d) > 30:
                    key = "pitch_dev" if t == "pitch" else t
                    self.phys[a].offset[key] = self.phys[a].offset.get(key, 0.0) + float(np.median(d))
        w = self.weather.step(4 * 24 * C.STEPS_PER_HOUR)  # healthy commissioning days (offsets applied)
        for a in ASSETS:
            hist = self.mon[a].buf
            calib = self.phys[a].step(w[a])
            b = self._commission(a, hist, calib)
            self.mon[a] = Monitor(b, params=b.params_global, turbine=a, source="simulation")
            self.mon[a].prime(calib)                       # fresh alarm state, commissioning rows as history

    @property
    def now(self):
        return self.weather.t

    def inject(self, turbine, scenario, severity="moderate", ramp_h=24, duration_h=48, seed=None):
        if turbine not in ASSETS:
            raise KeyError(f"unknown turbine {turbine}")
        meta = scenarios.get(scenario)                     # KeyError for an unknown id
        severity = severity.lower()
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        start = self.now
        inj = {"id": self.next_id, "turbine": turbine, "scenario": scenario, "name": meta["name"],
               "category": meta["category"], "component": meta["component"], "severity": severity,
               "ramp_h": float(ramp_h), "duration_h": float(duration_h),
               "seed": int(self.seed * 1000 + self.next_id if seed is None else seed),
               "start": start, "end": start + pd.Timedelta(hours=float(duration_h)),
               "detected_time": None, "detected_component": None, "lost_energy_pu_h": 0.0}
        self.next_id += 1
        self.injections.append(inj)
        return self._inj_out(inj)

    def repair(self, injection_id):
        for f in self.injections:
            if f["id"] == injection_id and f["end"] > self.now:
                f["end"] = self.now
        return self.state()

    # ------------------------------------------------------------------ stepping
    def step(self, n=6):
        w = self.weather.step(int(n))
        for a in ASSETS:
            df = self.phys[a].step(w[a])
            healthy_power = df[C.POWER].to_numpy(float).copy()
            active = [f for f in self.injections if f["turbine"] == a and f["start"] <= df["time_stamp"].iloc[-1]
                      and f["end"] > df["time_stamp"].iloc[0]]
            for f in active:
                df = scenarios.apply(df, self._spec(f))
            rows, cases = self.mon[a].update(df)
            self.rows[a].extend(rows)
            del self.rows[a][:-KEEP]
            self._account(a, df, rows, healthy_power, active)
            for c in cases:
                self._attach_truth(c)
                self.case_list[c["id"]] = c
        return self.state()

    def _spec(self, f):
        return {"scenario": f["scenario"], "severity": f["severity"], "ramp_h": f["ramp_h"],
                "duration_h": f["duration_h"], "seed": f["seed"], "start": f["start"], "end": f["end"]}

    def _account(self, a, df, rows, healthy_power, active):
        h = 1.0 / C.STEPS_PER_HOUR
        p = np.clip(df[C.POWER].to_numpy(float), 0, None)
        e = np.array([r["expected"] if r["expected"] is not None else 0.0 for r in rows])
        self.energy["actual"] += float(p.sum() * h)
        self.energy["expected"] += float(np.clip(e, 0, None).sum() * h)
        self.energy["lost"] += float(np.clip(e - p, 0, None).sum() * h)
        ts = df["time_stamp"]
        for f in active:
            m = ((ts >= f["start"]) & (ts < f["end"])).to_numpy()
            f["lost_energy_pu_h"] += float(np.clip(np.clip(healthy_power, 0, None) - p, 0, None)[m].sum() * h)

    def _attach_truth(self, c):
        """Link a case to the injection active on that turbine when the alarm fired (test oracle only)."""
        if c.get("truth") is not None:
            return
        t = pd.Timestamp(c["alarm_time"])
        f = next((f for f in self.injections if f["turbine"] == c["turbine"]
                  and f["start"] <= t < f["end"] + pd.Timedelta(hours=6)), None)
        if f is None:
            c["truth"] = None
            return
        delay = (t - f["start"]).total_seconds() / 3600
        c["truth"] = {"injection_id": f["id"], "scenario": f["scenario"], "delay_h": round(delay, 2)}
        if f["detected_time"] is None:
            f["detected_time"], f["detected_component"] = t, c["component"]

    # ------------------------------------------------------------------ outputs
    def _snapshot(self, a):
        r = self.rows[a][-1]
        return {"id": a, "label": f"T-{a:02d}", "t": r["t"], "source": "simulation", "state": r["state"],
                "health": r["health"], "power": r["power"], "expected": r["expected"],
                "expected_lo": r["expected_lo"], "expected_hi": r["expected_hi"], "wind": r["wind"],
                "wind_dir": r["wind_dir"], "nacelle_dir": r["nacelle_dir"], "yaw_error": r["yaw_error"],
                "rotor_rpm": r["rotor_rpm"], "pitch": r["pitch"], "ambient": r["ambient"],
                "top_component": r["top_component"], "components": r["components"], "quality": r["quality"],
                "operating_state": r["operating_state"],
                "injected": [f["name"] for f in self.injections if f["turbine"] == a and f["start"] <= self.now < f["end"]]}

    def state(self):
        snaps = [self._snapshot(a) for a in ASSETS if self.rows[a]]
        t_cut = self.now - pd.Timedelta(hours=24)
        w = snaps[0] if snaps else {}
        return {"t": self.now.isoformat(), "source": "simulation",
                "weather": {"scenario": self.weather.scenario, "wind": float(np.mean([s["wind"] or 0 for s in snaps])) if snaps else None,
                            "wind_dir": w.get("wind_dir"), "ambient": w.get("ambient")},
                "turbines": snaps,
                "farm": {"power": float(sum(max(0.0, s["power"] or 0) for s in snaps)),
                         "expected": float(sum(max(0.0, s["expected"] or 0) for s in snaps)),
                         "energy_actual": round(self.energy["actual"], 3),
                         "energy_expected": round(self.energy["expected"], 3),
                         "lost_energy": round(self.energy["lost"], 3)},
                "open_cases": sum(c["status"] != "closed" for c in self.case_list.values()),
                "alarms_24h": sum(pd.Timestamp(c["alarm_time"]) >= t_cut for c in self.case_list.values())}

    def series(self, tid, hours=48):
        if tid not in ASSETS:
            raise KeyError(tid)
        rows = self.rows[tid][-int(max(1, hours) * C.STEPS_PER_HOUR):]
        p = self.mon[tid].p
        signals = {"power": {"label": "Active power", "unit": "pu", "observed": [r["power"] for r in rows],
                             "expected": [r["expected"] for r in rows], "lo": [r["expected_lo"] for r in rows],
                             "hi": [r["expected_hi"] for r in rows]}}
        for t in self.bundles[tid].targets:
            v = [r["signals"].get(t, (None,) * 4) for r in rows]
            signals[t] = {"label": C.label_of(t), "unit": C.unit_of(t), "component": C.TARGET_COMPONENT.get(t),
                          "observed": [x[0] for x in v], "expected": [x[1] for x in v],
                          "lo": [x[2] for x in v], "hi": [x[3] for x in v]}
        names = list(dict.fromkeys(c["name"] for r in rows for c in r["components"]))
        comps = {n: {"score": [], "ewma": [], "cusum": []} for n in names}
        for r in rows:
            by = {c["name"]: c for c in r["components"]}
            for n in names:
                c = by.get(n, {})
                for k in ("score", "ewma", "cusum"):
                    comps[n][k].append(c.get(k))
        return {"id": tid, "t": [r["t"] for r in rows], "signals": signals, "components": comps,
                "state": [r["state"] for r in rows], "health": [r["health"] for r in rows],
                "quality": [r["quality"]["state"] for r in rows],
                "wind": [r["wind"] for r in rows], "ambient": [r["ambient"] for r in rows],
                "limits": {"z_band": 1.645, "ewma_warn": p.L if p.use_ewma else None, "ewma_watch": p.watch * p.L if p.use_ewma else None,
                           "cusum_h": p.h if p.use_cusum else None}}

    def cases(self):
        return sorted(self.case_list.values(), key=lambda c: c["alarm_time"], reverse=True)

    def _inj_out(self, f):
        d = None
        if f["detected_time"] is not None:
            d = round((f["detected_time"] - f["start"]).total_seconds() / 3600, 2)
        correct = None if f["detected_component"] is None else (f["detected_component"] == f["component"])
        return {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in f.items()} | {
            "active": bool(f["start"] <= self.now < f["end"]), "delay_h": d, "correct": correct,
            "detected_time": f["detected_time"].isoformat() if f["detected_time"] is not None else None,
            "lost_energy_pu_h": round(f["lost_energy_pu_h"], 4)}

    def injections_out(self):
        return [self._inj_out(f) for f in reversed(self.injections)]
