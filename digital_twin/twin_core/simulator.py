"""
Wind farm simulator + live digital twin (5 turbines)
====================================================
PHYSICAL side (simulated): farm weather -> each turbine's power, speeds, yaw, pitch and ~30 temperatures
from generator models trained on that real turbine (odd days) + realistic noise; injected anomalies
change these signals (temperature rise, pitch / yaw offset, power loss, trips).

DIGITAL side (the twin): exactly the project's monitoring chain on every new 10-minute row
  1. normal-behaviour models (fault_detection_v2, trained on even days) -> residual z-scores
  2. component scores (6 h and 24 h, relative to each component's normal level)
  3. RANDOM FOREST anomaly classifier (Step 1, approach C, 175 features) -> class + confidence,
     6-h averaged, alarm after 3 h of the same fault class
  4. residual detector (cross-check): highest component score above 2.75 for 3 h -> component
  5. expected power from the forecasting (power) model -> lost energy
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
os.environ.setdefault("OMP_NUM_THREADS", "1")          # tiny batches: single-thread prediction is faster
import warnings
warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd
import sklearn

sklearn.set_config(assume_finite=True, skip_parameter_validation=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin_core import paths                                   # noqa: E402
import fault_detection_v2 as fd                               # noqa: E402
import anomaly_classifier as ac                               # noqa: E402

STEPS_PER_HOUR = 6
P, W = fd.POWER, fd.WIND
DET_THRESHOLD, DET_HOURS = 2.75, 3          # residual detector (fault_detection_v2 cross-validated setting)
PERF_HOURS = 24                             # energy performance index window
TRIP_HOURS = 6                              # protection trips a severe overheating after 6 h at full level
RF_HOURS, RF_SMOOTH_H = 3, 6                # classifier: 6-h averaged probabilities, alarm after 3 h
HIST = 320                                  # rows kept for rolling features (> 24 h + lags)

SCENARIOS = {"Normal weather": (1.0, 0.0, 0.0), "Windy": (1.45, 0.0, -2.0), "Calm": (0.55, 0.0, 2.0),
             "Storm (very high wind)": (1.0, 9.0, -4.0), "Hot day": (0.9, 0.0, 12.0), "Cold day": (1.1, 0.0, -12.0)}

# ------------------------------------------------------------------ the 12 injectable anomalies
# cls: class the Random Forest was trained on (None = not one of the 4 recorded fault types)
# temp: {sensor: weight} rise in multiples of that sensor's normal noise; pitch: deg; yaw: deg; loss: power fraction
ANOMALIES = {
    "Gearbox HSS bearing wear":         dict(cls="gearbox", comp="Gearbox", temp={"sensor_11_avg": 1.0, "sensor_12_avg": 0.35},
                                             cause="Wear / pitting of the high-speed-shaft bearing, poor lubrication"),
    "Gearbox oil overheating":          dict(cls="gearbox", comp="Gearbox", temp={"sensor_12_avg": 1.0, "sensor_11_avg": 0.6},
                                             cause="Oil cooler or filter blocked, degraded / low oil"),
    "Generator bearing damage (DE)":    dict(cls="generator_bearing", comp="Generator bearings", temp={"sensor_13_avg": 1.0, "sensor_14_avg": 0.25},
                                             cause="Drive-end bearing wear, lubrication failure, shaft currents"),
    "Generator bearing damage (NDE)":   dict(cls="generator_bearing", comp="Generator bearings", temp={"sensor_14_avg": 1.0, "sensor_13_avg": 0.25},
                                             cause="Non-drive-end bearing wear or misalignment"),
    "Transformer phase overheating":    dict(cls="transformer", comp="Transformer", temp={"sensor_39_avg": 1.0, "sensor_38_avg": 0.2},
                                             cause="Loose connection / insulation problem on one phase"),
    "Transformer cooling failure":      dict(cls="transformer", comp="Transformer", temp={"sensor_38_avg": 1.0, "sensor_39_avg": 1.0, "sensor_40_avg": 1.0},
                                             cause="Transformer cell ventilation or fan failure"),
    "Hydraulic oil overheating":        dict(cls="hydraulic", comp="Hydraulic group", temp={"sensor_41_avg": 1.0}, pitch=0.15,
                                             cause="Hydraulic pump running too often (internal leakage), cooler blocked"),
    "Hydraulic pressure loss (pitch lag)": dict(cls="hydraulic", comp="Hub / pitch", temp={"sensor_41_avg": 0.4, "sensor_6_avg": 0.3}, pitch=1.0, loss=0.25,
                                             cause="Oil leak or accumulator pre-charge lost -> pitch responds slowly"),
    "Pitch misalignment":               dict(cls=None, comp="Hub / pitch", pitch=1.0, loss=0.4,
                                             cause="Blade pitch offset (encoder / calibration error)"),
    "Yaw misalignment":                 dict(cls=None, comp="Power performance", yaw=1.0,
                                             cause="Wind-vane offset or yaw controller fault -> rotor not facing the wind"),
    "Converter cooling failure":        dict(cls=None, comp="Converter", temp={"sensor_35_avg": 1.0, "sensor_36_avg": 1.0, "sensor_37_avg": 1.0, "sensor_21_avg": 0.6},
                                             cause="Converter coolant pump / fan failure"),
    "Blade icing / soiling":            dict(cls=None, comp="Power performance", loss=1.0,
                                             cause="Ice or dirt on the blades reduces aerodynamic efficiency"),
}
SEVERITY = {"Mild": dict(sigma=5.0, pitch=1.5, yaw=8.0, loss=0.06),
            "Moderate": dict(sigma=17.0, pitch=3.0, yaw=15.0, loss=0.15),
            "Severe": dict(sigma=25.0, pitch=6.0, yaw=30.0, loss=0.30)}
DEVELOPMENT = {"Sudden": 0, "Grows over 1 day": 24, "Grows over 3 days": 72}
CLASS_LABEL = {"normal": "Normal", "gearbox": "Gearbox", "generator_bearing": "Generator bearing",
               "hydraulic": "Hydraulic group", "transformer": "Transformer"}
COMP_TO_CLASS = {"Gearbox": "gearbox", "Generator bearings": "generator_bearing", "Transformer": "transformer",
                 "Hydraulic group": "hydraulic", "Hub / pitch": "hydraulic"}


def wrap(a):
    return ((np.asarray(a) + 180) % 360) - 180


# ------------------------------------------------------------------ weather
class FarmWeather:
    """One wind field for the whole farm (persistent AR(1) + daily cycle) + small per-turbine differences."""
    def __init__(self, models, seed=1, scenario="Normal weather", start=None):
        self.m = models
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario
        any_m = next(iter(models.values()))
        self.wind_mean = float(np.mean([mm["wind_mean"] for mm in models.values()]))
        self.wind_std = float(np.mean([mm["wind_std"] for mm in models.values()]))
        self.amb_mean = float(np.mean([mm["amb_mean"] for mm in models.values()]))
        self.t = pd.Timestamp(start or any_m["start"]).floor("10min") + pd.Timedelta(days=1)
        self.wind, self.amb_noise, self.dir = self.wind_mean, 0.0, 220.0
        self.local = {a: 0.0 for a in models}

    def step(self, n):
        fac, extra, amb_off = SCENARIOS.get(self.scenario, SCENARIOS["Normal weather"])
        mu, sd, phi = self.wind_mean * fac + extra, max(self.wind_std * (1.2 if extra else 1.0), 0.5), 0.985
        out = {a: [] for a in self.m}
        for i in range(n):
            ts = self.t + pd.Timedelta(minutes=10 * i)
            hour = ts.hour + ts.minute / 60
            self.wind = mu + phi * (self.wind - mu) + self.rng.normal(0, sd * np.sqrt(1 - phi ** 2))
            self.amb_noise = 0.97 * self.amb_noise + self.rng.normal(0, 0.25)
            self.dir = (self.dir + self.rng.normal(0, 3)) % 360
            amb = self.amb_mean + amb_off + 4.0 * np.sin(2 * np.pi * (hour - 9) / 24) + self.amb_noise
            base = self.wind + 0.6 * np.sin(2 * np.pi * (hour - 9) / 24)
            for a in self.m:
                self.local[a] = 0.9 * self.local[a] + self.rng.normal(0, 0.15)
                out[a].append({"time_stamp": ts, W: max(0.0, base * (1 + 0.03 * self.local[a])),
                               fd.AMBIENT: amb + self.rng.normal(0, 0.1), fd.WIND_DIR: (self.dir + self.rng.normal(0, 2)) % 360})
        self.t = self.t + pd.Timedelta(minutes=10 * n)
        return {a: pd.DataFrame(v) for a, v in out.items()}


# ------------------------------------------------------------------ one turbine (physical + twin)
class SimTurbine:
    def __init__(self, m, power_model=None, rf=None, seed=1, levels_known=True, power_sigma=None):
        self.m, self.pm, self.rf = m, power_model, rf
        if rf is not None and hasattr(rf.get("model"), "n_jobs"):
            rf["model"].set_params(n_jobs=1)                 # no thread pool for a few rows
        self.asset = m["asset"]
        self.rng = np.random.default_rng(seed)
        self.levels_known = levels_known and bool(m.get("levels"))
        self.H = pd.DataFrame()               # history: raw + derived + z + component columns
        self.out = pd.DataFrame()             # rows shown in the app (last 30 days)
        self.raw_fast, self.raw_slow = {}, {}
        self.noise = {k: 0.0 for k in m["noise"]}
        self.nacelle = 220.0
        self.crit = 0
        self.rf_run, self.rf_last = 0, "normal"
        self.prob_hist = pd.DataFrame()
        self.faults = []                      # active injections touching this turbine
        self.tripped = False
        self.full_since = None
        self.power_sigma = power_sigma or 0.03
        self.perf_hist = pd.DataFrame(columns=["a", "e"])
        self.perf_run = 0
        self.perf_samples = []                # used during calibration

    # ---- helpers
    def _noise(self, key, n):
        phi, s = self.m["noise"][key]
        e, out = self.noise.get(key, 0.0), np.empty(n)
        for i in range(n):
            e = phi * e + self.rng.normal(0, s)
            out[i] = e
        self.noise[key] = e
        return out

    def _gen(self, key, F, idx):
        X = F.loc[idx, self.m["gen_inputs"][key]].to_numpy(dtype=float)
        return self.m["gen"][key].predict(X) + self._noise(key, len(idx))

    def _levels(self, ts):
        """Current level (0..1) of every active injection at each timestamp."""
        out = []
        for f in self.faults:
            lv = np.array([0.0 if (t < f["start"] or t >= f["end"]) else
                           (1.0 if f["ramp_h"] == 0 else min(1.0, (t - f["start"]).total_seconds() / 3600 / f["ramp_h"]))
                           for t in ts])
            out.append((f, lv))
        return out

    # ---- one batch of new 10-minute rows
    def step(self, weather):
        n = len(weather)
        hist = self.H.tail(HIST) if len(self.H) else pd.DataFrame()
        F = pd.concat([hist, weather], ignore_index=True)
        idx = F.index[len(hist):]
        ts = F.loc[idx, "time_stamp"].tolist()
        lvls = self._levels(ts)
        F["wind_cubed"] = F[W].clip(lower=0) ** 3
        F["wind_std_1h"] = F[W].rolling(STEPS_PER_HOUR, min_periods=2).std()
        r = np.deg2rad(F[fd.WIND_DIR] % 360)
        F["dir_sin"], F["dir_cos"] = np.sin(r), np.cos(r)
        # -- yaw (nacelle follows the wind; a yaw fault holds an offset)
        yaw_off = np.zeros(n)
        loss = np.zeros(n)
        pitch_off = np.zeros(n)
        for f, lv in lvls:
            sev = SEVERITY[f["severity"]]
            spec = ANOMALIES[f["type"]]
            yaw_off += spec.get("yaw", 0) * sev["yaw"] * lv
            loss += spec.get("loss", 0) * sev["loss"] * lv
            pitch_off += spec.get("pitch", 0) * sev["pitch"] * lv
        nac = np.empty(n)
        for i, wd in enumerate(F.loc[idx, fd.WIND_DIR].to_numpy()):
            err = float(wrap(wd - self.nacelle))
            if abs(err) > 8:
                self.nacelle = (self.nacelle + err) % 360
            nac[i] = (self.nacelle + yaw_off[i]) % 360
        F.loc[idx, "sensor_42_avg"] = nac
        F.loc[idx, "sensor_2_avg"] = wrap(F.loc[idx, fd.WIND_DIR].to_numpy() - nac)
        # -- power
        p = self._gen(P, F, idx)
        p = np.clip(p, -0.006, self.m["p_max"] * 1.01)
        aero = np.cos(np.deg2rad(np.clip(np.abs(wrap(yaw_off)), 0, 89))) ** 3
        p = np.where(p > 0, p * aero * (1 - np.clip(loss, 0, 0.9)), p)
        p[F.loc[idx, W].to_numpy() > self.m["cut_out"]] = -0.003
        # trips: severe temperature faults stop the turbine after TRIP_HOURS at full level (turbine protection)
        trip = np.zeros(n, bool)
        for i, t in enumerate(ts):
            hot = any(lv[i] >= 1.0 and f["severity"] == "Severe" and ANOMALIES[f["type"]].get("temp") for f, lv in lvls)
            if hot:
                self.full_since = self.full_since or t
                if (t - self.full_since).total_seconds() >= TRIP_HOURS * 3600:
                    self.tripped = True
            else:
                self.full_since = None
                if self.tripped and not any(lv[i] > 0 for _, lv in lvls):
                    self.tripped = False
            trip[i] = self.tripped
        p[trip] = -0.003
        F.loc[idx, P] = p
        F["power_mean_1h"] = F[P].rolling(STEPS_PER_HOUR, min_periods=1).mean()
        running = p > 0.005 * self.m["p_max"]
        for k in (fd.ROTOR_RPM, fd.GEN_RPM):
            if k in self.m["gen"]:
                F.loc[idx, k] = np.clip(self._gen(k, F, idx), 0, None) * running
        for k in ("sensor_23_avg", "sensor_24_avg", "sensor_25_avg"):
            if k in self.m["gen"]:
                F.loc[idx, k] = np.clip(self._gen(k, F, idx), 0, None)
        for k in ("sensor_32_avg", "sensor_33_avg", "sensor_34_avg"):
            if k in self.m["gen"]:
                F.loc[idx, k] = self._gen(k, F, idx)
        F = fd.add_features(F)
        if fd.NACELLE in self.m["gen"]:
            F.loc[idx, fd.NACELLE] = self._gen(fd.NACELLE, F, idx)
        load = np.clip(F.loc[idx, "power_mean_1h"].to_numpy() / max(self.m["p_max"], 1e-9), 0, 1)
        for t in [k for k in self.m["gen"] if k.startswith("sensor_") and k not in
                  (fd.ROTOR_RPM, fd.GEN_RPM, fd.NACELLE) and k not in
                  ("sensor_23_avg", "sensor_24_avg", "sensor_25_avg", "sensor_32_avg", "sensor_33_avg", "sensor_34_avg")] + ["pitch_dev"]:
            if t not in self.m["gen"]:
                continue
            v = self._gen(t, F, idx)
            sd = self.m["calib"].get(t, (0, 1.0))[1]
            for f, lv in lvls:
                spec, sev = ANOMALIES[f["type"]], SEVERITY[f["severity"]]
                wgt = spec.get("temp", {}).get(t)
                if wgt:
                    v = v + wgt * sev["sigma"] * sd * lv * (0.6 + 0.4 * load)
            if t == "pitch_dev":
                v = v + pitch_off
            F.loc[idx, t] = v
        F.loc[idx, "sensor_5_avg"] = F.loc[idx, "pitch_dev"] % 360
        F = fd.add_features(F)
        new = F.loc[idx].copy()
        new["tripped"] = trip
        self._twin(new, F, idx, running)

    # ---- the digital twin on the new rows
    def _twin(self, new, F, idx, running):
        z = pd.DataFrame(index=new.index)
        for t, mdl in self.m["twin"].items():
            if t not in new:
                continue
            pred = mdl.predict(new[self.m["twin_inputs"][t]].to_numpy(dtype=float))
            mu, sd = self.m["calib"][t]
            zt = (new[t].to_numpy() - pred - mu) / sd
            if t == P:
                zt = -zt
                zt[~running] = np.nan
            if t in fd.TWO_SIDED:
                zt = np.abs(zt)
            zt[new["tripped"].to_numpy()] = np.nan          # turbine stopped by protection: not judged
            z[f"z_{t}"] = zt
        ae = self.m["ae"]
        Xa = np.clip(ae["scaler"].transform(new[ae["cols"]].fillna(ae["fill"])), -10, 10)
        R = Xa - ae["model"].predict(Xa)
        top3 = np.sort(np.abs((R - ae["r_mu"]) / ae["r_sd"]), axis=1)[:, -3:].mean(axis=1)
        z["z_ae_error"] = np.where(new["tripped"].to_numpy(), np.nan, (np.log(top3 + 1e-6) - ae["mu"]) / ae["sd"])
        for c in z:
            new[c] = z[c].to_numpy()
        # component scores (6 h and 24 h smoothing, relative to the component's normal level)
        Z = pd.concat([self.H.tail(150)[[c for c in z.columns if c in self.H]], z], ignore_index=True) if len(self.H) else z.reset_index(drop=True)
        fast = Z.rolling(36, min_periods=12).mean().clip(lower=0).tail(len(z))
        slow = Z.rolling(144, min_periods=48).mean().clip(lower=0).tail(len(z))
        comps = {}
        for name, sens in fd.SUBSYSTEMS.items():
            cols = [f"z_{s}" for s in sens if f"z_{s}" in Z]
            if not cols:
                continue
            fmax, smax = fast[cols].max(axis=1).to_numpy(), slow[cols].max(axis=1).to_numpy()
            if self.levels_known:
                lf, ls = self.m["levels"]["fast"].get(name, 1.0), self.m["levels"]["slow"].get(name, 1.0)
                comps[name] = np.fmax(fmax / lf, smax / ls)
            else:
                comps[name] = fmax
                self.raw_fast.setdefault(name, []).append(fmax); self.raw_slow.setdefault(name, []).append(smax)
        trip_rows = new["tripped"].to_numpy()
        for c, v in comps.items():
            v = np.array(v, dtype=float)                 # writable copy
            v[trip_rows] = np.nan                        # stopped by protection: not judged
            comps[c] = v
            new[c] = v
        self.H = pd.concat([self.H, new], ignore_index=True).tail(HIST + 200).reset_index(drop=True)
        if not self.levels_known:                          # calibration run: collect healthy 24-h energy ratios
            if self.pm is not None:
                e = self._expected(new)
                ph = pd.DataFrame({"a": np.where(e > 0.05, new[P].clip(lower=0).to_numpy(), np.nan),
                                   "e": np.where(e > 0.05, e, np.nan)})
                self.perf_hist = pd.concat([self.perf_hist, ph], ignore_index=True).tail(PERF_HOURS * STEPS_PER_HOUR + len(new)).reset_index(drop=True)
                roll = self.perf_hist.rolling(PERF_HOURS * STEPS_PER_HOUR, min_periods=8 * STEPS_PER_HOUR).sum().tail(len(new))
                self.perf_samples.extend([float(x) for x in (roll["a"] / roll["e"]) if not np.isnan(x)])
            return
        # Random Forest classifier on the 175 features of Step 1
        rf_cls = np.array(["normal"] * len(new), dtype=object)
        rf_conf = np.zeros(len(new))
        if self.rf is not None:
            E = ac.engineer(self.H.tail(HIST).copy(), self.m["baseline"]).tail(len(new))
            X = E.reindex(columns=self.rf["features"]).fillna(-999)
            prob = pd.DataFrame(self.rf["model"].predict_proba(X.to_numpy(dtype=float)), columns=self.rf["classes"])
            prob[~running] = np.nan                    # classifier only judges a running turbine
            self.prob_hist = pd.concat([self.prob_hist, prob], ignore_index=True).tail(60 + len(new)).reset_index(drop=True)
            sm = self.prob_hist.rolling(RF_SMOOTH_H * STEPS_PER_HOUR, min_periods=3).mean().tail(len(new))
            ok = sm.notna().any(axis=1)
            rf_cls = np.where(ok, sm.fillna(-1).idxmax(axis=1), "normal").astype(object)
            rf_conf = sm.max(axis=1).fillna(0).to_numpy()
        rf_alarm = np.zeros(len(new), bool)
        for i, c in enumerate(rf_cls):
            if not running[i]:
                continue
            self.rf_run = self.rf_run + 1 if (c != "normal" and c == self.rf_last) else (1 if c != "normal" else 0)
            self.rf_last = c
            rf_alarm[i] = self.rf_run >= RF_HOURS * STEPS_PER_HOUR
        # residual detector (cross-check)
        cm = pd.DataFrame(comps)
        score = cm.max(axis=1).fillna(0).to_numpy()
        top = cm.fillna(-1).idxmax(axis=1).to_numpy()
        det = np.zeros(len(new), bool)
        tripped = new["tripped"].to_numpy()
        for i, s in enumerate(score):
            if tripped[i]:                                  # counter frozen while tripped (as for abnormal status)
                det[i] = self.crit >= DET_HOURS * STEPS_PER_HOUR
                continue
            self.crit = min(self.crit + 1, 2 * DET_HOURS * STEPS_PER_HOUR) if s > DET_THRESHOLD else max(self.crit - 1, 0)
            det[i] = self.crit >= DET_HOURS * STEPS_PER_HOUR
        exp_ml = self._expected(new)
        # energy performance index: actual / expected energy over the last 24 h, vs this turbine's healthy level
        ph = pd.DataFrame({"a": np.where(exp_ml > 0.05, new[P].clip(lower=0).to_numpy(), np.nan),
                           "e": np.where(exp_ml > 0.05, exp_ml, np.nan)})
        ph[tripped] = np.nan
        self.perf_hist = pd.concat([self.perf_hist, ph], ignore_index=True).tail(PERF_HOURS * STEPS_PER_HOUR + len(new)).reset_index(drop=True)
        roll = self.perf_hist.rolling(PERF_HOURS * STEPS_PER_HOUR, min_periods=8 * STEPS_PER_HOUR).sum().tail(len(new))
        ratio = (roll["a"] / roll["e"]).to_numpy()
        pf = self.m.get("perf", {"mean": 1.0, "limit": 0.9})
        perf_index = ratio / pf["mean"]
        perf_alarm = np.zeros(len(new), bool)
        for i, v in enumerate(perf_index):
            if tripped[i] or np.isnan(v):
                perf_alarm[i] = self.perf_run >= DET_HOURS * STEPS_PER_HOUR
                continue
            self.perf_run = min(self.perf_run + 1, 2 * DET_HOURS * STEPS_PER_HOUR) if v < pf["limit"] else max(self.perf_run - 1, 0)
            perf_alarm[i] = self.perf_run >= DET_HOURS * STEPS_PER_HOUR
        self.perf_samples.extend([float(x) for x in ratio if not np.isnan(x)])
        top = np.where(perf_alarm & ~det, "Power performance", top)
        det = det | perf_alarm
        status = np.where(rf_alarm | det | new["tripped"].to_numpy(), "fault",
                          np.where((score > 0.6 * DET_THRESHOLD) | (rf_cls != "normal"), "warning", "normal"))
        o = pd.DataFrame({
            "time_stamp": new["time_stamp"].to_numpy(), "turbine": self.asset,
            "wind": new[W].to_numpy(), "wind_dir": new[fd.WIND_DIR].to_numpy(), "nacelle_dir": new["sensor_42_avg"].to_numpy(),
            "yaw_error": new["sensor_2_avg"].to_numpy(), "ambient": new[fd.AMBIENT].to_numpy(),
            "power": new[P].to_numpy(), "expected": exp_ml, "rotor_rpm": new[fd.ROTOR_RPM].to_numpy(),
            "gen_rpm": new[fd.GEN_RPM].to_numpy(), "pitch": new["pitch_dev"].to_numpy(),
            "nacelle_temp": new[fd.NACELLE].to_numpy(), "gearbox_bearing": new["sensor_11_avg"].to_numpy(),
            "gearbox_oil": new["sensor_12_avg"].to_numpy(), "gen_bearing_de": new["sensor_13_avg"].to_numpy(),
            "gen_bearing_nde": new["sensor_14_avg"].to_numpy(), "stator": new[["sensor_15_avg", "sensor_16_avg", "sensor_17_avg"]].mean(axis=1).to_numpy(),
            "transformer": new[["sensor_38_avg", "sensor_39_avg", "sensor_40_avg"]].max(axis=1).to_numpy(),
            "hydraulic_oil": new["sensor_41_avg"].to_numpy(), "converter": new[["sensor_35_avg", "sensor_36_avg", "sensor_37_avg"]].max(axis=1).to_numpy(),
            "running": running, "tripped": new["tripped"].to_numpy(),
            "rf_class": rf_cls, "rf_conf": rf_conf, "rf_alarm": rf_alarm,
            "det_score": score, "det_component": top, "det_alarm": det, "status": status,
            "perf_index": perf_index, "perf_alarm": perf_alarm,
            "injected": ", ".join(f["type"] for f in self.faults if f["end"] > new["time_stamp"].iloc[0]) or "",
        })
        for c in comps:
            o[f"comp_{c}"] = comps[c]
        self.out = pd.concat([self.out, o], ignore_index=True).tail(30 * 24 * STEPS_PER_HOUR).reset_index(drop=True)

    def _expected(self, new):
        if not self.pm:
            return np.full(len(new), np.nan)
        R, PSTD, RHO = 287.05, 101325.0, 1.225
        w, t = new[W].to_numpy(), new[fd.AMBIENT].to_numpy()
        rho = PSTD / (R * (t + 273.15))
        tsr = pd.to_datetime(new["time_stamp"])
        hour, doy = (tsr.dt.hour + tsr.dt.minute / 60).to_numpy(), tsr.dt.dayofyear.to_numpy()
        prevw = self.H[W].iloc[-len(new) - 1] if len(self.H) > len(new) else w[0]
        f = {"wind": w, "wind2": w ** 2, "wind3": w ** 3, "wind_norm": w * (rho / RHO) ** (1 / 3),
             "wind_std_1h": new["wind_std_1h"].fillna(0.5).to_numpy(), "wind_change": np.diff(np.concatenate([[prevw], w])),
             "air_density": rho, "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
             "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25), "temp": t,
             "dir_sin": new["dir_sin"].to_numpy(), "dir_cos": new["dir_cos"].to_numpy()}
        X = np.column_stack([np.asarray(f[c], dtype=float) for c in self.pm["features"]])
        e = np.clip(self.pm["model"].predict(X), 0, self.pm["rated_power"] * 1.02)
        e[new[W].to_numpy() > self.m["cut_out"]] = 0.0
        return e

    def raw_component_history(self):
        return (pd.DataFrame({k: np.concatenate(v) for k, v in self.raw_fast.items()}),
                pd.DataFrame({k: np.concatenate(v) for k, v in self.raw_slow.items()}))


# ------------------------------------------------------------------ the farm
class FarmSimulator:
    def __init__(self, models, power_model=None, rf=None, power_sigma=None, scenario="Normal weather", seed=1,
                 rated_mw=2.0, warmup_hours=30):
        self.models = models
        self.weather = FarmWeather(models, seed=seed, scenario=scenario)
        self.turbines = {a: SimTurbine(m, power_model, rf, seed=seed * 100 + a, power_sigma=power_sigma)
                         for a, m in models.items()}
        self.rated_mw = rated_mw
        self.injections = []                 # every injected anomaly (active and finished)
        self.events = []                     # detections (incl. ones without an injection = false alarms)
        self.energy = {a: {"actual": 0.0, "expected": 0.0} for a in models}
        self.next_id = 1
        self._last_alarm = {a: {"rf": False, "det": False, "trip": False} for a in models}
        if warmup_hours:
            self.step(int(warmup_hours * STEPS_PER_HOUR))

    @property
    def now(self):
        return self.weather.t

    def set_scenario(self, s):
        self.weather.scenario = s

    def inject(self, turbine, kind, severity="Moderate", development="Grows over 1 day", duration_h=48):
        f = dict(id=self.next_id, turbine=turbine, type=kind, severity=severity, development=development,
                 ramp_h=DEVELOPMENT[development], duration_h=duration_h, start=self.now,
                 end=self.now + pd.Timedelta(hours=duration_h), expected_class=ANOMALIES[kind]["cls"],
                 expected_component=ANOMALIES[kind]["comp"], cause=ANOMALIES[kind]["cause"],
                 rf_detected=None, rf_class=None, rf_conf=None, det_detected=None, det_component=None, trip_time=None,
                 energy_lost_mwh=0.0, tripped=False)
        self.next_id += 1
        self.injections.append(f)
        self.turbines[turbine].faults.append(f)
        return f

    def stop(self, fid):
        for f in self.injections:
            if f["id"] == fid and f["end"] > self.now:
                f["end"] = self.now

    def step(self, n=6):
        w = self.weather.step(n)
        for a, tb in self.turbines.items():
            tb.step(w[a])
            self._account(a, tb.out.tail(n))

    def _account(self, a, rows):
        h = 1.0 / STEPS_PER_HOUR
        act = rows["power"].clip(lower=0).to_numpy() * self.rated_mw * h
        exp = rows["expected"].fillna(0).to_numpy() * self.rated_mw * h
        self.energy[a]["actual"] += float(act.sum())
        self.energy[a]["expected"] += float(exp.sum())
        active = [f for f in self.injections if f["turbine"] == a and f["start"] <= rows["time_stamp"].iloc[-1]
                  and (f["end"] > rows["time_stamp"].iloc[0] or f.get("_tripping"))]
        for f in active:
            m = (rows["time_stamp"] >= f["start"]).to_numpy() & ((rows["time_stamp"] < f["end"]).to_numpy() | rows["tripped"].to_numpy())
            # lost energy vs THIS turbine's healthy level (its normal performance index), signed sum over time
            base = exp * self.models[a].get("perf", {}).get("mean", 1.0)
            f["_signed_mwh"] = f.get("_signed_mwh", 0.0) + float((base - act)[m].sum())
            f["energy_lost_mwh"] = max(0.0, f["_signed_mwh"])
            if rows["tripped"].to_numpy()[m].any():
                f["tripped"] = True
                if f["trip_time"] is None:
                    f["trip_time"] = rows.loc[m & rows["tripped"].to_numpy(), "time_stamp"].iloc[0]
            fa = rows[m & rows["rf_alarm"].to_numpy()]
            if f["rf_detected"] is None and len(fa):
                f["rf_detected"], f["rf_class"], f["rf_conf"] = fa["time_stamp"].iloc[0], fa["rf_class"].iloc[0], float(fa["rf_conf"].iloc[0])
            fd_ = rows[m & rows["det_alarm"].to_numpy()]
            if f["det_detected"] is None and len(fd_):
                f["det_detected"], f["det_component"] = fd_["time_stamp"].iloc[0], fd_["det_component"].iloc[0]
        # detection events (rising edges), labelled with the injection they belong to (if any)
        for kind, col in (("Random Forest", "rf_alarm"), ("Residual detector", "det_alarm"), ("Protection trip", "tripped")):
            key = {"rf_alarm": "rf", "det_alarm": "det", "tripped": "trip"}[col]
            vals = rows[col].to_numpy()
            for i, v in enumerate(vals):
                if v and not self._last_alarm[a][key]:
                    t = rows["time_stamp"].iloc[i]
                    inj = next((f for f in self.injections if f["turbine"] == a and f["start"] <= t < f["end"] + pd.Timedelta(hours=6)), None)
                    r = rows.iloc[i]
                    self.events.append(dict(time=t, turbine=a, detector=kind,
                                            result=CLASS_LABEL.get(r["rf_class"], r["rf_class"]) if key == "rf" else
                                            (r["det_component"] if key == "det" else f"turbine stopped (twin suspects {r['det_component']})"),
                                            confidence=float(r["rf_conf"]) if key == "rf" else float(r["det_score"]),
                                            injected=inj["type"] if inj else "", injection_id=inj["id"] if inj else None,
                                            cause=inj["cause"] if inj else ANOMALIES.get(next((k for k, v in ANOMALIES.items()
                                                  if v["comp"] == r["det_component"]), ""), {}).get("cause", "Unknown - check the component")))
                self._last_alarm[a][key] = bool(v)

    def frame(self, turbine):
        return self.turbines[turbine].out

    def latest(self):
        return {a: tb.out.iloc[-1] for a, tb in self.turbines.items() if len(tb.out)}
