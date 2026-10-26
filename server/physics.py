"""Physical side of the simulation: synthetic but realistic SCADA rows for one turbine.

Each signal comes from a regression model trained on the real turbine's healthy training period
(odd days only - the even days stay unseen by the generator) plus AR(1) noise estimated from the
model's errors. Faults are NOT modelled here: windtwin.scenarios transforms the generated rows.
Generator bundles are cached in artifacts/models/gen_<asset>.joblib.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")           # tiny batches: single-thread prediction is faster
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sklearn.set_config(assume_finite=True, skip_parameter_validation=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "anomaly_detection_model"))
import fault_detection_v2 as fd                                   # noqa: E402  (legacy feature code)

GEN_DIR = ROOT / "artifacts" / "models"
P, W = fd.POWER, fd.WIND
STEPS_PER_HOUR = 6
# order matters: drivers first, temperatures last (they use the power history)
AUX = {fd.ROTOR_RPM: [W, P, "power_mean_1h"], fd.GEN_RPM: [W, P, "power_mean_1h"],
       "sensor_23_avg": [P, "power_mean_1h", W], "sensor_24_avg": [P, "power_mean_1h", W],
       "sensor_25_avg": [P, "power_mean_1h", W], "sensor_32_avg": [P, W], "sensor_33_avg": [P, W],
       "sensor_34_avg": [P, W]}
# phases of one component share their noise (real phases move together; independent noise inflates spreads)
PHASES = [[f"sensor_{i}_avg" for i in g] for g in ((15, 16, 17), (35, 36, 37), (38, 39, 40), (23, 24, 25), (32, 33, 34))]
LEADER = {k: g[0] for g in PHASES for k in g[1:]}
TEMPS = [f"sensor_{i}_avg" for i in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
                                     35, 36, 37, 38, 39, 40, 41, 53)]


def _csv(path):
    return fd.read_csv(path, usecols=lambda c: c in ("time_stamp", "train_test", "status_type_id") or c.endswith("_avg"))


def _ar1(resid):
    """Robust AR(1) noise model so unlabelled stops do not inflate it."""
    r = pd.Series(resid).dropna()
    if len(r) < 50:
        return 0.0, float(r.std() or 1.0)
    med = r.median()
    rob = 1.4826 * (r - med).abs().median() or r.std() or 1.0
    r = r.clip(med - 3 * rob, med + 3 * rob)
    phi = float(np.clip(r.autocorr(1), 0, 0.995)) if r.std() > 0 else 0.0
    return phi, float(r.std() * np.sqrt(1 - phi ** 2))


def train_generator(asset, dataset_path, max_rows=8000, log=print):
    t0 = time.time()
    df = _csv(dataset_path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = fd.add_features(df.sort_values("time_stamp").reset_index(drop=True))
    healthy = df[(df["train_test"].astype(str).str.lower() == "train") & df["status_type_id"].isin((0, 2))]
    odd = (healthy["time_stamp"].dt.dayofyear % 2 == 1).to_numpy()
    fit_rows, noise_rows = healthy[odd], healthy[~odd]
    rng = np.random.default_rng(asset)

    def sub(d):
        return d if len(d) <= max_rows else d.iloc[np.sort(rng.choice(len(d), max_rows, replace=False))]

    m = {"asset": asset, "gen": {}, "inputs": {}, "noise": {}}
    for t in [P] + list(AUX) + ([fd.NACELLE] if fd.NACELLE in df else []) + TEMPS + ["pitch_dev"]:
        if t not in df or t in m["gen"]:
            continue
        X = [c for c in (AUX.get(t) or fd.inputs_for(t, df.columns)) if c in df and c != t]
        g, w = sub(fit_rows.dropna(subset=[t])), sub(noise_rows.dropna(subset=[t]))
        if len(g) < 300 or g[t].std() < 1e-9:
            continue
        mdl = fd.make_model().set_params(max_iter=150).fit(g[X], g[t])
        m["gen"][t], m["inputs"][t] = mdl, X
        m["noise"][t] = _ar1(w[t] - mdl.predict(w[X]))
    p_max = float(healthy[P].quantile(0.999))
    run = healthy[healthy[P] > 0.05 * p_max]
    m.update(p_max=p_max, cut_out=float(min(run[W].quantile(0.999) + 1.0, 25.0)) if len(run) else 25.0,
             wind_mean=float(healthy[W].mean()), wind_std=float(healthy[W].std()),
             amb_mean=float(healthy[fd.AMBIENT].mean()), start=str(healthy["time_stamp"].max()))
    log(f"  generator for asset {asset}: {len(m['gen'])} signal models ({time.time() - t0:.0f}s)")
    return m


def load_generator(asset, dataset_path, log=print):
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    p = GEN_DIR / f"gen_{asset}.joblib"
    try:
        return joblib.load(p)
    except Exception:                       # missing, or saved with another scikit-learn version
        m = train_generator(asset, dataset_path, log=log)
        joblib.dump(m, p, compress=3)
        return m


def wrap(a):
    return ((np.asarray(a) + 180) % 360) - 180


class PhysicalTurbine:
    """Generates healthy 10-min rows for one turbine from the farm weather."""
    HIST = 160                              # rows kept for rolling features (> 12 h + margin)

    def __init__(self, m, seed):
        self.m = m
        self.rng = np.random.default_rng(seed)
        self.noise = {k: 0.0 for k in m["noise"]}
        self.H = pd.DataFrame()
        self.nacelle = 220.0
        self.offset = {}                    # per-signal bias removed after the healthy calibration run
        self.power_fn = None                # optional weather -> power curve (the twin's power NBM)
        self.p_gain = None                  # (wind bins, gain): generated power curve aligned to the twin
        self._last = {}

    def _gen(self, key, F, idx):
        phi, s = self.m["noise"][key]
        lead = LEADER.get(key)
        if lead in self._last:                                  # follower phase: leader noise + 5 % own
            out = self._last[lead] * (s / max(self.m["noise"][lead][1], 1e-9)) + self.rng.normal(0, 0.05 * s, len(idx))
        else:
            e, out = self.noise[key], np.empty(len(idx))
            for i in range(len(idx)):
                e = phi * e + self.rng.normal(0, s)
                out[i] = e
            self.noise[key] = e
            self._last[key] = out
        X = F.loc[idx, self.m["inputs"][key]].to_numpy(dtype=float)
        return self.m["gen"][key].predict(X) + out - (0.0 if key == P else self.offset.get(key, 0.0))

    def step(self, weather):
        hist = self.H.tail(self.HIST)
        F = pd.concat([hist, weather], ignore_index=True)
        idx = F.index[len(hist):]
        F["wind_cubed"] = F[W].clip(lower=0) ** 3
        F["wind_std_1h"] = F[W].rolling(STEPS_PER_HOUR, min_periods=2).std()
        r = np.deg2rad(F[fd.WIND_DIR] % 360)
        F["dir_sin"], F["dir_cos"] = np.sin(r), np.cos(r)
        nac = np.empty(len(idx))
        for i, wd in enumerate(F.loc[idx, fd.WIND_DIR].to_numpy()):
            err = float(wrap(wd - self.nacelle))
            if abs(err) > 8:                                   # yaw controller deadband
                self.nacelle = (self.nacelle + err) % 360
            nac[i] = self.nacelle
        F.loc[idx, "sensor_42_avg"] = nac
        F.loc[idx, "sensor_2_avg"] = wrap(F.loc[idx, fd.WIND_DIR].to_numpy() - nac)
        F["wind_mean_1h"] = F[W].rolling(STEPS_PER_HOUR, min_periods=1).mean()
        hour = F["time_stamp"].dt.hour + F["time_stamp"].dt.minute / 60
        F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        if self.power_fn is not None:                    # plant power curve = twin's weather-only power model
            phi, sd = self.m["noise"][P]
            e, nz = self.noise[P], np.empty(len(idx))
            for i in range(len(idx)):
                e = phi * e + self.rng.normal(0, sd)
                nz[i] = e
            self.noise[P] = e
            p = np.clip(self.power_fn(F.loc[idx]) + nz, -0.006, self.m["p_max"] * 1.01)
        else:
            p = np.clip(self._gen(P, F, idx), -0.006, self.m["p_max"] * 1.01)
        if self.p_gain is not None:
            p = np.where(p > 0, p * np.interp(F.loc[idx, W].to_numpy(float), *self.p_gain), p)
        p[F.loc[idx, W].to_numpy() > self.m["cut_out"]] = -0.003
        F.loc[idx, P] = p
        F["power_mean_1h"] = F[P].rolling(STEPS_PER_HOUR, min_periods=1).mean()
        running = p > 0.005 * self.m["p_max"]
        for k in AUX:
            if k in self.m["gen"]:
                v = self._gen(k, F, idx)
                F.loc[idx, k] = np.clip(v, 0, None) * (running if k in (fd.ROTOR_RPM, fd.GEN_RPM) else 1)
        F = fd.add_features(F)
        if fd.NACELLE in self.m["gen"]:
            F.loc[idx, fd.NACELLE] = self._gen(fd.NACELLE, F, idx)
        for t in TEMPS + ["pitch_dev"]:
            if t in self.m["gen"]:
                F.loc[idx, t] = self._gen(t, F, idx)
        F.loc[idx, "sensor_5_avg"] = F.loc[idx, "pitch_dev"] % 360
        F = fd.add_features(F)
        new = F.loc[idx].reset_index(drop=True)
        self.H = pd.concat([self.H, new], ignore_index=True).tail(self.HIST + 50).reset_index(drop=True)
        return new


class FarmWeather:
    """One wind field for the whole farm (AR(1) + daily cycle) + small per-turbine differences."""
    SCENARIOS = {"normal": (1.0, 0.0, 0.0), "windy": (1.45, 0.0, -2.0), "calm": (0.55, 0.0, 2.0),
                 "storm": (1.0, 9.0, -4.0), "hot": (0.9, 0.0, 12.0), "cold": (1.1, 0.0, -12.0)}

    def __init__(self, gens, seed=1, scenario="normal"):
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario
        self.ids = list(gens)
        self.wind_mean = float(np.mean([g["wind_mean"] for g in gens.values()]))
        self.wind_std = float(np.mean([g["wind_std"] for g in gens.values()]))
        self.amb_mean = float(np.mean([g["amb_mean"] for g in gens.values()]))
        self.t = pd.Timestamp(max(g["start"] for g in gens.values())).floor("10min") + pd.Timedelta(days=1)
        self.wind, self.amb_noise, self.dir = self.wind_mean, 0.0, 220.0
        self.local = {a: 0.0 for a in self.ids}

    def step(self, n):
        fac, extra, amb_off = self.SCENARIOS.get(self.scenario, self.SCENARIOS["normal"])
        mu, sd, phi = self.wind_mean * fac + extra, max(self.wind_std * (1.2 if extra else 1.0), 0.5), 0.985
        out = {a: [] for a in self.ids}
        for i in range(n):
            ts = self.t + pd.Timedelta(minutes=10 * i)
            hour = ts.hour + ts.minute / 60
            self.wind = mu + phi * (self.wind - mu) + self.rng.normal(0, sd * np.sqrt(1 - phi ** 2))
            self.amb_noise = 0.97 * self.amb_noise + self.rng.normal(0, 0.25)
            self.dir = (self.dir + self.rng.normal(0, 3)) % 360
            amb = self.amb_mean + amb_off + 4.0 * np.sin(2 * np.pi * (hour - 9) / 24) + self.amb_noise
            base = self.wind + 0.6 * np.sin(2 * np.pi * (hour - 9) / 24)
            for a in self.ids:
                self.local[a] = 0.9 * self.local[a] + self.rng.normal(0, 0.15)
                out[a].append({"time_stamp": ts, W: max(0.0, base * (1 + 0.03 * self.local[a])),
                               fd.AMBIENT: amb + self.rng.normal(0, 0.1),
                               fd.WIND_DIR: (self.dir + self.rng.normal(0, 2)) % 360})
        self.t = self.t + pd.Timedelta(minutes=10 * n)
        return {a: pd.DataFrame(v) for a, v in out.items()}
