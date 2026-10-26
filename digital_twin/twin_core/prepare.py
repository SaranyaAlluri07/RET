"""
Prepare the simulated wind farm (run once; the app also runs it automatically on first start).

For each of the 5 turbines (real assets 0, 10, 11, 13, 21) it reads the healthy training period of
that turbine's normal dataset and trains:
  * GENERATOR models (odd days)  : how every sensor behaves given wind / load / temperature -> used to
                                   create new realistic data, plus a noise model (size + persistence)
  * TWIN models (even days)      : the monitoring side, exactly the normal-behaviour models of
                                   fault_detection_v2.py (so the twin never sees the generator's model)
  * an autoencoder over all simulated sensors (whole-turbine deviation)
  * component normal levels: the twin is run on 12 simulated healthy days and the 99.5 % level of every
    component score is stored (as fault_detection_v2.py does on its calibration period)

    python digital_twin/twin_core/prepare.py            (all 5 turbines, ~2 min each)
    python digital_twin/twin_core/prepare.py --assets 10
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin_core import paths                                   # noqa: E402
import fault_detection_v2 as fd                               # noqa: E402

STEPS_PER_HOUR = 6
P, W = fd.POWER, fd.WIND
RAW_TARGETS = [f"sensor_{i}_avg" for i in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
                                           35, 36, 37, 38, 39, 40, 41, 53)] + ["pitch_dev", P]
AUX = {fd.ROTOR_RPM: [W, P, "power_mean_1h"], fd.GEN_RPM: [W, P, "power_mean_1h"],
       "sensor_23_avg": [P, "power_mean_1h", W], "sensor_24_avg": [P, "power_mean_1h", W],
       "sensor_25_avg": [P, "power_mean_1h", W], "sensor_32_avg": [P, W], "sensor_33_avg": [P, W],
       "sensor_34_avg": [P, W]}
TWIN_TARGETS = [s for sens in fd.SUBSYSTEMS.values() for s in sens if s != "ae_error"]
AE_COLS = [f"sensor_{i}_avg" for i in (0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                       23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 52, 53)] + [W, P]


def ar1(resid):
    """Noise model; robust so unlabelled curtailment / stops do not inflate it."""
    r = pd.Series(resid).dropna()
    if len(r) < 50:
        return 0.0, float(r.std() or 1.0)
    med = r.median()
    rob = 1.4826 * (r - med).abs().median() or r.std() or 1.0
    r = r.clip(med - 3 * rob, med + 3 * rob)
    phi = float(np.clip(r.autocorr(1), 0, 0.995)) if r.std() > 0 else 0.0
    return phi, float(r.std() * np.sqrt(1 - phi ** 2))


def small_model():
    m = fd.make_model()
    m.set_params(max_iter=150)
    return m


def train_asset(asset, ds, max_rows=6000, log=print):
    path = paths.dataset_path(ds)
    if not path:
        raise FileNotFoundError(f"dataset {ds}.csv (Wind Farm A) not found - see data/README.md")
    t0 = time.time()
    df = fd.read_csv(path, usecols=lambda c: c in ("time_stamp", "train_test", "status_type_id") or c.endswith("_avg"))
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = fd.add_features(df.sort_values("time_stamp").reset_index(drop=True))
    healthy = df[(df["train_test"].astype(str).str.lower() == "train") & df["status_type_id"].isin(fd.NORMAL_STATUS)]
    odd = (healthy["time_stamp"].dt.dayofyear % 2 == 1).to_numpy()
    gen_rows, twin_rows = healthy[odd], healthy[~odd]
    rng = np.random.default_rng(asset)

    def sub(d):
        return d if len(d) <= max_rows else d.iloc[np.sort(rng.choice(len(d), max_rows, replace=False))]

    m = {"asset": asset, "base_dataset": ds, "gen": {}, "gen_inputs": {}, "noise": {}, "twin": {},
         "twin_inputs": {}, "calib": {}}
    gen_list = list(AUX) + ([fd.NACELLE] if fd.NACELLE in df else []) + RAW_TARGETS
    for t in gen_list:
        if t not in df or t in m["gen"]:
            continue
        X = AUX.get(t) or fd.inputs_for(t, df.columns)
        X = [c for c in X if c in df and c != t]
        g, w = sub(gen_rows.dropna(subset=[t])), sub(twin_rows.dropna(subset=[t]))
        if len(g) < 300 or g[t].std() < 1e-9:
            continue
        mg = small_model().fit(g[X], g[t])
        m["gen"][t], m["gen_inputs"][t] = mg, X
        m["noise"][t] = ar1(w[t] - mg.predict(w[X]))
    for t in TWIN_TARGETS:
        if t not in df:
            continue
        X = [c for c in fd.inputs_for(t, df.columns) if c in df]
        g, w = sub(gen_rows.dropna(subset=[t])), sub(twin_rows.dropna(subset=[t]))
        if len(w) < 300 or w[t].std() < 1e-9:
            continue
        mt = small_model().fit(w[X], w[t])
        r = g[t] - mt.predict(g[X])
        m["twin"][t], m["twin_inputs"][t] = mt, X
        m["calib"][t] = (float(r.mean()), float(r.std() + 1e-9))
    # autoencoder (whole turbine)
    cols = [c for c in AE_COLS if c in df]
    med = twin_rows[cols].median()
    sc = StandardScaler().fit(sub(twin_rows)[cols].fillna(med))
    Xa = np.clip(sc.transform(sub(twin_rows)[cols].fillna(med)), -10, 10)
    n = len(cols)
    ae = MLPRegressor(hidden_layer_sizes=(max(16, n // 2), max(4, n // 6), max(16, n // 2)), activation="tanh",
                      alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                      n_iter_no_change=8, random_state=0).fit(Xa, Xa)
    Xc = np.clip(sc.transform(sub(gen_rows)[cols].fillna(med)), -10, 10)
    R = Xc - ae.predict(Xc)
    r_mu, r_sd = R.mean(axis=0), R.std(axis=0) + 1e-9
    top3 = np.sort(np.abs((R - r_mu) / r_sd), axis=1)[:, -3:].mean(axis=1)
    e = np.log(top3 + 1e-6)
    m["ae"] = dict(model=ae, scaler=sc, cols=cols, fill=med, r_mu=r_mu, r_sd=r_sd, mu=float(e.mean()), sd=float(e.std() + 1e-9))
    # operating statistics
    p_max = float(healthy[P].quantile(0.999))
    run = healthy[healthy[P] > 0.05 * p_max]
    m.update(p_max=p_max, cut_out=float(min(run[W].quantile(0.999) + 1.0, 25.0)) if len(run) else 25.0,
             wind_mean=float(healthy[W].mean()), wind_std=float(healthy[W].std()),
             amb_mean=float(healthy[fd.AMBIENT].mean()), start=str(healthy["time_stamp"].max()),
             volt_mean=float(healthy["sensor_32_avg"].median()) if "sensor_32_avg" in healthy else 400.0)
    bpath = os.path.join(paths.PROCESSED, f"baseline_{ds}.json")
    if os.path.isfile(bpath):
        m["baseline"] = json.load(open(bpath))
    else:
        num = [c for c in df.columns if c.endswith("_avg") or c.endswith("_std")]
        m["baseline"] = {c: [float(healthy[c].median()), float(healthy[c].quantile(0.75) - healthy[c].quantile(0.25))]
                         for c in num if c in healthy}
    log(f"  asset {asset} (dataset {ds}): {len(m['gen'])} generator + {len(m['twin'])} twin models + autoencoder "
        f"({time.time() - t0:.0f}s)")
    return m


def calibrate_levels(m, days=40, log=print):
    """Run the healthy simulated turbine and store each component's normal level (99.5 %)."""
    from twin_core.simulator import SimTurbine, FarmWeather
    from twin_core import models_io
    m["levels"] = {}
    weather = FarmWeather({m["asset"]: m}, seed=7)
    tb = SimTurbine(m, power_model=models_io.power_model(), rf=None, seed=11, levels_known=False)
    n = days * 24 * STEPS_PER_HOUR
    for _ in range(n // 36):
        tb.step(weather.step(36)[m["asset"]])
    fast, slow = tb.raw_component_history()
    m["levels"] = {"fast": {c: float(max(1.0, fast[c].quantile(0.995))) for c in fast},
                   "slow": {c: float(max(1.0, slow[c].quantile(0.995))) for c in slow}}
    r = np.array(tb.perf_samples)
    if len(r) > 100:
        mean = float(np.mean(r))
        m["perf"] = {"mean": mean, "limit": float(min(0.85, np.quantile(r, 0.001) / mean - 0.04))}
    log(f"  asset {m['asset']}: component normal levels + energy performance level calibrated on {days} simulated healthy days"
        + (f" (healthy index {m['perf']['mean']:.3f}, alarm below {m['perf']['limit']:.2f})" if "perf" in m else ""))
    return m


def prepare(assets=None, log=print):
    os.makedirs(paths.MODEL_DIR, exist_ok=True)
    done = []
    for asset, ds in paths.ASSETS.items():
        if assets and asset not in assets:
            continue
        m = calibrate_levels(train_asset(asset, ds, log=log), log=log)
        joblib.dump(m, os.path.join(paths.MODEL_DIR, f"turbine_{asset}.joblib"), compress=3)
        done.append(asset)
    return done


def missing_models():
    out = []
    for a in paths.ASSETS:
        p = os.path.join(paths.MODEL_DIR, f"turbine_{a}.joblib")
        try:
            joblib.load(p)
        except Exception:                           # missing, or saved with another scikit-learn version
            out.append(a)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", type=int, nargs="*")
    a = ap.parse_args()
    print("Wind Farm A folder:", paths.find_farm_dir())
    prepare(a.assets)
    print("Saved to", paths.MODEL_DIR)
