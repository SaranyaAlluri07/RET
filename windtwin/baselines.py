"""Baselines, all reduced to the same per-dataset output on the prediction window:
point predictions (bool, for CARE), alarm activity (bool, for operational metrics) and a diagnosed
component at the first alarm (or None).

(a) legacy Random Forest (anomaly_detection_model/anomaly_classifier.py, approach C): its saved outputs are
    event-level only, so its leave-one-group-out CV is re-run with the legacy code (imported, unmodified);
    alarm = legacy event rule (>= 3 h in a row of a fault class); diagnosis = predicted class.
(b) legacy residual detector (fault_detection_v2 saved scores): point = anomaly_score > threshold,
    alarm = its own criticality >= 18 rule, diagnosis = top subsystem at the first alarm.
(c) IsolationForest and (d) MLP autoencoder on standardized Avg signals, one model per dataset fitted
    on the same fit rows as the NBMs; anomaly threshold = quantile q of the validation score (q chosen
    leave-one-turbine-out like the proposed methods); alarm = CARE criticality >= 72."""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import config as C, data, care

ANGLES = {C.WIND_DIR, C.REL_DIR, C.NACELLE_DIR, C.PITCH_RAW}
QUANTILES = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
LEGACY_SUB = {"Generator bearings": "Generator bearings", "Gearbox": "Gearbox", "Generator windings": "Generator windings",
              "Transformer": "Transformer", "Hydraulic group": "Hydraulic group", "Hub / pitch": "Hub / pitch",
              "Converter": "Converter", "Nacelle cabinets": "Nacelle cabinets", "Electrical balance": "Electrical balance",
              "Power performance": "Power performance", "Whole turbine (AE)": C.UNKNOWN}
RF_CLASS = {"hydraulic": "Hydraulic group", "gearbox": "Gearbox", "generator_bearing": "Generator bearings",
            "transformer": "Transformer"}


# ------------------------------------------------------------------ (b) legacy residual detector
def legacy_residual(ds_id, times):
    """Align saved legacy scores to the prediction timestamps."""
    f = C.LEGACY_SCORES / f"{ds_id}.csv"
    s = pd.read_csv(f, parse_dates=["time_stamp"]).drop_duplicates("time_stamp").set_index("time_stamp")
    s = s.reindex(pd.to_datetime(times))
    point = (s["anomaly_score"] > s["threshold"]).fillna(False).to_numpy()
    alarm = s["alarm"].astype("boolean").fillna(False).to_numpy(bool)
    subs = [c for c in LEGACY_SUB if c in s]
    top = s[subs].fillna(-1).idxmax(axis=1).map(LEGACY_SUB).to_numpy()
    return point, alarm, top


# ------------------------------------------------------------------ (a) legacy random forest
def legacy_rf(log=print):
    """Re-run the legacy RF leave-one-group-out CV. Returns DataFrame(dataset, time_stamp, pred class)."""
    cache = C.CACHE_DIR / "rf_legacy_oof.parquet"
    if cache.is_file():
        return pd.read_parquet(cache)
    sys.path.insert(0, str(C.LEGACY_DIR))
    import anomaly_classifier as ac      # legacy module, used unmodified
    t0 = time.time()
    dat, _ = ac.load_all(str(C.ROOT))
    run = dat[dat["running"]].copy()
    feats = ac.feature_columns(run)
    log(f"  RF: {len(run):,} running rows, {len(feats)} features, {run['group'].nunique()} CV groups")
    pred, _ = ac.run_cv(run, feats, "C", lambda s: None)
    out = pd.DataFrame({"dataset": run["dataset"].to_numpy(), "time_stamp": run["time_stamp"].to_numpy(),
                        "pred": pred.to_numpy().astype(str)})
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    log(f"  RF CV done ({time.time() - t0:.0f}s)")
    return out


def rf_outputs(rf, ds_id, times, hours=3):
    """Point = predicted fault class; alarm = run of >= `hours` of fault-class rows (legacy event rule)."""
    t = pd.to_datetime(pd.Series(times))
    g = rf[rf["dataset"] == ds_id].drop_duplicates("time_stamp").set_index("time_stamp")["pred"]
    p = g.reindex(t).fillna("normal").to_numpy()
    point = p != "normal"
    n = hours * C.STEPS_PER_HOUR
    runl = np.zeros(len(p), int)
    for i in range(len(p)):
        runl[i] = runl[i - 1] + 1 if (point[i] and i > 0) else int(point[i])
    alarm = np.zeros(len(p), bool)
    for s, e in care.alarm_episodes(point):
        if e - s >= n:
            alarm[s + n - 1:e] = True
    comp = np.array([RF_CLASS.get(x, C.UNKNOWN) if x != "normal" else None for x in p], dtype=object)
    return point, alarm, comp


# ------------------------------------------------------------------ (c, d) IsolationForest / autoencoder
def _matrix(df):
    cols = [c for c in df.columns if c.endswith("_avg") and c not in ANGLES] + ["pitch"]
    return cols


def unsupervised_scores(ds_id, seed=0):
    """Fit IF + AE on the dataset's fit rows; score validation rows and the monitoring segment."""
    cache = C.CACHE_DIR / f"unsup_{ds_id}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    df = data.prepare(data.load_dataset(ds_id))
    sp = data.split(df)
    cols = [c for c in _matrix(df) if df[c].iloc[sp["fit"]].notna().mean() > 0.95 and df[c].iloc[sp["fit"]].std() > 1e-9]
    X = df[cols].to_numpy(float)
    med = np.nanmedian(X[sp["fit"]], axis=0)
    X = np.where(np.isfinite(X), X, med)
    rng = np.random.default_rng(seed)
    fit = sp["fit"] if len(sp["fit"]) <= 20000 else np.sort(rng.choice(sp["fit"], 20000, replace=False))
    sc = StandardScaler().fit(X[fit])
    Xs = np.clip(sc.transform(X), -10, 10)
    iso = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1).fit(Xs[fit])
    n = Xs.shape[1]
    ae = MLPRegressor(hidden_layer_sizes=(max(16, n // 2), max(4, n // 8), max(16, n // 2)), activation="tanh",
                      alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                      n_iter_no_change=6, random_state=seed).fit(Xs[fit], Xs[fit])
    seg = np.arange(sp["monitor_start"], len(df))
    rows = np.concatenate([sp["val"], seg])
    s_if = -iso.score_samples(Xs[rows])
    s_ae = np.sqrt(np.mean((Xs[rows] - ae.predict(Xs[rows])) ** 2, axis=1))
    nv = len(sp["val"])
    out = {"if_val": s_if[:nv], "if_seg": s_if[nv:], "ae_val": s_ae[:nv], "ae_seg": s_ae[nv:],
           "times": df["time_stamp"].to_numpy()[seg], "is_pred": ~sp["is_train"][seg]}
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def threshold_outputs(val_scores, seg_scores, is_pred, q):
    """Point = score > validation quantile q; alarm = CARE criticality >= 72 (counter starts at context)."""
    thr = float(np.quantile(val_scores, q))
    point_seg = seg_scores > thr
    crit = care.criticality(point_seg)
    return point_seg[is_pred], (crit >= care.CRIT_THRESHOLD)[is_pred]
