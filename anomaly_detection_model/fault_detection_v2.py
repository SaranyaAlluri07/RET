"""
Wind turbine fault detection - VERSION 2 (more sensitive, same honest evaluation)
================================================================================
Changes compared with fault_detection.py (v1 is kept unchanged for the reported results):
  1. Per-component thresholds: every component score is divided by its OWN normal level
     (99.5 % quantile on the healthy calibration period). A noisy signal (pitch) no longer
     sets the bar for a quiet one (transformer), so small but real deviations count.
  2. Two time scales: 6 h smoothing (fast faults) AND 24 h smoothing (slow drifts that never
     look big in 6 h). The higher of the two (each on its own normal level) is used.
  3. Physics signals: gearbox bearing minus gearbox oil, generator bearing DE minus NDE,
     spread between the 3 stator / transformer / IGBT phases, hydraulic oil minus nacelle.
     A failing part heats up relative to its neighbours even when the absolute value looks normal.
  4. Robust training: rows of the healthy period with very large errors (hidden glitches) are
     removed and the model is trained again.
  5. Near-miss report: for every fault, how close the score came to the threshold and which
     component peaked - this tells WHY a fault was missed.
The alarm threshold and alarm time are still chosen by leave-one-dataset-out cross-validation.
Outputs go to outputs_fault_detection_v2/ (v1 results are not overwritten).

--- original description (v1) ---
Wind turbine fault detection (Digital Twin - Part 2)
Dataset: CARE to Compare - Wind Farm A   https://zenodo.org/records/14958989

Method: Normal Behaviour Models (NBM) + autoencoder
--------------------------------------------------
For every dataset (one turbine, one time window):

1. TRAIN  - On the healthy "train" period (normal status only) learn how each
            component temperature normally behaves given the operating
            conditions (wind, power, ambient temperature, rotor speed ...).
            One Gradient Boosting model per monitored signal (component
            temperatures, pitch angle, phase imbalance, power).
            PLUS an autoencoder over ~50 sensors that learns how they normally
            move together (catches faults no single sensor shows).
2. CALIBRATE - The last 20 % of the train period is held out. The model errors
            there tell us how big a "normal" error is (mean and std) and give
            the alarm threshold.
3. MONITOR - On the "prediction" period:
              residual  = measured - expected
              z-score   = (residual - normal mean) / normal std
              smoothed with a rolling mean (default 6 h)
            Temperatures: only "hotter than expected" counts.
            Power:        only "less power than expected" counts.
            Pitch:        deviation in both directions counts.
            Autoencoder:  larger reconstruction error than normal counts.
            Subsystem score = largest z of its sensors.
            Anomaly score   = largest subsystem score.
4. ALARM  - Criticality counter (as in the CARE paper): +1 for every anomalous
            10-min step while the turbine is in normal status, -1 for a normal
            step, unchanged during non-normal status. Alarm when it reaches
            72 (= 12 hours of net anomalous behaviour).
5. DIAGNOSE - The subsystem with the highest score at the first alarm is
            reported (for the autoencoder: the sensor it reconstructs worst),
            and compared with the fault description in event_info.csv.
6. SWEEP  - Detection rate vs false alarms for many thresholds / alarm
            limits (sweep.csv, sweep.png). Run on saved scores with --sweep-only.

Outputs (in outputs_fault_detection/):
    results_per_dataset.csv  detection, lead time, false alarms, diagnosis
    summary.json             overall detection rate, false alarm rate, ...
    sweep.csv / sweep.png    detection vs false alarms for many settings
    scores/<id>.csv          time series of all scores (used by the dashboard)
    plots/<id>.png           anomaly score plot for every dataset
    models/<id>.joblib       trained models (only with --save-models)

Usage
-----
    python fault_detection.py
    python fault_detection.py --smooth-hours 12 --crit-limit 72 --datasets 0 10 68
    python fault_detection.py --sweep-only      (re-analyse saved scores, no training)
    python fault_detection.py --no-ae           (skip the autoencoder, faster)
"""

import argparse
import glob
import warnings
import json
import os
# hide harmless sklearn/joblib version warnings, also in the parallel worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

STEPS_PER_HOUR = 6
V2_THRESHOLDS = np.round(np.arange(1.0, 6.01, 0.25), 2)      # relative scores
NORMAL_STATUS = (0, 2)          # 0 = normal operation, 2 = idling

# --------------------------------------------------------------------------
# Wind Farm A sensors (from feature_description.csv)
# --------------------------------------------------------------------------
SUBSYSTEMS = {
    "Generator bearings": ["sensor_13_avg",     # generator bearing 2 (drive end)
                           "sensor_14_avg",     # generator bearing 1 (non-drive end)
                           "gen_bearing_diff"], # DE - NDE (v2)
    "Gearbox":            ["sensor_11_avg",     # gearbox bearing, high speed shaft
                           "sensor_12_avg",     # gearbox oil
                           "gbx_bearing_minus_oil"],   # v2
    "Generator windings": ["sensor_15_avg", "sensor_16_avg", "sensor_17_avg", "stator_spread"],
    "Transformer":        ["sensor_38_avg", "sensor_39_avg", "sensor_40_avg", "transformer_spread"],
    "Hydraulic group":    ["sensor_41_avg",     # hydraulic oil temperature
                           "hydraulic_minus_nacelle"],  # v2
    "Hub / pitch":        ["sensor_6_avg",      # hub controller temperature
                           "sensor_53_avg",     # nose cone temperature
                           "pitch_dev"],        # pitch angle vs expected (both directions)
    "Converter":          ["sensor_21_avg", "sensor_35_avg", "sensor_36_avg", "sensor_37_avg", "igbt_spread"],
    "Nacelle cabinets":   ["sensor_7_avg", "sensor_8_avg", "sensor_9_avg", "sensor_10_avg",
                           "sensor_19_avg", "sensor_20_avg"],
    "Electrical balance": ["current_imbalance", "voltage_imbalance"],
    "Power performance":  ["power_30_avg"],     # grid power vs expected
    "Whole turbine (AE)": ["ae_error"],         # autoencoder over all sensors
}
TWO_SIDED = {"pitch_dev", "gen_bearing_diff"}
NOT_AE_INPUT = {"sensor_1_avg", "sensor_2_avg", "sensor_42_avg",   # directions (angles)
                "sensor_5_avg",                                     # pitch (angle wrapping)
                "sensor_44_avg", "sensor_45_avg", "sensor_46_avg", "sensor_47_avg",
                "sensor_48_avg", "sensor_49_avg", "sensor_50_avg", "sensor_51_avg"}  # energy counters
SENSOR_NAMES = {
    "sensor_13_avg": "Gen. bearing DE temp", "sensor_14_avg": "Gen. bearing NDE temp",
    "sensor_11_avg": "Gearbox HSS bearing temp", "sensor_12_avg": "Gearbox oil temp",
    "sensor_15_avg": "Stator winding L1 temp", "sensor_16_avg": "Stator winding L2 temp",
    "sensor_17_avg": "Stator winding L3 temp", "sensor_38_avg": "HV transformer L1 temp",
    "sensor_39_avg": "HV transformer L2 temp", "sensor_40_avg": "HV transformer L3 temp",
    "sensor_41_avg": "Hydraulic oil temp", "sensor_21_avg": "Grid-side IGBT temp",
    "sensor_35_avg": "Rotor-side IGBT L1 temp", "sensor_36_avg": "Rotor-side IGBT L2 temp",
    "sensor_37_avg": "Rotor-side IGBT L3 temp", "power_30_avg": "Grid power",
    "sensor_6_avg": "Hub controller temp", "sensor_53_avg": "Nose cone temp",
    "pitch_dev": "Pitch angle", "sensor_7_avg": "Top nacelle controller temp",
    "sensor_8_avg": "Choke coils temp", "sensor_9_avg": "VCP-board temp",
    "sensor_10_avg": "VCS cooling water temp", "sensor_19_avg": "Split ring chamber temp",
    "sensor_20_avg": "Busbar temp", "current_imbalance": "Phase current imbalance",
    "voltage_imbalance": "Phase voltage imbalance", "ae_error": "Autoencoder error",
    "gen_bearing_diff": "Gen. bearing DE - NDE", "gbx_bearing_minus_oil": "Gearbox bearing - oil",
    "stator_spread": "Stator phase spread", "transformer_spread": "Transformer phase spread",
    "igbt_spread": "IGBT phase spread", "hydraulic_minus_nacelle": "Hydraulic oil - nacelle",
}
SENSOR_TO_SUBSYSTEM = {s: name for name, sens in SUBSYSTEMS.items() for s in sens}

# Operating-condition inputs (things that CAUSE temperatures, not other component temps)
WIND, POWER, AMBIENT = "wind_speed_3_avg", "power_30_avg", "sensor_0_avg"
NACELLE, GEN_RPM, ROTOR_RPM, WIND_DIR = "sensor_43_avg", "sensor_18_avg", "sensor_52_avg", "sensor_1_avg"
BASE_INPUTS = [WIND, POWER, AMBIENT, NACELLE, GEN_RPM, ROTOR_RPM, WIND_DIR]

# which subsystem should catch which fault description (for the diagnosis check)
FAULT_TO_SUBSYSTEM = {          # fault text -> subsystems that count as a correct diagnosis
    "generator bearing": ["Generator bearings"],
    "gearbox": ["Gearbox"],
    "transformer": ["Transformer"],
    "hydraulic": ["Hydraulic group", "Hub / pitch"],   # hydraulic pitch system sits in the hub
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def sniff_sep(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    return ";" if first.count(";") > first.count(",") else ","


def read_csv(path, **kw):
    try:
        return pd.read_csv(path, sep=sniff_sep(path), **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sniff_sep(path), encoding="latin-1", **kw)


def find_farm_dir(here):
    for base in (here, os.path.dirname(here)):              # this folder, then the parent
        hits = [d for d in glob.glob(os.path.join(base, "**", "Wind Farm A"), recursive=True)
                if os.path.isfile(os.path.join(d, "event_info.csv"))]
        if hits:
            return hits[0]
    return None


def dataset_files(farm_dir):
    out = {}
    for f in glob.glob(os.path.join(farm_dir, "**", "*.csv"), recursive=True):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.isdigit():
            out[int(stem)] = f
    return dict(sorted(out.items()))


def add_features(df):
    """Thermal inertia: temperatures follow the recent load, not just the current one."""
    p = df[POWER]
    for h in (1, 3, 6, 12):
        df[f"power_mean_{h}h"] = p.rolling(h * STEPS_PER_HOUR, min_periods=1).mean()
    df["power_sq"] = p.clip(lower=0) ** 2                 # copper losses ~ I^2
    df["ambient_mean_6h"] = df[AMBIENT].rolling(6 * STEPS_PER_HOUR, min_periods=1).mean()
    df["wind_cubed"] = df[WIND].clip(lower=0) ** 3
    df["wind_std_1h"] = df[WIND].rolling(STEPS_PER_HOUR, min_periods=2).std()
    if WIND_DIR in df:
        r = np.deg2rad(df[WIND_DIR] % 360)
        df["dir_sin"], df["dir_cos"] = np.sin(r), np.cos(r)
    hour = df["time_stamp"].dt.hour + df["time_stamp"].dt.minute / 60
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)

    # derived health signals
    def diff(a, b):
        return df[a] - df[b] if a in df and b in df else None
    def spread(cols):
        return df[cols].max(axis=1) - df[cols].min(axis=1) if all(c in df for c in cols) else None
    for name, v in (("gen_bearing_diff", diff("sensor_13_avg", "sensor_14_avg")),
                    ("gbx_bearing_minus_oil", diff("sensor_11_avg", "sensor_12_avg")),
                    ("hydraulic_minus_nacelle", diff("sensor_41_avg", NACELLE)),
                    ("stator_spread", spread(["sensor_15_avg", "sensor_16_avg", "sensor_17_avg"])),
                    ("transformer_spread", spread(["sensor_38_avg", "sensor_39_avg", "sensor_40_avg"])),
                    ("igbt_spread", spread(["sensor_35_avg", "sensor_36_avg", "sensor_37_avg"]))):
        if v is not None:
            df[name] = v
    if "sensor_5_avg" in df:                              # pitch: undo 0/360 wrapping
        df["pitch_dev"] = ((df["sensor_5_avg"] + 180) % 360) - 180
    for name, cols in (("current_imbalance", ["sensor_23_avg", "sensor_24_avg", "sensor_25_avg"]),
                       ("voltage_imbalance", ["sensor_32_avg", "sensor_33_avg", "sensor_34_avg"])):
        if all(c in df for c in cols):
            ph = df[cols]
            mean = ph.mean(axis=1)
            ok = mean.abs() > 0.05 * mean.abs().quantile(0.99)   # only when there is load
            df[name] = (ph.std(axis=1) / mean.abs()).where(ok)
    return df


def inputs_for(target, columns):
    """Inputs for the model of one target sensor."""
    if target == POWER:                          # power depends on weather only
        cand = [WIND, "wind_cubed", "wind_std_1h", AMBIENT, "dir_sin", "dir_cos"]
    else:
        cand = BASE_INPUTS + ["power_mean_1h", "power_mean_3h", "power_mean_6h", "power_mean_12h",
                              "power_sq", "ambient_mean_6h", "wind_std_1h",
                              "dir_sin", "dir_cos", "hour_sin", "hour_cos"]
        cand = [c for c in cand if c != WIND_DIR]
    return [c for c in cand if c in columns and c != target]


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=0)


def criticality(anomalous, normal_status, limit):
    crit = np.zeros(len(anomalous), dtype=int)
    c = 0
    for i, (a, n) in enumerate(zip(anomalous, normal_status)):
        if n:
            c = c + 1 if a else max(c - 1, 0)
        crit[i] = c
    return crit, crit >= limit


def health_trend(scores, threshold, window_steps=144):
    """Trend of smoothed subsystem scores over the last `window_steps` (default 24 h).
    scores: DataFrame (rows = time, columns = subsystems).
    Returns (slope_per_hour, hours_to_threshold) DataFrames. hours_to_threshold is
    NaN when the score is falling/flat or already above the threshold."""
    t = pd.Series(np.arange(len(scores), dtype=float), index=scores.index)
    mp = max(STEPS_PER_HOUR, window_steps // 3)
    mt = t.rolling(window_steps, min_periods=mp).mean()
    mt2 = (t * t).rolling(window_steps, min_periods=mp).mean()
    var_t = (mt2 - mt * mt).replace(0, np.nan)
    slope = pd.DataFrame(index=scores.index)
    for c in scores.columns:
        s = scores[c].astype(float)
        mts = (s * t).rolling(window_steps, min_periods=mp).mean()
        ms = s.rolling(window_steps, min_periods=mp).mean()
        slope[c] = (mts - mt * ms) / var_t * STEPS_PER_HOUR          # score units per hour
    rising = (slope > 1e-6) & (scores < threshold)
    hours = ((threshold - scores) / slope).where(rising)
    return slope, hours


# --------------------------------------------------------------------------
# One dataset
# --------------------------------------------------------------------------
def process_dataset(ds_id, path, event, args):
    header = read_csv(path, nrows=0).columns
    meta = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
    avg_cols = [c for c in header if c.endswith("_avg")]   # Min/Max/Std are unreliable
    df = read_csv(path, usecols=lambda c: c in meta or c in avg_cols)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.sort_values("time_stamp").reset_index(drop=True)
    df = add_features(df)
    targets = [s for sens in SUBSYSTEMS.values() for s in sens if s in df.columns]
    rng = np.random.default_rng(0)

    normal = df["status_type_id"].isin(NORMAL_STATUS).to_numpy()
    is_train = (df["train_test"].astype(str).str.lower() == "train").to_numpy()
    pred_mask = ~is_train
    pred_status_normal_share = float(normal[pred_mask].mean()) if pred_mask.any() else float("nan")

    # In Wind Farm A the status labels come from the failure logbook, so the
    # fault window itself is labelled "not normal". The dataset authors say to
    # use them only to filter TRAINING data. -> in the prediction period every
    # timestamp is scored (unless --use-status-in-prediction).
    monitored = normal.copy()
    if not args.use_status_in_prediction:
        monitored[pred_mask] = True

    # power can only be "too low" while the turbine is actually running
    p_ref = float(df.loc[is_train & normal, POWER].quantile(0.999)) if POWER in df else 1.0
    running = (df[POWER] > 0.02 * p_ref).to_numpy() if POWER in df else np.ones(len(df), bool)

    train_idx = np.where(is_train & normal)[0]
    split = int(len(train_idx) * (1 - args.calib_frac))
    fit_idx, cal_idx = train_idx[:split], train_idx[split:]

    z = pd.DataFrame(index=df.index)
    models = {}
    for t in targets:
        X_cols = inputs_for(t, df.columns)
        ok_fit = fit_idx[df.loc[fit_idx, t].notna().to_numpy()]
        ok_cal = cal_idx[df.loc[cal_idx, t].notna().to_numpy()]
        if len(ok_fit) < 500 or len(ok_cal) < 100 or df.loc[ok_fit, t].std() < 1e-9:
            continue
        if len(ok_fit) > args.max_fit_rows:      # speed: a random subset is enough
            ok_fit = np.sort(rng.choice(ok_fit, args.max_fit_rows, replace=False))
        m = make_model().fit(df.loc[ok_fit, X_cols], df.loc[ok_fit, t])
        r_fit = df.loc[ok_fit, t] - m.predict(df.loc[ok_fit, X_cols])          # v2: robust refit
        keep = (r_fit - r_fit.median()).abs() <= 4 * 1.4826 * (r_fit - r_fit.median()).abs().median() + 1e-9
        if 0.9 < keep.mean() < 1.0:
            m = make_model().fit(df.loc[ok_fit[keep.to_numpy()], X_cols], df.loc[ok_fit[keep.to_numpy()], t])
        resid = df[t] - m.predict(df[X_cols])
        mu, sd = resid.iloc[ok_cal].mean(), resid.iloc[ok_cal].std() + 1e-9
        zt = (resid - mu) / sd
        zt = -zt if t == POWER else zt          # power: shortfall is bad
        zt = zt.abs() if t in TWO_SIDED else zt
        zt[~monitored] = np.nan                 # ignore stops/maintenance in TRAIN period
        if t == POWER:
            zt[~running] = np.nan               # turbine stopped -> no power expectation
        z[t] = zt
        models[t] = {"model": m, "inputs": X_cols, "mu": mu, "sd": sd}

    ae_feat_err, ae_cols = None, []
    if not args.no_ae:
        ae_cols = [c for c in avg_cols if c not in NOT_AE_INPUT and c in df
                   and df.loc[fit_idx, c].notna().mean() > 0.9 and df.loc[fit_idx, c].std() > 1e-9]
    if len(ae_cols) >= 5:
        med = df.loc[fit_idx, ae_cols].median()
        X_all = df[ae_cols].fillna(med)
        fit_rows = fit_idx if len(fit_idx) <= args.max_fit_rows else \
            np.sort(rng.choice(fit_idx, args.max_fit_rows, replace=False))
        scaler = StandardScaler().fit(X_all.iloc[fit_rows])
        Xs = np.clip(scaler.transform(X_all), -10, 10)
        n = len(ae_cols)
        ae = MLPRegressor(hidden_layer_sizes=(max(16, n // 2), max(4, n // 6), max(16, n // 2)),
                          activation="tanh", alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                          max_iter=80, early_stopping=True, n_iter_no_change=8, random_state=0)
        ae.fit(Xs[fit_rows], Xs[fit_rows])
        R = Xs - ae.predict(Xs)                                 # reconstruction error per sensor
        r_mu, r_sd = R[cal_idx].mean(axis=0), R[cal_idx].std(axis=0) + 1e-9
        ae_feat_err = np.abs((R - r_mu) / r_sd)                 # how unusual, per sensor
        # mean of the 3 most unusual sensors: one sick component is not averaged away
        top3 = np.sort(ae_feat_err, axis=1)[:, -3:].mean(axis=1)
        err = pd.Series(np.log(top3 + 1e-6), index=df.index)
        mu, sd = err.iloc[cal_idx].mean(), err.iloc[cal_idx].std() + 1e-9
        za = (err - mu) / sd
        za[~monitored] = np.nan
        z["ae_error"] = za
        models["ae_error"] = {"model": ae, "scaler": scaler, "inputs": ae_cols, "fill": med,
                              "r_mu": r_mu, "r_sd": r_sd, "mu": mu, "sd": sd}

    # smooth and combine
    def subsystem_scores(hours):
        w = int(hours * STEPS_PER_HOUR)
        zz = z.rolling(w, min_periods=max(1, w // 3)).mean().clip(lower=0)
        sb = pd.DataFrame({name: zz[[s for s in sens if s in zz]].max(axis=1)
                           for name, sens in SUBSYSTEMS.items() if any(s in zz for s in sens)})
        return zz, sb

    def own_level(sb):          # v2: each component relative to its own normal level
        q = sb.iloc[cal_idx].quantile(args.sub_q).fillna(args.sub_floor).clip(lower=args.sub_floor)
        return sb / q

    zs, sub_fast = subsystem_scores(args.smooth_hours)
    _, sub_slow = subsystem_scores(args.slow_hours)
    nf, ns = own_level(sub_fast), own_level(sub_slow)
    sub = pd.DataFrame(np.fmax(nf.to_numpy(), ns.to_numpy()), index=nf.index, columns=nf.columns)
    score = sub.max(axis=1)

    # threshold from the held-out healthy calibration period
    cal_scores = score.iloc[cal_idx].dropna()
    threshold = max(args.min_threshold, float(cal_scores.quantile(args.threshold_q))) \
        if len(cal_scores) else args.min_threshold

    anomalous = (score > threshold).fillna(False).to_numpy()
    crit = np.zeros(len(df), dtype=int)
    alarm = np.zeros(len(df), dtype=bool)
    p_idx = np.where(pred_mask)[0]
    crit[p_idx], alarm[p_idx] = criticality(anomalous[p_idx], monitored[p_idx], args.crit_limit)

    out = df.loc[pred_mask, ["time_stamp", "id", "status_type_id"]].copy()
    out["monitored"] = monitored[pred_mask]
    out = out.join(sub.loc[pred_mask].round(3)).join(z.loc[pred_mask].round(3).add_prefix("z_"))
    out["anomaly_score"] = score[pred_mask].round(3)
    out["threshold"] = round(threshold, 3)
    out["criticality"] = crit[pred_mask]
    out["alarm"] = alarm[pred_mask]

    # ---------------- evaluation ----------------
    label = str(event.get("event_label", "unknown")).lower()
    ev_start, ev_end = event.get("event_start_id"), event.get("event_end_id")
    in_event = out["id"].between(ev_start, ev_end) if label == "anomaly" else pd.Series(False, index=out.index)
    out["in_event_window"] = in_event.to_numpy()

    res = {"dataset_id": ds_id, "asset": event.get("asset"), "label": label,
           "fault": event.get("event_description") if label == "anomaly" else "",
           "threshold": round(threshold, 2), "n_models": len(models),
           "pred_status_normal_share": round(pred_status_normal_share, 3)}

    # false alarm = alarm in a normal dataset, or BEFORE the fault window of an
    # anomaly dataset (after the fault the turbine is broken / being repaired)
    if label == "anomaly":
        alarms_outside = out["alarm"] & (out["id"] < ev_start)
    else:
        alarms_outside = out["alarm"]
    res["false_alarm"] = bool(alarms_outside.any())
    res["false_alarm_hours"] = round(alarms_outside.sum() / STEPS_PER_HOUR, 1)
    # accuracy on normal-status timestamps outside the fault window
    nm = out["monitored"] & (~out["in_event_window"] if label != "anomaly" else (out["id"] < ev_start))
    res["normal_accuracy"] = round(1 - out.loc[nm, "alarm"].mean(), 4) if nm.any() else np.nan

    if label == "anomaly":
        hits = out[out["alarm"] & out["in_event_window"]]
        res["detected"] = not hits.empty
        if res["detected"]:
            first = hits.iloc[0]
            end_ts = out.loc[out["id"] == ev_end, "time_stamp"]
            end_ts = end_ts.iloc[0] if len(end_ts) else out["time_stamp"].iloc[-1]
            res["first_alarm"] = str(first["time_stamp"])
            res["lead_time_days"] = round((end_ts - first["time_stamp"]).total_seconds() / 86400, 2)
            diag = sub.loc[first.name].idxmax()
            res["top_sensor"] = ""
            if diag == "Whole turbine (AE)" and ae_feat_err is not None:
                # which sensor does the autoencoder reconstruct worst (last 6 h)?
                i0 = df.index.get_loc(first.name)
                worst = ae_cols[int(ae_feat_err[max(0, i0 - 36):i0 + 1].mean(axis=0).argmax())]
                res["top_sensor"] = SENSOR_NAMES.get(worst, worst)
                diag = SENSOR_TO_SUBSYSTEM.get(worst, diag)
            res["diagnosed_subsystem"] = diag
            exp = next((v for k, v in FAULT_TO_SUBSYSTEM.items()
                        if k in str(res["fault"]).lower()), None)
            res["expected_subsystem"] = " / ".join(exp) if exp else None
            res["diagnosis_correct"] = (diag in exp) if exp else None
        # v2 near-miss report: how close did the score get inside the fault window?
        wsc = sub.loc[out.index[out["in_event_window"]]]
        if len(wsc) and wsc.notna().any().any():
            peak = wsc.max()
            res["peak_component_in_window"] = peak.idxmax()
            res["peak_score_vs_threshold"] = round(float(peak.max() / threshold), 2)
            res["hours_above_threshold_in_window"] = round(float((wsc.max(axis=1) > threshold).sum()) / STEPS_PER_HOUR, 1)
        # coverage: share of the event window that is flagged anomalous
        w = out[out["in_event_window"] & out["monitored"]]
        res["event_coverage"] = round(float((w["anomaly_score"] > threshold).mean()), 3) if len(w) else np.nan
    return res, out, models, threshold


def plot_dataset(ds_id, out, res, threshold, crit_limit, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    t = out["time_stamp"]
    sub_cols = [c for c in SUBSYSTEMS if c in out]
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[::2].tolist() + ["#000000"]
    for c, col in zip(sub_cols, colors):
        ax1.plot(t, out[c], lw=1, color=col, label=c)
    ax1.axhline(threshold, color="black", ls="--", lw=1, label=f"Threshold ({threshold:.1f})")
    if out["in_event_window"].any():
        w = out.loc[out["in_event_window"], "time_stamp"]
        for ax in (ax1, ax2):
            ax.axvspan(w.min(), w.max(), color="#d1492e", alpha=0.12)
        ax1.text(w.min(), ax1.get_ylim()[1] * 0.95, "  fault window", color="#d1492e", va="top")
    ax1.set_ylabel("Smoothed z-score")
    title = f"Dataset {ds_id} - asset {res['asset']} - {res['label']}"
    if res["fault"]:
        title += f" ({res['fault']})"
    ax1.set_title(title)
    ax1.legend(fontsize=7, ncol=3, loc="upper left")
    ax2.plot(t, out["criticality"], color="#33404d", lw=1)
    ax2.axhline(crit_limit, color="#d1492e", ls="--", lw=1)
    ax2.set_ylabel("Criticality")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------
# Threshold sweep: detection vs false alarms for many settings
# --------------------------------------------------------------------------
def run_sweep(score_dir, out_dir, thresholds=None, crit_limits=(18, 36, 72, 144)):
    thresholds = thresholds if thresholds is not None else np.arange(1.0, 10.01, 0.5)
    frames = {}
    for f in glob.glob(os.path.join(score_dir, "*.csv")):
        d = pd.read_csv(f, usecols=lambda c: c in ("id", "anomaly_score", "monitored", "in_event_window"))
        if "monitored" not in d:
            d["monitored"] = True
        frames[int(os.path.splitext(os.path.basename(f))[0])] = d
    if not frames:
        print("No score files found for the sweep."); return None

    # outcome of every dataset for every setting
    combos = [(float(T), int(C)) for T in thresholds for C in crit_limits]
    outcome = []                        # rows: dataset, T, C, is_anomaly, detected, false_alarm
    for ds, d in frames.items():
        win = d["in_event_window"].to_numpy(bool)
        is_anom = bool(win.any())
        ev_start = d.loc[win, "id"].min() if is_anom else None
        mon = d["monitored"].to_numpy(bool)
        for T, C in combos:
            anom = (d["anomaly_score"] > T).fillna(False).to_numpy()
            _, alarm = criticality(anom, mon, C)
            det = bool((alarm & win).any()) if is_anom else False
            fa = bool((alarm & (d["id"].to_numpy() < ev_start)).any()) if is_anom else bool(alarm.any())
            outcome.append((ds, T, C, is_anom, det, fa))
    oc = pd.DataFrame(outcome, columns=["dataset", "T", "C", "is_anom", "detected", "false_alarm"])

    def score_table(sub):
        g = sub.groupby(["T", "C"])
        t = pd.DataFrame({
            "detected": g.apply(lambda x: int(x.loc[x.is_anom, "detected"].sum()), include_groups=False),
            "anomaly_datasets": g.apply(lambda x: int(x.is_anom.sum()), include_groups=False),
            "normal_with_false_alarm": g.apply(lambda x: int(x.loc[~x.is_anom, "false_alarm"].sum()), include_groups=False),
            "normal_datasets": g.apply(lambda x: int((~x.is_anom).sum()), include_groups=False),
            "anomaly_with_alarm_before_fault": g.apply(lambda x: int(x.loc[x.is_anom, "false_alarm"].sum()), include_groups=False),
        }).reset_index()
        tp, fp = t["detected"], t["normal_with_false_alarm"]
        t["detection_rate"] = tp / t["anomaly_datasets"].clip(lower=1)
        t["event_precision"] = (tp / (tp + fp)).where(tp + fp > 0)
        t["event_F1"] = (2 * tp / (2 * tp + fp + (t["anomaly_datasets"] - tp))).fillna(0)
        # balanced accuracy: mean of "faults caught" and "healthy turbines left alone".
        # Alarming on everything scores 0.5 (= guessing), unlike F1 which rewards it.
        # Alarms before the fault window of an anomaly dataset also count as false alarms.
        fa_any = t["normal_with_false_alarm"] + t["anomaly_with_alarm_before_fault"]
        n_all = t["normal_datasets"] + t["anomaly_datasets"]
        t["false_alarm_rate"] = fa_any / n_all.clip(lower=1)
        t["balanced_accuracy"] = 0.5 * (t["detection_rate"]
                                        + 1 - t["normal_with_false_alarm"] / t["normal_datasets"].clip(lower=1))
        t["selection_score"] = t["balanced_accuracy"] - 0.5 * t["anomaly_with_alarm_before_fault"] / \
            t["anomaly_datasets"].clip(lower=1) * t["detection_rate"].clip(upper=1)
        return t

    def pick_best(t):   # best balanced accuracy; ties -> fewer false alarms, stricter setting
        return t.sort_values(["selection_score", "normal_with_false_alarm", "detected", "T", "C"],
                             ascending=[False, True, False, False, False]).iloc[0]

    sw = score_table(oc).rename(columns={"T": "threshold", "C": "crit_limit_steps"})
    sw["crit_limit_hours"] = sw["crit_limit_steps"] / STEPS_PER_HOUR
    sw.to_csv(os.path.join(out_dir, "sweep.csv"), index=False)

    # ---- leave-one-dataset-out cross-validation (honest estimate) ----
    cv_rows = []
    for ds in frames:
        best = pick_best(score_table(oc[oc.dataset != ds]))
        r = oc[(oc.dataset == ds) & (oc["T"] == best["T"]) & (oc["C"] == best["C"])].iloc[0]
        cv_rows.append({"dataset": ds, "is_anomaly": bool(r.is_anom), "chosen_threshold": best["T"],
                        "chosen_alarm_hours": best["C"] / STEPS_PER_HOUR,
                        "detected": bool(r.detected), "false_alarm": bool(r.false_alarm)})
    cv = pd.DataFrame(cv_rows).sort_values("dataset")
    cv.to_csv(os.path.join(out_dir, "sweep_crossvalidation.csv"), index=False)
    an, no = cv[cv.is_anomaly], cv[~cv.is_anomaly]
    cv_tp, cv_fp = int(an.detected.sum()), int(no.false_alarm.sum())
    cv_f1 = 2 * cv_tp / (2 * cv_tp + cv_fp + (len(an) - cv_tp)) if len(an) else 0
    cv_bacc = 0.5 * (cv_tp / max(len(an), 1) + 1 - cv_fp / max(len(no), 1))
    cv_summary = {"detected": cv_tp, "anomaly_datasets": int(len(an)),
                  "balanced_accuracy": round(cv_bacc, 3),
                  "normal_with_false_alarm": cv_fp, "normal_datasets": int(len(no)),
                  "anomaly_with_alarm_before_fault": int(an.false_alarm.sum()),
                  "event_F1": round(cv_f1, 3),
                  "most_chosen_setting": cv.groupby(["chosen_threshold", "chosen_alarm_hours"]).size().idxmax()}
    with open(os.path.join(out_dir, "sweep_crossvalidation.json"), "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for C, g in sw.groupby("crit_limit_steps"):
        g = g.sort_values("threshold")
        ax.plot(g["normal_with_false_alarm"], g["detection_rate"] * 100, marker="o", ms=3,
                label=f"alarm after {C / STEPS_PER_HOUR:g} h")
    ax.scatter([cv_fp], [100 * cv_tp / max(len(an), 1)], s=120, marker="*", color="black", zorder=5,
               label="cross-validated result")
    ax.set_xlabel(f"Normal datasets with a false alarm (of {sw['normal_datasets'].iloc[0]})")
    ax.set_ylabel("Faults detected (%)")
    ax.set_title("Fault detection trade-off (each point = one threshold)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "sweep.png"), dpi=140); plt.close(fig)

    best5 = sw.sort_values(["selection_score", "normal_with_false_alarm", "detection_rate"],
                           ascending=[False, True, False]).head(5)
    print("\n==========  THRESHOLD SWEEP (best 5 by balanced accuracy, optimistic)  ==========")
    print("threshold  alarm-after  detected  normal-false-alarms  early-alarms  bal.acc")
    for _, b in best5.iterrows():
        print(f"{b['threshold']:>8.1f}  {b['crit_limit_hours']:>8.0f} h  "
              f"{int(b['detected']):>4}/{int(b['anomaly_datasets'])}  "
              f"{int(b['normal_with_false_alarm']):>10}/{int(b['normal_datasets'])}  "
              f"{int(b['anomaly_with_alarm_before_fault']):>8}/{int(b['anomaly_datasets'])}  "
              f"{b['balanced_accuracy']:.2f}")
    print("\n==========  CROSS-VALIDATED (honest - report this)  ==========")
    print(f"Faults detected              : {cv_tp}/{len(an)}")
    print(f"Normal datasets false alarm  : {cv_fp}/{len(no)}")
    print(f"Alarms before fault window   : {cv_summary['anomaly_with_alarm_before_fault']}/{len(an)}")
    print(f"Balanced accuracy            : {cv_bacc:.2f}   (0.5 = guessing, 1.0 = perfect)")
    print(f"Event F1                     : {cv_f1:.2f}")
    print(f"Most often chosen setting    : threshold {cv_summary['most_chosen_setting'][0]}, "
          f"alarm after {cv_summary['most_chosen_setting'][1]:g} h")
    return sw


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--farm-dir")
    ap.add_argument("--out")
    ap.add_argument("--datasets", type=int, nargs="*", help="only these dataset ids")
    ap.add_argument("--smooth-hours", type=float, default=6)
    ap.add_argument("--slow-hours", type=float, default=24, help="v2: second, slower time scale")
    ap.add_argument("--sub-q", type=float, default=0.995, help="v2: quantile = each component's normal level")
    ap.add_argument("--sub-floor", type=float, default=1.0, help="v2: smallest normal level (z units)")
    ap.add_argument("--threshold-q", type=float, default=0.995,
                    help="quantile of the healthy calibration score used as threshold")
    ap.add_argument("--min-threshold", type=float, default=1.5,
                    help="v2 scores are relative: 1.0 = top of the component's normal range")
    ap.add_argument("--crit-limit", type=int, default=18, help="18 x 10 min = 3 h")
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--no-ae", action="store_true", help="skip the autoencoder (faster)")
    ap.add_argument("--max-fit-rows", type=int, default=25000,
                    help="max training rows per model (random subset, for speed)")
    ap.add_argument("--sweep-only", action="store_true",
                    help="only run the threshold sweep on saved scores (no training)")
    ap.add_argument("--use-status-in-prediction", action="store_true",
                    help="also skip non-normal status in the prediction period "
                         "(NOT recommended for Wind Farm A, see README)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.sweep_only:
        out_dir = args.out or os.path.join(here, "outputs_fault_detection_v2")
        run_sweep(os.path.join(out_dir, "scores"), out_dir, thresholds=V2_THRESHOLDS)
        return
    farm = args.farm_dir or find_farm_dir(here)
    if not farm:
        sys.exit("Could not find 'Wind Farm A'. Pass --farm-dir.")
    out_dir = args.out or os.path.join(here, "outputs_fault_detection_v2")
    for sub in ("scores", "plots", "models"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    print(f"Wind Farm A folder: {farm}")

    events = read_csv(os.path.join(farm, "event_info.csv"))
    events = {int(r["event_id"]): r.to_dict() for _, r in events.iterrows()}
    files = dataset_files(farm)
    if args.datasets:
        files = {k: v for k, v in files.items() if k in args.datasets}

    results = []
    t0 = time.time()
    for i, (ds_id, path) in enumerate(files.items(), 1):
        ev = events.get(ds_id, {})
        res, out, models, thr = process_dataset(ds_id, path, ev, args)
        results.append(res)
        out.to_csv(os.path.join(out_dir, "scores", f"{ds_id}.csv"), index=False)
        plot_dataset(ds_id, out, res, thr, args.crit_limit, os.path.join(out_dir, "plots", f"{ds_id}.png"))
        if args.save_models:
            joblib.dump(models, os.path.join(out_dir, "models", f"{ds_id}.joblib"))

        if res["label"] == "anomaly":
            status = (f"DETECTED {res['lead_time_days']:>5} days before end -> {res['diagnosed_subsystem']}"
                      + (f" ({res['top_sensor']})" if res.get("top_sensor") else "")
                      if res.get("detected") else "missed")
        else:
            status = "false alarm!" if res["false_alarm"] else "ok (no alarm)"
        print(f"[{i:>2}/{len(files)}] dataset {ds_id:>3} asset {str(res['asset']):>3} "
              f"{res['label']:<8} {str(res['fault'] or ''):<26} {status}"
              f"   [normal status in prediction: {res['pred_status_normal_share']:.0%}]"
              f"   ({time.time() - t0:.0f}s)")

    r = pd.DataFrame(results)
    r.to_csv(os.path.join(out_dir, "results_per_dataset.csv"), index=False)

    an, no = r[r["label"] == "anomaly"], r[r["label"] == "normal"]
    det = an[an.get("detected", False) == True] if "detected" in an else an.iloc[0:0]
    tp, fn = len(det), len(an) - len(det)
    fp_norm = int(no["false_alarm"].sum())
    precision = tp / (tp + fp_norm) if (tp + fp_norm) else float("nan")
    recall = tp / len(an) if len(an) else float("nan")
    diag = det["diagnosis_correct"].dropna() if "diagnosis_correct" in det else pd.Series(dtype=float)
    summary = {
        "anomaly_datasets": int(len(an)), "normal_datasets": int(len(no)),
        "detected": tp, "missed": fn,
        "detection_rate": round(recall, 3),
        "normal_datasets_with_false_alarm": fp_norm,
        "event_precision": round(precision, 3),
        "event_F1": round(2 * precision * recall / (precision + recall), 3) if tp else 0.0,
        "mean_lead_time_days": round(float(det["lead_time_days"].mean()), 2) if tp else None,
        "median_lead_time_days": round(float(det["lead_time_days"].median()), 2) if tp else None,
        "mean_normal_accuracy": round(float(r["normal_accuracy"].mean()), 4),
        "mean_event_coverage": round(float(an["event_coverage"].mean()), 3) if "event_coverage" in an else None,
        "diagnosis_correct": f"{int(diag.sum())}/{len(diag)}" if len(diag) else "n/a",
        "settings": vars(args) | {"farm_dir": farm},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=====================  SUMMARY  =====================")
    print(f"Fault datasets detected      : {tp}/{len(an)}  (detection rate {recall:.0%})")
    print(f"Normal datasets false alarm  : {fp_norm}/{len(no)}")
    print(f"Event precision / F1         : {precision:.2f} / {summary['event_F1']}")
    if tp:
        print(f"Warning before fault (days)  : mean {summary['mean_lead_time_days']}, "
              f"median {summary['median_lead_time_days']}")
    print(f"Accuracy on normal timestamps: {summary['mean_normal_accuracy']:.3f}")
    print(f"Correct subsystem diagnosed  : {summary['diagnosis_correct']}")
    if not args.datasets:
        run_sweep(os.path.join(out_dir, "scores"), out_dir, thresholds=V2_THRESHOLDS)
    near = r[r["label"] == "anomaly"][[c for c in ("dataset_id", "fault", "detected", "peak_component_in_window",
                                                 "peak_score_vs_threshold", "hours_above_threshold_in_window")
                                       if c in r]]
    print("\n==========  FAULTS: how close did the score get?  (peak / threshold, 1.0 = reached)  ==========")
    print(near.to_string(index=False))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
