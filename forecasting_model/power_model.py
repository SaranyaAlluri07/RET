"""
Wind turbine power prediction model (Digital Twin - Part 1)
Dataset: CARE to Compare - Wind Turbine SCADA Data For Early Fault Detection
         https://zenodo.org/records/14958989  (use Wind Farm A)

What it does
------------
1. Loads every Wind Farm A dataset + event_info.csv + feature_description.csv
2. Finds the wind speed, active power, ambient temperature and wind direction
   columns automatically (only the *_avg columns are used, as the dataset
   authors recommend)
3. Cleans the data: keeps status_type_id == 0 (normal operation) for training,
   removes duplicates and obvious stoppage/curtailment points
4. Builds weather features: wind speed (and v^2, v^3), air density from
   ambient temperature, density-corrected wind speed (IEC 61400-12),
   wind direction (sin/cos), turbulence proxy, time-of-day / season
5. Trains two models
     a) PHYSICS baseline : binned power curve (IEC method of bins)
     b) ML model         : Gradient Boosting regressor
6. Evaluates on the "prediction" part of the NORMAL datasets
   (unseen data, no fault in it) and writes metrics, plots,
   the saved model and predictions for every test row
   (those predictions are reused later for fault detection)

Usage
-----
    pip install pandas numpy scikit-learn matplotlib joblib
    python power_model.py --farm-dir "path/to/Care_To_Compare/Wind Farm A"

If the automatic column detection picks the wrong sensor, override it, e.g.
    python power_model.py --farm-dir "..." --wind-col wind_speed_3_avg --power-col power_30_avg
"""

import argparse
import glob
import json
import os
# hide harmless sklearn/joblib version warnings, also in the parallel worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

R_DRY_AIR = 287.05      # J/(kg K)
P_STD = 101325.0        # Pa (no pressure sensor in Farm A -> standard pressure)
RHO_REF = 1.225         # kg/m^3, IEC reference density
SAMPLES_PER_HOUR = 6    # 10-minute SCADA data


# --------------------------------------------------------------------------
# File helpers
# --------------------------------------------------------------------------
def sniff_sep(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    return ";" if first.count(";") > first.count(",") else ","


def read_csv(path, **kw):
    try:
        return pd.read_csv(path, sep=sniff_sep(path), **kw)
    except UnicodeDecodeError:          # e.g. the ° sign saved in Windows encoding
        return pd.read_csv(path, sep=sniff_sep(path), encoding="latin-1", **kw)


def find_file(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    return hits[0] if hits else None


def find_dataset_files(root):
    files = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    out = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.isdigit():
            out[int(stem)] = f
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# Column detection
# --------------------------------------------------------------------------
AVG_SUFFIX = re.compile(r"_(avg|average|mean)$", re.I)


def base_name(col):
    return AVG_SUFFIX.sub("", col)


def load_descriptions(feat_desc_path):
    """Return ({sensor_base_name: lowercase description}, {sensor_base_name: unit})."""
    if not feat_desc_path:
        return {}, {}
    fd = read_csv(feat_desc_path)
    cols = {c.lower(): c for c in fd.columns}
    name_col = cols.get("sensor_name", fd.columns[0])
    desc_col = cols.get("description")
    unit_col = cols.get("unit")
    desc, units = {}, {}
    for _, row in fd.iterrows():
        name = str(row[name_col]).strip()
        desc[name] = str(row[desc_col]).lower() if desc_col and pd.notna(row[desc_col]) else ""
        units[name] = str(row[unit_col]).strip() if unit_col and pd.notna(row[unit_col]) else ""
    return desc, units


# Known correct columns for CARE-to-Compare Wind Farm A (from feature_description.csv)
FARM_A_DEFAULTS = {"wind": "wind_speed_3_avg",      # Windspeed (m/s)
                   "power": "power_30_avg",         # Grid power (kW)
                   "temp": "sensor_0_avg",          # Ambient temperature
                   "direction": "sensor_1_avg"}     # Wind absolute direction
ENERGY_UNITS = {"wh", "kwh", "varh", "kvarh", "mwh"}


def pick_column(avg_cols, descriptions, include, exclude=(), prefer=(), name_hint=None,
                units=None, bad_units=()):
    """Score every *_avg column by its description and name, return the best."""
    best, best_score = None, 0
    units = units or {}
    for col in avg_cols:
        b = base_name(col)
        if units.get(b, "").lower() in bad_units:
            continue
        text = descriptions.get(b, "") + " " + b.lower().replace("_", " ")
        if not all(any(k in text for k in group) for group in include):
            continue
        if any(k in text for k in exclude):
            continue
        score = 10 + sum(3 for k in prefer if k in text)
        if name_hint and b.lower().startswith(name_hint):
            score += 2
        if score > best_score:
            best, best_score = col, score
    return best


def detect_columns(avg_cols, descriptions, units, args):
    overrides = {"wind": args.wind_col, "power": args.power_col,
                 "temp": args.temp_col, "direction": args.dir_col}
    if all(c in avg_cols for c in FARM_A_DEFAULTS.values()):
        return {k: overrides[k] or v for k, v in FARM_A_DEFAULTS.items()}

    cols = {
        "wind": args.wind_col or pick_column(
            avg_cols, descriptions,
            include=[("wind speed", "windspeed")],
            exclude=("std", "estimated", "theoretical"),
            prefer=("average", "nacelle", "anemometer"),
            name_hint="wind_speed"),
        "power": args.power_col or pick_column(
            avg_cols, descriptions,
            include=[("power",), ("active", "grid", "produced", "output")],
            exclude=("reactive", "possible", "available", "setpoint", "set point",
                     "limit", "consum", "converter", "inverter", "apparent"),
            prefer=("grid", "active power"),
            name_hint="power", units=units, bad_units=ENERGY_UNITS),
        "temp": args.temp_col or pick_column(
            avg_cols, descriptions,
            include=[("ambient", "outside", "outdoor", "external")],
            exclude=(),
            prefer=("temperature", "ambient temperature"),
            name_hint="sensor"),
        "direction": args.dir_col or pick_column(
            avg_cols, descriptions,
            include=[("wind direction", "absolute direction", "absolute wind")],
            exclude=("relative",),
            prefer=("absolute",)),
    }
    if cols["direction"] is None:
        cols["direction"] = pick_column(
            avg_cols, descriptions,
            include=[("nacelle direction", "nacelle position", "yaw position")])
    return cols


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_farm(farm_dir, args):
    event_path = find_file(farm_dir, "event_info.csv")
    feat_path = find_file(farm_dir, "feature_description.csv")
    ds_files = find_dataset_files(farm_dir)
    if not ds_files:
        sys.exit(f"No dataset CSVs (0.csv, 1.csv, ...) found under {farm_dir}")

    events = read_csv(event_path) if event_path else pd.DataFrame()
    descriptions, units = load_descriptions(feat_path)

    header = read_csv(next(iter(ds_files.values())), nrows=0).columns.tolist()
    avg_cols = [c for c in header if AVG_SUFFIX.search(c)]
    cols = detect_columns(avg_cols, descriptions, units, args)

    print("\nSelected SCADA columns (check these!):")
    for k, v in cols.items():
        d = descriptions.get(base_name(v), "") if v else ""
        print(f"  {k:<10} -> {v}   {('| ' + d[:70]) if d else ''}")
    if cols["wind"] is None or cols["power"] is None:
        sys.exit("Could not find wind speed / power column. Pass --wind-col and --power-col.")

    meta = [c for c in ("time_stamp", "asset_id", "id", "train_test", "status_type_id") if c in header]
    use = meta + [c for c in cols.values() if c]

    # event label per dataset (normal / anomaly)
    label = {}
    if not events.empty:
        id_col = next((c for c in events.columns if c.lower() in ("event_id", "id")), events.columns[0])
        lab_col = next((c for c in events.columns if "label" in c.lower()), None)
        if lab_col:
            label = dict(zip(events[id_col].astype(int), events[lab_col].astype(str).str.lower()))

    frames = []
    for ds_id, path in ds_files.items():
        df = read_csv(path, usecols=lambda c: c in use)
        df["dataset_id"] = ds_id
        df["event_label"] = label.get(ds_id, "unknown")
        frames.append(df)
        print(f"  loaded dataset {ds_id:>3}: {len(df):>7} rows  ({label.get(ds_id, 'unknown')})")
    data = pd.concat(frames, ignore_index=True)

    rename = {v: k for k, v in cols.items() if v}
    data = data.rename(columns=rename)
    data["time_stamp"] = pd.to_datetime(data["time_stamp"])
    if "asset_id" not in data:
        data["asset_id"] = 0
    if "train_test" not in data:
        data["train_test"] = "train"
    if "status_type_id" not in data:
        data["status_type_id"] = 0
    data["train_test"] = data["train_test"].astype(str).str.lower()
    return data, cols


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def add_features(df):
    df = df.sort_values(["dataset_id", "time_stamp"]).copy()

    df["wind"] = df["wind"].clip(lower=0)
    df["wind2"] = df["wind"] ** 2
    df["wind3"] = df["wind"] ** 3

    # turbulence proxy: rolling std of the 10-min average wind speed (1 hour)
    df["wind_std_1h"] = (df.groupby("dataset_id")["wind"]
                           .transform(lambda s: s.rolling(SAMPLES_PER_HOUR, min_periods=2).std()))
    df["wind_change"] = df.groupby("dataset_id")["wind"].diff()

    if "temp" in df:
        t = df["temp"].astype(float)
        t_k = np.where(t.median() > 200, t, t + 273.15)       # handle Kelvin or Celsius
        df["air_density"] = P_STD / (R_DRY_AIR * t_k)
    else:
        df["air_density"] = RHO_REF
    # IEC 61400-12 density-normalised wind speed
    df["wind_norm"] = df["wind"] * (df["air_density"] / RHO_REF) ** (1 / 3)

    if "direction" in df:
        rad = np.deg2rad(df["direction"].astype(float) % 360)
        df["dir_sin"], df["dir_cos"] = np.sin(rad), np.cos(rad)

    # timestamps are shifted by whole years only, so hour / season still make sense
    hour = df["time_stamp"].dt.hour + df["time_stamp"].dt.minute / 60
    doy = df["time_stamp"].dt.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return df


def feature_list(df):
    feats = ["wind", "wind2", "wind3", "wind_norm", "wind_std_1h", "wind_change",
             "air_density", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    if "temp" in df:
        feats.append("temp")
    if "dir_sin" in df:
        feats += ["dir_sin", "dir_cos"]
    return feats


def clean_for_training(df, p_max):
    """Normal operation only, and drop stopped / curtailed points at high wind."""
    m = (df["status_type_id"] == 0) & df["wind"].notna() & df["power"].notna()
    m &= ~((df["wind"] > 5) & (df["power"] <= 0.01 * p_max))   # stopped although windy
    m &= df["power"] >= -0.02 * p_max
    return df[m]


# --------------------------------------------------------------------------
# Physics baseline: binned power curve
# --------------------------------------------------------------------------
class BinnedPowerCurve:
    def __init__(self, bin_width=0.5):
        self.bin_width = bin_width

    def fit(self, wind_norm, power):
        bins = np.floor(wind_norm / self.bin_width)
        tab = pd.DataFrame({"b": bins, "v": wind_norm, "p": power}).groupby("b").agg(
            v=("v", "mean"), p=("p", "mean"), n=("p", "size"))
        tab = tab[tab["n"] >= 5]
        self.v_, self.p_ = tab["v"].to_numpy(), tab["p"].to_numpy()
        return self

    def predict(self, wind_norm):
        return np.interp(wind_norm, self.v_, self.p_, left=0.0, right=self.p_[-1])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def metrics(y, yhat, p_max):
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae = float(mean_absolute_error(y, yhat))
    return {"RMSE": rmse, "MAE": mae, "R2": float(r2_score(y, yhat)),
            "nRMSE_%": 100 * rmse / p_max, "nMAE_%": 100 * mae / p_max, "n": int(len(y))}


def make_plots(test, train, curve, model, feats, out):
    os.makedirs(out, exist_ok=True)
    s = test.sample(min(20000, len(test)), random_state=0)

    # 1. power curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(s["wind"], s["power"], s=2, alpha=0.25, color="#9aa5b1", label="Measured (test)")
    ax.scatter(s["wind"], s["pred_ml"], s=2, alpha=0.25, color="#2f6fde", label="ML prediction")
    v = np.linspace(0, curve.v_.max(), 200)
    ax.plot(v, curve.predict(v), color="#d1492e", lw=2, label="Binned power curve (physics)")
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Active power")
    ax.set_title("Power curve - measured vs model"); ax.legend(markerscale=5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "power_curve.png"), dpi=140); plt.close(fig)

    # 2. predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(test["power"], test["pred_ml"], gridsize=60, bins="log", cmap="Blues")
    lim = [test["power"].min(), test["power"].max()]
    ax.plot(lim, lim, color="#d1492e", lw=1)
    ax.set_xlabel("Actual power"); ax.set_ylabel("Predicted power")
    ax.set_title("ML model: predicted vs actual (test)")
    fig.colorbar(hb, ax=ax, label="log count")
    fig.tight_layout(); fig.savefig(os.path.join(out, "pred_vs_actual.png"), dpi=140); plt.close(fig)

    # 3. one week time series
    ds = test["dataset_id"].value_counts().index[0]
    w = test[test["dataset_id"] == ds].sort_values("time_stamp").head(7 * 24 * SAMPLES_PER_HOUR)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(w["time_stamp"], w["power"], color="#33404d", lw=1, label="Actual")
    ax.plot(w["time_stamp"], w["pred_phys"], color="#d1492e", lw=1, alpha=0.8, label="Physics curve")
    ax.plot(w["time_stamp"], w["pred_ml"], color="#2f6fde", lw=1, label="ML model")
    ax.set_title(f"One week of test data - dataset {ds}"); ax.set_ylabel("Active power"); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(out, "timeseries_week.png"), dpi=140); plt.close(fig)

    # 4. error by wind speed bin
    t = test.copy()
    t["bin"] = (t["wind"] // 1).clip(upper=25)
    err = t.groupby("bin").apply(lambda g: pd.Series({
        "Physics": np.sqrt(np.mean((g["power"] - g["pred_phys"]) ** 2)),
        "ML": np.sqrt(np.mean((g["power"] - g["pred_ml"]) ** 2))}), include_groups=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(err.index, err["Physics"], marker="o", color="#d1492e", label="Physics curve")
    ax.plot(err.index, err["ML"], marker="o", color="#2f6fde", label="ML model")
    ax.set_xlabel("Wind speed bin (m/s)"); ax.set_ylabel("RMSE"); ax.set_title("Error by wind speed")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "error_by_windspeed.png"), dpi=140); plt.close(fig)

    # 5. permutation feature importance
    ps = test.sample(min(5000, len(test)), random_state=1)
    imp = permutation_importance(model, ps[feats], ps["power"], n_repeats=5, random_state=0, n_jobs=-1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feats)[order], imp.importances_mean[order], color="#2f6fde")
    ax.set_xlabel("Drop in R² when feature is shuffled"); ax.set_title("Feature importance")
    fig.tight_layout(); fig.savefig(os.path.join(out, "feature_importance.png"), dpi=140); plt.close(fig)
    return dict(zip(feats, map(float, imp.importances_mean)))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--farm-dir", default=None,
                    help="Folder of Wind Farm A (contains datasets/, event_info.csv). "
                         "If omitted, it is searched for next to this script.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wind-col"); ap.add_argument("--power-col")
    ap.add_argument("--temp-col"); ap.add_argument("--dir-col")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.farm_dir is None:
        hits = []
        for base in (here, os.path.dirname(here)):          # this folder, then the parent
            hits += [d for d in glob.glob(os.path.join(base, "**", "Wind Farm A"), recursive=True)
                     if os.path.isfile(os.path.join(d, "event_info.csv"))]
            if hits:
                break
        if not hits:
            sys.exit("Could not find a 'Wind Farm A' folder next to this script. "
                     "Pass it with --farm-dir \"path\\to\\Wind Farm A\"")
        args.farm_dir = hits[0]
    if args.out is None:
        args.out = os.path.join(here, "outputs_power_model")
    print(f"Wind Farm A folder: {args.farm_dir}")

    print("Loading data ...")
    data, cols = load_farm(args.farm_dir, args)
    data = data.drop_duplicates(subset=["asset_id", "time_stamp", "dataset_id"])
    data = add_features(data)
    feats = feature_list(data)

    p_max = float(data["power"].quantile(0.999))
    print(f"\nRated power estimate (99.9th pct): {p_max:.3f}")

    # ---- split: train part of all datasets, test = prediction part of NORMAL datasets
    train = clean_for_training(data[data["train_test"] == "train"], p_max)
    # several datasets come from the same turbine and can overlap in time.
    # Timestamps were shifted per dataset, so a row only counts as duplicate
    # if the measured values are identical too.
    dup_key = ["asset_id", "time_stamp", "wind", "power"]
    train = train.dropna(subset=["wind"]).drop_duplicates(subset=dup_key)
    test_all = data[(data["train_test"] != "train")].dropna(subset=["wind", "power"])
    normal_ids = data.loc[data["event_label"] == "normal", "dataset_id"].unique()
    test = clean_for_training(test_all[test_all["dataset_id"].isin(normal_ids)], p_max)
    if test.empty:   # fall back if labels missing
        test = clean_for_training(test_all, p_max)

    # avoid the same timestamps of one turbine sitting in train and test
    test = test[~test.set_index(dup_key).index.isin(train.set_index(dup_key).index)]
    print(f"Train rows: {len(train):,}   Test rows (normal datasets): {len(test):,}")

    # ---- a) physics baseline
    curve = BinnedPowerCurve().fit(train["wind_norm"].to_numpy(), train["power"].to_numpy())

    # ---- b) ML model
    model = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=42)
    model.fit(train[feats], train["power"])

    for df in (test, test_all):
        df["pred_phys"] = curve.predict(df["wind_norm"].to_numpy())
        df["pred_ml"] = np.clip(model.predict(df[feats]), 0, p_max * 1.02)

    results = {
        "columns_used": cols,
        "features": feats,
        "rated_power_estimate": p_max,
        "physics_power_curve": metrics(test["power"], test["pred_phys"], p_max),
        "ml_gradient_boosting": metrics(test["power"], test["pred_ml"], p_max),
        "ml_per_turbine": {
            str(a): metrics(g["power"], g["pred_ml"], p_max) for a, g in test.groupby("asset_id")},
    }

    print("\n================  TEST RESULTS (unseen normal data)  ================")
    print(f"{'Model':<24}{'RMSE':>10}{'MAE':>10}{'R²':>8}{'nRMSE %':>10}{'nMAE %':>9}")
    for name, key_ in (("Physics power curve", "physics_power_curve"),
                       ("Gradient Boosting (ML)", "ml_gradient_boosting")):
        m = results[key_]
        print(f"{name:<24}{m['RMSE']:>10.3f}{m['MAE']:>10.3f}{m['R2']:>8.3f}"
              f"{m['nRMSE_%']:>10.2f}{m['nMAE_%']:>9.2f}")
    print("\nPer turbine (ML):")
    for a, m in results["ml_per_turbine"].items():
        print(f"  asset {a:<6} R²={m['R2']:.3f}  nMAE={m['nMAE_%']:.2f}%  n={m['n']}")

    print("\nMaking plots ...")
    results["feature_importance"] = make_plots(test, train, curve, model, feats, args.out)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump({"model": model, "power_curve": curve, "features": feats,
                 "columns": cols, "rated_power": p_max},
                os.path.join(args.out, "power_model.joblib"))

    # predictions for ALL test rows (normal + anomaly datasets) -> used in fault detection step
    test_all["residual"] = test_all["power"] - test_all["pred_ml"]
    keep = ["dataset_id", "asset_id", "event_label", "time_stamp", "status_type_id",
            "wind", "power", "pred_phys", "pred_ml", "residual"]
    test_all[keep].to_csv(os.path.join(args.out, "test_predictions.csv"), index=False)

    print(f"\nDone. Everything saved in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
