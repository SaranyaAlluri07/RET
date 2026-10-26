"""
Short-term power forecasting (Digital Twin - future prediction)
===============================================================
Predicts the turbine's power 10 min, 1 h, 2 h and 3 h ahead from its recent
history (power, wind, direction, temperature, time of day/season).

Method
------
* "Direct" multi-horizon forecasting: one Gradient Boosting model per horizon.
* Uncertainty: two extra quantile models per horizon (10 % and 90 %), giving
  an 80 % prediction interval.
* Baseline: PERSISTENCE (power in h hours = power now), the standard benchmark.
  Skill score = 1 - RMSE_model / RMSE_persistence  (>0 means better than persistence).
* Training: "train" periods of all 22 datasets (normal status only).
  Test: unseen "prediction" periods of the 10 normal datasets (rows that also
  appear in another dataset's training data are removed).

Limitation: no weather-forecast input is available (timestamps are anonymised),
so accuracy decreases quickly with the horizon - this is reported honestly.

Outputs (outputs_forecast/):
    forecasts.csv          forecast + 80 % band for every prediction-period row (all datasets)
    metrics.json           RMSE / MAE / skill / interval coverage per horizon
    forecast_skill.png     model vs persistence error per horizon
    forecast_example.png   example of the forecast fan on test data

Usage:
    python forecast_model.py
    python forecast_model.py --farm-dir "path/to/Wind Farm A"
"""

import argparse
import glob
import json
import os
# hide harmless sklearn/joblib version warnings, also in the parallel worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)

STEP = pd.Timedelta(minutes=10)
HORIZONS = {1: "10 min", 6: "1 h", 12: "2 h", 18: "3 h"}       # steps ahead -> label
NORMAL_STATUS = (0, 2)
COLS = {"wind_speed_3_avg": "wind", "power_30_avg": "power", "sensor_0_avg": "temp", "sensor_1_avg": "direction"}


# ------------------------------------------------------------------ helpers
def sniff_sep(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    return ";" if first.count(";") > first.count(",") else ","


def find_farm_dir(here):
    for base in (here, os.path.dirname(here)):
        hits = [d for d in glob.glob(os.path.join(base, "**", "Wind Farm A"), recursive=True)
                if os.path.isfile(os.path.join(d, "event_info.csv"))]
        if hits:
            return hits[0]
    return None


def load_dataset(path):
    use = ["time_stamp", "asset_id", "train_test", "status_type_id", *COLS]
    df = pd.read_csv(path, sep=sniff_sep(path), usecols=lambda c: c in use)
    df = df.rename(columns=COLS)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.drop_duplicates("time_stamp").sort_values("time_stamp")
    return df


def make_features(df):
    """Features at time t use only information up to t. Gaps are respected by
    re-indexing to a regular 10-min grid before shifting."""
    g = df.set_index("time_stamp").asfreq(STEP)
    g["present"] = g["power"].notna() | g["wind"].notna()
    p, w = g["power"], g["wind"]
    f = pd.DataFrame(index=g.index)
    for lag in (0, 1, 2, 3, 6, 12, 18, 36):
        f[f"p_lag{lag}"] = p.shift(lag)
        f[f"w_lag{lag}"] = w.shift(lag)
    for h in (6, 18, 36):
        f[f"p_mean{h}"] = p.rolling(h, min_periods=max(2, h // 2)).mean()
        f[f"w_mean{h}"] = w.rolling(h, min_periods=max(2, h // 2)).mean()
        f[f"w_std{h}"] = w.rolling(h, min_periods=max(2, h // 2)).std()
    f["p_diff1"], f["p_diff6"] = p.diff(1), p.diff(6)
    f["w_diff1"], f["w_diff6"] = w.diff(1), w.diff(6)
    f["w3"] = w.clip(lower=0) ** 3
    if "direction" in g:
        r = np.deg2rad(g["direction"] % 360)
        f["dir_sin"], f["dir_cos"] = np.sin(r), np.cos(r)
    if "temp" in g:
        f["temp"] = g["temp"]
        f["rho"] = 101325.0 / (287.05 * (g["temp"] + 273.15))
    hour = g.index.hour + g.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    doy = g.index.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for h in HORIZONS:
        f[f"y{h}"] = p.shift(-h)
        f[f"ystat{h}"] = g["status_type_id"].shift(-h)
    meta = g[["asset_id", "train_test", "status_type_id", "present", "power"]].rename(columns={"power": "power_now"})
    out = pd.concat([meta, f], axis=1)
    out = out[out["present"]].drop(columns="present").reset_index()
    return out


def model(loss="squared_error", q=None):
    kw = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=60,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
              n_iter_no_change=15, random_state=0)
    return HistGradientBoostingRegressor(loss=loss, quantile=q, **kw) if loss == "quantile" \
        else HistGradientBoostingRegressor(loss=loss, **kw)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--farm-dir")
    ap.add_argument("--out")
    ap.add_argument("--max-train-rows", type=int, default=250000)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    farm = args.farm_dir or find_farm_dir(here)
    if not farm:
        sys.exit("Could not find 'Wind Farm A'. Pass --farm-dir.")
    out_dir = args.out or os.path.join(here, "outputs_forecast")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Wind Farm A folder: {farm}")

    ev = pd.read_csv(os.path.join(farm, "event_info.csv"), sep=sniff_sep(os.path.join(farm, "event_info.csv")))
    label = dict(zip(ev["event_id"].astype(int), ev["event_label"].astype(str).str.lower()))

    frames = []
    for path in sorted(glob.glob(os.path.join(farm, "**", "*.csv"), recursive=True)):
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.isdigit():
            continue
        ds = int(stem)
        f = make_features(load_dataset(path))
        f["dataset_id"], f["label"] = ds, label.get(ds, "unknown")
        frames.append(f)
        print(f"  dataset {ds:>3}: {len(f):>6} rows")
    data = pd.concat(frames, ignore_index=True)
    feats = [c for c in data.columns if c not in
             {"time_stamp", "asset_id", "train_test", "status_type_id", "power_now", "dataset_id", "label"}
             and not c.startswith("y")]

    is_train = data["train_test"].astype(str).str.lower() == "train"
    normal = data["status_type_id"].isin(NORMAL_STATUS)
    p_ref = float(data.loc[is_train & normal, "power_now"].quantile(0.999)) or 1.0
    # a stopped turbine draws a little power for its own systems -> measured power can be slightly
    # negative; forecasts and their range must be allowed down to that level, not cut at 0
    p_min = min(0.0, float(data.loc[is_train, "power_now"].quantile(0.001)))

    # test = prediction period of normal datasets, minus rows also present in training data
    train_keys = set(zip(data.loc[is_train, "asset_id"], data.loc[is_train, "time_stamp"]))
    pred_rows = ~is_train
    test_mask = pred_rows & (data["label"] == "normal") & normal
    test_mask &= ~pd.Series([k in train_keys for k in zip(data["asset_id"], data["time_stamp"])], index=data.index)

    rng = np.random.default_rng(0)
    out = data.loc[pred_rows, ["dataset_id", "time_stamp", "power_now"]].copy()
    metrics = {"rated_power_estimate": p_ref, "features": feats, "horizons": {}}
    print(f"\nTrain rows (before subsampling): {int((is_train & normal).sum()):,}   Test rows: {int(test_mask.sum()):,}")
    print(f"\n{'Horizon':<8}{'nRMSE model':>13}{'nRMSE persist.':>16}{'skill':>8}{'80% band coverage':>20}")

    for h, lab in HORIZONS.items():
        y, ys = f"y{h}", f"ystat{h}"
        tr = data[is_train & normal & data[y].notna() & data[ys].isin(NORMAL_STATUS)]
        if len(tr) > args.max_train_rows:
            tr = tr.iloc[np.sort(rng.choice(len(tr), args.max_train_rows, replace=False))]
        m50 = model().fit(tr[feats], tr[y])
        m10 = model("quantile", 0.10).fit(tr[feats], tr[y])
        m90 = model("quantile", 0.90).fit(tr[feats], tr[y])

        P = data.loc[pred_rows, feats]
        f50 = np.clip(m50.predict(P), p_min, p_ref * 1.02)
        lo, hi = np.clip(m10.predict(P), p_min, None), np.clip(m90.predict(P), p_min, p_ref * 1.02)
        lo, hi = np.minimum(lo, f50), np.maximum(hi, f50)
        out[f"f{h}"], out[f"lo{h}"], out[f"hi{h}"] = f50, lo, hi

        te = data[test_mask & data[y].notna()]
        Pt = te[feats]
        pf = np.clip(m50.predict(Pt), p_min, p_ref * 1.02)
        pl, ph = np.clip(m10.predict(Pt), p_min, None), np.clip(m90.predict(Pt), p_min, p_ref * 1.02)
        pers = te["power_now"].to_numpy()
        yt = te[y].to_numpy()
        ok = ~np.isnan(pers)
        rmse = float(np.sqrt(np.mean((yt[ok] - pf[ok]) ** 2)))
        rmse_p = float(np.sqrt(np.mean((yt[ok] - pers[ok]) ** 2)))
        mae = float(np.mean(np.abs(yt[ok] - pf[ok])))
        cov = float(np.mean((yt >= np.minimum(pl, pf)) & (yt <= np.maximum(ph, pf))))
        metrics["horizons"][lab] = {
            "steps_ahead": h, "nRMSE_%": 100 * rmse / p_ref, "nMAE_%": 100 * mae / p_ref,
            "nRMSE_persistence_%": 100 * rmse_p / p_ref, "skill_vs_persistence": 1 - rmse / rmse_p,
            "interval_coverage_80": cov, "test_rows": int(ok.sum())}
        mh = metrics["horizons"][lab]
        print(f"{lab:<8}{mh['nRMSE_%']:>12.2f}%{mh['nRMSE_persistence_%']:>15.2f}%{mh['skill_vs_persistence']:>8.2f}"
              f"{100 * cov:>19.0f}%")

    out.to_csv(os.path.join(out_dir, "forecasts.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- plots
    labs = list(metrics["horizons"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(labs, [metrics["horizons"][l]["nRMSE_persistence_%"] for l in labs], "o-", color="#9aa5b1", label="Persistence")
    ax.plot(labs, [metrics["horizons"][l]["nRMSE_%"] for l in labs], "o-", color="#0E9F8E", lw=2, label="Twin forecast (GBM)")
    for l in labs:
        ax.annotate(f"skill {metrics['horizons'][l]['skill_vs_persistence']:.2f}",
                    (l, metrics["horizons"][l]["nRMSE_%"]), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color="#0E9F8E")
    ax.set_ylabel("nRMSE (% of rated power)"); ax.set_xlabel("Forecast horizon")
    ax.set_title("Forecast error vs horizon (unseen normal data)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "forecast_skill.png"), dpi=140); plt.close(fig)

    ex_ds = data.loc[test_mask, "dataset_id"].value_counts().index[0] if test_mask.any() else out["dataset_id"].iloc[0]
    ex = out[out["dataset_id"] == ex_ds].head(3 * 144)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ex["time_stamp"], ex["power_now"] * 100 / p_ref, color="#33404d", lw=1.2, label="Actual")
    ax.plot(ex["time_stamp"] + 6 * STEP, ex["f6"] * 100 / p_ref, color="#0E9F8E", lw=1.2, label="Forecast made 1 h earlier")
    ax.fill_between(ex["time_stamp"] + 6 * STEP, ex["lo6"] * 100 / p_ref, ex["hi6"] * 100 / p_ref,
                    color="#0E9F8E", alpha=0.15, lw=0, label="80 % band")
    ax.set_ylabel("% of rated power"); ax.set_title(f"1-hour-ahead forecast - dataset {ex_ds}"); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "forecast_example.png"), dpi=140)
    plt.close(fig)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
