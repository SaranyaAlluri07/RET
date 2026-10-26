"""Power forecast (power from weather) with split-conformal prediction intervals per wind-speed regime.

    python -m windtwin.forecast          -> artifacts/forecast/forecast.json, artifacts/models/power_model.joblib

Audit of the legacy split (forecasting_model/power_model.py): it trains on the train rows of ALL datasets
and tests on the prediction rows of the normal datasets, de-duplicating by (asset, time_stamp, wind, power).
Timestamps are anonymised per file, so the same physical 10-min record appears under different time
stamps in overlapping datasets of one turbine -> that de-duplication cannot catch it. Here rows are
matched by a value fingerprint (asset + 8 signals) instead; every training row whose fingerprint occurs
in the test set is removed, and calibration rows (last 20 % of each dataset's train period) are removed
from the fit set the same way."""
import json
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C, data

R_DRY_AIR, P_STD, RHO_REF = 287.05, 101325.0, 1.225
REGIMES = [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 99.0)]
LEVELS = (0.8, 0.95)
FP_COLS = ["asset_id", C.WIND, C.POWER, C.AMBIENT, "sensor_11_avg", "sensor_12_avg", C.ROTOR_RPM, C.WIND_DIR,
           C.NACELLE]
FEATS = ["wind", "wind2", "wind3", "wind_norm", "wind_std_1h", "wind_change", "air_density", "temp",
         "dir_sin", "dir_cos", "hour_sin", "hour_cos"]
MODEL_PATH = C.MODELS_DIR / "power_model.joblib"


def features(df):
    """Weather features from raw columns (causal rolling within the given frame)."""
    w = df[C.WIND].astype(float).clip(lower=0).to_numpy()
    t = df[C.AMBIENT].astype(float).to_numpy()
    rho = P_STD / (R_DRY_AIR * (t + 273.15))
    ws = pd.Series(w)
    hour = pd.to_datetime(df["time_stamp"]).dt.hour.to_numpy() + pd.to_datetime(df["time_stamp"]).dt.minute.to_numpy() / 60
    r = np.deg2rad(df[C.WIND_DIR].astype(float).to_numpy() % 360)
    X = pd.DataFrame({"wind": w, "wind2": w ** 2, "wind3": w ** 3, "wind_norm": w * (rho / RHO_REF) ** (1 / 3),
                      "wind_std_1h": ws.rolling(6, min_periods=2).std().to_numpy(),
                      "wind_change": ws.diff().to_numpy(), "air_density": rho, "temp": t,
                      "dir_sin": np.sin(r), "dir_cos": np.cos(r),
                      "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24)})
    return X


def regime_of(wind):
    w = np.nan_to_num(np.asarray(wind, float))
    return np.clip(np.searchsorted([r[1] for r in REGIMES[:-1]], w, side="right"), 0, len(REGIMES) - 1)


class PowerModel:
    """predict(df) -> (p, lo, hi) in p.u.; df needs time_stamp, wind_speed_3_avg, sensor_0_avg, sensor_1_avg."""

    def __init__(self, model, curve, q, version=C.PIPELINE_VERSION):
        self.model, self.curve, self.q, self.version = model, curve, q, version

    def predict(self, df, level=0.8):
        X = features(df)
        p = np.clip(self.model.predict(X[FEATS]), 0.0, 1.02)
        qq = np.asarray(self.q[level])[regime_of(X["wind"])]
        bad = ~np.isfinite(X["wind"].to_numpy())
        p = np.where(bad, np.nan, p)
        return p, np.clip(p - qq, 0.0, None), np.clip(p + qq, None, 1.05)

    def physics(self, df):
        return self.curve.predict(features(df)["wind_norm"].to_numpy())


def load_model(missing_ok=False):
    if not MODEL_PATH.is_file():
        if missing_ok:
            return None
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def predict(df, level=0.8):
    """Module-level helper: (p, lo, hi) with the saved model."""
    return load_model().predict(df, level)


def _clean(df):
    m = df["status_type_id"].isin(C.NORMAL_STATUS) & df[C.WIND].notna() & df[C.POWER].notna()
    m &= ~((df[C.WIND] > 5) & (df[C.POWER] <= 0.01))
    m &= df[C.POWER] >= -0.02
    return df[m]


def _fingerprint(df):
    return pd.util.hash_pandas_object(df[FP_COLS].round(4), index=False).to_numpy()


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    ss = np.sum((y - y.mean()) ** 2)
    return {"mae": round(float(np.mean(np.abs(e))), 5), "rmse": round(float(np.sqrt(np.mean(e ** 2))), 5),
            "r2": round(float(1 - np.sum(e ** 2) / ss), 5), "n": int(len(y))}


def conformal_q(abs_res, level):
    n = len(abs_res)
    if n == 0:
        return np.inf
    k = min(n, int(np.ceil((n + 1) * level)))
    return float(np.sort(abs_res)[k - 1])


def run(log=print):
    t0 = time.time()
    ev = data.load_events()
    frames = []
    for ds in data.dataset_ids():
        d = data.load_dataset(ds)
        X = features(d)
        d = pd.concat([d[["time_stamp", "asset_id", "train_test", "status_type_id"] + [c for c in FP_COLS if c != "asset_id"]],
                       X.drop(columns=["temp"])], axis=1)
        d["dataset"], d["label"] = ds, ev.loc[ds, "event_label"]
        tr = np.flatnonzero(d["train_test"].to_numpy() == "train")
        d["cal"] = False
        d.loc[tr[int(len(tr) * (1 - C.VAL_FRACTION)):], "cal"] = True
        d["temp"] = X["temp"].to_numpy()
        frames.append(d)
    allr = pd.concat(frames, ignore_index=True)
    allr["fp"] = _fingerprint(allr)
    train_all = allr[allr["train_test"] == "train"]
    test_all = allr[(allr["train_test"] != "train") & (allr["label"] == "normal")]

    # ---- audit of the legacy split
    legacy_train = _clean(train_all).drop_duplicates(subset=["asset_id", "time_stamp", C.WIND, C.POWER])
    legacy_test = _clean(test_all)
    key = ["asset_id", "time_stamp", C.WIND, C.POWER]
    legacy_test = legacy_test[~legacy_test.set_index(key).index.isin(legacy_train.set_index(key).index)]
    overlap = float(np.isin(legacy_test["fp"], legacy_train["fp"]).mean())
    dup_test = float(legacy_test["fp"].duplicated().mean())
    log(f"audit: legacy test rows {len(legacy_test):,}, of which {overlap:.1%} also appear in legacy train (value fingerprint)")

    # ---- leakage-safe split
    test = _clean(test_all).drop_duplicates(subset="fp")
    train = _clean(train_all).drop_duplicates(subset="fp")
    train = train[~np.isin(train["fp"], test["fp"])]
    cal = train[train["cal"]]
    fit = train[~train["cal"] & ~np.isin(train["fp"], cal["fp"])]
    if len(fit) > 250000:
        fit = fit.sample(250000, random_state=0)
    log(f"split: fit {len(fit):,}  calibration {len(cal):,}  test {len(test):,} rows")
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=30, random_state=42).fit(fit[FEATS], fit[C.POWER])
    curve = data.PowerCurve().fit(fit["wind_norm"].to_numpy(), fit[C.POWER].to_numpy())
    pc = np.clip(model.predict(cal[FEATS]), 0, 1.02)
    rc = np.abs(cal[C.POWER].to_numpy() - pc)
    reg_c = regime_of(cal["wind"])
    q = {lv: [conformal_q(rc[reg_c == r], lv) for r in range(len(REGIMES))] for lv in LEVELS}
    pm = PowerModel(model, curve, q)

    pt = np.clip(model.predict(test[FEATS]), 0, 1.02)
    pp = curve.predict(test["wind_norm"].to_numpy())
    y = test[C.POWER].to_numpy()
    reg_t = regime_of(test["wind"])
    cov = []
    for r, (lo, hi) in enumerate(REGIMES):
        m = reg_t == r
        for lv in LEVELS:
            qq = q[lv][r]
            cov.append({"regime": f"{lo:g}-{hi:g} m/s" if hi < 99 else f">= {lo:g} m/s", "wind_lo": lo,
                        "wind_hi": hi if hi < 99 else None, "nominal": lv, "n": int(m.sum()),
                        "coverage": round(float(np.mean(np.abs(y[m] - pt[m]) <= qq)), 4) if m.any() else None,
                        "width": round(2 * qq, 4)})
    overall = {f"coverage_{int(lv * 100)}": round(float(np.mean(np.abs(y - pt) <= np.asarray(q[lv])[reg_t])), 4) for lv in LEVELS}
    ws = np.arange(0, 25.5, 0.5)
    grid = pd.DataFrame({"time_stamp": pd.Timestamp("2023-01-01 12:00"), C.WIND: ws, C.AMBIENT: 15.0, C.WIND_DIR: 180.0})
    gp, glo, ghi = pm.predict(grid, 0.8)
    out = {
        "metrics": {"ml": _metrics(y, pt), "physics": _metrics(y, pp),
                    "legacy_reported": {"r2": 0.9946, "rmse": 0.0182, "mae": 0.0115, "note": "from USER_MANUAL / legacy README"}},
        "intervals": {"nominal": list(LEVELS), "method": "split-conformal |residual| quantile per wind-speed regime",
                      "coverage_overall": overall, "coverage_by_regime": cov,
                      "q": {str(k): [round(v, 4) for v in vals] for k, vals in q.items()}},
        "power_curve": [{"wind": float(w), "p_med": round(float(a), 4), "p_lo": round(float(b), 4), "p_hi": round(float(c), 4)}
                        for w, a, b, c in zip(ws, gp, glo, ghi)],
        "audit": {
            "split": "fit = first 80 % of train rows of all 22 datasets; calibration = last 20 % of each dataset's train "
                     "period; test = prediction rows of the 10 normal datasets. Rows matched by value fingerprint "
                     "(asset + 8 signals) are removed from fit/calibration when they also occur in test/calibration.",
            "leakage_checks": [
                {"check": "legacy split (forecasting_model/power_model.py): test rows absent from its training set",
                 "passed": bool(overlap <= 0.01), "result": round(overlap, 4), "n_test": int(len(legacy_test)),
                 "detail": f"{overlap:.1%} of the legacy test rows ({len(legacy_test):,}) also occur in its training set: "
                           "overlapping datasets of the same turbine share physical records under different anonymised "
                           "timestamps, so the legacy de-duplication on (asset, time_stamp, wind, power) misses them. "
                           "The legacy R2 0.995 / RMSE 1.82 % is therefore partly in-sample."},
                {"check": "legacy test rows duplicated between normal datasets", "passed": bool(dup_test <= 0.01),
                 "result": round(dup_test, 4), "detail": f"{dup_test:.1%} of legacy test rows appear twice (inflates n)"},
                {"check": "new split: test fingerprints in fit or calibration", "passed": True,
                 "result": int(np.isin(test["fp"], train["fp"]).sum()), "detail": "removed by fingerprint matching"},
                {"check": "new split: calibration fingerprints in fit", "passed": True,
                 "result": int(np.isin(cal["fp"], fit["fp"]).sum()), "detail": "calibration rows removed from fit"},
                {"check": "status / event labels as model inputs", "passed": True, "detail": "not used (status 0/2 only filters training rows)"},
                {"check": "day-of-year feature", "passed": True, "detail": "dropped: timestamps are anonymised per file"},
            ]},
        "features": FEATS, "dataset": {"record": C.DATASET_RECORD, "doi": C.DATASET_DOI, "version": C.DATASET_VERSION},
        "pipeline_version": C.PIPELINE_VERSION, "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    C.FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (C.FORECAST_DIR / "forecast.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    joblib.dump(pm, MODEL_PATH, compress=3)
    log(f"forecast: ML {out['metrics']['ml']}  physics {out['metrics']['physics']}  coverage {overall}  ({time.time() - t0:.0f}s)")
    return out


if __name__ == "__main__":
    from windtwin import forecast as _forecast       # pickle PowerModel under windtwin.forecast, not __main__
    _forecast.run()
