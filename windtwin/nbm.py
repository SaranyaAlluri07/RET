"""Normal-behaviour models: one regression per monitored target, inputs = operating conditions only.

Leakage control: a target never sees its own sensors, sibling sensors of the same component, or
inputs from its correlation group (e.g. nacelle air temperature for nacelle-cabinet targets);
an automatic guard also drops any non-driver input with |corr| > 0.98 to the target on fit rows.
Robust fit: fit -> trim |r - median| > 4 robust sigma -> refit. Ridge and HistGradientBoosting are
both fitted; HGB is kept only when it lowers the validation robust sigma by more than 10 %
(prior from the literature: linear NBMs are competitive and more robust in transients)."""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

THERMAL_INPUTS = [C.WIND, C.POWER, "power_mean_1h", "power_mean_3h", "power_mean_6h", "power_mean_12h",
                  "power_sq", C.AMBIENT, "ambient_mean_6h", C.NACELLE, C.ROTOR_RPM, "rotor_mean_1h", C.GEN_RPM,
                  "wind_std_1h", "hour_sin", "hour_cos"]
WIND_INPUTS = [C.WIND, "wind_mean_1h", "wind_std_1h", "wind_cubed", C.AMBIENT, "dir_sin", "dir_cos",
               "hour_sin", "hour_cos"]
ELECTRIC_INPUTS = [C.POWER, "power_mean_1h", "power_sq", C.AMBIENT, "hour_sin", "hour_cos"]

# targets whose correlation group contains an otherwise allowed input
GROUP_EXCLUDE = {
    "hydraulic_minus_nacelle": {C.NACELLE},
    **{t: {C.NACELLE} for t in C.SUBSYSTEMS["Nacelle cabinets"]},
}
DRIVERS = {C.WIND, C.POWER}
PRODUCTION_ONLY = {C.POWER, "pitch", "current_imbalance", "voltage_imbalance"}
HGB_GAIN = 0.10
MAX_HGB_ROWS = 10000
MIN_FIT, MIN_VAL = 2000, 300


def candidate_inputs(target):
    if target == C.POWER or target == "pitch":
        cand = WIND_INPUTS
    elif target in ("current_imbalance", "voltage_imbalance"):
        cand = ELECTRIC_INPUTS
    else:
        cand = THERMAL_INPUTS
    banned = set(C.raw_sources(target)) | {target} | GROUP_EXCLUDE.get(target, set())
    comp = C.TARGET_COMPONENT.get(target)
    if comp:                                   # siblings of the same component
        for sib in C.SUBSYSTEMS[comp]:
            banned |= set(C.raw_sources(sib)) | {sib}
    return [c for c in cand if c not in banned]


def robust_sigma(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _hgb():
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.12, max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=15, random_state=0)


class TargetModel:
    """Model of one target. predict(X frame) -> expected value."""

    def __init__(self, target, inputs):
        self.target, self.inputs = target, list(inputs)
        self.kind, self.model, self.fill = None, None, None
        self.metrics = {}

    def _X(self, df, impute):
        """df: DataFrame or dict of column -> numpy array (fast path used by the monitor)."""
        if isinstance(df, dict):
            X = np.column_stack([df[c] for c in self.inputs])
        else:
            X = df[self.inputs].to_numpy(float)
        if impute:
            X = np.where(np.isfinite(X), X, self.fill)
        return X

    def _robust_fit(self, make, X, y):
        m = make().fit(X, y)
        r = y - m.predict(X)
        med, s = np.median(r), robust_sigma(r)
        keep = np.abs(r - med) <= 4 * s + 1e-9
        if 0.8 < keep.mean() < 1.0:
            m = make().fit(X[keep], y[keep])
        return m

    def fit(self, df, fit_rows, val_rows, rng):
        y_fit = df[self.target].to_numpy(float)[fit_rows]
        self.fill = np.nanmedian(df[self.inputs].to_numpy(float)[fit_rows], axis=0)
        self.fill = np.where(np.isfinite(self.fill), self.fill, 0.0)
        Xf = self._X(df.iloc[fit_rows], True)
        Xv = self._X(df.iloc[val_rows], True)
        y_val = df[self.target].to_numpy(float)[val_rows]
        ridge = self._robust_fit(_ridge, Xf, y_fit)
        sub = np.arange(len(fit_rows))
        if len(sub) > MAX_HGB_ROWS:
            sub = np.sort(rng.choice(sub, MAX_HGB_ROWS, replace=False))
        hgb = self._robust_fit(_hgb, Xf[sub], y_fit[sub])
        s_r = robust_sigma(y_val - ridge.predict(Xv))
        s_h = robust_sigma(y_val - hgb.predict(Xv))
        use_hgb = s_h < (1 - HGB_GAIN) * s_r
        self.kind, self.model = ("hgb", hgb) if use_hgb else ("ridge", ridge)
        pred_v = self.model.predict(Xv)
        ss = np.sum((y_val - y_val.mean()) ** 2)
        self.metrics = {"val_sigma_ridge": round(s_r, 4), "val_sigma_hgb": round(s_h, 4), "chosen": self.kind,
                        "val_r2": round(float(1 - np.sum((y_val - pred_v) ** 2) / ss), 4) if ss > 0 else None,
                        "n_fit": int(len(fit_rows)), "n_val": int(len(val_rows))}
        return self

    def predict(self, df):
        return self.model.predict(self._X(df, True))


def correlation_guard(df, target, inputs, rows):
    """Drop non-driver inputs that are near-duplicates of the target (|corr| > 0.98)."""
    y = df[target].to_numpy(float)[rows]
    out = []
    for c in inputs:
        x = df[c].to_numpy(float)[rows]
        ok = np.isfinite(x) & np.isfinite(y)
        if c not in DRIVERS and ok.sum() > 100 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            if abs(np.corrcoef(x[ok], y[ok])[0, 1]) > 0.98:
                continue
        out.append(c)
    return out


def fit_all(df, fit_idx, val_idx, flags, state, seed=0, log=None):
    """Fit every target. Rows: given fit/val indices, target finite, target sources unflagged,
    production state for production-only targets. Returns {target: TargetModel}."""
    rng = np.random.default_rng(seed)
    models = {}
    prod = state == "production"
    for t in C.TARGETS:
        if t not in df:
            continue
        ok = np.isfinite(df[t].to_numpy(float)) & (flags[t].to_numpy() == 0)
        if t in PRODUCTION_ONLY:
            ok &= prod
        else:
            ok &= state != "stopped"
        fr, vr = fit_idx[ok[fit_idx]], val_idx[ok[val_idx]]
        if len(fr) < MIN_FIT or len(vr) < MIN_VAL or np.nanstd(df[t].to_numpy(float)[fr]) < 1e-6:
            continue
        inputs = correlation_guard(df, t, candidate_inputs(t), fr)
        t0 = time.time()
        models[t] = TargetModel(t, inputs).fit(df, fr, vr, rng)
        if log:
            log(f"  {t}: {models[t].kind} {time.time() - t0:.1f}s")
    return models


def target_valid(t, state, flags_t):
    """Rows where the residual of target t is judged (others -> NaN, statistics freeze)."""
    ok = flags_t == 0
    if t in PRODUCTION_ONLY:
        return ok & (state == "production")
    return ok & (state != "stopped")
