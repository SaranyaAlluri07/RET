"""Per-turbine model bundle (NBMs + residual normalizer + alarm calibration) and the streaming Monitor
used by replay, the benchmark and the server simulation.

    b = build_bundle(24); save_bundle(b)              # train on dataset 24's train rows only
    m = Monitor.for_dataset(24)                        # or Monitor(bundle)
    m.prime(history_df)                                # optional warm-up rows (no alarms)
    rows, cases = m.update(new_rows_df)                # per-row outputs + opened/updated Case dicts
"""
import math

import numpy as np
import pandas as pd
import joblib

from . import config as C, data, quality, nbm, residuals, alarms, diagnosis

LAMBDAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 1.0)   # EWMA grid (1.0 = no smoothing / simple threshold)
KS = (0.5, 1.0, 2.0, 3.0)                          # CUSUM reference values
DEFAULT_PARAMS = alarms.AlarmParams()


class Bundle:
    """Everything needed to monitor one turbine, fitted on one dataset's train rows."""

    def __init__(self, ds_id, asset, targets, models, normalizer, curve, scales, meta):
        self.ds_id, self.asset, self.targets = ds_id, asset, targets
        self.models, self.normalizer, self.curve, self.scales, self.meta = models, normalizer, curve, scales, meta
        self.two_sided = residuals.two_sided_mask(targets)
        self.comp_idx = alarms.comp_index(targets)
        self.params = DEFAULT_PARAMS.to_dict()          # held-out (leave-one-turbine-out) alarm params
        self.params_global = DEFAULT_PARAMS.to_dict()   # params tuned on all turbines (simulation)
        self.version = C.PIPELINE_VERSION

    def cscale_for(self, k):
        cs = getattr(self, "cscales", {}) or {}
        if not cs:
            return np.ones(len(self.targets))
        return cs[min(cs, key=lambda x: abs(x - float(k)))]

    def scale_for(self, lam):
        lam = float(lam)
        if lam not in self.scales:
            k = min(self.scales, key=lambda x: abs(x - lam))
            return self.scales[k]
        return self.scales[lam]


def bundle_path(ds_id):
    return C.MODELS_DIR / f"ds_{ds_id}.joblib"


def save_bundle(b):
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, bundle_path(b.ds_id), compress=3)


def load_bundle(ds_id):
    return joblib.load(bundle_path(ds_id))


# ------------------------------------------------------------------ scoring
def score_frame(b, df, flags=None, q=None):
    """df: prepared frame (data.prepare). Returns dict of per-row arrays for the bundle's targets."""
    if flags is None:
        flags, q = quality.assess(df)
    state = data.operating_state(df, b.curve)
    n, T = len(df), len(b.targets)
    Y = np.full((n, T), np.nan)
    Yhat = np.full((n, T), np.nan)
    valid = np.zeros((n, T), bool)
    bad = q == "bad"
    need = {c for t in b.targets for c in b.models[t].inputs}
    F = {c: df[c].to_numpy(float) for c in need}
    for j, t in enumerate(b.targets):
        Y[:, j] = df[t].to_numpy(float) if t in df else np.nan
        Yhat[:, j] = b.models[t].predict(F)
        valid[:, j] = nbm.target_valid(t, state, flags[t].to_numpy()) & np.isfinite(Y[:, j]) & ~bad
    R = Y - Yhat
    power, amb = df[C.POWER].to_numpy(float), df[C.AMBIENT].to_numpy(float)
    Z = b.normalizer.transform(R, power, amb) if b.normalizer is not None else np.full_like(R, np.nan)
    Z[~valid] = np.nan
    return {"Y": Y, "Yhat": Yhat, "R": R, "Z": Z, "valid": valid, "state": state, "quality": q, "flags": flags,
            "bad": bad, "off": np.isin(state, ("stopped", "idle")), "power": power, "ambient": amb}


def build_bundle(ds_id, events=None):
    """Fit NBMs on the dataset's normal train rows (first 80 %), calibrate the normalizer and EWMA
    scales on the chronological validation split (last 20 %). Prediction rows are never touched."""
    events = data.load_events() if events is None else events
    raw = data.load_dataset(ds_id)
    df = data.prepare(raw)
    sp = data.split(df)
    flags, q = quality.assess(df)
    fit = sp["fit"]
    curve = data.PowerCurve().fit(df[C.WIND].to_numpy(float)[fit], df[C.POWER].to_numpy(float)[fit])
    state = data.operating_state(df, curve)
    models = nbm.fit_all(df, fit, sp["val"], flags, state)
    targets = [t for t in C.TARGETS if t in models]
    asset = int(events.loc[ds_id, "asset"]) if ds_id in events.index else int(raw["asset_id"].iloc[0])
    b = Bundle(ds_id, asset, targets, models, None, curve, {}, {})
    calibrate(b, df, sp, flags, q)
    b.meta = {"targets": {t: models[t].metrics | {"inputs": models[t].inputs} for t in targets},
              "n_fit": int(len(fit)), "n_val": int(len(sp["val"])), "quality": quality.summary(flags.iloc[:sp["pred_start"]]),
              "dropped_targets": [t for t in C.TARGETS if t not in models]}
    return b


def calibrate(b, df=None, sp=None, flags=None, q=None):
    """(Re)fit the regime normalizer and EWMA scales on the validation segment (models unchanged)."""
    if df is None:
        df = data.prepare(data.load_dataset(b.ds_id))
        sp = data.split(df)
        flags, q = quality.assess(df)
    v0, p0 = sp["val_start"], sp["pred_start"]
    seg = df.iloc[v0:p0]
    b.normalizer = None
    s = score_frame(b, seg, flags.iloc[v0:p0], q[v0:p0])
    normal_seg = sp["normal"][v0:p0]
    valid = s["valid"] & normal_seg[:, None]
    norm = residuals.RegimeNormalizer(b.targets).fit(s["R"][normal_seg], s["power"][normal_seg],
                                                     s["ambient"][normal_seg], valid[normal_seg])
    b.normalizer = norm
    Z = norm.transform(s["R"], s["power"], s["ambient"])
    Z[~valid] = np.nan
    b.scales = {lam: alarms.ewma_scale(Z, lam, b.two_sided) for lam in LAMBDAS}
    b.cscales = {k: alarms.cusum_scale(Z, k, b.two_sided) for k in KS}
    b.version = C.PIPELINE_VERSION
    return b


def segment(ds_id):
    """Raw monitoring segment: warm-up + 2-day context + prediction window; index of monitor start."""
    raw = data.load_dataset(ds_id)
    df = data.prepare(raw)
    sp = data.split(df)
    s0 = sp["segment_start"]
    return raw.iloc[s0:].reset_index(drop=True), sp["monitor_start"] - s0, sp["pred_start"] - s0


# ------------------------------------------------------------------ streaming monitor
class Monitor:
    HISTORY = C.WARMUP_STEPS
    ZHIST = 3 * 24 * C.STEPS_PER_HOUR        # max diagnosis window (3 days)

    def __init__(self, bundle, params=None, forecast_model=None, turbine=None, source="replay", use_forecast=True):
        self.b = bundle
        self.p = alarms.AlarmParams.from_dict(params or bundle.params)
        lam = self.p.lam if self.p.use_ewma else 1.0
        self.stream = alarms.AlarmStream(self.p, bundle.scale_for(lam), bundle.two_sided, bundle.comp_idx,
                                         bundle.cscale_for(self.p.k))
        self.turbine = bundle.asset if turbine is None else turbine
        self.source = source
        self.fm = forecast_model
        if self.fm is None and use_forecast:
            from . import forecast
            self.fm = forecast.load_model(missing_ok=True)
        self.buf = None
        self.t_hist, self.z_hist = [], []
        self.n_steps = 0
        self.cases, self.open_case, self._bad_run, self._ok_run, self.dq_case = [], None, 0, 0, None

    @classmethod
    def for_dataset(cls, ds_id, **kw):
        return cls(load_bundle(ds_id), **kw)

    # -------------------------------------------------------------- input handling
    def _append(self, df_new):
        df_new = df_new.copy()
        df_new["time_stamp"] = pd.to_datetime(df_new["time_stamp"])
        for c in quality.monitored_raw() + [C.WIND_DIR, C.REL_DIR, C.NACELLE_DIR, C.PITCH_RAW, C.GEN_RPM]:
            if c not in df_new:
                df_new[c] = np.nan
        buf = df_new if self.buf is None else pd.concat([self.buf, df_new], ignore_index=True)
        return buf

    def prime(self, df_hist):
        """Feed history rows (features / quality context) without running the alarm logic."""
        buf = self._append(df_hist)
        self.buf = buf.iloc[-self.HISTORY:].reset_index(drop=True)

    def update(self, df_new):
        """Process new rows. Returns (rows: list of per-row dicts, cases: list of Case dicts changed)."""
        if df_new is None or len(df_new) == 0:
            return [], []
        n_new = len(df_new)
        buf = self._append(df_new)
        full = data.prepare(buf)                       # features / quality need the history rows
        flags_all, q_all = quality.assess(full)
        lo = len(full) - n_new
        df = full.iloc[lo:].reset_index(drop=True)     # only the new rows are scored (row-wise models)
        flags = flags_all.iloc[lo:].reset_index(drop=True)
        s = score_frame(self.b, df, flags, q_all[lo:])
        if self.fm is not None:
            hist = full.iloc[max(0, lo - 6):]           # forecast features use a 1 h rolling window
            pe, pl, ph = (a[-n_new:] for a in self.fm.predict(hist))
        else:
            j = self.b.targets.index(C.POWER) if C.POWER in self.b.targets else None
            pe = s["Yhat"][:, j] if j is not None else np.full(n_new, np.nan)
            pl, ph = pe - 0.1, pe + 0.1
        # expected values and normal bands of every target (vectorised)
        T = len(self.b.targets)
        EXP, HALF = np.full((n_new, T), np.nan), np.full((n_new, T), np.nan)
        for j, t in enumerate(self.b.targets):
            cen, half = self.b.normalizer.band(t, s["power"], s["ambient"])
            EXP[:, j], HALF[:, j] = s["Yhat"][:, j] + cen, half
        s["EXP"], s["LO"], s["HI"] = EXP, EXP - HALF, EXP + HALF
        s["cols"] = {c: (df[c].to_numpy(float) if c in df else np.full(n_new, np.nan))
                     for c in (C.POWER, C.WIND, C.WIND_DIR, C.NACELLE_DIR, C.REL_DIR, C.ROTOR_RPM, "pitch", C.AMBIENT)}
        times = df["time_stamp"].tolist()
        rows, changed = [], {}
        for i in range(n_new):
            z = s["Z"][i]
            o = self.stream.step(z, s["bad"][i], s["off"][i])
            self.t_hist.append(times[i])
            self.z_hist.append(z)
            if len(self.z_hist) > self.ZHIST:
                self.t_hist.pop(0), self.z_hist.pop(0)
            row = self._row(times[i], s, i, o, flags, pe[i], pl[i], ph[i])
            rows.append(row)
            for c in self._cases(row, o, s, i):
                changed[c["id"]] = c
            self.n_steps += 1
        self.buf = buf.iloc[-self.HISTORY:].reset_index(drop=True)
        return rows, list(changed.values())

    # -------------------------------------------------------------- outputs
    def _row(self, t_i, s, i, o, flags, pe, pl, ph):
        p = self.p
        comps = []
        ev = alarms.evidence(s["Z"][i][None, :], 1.0, self.b.two_sided)[0]
        for c, name in enumerate(C.COMPONENTS):
            idx = self.b.comp_idx[c]
            if not len(idx):
                continue
            e, cs = o["Ec"][c], o["Sc"][c]
            zc = ev[idx]
            score = float(np.nanmax(zc)) if np.isfinite(zc).any() else None
            st = "normal"
            hi = (p.use_ewma and np.isfinite(e) and e > p.L) or (p.use_cusum and np.isfinite(cs) and cs > p.h)
            near = (p.use_ewma and np.isfinite(e) and e > p.watch * p.L) or (p.use_cusum and np.isfinite(cs) and cs > p.watch * p.h)
            st = "warning" if hi else ("watch" if near else "normal")
            comps.append({"name": name, "score": _r(score), "ewma": _r(e), "cusum": _r(cs), "state": st})
        top = None
        if o["level"] >= 1:
            ranked = [c for c in comps if c["state"] != "normal"]
            if ranked:
                top = max(ranked, key=lambda c: max((c["ewma"] or 0) / p.L if p.use_ewma else 0,
                                                     (c["cusum"] or 0) / p.h if p.use_cusum else 0))["name"]
        qs = s["quality"][i]
        Y, EXP, LO, HI = s["Y"][i].tolist(), s["EXP"][i].tolist(), s["LO"][i].tolist(), s["HI"][i].tolist()
        sig = {t: (_r(Y[j]), _r(EXP[j]), _r(LO[j]), _r(HI[j])) for j, t in enumerate(self.b.targets)}
        cols = s["cols"]
        g = lambda c: _r(cols[c][i])
        return {"t": t_i.isoformat(), "state": o["state"], "level": int(o["level"]),
                "alarm": bool(o["alarm"]), "health": round(o["health"], 1),
                "power": g(C.POWER), "expected": _r(pe), "expected_lo": _r(pl), "expected_hi": _r(ph),
                "wind": g(C.WIND), "wind_dir": g(C.WIND_DIR), "nacelle_dir": g(C.NACELLE_DIR), "yaw_error": g(C.REL_DIR),
                "rotor_rpm": g(C.ROTOR_RPM), "pitch": g("pitch"), "ambient": g(C.AMBIENT),
                "operating_state": s["state"][i], "top_component": top, "components": comps,
                "quality": {"state": qs, "issues": quality.issues(flags.iloc[i]) if qs != "ok" else []},
                "signals": sig, "E": _r(o["E"]), "S": _r(o["S"])}

    def _comp_stat(self, o):
        """Component statistic relative to its limit at this step (what made the alarm fire)."""
        p = self.p
        st = {}
        for c, name in enumerate(C.COMPONENTS):
            v = 0.0
            if p.use_ewma and np.isfinite(o["Ec"][c]):
                v = max(v, o["Ec"][c] / p.L)
            if p.use_cusum and np.isfinite(o["Sc"][c]):
                v = max(v, o["Sc"][c] / p.h)
            st[name] = float(v)
        return st

    def _diagnose(self, dev_step, row, o=None):
        n_back = self.n_steps - dev_step + 1 if dev_step >= 0 else 36
        n_back = int(min(max(n_back, 36), len(self.z_hist)))
        Zw = np.array(self.z_hist[-n_back:])
        obs = {t: v[0] for t, v in row["signals"].items()}
        exp = {t: v[1] for t, v in row["signals"].items()}
        obs = {k: (np.nan if v is None else v) for k, v in obs.items()}
        exp = {k: (np.nan if v is None else v) for k, v in exp.items()}
        cs = self._comp_stat(o) if o is not None else None
        return diagnosis.diagnose(Zw, self.b.targets, self.b.two_sided, obs, exp, comp_stat=cs), n_back

    def _cases(self, row, o, s, i):
        changed = []
        t = row["t"]
        lost = 0.0
        if row["expected"] is not None and row["power"] is not None and row["operating_state"] == "production":
            lost = max(0.0, row["expected"] - row["power"]) / C.STEPS_PER_HOUR
        c = self.open_case
        if c is None and o["level"] >= 2:
            d, n_back = self._diagnose(o["dev_start"], row, o)
            dev_t = self.t_hist[-n_back].isoformat() if o["dev_start"] >= 0 else t
            c = {"id": f"{self.source[:3]}-{self.turbine}-{len(self.cases) + 1}", "turbine": self.turbine,
                 "source": self.source, "status": "open", "state": "fault" if o["level"] >= 3 else "warning",
                 "component": d["component"], "hypothesis": d["hypothesis"], "deviation_start": dev_t,
                 "alarm_time": t, "cleared_time": None, "severity": round(1 - row["health"] / 100, 3),
                 "confidence": d["confidence"], "evidence": d["evidence"], "shares": d["shares"],
                 "top_component": d["top_component"], "lost_energy_pu_h": 0.0, "truth": None}
            self.cases.append(c)
            self.open_case = c
            changed.append(c)
        elif c is not None:
            upd = False
            if o["level"] >= 3 and c["state"] != "fault":
                d, _ = self._diagnose(o["dev_start"], row, o)
                c.update(state="fault", component=d["component"], hypothesis=d["hypothesis"],
                         evidence=d["evidence"], shares=d["shares"], top_component=d["top_component"])
                upd = True
            sev = round(1 - row["health"] / 100, 3)
            if sev > c["severity"] + 0.05:
                c["severity"], upd = sev, True
            if o["level"] < 2:
                c.update(status="closed", cleared_time=t)
                self.open_case, upd = None, True
            if upd:
                changed.append(c)
        if self.open_case is not None:
            self.open_case["lost_energy_pu_h"] = round(self.open_case["lost_energy_pu_h"] + lost, 4)
        # data-quality case: 'unknown' state persisting >= 6 h
        bad = row["quality"]["state"] == "bad"
        self._bad_run = self._bad_run + 1 if bad else 0
        self._ok_run = 0 if bad else self._ok_run + 1
        if self.dq_case is None and self._bad_run >= 36:
            dq = {"id": f"{self.source[:3]}-{self.turbine}-{len(self.cases) + 1}", "turbine": self.turbine,
                  "source": self.source, "status": "open", "state": "unknown", "component": None,
                  "hypothesis": "Data quality — monitoring not possible: " + "; ".join(row["quality"]["issues"][:3]),
                  "deviation_start": self.t_hist[-36].isoformat(), "alarm_time": t, "cleared_time": None,
                  "severity": 0.0, "confidence": None, "evidence": [], "lost_energy_pu_h": 0.0, "truth": None}
            self.cases.append(dq)
            self.dq_case = dq
            changed.append(dq)
        elif self.dq_case is not None and self._ok_run >= 36:
            self.dq_case.update(status="closed", cleared_time=t)
            changed.append(self.dq_case)
            self.dq_case = None
        return changed


def _r(v, nd=4):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None
