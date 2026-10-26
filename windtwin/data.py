"""Load CARE Wind Farm A datasets (Avg signals only), derived physics signals, features, operating state."""
import numpy as np
import pandas as pd

from . import config as C

META = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
H = C.STEPS_PER_HOUR


# ------------------------------------------------------------------ loading
def load_events():
    ev = pd.read_csv(C.EVENT_INFO, sep=";")
    for c in ("event_start", "event_end"):
        ev[c] = pd.to_datetime(ev[c])
    ev["event_description"] = ev["event_description"].fillna("").astype(str)
    ev = ev.rename(columns={"asset": "asset"})
    return ev.set_index("event_id", drop=False).sort_index()


def dataset_ids():
    return sorted(int(p.stem) for p in C.DATASETS_DIR.glob("*.csv") if p.stem.isdigit())


def load_dataset(ds_id, use_cache=True):
    """Raw dataset: meta columns + every *_avg column, sorted by time. Cached as parquet."""
    cache = C.CACHE_DIR / f"raw_{ds_id}.parquet"
    if use_cache and cache.is_file():
        return pd.read_parquet(cache)
    path = C.DATASETS_DIR / f"{ds_id}.csv"
    header = pd.read_csv(path, sep=";", nrows=0).columns
    keep = [c for c in header if c in META or c.endswith("_avg")]
    df = pd.read_csv(path, sep=";", usecols=keep)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df["train_test"] = df["train_test"].astype(str).str.lower()
    df = df.sort_values("time_stamp").reset_index(drop=True)
    for c in df.columns:
        if c.endswith("_avg"):
            df[c] = df[c].astype("float64")
    if use_cache:
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


# ------------------------------------------------------------------ derived signals + features
def unwrap_pitch(raw):
    """Pitch angle is stored with 0/360 wrapping ambiguities -> map to (-180, 180]."""
    return ((np.asarray(raw, float) + 180.0) % 360.0) - 180.0


def add_derived(df):
    """Derived physics signals (differences, phase spreads, imbalance, unwrapped pitch). In place."""
    def col(c):
        return df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)
    df["pitch"] = unwrap_pitch(col(C.PITCH_RAW))
    df["gen_bearing_diff"] = col("sensor_13_avg") - col("sensor_14_avg")
    df["gbx_bearing_minus_oil"] = col("sensor_11_avg") - col("sensor_12_avg")
    df["hydraulic_minus_nacelle"] = col("sensor_41_avg") - col(C.NACELLE)
    for name, cols in (("stator_spread", [15, 16, 17]), ("transformer_spread", [38, 39, 40]),
                       ("igbt_spread", [35, 36, 37])):
        m = np.column_stack([col(f"sensor_{i}_avg") for i in cols])
        df[name] = m.max(axis=1) - m.min(axis=1)
    for name, cols, min_mean in (("current_imbalance", [23, 24, 25], 100.0),
                                 ("voltage_imbalance", [32, 33, 34], 100.0)):
        m = np.column_stack([col(f"sensor_{i}_avg") for i in cols])
        mean = np.abs(m.mean(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            v = m.std(axis=1) / mean
        df[name] = np.where(mean > min_mean, v, np.nan)      # only meaningful under load
    return df


def add_features(df):
    """Operating-condition features (inputs of the normal-behaviour models). In place, causal."""
    p = df[C.POWER].clip(lower=-0.05)
    for h in (1, 3, 6, 12):
        df[f"power_mean_{h}h"] = p.rolling(h * H, min_periods=1).mean()
    df["power_sq"] = p.clip(lower=0) ** 2
    df["wind_cubed"] = df[C.WIND].clip(lower=0, upper=25) ** 3
    df["wind_std_1h"] = df[C.WIND].rolling(H, min_periods=2).std()
    df["wind_mean_1h"] = df[C.WIND].rolling(H, min_periods=1).mean()
    df["ambient_mean_6h"] = df[C.AMBIENT].rolling(6 * H, min_periods=1).mean()
    df["rotor_mean_1h"] = df[C.ROTOR_RPM].rolling(H, min_periods=1).mean()
    hour = df["time_stamp"].dt.hour + df["time_stamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    r = np.deg2rad(df[C.WIND_DIR] % 360)
    df["dir_sin"], df["dir_cos"] = np.sin(r), np.cos(r)
    return df


def prepare(df):
    """Copy + derived signals + features."""
    df = df.copy()
    add_derived(df)
    add_features(df)
    return df


# ------------------------------------------------------------------ operating state
class PowerCurve:
    """Binned median power curve (method of bins) used for operating-state classification."""

    def __init__(self, bin_width=0.5):
        self.bin_width = bin_width
        self.v_, self.p_ = np.array([0.0, 25.0]), np.array([0.0, 1.0])

    def fit(self, wind, power):
        ok = np.isfinite(wind) & np.isfinite(power)
        t = pd.DataFrame({"b": np.floor(wind[ok] / self.bin_width), "v": wind[ok], "p": power[ok]})
        g = t.groupby("b").agg(v=("v", "median"), p=("p", "median"), n=("p", "size"))
        g = g[g["n"] >= 20]
        if len(g) >= 3:
            self.v_, self.p_ = g["v"].to_numpy(), np.maximum.accumulate(g["p"].clip(lower=0).to_numpy())
        return self

    def predict(self, wind):
        return np.interp(np.asarray(wind, float), self.v_, self.p_, left=0.0, right=float(self.p_[-1]))


STATES = ("production", "curtailed", "idle", "stopped")


def operating_state(df, curve):
    """production / curtailed (power far below the curve with pitch out of fine pitch) / idle / stopped."""
    p = df[C.POWER].to_numpy(float)
    w = df[C.WIND].to_numpy(float)
    rpm = df[C.ROTOR_RPM].to_numpy(float) if C.ROTOR_RPM in df else np.full(len(df), np.nan)
    pitch = df["pitch"].to_numpy(float) if "pitch" in df else np.zeros(len(df))
    exp = curve.predict(np.nan_to_num(w))
    st = np.full(len(df), "idle", dtype=object)
    prod = p > C.P_PRODUCTION
    st[prod] = "production"
    curt = prod & (w > 5.0) & (p < 0.6 * exp) & (pitch > 3.0)
    st[curt] = "curtailed"
    stopped = (p <= C.P_STOPPED) & ((rpm < C.RPM_STOPPED) | ~np.isfinite(rpm))
    st[stopped] = "stopped"
    return st


# ------------------------------------------------------------------ split
def split(df):
    """Chronological protocol: normal train rows -> first 80 % fit, last 20 % validation.
    Returns dict of integer index arrays + the monitoring segment start."""
    is_train = (df["train_test"] == "train").to_numpy()
    normal = df["status_type_id"].isin(C.NORMAL_STATUS).to_numpy()
    tr = np.where(is_train & normal)[0]
    cut = int(len(tr) * (1 - C.VAL_FRACTION))
    pred = np.where(~is_train)[0]
    p0 = int(pred[0]) if len(pred) else len(df)
    return {"fit": tr[:cut], "val": tr[cut:], "val_start": int(tr[cut]) if cut < len(tr) else p0,
            "pred": pred, "pred_start": p0,
            "monitor_start": max(0, p0 - C.CONTEXT_STEPS),
            "segment_start": max(0, p0 - C.CONTEXT_STEPS - C.WARMUP_STEPS),
            "normal": normal, "is_train": is_train}
