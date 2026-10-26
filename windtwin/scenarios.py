"""Reproducible synthetic scenario library (simulation / stress tests only - never real-data evidence).

Each scenario is a pure function of (row values, time since start, severity, seed): the effect level at a
timestamp depends only on (time_stamp - start), so splitting a run into batches never changes the result,
and the same seed + timestamps always give the same output. Rows outside [start, end) are unchanged
(plus an optional linear recovery tail of `recovery_h` hours after `end`).

    catalog()                       -> GET /api/sim/scenarios list
    get(id)                         -> scenario metadata (KeyError if unknown)
    apply(df, spec)                 -> modified copy of df
    truth(df, spec)                 -> ground-truth labels per row (active, level, component, category)
    inject(df, spec)                -> (modified df, truth frame)   convenience for offline tests
"""
import numpy as np
import pandas as pd

from . import config as C

SEVERITY = {"mild": 0.4, "moderate": 1.0, "severe": 2.0}
PARAMS = {"severity": list(SEVERITY), "ramp_h": [0, 6, 24, 72, 168], "duration_h": [12, 48, 168, 336]}

# id -> (name, category, component, description, default signal)
_SCEN = {
    "gearbox_bearing": ("Gearbox HSS bearing overheating", "component", "Gearbox",
                        "Gearbox high-speed-shaft bearing runs hotter under load (+5/+12/+24 °C at full load).", "sensor_11_avg"),
    "gearbox_oil": ("Gearbox oil overheating", "component", "Gearbox",
                    "Gearbox oil temperature rises (cooler fouling); bearing follows at 60 %.", "sensor_12_avg"),
    "gen_bearing_de": ("Generator bearing DE overheating", "component", "Generator bearings",
                       "Drive-end generator bearing heats up relative to the non-drive end.", "sensor_13_avg"),
    "gen_bearing_nde": ("Generator bearing NDE overheating", "component", "Generator bearings",
                        "Non-drive-end generator bearing heats up relative to the drive end.", "sensor_14_avg"),
    "transformer_phase": ("Transformer phase hot spot", "component", "Transformer",
                          "One HV transformer phase (L1) runs hotter -> phase spread grows.", "sensor_38_avg"),
    "transformer_cooling": ("Transformer cooling loss", "component", "Transformer",
                            "All three HV transformer phases heat up with load (cooling degraded).", "sensor_39_avg"),
    "hydraulic_overheating": ("Hydraulic oil overheating", "component", "Hydraulic group",
                              "Hydraulic group oil temperature rises above its normal level.", "sensor_41_avg"),
    "hydraulic_pitch_lag": ("Hydraulic pressure loss / pitch lag", "component", "Hub / pitch",
                            "Low hydraulic pressure: pitch lags the controller above rated wind, power fluctuates, "
                            "hydraulic oil warms slightly.", "sensor_5_avg"),
    "converter_cooling": ("Converter cooling fault", "component", "Converter",
                          "Converter IGBTs and VCS cooling water run hotter under load.", "sensor_21_avg"),
    "pitch_misalignment": ("Pitch misalignment", "performance", "Power performance",
                           "Blade pitch offset (+1/+2.5/+5 °) reduces power in partial load.", "sensor_5_avg"),
    "yaw_misalignment": ("Yaw misalignment", "performance", "Power performance",
                         "Static yaw error (+5/+12/+24 °) reduces power ~cos^3.", "sensor_2_avg"),
    "blade_icing_soiling": ("Blade icing / soiling", "performance", "Power performance",
                            "Aerodynamic degradation reduces power by 3/8/16 %.", "power_30_avg"),
    "sensor_bias": ("Sensor bias", "sensor", None,
                    "Constant offset on one sensor (default gearbox oil temp: +3/+6/+12 °C).", "sensor_12_avg"),
    "sensor_drift": ("Sensor drift", "sensor", None,
                     "Offset growing linearly over the injection (default gearbox oil temp, up to +4/+10/+20 °C).",
                     "sensor_12_avg"),
    "sensor_stuck": ("Stuck sensor", "sensor", None,
                     "Sensor repeats the value it had at onset (default gearbox oil temp).", "sensor_12_avg"),
    "sensor_noise": ("Sensor noise increase", "sensor", None,
                     "Extra Gaussian noise on one sensor (default gearbox oil temp: 1/2.5/5 °C).", "sensor_12_avg"),
    "missing_data": ("Missing data", "sensor", None,
                     "Telemetry drop-out: sensor values missing (default: all temperatures, wind and power).", None),
}


def catalog():
    return [{"id": k, "name": v[0], "category": v[1], "component": v[2], "description": v[3],
             "signal": v[4], "params": PARAMS} for k, v in _SCEN.items()]


def get(sid):
    if sid not in _SCEN:
        raise KeyError(f"unknown scenario '{sid}'")
    v = _SCEN[sid]
    return {"id": sid, "name": v[0], "category": v[1], "component": v[2], "description": v[3], "signal": v[4],
            "params": PARAMS}


# ------------------------------------------------------------------ profile + deterministic noise
def level(times, spec):
    """Effect level 0..1 at each timestamp: linear ramp over ramp_h from start, full until end,
    optional linear recovery over recovery_h after end."""
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    start = np.datetime64(pd.Timestamp(spec["start"]))
    end = spec.get("end")
    end = np.datetime64(pd.Timestamp(end)) if end is not None else start + np.timedelta64(int(float(spec.get("duration_h", 48)) * 3600), "s")
    h = (t - start) / np.timedelta64(1, "h")
    ramp = float(spec.get("ramp_h", 0) or 0)
    lv = np.where(h < 0, 0.0, np.minimum(1.0, h / ramp) if ramp > 0 else 1.0)
    lv = np.where(h < 0, 0.0, lv)
    after = (t >= end)
    rec = float(spec.get("recovery_h", 0) or 0)
    if rec > 0:
        h_end = (t - end) / np.timedelta64(1, "h")
        h_dur = (end - start) / np.timedelta64(1, "h")
        lv_end = min(1.0, h_dur / ramp) if ramp > 0 else 1.0
        lv = np.where(after, np.clip(lv_end * (1 - h_end / rec), 0.0, 1.0), lv)
    else:
        lv = np.where(after, 0.0, lv)
    return np.asarray(lv, float)


def _noise(times, seed, salt=0):
    """Standard normal per timestamp, keyed by (seed, salt, 10-min step) -> batching-invariant."""
    ts = pd.to_datetime(pd.Series(times)).astype("int64").to_numpy() // (600 * 10 ** 9)
    return np.array([np.random.default_rng([int(seed) & 0xFFFFFFFF, salt, int(k) & 0xFFFFFFFFFFFF]).standard_normal()
                     for k in ts])


def _load(df):
    p = df[C.POWER].to_numpy(float) if C.POWER in df else np.zeros(len(df))
    return 0.3 + 0.7 * np.clip(np.nan_to_num(p), 0, 1)


def _add(df, col, delta):
    if col in df:
        df[col] = df[col].to_numpy(float) + delta


# ------------------------------------------------------------------ apply
def apply(df, spec):
    """Return a modified copy of df (raw rows with time_stamp)."""
    sid = spec["scenario"]
    meta = get(sid)
    sev = SEVERITY[str(spec.get("severity", "moderate")).lower()]
    seed = int(spec.get("seed", 0) or 0)
    out = df.copy()
    lv = level(out["time_stamp"], spec)
    if not np.any(lv > 0) and sid != "sensor_stuck":
        return out
    act = lv > 0
    load = _load(out)
    sig = spec.get("signal") or meta["signal"]
    if sid == "gearbox_bearing":
        _add(out, "sensor_11_avg", lv * 12 * sev * load)
    elif sid == "gearbox_oil":
        d = lv * 8 * sev * load
        _add(out, "sensor_12_avg", d)
        _add(out, "sensor_11_avg", 0.6 * d)
    elif sid == "gen_bearing_de":
        _add(out, "sensor_13_avg", lv * 10 * sev * load)
    elif sid == "gen_bearing_nde":
        _add(out, "sensor_14_avg", lv * 10 * sev * load)
    elif sid == "transformer_phase":
        _add(out, "sensor_38_avg", lv * 10 * sev * load)
    elif sid == "transformer_cooling":
        for c in ("sensor_38_avg", "sensor_39_avg", "sensor_40_avg"):
            _add(out, c, lv * 8 * sev * load)
    elif sid == "hydraulic_overheating":
        _add(out, "sensor_41_avg", lv * 8 * sev)
    elif sid == "hydraulic_pitch_lag":
        w = out[C.WIND].to_numpy(float) if C.WIND in out else np.zeros(len(out))
        above = np.clip((np.nan_to_num(w) - 11.0) / 4.0, 0, 1)
        n = _noise(out["time_stamp"], seed, 1)
        _add(out, C.PITCH_RAW, -lv * above * 2.0 * sev + lv * above * 0.8 * sev * n)
        if C.POWER in out:
            out[C.POWER] = out[C.POWER].to_numpy(float) * (1 - lv * above * 0.03 * sev * np.abs(n))
        _add(out, "sensor_41_avg", lv * 2.5 * sev)
    elif sid == "converter_cooling":
        for c in ("sensor_21_avg", "sensor_35_avg", "sensor_36_avg", "sensor_37_avg"):
            _add(out, c, lv * 9 * sev * load)
        _add(out, "sensor_10_avg", lv * 4 * sev * load)
    elif sid == "pitch_misalignment":
        off = 2.5 * sev
        _add(out, C.PITCH_RAW, lv * off)
        if C.POWER in out:
            p = out[C.POWER].to_numpy(float)
            partial = np.clip(1.0 - np.nan_to_num(p) / 0.95, 0, 1)
            out[C.POWER] = p * (1 - lv * partial * 0.025 * off)
    elif sid == "yaw_misalignment":
        ang = 12.0 * sev
        _add(out, C.REL_DIR, lv * ang)
        if C.POWER in out:
            out[C.POWER] = out[C.POWER].to_numpy(float) * (1 - lv * (1 - np.cos(np.deg2rad(ang)) ** 3))
    elif sid == "blade_icing_soiling":
        if C.POWER in out:
            out[C.POWER] = out[C.POWER].to_numpy(float) * (1 - lv * 0.08 * sev)
    elif sid == "sensor_bias":
        _add(out, sig, np.where(act, 6.0 * sev, 0.0))
    elif sid == "sensor_drift":
        dur = max(float(spec.get("duration_h", 48) or 48), 1.0)
        h = (pd.to_datetime(out["time_stamp"]) - pd.Timestamp(spec["start"])).dt.total_seconds().to_numpy() / 3600
        _add(out, sig, np.where(act, 10.0 * sev * np.clip(h / dur, 0, 1), 0.0))
    elif sid == "sensor_stuck":
        stuck = spec.get("stuck_value")
        if stuck is None:
            # value at onset: deterministic from the seed when the onset row is not in this batch
            stuck = float(spec.get("onset_value", 45.0 + 10.0 * _noise([spec["start"]], seed, 7)[0] / 3))
        if sig in out:
            out[sig] = np.where(act, round(float(stuck)), out[sig].to_numpy(float))
    elif sid == "sensor_noise":
        _add(out, sig, np.where(act, 2.5 * sev * _noise(out["time_stamp"], seed, 3), 0.0))
    elif sid == "missing_data":
        cols = [sig] if sig else [c for c in out.columns if c.endswith("_avg") and
                                  C.kind_of(c) in ("temp", "wind", "power")]
        for c in cols:
            if c in out:
                out[c] = np.where(act, np.nan, out[c].to_numpy(float))
    return out


def truth(df, spec):
    meta = get(spec["scenario"])
    lv = level(df["time_stamp"], spec)
    return pd.DataFrame({"time_stamp": pd.to_datetime(df["time_stamp"]).to_numpy(), "active": lv > 0, "level": lv,
                         "scenario": spec["scenario"], "category": meta["category"], "component": meta["component"]})


def inject(df, spec):
    return apply(df, spec), truth(df, spec)
