import numpy as np
import pandas as pd

from windtwin import quality, data, config as C


def test_signal_flags():
    v = np.r_[np.arange(10.0), [np.nan], [500.0], [20.0]]
    f = quality.signal_flags(v, "temp")
    assert f[10] & quality.MISSING
    assert f[11] & quality.RANGE
    assert f[12] & quality.JUMP and not f[3]   # 500 -> 20


def test_flatline_wind():
    v = np.r_[np.linspace(3, 6, 10), np.full(30, 7.3)]
    f = quality.signal_flags(v, "wind")
    assert not (f[:27] & quality.FLATLINE).any()
    assert (f[-5:] & quality.FLATLINE).all()


def test_run_length_causal():
    assert quality.run_length([1, 1, 2, 2, 2, np.nan, np.nan]).tolist() == [1, 2, 1, 2, 3, 1, 1]


def test_row_quality_bad_on_core_input():
    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(40, 1, n) for c in quality.monitored_raw()})
    for c in quality.monitored_raw():
        if C.kind_of(c) == "voltage":
            df[c] = rng.normal(400, 1, n)
        elif C.kind_of(c) == "current":
            df[c] = rng.normal(800, 5, n)
    df[C.WIND] = rng.uniform(3, 10, n)
    df[C.POWER] = rng.uniform(0, 1, n)
    df[C.ROTOR_RPM] = rng.uniform(8, 14, n)
    df.loc[20, C.WIND] = np.nan
    df.loc[30, "sensor_11_avg"] = 400
    fl, q = quality.assess(df)
    assert q[20] == "bad" and q[30] == "degraded" and q[0] == "ok"
    assert fl.loc[30, "gbx_bearing_minus_oil"] & quality.RANGE


def test_pitch_unwrap():
    assert np.allclose(data.unwrap_pitch([359.0, 0.5, 90.0, 358.0]), [-1.0, 0.5, 90.0, -2.0])
