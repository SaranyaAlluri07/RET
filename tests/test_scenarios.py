import numpy as np
import pandas as pd

from windtwin import scenarios, config as C


def frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"time_stamp": pd.date_range("2023-03-01", periods=n, freq="10min")})
    for c in C.SIGNALS:
        df[c] = rng.normal(40, 2, n)
    df[C.POWER] = rng.uniform(0, 1, n)
    df[C.WIND] = rng.uniform(3, 16, n)
    return df


def spec(sid, seed=7):
    return {"scenario": sid, "severity": "moderate", "ramp_h": 6, "duration_h": 24, "seed": seed,
            "start": pd.Timestamp("2023-03-01 10:00"), "end": pd.Timestamp("2023-03-02 10:00")}


def test_catalog_shape():
    cat = scenarios.catalog()
    assert len(cat) >= 17
    for s in cat:
        assert {"id", "name", "category", "component", "description", "params"} <= set(s)
        assert s["category"] in ("component", "sensor", "performance")
        assert s["component"] is None or s["component"] in C.COMPONENTS
        assert s["params"]["severity"] == ["mild", "moderate", "severe"]
        assert scenarios.get(s["id"])["name"] == s["name"]


def test_same_seed_same_output_and_batching_invariance():
    df = frame()
    for s in scenarios.catalog():
        a = scenarios.apply(df, spec(s["id"]))
        b = scenarios.apply(df, spec(s["id"]))
        pd.testing.assert_frame_equal(a, b)
        parts = pd.concat([scenarios.apply(df.iloc[i:i + 36], spec(s["id"])) for i in range(0, len(df), 36)])
        pd.testing.assert_frame_equal(a, parts)


def test_rows_outside_window_unchanged_and_truth():
    df = frame()
    sp = spec("gearbox_bearing")
    out, truth = scenarios.inject(df, sp)
    outside = (df["time_stamp"] < sp["start"]) | (df["time_stamp"] >= sp["end"])
    pd.testing.assert_frame_equal(out[outside], df[outside])
    assert (out.loc[~outside, "sensor_11_avg"] >= df.loc[~outside, "sensor_11_avg"]).all()
    assert truth["active"].sum() == (~outside).sum() - 1 or truth["active"].sum() == (~outside).sum()
    assert truth["component"].iloc[0] == "Gearbox"


def test_noise_depends_on_seed():
    df = frame()
    a = scenarios.apply(df, spec("sensor_noise", 1))
    b = scenarios.apply(df, spec("sensor_noise", 2))
    assert not np.allclose(a["sensor_12_avg"], b["sensor_12_avg"])


def test_unknown_id():
    try:
        scenarios.get("nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
