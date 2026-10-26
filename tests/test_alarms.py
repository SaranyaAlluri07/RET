"""Batch and streaming alarm logic must give identical results."""
import numpy as np

from windtwin import alarms, config as C


def _data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    T = len(C.TARGETS)
    Z = rng.standard_normal((n, T))
    Z[600:900, 0] += 4.0                    # sustained gearbox deviation
    Z[1200:1210, 5] += 8.0                  # short spike
    Z[rng.random((n, T)) < 0.05] = np.nan   # not judged
    Z[300:340] = np.nan                     # stopped
    bad = np.zeros(n, bool)
    bad[1000:1030] = True
    Z[bad] = np.nan
    off = np.zeros(n, bool)
    off[300:340] = True
    two = np.array([C.DIRECTION.get(t, 1) == 0 for t in C.TARGETS])
    return Z, bad, off, two, alarms.comp_index(C.TARGETS), rng.uniform(0.2, 1.0, T)


def _check(p):
    Z, bad, off, two, ci, scale = _data()
    cscale = np.linspace(1.0, 20.0, len(scale))
    b = alarms.run_batch(Z, bad, off, p, scale, two, ci, cscale)
    s = alarms.AlarmStream(p, scale, two, ci, cscale)
    out = [s.step(Z[i], bad[i], off[i]) for i in range(len(Z))]
    assert [o["level"] for o in out] == b["level"].tolist()
    assert [o["state"] for o in out] == b["state"].tolist()
    assert [o["dev_start"] for o in out] == b["dev_start"].tolist()
    np.testing.assert_allclose([o["E"] for o in out], b["E"], rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose([o["S"] for o in out], b["S"], rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.array([o["Ec"] for o in out]), b["Ec"], rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose([o["health"] for o in out], b["health"], rtol=1e-12)
    return b


def test_batch_equals_streaming_selected():
    b = _check(alarms.AlarmParams(lam=0.05, L=4, k=1.0, h=40, persist=6, cooldown=36, clear=0.5))
    assert b["alarm"][600:900].any() and not b["alarm"][:300].any()
    assert set(b["state"][300:340]) <= {"offline", "unknown"}


def test_batch_equals_streaming_variants():
    _check(alarms.AlarmParams(lam=1.0, L=3, use_cusum=False, persist=1, cooldown=1, clear=1.0))
    _check(alarms.AlarmParams(use_ewma=False, k=0.5, h=20, persist=1, cooldown=1, clear=1.0))
    _check(alarms.AlarmParams(lam=0.1, L=5, use_cusum=False, persist=1, cooldown=1, clear=1.0))
    _check(alarms.AlarmParams(lam=0.05, L=2, k=1.0, h=2, persist=6, cooldown=36, clear=0.5, cusum_cap=1.5))
    _check(alarms.AlarmParams(lam=0.05, L=2, k=1.0, h=2, persist=6, cooldown=36, clear=0.5, combine="and"))


def _one(Z, p):
    n = len(Z)
    return alarms.run_batch(Z, np.zeros(n, bool), np.zeros(n, bool), p, np.ones(1), np.array([False]), [np.array([0])])


def test_threshold_variant_is_pointwise():
    Z = np.zeros((10, 1))
    Z[3, 0] = 5
    b = _one(Z, alarms.AlarmParams(lam=1.0, L=3, use_cusum=False, persist=1, cooldown=1, clear=1.0))
    assert b["alarm"].tolist() == [False, False, False, True] + [False] * 6


def test_persistence_hysteresis_cooldown():
    Z = np.zeros((100, 1))
    Z[10:30, 0] = 10
    b = _one(Z, alarms.AlarmParams(lam=1.0, L=3, use_cusum=False, persist=5, cooldown=10, clear=0.5))
    a = np.flatnonzero(b["alarm"])
    assert a[0] == 14                  # 5 consecutive exceedances needed
    assert a[-1] == 38                 # 10 calm steps after the deviation before clearing
    assert b["state"][12] == "watch"


def test_unknown_freezes():
    Z = np.zeros((50, 1))
    Z[5:40, 0] = 10
    bad = np.zeros(50, bool)
    bad[8:30] = True
    Z[bad] = np.nan
    p = alarms.AlarmParams(lam=1.0, L=3, use_cusum=False, persist=5, cooldown=3, clear=0.5)
    b = alarms.run_batch(Z, bad, np.zeros(50, bool), p, np.ones(1), np.array([False]), [np.array([0])])
    assert set(b["state"][8:30]) == {"unknown"}
    assert not b["alarm"][:30].any()   # 3 judged hits before the gap are not enough
    assert b["alarm"][31]              # persistence resumes after the gap (3 + 2 = 5)
