"""CARE score against hand-computed toy cases."""
import numpy as np
import pandas as pd
import pytest

from windtwin import care


def times(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq="10min")


def test_fbeta_hand():
    # tp=2 fp=1 fn=1 -> (1.25*2)/(1.25*2 + 0.25*1 + 1) = 2.5/3.75
    assert care.fbeta([1, 1, 1, 0], [1, 1, 0, 1], 0.5) == pytest.approx(2.5 / 3.75)
    assert care.fbeta([0, 0], [0, 0]) == 0.0


def test_criticality_counter():
    c = care.criticality([1, 1, 0, 0, 0, 1], [True, True, True, False, True, True])
    assert c.tolist() == [1, 2, 1, 1, 0, 1]


def test_earliness_weights_quarter():
    w = care.earliness_weights(8, (1, 4))
    # flat for the first quarter (2 of 8 points), then linear decay to ~0
    assert w[0] == pytest.approx(1.0) and w[1] == pytest.approx(1.0)
    assert np.all(np.diff(w[2:]) < 0) and w[-1] < 0.2
    assert care.weighted_score(np.ones(8)) == pytest.approx(1.0)
    assert care.weighted_score(np.zeros(8)) == 0.0
    early = np.r_[np.ones(4), np.zeros(4)]
    assert care.weighted_score(early) > care.weighted_score(early[::-1])


def test_event_and_final_score():
    t = times(200)
    start, end = t[100], t[179]
    pred = np.zeros(200, bool)
    pred[100:180] = True                               # perfect detection of the event window
    a = care.evaluate_event(t, pred, start, end, "anomaly")
    assert a["f_beta_score"] == pytest.approx(1.0) and a["weighted_score"] == pytest.approx(1.0)
    assert a["max_criticality"] == 80 and a["detected"]
    n = care.evaluate_event(t, np.zeros(200, bool), t[50], t[150], "normal")
    assert n["accuracy"] == 1.0 and not n["detected"]
    assert care.final_score([a, n])["care"] == pytest.approx(1.0)
    # all-normal predictions -> nothing detected -> CARE 0
    a0 = care.evaluate_event(t, np.zeros(200, bool), start, end, "anomaly")
    assert care.final_score([a0, n])["care"] == 0.0
    # all-anomaly -> accuracy 0 <= 0.5 -> CARE = accuracy = 0
    a1 = care.evaluate_event(t, np.ones(200, bool), start, end, "anomaly")
    n1 = care.evaluate_event(t, np.ones(200, bool), t[50], t[150], "normal")
    assert care.final_score([a1, n1])["care"] == 0.0


def test_partial_case_hand():
    t = times(100)
    pred = np.zeros(100, bool)
    pred[40:60] = True                         # event 50..99 -> tp=10, fp=10 (40..49), fn=40
    a = care.evaluate_event(t, pred, t[50], t[99], "anomaly")
    assert (a["tp"], a["fp"], a["fn"]) == (10, 10, 40)
    assert a["f_beta_score"] == pytest.approx(12.5 / (12.5 + 10 + 10))
    assert a["max_criticality"] == 20 and not a["detected"]
    n = care.evaluate_event(t, np.r_[np.ones(80, bool), np.zeros(20, bool)], t[0], t[99], "normal")
    assert n["accuracy"] == pytest.approx(0.2) and n["detected"]
    fs = care.final_score([a, n])
    assert fs["accuracy"] == pytest.approx(0.2) and fs["care"] == pytest.approx(0.2)   # accuracy cap
    assert fs["reliability"] == 0.0           # labels [anomaly, normal], detected [False, True]


def test_weighted_average_hand():
    t = times(100)
    pa = np.zeros(100, bool)
    pa[50:100] = True                          # whole event, nothing else -> F=1, WS=1, crit 50 (<72)
    a = care.evaluate_event(t, pa, t[50], t[99], "anomaly", threshold=72)
    pn = np.zeros(100, bool)
    pn[:80] = True
    n = care.evaluate_event(t, pn, t[0], t[99], "normal")
    n_ok = care.evaluate_event(t, np.zeros(100, bool), t[0], t[99], "normal")
    # one detected (normal, false), acc mean = (0.2 + 1)/2 = 0.6, reliability F0.5 = 0 (tp 0)
    fs = care.final_score([a, n, n_ok])
    assert fs["accuracy"] == pytest.approx(0.6)
    assert fs["care"] == pytest.approx((1 + 1 + 0 + 2 * 0.6) / 5)


def test_operational():
    t = times(100)
    alarm = np.zeros(100, bool)
    alarm[10:15] = True
    alarm[60:70] = True
    op = care.operational(t, alarm, t[50], t[99], "anomaly")
    assert op["detected"] and op["pre_event_alarms"] == 1 and op["delay_h"] == pytest.approx(10 / 6)
    op = care.operational(t, alarm, t[50], t[99], "normal")
    assert op["false_alarms"] == 2
