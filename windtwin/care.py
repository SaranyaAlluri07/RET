"""CARE score (Gück et al. 2024, Data 9(12) 138) re-implemented to match the official
EnergyFaultDetector `CAREScore` defaults: coverage/reliability beta 0.5, weights (cov 1, acc 2, earl 1,
rel 1), criticality threshold 72 (>=), criticality cap 1000, criticality computed on predictions up to
event_end, earliness weights flat for the first 1/4 of the event then linear decay to 0
(ws_start_of_descend=(1, 4); the paper text says 1/2 - pass descend=(1, 2) for that), CARE = 0 when no
event is detected, = mean accuracy when that is <= 0.5. Wind Farm A: ignore_normal_index=True (every
prediction timestamp is evaluated; status labels are only used to filter training data).

Also: operational metrics (event recall, false-alarm events per turbine-month, detection delay,
component precision/recall with an 'unknown' bucket, confusion matrix)."""
import numpy as np
import pandas as pd

from . import config as C

CRIT_THRESHOLD = 72
MAX_CRIT = 1000


def fbeta(y_true, y_pred, beta=0.5):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / den if den > 0 else 0.0          # sklearn zero_division -> 0


def criticality(pred, normal=None, max_crit=MAX_CRIT):
    """CARE criticality counter: +1 on normal-status anomalous step, -1 (floor 0) on normal-status
    normal step, unchanged on abnormal-status steps."""
    pred = np.asarray(pred, bool)
    normal = np.ones(len(pred), bool) if normal is None else np.asarray(normal, bool)
    out = np.zeros(len(pred), int)
    c = 0
    for i in range(len(pred)):
        if normal[i]:
            c = min(max_crit, c + 1) if pred[i] else max(c - 1, 0)
        out[i] = c
    return out


def earliness_weights(n, descend=(1, 4)):
    """Weights of the official `_calculate_weighted_score` (flat, then linear decay to 0)."""
    if n == 0:
        return np.zeros(0)
    num, scale = descend
    s = num / scale
    L = n * scale
    cp = int(L * s)
    x = np.linspace(0, scale, L)
    w = np.empty(L)
    w[:cp] = scale
    w[cp:] = scale / (1 - s) - x[cp:] / (1 - s)
    return w[::scale] / scale


def weighted_score(pred_event, descend=(1, 4)):
    p = np.asarray(pred_event, float)
    w = earliness_weights(len(p), descend)
    return float(np.sum(p * w) / np.sum(w)) if np.sum(w) > 0 else 0.0


def evaluate_event(times, pred, event_start, event_end, label, normal=None, ignore_normal_index=True,
                   threshold=CRIT_THRESHOLD, descend=(1, 4)):
    """One dataset. times: timestamps of the prediction window; pred: bool anomaly predictions."""
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    pred = np.asarray(pred, bool)
    nidx = np.ones(len(pred), bool) if (normal is None or ignore_normal_index) else np.asarray(normal, bool)
    gt = ~nidx.copy()
    es, ee = np.datetime64(pd.Timestamp(event_start)), np.datetime64(pd.Timestamp(event_end))
    in_ev = (t >= es) & (t <= ee)
    if label == "anomaly":
        gt = gt | in_ev
    g, p = gt[nidx], pred[nidx]
    tp, fp = int(np.sum(g & p)), int(np.sum(~g & p))
    fn, tn = int(np.sum(g & ~p)), int(np.sum(~g & ~p))
    upto = t <= ee
    crit = criticality(pred[upto], nidx[upto])
    max_crit = int(crit.max()) if len(crit) else 0
    first = None
    if max_crit >= threshold:
        first = pd.Timestamp(t[upto][int(np.argmax(crit >= threshold))])
    out = {"event_label": label, "max_criticality": max_crit, "first_time_detected": first,
           "detected": max_crit >= threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "f_beta_score": None, "accuracy": None, "weighted_score": None}
    if label == "anomaly":
        out["f_beta_score"] = fbeta(g, p, 0.5)
        out["weighted_score"] = weighted_score(pred[in_ev], descend)
    else:
        out["accuracy"] = float((tn + tp) / max(1, len(g)))
    return out


def final_score(events):
    """events: list of evaluate_event dicts. Returns dict with CARE and its four sub-scores."""
    an = [e for e in events if e["event_label"] == "anomaly"]
    no = [e for e in events if e["event_label"] != "anomaly"]
    if not an or not no:
        raise ValueError("CARE needs at least one anomaly and one normal event")
    cov = float(np.mean([e["f_beta_score"] for e in an]))
    earl = float(np.mean([e["weighted_score"] for e in an]))
    acc = float(np.mean([e["accuracy"] for e in no]))
    rel = fbeta([e["event_label"] == "anomaly" for e in events], [e["detected"] for e in events], 0.5)
    if not any(e["detected"] for e in events):
        care = 0.0
    elif acc <= 0.5:
        care = acc
    else:
        care = (1 * cov + 1 * earl + 1 * rel + 2 * acc) / 5
    return {"care": care, "coverage": cov, "accuracy": acc, "reliability": rel, "earliness": earl,
            "events_detected_care": int(sum(e["detected"] for e in an)),
            "normal_detected_care": int(sum(e["detected"] for e in no))}


# ------------------------------------------------------------------ operational metrics
def alarm_episodes(alarm):
    a = np.asarray(alarm, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], a, [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def operational(times, alarm, event_start, event_end, label):
    """Per-dataset operational outcome from alarm episodes (alarm = method's own alarm decision).
    anomaly: detected if an alarm is active inside [event_start, event_end]; delay = first active alarm
    in the window - event_start; alarms that start before event_start are 'pre-event' alarms.
    normal: every alarm episode in the prediction window is a false-alarm event."""
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    alarm = np.asarray(alarm, bool)
    eps = alarm_episodes(alarm)
    es, ee = pd.Timestamp(event_start), pd.Timestamp(event_end)
    days = (len(t) * C.STEP_MINUTES) / 1440.0
    res = {"episodes": len(eps), "days": days, "detected": False, "first_alarm": None, "delay_h": None,
           "lead_to_end_h": None, "false_alarms": 0, "pre_event_alarms": 0, "alarm_index": None}
    if label == "anomaly":
        win = ((t >= es) & (t <= ee)).to_numpy()
        hit = np.flatnonzero(alarm & win)
        res["pre_event_alarms"] = int(sum(1 for s, _ in eps if t[s] < es))
        if len(hit):
            i = int(hit[0])
            res.update(detected=True, first_alarm=t[i], alarm_index=i,
                       delay_h=(t[i] - es).total_seconds() / 3600, lead_to_end_h=(ee - t[i]).total_seconds() / 3600)
    else:
        res["false_alarms"] = len(eps)
    return res


def summarize(rows):
    """rows: per-dataset dicts with label, care event dict, operational dict, diagnosis info.
    Returns the MethodResult metric fields."""
    care_events = [r["care_event"] for r in rows]
    fs = final_score(care_events)
    an = [r for r in rows if r["label"] == "anomaly"]
    no = [r for r in rows if r["label"] != "anomaly"]
    det = [r for r in an if r["op"]["detected"]]
    delays = np.array([r["op"]["delay_h"] for r in det], float)
    fa = int(sum(r["op"]["false_alarms"] for r in no))
    months = sum(r["op"]["days"] for r in no) / 30.44
    diag = [r for r in det if r.get("diagnosis_correct") is not None or r.get("diagnosis_unknown")]
    return {
        "care": round(fs["care"], 4), "coverage": round(fs["coverage"], 4), "accuracy": round(fs["accuracy"], 4),
        "reliability": round(fs["reliability"], 4), "earliness": round(fs["earliness"], 4),
        "events_detected": len(det), "events_total": len(an),
        "false_alarm_events": fa, "normal_datasets": len(no),
        "fa_per_turbine_month": round(fa / months, 3) if months > 0 else None,
        "normal_datasets_flagged": int(sum(1 for r in no if r["op"]["false_alarms"] > 0)),
        "pre_event_alarms": int(sum(r["op"]["pre_event_alarms"] for r in an)),
        "delay_median_h": round(float(np.median(delays)), 1) if len(delays) else None,
        "delay_p90_h": round(float(np.percentile(delays, 90)), 1) if len(delays) else None,
        "component_correct": int(sum(1 for r in det if r.get("diagnosis_correct") is True)),
        "component_unknown": int(sum(1 for r in det if r.get("diagnosis_unknown"))),
        "care_events_detected": fs["events_detected_care"], "care_normal_flagged": fs["normal_detected_care"],
        "normal_months": round(months, 2), "diagnosed": len(diag),
    }


def component_metrics(rows):
    """Per fault class precision / recall of the diagnosis over detected events, incl. 'Unknown'.
    Recall denominator = all events of that class (missed events count as misses)."""
    classes = sorted({C.fault_class(r["description"]) for r in rows if r["label"] == "anomaly"} - {None})
    out = {}
    for c in classes:
        truth = [r for r in rows if r["label"] == "anomaly" and C.fault_class(r["description"]) == c]
        said = [r for r in rows if r["label"] == "anomaly" and r["op"]["detected"] and r.get("diagnosis_class") == c]
        correct = [r for r in truth if r["op"]["detected"] and r.get("diagnosis_correct")]
        out[c] = {"events": len(truth), "correct": len(correct), "predicted": len(said),
                  "recall": round(len(correct) / len(truth), 3) if truth else None,
                  "precision": round(sum(1 for r in said if r.get("diagnosis_correct")) / len(said), 3) if said else None}
    return out


def confusion(rows):
    """Event-level confusion matrix (square): rows = logged fault class ('normal' for normal datasets),
    columns = diagnosis at the first alarm ('missed' = no alarm in the event window / no false alarm)."""
    def true_of(r):
        return (C.fault_class(r["description"]) or "other fault") if r["label"] == "anomaly" else "normal"

    def pred_of(r):
        alarmed = r["op"]["detected"] if r["label"] == "anomaly" else r["op"]["false_alarms"] > 0
        if not alarmed:
            return "normal" if r["label"] != "anomaly" else "missed"
        d = r.get("diagnosis")
        return "Unknown" if (d is None or d == C.UNKNOWN) else d

    pairs = [(true_of(r), pred_of(r)) for r in rows]
    comps = [c for c in C.COMPONENTS if any(c in p for p in pairs)]
    labels = comps + sorted({t for t, _ in pairs} - set(comps) - {"normal"}) + ["normal", "Unknown", "missed"]
    M = np.zeros((len(labels), len(labels)), int)
    for t, p in pairs:
        M[labels.index(t), labels.index(p)] += 1
    return {"labels": labels, "matrix": M.tolist(),
            "note": "rows = logged fault class, columns = diagnosis at first alarm; 'missed' = no alarm in the "
                    "event window; for normal datasets 'normal' = no false alarm, otherwise the false alarm's diagnosis"}
