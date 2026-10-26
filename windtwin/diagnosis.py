"""Evidence-based diagnosis. Components are ranked by the alarm statistic that fired (normalised
EWMA / CUSUM relative to its limit, at alarm time) when available, otherwise by the mean standardized
residual evidence over the alarm window. Explicit 'Unknown — insufficient evidence' when the leader is
weak, the runner-up is close, or several unrelated components are above their limits together
(common-mode / weather / input problem).
Confidence is reported only when calibrated (it is not: 12 logged events are too few) -> None."""
import numpy as np

from . import config as C

MIN_MARGIN = 0.25       # leader must exceed the runner-up by >= 25 % (ratio 1.25)
MAX_SPREAD = 3          # >= 3 components above their limit at once -> common-mode, unknown
MIN_EVIDENCE = 0.5      # leader's statistic/limit (or window mean positive z) must reach this

HYPOTHESIS = {
    "Gearbox": "Gearbox — bearing/oil temperatures higher than expected for the load",
    "Generator bearings": "Generator bearing — bearing temperature(s) higher than expected",
    "Generator windings": "Generator windings — stator temperature/phase spread above normal",
    "Transformer": "Transformer — HV phase temperature(s)/spread above normal",
    "Hydraulic group": "Hydraulic group — oil temperature higher than expected",
    "Hub / pitch": "Hub / pitch system — pitch or hub temperature deviates from normal",
    "Converter": "Converter — IGBT / cooling temperatures above normal",
    "Nacelle cabinets": "Nacelle cabinets — cabinet temperatures above normal",
    "Electrical balance": "Electrical balance — phase current/voltage imbalance",
    "Power performance": "Power performance — output below the expected power curve",
}


def evidence_per_target(Z, two_sided):
    """Mean positive evidence per target over the window (two-sided: |z|)."""
    X = np.where(two_sided, np.abs(Z), np.maximum(Z, 0.0))
    with np.errstate(all="ignore"):
        m = np.nanmean(X, axis=0)
    return np.nan_to_num(m, nan=0.0)


def diagnose(Z_win, targets, two_sided, observed=None, expected=None, quality_bad_share=0.0, comp_stat=None):
    """Z_win: signed z over the alarm window (n x T). observed / expected: dict target -> value at alarm.
    Returns dict(component, hypothesis, shares, evidence, confidence, unknown)."""
    ev = evidence_per_target(Z_win, two_sided)
    comp_ev = {}
    for c in C.COMPONENTS:
        idx = [i for i, t in enumerate(targets) if C.TARGET_COMPONENT.get(t) == c]
        comp_ev[c] = float(max(ev[idx])) if idx else 0.0
    if comp_stat:                       # rank by the statistic that fired (relative to its limit)
        comp_ev = {c: float(comp_stat.get(c, 0.0)) for c in C.COMPONENTS}
    tot = sum(comp_ev.values())
    shares = {c: (v / tot if tot > 0 else 0.0) for c, v in comp_ev.items()}
    ranked = sorted(shares, key=shares.get, reverse=True)
    top, second = ranked[0], ranked[1]
    reasons = []
    if tot <= 0 or comp_ev[top] < MIN_EVIDENCE:
        reasons.append("weak evidence")
    if comp_ev[second] * (1 + MIN_MARGIN) > comp_ev[top]:
        reasons.append("runner-up too close")
    if comp_stat:
        above = sum(1 for c in ranked if comp_ev[c] >= 1.0)
    else:
        above = sum(1 for c in ranked if shares[c] >= 0.2)
    if above >= MAX_SPREAD:
        reasons.append("several unrelated components deviate (common mode)")
    if quality_bad_share > 0.3:
        reasons.append("data quality")
    unknown = bool(reasons)
    total_ev = float(ev.sum()) or 1.0
    order = np.argsort(-ev)
    evidence = []
    for i in order[:6]:
        if ev[i] <= 0:
            break
        t = targets[i]
        evidence.append({"signal": t, "label": C.label_of(t), "component": C.TARGET_COMPONENT.get(t),
                         "observed": _r(observed.get(t)) if observed else None,
                         "expected": _r(expected.get(t)) if expected else None,
                         "z": _r(float(np.nanmean(Z_win[:, i])) if np.isfinite(Z_win[:, i]).any() else None),
                         "contribution": round(float(ev[i] / total_ev), 3)})
    return {"component": C.UNKNOWN if unknown else top, "top_component": top,
            "hypothesis": (C.UNKNOWN + f" (leading: {top}; {', '.join(reasons)})") if unknown else HYPOTHESIS[top],
            "shares": {c: round(v, 3) for c, v in shares.items()}, "evidence": evidence,
            "confidence": None, "unknown": unknown, "reasons": reasons}


def _r(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 3)
