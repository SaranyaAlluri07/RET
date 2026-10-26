"""
Which change really improves fault detection?  (fast - no retraining)
=====================================================================
Uses the per-sensor deviations (z_ columns) that fault_detection.py (v1) and
fault_detection_v2.py already saved in their scores/ folders, and tests each idea
separately with the SAME honest leave-one-dataset-out cross-validation:

  A  v1 exactly as reported
  B  v2 models (robust training), v1 signals, 6 h smoothing
  C  B + physics signals (bearing-oil, DE-NDE, phase spreads, hydraulic-nacelle)
  D  best of A-C with 12 h smoothing
  E  best of A-D + component selection: components that cause false alarms on the
     healthy turbines are switched off - chosen INSIDE each cross-validation fold,
     so the held-out turbine never influences the choice (no cheating).

Run (after both fault_detection.py and fault_detection_v2.py have run):
    python improve_detection.py
Output: outputs_improve/variants.csv, variants.png, best_setting.json  (~2-5 min)
"""
import os
# hide harmless sklearn/joblib version warnings, also in the parallel worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import glob
import json
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS_PER_HOUR = 6
THRESHOLDS = np.round(np.arange(1.0, 12.01, 0.25), 2)
CRIT_LIMITS = np.array([18, 36, 72])          # 3 h, 6 h, 12 h

V1_SUBSYSTEMS = {
    "Generator bearings": ["sensor_13_avg", "sensor_14_avg"],
    "Gearbox": ["sensor_11_avg", "sensor_12_avg"],
    "Generator windings": ["sensor_15_avg", "sensor_16_avg", "sensor_17_avg"],
    "Transformer": ["sensor_38_avg", "sensor_39_avg", "sensor_40_avg"],
    "Hydraulic group": ["sensor_41_avg"],
    "Hub / pitch": ["sensor_6_avg", "sensor_53_avg", "pitch_dev"],
    "Converter": ["sensor_21_avg", "sensor_35_avg", "sensor_36_avg", "sensor_37_avg"],
    "Nacelle cabinets": ["sensor_7_avg", "sensor_8_avg", "sensor_9_avg", "sensor_10_avg", "sensor_19_avg", "sensor_20_avg"],
    "Electrical balance": ["current_imbalance", "voltage_imbalance"],
    "Power performance": ["power_30_avg"],
    "Whole turbine (AE)": ["ae_error"],
}
PHYSICS = {"Generator bearings": ["gen_bearing_diff"], "Gearbox": ["gbx_bearing_minus_oil"],
           "Generator windings": ["stator_spread"], "Transformer": ["transformer_spread"],
           "Hydraulic group": ["hydraulic_minus_nacelle"], "Converter": ["igbt_spread"]}
FAULT_TO_SUBSYSTEM = {"generator bearing": ["Generator bearings"], "gearbox": ["Gearbox"],
                      "transformer": ["Transformer"], "hydraulic": ["Hydraulic group", "Hub / pitch"]}


# ---------------------------------------------------------------- data
def load(out_dir):
    d = {}
    for f in glob.glob(os.path.join(out_dir, "scores", "*.csv")):
        df = pd.read_csv(f)
        df["in_event_window"] = df["in_event_window"].astype(bool)
        df["monitored"] = df["monitored"].astype(bool) if "monitored" in df else True
        d[int(os.path.splitext(os.path.basename(f))[0])] = df
    return d


def component_scores(df, subsystems, hours):
    """Smoothed score per component (same rule as the detection scripts)."""
    w = int(hours * STEPS_PER_HOUR)
    out = {}
    for comp, sens in subsystems.items():
        cols = [f"z_{s}" for s in sens if f"z_{s}" in df]
        if cols:
            zz = df[cols].rolling(w, min_periods=max(1, w // 3)).mean().clip(lower=0)
            out[comp] = zz.max(axis=1).to_numpy()
    return pd.DataFrame(out)


def prepare(data, subsystems, hours):
    prep = {}
    for ds, df in data.items():
        cs = component_scores(df, subsystems, hours)
        win = df["in_event_window"].to_numpy(bool)
        ids = df["id"].to_numpy()
        prep[ds] = dict(comp=cs.fillna(0).to_numpy(), names=list(cs.columns), win=win,
                        mon=df["monitored"].to_numpy(bool), is_anom=bool(win.any()),
                        before=(ids < ids[win].min()) if win.any() else np.ones(len(ids), bool))
    return prep


# ---------------------------------------------------------------- fast evaluation
def outcomes(p, comps_mask, thresholds=None):
    """detected / false alarm for every (threshold, alarm limit), vectorised.
    Criticality counter c_t = max(c_{t-1} + x_t, 0) is computed with the Lindley formula
    c_t = S_t - min(0, min_k<=t S_k), S = cumsum(x)."""
    score = p["comp"][:, comps_mask].max(axis=1) if comps_mask.any() else np.zeros(len(p["win"]))
    th = THRESHOLDS if thresholds is None else thresholds
    anom = score[None, :] > th[:, None]                               # (nT, n)
    x = np.where(p["mon"][None, :], np.where(anom, 1, -1), 0)
    S = np.cumsum(x, axis=1)
    c = S - np.minimum(0, np.minimum.accumulate(S, axis=1))
    alarm = c[:, None, :] >= CRIT_LIMITS[None, :, None]               # (nT, nC, n)
    det = (alarm & p["win"]).any(axis=2) if p["is_anom"] else np.zeros(alarm.shape[:2], bool)
    fa = (alarm & p["before"]).any(axis=2)
    return det, fa


def table(prep, ds_list, mask_of, thresholds=None):
    th = THRESHOLDS if thresholds is None else thresholds
    det = np.zeros((len(th), len(CRIT_LIMITS))); fa_norm = det.copy(); early = det.copy()
    n_an = sum(prep[d]["is_anom"] for d in ds_list); n_no = len(ds_list) - n_an
    for d in ds_list:
        de, fa = outcomes(prep[d], mask_of(d), th)
        if prep[d]["is_anom"]:
            det += de; early += fa
        else:
            fa_norm += fa
    rate = det / max(n_an, 1)
    bacc = 0.5 * (rate + 1 - fa_norm / max(n_no, 1))
    sel = bacc - 0.5 * early / max(n_an, 1) * np.minimum(rate, 1)
    return sel, bacc, det, fa_norm, early


def pick(sel, fa_norm, det, thresholds=None):
    # best selection score; ties -> fewer false alarms, more detections, stricter setting
    th = THRESHOLDS if thresholds is None else thresholds
    best = None
    for i in range(len(th)):
        for j in range(len(CRIT_LIMITS)):
            key = (round(sel[i, j], 9), -fa_norm[i, j], det[i, j], th[i], CRIT_LIMITS[j])
            if best is None or key > best[0]:
                best = (key, i, j)
    return best[1], best[2]


def select_components(prep, train_ds, names, max_drop=4):
    """Greedy: switch off components while it improves the training-fold score."""
    keep = np.ones(len(names), bool)
    def score(k):
        sel, _, det, fa, _ = table(prep, train_ds, lambda d: k)
        i, j = pick(sel, fa, det)
        return sel[i, j]
    cur = score(keep)
    for _ in range(max_drop):
        trials = []
        for c in np.where(keep)[0]:
            k = keep.copy(); k[c] = False
            if k.any():
                trials.append((score(k), c))
        if not trials:
            break
        s, c = max(trials)
        if s <= cur + 1e-9:
            break
        keep[c] = False; cur = s
    return keep


def evaluate(prep, select=False):
    ds_all = sorted(prep)
    names = prep[ds_all[0]]["names"]
    full = np.ones(len(names), bool)
    sel, bacc, det, fa, early = table(prep, ds_all, lambda d: full)
    i, j = pick(sel, fa, det)
    optimistic = dict(threshold=THRESHOLDS[i], alarm_h=CRIT_LIMITS[j] / STEPS_PER_HOUR,
                      detected=int(det[i, j]), false_alarms=int(fa[i, j]), bacc=round(float(bacc[i, j]), 3))
    rows, dropped = [], []
    for held in ds_all:
        train = [d for d in ds_all if d != held]
        keep = select_components(prep, train, names) if select else full
        dropped += [names[k] for k in np.where(~keep)[0]]
        s, b, de, f, e = table(prep, train, lambda d: keep)
        ii, jj = pick(s, f, de)
        dd, ff = outcomes(prep[held], keep)
        rows.append(dict(dataset=held, is_anom=prep[held]["is_anom"], detected=bool(dd[ii, jj]),
                         false_alarm=bool(ff[ii, jj]), T=THRESHOLDS[ii], C=CRIT_LIMITS[jj]))
    cv = pd.DataFrame(rows)
    an, no = cv[cv.is_anom], cv[~cv.is_anom]
    tp, fp = int(an.detected.sum()), int(no.false_alarm.sum())
    res = dict(cv_detected=tp, anomaly=len(an), cv_false_alarms=fp, normal=len(no),
               cv_early_alarms=int(an.false_alarm.sum()),
               cv_bacc=round(0.5 * (tp / max(len(an), 1) + 1 - fp / max(len(no), 1)), 3),
               optimistic=optimistic, per_dataset=cv)
    if select:
        res["dropped_counts"] = pd.Series(dropped).value_counts().to_dict() if dropped else {}
    return res



# ---------------------------------------------------------------- F: per-component thresholds from healthy turbines
REL_THRESHOLDS = np.round(np.arange(0.6, 2.51, 0.05), 2)


def healthy_levels(prep, normal_ds, q):
    """Normal level of every component = high quantile of its score on the healthy turbines."""
    pooled = np.concatenate([prep[d]["comp"] for d in normal_ds], axis=0)
    return np.maximum(np.quantile(pooled, q, axis=0), 1.0)


def scaled(prep, levels):
    return {d: dict(p, comp=p["comp"] / levels) for d, p in prep.items()}


def evaluate_per_component(prep, q):
    ds_all = sorted(prep)
    full = np.ones(len(prep[ds_all[0]]["names"]), bool)
    rows = []
    for held in ds_all:
        train = [d for d in ds_all if d != held]
        lv = healthy_levels(prep, [d for d in train if not prep[d]["is_anom"]], q)   # held-out turbine NOT used
        sp = scaled(prep, lv)
        s, b, de, f, e = table(sp, train, lambda d: full, REL_THRESHOLDS)
        ii, jj = pick(s, f, de, REL_THRESHOLDS)
        dd, ff = outcomes(sp[held], full, REL_THRESHOLDS)
        rows.append(dict(dataset=held, is_anom=prep[held]["is_anom"], detected=bool(dd[ii, jj]),
                         false_alarm=bool(ff[ii, jj]), T=REL_THRESHOLDS[ii], C=CRIT_LIMITS[jj]))
    cv = pd.DataFrame(rows)
    an, no = cv[cv.is_anom], cv[~cv.is_anom]
    tp, fp = int(an.detected.sum()), int(no.false_alarm.sum())
    lv = healthy_levels(prep, [d for d in ds_all if not prep[d]["is_anom"]], q)
    s, b, de, f, e = table(scaled(prep, lv), ds_all, lambda d: full, REL_THRESHOLDS)
    i, j = pick(s, f, de, REL_THRESHOLDS)
    return dict(cv_detected=tp, anomaly=len(an), cv_false_alarms=fp, normal=len(no),
                cv_early_alarms=int(an.false_alarm.sum()),
                cv_bacc=round(0.5 * (tp / max(len(an), 1) + 1 - fp / max(len(no), 1)), 3),
                optimistic=dict(threshold=REL_THRESHOLDS[i], alarm_h=CRIT_LIMITS[j] / STEPS_PER_HOUR,
                                detected=int(de[i, j]), false_alarms=int(f[i, j]), bacc=round(float(b[i, j]), 3)),
                per_dataset=cv)


# ---------------------------------------------------------------- separability: can this fault be caught at all?
def separability(prep, faults):
    names = prep[sorted(prep)[0]]["names"]
    normal = [d for d in prep if not prep[d]["is_anom"]]
    healthy_max = pd.DataFrame({d: prep[d]["comp"].max(axis=0) for d in normal}, index=names)   # comp x normal ds
    rows = []
    for d in sorted(prep):
        p = prep[d]
        if not p["is_anom"]:
            continue
        win_peak = pd.Series(p["comp"][p["win"]].max(axis=0), index=names)
        ratio = win_peak / healthy_max.max(axis=1).clip(lower=1e-6)
        fault = str(faults.get(d, ""))
        exp = next((v for k, v in FAULT_TO_SUBSYSTEM.items() if k in fault.lower()), [])
        exp = [e for e in exp if e in names]
        e_best = max(exp, key=lambda c: ratio[c]) if exp else None
        rows.append({"dataset": d, "fault": fault,
                     "expected component": e_best or "-",
                     "its peak in fault window": round(float(win_peak[e_best]), 1) if e_best else None,
                     "same component, max on healthy turbines": round(float(healthy_max.loc[e_best].max()), 1) if e_best else None,
                     "ratio (expected)": round(float(ratio[e_best]), 2) if e_best else None,
                     "best component": ratio.idxmax(), "ratio (best)": round(float(ratio.max()), 2)})
    sep = pd.DataFrame(rows)
    culprits = pd.DataFrame({"highest on healthy turbines": healthy_max.max(axis=1).round(1),
                             "in which dataset": healthy_max.idxmax(axis=1),
                             "median healthy turbine": healthy_max.median(axis=1).round(1)}).sort_values(
                                 "highest on healthy turbines", ascending=False)
    return sep, culprits


# ---------------------------------------------------------------- main
def main():
    v1_dir = os.path.join(HERE, "outputs_fault_detection")
    v2_dir = os.path.join(HERE, "outputs_fault_detection_v2")
    out = os.path.join(HERE, "outputs_improve"); os.makedirs(out, exist_ok=True)
    if not os.path.isdir(os.path.join(v1_dir, "scores")):
        sys.exit("Run fault_detection.py first.")
    t0 = time.time()
    v1 = load(v1_dir)
    v2 = load(v2_dir) if os.path.isdir(os.path.join(v2_dir, "scores")) else None
    with_phys = {k: v + PHYSICS.get(k, []) for k, v in V1_SUBSYSTEMS.items()}

    variants = [("A  v1 as reported (6 h)", v1, V1_SUBSYSTEMS, 6)]
    if v2:
        variants += [("B  robust training (v2 models), 6 h", v2, V1_SUBSYSTEMS, 6),
                     ("C  B + physics signals, 6 h", v2, with_phys, 6)]
    results = []
    for name, data, subs, h in variants:
        r = evaluate(prepare(data, subs, h)); r["name"] = name; r["cfg"] = (data is v2, subs is with_phys, h)
        results.append(r); print(f"{name:<48} CV {r['cv_detected']}/{r['anomaly']} faults, "
                                 f"{r['cv_false_alarms']}/{r['normal']} false alarms, bal.acc {r['cv_bacc']}  ({time.time() - t0:.0f}s)")
    best = max(results, key=lambda r: (r["cv_bacc"], -r["cv_false_alarms"]))
    use_v2, phys, _ = best["cfg"]
    data, subs = (v2 if use_v2 else v1), (with_phys if phys else V1_SUBSYSTEMS)
    r = evaluate(prepare(data, subs, 12)); r["name"] = f"D  best so far ({best['name'][:1]}) with 12 h smoothing"
    r["cfg"] = (use_v2, phys, 12); results.append(r)
    print(f"{r['name']:<48} CV {r['cv_detected']}/{r['anomaly']} faults, {r['cv_false_alarms']}/{r['normal']} false alarms, "
          f"bal.acc {r['cv_bacc']}  ({time.time() - t0:.0f}s)")
    best = max(results, key=lambda r: (r["cv_bacc"], -r["cv_false_alarms"]))
    use_v2, phys, h = best["cfg"]
    data, subs = (v2 if use_v2 else v1), (with_phys if phys else V1_SUBSYSTEMS)
    print("E: choosing components inside every cross-validation fold (takes a few minutes) ...")
    r = evaluate(prepare(data, subs, h), select=True); r["name"] = f"E  {best['name'][:1]} + component selection in CV"
    r["cfg"] = (use_v2, phys, h); results.append(r)
    print(f"{r['name']:<48} CV {r['cv_detected']}/{r['anomaly']} faults, {r['cv_false_alarms']}/{r['normal']} false alarms, "
          f"bal.acc {r['cv_bacc']}  ({time.time() - t0:.0f}s)")
    if r.get("dropped_counts"):
        n = len(r["per_dataset"])
        print("   components switched off (in how many of the", n, "folds):",
              ", ".join(f"{k} {v}" for k, v in r["dropped_counts"].items()))

    for base_name, data_f, subs_f in ([("A", v1, V1_SUBSYSTEMS)] + ([("C", v2, with_phys)] if v2 else [])):
        prep_f = prepare(data_f, subs_f, 6)
        for q in (0.999, 0.9999):
            r = evaluate_per_component(prep_f, q)
            r["name"] = f"F  {base_name} + per-component thresholds (healthy q={q})"
            r["cfg"] = (data_f is v2, subs_f is with_phys, 6)
            results.append(r)
            print(f"{r['name']:<48} CV {r['cv_detected']}/{r['anomaly']} faults, {r['cv_false_alarms']}/{r['normal']} "
                  f"false alarms, bal.acc {r['cv_bacc']}  ({time.time() - t0:.0f}s)")

    # separability report (v1 signals + physics if available)
    res_csv = os.path.join(v1_dir, "results_per_dataset.csv")
    faults = pd.read_csv(res_csv).set_index("dataset_id")["fault"].to_dict() if os.path.isfile(res_csv) else {}
    sep, culprits = separability(prepare(v2 if v2 else v1, with_phys if v2 else V1_SUBSYSTEMS, 6), faults)
    sep.to_csv(os.path.join(out, "separability.csv"), index=False); culprits.to_csv(os.path.join(out, "healthy_levels.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n==========  CAN EACH FAULT BE SEPARATED FROM HEALTHY TURBINES?  (ratio > 1 = higher than any healthy turbine)  ==========")
        print(sep.to_string(index=False))
        print("\n==========  WHAT SETS THE BAR: highest score of each component on the 10 healthy turbines  ==========")
        print(culprits.to_string())

    tab = pd.DataFrame([{"variant": r["name"], "CV faults detected": f"{r['cv_detected']}/{r['anomaly']}",
                         "CV false alarms (healthy)": f"{r['cv_false_alarms']}/{r['normal']}",
                         "CV alarms before fault": r["cv_early_alarms"], "CV balanced accuracy": r["cv_bacc"],
                         "best setting (all data, optimistic)": f"T={r['optimistic']['threshold']}, "
                                                                f"{r['optimistic']['alarm_h']:g} h -> "
                                                                f"{r['optimistic']['detected']}/{r['anomaly']}, "
                                                                f"{r['optimistic']['false_alarms']} FA"} for r in results])
    tab.to_csv(os.path.join(out, "variants.csv"), index=False)
    print("\n" + tab.to_string(index=False))

    winner = max(results, key=lambda r: (r["cv_bacc"], -r["cv_false_alarms"], r["cv_detected"]))
    per = winner["per_dataset"]
    missed = per[per.is_anom & ~per.detected]["dataset"].tolist()
    json.dump({"best_variant": winner["name"], "cv_detected": winner["cv_detected"],
               "cv_false_alarms": winner["cv_false_alarms"], "cv_balanced_accuracy": winner["cv_bacc"],
               "uses_v2_models": winner["cfg"][0], "physics_signals": winner["cfg"][1], "smoothing_hours": winner["cfg"][2],
               "components_switched_off": winner.get("dropped_counts", {}), "still_missed_datasets": missed},
              open(os.path.join(out, "best_setting.json"), "w"), indent=2, default=str)
    print(f"\nBEST (honest, cross-validated): {winner['name']}  ->  {winner['cv_detected']}/{winner['anomaly']} faults, "
          f"{winner['cv_false_alarms']}/{winner['normal']} false alarms.  Still missed: {missed}")

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(results))[::-1]
    ax.barh(y + 0.18, [r["cv_detected"] / max(r["anomaly"], 1) * 100 for r in results], 0.36, color="#0E9F8E", label="faults detected (%)")
    ax.barh(y - 0.18, [r["cv_false_alarms"] / max(r["normal"], 1) * 100 for r in results], 0.36, color="#E5484D", label="healthy turbines with false alarm (%)")
    ax.set_yticks(y, [r["name"] for r in results], fontsize=9); ax.set_xlim(0, 105)
    ax.set_xlabel("% (cross-validated)"); ax.set_title("What actually improves detection (leave-one-dataset-out)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "variants.png"), dpi=140); plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
