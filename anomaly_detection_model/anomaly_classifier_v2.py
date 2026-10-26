"""
Step 1, attempt 2 - low false alarms + fault-type classification (Wind Farm A, real data)
=========================================================================================
Lessons from attempt 1: 175-feature classifiers learned event-specific patterns -> many false
alarms on healthy turbines and fault types no better than guessing. This attempt:

  * monitors ONLY the components that correspond to a fault type in this dataset
      gearbox -> "gearbox", generator bearings -> "generator_bearing", transformer -> "transformer",
      hydraulic group + hub/pitch -> "hydraulic"
    (decided from the physics, not tuned on the test turbine; components without any recorded
     fault - cabinets, converter, windings, electrical, power, autoencoder - can only add false alarms)
  * each component score is divided by its NORMAL LEVEL, learned in every fold from the training
    turbines' healthy rows (healthy datasets + time before each fault): a quantile q of that score
  * alarm when the highest relative score stays above T for H hours (criticality counter on running rows)
  * q, T, H are chosen INSIDE each fold with a strict rule: on the training turbines at most 10 %
    of healthy datasets may raise an alarm; among the settings that satisfy it, the one with the
    most correctly classified training faults wins (ties -> stricter)
  * fault type: E = physics rule (component furthest above its normal level),
                F = small Gradient Boosting classifier on 16 physics features, trained on training-fold fault rows

Same honest evaluation as attempt 1: leave-one-group-out (overlapping datasets together), running
rows only, logbook status flags NOT used as inputs, out-of-fold predictions for every row.
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

import anomaly_classifier as a1          # same data loader, metrics and plots as attempt 1

STEPS_PER_HOUR = 6
COMP_CLASS = {"Gearbox": "gearbox", "Generator bearings": "generator_bearing", "Transformer": "transformer",
              "Hydraulic group": "hydraulic", "Hub / pitch": "hydraulic"}
COMPS = list(COMP_CLASS)
QS = [0.99, 0.995, 0.999, 1.0]                      # 1.0 = max seen on healthy training rows
TS = np.round(np.arange(0.8, 3.01, 0.1), 2)
HS = [1, 3, 6]
CLS_FEATS = ["d_gbx_bearing_oil", "d_gen_de_nde", "d_hyd_nacelle", "d_hub_nacelle", "spread_stator", "spread_trafo",
             "n_s11", "n_s12", "n_s13", "n_s14", "n_s41", "n_s38", "n_s39", "n_s40", "n_s6", "pitch"]


def lindley(x):
    S = np.cumsum(x, axis=-1)
    return S - np.minimum(0, np.minimum.accumulate(S, axis=-1))


def prep_groups(run):
    """Per dataset: component score matrix (running rows, in time order) and labels."""
    out = {}
    for ds, g in run.groupby("dataset"):
        g = g.sort_values("time_stamp")
        out[ds] = dict(idx=g.index.to_numpy(), comp=g[COMPS].fillna(0).to_numpy(), y=g["y"].to_numpy(),
                       is_anom=g["event_label"].iloc[0] == "anomaly", group=g["group"].iloc[0])
        out[ds]["true"] = next((c for c in out[ds]["y"] if c != "normal"), "normal")
        out[ds]["win"] = out[ds]["y"] != "normal"
    return out


def levels(P, train_ds, q):
    rows = [P[d]["comp"][~P[d]["win"]] for d in train_ds]            # healthy rows only
    pooled = np.concatenate(rows, axis=0)
    lv = np.quantile(pooled, q, axis=0) if q < 1 else pooled.max(axis=0)
    return np.maximum(lv, 0.5)


def decide(p, lv, T, H):
    rel = p["comp"] / lv
    top = rel.max(axis=1)
    cls = np.array([COMP_CLASS[COMPS[i]] for i in rel.argmax(axis=1)])
    crit = lindley(np.where(top > T, 1, -1))
    alarm = crit >= H * STEPS_PER_HOUR
    return alarm, cls


def outcome(p, alarm, cls):
    """detected / correct class (majority of alarm rows in window) / false alarm."""
    if p["is_anom"]:
        a_win = alarm & p["win"]
        det = bool(a_win.any())
        corr = det and pd.Series(cls[a_win]).value_counts().index[0] == p["true"]
        early = bool((alarm & ~p["win"] & (np.cumsum(p["win"]) == 0)).any())
        return det, corr, early
    return False, False, bool(alarm.any())


def choose_setting(P, train_ds, max_fa_rate=0.10):
    normals = [d for d in train_ds if not P[d]["is_anom"]]
    best = None
    for q in QS:
        lv = levels(P, train_ds, q)
        for T in TS:
            for H in HS:
                corr = det = fa = 0
                for d in train_ds:
                    al, cl = decide(P[d], lv, T, H)
                    de, c, f = outcome(P[d], al, cl)
                    corr += c; det += de; fa += f if not P[d]["is_anom"] else 0
                ok = fa <= max_fa_rate * max(1, len(normals))
                key = (ok, corr - (0 if ok else fa), det, T, H, -q if q < 1 else -1)
                if best is None or key > best[0]:
                    best = (key, q, T, H)
    return best[1], best[2], best[3]


def run(run_df, use_ml, log):
    P = prep_groups(run_df)
    pred = pd.Series("normal", index=run_df.index, dtype=object)
    chosen = []
    groups = sorted({P[d]["group"] for d in P})
    for g in groups:
        test = [d for d in P if P[d]["group"] == g]
        train = [d for d in P if P[d]["group"] != g]
        q, T, H = choose_setting(P, train)
        lv = levels(P, train, q)
        chosen.append(dict(group=g, q=q, T=T, H=H))
        clf = None
        if use_ml:
            tr_rows = run_df[run_df["dataset"].isin(train) & (run_df["y"] != "normal")]
            clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
                                                 min_samples_leaf=60, l2_regularization=2.0, random_state=0)
            clf.fit(tr_rows[CLS_FEATS], tr_rows["y"], sample_weight=compute_sample_weight("balanced", tr_rows["y"]))
        for d in test:
            al, cl = decide(P[d], lv, T, H)
            if use_ml and al.any():
                rows = run_df.loc[P[d]["idx"][al], CLS_FEATS]
                pr = pd.DataFrame(clf.predict_proba(rows), columns=clf.classes_)
                pr = pr.rolling(6 * STEPS_PER_HOUR, min_periods=1).mean()
                cl = cl.copy(); cl[al] = pr.idxmax(axis=1).to_numpy()
            pred[P[d]["idx"]] = np.where(al, cl, "normal")
    log(f"   {'F (ML classifier)' if use_ml else 'E (physics rule)'}: settings chosen per fold: " +
        pd.DataFrame(chosen)[["q", "T", "H"]].value_counts().head(3).to_string().replace("\n", " | "))
    return pred, pd.DataFrame(chosen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_attempt2"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(str(s))
    t0 = time.time()
    data, ev = a1.load_all(a.src)
    runrows = data[data["running"]].copy()
    log(f"Running rows: {len(runrows):,}   CV groups: {runrows['group'].nunique()}   components: {COMPS}")

    preds = {"A": a1.baseline_v1(runrows)}
    preds["E"], chE = run(runrows, False, log)
    preds["F"], chF = run(runrows, True, log)
    names = {"A": "A  old model (v1 threshold + diagnosis)",
             "E": "E  fault-component detector + physics rule (new)",
             "F": "F  fault-component detector + ML fault-type classifier (new)"}
    prev = pd.read_csv(os.path.join(os.path.dirname(a.out), "results_attempt1", "before_vs_after.csv")) \
        if os.path.isfile(os.path.join(os.path.dirname(a.out), "results_attempt1", "before_vs_after.csv")) else None
    summary, res = [], {}
    for k, p in preds.items():
        tab, cmdf, acc, mf1 = a1.row_metrics(runrows["y"], p)
        evm = a1.event_metrics(runrows, p)
        an, no = evm[evm.true != "normal"], evm[evm.true == "normal"]
        # event-level classification with the majority rule used for E/F
        res[k] = (tab, cmdf, acc, mf1, evm)
        fault_rows = runrows["y"] != "normal"
        flagged = p != "normal"
        summary.append({"approach": names[k], "accuracy": round(acc, 3), "macro-F1": round(mf1, 3),
                        "precision of fault alarms (rows)": round(float((flagged & fault_rows).sum() / max(1, flagged.sum())), 3),
                        "correct type when flagged (rows)": round(float(((p == runrows["y"]) & flagged & fault_rows).sum()
                                                                        / max(1, (flagged & fault_rows).sum())), 3),
                        "events detected": f"{int(an.detected.sum())}/{len(an)}",
                        "events correct class": f"{int(an.correct_class.sum())}/{len(an)}",
                        "false alarms (healthy)": f"{int(no.false_alarm.sum())}/{len(no)}"})
    summ = pd.DataFrame(summary)
    log("\n=== Attempt 2 vs old model (leave-one-group-out, out-of-fold, running rows) ===")
    log(summ.to_string(index=False))
    if prev is not None:
        log("\n=== Attempt 1 (for comparison) ===")
        log(prev.to_string(index=False))
    for k in ("E", "F"):
        tab, cmdf, acc, mf1, evm = res[k]
        log(f"\n--- {names[k]}: per-class metrics (rows) ---\n{tab.to_string()}\naccuracy {acc:.3f}   macro-F1 {mf1:.3f}")
        log(f"--- confusion matrix (rows) ---\n{cmdf.to_string()}")
        log(f"--- event level ---\n{evm.to_string(index=False)}")
        a1.plot_cm(cmdf, f"{names[k]} - confusion matrix (out-of-fold)", os.path.join(a.out, f"confusion_{k}.png"))
        tab.to_csv(os.path.join(a.out, f"per_class_{k}.csv")); cmdf.to_csv(os.path.join(a.out, f"confusion_{k}.csv"))
        evm.to_csv(os.path.join(a.out, f"events_{k}.csv"), index=False)
    summ.to_csv(os.path.join(a.out, "attempt2_summary.csv"), index=False)

    # final model on all data (settings chosen on all datasets with the same strict rule)
    P = prep_groups(runrows)
    q, T, H = choose_setting(P, list(P))
    lv = levels(P, list(P), q)
    fault = runrows[runrows["y"] != "normal"]
    clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=60,
                                         l2_regularization=2.0, random_state=0).fit(
        fault[CLS_FEATS], fault["y"], sample_weight=compute_sample_weight("balanced", fault["y"]))
    joblib.dump({"components": COMPS, "component_class": COMP_CLASS, "normal_levels": dict(zip(COMPS, lv.tolist())),
                 "quantile": q, "threshold_relative": float(T), "alarm_hours": int(H),
                 "type_classifier": clf, "classifier_features": CLS_FEATS},
                os.path.join(os.path.dirname(a.out), "anomaly_detector_v2.joblib"))
    log(f"\nFinal setting (all data): normal level = q{q}, threshold {T} x normal level, alarm after {H} h")
    open(os.path.join(a.out, "attempt2_report.txt"), "w").write("\n".join(lines))
    log(f"Saved to {a.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
