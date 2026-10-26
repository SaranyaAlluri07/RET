"""
Step 1 - Improved anomaly detection AND classification (Wind Farm A, real data)
================================================================================
Task: for every 10-minute row of a RUNNING turbine, say whether it is normal or which fault
type is developing: hydraulic group, gearbox, generator bearing, transformer.
(Farm A has 12 fault EVENTS of 4 fault TYPES - not 12 classes.)

Ground truth : event_info.csv windows (event_start_id .. event_end_id) of the anomaly datasets;
               normal datasets and the time before each fault window = "normal".
Status flags : Farm A status labels come from the maintenance logbook and mark the fault
               windows themselves -> NOT used as a feature (that would be reading the answer).
               They are only shown in a leakage demonstration (approach X).
Rows used    : turbine running (rotor rpm > 1 and power > 2 % of rated), so the model cannot
               learn "turbine stopped for repair" = fault.

Features (per row, all relative to the same turbine's own healthy behaviour):
  * residual z-scores of 30 normal-behaviour models (fault_detection_v2 scores), incl. physics
    differences (gearbox bearing - oil, generator DE - NDE, phase spreads, hydraulic - nacelle)
  * power residual (actual - expected), autoencoder error
  * raw signals normalised by the turbine's healthy median / IQR: temperature deltas, pitch,
    yaw misalignment, rotor/generator speed ratio, phase current imbalance
  * rolling mean / max / std over 6 h and 24 h, lags 1 h / 6 h / 24 h of the component scores

Evaluation   : leave-one-GROUP-out cross-validation. A group = all datasets of the same turbine
               whose periods overlap in time (they share rows), so the test turbine-period is never
               seen in training (no leakage, time-aware). Out-of-fold predictions for every row ->
               per-class precision / recall / F1, confusion matrix, accuracy, macro-F1, and
               event-level detection (sustained 3 h of the right class inside the fault window).

Approaches   : A  old model (v1 threshold alarm + diagnosed component)
               B  Gradient Boosting multiclass, class-balanced weights
               C  Random Forest, balanced subsample
               D  hybrid: residual detector (GBM, normal vs fault) + fault-type classifier (GBM)
               X  leakage demo: B + status flags + random row split (NOT valid, shown for contrast)
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore")
import argparse
import glob
import json
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix, accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

STEPS_PER_HOUR = 6
CLASSES = ["normal", "hydraulic", "gearbox", "generator_bearing", "transformer"]
FAULT_MAP = [("hydraulic", "hydraulic"), ("gearbox", "gearbox"), ("generator bearing", "generator_bearing"),
             ("transformer", "transformer")]
SUB_TO_CLASS = {"Hydraulic group": "hydraulic", "Hub / pitch": "hydraulic", "Gearbox": "gearbox",
                "Generator bearings": "generator_bearing", "Transformer": "transformer"}
SUBSYSTEMS = ["Generator bearings", "Gearbox", "Generator windings", "Transformer", "Hydraulic group",
              "Hub / pitch", "Converter", "Nacelle cabinets", "Electrical balance", "Power performance",
              "Whole turbine (AE)"]


def fault_class(desc):
    d = str(desc).lower()
    return next((c for k, c in FAULT_MAP if k in d), "other")


# ------------------------------------------------------------------ data
def _find(src, *parts):
    """Locate a folder anywhere below src (works for the original project and the bundle layout)."""
    direct = os.path.join(src, *parts)
    if os.path.isdir(direct):
        return direct
    hits = glob.glob(os.path.join(src, "**", *parts), recursive=True)
    if not hits:
        raise FileNotFoundError(os.path.join(*parts) + " not found below " + src)
    return hits[0]


def load_all(src):
    ev = pd.read_csv(glob.glob(os.path.join(src, "**", "event_info.csv"), recursive=True)[0], sep=";")
    proc, s_v2, s_v1 = _find(src, "data", "processed"), _find(src, "outputs_fault_detection_v2", "scores"), \
        _find(src, "outputs_fault_detection", "scores")
    frames = []
    for _, e in ev.iterrows():
        ds = int(e["event_id"])
        raw = pd.read_csv(os.path.join(proc, f"raw_{ds}.csv.gz"), parse_dates=["time_stamp"])
        base = json.load(open(os.path.join(proc, f"baseline_{ds}.json")))
        v2 = pd.read_csv(os.path.join(s_v2, f"{ds}.csv"), parse_dates=["time_stamp"])
        v1 = pd.read_csv(os.path.join(s_v1, f"{ds}.csv"), parse_dates=["time_stamp"])
        v1 = v1[["time_stamp", "anomaly_score", "alarm", "criticality"] + [s for s in SUBSYSTEMS if s in v1]]
        v1 = v1.rename(columns={c: f"v1_{c}" for c in v1.columns if c != "time_stamp"})
        df = raw.merge(v2.drop(columns=["id", "status_type_id", "in_event_window", "monitored", "threshold",
                                        "criticality", "alarm"], errors="ignore"), on="time_stamp", how="left")
        df = df.merge(v1, on="time_stamp", how="left")
        df = engineer(df, base)
        df = df[df["train_test"].astype(str).str.lower() != "train"].copy()        # prediction period only
        label = str(e["event_label"]).lower()
        df["dataset"], df["asset"], df["event_label"] = ds, int(e["asset"]), label
        df["fault"] = str(e.get("event_description") or "") if label == "anomaly" else ""
        if label == "anomaly":
            cls = fault_class(e["event_description"])
            inwin = df["id"].between(e["event_start_id"], e["event_end_id"])
            after = df["id"] > e["event_end_id"]
            df["y"] = np.where(inwin, cls, "normal")
            df["pos_in_event"] = np.where(inwin, (df["id"] - e["event_start_id"]) / max(1, e["event_end_id"] - e["event_start_id"]), np.nan)
            df = df[~after]
        else:
            df["y"], df["pos_in_event"] = "normal", np.nan
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["running"] = (data["sensor_52_avg"] > 1) & (data["power_30_avg"] > 0.02)
    data["group"] = overlap_groups(data)
    return data, ev


def overlap_groups(data):
    """Datasets of the same turbine whose periods overlap share rows -> same CV group."""
    span = data.groupby("dataset").agg(asset=("asset", "first"), t0=("time_stamp", "min"), t1=("time_stamp", "max"))
    parent = {d: d for d in span.index}
    def find(d):
        while parent[d] != d:
            d = parent[d]
        return d
    for a in span.index:
        for b in span.index:
            if a < b and span.at[a, "asset"] == span.at[b, "asset"] and \
               span.at[a, "t0"] <= span.at[b, "t1"] and span.at[b, "t0"] <= span.at[a, "t1"]:
                parent[find(a)] = find(b)
    return data["dataset"].map(lambda d: find(d))


def engineer(df, base):
    def nrm(c):
        if c not in df:
            return pd.Series(np.nan, index=df.index)
        med, iqr = base.get(c, [np.nan, np.nan])
        return (df[c] - med) / (iqr if iqr and iqr > 1e-6 else 1.0)
    f = df
    f["d_gbx_bearing_oil"] = nrm("sensor_11_avg") - nrm("sensor_12_avg")
    f["d_gen_de_nde"] = nrm("sensor_13_avg") - nrm("sensor_14_avg")
    f["d_hyd_nacelle"] = nrm("sensor_41_avg") - nrm("sensor_43_avg")
    f["d_hub_nacelle"] = nrm("sensor_6_avg") - nrm("sensor_43_avg")
    for name, cols in (("spread_stator", [15, 16, 17]), ("spread_trafo", [38, 39, 40]), ("spread_igbt", [35, 36, 37])):
        cc = [f"sensor_{i}_avg" for i in cols]
        f[name] = df[cc].max(axis=1) - df[cc].min(axis=1)
    for c in (11, 12, 13, 14, 41, 38, 39, 40, 6, 53, 43):
        f[f"n_s{c}"] = nrm(f"sensor_{c}_avg") - nrm("sensor_0_avg")            # above ambient, vs normal
    f["pitch"] = ((df["sensor_5_avg"] + 180) % 360) - 180
    f["yaw_misalign"] = (((df["sensor_2_avg"] + 180) % 360) - 180).abs()
    f["rpm_ratio"] = (df["sensor_18_avg"] / df["sensor_52_avg"].where(df["sensor_52_avg"] > 1))
    f["rpm_ratio_dev"] = f["rpm_ratio"] - f["rpm_ratio"].median()
    cur = df[["sensor_23_avg", "sensor_24_avg", "sensor_25_avg"]]
    f["current_imb"] = cur.std(axis=1) / cur.mean(axis=1).abs().clip(lower=1)
    f["wind"], f["power"] = df["wind_speed_3_avg"], df["power_30_avg"]
    keys = [s for s in SUBSYSTEMS if s in f] + ["z_power_30_avg", "z_pitch_dev", "z_ae_error",
                                                 "d_gbx_bearing_oil", "d_gen_de_nde", "d_hyd_nacelle", "yaw_misalign"]
    for c in keys:
        s = f[c]
        f[f"{c}_m6h"] = s.rolling(36, min_periods=6).mean()
        f[f"{c}_m24h"] = s.rolling(144, min_periods=24).mean()
        f[f"{c}_x24h"] = s.rolling(144, min_periods=24).max()
        f[f"{c}_s6h"] = s.rolling(36, min_periods=6).std()
    for c in [s for s in SUBSYSTEMS if s in f]:
        for lag, n in (("1h", 6), ("6h", 36), ("24h", 144)):
            f[f"{c}_lag{lag}"] = f[c].shift(n)
    return f


def feature_columns(data):
    drop = {"time_stamp", "asset_id", "id", "train_test", "status_type_id", "dataset", "asset", "event_label",
            "fault", "y", "pos_in_event", "running", "group", "anomaly_score"}
    raw_abs = {c for c in data.columns if c.startswith("sensor_") or c.startswith("wind_speed") or c == "power_30_avg"}
    return [c for c in data.columns if c not in drop and c not in raw_abs and not c.startswith("v1_")
            and data[c].dtype != object]


# ------------------------------------------------------------------ models
def gbm():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=80,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)


def rf():
    return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=0)


def fit_predict(kind, Xtr, ytr, Xte):
    w = compute_sample_weight("balanced", ytr)
    if kind == "B":
        m = gbm().fit(Xtr, ytr, sample_weight=w)
        return m.predict_proba(Xte), list(m.classes_), m
    if kind == "C":
        m = rf().fit(Xtr.fillna(-999), ytr)
        return m.predict_proba(Xte.fillna(-999)), list(m.classes_), m
    if kind == "D":                                       # detector + fault-type classifier
        yb = (ytr != "normal").astype(int)
        det = gbm().fit(Xtr, yb, sample_weight=compute_sample_weight("balanced", yb))
        fx = ytr != "normal"
        typ = gbm().fit(Xtr[fx], ytr[fx], sample_weight=compute_sample_weight("balanced", ytr[fx]))
        p_f = det.predict_proba(Xte)[:, 1]
        pt = typ.predict_proba(Xte)
        cls = ["normal"] + list(typ.classes_)
        return np.column_stack([1 - p_f, pt * p_f[:, None]]), cls, (det, typ)
    raise ValueError(kind)


def smooth_decide(prob, classes, keys, hours=6):
    """6-h rolling mean of the class probabilities within each dataset, then argmax."""
    P = pd.DataFrame(prob, columns=classes)
    P["k"] = keys
    P = P.groupby("k", group_keys=False)[classes].apply(lambda g: g.rolling(hours * STEPS_PER_HOUR, min_periods=1).mean())
    return P.idxmax(axis=1).to_numpy(), P


def run_cv(data, feats, kind, log):
    idx = data.index
    pred = pd.Series("normal", index=idx, dtype=object)
    probs = pd.DataFrame(0.0, index=idx, columns=CLASSES)
    for g in sorted(data["group"].unique()):
        te = data["group"] == g
        tr = (~te) & (data["id"] % 2 == 0)        # every 2nd 10-min row: neighbours are nearly identical
        p, cls, _ = fit_predict(kind, data.loc[tr, feats], data.loc[tr, "y"].to_numpy(), data.loc[te, feats])
        lab, P = smooth_decide(p, cls, data.loc[te, "dataset"].to_numpy())
        pred[te] = lab
        for c in cls:
            probs.loc[te, c] = P[c].to_numpy()
        log(f"      {kind} fold group {g} done")
    log(f"   {kind}: done ({data['group'].nunique()} folds)")
    return pred, probs


def baseline_v1(data):
    """Old model: alarm -> class of the diagnosed (highest) component, otherwise normal."""
    subs = [f"v1_{s}" for s in SUBSYSTEMS if f"v1_{s}" in data]
    top = data[subs].fillna(-1).idxmax(axis=1).str.replace("v1_", "", regex=False)
    cls = top.map(SUB_TO_CLASS).fillna("other_component")
    return pd.Series(np.where(data["v1_alarm"].fillna(False).astype(bool), cls, "normal"), index=data.index)


# ------------------------------------------------------------------ metrics
def row_metrics(y, p):
    labels = CLASSES
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    tab = pd.DataFrame({"precision": pr, "recall": rc, "F1": f1, "support (rows)": sup}, index=labels).round(3)
    extra = sorted(set(p) - set(labels))
    cm = confusion_matrix(y, p, labels=labels + extra)
    cmdf = pd.DataFrame(cm[:len(labels)], index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels + extra])
    return tab, cmdf, accuracy_score(y, p), f1_score(y, p, labels=labels, average="macro", zero_division=0)


def event_metrics(data, pred, hours=3):
    """Event level: detected AND classified if the correct class is predicted for >= `hours`
    in a row inside the fault window; false alarm if a normal dataset (or the time before a
    fault) has >= `hours` in a row of any fault class."""
    n = hours * STEPS_PER_HOUR
    rows = []
    for ds, g in data.groupby("dataset"):
        p = pred[g.index].to_numpy()
        lab = g["event_label"].iloc[0]
        def run_of(mask):
            best = cur = 0
            for v in mask:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            return best
        if lab == "anomaly":
            cls = g.loc[g["y"] != "normal", "y"]
            cls = cls.iloc[0] if len(cls) else "?"
            win = (g["y"] != "normal").to_numpy()
            right = run_of((p == cls) & win) >= n
            any_fault = run_of((p != "normal") & win) >= n
            early = run_of((p != "normal") & ~win) >= n
            # which class did it say most inside the window?
            said = pd.Series(p[win & (p != "normal")]).value_counts()
            rows.append(dict(dataset=ds, true=cls, detected=bool(any_fault), correct_class=bool(right),
                             predicted_mostly=said.index[0] if len(said) else "normal", alarm_before_fault=bool(early),
                             running_rows_in_window=int(win.sum())))
        else:
            rows.append(dict(dataset=ds, true="normal", false_alarm=bool(run_of(p != "normal") >= n)))
    return pd.DataFrame(rows)


def plot_cm(cmdf, title, path):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    m = cmdf.to_numpy().astype(float)
    norm = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(m.shape[1]), [c.replace("pred ", "") for c in cmdf.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(m.shape[0]), [c.replace("true ", "") for c in cmdf.index], fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{int(m[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else "#1F2D3D")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ------------------------------------------------------------------ main
def train_final(src=None, out_path=None, approach="C", log=print):
    """Train only the final model (no cross-validation, ~1-2 min) - used to rebuild the model file
    on another computer / scikit-learn version. Same data, features and settings as the evaluated model."""
    src = src or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = out_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "anomaly_classifier.joblib")
    data, _ = load_all(src)
    run = data[data["running"]].copy()
    feats = feature_columns(run)
    log(f"Training final model {approach} on {len(run):,} rows, {len(feats)} features ...")
    _, cls, model = fit_predict(approach, run[feats], run["y"].to_numpy(), run[feats].iloc[:5])
    joblib.dump({"approach": approach, "model": model, "features": feats, "classes": cls,
                 "smoothing_hours": 6, "event_rule_hours": 3}, out_path)
    log(f"Saved {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="project folder (with data/processed, outputs_fault_detection*, event_info.csv)")
    ap.add_argument("--final-only", action="store_true", help="only train and save the final model C (no cross-validation)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_attempt1"))
    ap.add_argument("--skip-rf", action="store_true")
    a = ap.parse_args()
    if a.final_only:
        train_final(a.src)
        return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(str(s))

    data, ev = load_all(a.src)
    run = data[data["running"]].copy()
    feats = feature_columns(run)
    log(f"Rows (prediction periods, turbine running): {len(run):,} of {len(data):,}   features: {len(feats)}   "
        f"CV groups: {run['group'].nunique()}")
    log("Rows per class: " + ", ".join(f"{k} {v:,}" for k, v in run["y"].value_counts().items()))
    log("Events per class: " + ", ".join(f"{k} {v}" for k, v in
                                       ev[ev.event_label == 'anomaly']['event_description'].map(fault_class).value_counts().items()))
    grp = run.groupby("group")["dataset"].unique()
    log("Overlapping datasets grouped together: " + "; ".join(str(sorted(v.tolist())) for v in grp if len(v) > 1))

    results, preds = {}, {}
    preds["A"] = baseline_v1(run)
    for k in ["B", "C", "D"]:
        if k == "C" and a.skip_rf:
            continue
        preds[k], _ = run_cv(run, feats, k, log)
    names = {"A": "A  old model (v1 threshold + diagnosis)", "B": "B  Gradient Boosting, class-balanced",
             "C": "C  Random Forest, balanced", "D": "D  residual detector + fault-type classifier"}
    summary = []
    for k, p in preds.items():
        tab, cmdf, acc, mf1 = row_metrics(run["y"], p)
        evm = event_metrics(run, p)
        an, no = evm[evm.true != "normal"], evm[evm.true == "normal"]
        results[k] = (tab, cmdf, acc, mf1, evm)
        summary.append({"approach": names[k], "accuracy": round(acc, 3), "macro-F1": round(mf1, 3),
                        "fault rows recall": round(float(((p != "normal") & (run["y"] != "normal")).sum() / max(1, (run["y"] != "normal").sum())), 3),
                        "events detected": f"{int(an.detected.sum())}/{len(an)}",
                        "events correct class": f"{int(an.correct_class.sum())}/{len(an)}",
                        "false alarms (healthy)": f"{int(no.false_alarm.sum())}/{len(no)}"})
    summ = pd.DataFrame(summary)
    log("\n=== Before vs after (leave-one-group-out, out-of-fold, running rows) ===")
    log(summ.to_string(index=False))
    best = max([k for k in preds if k != "A"], key=lambda k: (results[k][3], int(results[k][4].query("true!='normal'").correct_class.sum())))
    log(f"\nBest new approach: {names[best]}")
    for k in ("A", best):
        tab, cmdf, acc, mf1, evm = results[k]
        log(f"\n--- {names[k]}: per-class metrics (rows) ---\n{tab.to_string()}")
        log(f"accuracy {acc:.3f}   macro-F1 {mf1:.3f}")
        log(f"--- confusion matrix (rows) ---\n{cmdf.to_string()}")
        log(f"--- event level ---\n{evm.to_string(index=False)}")
        plot_cm(cmdf, f"{names[k]} - confusion matrix (out-of-fold)", os.path.join(a.out, f"confusion_{k}.png"))
        tab.to_csv(os.path.join(a.out, f"per_class_{k}.csv")); cmdf.to_csv(os.path.join(a.out, f"confusion_{k}.csv"))
        evm.to_csv(os.path.join(a.out, f"events_{k}.csv"), index=False)
    summ.to_csv(os.path.join(a.out, "before_vs_after.csv"), index=False)

    # ---- leakage demonstration (NOT a valid result) ----
    Xl = run[feats].assign(status=run["status_type_id"])
    Xtr, Xte, ytr, yte = train_test_split(Xl, run["y"], test_size=0.3, random_state=0, stratify=run["y"])
    m = gbm().fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    pl = m.predict(Xte)
    log(f"\n=== X  leakage demo: + logbook status flags + random row split (INVALID) ===\n"
        f"accuracy {accuracy_score(yte, pl):.3f}, macro-F1 {f1_score(yte, pl, average='macro'):.3f}  <- looks excellent, "
        f"but it reads the logbook and tests on rows next to training rows of the same event.")

    # ---- final model on all data + feature importance proxy ----
    p, cls, model = fit_predict(best, run[feats], run["y"].to_numpy(), run[feats].iloc[:5])
    joblib.dump({"approach": best, "model": model, "features": feats, "classes": cls,
                 "smoothing_hours": 6, "event_rule_hours": 3}, os.path.join(a.out, "..", "anomaly_classifier.joblib"))
    json.dump({"summary": summary, "best": names[best]}, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    open(os.path.join(a.out, "step1_report.txt"), "w").write("\n".join(lines))
    log(f"\nSaved to {a.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
