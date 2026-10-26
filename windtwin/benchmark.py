"""Benchmark CLI: every method under the same protocol and the same CARE / operational metrics.

    python -m windtwin.benchmark [--jobs 8] [--datasets 0 24 ...] [--rebuild]

Stages (cached in artifacts/): per-dataset NBM bundles + scored monitoring segments, baseline scores,
forecast model, alarm hyper-parameter grid evaluation, leave-one-turbine-out selection, final held-out
evaluation with diagnosis -> artifacts/benchmark/{benchmark.json, <method>.json, REPORT.md}."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C, data, alarms, care, monitor, baselines, diagnosis

warnings.filterwarnings("ignore")
LOG = None

METHODS = {
    "legacy_rf": ("Legacy Random Forest classifier (re-run LOGO CV)", "baseline"),
    "legacy_residual": ("Legacy residual detector v2 (saved scores)", "baseline"),
    "iforest": ("Isolation Forest on Avg signals", "baseline"),
    "autoencoder": ("MLP autoencoder on Avg signals", "baseline"),
    "nbm_threshold": ("NBM + regime-normalized residual + threshold", "proposed"),
    "nbm_ewma": ("NBM + normalized residual + EWMA", "proposed"),
    "nbm_cusum": ("NBM + normalized residual + CUSUM", "proposed"),
    "nbm_ewma_cusum_free": ("NBM + EWMA + CUSUM, persistence/cooldown also tuned (iteration 2)", "proposed"),
    "nbm_ewma_cusum": ("NBM + EWMA + CUSUM + persistence/hysteresis (6 h / 24 h)", "proposed"),
}
SELECTED = "nbm_ewma_cusum"


def grid(method):
    """Alarm hyper-parameter grids. Statistics are in units of their validation 99.5th percentile."""
    P = alarms.AlarmParams
    if method == "nbm_threshold":
        return [P(lam=1.0, L=L, use_cusum=False, persist=1, cooldown=1, clear=1.0) for L in (1, 1.5, 2, 3, 4, 6)]
    if method == "nbm_ewma":
        return [P(lam=lam, L=L, use_cusum=False, persist=1, cooldown=1, clear=1.0)
                for lam in (0.02, 0.05, 0.1, 0.2) for L in (1, 1.5, 2, 3, 4)]
    if method == "nbm_cusum":
        return [P(use_ewma=False, k=k, h=h, persist=1, cooldown=1, clear=1.0)
                for k in (0.5, 1.0, 2.0) for h in (1, 1.5, 2, 3, 4, 6)]
    if method == "nbm_ewma_cusum_free":
        return [P(lam=lam, L=L, k=k, h=h, persist=ps, cooldown=cd, clear=0.5, cusum_cap=1.5)
                for lam, L, k, h, ps, cd in itertools.product((0.02, 0.05, 0.1), (1.5, 2, 3), (0.5, 1.0, 2.0),
                                                              (1.5, 2, 3, 4), (6, 36), (36, 144))]
    if method == "nbm_ewma_cusum":
        # persistence 6 h and clear-cooldown 24 h fixed (literature: CARE 12 h criticality, 24 h hysteresis);
        # only the detector limits are tuned -> fewer free parameters, stable leave-one-turbine-out choices
        return [P(lam=lam, L=L, k=k, h=h, persist=36, cooldown=144, clear=0.5, cusum_cap=1.5)
                for lam, L, k, h in itertools.product((0.02, 0.05, 0.1), (1.5, 2, 3), (0.5, 1.0, 2.0), (1.5, 2, 3, 4))]
    raise KeyError(method)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------ stage 1: bundles + segments
def prepare_dataset(ds, rebuild=False, rescore=False):
    """Fit (or load) the bundle and cache the scored monitoring segment. rescore: keep the fitted
    models but recalibrate normalizer / EWMA scales (validation split only) and re-score."""
    seg_path = C.CACHE_DIR / f"seg_{ds}.npz"
    t0 = time.time()
    if not (rebuild or rescore) and seg_path.is_file() and monitor.bundle_path(ds).is_file():
        return ds, 0.0
    if rebuild or not monitor.bundle_path(ds).is_file():
        b = monitor.build_bundle(ds)
        monitor.save_bundle(b)
    else:
        b = monitor.load_bundle(ds)
        if rescore:
            monitor.calibrate(b)
            monitor.save_bundle(b)
    raw, mi, pi = monitor.segment(ds)
    df = data.prepare(raw)
    s = monitor.score_frame(b, df)
    np.savez_compressed(seg_path, Z=s["Z"][mi:].astype(np.float64), bad=s["bad"][mi:], off=s["off"][mi:],
                        times=df["time_stamp"].to_numpy()[mi:], is_pred=(df["train_test"] != "train").to_numpy()[mi:],
                        targets=np.array(b.targets), scales=np.array([b.scales[l] for l in monitor.LAMBDAS]),
                        lambdas=np.array(monitor.LAMBDAS), cscales=np.array([b.cscales[k] for k in monitor.KS]),
                        ks=np.array(monitor.KS))
    return ds, time.time() - t0


def load_seg(ds):
    z = np.load(C.CACHE_DIR / f"seg_{ds}.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["targets"] = [str(t) for t in d["targets"]]
    d["two_sided"] = np.array([C.DIRECTION.get(t, 1) == 0 for t in d["targets"]])
    d["comp_idx"] = alarms.comp_index(d["targets"])
    d["scale"] = {float(l): s for l, s in zip(d["lambdas"], d["scales"])}
    d["cscale"] = {float(k): s for k, s in zip(d["ks"], d["cscales"])}
    return d


# ------------------------------------------------------------------ evaluation helpers
def dataset_row(ds, ev, times, point, alarm, diag=None, extra=None):
    e = ev.loc[ds]
    label = e["event_label"]
    ce = care.evaluate_event(times, point, e["event_start"], e["event_end"], label)
    op = care.operational(times, alarm, e["event_start"], e["event_end"], label)
    row = {"id": int(ds), "asset": int(e["asset"]), "label": label, "description": e["event_description"],
           "care_event": ce, "op": op, "diagnosis": None, "diagnosis_correct": None, "diagnosis_unknown": False,
           "diagnosis_class": None}
    if diag is not None:
        row.update(diag)
    if extra:
        row.update(extra)
    return row


def diag_fields(comp, label, desc):
    """Normalise a diagnosed component into the result fields."""
    if comp is None:
        return {}
    unknown = comp == C.UNKNOWN
    acc = C.fault_components(desc) if label == "anomaly" else []
    return {"diagnosis": comp, "diagnosis_unknown": unknown, "diagnosis_class": None if unknown else comp,
            "diagnosis_correct": (None if label != "anomaly" else (False if unknown else comp in acc))}


def first_alarm_index(row, alarm):
    if row["label"] == "anomaly":
        return row["op"]["alarm_index"]
    eps = care.alarm_episodes(alarm)
    return int(eps[0][0]) if eps else None


def proposed_eval(seg, ev, ds, p, with_diag=False, Ecache=None, Scache=None):
    """Run the alarm logic for one dataset with params p; returns the dataset row."""
    Z, two, ci = seg["Z"], seg["two_sided"], seg["comp_idx"]
    lam = p.lam if p.use_ewma else 1.0
    key_e, key_s = lam, ((p.k, p.cusum_cap * p.h) if p.use_cusum else None)
    if Ecache is not None and key_e in Ecache:
        Ec = Ecache[key_e]
    else:
        Ev = alarms.evidence(alarms.ewma(Z, lam), seg["scale"][lam], two)
        seen = np.maximum.accumulate(np.isfinite(Z), axis=0)
        Ec = alarms.component_max(Ev, ci, seen)
        if Ecache is not None:
            Ecache[key_e] = Ec
    if Scache is not None and key_s in Scache:
        Sc = Scache[key_s]
    else:
        if p.use_cusum:
            seen = np.maximum.accumulate(np.isfinite(Z), axis=0)
            cs = seg["cscale"][float(p.k)]
            Sc = alarms.component_max(alarms.cusum(Z, p.k, two, cap=alarms.cap_of(p, cs)) / cs, ci, seen)
        else:
            Sc = np.zeros_like(Ec)
        if Scache is not None:
            Scache[key_s] = Sc
    out = alarms._decide(Ec, Sc, seg["bad"], seg["off"], p, np.isfinite(Z).any(axis=1))
    ip = seg["is_pred"]
    times = seg["times"][ip]
    row = dataset_row(ds, ev, times, out["alarm"][ip], out["alarm"][ip])
    if with_diag:
        i = first_alarm_index(row, out["alarm"][ip])
        if i is not None:
            gi = int(np.flatnonzero(ip)[i])
            d0 = out["dev_start"][gi]
            d0 = gi - 35 if d0 < 0 else min(d0, gi - 35)
            cstat = {}
            for c, name in enumerate(C.COMPONENTS):
                v = 0.0
                if p.use_ewma and np.isfinite(Ec[gi, c]):
                    v = max(v, Ec[gi, c] / p.L)
                if p.use_cusum and np.isfinite(Sc[gi, c]):
                    v = max(v, Sc[gi, c] / p.h)
                cstat[name] = float(v)
            d = diagnosis.diagnose(Z[max(0, d0):gi + 1], seg["targets"], two, comp_stat=cstat)
            row.update(diag_fields(d["component"], row["label"], row["description"]))
            row["diagnosis_detail"] = {"top_component": d["top_component"], "shares": d["shares"],
                                       "reasons": d["reasons"], "evidence": d["evidence"][:4]}
    return row


def eval_grid_dataset(ds, method, combos, ev):
    seg = load_seg(ds)
    Ecache, Scache = {}, {}
    res = []
    for p in combos:
        r = proposed_eval(seg, ev, ds, p, Ecache=Ecache, Scache=Scache)
        res.append({"care_event": r["care_event"], "op": r["op"], "label": r["label"], "asset": r["asset"], "id": r["id"]})
    return ds, res


def select(rows_by_combo, ds_list, assets, combos):
    """Index of the combo maximising CARE on datasets of `assets` (ties: fewer false alarms, then later =
    more conservative grid entries)."""
    best, best_key = None, None
    for ci in range(len(combos)):
        rows = [rows_by_combo[ci][d] for d in ds_list if rows_by_combo[ci][d]["asset"] in assets]
        try:
            fs = care.final_score([r["care_event"] for r in rows])
        except ValueError:
            continue
        fa = sum(r["op"]["false_alarms"] for r in rows if r["label"] != "anomaly")
        det = sum(1 for r in rows if r["label"] == "anomaly" and r["op"]["detected"])
        key = (round(fs["care"], 6), det - fa, ci)
        if best_key is None or key > best_key:
            best, best_key = ci, key
    return best


def method_result(mid, rows, notes=""):
    s = care.summarize(rows)
    return {"id": mid, "name": METHODS[mid][0], "kind": METHODS[mid][1], **s, "notes": notes}


def dataset_results(rows):
    out = []
    for r in rows:
        ce, op = r["care_event"], r["op"]
        out.append({"id": r["id"], "asset": r["asset"], "label": r["label"], "description": r["description"],
                    "detected": bool(op["detected"]) if r["label"] == "anomaly" else False,
                    "first_alarm": op["first_alarm"].isoformat() if op["first_alarm"] is not None else None,
                    "delay_h": round(op["delay_h"], 1) if op["delay_h"] is not None else None,
                    "lead_to_end_h": round(op["lead_to_end_h"], 1) if op["lead_to_end_h"] is not None else None,
                    "false_alarms": int(op["false_alarms"]), "pre_event_alarms": int(op["pre_event_alarms"]),
                    "diagnosis": r.get("diagnosis"), "diagnosis_correct": r.get("diagnosis_correct"),
                    "params": r.get("params"), "diagnosis_detail": r.get("diagnosis_detail"),
                    "care_parts": {"f_beta": _rd(ce["f_beta_score"]), "accuracy": _rd(ce["accuracy"]),
                                   "weighted_score": _rd(ce["weighted_score"]), "max_criticality": ce["max_criticality"],
                                   "detected_care": bool(ce["detected"]), "tp": ce["tp"], "fp": ce["fp"],
                                   "fn": ce["fn"], "tn": ce["tn"]}})
    return out


def _rd(v):
    return None if v is None else round(float(v), 4)


# ------------------------------------------------------------------ main
def main(argv=None):
    global LOG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--datasets", type=int, nargs="*")
    ap.add_argument("--rebuild", action="store_true", help="refit bundles even if cached")
    ap.add_argument("--rescore", action="store_true", help="recalibrate + re-score cached bundles (no refit)")
    ap.add_argument("--skip-rf", action="store_true")
    a = ap.parse_args(argv)
    for d in (C.BENCH_DIR, C.CACHE_DIR, C.MODELS_DIR, C.ARTIFACTS / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    LOG = C.ARTIFACTS / "logs" / "benchmark.log"
    LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    ev = data.load_events()
    ds_list = a.datasets or data.dataset_ids()
    assets = sorted({int(ev.loc[d, "asset"]) for d in ds_list})
    log(f"datasets {ds_list}  assets {assets}  dataset v{C.DATASET_VERSION} ({C.DATASET_DOI})")

    # stage 1
    for ds, dt in Parallel(n_jobs=a.jobs)(delayed(prepare_dataset)(d, a.rebuild, a.rescore) for d in ds_list):
        log(f"  bundle+segment {ds}: {dt:.0f}s")
    log(f"stage 1 done ({time.time() - t0:.0f}s)")
    # baselines
    for ds in Parallel(n_jobs=a.jobs)(delayed(lambda d: (baselines.unsupervised_scores(d), d)[1])(d) for d in ds_list):
        pass
    log(f"IF/AE scores done ({time.time() - t0:.0f}s)")
    if not (C.MODELS_DIR / "power_model.joblib").is_file():
        from . import forecast
        forecast.run(log)
    rf = None if a.skip_rf else baselines.legacy_rf(log)

    results, per_method, fold_info = [], {}, {}
    segs = {d: load_seg(d) for d in ds_list}

    # ---- legacy baselines (fixed, no tuning here)
    for mid in ("legacy_rf", "legacy_residual"):
        if mid == "legacy_rf" and rf is None:
            continue
        rows = []
        for d in ds_list:
            ip = segs[d]["is_pred"]
            times = segs[d]["times"][ip]
            if mid == "legacy_residual":
                point, alarm, comp = baselines.legacy_residual(d, times)
            else:
                point, alarm, comp = baselines.rf_outputs(rf, d, times)
            r = dataset_row(d, ev, times, point, alarm)
            i = first_alarm_index(r, alarm)
            if i is not None:
                r.update(diag_fields(comp[i] if comp[i] is not None else C.UNKNOWN, r["label"], r["description"]))
            rows.append(r)
        per_method[mid] = rows
    # ---- IF / AE: threshold quantile chosen leave-one-turbine-out
    for mid, key in (("iforest", "if"), ("autoencoder", "ae")):
        sc = {d: baselines.unsupervised_scores(d) for d in ds_list}
        by_q = []
        for q in baselines.QUANTILES:
            rq = {}
            for d in ds_list:
                point, alarm = baselines.threshold_outputs(sc[d][f"{key}_val"], sc[d][f"{key}_seg"], sc[d]["is_pred"], q)
                times = sc[d]["times"][sc[d]["is_pred"]]
                rq[d] = dataset_row(d, ev, times, point, alarm, diag_fields(C.UNKNOWN, ev.loc[d, "event_label"], ""))
            by_q.append(rq)
        rows, folds = [], {}
        for a_out in assets:
            ci = select(by_q, ds_list, set(assets) - {a_out}, baselines.QUANTILES)
            folds[a_out] = baselines.QUANTILES[ci]
            for d in ds_list:
                if int(ev.loc[d, "asset"]) == a_out:
                    rows.append(by_q[ci][d] | {"params": {"q": baselines.QUANTILES[ci]}})
        per_method[mid], fold_info[mid] = rows, {str(k): {"q": v} for k, v in folds.items()}
    log(f"baselines done ({time.time() - t0:.0f}s)")

    # ---- proposed: grid on every dataset once, then LOTO selection
    sel_params = {}
    for mid in ("nbm_threshold", "nbm_ewma", "nbm_cusum", "nbm_ewma_cusum_free", "nbm_ewma_cusum"):
        combos = grid(mid)
        out = Parallel(n_jobs=a.jobs)(delayed(eval_grid_dataset)(d, mid, combos, ev) for d in ds_list)
        by_combo = [{} for _ in combos]
        for d, res in out:
            for ci, r in enumerate(res):
                by_combo[ci][d] = r
        rows, folds = [], {}
        for a_out in assets:
            ci = select(by_combo, ds_list, set(assets) - {a_out}, combos)
            folds[a_out] = combos[ci]
            for d in ds_list:
                if int(ev.loc[d, "asset"]) == a_out:
                    r = proposed_eval(segs[d], ev, d, combos[ci], with_diag=True)
                    r["params"] = combos[ci].to_dict()
                    rows.append(r)
        gi = select(by_combo, ds_list, set(assets), combos)
        sel_params[mid] = {"folds": folds, "global": combos[gi]}
        per_method[mid] = sorted(rows, key=lambda r: r["id"])
        fold_info[mid] = {str(k): v.to_dict() for k, v in folds.items()} | {"global(all assets)": combos[gi].to_dict()}
        log(f"{mid}: {len(combos)} combos, LOTO choices {[(k, v.to_dict()) for k, v in folds.items()]}")

    # ---- write bundle params for the selected method
    for d in ds_list:
        b = monitor.load_bundle(d)
        b.params = sel_params[SELECTED]["folds"][b.asset].to_dict()
        b.params_global = sel_params[SELECTED]["global"].to_dict()
        monitor.save_bundle(b)

    # ---- outputs
    methods = []
    notes = {
        "legacy_rf": "supervised; trained on other turbine-periods' event labels (leave-one-group-out); running rows only",
        "legacy_residual": "saved v2 scores; its threshold/alarm settings were chosen on the same 22 datasets (optimistic)",
        "iforest": "threshold quantile chosen leave-one-turbine-out; alarm = CARE criticality >= 72; no diagnosis",
        "autoencoder": "threshold quantile chosen leave-one-turbine-out; alarm = CARE criticality >= 72; no diagnosis",
    }
    for mid, rows in per_method.items():
        rows = sorted(rows, key=lambda r: r["id"])
        mr = method_result(mid, rows, notes.get(mid, "alarm params chosen leave-one-turbine-out (pooled held-out)"))
        methods.append(mr)
        doc = {"method": mid, "name": METHODS[mid][0], "per_dataset": dataset_results(rows),
               "confusion": care.confusion(rows), "component_metrics": care.component_metrics(rows),
               "fold_params": fold_info.get(mid), "summary": mr}
        (C.BENCH_DIR / f"{mid}.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    order = list(METHODS)
    methods.sort(key=lambda m: order.index(m["id"]))
    bench = {"protocol": protocol_text(assets), "methods": methods, "selected": SELECTED, "baseline": "legacy_residual",
             "acceptance": acceptance(methods), "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
             "dataset": {"record": C.DATASET_RECORD, "doi": C.DATASET_DOI, "version": C.DATASET_VERSION},
             "pipeline_version": C.PIPELINE_VERSION, "fold_params": fold_info}
    (C.BENCH_DIR / "benchmark.json").write_text(json.dumps(bench, indent=1, default=str), encoding="utf-8")
    from .report import write_report
    write_report(bench, per_method)
    log(f"benchmark done ({time.time() - t0:.0f}s)")
    for m in methods:
        log(f"  {m['id']:<16} CARE {m['care']:.3f}  cov {m['coverage']:.3f} acc {m['accuracy']:.3f} rel {m['reliability']:.3f} "
            f"earl {m['earliness']:.3f}  events {m['events_detected']}/{m['events_total']}  FA {m['false_alarm_events']} "
            f"({m['normal_datasets_flagged']}/{m['normal_datasets']} normal flagged)  delay med {m['delay_median_h']} p90 {m['delay_p90_h']}"
            f"  comp {m['component_correct']} unk {m['component_unknown']}")
    return bench


def acceptance(methods):
    by = {m["id"]: m for m in methods}
    if "legacy_residual" not in by or SELECTED not in by:
        return None
    ref, sel = by["legacy_residual"], by[SELECTED]
    fa_red = (1 - sel["false_alarm_events"] / ref["false_alarm_events"]) if ref["false_alarm_events"] else None
    ok_fa = fa_red is not None and fa_red >= 0.25
    ok_rec = sel["events_detected"] >= ref["events_detected"]
    return {"target": ">= 25 % fewer false-alarm events than the legacy residual detector at >= equal event recall",
            "reference": "legacy_residual", "false_alarm_events": [ref["false_alarm_events"], sel["false_alarm_events"]],
            "fa_reduction": None if fa_red is None else round(fa_red, 3),
            "event_recall": [f"{ref['events_detected']}/{ref['events_total']}", f"{sel['events_detected']}/{sel['events_total']}"],
            "met": bool(ok_fa and ok_rec)}


def protocol_text(assets):
    return {
        "dataset": f"CARE to Compare v{C.DATASET_VERSION}, Wind Farm A, 22 datasets (12 anomaly / 10 normal), "
                   f"record {C.DATASET_RECORD}, DOI {C.DATASET_DOI}",
        "training": "per dataset: NBMs fitted on train rows with status 0/2 only, chronological first 80 %; "
                    "regime sigma/median and EWMA scales from the last 20 % (validation). Prediction rows never used "
                    "for fitting. Status/event labels are never model inputs.",
        "hyperparameters": f"alarm parameters (EWMA lambda/L, CUSUM k/h, persistence, cooldown) and the IF/AE "
                           f"threshold quantiles chosen leave-one-turbine-out over assets {assets}: maximise CARE on "
                           f"the datasets of 4 assets, apply to the 5th; reported metrics pool the 5 held-out folds.",
        "monitoring": "causal; statistics start 2 days before the prediction window (context from the end of the "
                      "train period) and freeze on bad-quality / not-judged rows.",
        "care": "official EnergyFaultDetector CAREScore defaults: beta 0.5, weights cov 1/acc 2/earl 1/rel 1, "
                "criticality >= 72 computed up to event_end, earliness decay from 1/4 of the event "
                "(paper text says 1/2), Farm A ignore_normal_index=True. Point prediction = alarm state "
                "(warning/fault) for the NBM methods.",
        "operational": "event detected = method's alarm active inside [event_start, event_end]; delay from logged "
                       "event_start (Farm A starts are derived from the logbook: treat delays as approximate); "
                       "false-alarm events = alarm episodes in normal datasets' prediction windows; per turbine-month "
                       "= / (normal prediction days / 30.44).",
        "diagnosis": "component at first alarm; correct if it matches the logged fault component "
                     "(hydraulic group also accepts Hub / pitch); 'Unknown' counted separately; confidence not "
                     "calibrated (null).",
        "acceptance": ">= 25 % fewer false-alarm events than the legacy residual detector at >= equal event recall",
    }


if __name__ == "__main__":
    main()
