"""In-depth investigation of missed gearbox faults, autoencoder overlap, and sensitivity sweep.

Produces event-level timelines, feature analysis, and detector fusion tables.
"""
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd

from windtwin import config as C, data, monitor, care


def load_replay(ds_id: int):
    path = C.ROOT / "artifacts" / "replay" / f"{ds_id}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def analyze_gearbox_events():
    gearbox_ids = [10, 51, 72]
    reports = {}

    for ds_id in gearbox_ids:
        rep = load_replay(ds_id)
        t = np.array(rep["t"])
        ev_start = rep.get("event_start")
        ev_end = rep.get("event_end")

        s11 = rep["signals"].get("sensor_11_avg", {})
        s12 = rep["signals"].get("sensor_12_avg", {})
        power = np.array(rep["power"])
        expected_p = np.array(rep["expected"])
        wind = np.array(rep["wind"])
        ambient = np.array(rep["ambient"])
        states = np.array(rep["state"])
        op_states = np.array(rep["operating_state"])

        obs11 = np.array(s11.get("observed", []))
        exp11 = np.array(s11.get("expected", []))
        obs12 = np.array(s12.get("observed", []))
        exp12 = np.array(s12.get("expected", []))

        # Event mask
        in_ev = (t >= ev_start) & (t <= ev_end)
        pre_ev = t < ev_start

        # Residuals inside event window
        r11_ev = (obs11 - exp11)[in_ev] if len(obs11) == len(t) else np.array([])
        r12_ev = (obs12 - exp12)[in_ev] if len(obs12) == len(t) else np.array([])

        # Component scores
        comps = rep["components"]
        gb_comp = comps.get("Gearbox", {})
        gb_ewma = np.array(gb_comp.get("ewma", []))
        gb_cusum = np.array(gb_comp.get("cusum", []))

        # Check if alarm fired pre-event
        pre_alarms = np.sum(np.isin(states[pre_ev], ["warning", "fault"]))
        ev_alarms = np.sum(np.isin(states[in_ev], ["warning", "fault"]))

        # Non-production rows in event
        stopped_in_ev = np.mean(op_states[in_ev] != "production") if in_ev.sum() > 0 else 0.0

        reports[ds_id] = {
            "dataset": ds_id,
            "asset": rep["asset"],
            "description": rep["description"],
            "event_start": ev_start,
            "event_end": ev_end,
            "duration_days": round(in_ev.sum() * 10 / 60 / 24, 2),
            "mean_power_in_ev": round(float(np.nanmean(power[in_ev])), 3) if in_ev.sum() > 0 else None,
            "stopped_fraction_in_ev": round(float(stopped_in_ev), 3),
            "s11_bearing_residual_mean": round(float(np.nanmean(r11_ev)), 2) if len(r11_ev) else None,
            "s11_bearing_residual_max": round(float(np.nanmax(r11_ev)), 2) if len(r11_ev) else None,
            "s12_oil_residual_mean": round(float(np.nanmean(r12_ev)), 2) if len(r12_ev) else None,
            "s12_oil_residual_max": round(float(np.nanmax(r12_ev)), 2) if len(r12_ev) else None,
            "max_gearbox_ewma_in_ev": round(float(np.nanmax(gb_ewma[in_ev])), 2) if len(gb_ewma) and in_ev.sum() > 0 else None,
            "max_gearbox_cusum_in_ev": round(float(np.nanmax(gb_cusum[in_ev])), 2) if len(gb_cusum) and in_ev.sum() > 0 else None,
            "pre_event_alarm_steps": int(pre_alarms),
            "event_alarm_steps": int(ev_alarms),
            "detected": bool(ev_alarms > 0),
        }

    return reports


def analyze_detector_overlap():
    art = C.ROOT / "artifacts" / "benchmark"
    with open(art / "nbm_ewma_cusum.json", "r", encoding="utf-8") as f:
        nbm = json.load(f)
    with open(art / "autoencoder.json", "r", encoding="utf-8") as f:
        ae = json.load(f)
    with open(art / "legacy_residual.json", "r", encoding="utf-8") as f:
        leg_res = json.load(f)
    with open(art / "legacy_rf.json", "r", encoding="utf-8") as f:
        leg_rf = json.load(f)

    # Build per-dataset overlap map
    nbm_res = {d["id"]: d for d in nbm["per_dataset"]}
    ae_res = {d["id"]: d for d in ae["per_dataset"]}
    leg_res_map = {d["id"]: d for d in leg_res["per_dataset"]}
    leg_rf_map = {d["id"]: d for d in leg_rf["per_dataset"]}

    all_ids = sorted(nbm_res.keys())
    rows = []

    union_detected = 0
    union_fa = 0

    for ds_id in all_ids:
        n = nbm_res[ds_id]
        a = ae_res[ds_id]
        lr = leg_res_map[ds_id]
        lrf = leg_rf_map[ds_id]

        label = n["label"]
        desc = n.get("description", "")
        asset = n["asset"]

        n_det = n.get("detected", False)
        a_det = a.get("detected", False)
        lr_det = lr.get("detected", False)
        lrf_det = lrf.get("detected", False)

        n_fa = n.get("false_alarms", 0)
        a_fa = a.get("false_alarms", 0)
        lr_fa = lr.get("false_alarms", 0)
        lrf_fa = lrf.get("false_alarms", 0)

        # Union detection
        u_det = n_det or a_det
        # Union FA on normal dataset: any method alarming
        u_fa = n_fa + a_fa if label == "normal" else 0

        if label == "anomaly" and u_det:
            union_detected += 1
        if label == "normal":
            union_fa += (1 if (n_fa > 0 or a_fa > 0) else 0)

        rows.append({
            "id": ds_id,
            "asset": asset,
            "label": label,
            "description": desc,
            "nbm_detected": n_det,
            "nbm_fa": n_fa,
            "ae_detected": a_det,
            "ae_fa": a_fa,
            "legacy_res_detected": lr_det,
            "legacy_res_fa": lr_fa,
            "legacy_rf_detected": lrf_det,
            "legacy_rf_fa": lrf_fa,
            "union_detected": u_det,
        })

    summary = {
        "nbm": {"detected": sum(1 for r in rows if r["label"] == "anomaly" and r["nbm_detected"]),
                "fa_events": sum(r["nbm_fa"] for r in rows if r["label"] == "normal")},
        "ae": {"detected": sum(1 for r in rows if r["label"] == "anomaly" and r["ae_detected"]),
               "fa_events": sum(r["ae_fa"] for r in rows if r["label"] == "normal")},
        "legacy_res": {"detected": sum(1 for r in rows if r["label"] == "anomaly" and r["legacy_res_detected"]),
                       "fa_events": sum(r["legacy_res_fa"] for r in rows if r["label"] == "normal")},
        "legacy_rf": {"detected": sum(1 for r in rows if r["label"] == "anomaly" and r["legacy_rf_detected"]),
                      "fa_events": sum(r["legacy_rf_fa"] for r in rows if r["label"] == "normal")},
        "union_nbm_ae": {"detected": union_detected, "fa_events": union_fa},
    }

    return rows, summary


if __name__ == "__main__":
    gb_reports = analyze_gearbox_events()
    print("=== GEARBOX EVENT INVESTIGATION ===")
    print(json.dumps(gb_reports, indent=2))

    overlap_rows, summary = analyze_detector_overlap()
    print("\n=== DETECTOR OVERLAP SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\n=== PER-DATASET OVERLAP DETAILS ===")
    df = pd.DataFrame(overlap_rows)
    print(df[["id", "asset", "label", "description", "nbm_detected", "ae_detected", "legacy_res_detected", "union_detected"]].to_string())
