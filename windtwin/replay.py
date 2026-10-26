"""Replay CLI: run the selected method causally over every dataset's prediction window (+2 days context)
with the streaming Monitor and write artifacts/replay/<id>.json.gz (ReplayFile) + artifacts/datasets.json.

    python -m windtwin.replay [--jobs 8] [--datasets ...]

Each dataset uses the alarm parameters chosen with its own turbine held out (bundle.params), so the replay
reproduces the held-out benchmark result for that dataset (checked and logged)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import gzip
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C, data, monitor, care

warnings.filterwarnings("ignore")


def replay_dataset(ds, ev):
    t0 = time.time()
    b = monitor.load_bundle(ds)
    raw, mi, pi = monitor.segment(ds)
    m = monitor.Monitor(b, source="replay")
    m.prime(raw.iloc[:mi])
    rows, _ = m.update(raw.iloc[mi:])
    e = ev.loc[ds]
    comps = {c["name"]: {"score": [], "ewma": [], "cusum": [], "state": []} for c in rows[0]["components"]}
    for r in rows:
        for c in r["components"]:
            d = comps[c["name"]]
            d["score"].append(c["score"]), d["ewma"].append(c["ewma"]), d["cusum"].append(c["cusum"]), d["state"].append(c["state"])
    sig = {t: {"label": C.label_of(t), "unit": C.unit_of(t), "component": C.TARGET_COMPONENT.get(t),
               "observed": [r["signals"][t][0] for r in rows], "expected": [r["signals"][t][1] for r in rows],
               "lo": [r["signals"][t][2] for r in rows], "hi": [r["signals"][t][3] for r in rows]} for t in b.targets}
    p = m.p
    t = [r["t"] for r in rows]
    alarm = np.array([r["alarm"] for r in rows])
    is_pred = (raw["train_test"].iloc[mi:] != "train").to_numpy()
    op = care.operational(np.array(t)[is_pred], alarm[is_pred], e["event_start"], e["event_end"], e["event_label"])
    out = {"id": int(ds), "asset": int(e["asset"]), "label": e["event_label"], "description": e["event_description"],
           "event_start": e["event_start"].isoformat(), "event_end": e["event_end"].isoformat(),
           "prediction_start": raw["time_stamp"].iloc[pi].isoformat(), "pipeline_version": C.PIPELINE_VERSION,
           "source": "replay", "t": t,
           "power": [r["power"] for r in rows], "expected": [r["expected"] for r in rows],
           "expected_lo": [r["expected_lo"] for r in rows], "expected_hi": [r["expected_hi"] for r in rows],
           "wind": [r["wind"] for r in rows], "ambient": [r["ambient"] for r in rows],
           "pitch": [r["pitch"] for r in rows], "rotor_rpm": [r["rotor_rpm"] for r in rows],
           "wind_dir": [r["wind_dir"] for r in rows], "nacelle_dir": [r["nacelle_dir"] for r in rows],
           "yaw_error": [r["yaw_error"] for r in rows],
           "state": [r["state"] for r in rows], "health": [r["health"] for r in rows],
           "operating_state": [r["operating_state"] for r in rows], "alarm": alarm.tolist(),
           "top_component": [r["top_component"] for r in rows],
           "components": comps, "signals": sig, "quality": [r["quality"]["state"] for r in rows],
           "quality_issues": {r["t"]: r["quality"]["issues"] for r in rows if r["quality"]["issues"]},
           "cases": m.cases,
           "limits": {"ewma_warn": p.L if p.use_ewma else None, "ewma_watch": p.watch * p.L if p.use_ewma else None,
                      "ewma_clear": p.clear * p.L if p.use_ewma else None, "cusum_h": p.h if p.use_cusum else None,
                      "cusum_k": p.k if p.use_cusum else None, "persist_steps": p.persist, "cooldown_steps": p.cooldown,
                      "lambda": p.lam, "z_band": 1.645},
           "params": p.to_dict(),
           "outcome": {"detected": op["detected"], "first_alarm": op["first_alarm"].isoformat() if op["first_alarm"] is not None else None,
                       "delay_h": op["delay_h"], "false_alarms": op["false_alarms"], "pre_event_alarms": op["pre_event_alarms"]}}
    C.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    path = C.REPLAY_DIR / f"{ds}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), default=str)
    return ds, time.time() - t0, path.stat().st_size, op


def datasets_json(ev, ids):
    out = []
    for ds in ids:
        raw = data.load_dataset(ds)
        pred = raw[raw["train_test"] != "train"]
        e = ev.loc[ds]
        out.append({"id": int(ds), "asset": int(e["asset"]), "label": e["event_label"], "description": e["event_description"],
                    "event_start": e["event_start"].isoformat(), "event_end": e["event_end"].isoformat(),
                    "prediction_start": pred["time_stamp"].iloc[0].isoformat(),
                    "prediction_end": pred["time_stamp"].iloc[-1].isoformat(), "n_rows": int(len(pred)),
                    "n_train_rows": int(len(raw) - len(pred)),
                    "simulation_asset": next((a for a, d in C.SIM_DATASETS.items() if d == ds), None)})
    (C.ARTIFACTS / "datasets.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--datasets", type=int, nargs="*")
    a = ap.parse_args(argv)
    ev = data.load_events()
    ids = a.datasets or data.dataset_ids()
    t0 = time.time()
    datasets_json(ev, data.dataset_ids())
    bench = {}
    sel = C.BENCH_DIR / "nbm_ewma_cusum.json"
    if sel.is_file():
        bench = {r["id"]: r for r in json.loads(sel.read_text(encoding="utf-8"))["per_dataset"]}
    mism = 0
    for ds, dt, size, op in Parallel(n_jobs=a.jobs)(delayed(replay_dataset)(d, ev) for d in ids):
        ref = bench.get(ds)
        same = ref is None or (ref["false_alarms"] == op["false_alarms"] and
                               (ref["label"] != "anomaly" or ref["detected"] == op["detected"]))
        mism += not same
        print(f"replay {ds:>3}: {size / 1e6:.2f} MB  {dt:.0f}s  detected={op['detected']} FA={op['false_alarms']}"
              f"{'' if same else '  MISMATCH vs benchmark'}", flush=True)
    print(f"replay done ({time.time() - t0:.0f}s), mismatches vs benchmark: {mism}")


if __name__ == "__main__":
    main()
