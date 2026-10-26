"""REPORT.md writer for the benchmark (tables with counts + acceptance verdict + caveats)."""
import json

from . import config as C


def _f(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_report(bench, per_method):
    m = bench["methods"]
    acc = bench.get("acceptance") or {}
    sel = bench["selected"]
    L = [f"# WindTwin benchmark report", "",
         f"Dataset: CARE to Compare v{C.DATASET_VERSION} Wind Farm A (record {C.DATASET_RECORD}, DOI {C.DATASET_DOI}); "
         f"pipeline {C.PIPELINE_VERSION}; generated {bench['generated_at']}.", "",
         "All methods are scored by the same `windtwin/care.py` on the same 22 prediction windows "
         "(12 anomaly events, 10 normal datasets). Hyper-parameters of the tunable methods were chosen "
         "leave-one-turbine-out; numbers below pool the 5 held-out folds.", "",
         "## Main results", "",
         "| Method | kind | CARE | Coverage | Accuracy | Reliability | Earliness | Events detected | FA events (normal) | FA / turbine-month | Normal datasets flagged | Pre-event alarms | Delay median / p90 (h) | Diagnosis correct / unknown (of detected) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in m:
        star = " **(selected)**" if x["id"] == sel else ""
        L.append(f"| {x['name']}{star} | {x['kind']} | {_f(x['care'])} | {_f(x['coverage'])} | {_f(x['accuracy'])} | "
                 f"{_f(x['reliability'])} | {_f(x['earliness'])} | {x['events_detected']}/{x['events_total']} | "
                 f"{x['false_alarm_events']} | {_f(x['fa_per_turbine_month'], 2)} | {x['normal_datasets_flagged']}/{x['normal_datasets']} | "
                 f"{x['pre_event_alarms']} | {_f(x['delay_median_h'], 1)} / {_f(x['delay_p90_h'], 1)} | "
                 f"{x['component_correct']} / {x['component_unknown']} (of {x['events_detected']}) |")
    L += ["", f"Normal prediction data: {m[0]['normal_months']} turbine-months. CARE-detected events "
          "(criticality >= 72 on point predictions) per method: " +
          ", ".join(f"{x['id']} {x['care_events_detected']}/{x['events_total']} anomaly + {x['care_normal_flagged']} normal" for x in m), ""]
    L += ["## Acceptance target", ""]
    if acc:
        L.append(f"Target: {acc['target']}.  Legacy residual detector: {acc['false_alarm_events'][0]} FA events, "
                 f"recall {acc['event_recall'][0]}; selected `{sel}`: {acc['false_alarm_events'][1]} FA events, "
                 f"recall {acc['event_recall'][1]} (FA reduction {_f(acc['fa_reduction'])}).  "
                 f"**Target {'MET' if acc['met'] else 'NOT MET'}.**")
    L += ["", "## Selected method per dataset (held-out)", "",
          "| Dataset | asset | label | logged fault | detected | delay h | lead to end h | FA episodes | pre-event | diagnosis | correct | CARE F0.5 / Acc / WS | max crit |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    doc = json.loads((C.BENCH_DIR / f"{sel}.json").read_text(encoding="utf-8"))
    for r in doc["per_dataset"]:
        cp = r["care_parts"]
        L.append(f"| {r['id']} | {r['asset']} | {r['label']} | {r['description'] or ''} | {r['detected'] if r['label'] == 'anomaly' else ''} | "
                 f"{_f(r['delay_h'], 1)} | {_f(r['lead_to_end_h'], 1)} | {r['false_alarms']} | {r['pre_event_alarms']} | "
                 f"{r['diagnosis'] or ''} | {'' if r['diagnosis_correct'] is None else r['diagnosis_correct']} | "
                 f"{_f(cp['f_beta'])} / {_f(cp['accuracy'])} / {_f(cp['weighted_score'])} | {cp['max_criticality']} |")
    cf = doc["confusion"]
    L += ["", "## Event-level confusion (selected): true fault class vs diagnosis at first alarm", "",
          "| true \\ predicted | " + " | ".join(cf["labels"]) + " |", "|---" * (len(cf["labels"]) + 1) + "|"]
    for lab, row in zip(cf["labels"], cf["matrix"]):
        L.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    L += ["", "Component precision / recall (selected; recall over all logged events of the class):", ""]
    for c, v in doc["component_metrics"].items():
        L.append(f"- {c}: {v['correct']}/{v['events']} correct (recall {_f(v['recall'])}), precision {_f(v['precision'])} "
                 f"over {v['predicted']} diagnoses")
    L += ["", "## Leave-one-turbine-out parameter choices", ""]
    for mid, folds in (bench.get("fold_params") or {}).items():
        L.append(f"- `{mid}`: " + "; ".join(f"held-out {k}: {_short(v)}" for k, v in folds.items()))
    L += ["", "## Protocol", ""] + [f"- **{k}**: {v}" for k, v in bench["protocol"].items()]
    L += ["", "## Discussion", "", DISCUSSION, ""]
    (C.BENCH_DIR / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def _short(p):
    if not isinstance(p, dict):
        return str(p)
    if "q" in p:
        return f"q={p['q']}"
    keys = []
    if p.get("use_ewma"):
        keys.append(f"λ={p['lam']} L={p['L']}")
    if p.get("use_cusum"):
        keys.append(f"k={p['k']} h={p['h']}")
    keys.append(f"persist={p['persist']} cooldown={p['cooldown']}")
    return ", ".join(keys)


# Written after the final run; numbers refer to artifacts/benchmark/benchmark.json of that run.
DISCUSSION = """\
**Headline.** The selected detector (regime-normalized NBM residuals + EWMA + CUSUM with 6 h persistence /
24 h hysteresis) reaches CARE 0.637 on held-out turbines, vs 0.576 for the legacy residual detector and 0.614
for the legacy Random Forest. It raises 3 false-alarm events on the 10 normal datasets (legacy residual: 14,
-79 %) but detects 8/12 logged faults vs 9/12 for the legacy residual detector. **The pre-registered
acceptance target (>= 25 % fewer false alarms at >= equal event recall) is therefore NOT met**: the false-alarm
reduction is achieved at the cost of one more missed fault. The small MLP autoencoder has the highest CARE
(0.644, zero false alarms) but detects only 5/12 events and cannot name a component.

**What drives the numbers (counts, not just rates).**
- Missed events: all three gearbox failures (datasets 10, 72, 51 "gearbox bearings damaged") and one hydraulic
  event (26). No gearbox temperature residual left its validation range inside these windows. On 10 an alarm fired
  and cleared *before* the logged window (counted as pre-event alarm, not detection). The legacy detector caught 10
  through a nacelle-cabinet temperature, not through the gearbox.
- False alarms: 3/10 normal datasets. Dataset 17 shows a large sustained hydraulic-oil residual (~18x its
  validation level) that no setting can ignore without also missing the hydraulic faults; 24 a generator-bearing
  and 38 a converter residual. Every method sensitive enough to detect >= 8 faults also alarms on 17. These may
  be unlogged deviations, but they are counted as false alarms.
- Delays are long (median 106 h, p90 302 h after the logged event start): several detections come late in the
  7-14-day windows. Farm A event starts are derived from the logbook (fault timestamp - ~14 days), so delays
  relative to the start are indicative only; lead time to the logged fault (event end) is 22-600 h (per dataset).
- Diagnosis is weak, as expected with 12 events: of 8 detections the component is correct 2x (generator bearing 0,
  hydraulic 22), 'Unknown' 1x (45), wrong 5x - hydraulic events often appear as nacelle-cabinet, converter or
  generator-bearing residuals. No component confidence is reported (not calibratable on 12 events).

**Trade-offs between the alternatives (all held-out, same scorer).**
- Simple threshold on z and EWMA alone detect every event (12/12) but produce hundreds of alarm episodes on
  normal data (1018 / 280) - operationally useless even though their CARE (0.58 / 0.55) looks acceptable,
  because CARE scores pointwise accuracy, not alarm chatter.
- CUSUM alone: 7/12 events, 14 false alarms. Combining EWMA + CUSUM + persistence + hysteresis is what removes
  the chatter.
- Letting the leave-one-turbine-out search also choose persistence/cooldown (`nbm_ewma_cusum_free`, 432 settings)
  overfits the 4 training turbines (fold choices differ; CARE 0.582, 7/12, 5 false alarms). Fixing persistence 6 h
  and cooldown 24 h (literature: CARE 12 h criticality, 24 h hysteresis) and tuning only lambda/L/k/h gives stable
  choices (4 of 5 folds agree) and the better held-out result.
- In-sample (all 22 datasets, NOT a valid estimate) the same family has settings with 9/12 events and 3 false
  alarms (CARE ~0.67). This bounds what better threshold selection alone could reach; closing the gap honestly
  needs more labelled events or stronger gearbox-sensitive features.

**Iteration log (each step re-evaluated with the same held-out protocol).**
1. NBM residuals; EWMA scaled by validation MAD; raw CUSUM; 216-setting grid -> CARE 0.496, 8/12, 23 false alarms
   (raw CUSUM on strongly autocorrelated temperature residuals - transformer, generator bearings - grew without
   bound on normal data).
2. EWMA and CUSUM expressed in units of their own 99.5th percentile on the validation split (label-free, absorbs
   autocorrelation); bounded CUSUM; diagnosis ranked by the statistic that fired -> CARE 0.582, 7/12, 5.
3. Persistence/cooldown fixed (6 h / 24 h), only detector limits tuned -> CARE 0.637, 8/12, 3 (selected).
Steps 2-3 were decided after looking at pooled held-out results of the previous step. That is a researcher
degree of freedom on a 22-dataset benchmark, so the selected method's held-out CARE is somewhat optimistic. The
legacy residual detector's settings were likewise chosen on the same 22 datasets (its documented sweep).

**Protocol caveats.** 12 anomaly / 10 normal datasets: one event = 8 percentage points of recall, so +-1 event
differences are within noise. CARE follows the official EnergyFaultDetector code (earliness decay from 1/4 of the
event; the paper text says 1/2; criticality up to event_end; Farm A status index ignored). Our values are Farm A
only on v6 data and are not comparable to the paper's 0.66 (pooled over 95 datasets). The legacy RF is supervised
(trained on other turbine-periods' event labels, leave-one-group-out, running rows only) and keeps its own alarm
rule (>= 3 h of a fault class). IsolationForest / autoencoder have no component diagnosis (counted 'Unknown').
"""
