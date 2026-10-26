# WindTwin benchmark report

Dataset: CARE to Compare v6 Wind Farm A (record 15846963, DOI 10.5281/zenodo.15846963); pipeline windtwin-2.0.0; generated 2026-10-26T09:25:51.

All methods are scored by the same `windtwin/care.py` on the same 22 prediction windows (12 anomaly events, 10 normal datasets). Hyper-parameters of the tunable methods were chosen leave-one-turbine-out; numbers below pool the 5 held-out folds.

## Main results

| Method | kind | CARE | Coverage | Accuracy | Reliability | Earliness | Events detected | FA events (normal) | FA / turbine-month | Normal datasets flagged | Pre-event alarms | Delay median / p90 (h) | Diagnosis correct / unknown (of detected) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Legacy Random Forest classifier (re-run LOGO CV) | baseline | 0.614 | 0.340 | 0.935 | 0.673 | 0.187 | 8/12 | 27 | 4.54 | 5/10 | 4 | 20.6 / 82.5 | 1 / 0 (of 8) |
| Legacy residual detector v2 (saved scores) | baseline | 0.576 | 0.256 | 0.951 | 0.625 | 0.097 | 9/12 | 14 | 2.35 | 7/10 | 1 | 94.2 / 241.6 | 3 / 0 (of 9) |
| Isolation Forest on Avg signals | baseline | 0.520 | 0.295 | 0.911 | 0.357 | 0.127 | 2/12 | 2 | 0.34 | 2/10 | 0 | 163.8 / 290.5 | 0 / 2 (of 2) |
| MLP autoencoder on Avg signals | baseline | 0.644 | 0.414 | 0.925 | 0.781 | 0.174 | 5/12 | 0 | 0.00 | 0/10 | 1 | 28.2 / 90.5 | 0 / 5 (of 5) |
| NBM + regime-normalized residual + threshold | proposed | 0.584 | 0.448 | 0.859 | 0.568 | 0.185 | 12/12 | 1018 | 171.20 | 10/10 | 122 | 2.1 / 8.0 | 0 / 4 (of 12) |
| NBM + normalized residual + EWMA | proposed | 0.549 | 0.493 | 0.754 | 0.486 | 0.257 | 12/12 | 280 | 47.09 | 10/10 | 40 | 7.8 / 76.4 | 0 / 1 (of 12) |
| NBM + normalized residual + CUSUM | proposed | 0.588 | 0.309 | 0.923 | 0.577 | 0.205 | 7/12 | 14 | 2.35 | 6/10 | 1 | 102.0 / 275.2 | 1 / 2 (of 7) |
| NBM + EWMA + CUSUM, persistence/cooldown also tuned (iteration 2) | proposed | 0.582 | 0.349 | 0.823 | 0.667 | 0.249 | 7/12 | 5 | 0.84 | 4/10 | 2 | 104.2 / 218.7 | 1 / 2 (of 7) |
| NBM + EWMA + CUSUM + persistence/hysteresis (6 h / 24 h) **(selected)** | proposed | 0.637 | 0.380 | 0.896 | 0.750 | 0.261 | 8/12 | 3 | 0.51 | 3/10 | 2 | 106.0 / 302.2 | 2 / 1 (of 8) |

Normal prediction data: 5.95 turbine-months. CARE-detected events (criticality >= 72 on point predictions) per method: legacy_rf 7/12 anomaly + 3 normal, legacy_residual 6/12 anomaly + 3 normal, iforest 2/12 anomaly + 2 normal, autoencoder 5/12 anomaly + 0 normal, nbm_threshold 5/12 anomaly + 3 normal, nbm_ewma 7/12 anomaly + 8 normal, nbm_cusum 6/12 anomaly + 4 normal, nbm_ewma_cusum_free 8/12 anomaly + 4 normal, nbm_ewma_cusum 9/12 anomaly + 3 normal

## Acceptance target

Target: >= 25 % fewer false-alarm events than the legacy residual detector at >= equal event recall.  Legacy residual detector: 14 FA events, recall 9/12; selected `nbm_ewma_cusum`: 3 FA events, recall 8/12 (FA reduction 0.786).  **Target NOT MET.**

## Selected method per dataset (held-out)

| Dataset | asset | label | logged fault | detected | delay h | lead to end h | FA episodes | pre-event | diagnosis | correct | CARE F0.5 / Acc / WS | max crit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | anomaly | Generator bearing failure | True | 297.3 | 38.7 | 0 | 0 | Generator bearings | True | 0.229 / – / 0.014 | 233 |
| 3 | 10 | normal |  |  | – | – | 0 | 0 |  |  | – / 1.000 / – | 0 |
| 10 | 10 | anomaly | Gearbox failure | False | – | – | 0 | 1 |  |  | 0.000 / – / 0.000 | 222 |
| 13 | 21 | normal |  |  | – | – | 0 | 0 |  |  | – / 1.000 / – | 0 |
| 14 | 13 | normal |  |  | – | – | 0 | 0 |  |  | – / 1.000 / – | 0 |
| 17 | 10 | normal |  |  | – | – | 1 | 0 | Hydraulic group |  | – / 0.465 / – | 912 |
| 22 | 21 | anomaly | Hydraulic group | True | 12.8 | 155.3 | 0 | 0 | Hydraulic group | True | 0.877 / – / 0.877 | 928 |
| 24 | 0 | normal |  |  | – | – | 1 | 0 | Generator bearings |  | – / 0.709 / – | 536 |
| 25 | 11 | normal |  |  | – | – | 0 | 0 |  |  | – / 1.000 / – | 0 |
| 26 | 0 | anomaly | Hydraulic group | False | – | – | 0 | 0 |  |  | 0.000 / – / 0.000 | 0 |
| 38 | 13 | normal |  |  | – | – | 1 | 0 | Converter |  | – / 0.790 / – | 533 |
| 40 | 10 | anomaly | Generator bearing failure | True | 155.5 | 601.5 | 0 | 0 | Nacelle cabinets | False | 0.536 / – / 0.154 | 593 |
| 42 | 10 | anomaly | Hydraulic group | True | 104.2 | 63.8 | 0 | 0 | Generator bearings | False | 0.454 / – / 0.155 | 384 |
| 45 | 13 | anomaly | Hydraulic group | True | 22.7 | 145.3 | 0 | 0 | Unknown — insufficient evidence | False | 0.970 / – / 0.784 | 872 |
| 51 | 21 | anomaly | Gearbox bearings damaged | False | – | – | 0 | 0 |  |  | 0.000 / – / 0.000 | 0 |
| 68 | 11 | anomaly | Transformer failure | True | 313.5 | 22.3 | 0 | 0 | Hydraulic group | False | 0.183 / – / 0.005 | 135 |
| 69 | 11 | normal |  |  | – | – | 0 | 0 |  |  | – / 1.000 / – | 0 |
| 71 | 0 | normal |  |  | – | – | 0 | 0 |  |  | – / 1.000 / – | 0 |
| 72 | 21 | anomaly | Gearbox failure | False | – | – | 0 | 0 |  |  | 0.000 / – / 0.000 | 0 |
| 73 | 0 | anomaly | Hydraulic group | True | 0.0 | 168.0 | 0 | 1 | Nacelle cabinets | False | 0.714 / – / 1.000 | 1000 |
| 84 | 13 | anomaly | Hydraulic group | True | 107.8 | 60.2 | 0 | 0 | Converter | False | 0.594 / – / 0.136 | 358 |
| 92 | 11 | normal |  |  | – | – | 0 | 0 |  |  | – / 1.000 / – | 0 |

## Event-level confusion (selected): true fault class vs diagnosis at first alarm

| true \ predicted | Gearbox | Generator bearings | Transformer | Hydraulic group | Converter | Nacelle cabinets | normal | Unknown | missed |
|---|---|---|---|---|---|---|---|---|---|
| Gearbox | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| Generator bearings | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| Transformer | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Hydraulic group | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 1 |
| Converter | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Nacelle cabinets | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| normal | 0 | 1 | 0 | 1 | 1 | 0 | 7 | 0 | 0 |
| Unknown | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| missed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Component precision / recall (selected; recall over all logged events of the class):

- Gearbox: 0/3 correct (recall 0.000), precision – over 0 diagnoses
- Generator bearings: 1/2 correct (recall 0.500), precision 0.500 over 2 diagnoses
- Hydraulic group: 1/6 correct (recall 0.167), precision 0.500 over 2 diagnoses
- Transformer: 0/1 correct (recall 0.000), precision – over 0 diagnoses

## Leave-one-turbine-out parameter choices

- `iforest`: held-out 0: q=0.98; held-out 10: q=0.9; held-out 11: q=0.9; held-out 13: q=0.9; held-out 21: q=0.9
- `autoencoder`: held-out 0: q=0.9; held-out 10: q=0.9; held-out 11: q=0.9; held-out 13: q=0.9; held-out 21: q=0.9
- `nbm_threshold`: held-out 0: λ=1.0 L=1, persist=1 cooldown=1; held-out 10: λ=1.0 L=1, persist=1 cooldown=1; held-out 11: λ=1.0 L=1, persist=1 cooldown=1; held-out 13: λ=1.0 L=1, persist=1 cooldown=1; held-out 21: λ=1.0 L=1, persist=1 cooldown=1; held-out global(all assets): λ=1.0 L=1, persist=1 cooldown=1
- `nbm_ewma`: held-out 0: λ=0.1 L=1, persist=1 cooldown=1; held-out 10: λ=0.02 L=1, persist=1 cooldown=1; held-out 11: λ=0.2 L=1, persist=1 cooldown=1; held-out 13: λ=0.02 L=1, persist=1 cooldown=1; held-out 21: λ=0.2 L=1, persist=1 cooldown=1; held-out global(all assets): λ=0.2 L=1, persist=1 cooldown=1
- `nbm_cusum`: held-out 0: k=1.0 h=3, persist=1 cooldown=1; held-out 10: k=0.5 h=4, persist=1 cooldown=1; held-out 11: k=1.0 h=3, persist=1 cooldown=1; held-out 13: k=1.0 h=3, persist=1 cooldown=1; held-out 21: k=1.0 h=3, persist=1 cooldown=1; held-out global(all assets): k=1.0 h=3, persist=1 cooldown=1
- `nbm_ewma_cusum_free`: held-out 0: λ=0.1 L=2, k=2.0 h=4, persist=6 cooldown=144; held-out 10: λ=0.02 L=3, k=1.0 h=3, persist=36 cooldown=144; held-out 11: λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144; held-out 13: λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144; held-out 21: λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144; held-out global(all assets): λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144
- `nbm_ewma_cusum`: held-out 0: λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144; held-out 10: λ=0.02 L=3, k=1.0 h=3, persist=36 cooldown=144; held-out 11: λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144; held-out 13: λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144; held-out 21: λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144; held-out global(all assets): λ=0.1 L=2, k=1.0 h=3, persist=36 cooldown=144

## Protocol

- **dataset**: CARE to Compare v6, Wind Farm A, 22 datasets (12 anomaly / 10 normal), record 15846963, DOI 10.5281/zenodo.15846963
- **training**: per dataset: NBMs fitted on train rows with status 0/2 only, chronological first 80 %; regime sigma/median and EWMA scales from the last 20 % (validation). Prediction rows never used for fitting. Status/event labels are never model inputs.
- **hyperparameters**: alarm parameters (EWMA lambda/L, CUSUM k/h, persistence, cooldown) and the IF/AE threshold quantiles chosen leave-one-turbine-out over assets [0, 10, 11, 13, 21]: maximise CARE on the datasets of 4 assets, apply to the 5th; reported metrics pool the 5 held-out folds.
- **monitoring**: causal; statistics start 2 days before the prediction window (context from the end of the train period) and freeze on bad-quality / not-judged rows.
- **care**: official EnergyFaultDetector CAREScore defaults: beta 0.5, weights cov 1/acc 2/earl 1/rel 1, criticality >= 72 computed up to event_end, earliness decay from 1/4 of the event (paper text says 1/2), Farm A ignore_normal_index=True. Point prediction = alarm state (warning/fault) for the NBM methods.
- **operational**: event detected = method's alarm active inside [event_start, event_end]; delay from logged event_start (Farm A starts are derived from the logbook: treat delays as approximate); false-alarm events = alarm episodes in normal datasets' prediction windows; per turbine-month = / (normal prediction days / 30.44).
- **diagnosis**: component at first alarm; correct if it matches the logged fault component (hydraulic group also accepts Hub / pitch); 'Unknown' counted separately; confidence not calibrated (null).
- **acceptance**: >= 25 % fewer false-alarm events than the legacy residual detector at >= equal event recall

## Discussion

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

