# WindTwin — architecture and contracts (v2)

Direction source: `correct_method.txt` (research review). Core idea: **reliable, early, explainable fault
alarms** from a regime-aware normal-behaviour model (NBM) + normalized residuals + temporal alarm logic
(EWMA / CUSUM / persistence + hysteresis) + evidence-based diagnosis with an explicit *unknown* outcome,
evaluated on CARE to Compare v6 (Wind Farm A) with CARE score + operational metrics, and demonstrated in a
digital twin that keeps **historical replay (real data)** and **synthetic simulation** strictly separate.
Nothing is labelled "live telemetry".

Python: `C:/Users/Abcom/AppData/Local/Programs/Python/Python313/python.exe` (3.13; sklearn 1.9, pandas 3.1,
numpy 2.5, fastapi, uvicorn, orjson, pyarrow). No Node — the frontend is a no-build ES-module SPA.

## Layout

```
data/CARE_To_Compare/Wind Farm A/      raw v6 data (fetched by data/fetch_care.py; record 15846963)
windtwin/                               core library (pure Python, no web code)
  config.py      signal map, subsystems, paths, versions (DATASET_DOI, PIPELINE_VERSION)
  data.py        load a dataset, split train/val/prediction, operating state, feature engineering
  quality.py     data-quality flags: missing, flatline/stuck, out-of-range, implausible, pitch wrap
  nbm.py         normal-behaviour models (one per target signal), leakage-safe inputs, robust fit
  residuals.py   regime-aware residual normalization (sigma per operating-regime bin, fit on validation)
  alarms.py      EWMA, CUSUM, persistence/hysteresis/cooldown state machine; streaming + batch
  diagnosis.py   component ranking from residual evidence, calibrated confidence, 'unknown'
  care.py        CARE score (coverage, accuracy, reliability, earliness) + operational metrics
  baselines.py   wrappers for legacy RF, legacy residual detector, IsolationForest, autoencoder
  forecast.py    power forecast + prediction intervals (split-conformal, per wind-speed regime)
  scenarios.py   reproducible synthetic scenario library (seeded; severity, onset, duration)
  monitor.py     Monitor: streaming facade used by the server (replay + simulation)
  benchmark.py   CLI: run all methods under identical protocols -> artifacts/benchmark/
  replay.py      CLI: precompute replay files for every dataset -> artifacts/replay/
artifacts/        generated: models/, replay/<id>.json.gz, benchmark/*.json, forecast/*.json
server/           FastAPI app (main.py) + simulation engine (sim.py) that reuses windtwin
web/              SPA: index.html, app.js, css/, views/*.js, vendor/ (echarts 5.5.1, three 0.160.0)
tests/            pytest: care score, alarms, quality, scenarios, API smoke
legacy/           (untouched) old Streamlit app, RF classifier and scripts remain in their folders
```

## Evaluation protocol (fixed before experiments)

* Per CARE dataset: models trained on its `train` rows with normal status (0/2) only, using
  **chronological** split: first 80 % of train = fit, last 20 % = validation (residual sigma, thresholds of
  the per-dataset normalization). The `prediction` rows are never used for fitting anything.
* Global alarm hyper-parameters (EWMA lambda / L, CUSUM k / h, persistence) are chosen by
  **leave-one-turbine-out** over the 5 assets: tune on datasets of 4 assets, score the 5th; report the
  pooled held-out result. No status labels or event labels are model inputs.
* Metrics: CARE score + 4 sub-scores; event recall (n/N), false-alarm events per turbine-month on normal
  datasets, detection delay median/p90 (relative to logged event start, caveat: Farm A logbook times),
  component-level precision/recall incl. an 'unknown' bucket, confusion matrix; always with counts.
* Acceptance target: ≥25 % fewer false-alarm events than the legacy residual detector at ≥ equal event recall.

## Alarm states

`normal` → `watch` (EWMA above warning limit) → `warning` (persistence met) → `fault` (sustained / CUSUM
decision + persistence) ; `unknown` = data quality insufficient to judge; `offline` = turbine stopped /
non-production state; `tripped` (simulation protection trip). Hysteresis: clear only after the statistic stays
below the clear limit for the cooldown period.

## REST API (served by server/main.py at http://127.0.0.1:8000, SPA at /)

All JSON. Times ISO-8601 strings. Power is per-unit of rated (dataset is normalized 0–1); UI shows %
of rated and an *illustrative* MW figure using a user-entered rating, clearly labelled.

```
GET  /api/meta                -> {dataset:{name,record,doi,version}, pipeline_version, assets:[{id,label,x,y}],
                                  modes:["replay","simulation"], rated_mw_default:2.0}
GET  /api/benchmark           -> {protocol:{...text}, methods:[MethodResult], selected:"<method id>",
                                  generated_at}
GET  /api/benchmark/{method}  -> {method, per_dataset:[DatasetResult], confusion:{labels:[], matrix:[[]]}}
GET  /api/forecast            -> {metrics:{ml:{mae,rmse,r2,n}, physics:{...}}, intervals:{nominal, coverage_by_regime:[...]},
                                  power_curve:[{wind, p_med, p_lo, p_hi}], audit:{split, leakage_checks:[...]}}
GET  /api/datasets            -> [{id, asset, label:"anomaly"|"normal", description, event_start, event_end,
                                  prediction_start, prediction_end, n_rows}]
GET  /api/replay/{id}         -> ReplayFile (precomputed, see below)
GET  /api/sim/state           -> FleetState
POST /api/sim/reset   {seed?, weather?}         -> FleetState
POST /api/sim/step    {steps}  (10-min steps)    -> FleetState
POST /api/sim/inject  {turbine, scenario, severity, ramp_h, duration_h, seed?} -> {injection, state:FleetState}
POST /api/sim/repair  {injection_id}            -> FleetState
GET  /api/sim/scenarios       -> [{id, name, category:"component"|"sensor"|"performance", component, description,
                                  params:{severity:[..], ramp_h:[..]}}]
GET  /api/sim/turbine/{id}?hours=48 -> TurbineSeries
GET  /api/sim/cases           -> [Case]
GET  /api/sim/injections      -> [Injection]   (test-oracle view: ground truth vs detection)
```

```
Injection = {id, turbine, scenario, name, category, component, severity, ramp_h, duration_h, seed,
  start, end, active:bool, detected_time|null, delay_h|null, detected_component|null, correct:bool|null,
  lost_energy_pu_h}
```

### Shapes

```
MethodResult = {id, name, kind:"baseline"|"proposed", care, coverage, accuracy, reliability, earliness,
                events_detected, events_total, false_alarm_events, normal_datasets, fa_per_turbine_month,
                normal_datasets_flagged, delay_median_h, delay_p90_h, component_correct, component_unknown,
                notes}
DatasetResult = {id, asset, label, description, detected:bool, first_alarm, delay_h, false_alarms:int,
                 diagnosis, diagnosis_correct:bool|null, care_parts:{...}}

TurbineSnapshot = {id, label, t, source:"replay"|"simulation", state, health:0..100,
  power, expected, expected_lo, expected_hi, wind, wind_dir, nacelle_dir, yaw_error, rotor_rpm, pitch, ambient,
  top_component|null, components:[{name, score, ewma, cusum, state}], quality:{state:"ok"|"degraded"|"bad", issues:[str]},
  injected:[str] (simulation only, ground truth shown in a separate 'test oracle' area)}
FleetState = {t, source, weather, turbines:[TurbineSnapshot], farm:{power, expected, energy_actual, energy_expected,
  lost_energy}, open_cases:int, alarms_24h:int}
TurbineSeries = {id, t:[...], signals:{<key>:{label, unit, observed:[], expected:[], lo:[], hi:[]}},
  components:{<name>:{score:[], ewma:[], cusum:[]}}, state:[...], limits:{ewma_warn, ewma_fault, cusum_h}}
Case = {id, turbine, source, status:"open"|"acknowledged"|"closed", state:"warning"|"fault"|"unknown",
  component, hypothesis, deviation_start, alarm_time, cleared_time|null, severity:0..1, confidence|null,
  evidence:[{signal, label, observed, expected, z, contribution}], lost_energy_pu_h,
  truth:{injection_id, scenario, delay_h}|null}
ReplayFile = {id, asset, label, description, event_start, event_end, pipeline_version,
  t:[...], power, expected, expected_lo, expected_hi, wind, ambient, state:[...], health:[...],
  components:{<name>:{score:[], ewma:[], cusum:[]}}, signals:{<key>:{label, unit, observed:[], expected:[]}},
  quality:[...], cases:[Case], limits:{...}}
```
