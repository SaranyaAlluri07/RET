# Anomaly detection model

| File | What |
|---|---|
| `anomaly_classifier.joblib` | **the model used by the app: Random Forest (Step 1, approach C)**, 175 features, classes normal / hydraulic / gearbox / generator_bearing / transformer |
| `anomaly_classifier.py` | Step 1: compares the old model (A) with Gradient Boosting (B), Random Forest (C) and detector + classifier (D); leave-one-turbine-out cross-validation; per-class metrics, confusion matrix, before vs after, leakage demonstration |
| `anomaly_classifier_v2.py` | attempt 2: low-false-alarm detector + fault-type classification (E, F) and the upper-limit analysis |
| `results_attempt1/`, `results_attempt2/` | per-class metrics, confusion matrices (CSV + PNG), event-level results, reports |
| `fault_detection.py`, `fault_detection_v2.py` | normal-behaviour models (residual z-scores) whose scores are features of the classifier; `outputs_fault_detection*/scores/` = their outputs on the real data |
| `improve_detection.py` | variant comparison + "can each fault be separated from healthy turbines?" report |

Retrain:
```
python anomaly_detection_model/anomaly_classifier.py --final-only   # only the final model C (~1 min)
python anomaly_detection_model/anomaly_classifier.py                # full comparison with cross-validation (~50 min)
python anomaly_detection_model/anomaly_classifier_v2.py             # attempt 2 (~5 min)
```
The app retrains the model automatically (`--final-only`) if the file cannot be loaded with your scikit-learn version.

Real-data results (leave-one-turbine-out, running turbine only): see `results_attempt1/before_vs_after.csv`.
Random Forest (C): 8/12 faults detected, 1/12 correct fault type, false alarms on 6/10 healthy turbines.
The fault type cannot be classified reliably with 1–6 events per type (upper limit 3/12, see attempt 2).
