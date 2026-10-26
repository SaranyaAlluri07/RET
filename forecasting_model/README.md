# Forecasting model (power from weather)

| File | What |
|---|---|
| `power_model.py` | trains the physics power curve (IEC 61400-12, air-density corrected) and the Gradient Boosting model; writes `outputs_power_model/` (model `power_model.joblib`, `metrics.json`, `test_predictions.csv`) |
| `evaluate_forecasting.py` | MAE, RMSE, R², MAPE (rows > 10 % of rated), per turbine, residual analysis, figures -> `results/` |
| `forecast_model.py` | optional: short-term forecast 10 min – 3 h ahead with 80 % range |
| `outputs_power_model/` | the trained model used by the digital twin (trained on the project laptop) |
| `results/` | evaluation results and figures |

Retrain (needs the raw dataset, see `data/README.md`):
```
python forecasting_model/power_model.py
python forecasting_model/evaluate_forecasting.py
```
Results on unseen data (power_model.py test set, 7,815 rows): R² 0.9946, RMSE 1.82 %, MAE 1.15 % of rated (physics curve: R² 0.9927, RMSE 2.12 %).
