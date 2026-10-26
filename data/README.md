# Data

## Raw dataset (needed once, to build the simulated turbines)
CARE to Compare – "Wind Turbine SCADA Data For Early Fault Detection", Zenodo record 14958989.
Extract the zip so that this path exists (any one of these places works):

* `data/CARE_To_Compare/.../Wind Farm A/datasets/<id>.csv`  (inside this bundle), or
* the folder that **contains** this bundle (e.g. your old project folder `D:\Projects\RET -- Claude\CARE_To_Compare\...`)

The app looks for a folder called `Wind Farm A` with `event_info.csv` and the dataset CSV files.
The simulated turbines use the healthy datasets 24 (asset 0), 3 (asset 10), 25 (asset 11), 14 (asset 13), 13 (asset 21).

## processed/ (included)
* `raw_<id>.csv.gz` – selected Avg signals of each dataset (prediction period + 2 days of history)
* `baseline_<id>.json` – each sensor's healthy median / IQR (training period, normal status)
* `event_info.csv`, `feature_description.csv` – copied from the dataset
* made by `extract_signals.py`; used to train / retrain the Random Forest classifier
