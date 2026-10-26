"""All paths are relative to the bundle folder, so the project runs on any laptop."""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWIN_DIR = os.path.join(ROOT, "digital_twin")
MODEL_DIR = os.path.join(TWIN_DIR, "models")
FORECAST_DIR = os.path.join(ROOT, "forecasting_model")
POWER_OUT = os.path.join(FORECAST_DIR, "outputs_power_model")
ANOMALY_DIR = os.path.join(ROOT, "anomaly_detection_model")
DASH_DIR = os.path.join(ROOT, "dashboard")
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED = os.path.join(DATA_DIR, "processed")

for p in (ANOMALY_DIR, FORECAST_DIR, DASH_DIR, TWIN_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# the 5 simulated turbines: real asset id -> healthy dataset it imitates, position in the farm (m)
ASSETS = {0: 24, 10: 3, 11: 25, 13: 14, 21: 13}
LAYOUT = {0: (-520, -160), 10: (-260, 120), 11: (0, -120), 13: (260, 140), 21: (520, -110)}


def find_farm_dir():
    """Wind Farm A folder: data/, the bundle, or the folder that contains the bundle."""
    for base in (DATA_DIR, ROOT, os.path.dirname(ROOT)):
        for d in glob.glob(os.path.join(base, "**", "Wind Farm A"), recursive=True):
            if os.path.isfile(os.path.join(d, "event_info.csv")) and glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
                if any(os.path.basename(f)[:-4].isdigit() for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)):
                    return d
    return None


def dataset_path(ds):
    farm = find_farm_dir()
    if not farm:
        return None
    hits = glob.glob(os.path.join(farm, "**", f"{ds}.csv"), recursive=True)
    return hits[0] if hits else None
