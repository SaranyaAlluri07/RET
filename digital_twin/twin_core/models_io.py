"""Loading the trained models (forecasting / power model, Random Forest anomaly classifier, turbines)."""
import glob
import json
import os

import joblib
import numpy as np

from twin_core import paths


class BinnedPowerCurve:                       # class of the pickled physics power curve (power_model.py)
    def __init__(self, bin_width=0.5):
        self.bin_width = bin_width

    def predict(self, wind_norm):
        return np.interp(wind_norm, self.v_, self.p_, left=0.0, right=self.p_[-1])


def _register():
    import __main__
    __main__.BinnedPowerCurve = BinnedPowerCurve


def power_model():
    _register()
    p = os.path.join(paths.POWER_OUT, "power_model.joblib")
    return joblib.load(p) if os.path.isfile(p) else None


def power_sigma():
    """Std of the power model error on unseen normal data (for the error band)."""
    import pandas as pd
    p = os.path.join(paths.POWER_OUT, "test_predictions.csv")
    if not os.path.isfile(p):
        return 0.03
    d = pd.read_csv(p)
    if "event_label" in d:
        d = d[d["event_label"].astype(str).str.lower() == "normal"]
    r = (d["power"] - d["pred_ml"]).dropna()
    return float(r.std()) if len(r) else 0.03


def anomaly_classifier(name="anomaly_classifier.joblib", log=print):
    """Random Forest (Step 1, model C). If the file cannot be loaded with this scikit-learn version it is
    retrained here from the bundled data (same data, features and settings; ~1-2 min, no cross-validation)."""
    p = os.path.join(paths.ANOMALY_DIR, name)
    try:
        return joblib.load(p)
    except Exception as e:                          # missing or saved with another scikit-learn version
        log(f"Anomaly classifier could not be loaded ({type(e).__name__}) - retraining it on this computer ...")
        import anomaly_classifier as ac
        ac.train_final(paths.ROOT, p, log=log)
        return joblib.load(p)


def turbines():
    out = {}
    for a in paths.ASSETS:
        p = os.path.join(paths.MODEL_DIR, f"turbine_{a}.joblib")
        if os.path.isfile(p):
            try:
                out[a] = joblib.load(p)
            except Exception:                       # other scikit-learn version -> rebuilt by prepare
                os.replace(p, p + ".incompatible")
    return out


def json_or_none(path):
    return json.load(open(path)) if os.path.isfile(path) else None
