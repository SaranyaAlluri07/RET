"""
Evaluation of the forecasting (power) model on UNSEEN data
==========================================================
Reads outputs_power_model/ (written by power_model.py) and computes, for the Gradient Boosting
model and the physics power curve:
  MAE, RMSE, R2 (all rows of the unseen normal test period, normal operation)
  MAPE (only rows with actual power > 10 % of rated - below that the percentage error explodes)
  per-turbine metrics, residual statistics, and the figures used by the app / PDF report.

    python forecasting_model/evaluate_forecasting.py
Output: forecasting_model/results/forecasting_results.json + PNG figures
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "outputs_power_model")
OUT = os.path.join(HERE, "results")


def find_farm():
    import glob
    root = os.path.dirname(HERE)
    for base in (os.path.join(root, "data"), root, os.path.dirname(root)):
        for d in glob.glob(os.path.join(base, "**", "Wind Farm A"), recursive=True):
            files = [f for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True) if os.path.basename(f)[:-4].isdigit()]
            if files:
                return files
    return []


def load_test():
    """Same test set as power_model.py: unseen prediction period of the healthy datasets, normal operation,
    minus rows that also appear in any dataset's TRAINING period (overlapping datasets)."""
    d = pd.read_csv(os.path.join(SRC, "test_predictions.csv"), parse_dates=["time_stamp"])
    d = d[(d["event_label"].astype(str).str.lower() == "normal") & d["status_type_id"].isin([0, 2])].copy()
    files = find_farm()
    note = "all healthy prediction-period rows (raw data not found, overlap filter not applied)"
    if files:
        keys = []
        for f in files:
            sep = ";" if open(f, encoding="utf-8", errors="ignore").readline().count(";") > 1 else ","
            r = pd.read_csv(f, sep=sep, usecols=["time_stamp", "asset_id", "train_test"])
            r = r[r["train_test"].astype(str).str.lower() == "train"]
            keys.append(pd.DataFrame({"asset_id": r["asset_id"], "time_stamp": pd.to_datetime(r["time_stamp"])}))
        k = pd.concat(keys).drop_duplicates()
        d = d.merge(k.assign(_train=1), on=["asset_id", "time_stamp"], how="left")
        d = d[d["_train"].isna()].drop(columns="_train")
        note = "unseen healthy rows, rows overlapping any training period removed (same as power_model.py)"
    return d.drop_duplicates(["asset_id", "time_stamp"]), note


def scores(y, p, rated):
    e = p - y
    rmse, mae = float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))
    r2 = float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
    big = y > 0.1 * rated
    mape = float(np.mean(np.abs(e[big] / y[big])) * 100) if big.any() else float("nan")
    return {"MAE_%rated": 100 * mae / rated, "RMSE_%rated": 100 * rmse / rated, "R2": r2,
            "MAPE_%": mape, "MAPE_rows": int(big.sum()), "n": int(len(y))}


def main():
    os.makedirs(OUT, exist_ok=True)
    m = json.load(open(os.path.join(SRC, "metrics.json")))
    rated = float(m.get("rated_power_estimate", 1.0))
    d, note = load_test()
    y = d["power"].to_numpy()
    res = {"test_rows": int(len(d)), "test_set": note, "rated_power_estimate": rated,
           "official_power_model_metrics": {k: m[k] for k in ("ml_gradient_boosting", "physics_power_curve") if k in m},
           "ml_gradient_boosting": scores(y, d["pred_ml"].to_numpy(), rated),
           "physics_power_curve": scores(y, d["pred_phys"].to_numpy(), rated),
           "per_turbine": {str(a): scores(g["power"].to_numpy(), g["pred_ml"].to_numpy(), rated)
                           for a, g in d.groupby("asset_id")},
           "feature_importance": m.get("feature_importance", {}),
           "features": m.get("features", []), "train_test_note": "Trained on the training periods (normal operation); "
           "tested on the unseen prediction periods of the healthy datasets."}
    r = d["pred_ml"] - d["power"]
    res["residual"] = {"mean": float(r.mean()), "std": float(r.std()), "p05": float(r.quantile(0.05)), "p95": float(r.quantile(0.95))}
    json.dump(res, open(os.path.join(OUT, "forecasting_results.json"), "w"), indent=2)

    k = 100 / rated
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(d["power"] * k, d["pred_ml"] * k, s=3, alpha=0.25, color="#2F80ED")
    ax.plot([0, 100], [0, 100], color="#1F2D3D", lw=1, ls="--")
    ax.set_xlabel("Actual power (% of rated)"); ax.set_ylabel("Predicted power (% of rated)")
    ax.set_title("Predicted vs actual (unseen data)", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pred_vs_actual.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(r * k, bins=80, color="#0E9F8E", alpha=0.85)
    ax.axvline(0, color="#1F2D3D", lw=1)
    ax.set_xlabel("Error = predicted - actual (% of rated)"); ax.set_ylabel("rows")
    ax.set_title("Residual distribution", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "residual_hist.png"), dpi=140); plt.close(fig)

    b = d.assign(bin=(d["wind"] // 1).clip(upper=24), res=r * k).groupby("bin")["res"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.fill_between(b.mean().index, b.quantile(0.1), b.quantile(0.9), color="#2F80ED", alpha=0.18, label="80 % of errors")
    ax.plot(b.mean().index, b.mean(), color="#2F80ED", lw=2, label="mean error")
    ax.axhline(0, color="#1F2D3D", lw=1)
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Error (% of rated)"); ax.legend(frameon=False)
    ax.set_title("Residual vs wind speed", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "residual_vs_wind.png"), dpi=140); plt.close(fig)

    g = d.assign(bin=(d["wind"] * 2).round() / 2).groupby("bin")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d["wind"], d["power"] * k, s=2, alpha=0.15, color="#A3B1C2", label="measured")
    ax.plot(g["pred_phys"].mean().index, g["pred_phys"].mean() * k, color="#F59E0B", lw=2, label="physics power curve")
    ax.plot(g["pred_ml"].mean().index, g["pred_ml"].mean() * k, color="#2F80ED", lw=2, ls="--", label="ML model")
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("% of rated power"); ax.legend(frameon=False)
    ax.set_title("Power curve (unseen data)", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "power_curve.png"), dpi=140); plt.close(fig)

    fi = pd.Series(res["feature_importance"]).sort_values()
    if len(fi):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.barh(fi.index, fi.values, color="#0E9F8E")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("Drop in R² when the feature is shuffled (permutation importance)")
        ax.set_title("Feature importance", loc="left", fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "feature_importance.png"), dpi=140); plt.close(fig)

    ml = res["ml_gradient_boosting"]
    print(f"Test rows: {len(d):,}  ({note})")
    print(f"ML  : MAE {ml['MAE_%rated']:.2f} %  RMSE {ml['RMSE_%rated']:.2f} %  R2 {ml['R2']:.4f}  MAPE {ml['MAPE_%']:.1f} % (rows > 10 % rated)")
    ph = res["physics_power_curve"]
    print(f"Phys: MAE {ph['MAE_%rated']:.2f} %  RMSE {ph['RMSE_%rated']:.2f} %  R2 {ph['R2']:.4f}  MAPE {ph['MAPE_%']:.1f} %")
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
