"""
PDF export of the results dashboard (matplotlib only - works offline).
    python dashboard/export_pdf.py            -> dashboard/results_dashboard.pdf (static results)
In the app: Results dashboard -> "Build PDF report" (includes the live simulation results).
"""
import io
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import results_dashboard as rd                                   # noqa: E402

FC_DIR = os.path.join(rd.ROOT, "forecasting_model", "results")
INK, MUTED, TEAL, RED, GREY = "#1F2D3D", "#6B7C93", "#0E9F8E", "#E5484D", "#A3B1C2"


def _text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", color=INK)
    y = 0.90
    for ln in lines:
        for part in textwrap.wrap(ln, 95) or [""]:
            fig.text(0.08, y, part, fontsize=10, color=INK, family="monospace" if ln.startswith("  ") else None)
            y -= 0.022
        if y < 0.06:
            pdf.savefig(fig); plt.close(fig); fig = plt.figure(figsize=(8.27, 11.69)); y = 0.94
    pdf.savefig(fig); plt.close(fig)


def _table_page(pdf, title, df, note=None):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off"); ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK)
    tb = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns), loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.4)
    if note:
        fig.text(0.05, 0.05, textwrap.fill(note, 160), fontsize=9, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def _image_page(pdf, title, paths):
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return
    n = len(paths); cols = 2 if n > 1 else 1; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, paths):
        ax.imshow(plt.imread(p))
    fig.suptitle(title, x=0.05, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def build(farm=None, path=None):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fm = rd.forecast_metrics()
        lines = ["Wind farm digital twin - results report", "",
                 "Data: CARE to Compare, Wind Farm A (5 turbines, 22 datasets, 12 recorded faults of 4 types).", "",
                 "FORECASTING (power from weather, unseen real data)"]
        if fm:
            lines += [f"  MAE  {fm.get('MAE', float('nan')):.2f} % of rated", f"  RMSE {fm.get('RMSE', float('nan')):.2f} % of rated",
                      f"  R2   {fm.get('R2', float('nan')):.3f}", f"  MAPE {fm.get('MAPE', float('nan')):.1f} % (rows above 10 % of rated)"]
        t = rd.anomaly_tables()
        lines += ["", "ANOMALY DETECTION (real data, leave-one-turbine-out)",
                  "  Model in the app: Random Forest (Step 1, approach C), 175 features, class-balanced.",
                  "  Cross-check: residual detector (normal-behaviour models) + 24-h energy performance monitor."]
        if t["before_after"] is not None:
            for _, r in t["before_after"].iterrows():
                lines.append(f"  {r['approach'][:46]:<46} detected {r['events detected']:>5}  correct {r['events correct class']:>5}  "
                             f"false alarms {r['false alarms (healthy)']:>5}")
        lines += ["", "  With 1-6 events per fault type the fault type cannot be classified reliably on unseen turbines;",
                  "  even with settings tuned on all data at most 3 of 12 faults are classified correctly."]
        et = rd.energy_table(farm)
        if et is not None:
            lines += ["", f"SIMULATION SESSION (rated power {farm.rated_mw:.1f} MW per turbine, until {pd.Timestamp(farm.now):%d %b %Y %H:%M})"]
            for _, r in et.iterrows():
                lines.append(f"  {r['turbine']:<6} actual {r['actual MWh']:8.1f} MWh  expected {r['expected MWh']:8.1f} MWh  "
                             f"lost to anomalies {r['lost to injected anomalies MWh']:6.2f} MWh")
        _text_page(pdf, "Results dashboard", lines)
        _image_page(pdf, "Forecasting model", [os.path.join(FC_DIR, f) for f in
                                               ("pred_vs_actual.png", "power_curve.png", "residual_hist.png", "feature_importance.png")])
        if t["per_class"] is not None:
            _table_page(pdf, "Random Forest (C) - per-class metrics (real data, out-of-fold)", t["per_class"].reset_index().rename(columns={"index": "class"}))
        _image_page(pdf, "Confusion matrices (real data)", [os.path.join(rd.A1, "confusion_C.png"), os.path.join(rd.A2, "confusion_E.png")])
        it = rd.injection_table(farm)
        if it is not None:
            _table_page(pdf, "Live simulation - injected anomalies and detection",
                        it[["#", "turbine", "anomaly", "severity", "RF class", "RF delay h", "residual component", "residual delay h", "energy lost MWh", "tripped"]],
                        "RF = Random Forest classifier; residual = normal-behaviour residual detector + energy performance monitor. "
                        "Synthetic faults show how the twin reacts; accuracy is reported on real data only.")
            fig, ax = plt.subplots(figsize=(11.69, 5))
            ax.bar(it["#"].astype(str) + " " + it["anomaly"].str.slice(0, 22), it["energy lost MWh"], color=RED)
            ax.set_ylabel("MWh"); ax.set_title("Energy lost per injected anomaly", loc="left", fontweight="bold")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        if farm is not None:
            fig, ax = plt.subplots(figsize=(11.69, 5))
            for a, tb in farm.turbines.items():
                d = tb.out.tail(7 * 144)
                ax.plot(d["time_stamp"], d["power"].clip(lower=0) * farm.rated_mw, lw=1, label=f"T-{a}")
            ax.set_ylabel("MW"); ax.legend(ncol=5, frameon=False); ax.set_title("Power of each turbine (last 7 simulated days)", loc="left", fontweight="bold")
            fig.autofmt_xdate(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    data = buf.getvalue()
    if path:
        open(path, "wb").write(data)
    return data


if __name__ == "__main__":
    out = os.path.join(HERE, "results_dashboard.pdf")
    build(None, out)
    print("Saved", out)
