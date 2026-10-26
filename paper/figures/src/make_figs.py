"""Regenerate the data figures of the paper from the project artifacts.

    python paper/figures/src/make_figs.py        (from the project root)
Reads artifacts/benchmark/benchmark.json and artifacts/replay/22.json.gz; writes vector PDFs to paper/figures/.
"""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
SHORT = {"legacy_rf": "Legacy RF", "legacy_residual": "Legacy residual", "iforest": "Isolation forest",
         "autoencoder": "Autoencoder", "nbm_threshold": "NBM + threshold", "nbm_ewma": "NBM + EWMA",
         "nbm_cusum": "NBM + CUSUM", "nbm_ewma_cusum_free": "NBM + E/C (free)", "nbm_ewma_cusum": "NBM + E/C (sel.)"}


def _preview(fig, name):
    """Optional PNG preview: set WT_PREVIEW_DIR to a folder."""
    import os
    if os.environ.get("WT_PREVIEW_DIR"):
        fig.savefig(Path(os.environ["WT_PREVIEW_DIR"]) / f"{name}.png", dpi=170)


def fig_tradeoff():
    b = json.load(open(ROOT / "artifacts/benchmark/benchmark.json"))
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    for m in b["methods"]:
        x, y = m["false_alarm_events"], m["events_detected"]
        prop = m["kind"] == "proposed"
        ax.scatter(max(x, 0.7), y, s=22, marker="o" if prop else "s",
                   facecolor="black" if m["id"] == b["selected"] else ("white" if prop else "0.6"),
                   edgecolor="black", linewidth=0.7, zorder=3)
        dx, dy, ha = 1.15, 0.0, "left"
        if m["id"] in ("nbm_cusum", "nbm_ewma_cusum"):
            dy = -0.5 if m["id"] == "nbm_cusum" else 0.45
        if m["id"] in ("nbm_ewma", "nbm_ewma_cusum_free"):
            dx, ha = 0.85, "right"
        ax.annotate(SHORT[m["id"]], (max(x, 0.7) * dx, y + dy), fontsize=6.3, ha=ha, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 5000)
    ax.set_ylim(1, 13)
    ax.set_xlabel("False-alarm events on 10 normal datasets (log; 0 plotted at 0.7)")
    ax.set_ylabel("Fault events detected (of 12)")
    ax.grid(True, which="major", linewidth=0.3, color="0.85")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "tradeoff.pdf")
    _preview(fig, "tradeoff")


def fig_replay(ds=22):
    d = json.load(gzip.open(ROOT / f"artifacts/replay/{ds}.json.gz"))
    t = pd.to_datetime(pd.Series(d["t"]))
    es, ee = pd.Timestamp(d["event_start"]), pd.Timestamp(d["event_end"])
    case = d["cases"][0]
    alarm, dev = pd.Timestamp(case["alarm_time"]), pd.Timestamp(case["deviation_start"])
    win = (t >= es - pd.Timedelta(days=2)) & (t <= es + pd.Timedelta(days=4))
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.1), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    sig = d["signals"]["hydraulic_minus_nacelle"]
    hr = pd.DataFrame({"o": sig["observed"], "e": sig["expected"]}, dtype=float).set_index(t)
    hr = hr[win.to_numpy()].resample("1h").mean()                 # hourly means for legibility
    axs[0].plot(hr.index, hr["o"], color="black", lw=0.8, label="observed")
    axs[0].plot(hr.index, hr["e"], color="0.5", lw=0.8, ls="--", label="NBM expected")
    axs[0].set_ylabel("Hyd. oil $-$ nacelle (K)")
    axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=2, borderaxespad=0.1)
    c = d["components"]["Hydraulic group"]
    e = np.array([np.nan if v is None else v for v in c["ewma"]], float)
    s = np.array([np.nan if v is None else v for v in c["cusum"]], float)
    axs[1].plot(t[win], e[win], color="black", lw=0.7, label="EWMA evidence")
    axs[1].plot(t[win], s[win], color="0.55", lw=0.7, label="CUSUM")
    axs[1].axhline(d["limits"]["ewma_warn"], color="black", lw=0.5, ls=":")
    axs[1].axhline(d["limits"]["cusum_h"], color="0.55", lw=0.5, ls=":")
    axs[1].set_ylabel("Statistic (scaled)")
    axs[1].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=2, borderaxespad=0.1)
    for ax in axs:
        ax.axvspan(es, min(ee, t[win].max()), color="0.92", lw=0)
        ax.axvline(alarm, color="black", lw=0.8)
        ax.axvline(dev, color="black", lw=0.5, ls="--")
        ax.grid(True, linewidth=0.3, color="0.88")
    axs[1].set_ylim(0, max(6, np.nanmax(s[win]) * 1.05))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(OUT / "replay22.pdf")
    _preview(fig, "replay22")
    return {"event_start": str(es), "alarm": str(alarm), "deviation_start": str(dev),
            "delay_h": (alarm - es).total_seconds() / 3600}


if __name__ == "__main__":
    fig_tradeoff()
    print(fig_replay())
