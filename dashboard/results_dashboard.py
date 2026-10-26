"""
Results dashboard - everything in one place
===========================================
* energy output of the farm and of every turbine (simulation session) and energy lost to anomalies
* forecasting accuracy (unseen real data)
* anomaly detection metrics on REAL data (Step 1): per-class table, confusion matrix, before vs after
* live detection results of the simulation: injected anomalies, detection delay, history timeline
* export: PDF report (button) and PNG of any chart (camera icon on the chart)

Used inside the digital twin app (page "Results dashboard"); can also run alone:
    streamlit run dashboard/results_dashboard.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "digital_twin"))
from twin_core import ui                                        # noqa: E402

FC_RES = os.path.join(ROOT, "forecasting_model", "results", "forecasting_results.json")
PM_METRICS = os.path.join(ROOT, "forecasting_model", "outputs_power_model", "metrics.json")
A1 = os.path.join(ROOT, "anomaly_detection_model", "results_attempt1")
A2 = os.path.join(ROOT, "anomaly_detection_model", "results_attempt2")
CLASS_ORDER = ["normal", "hydraulic", "gearbox", "generator_bearing", "transformer"]


def _json(p):
    return json.load(open(p)) if os.path.isfile(p) else None


def _csv(p, **kw):
    return pd.read_csv(p, **kw) if os.path.isfile(p) else None


def forecast_metrics():
    r, m = _json(FC_RES), _json(PM_METRICS)
    out = {}
    if m and "ml_gradient_boosting" in m:
        ml = m["ml_gradient_boosting"]
        out.update(MAE=ml["nMAE_%"], RMSE=ml["nRMSE_%"], R2=ml["R2"], n=ml.get("n"))
    if r:
        out["MAPE"] = r["ml_gradient_boosting"]["MAPE_%"]
        out["MAPE_rows"] = r["ml_gradient_boosting"]["MAPE_rows"]
        out.setdefault("MAE", r["ml_gradient_boosting"]["MAE_%rated"])
        out.setdefault("RMSE", r["ml_gradient_boosting"]["RMSE_%rated"])
        out.setdefault("R2", r["ml_gradient_boosting"]["R2"])
    return out


def anomaly_tables():
    return dict(before_after=_csv(os.path.join(A1, "before_vs_after.csv")),
                per_class=_csv(os.path.join(A1, "per_class_C.csv"), index_col=0),
                confusion=_csv(os.path.join(A1, "confusion_C.csv"), index_col=0),
                events=_csv(os.path.join(A1, "events_C.csv")),
                attempt2=_csv(os.path.join(A2, "attempt2_summary.csv")))


def energy_table(farm):
    if farm is None:
        return None
    rows = []
    lost = {a: 0.0 for a in farm.energy}
    for f in farm.injections:
        lost[f["turbine"]] += f["energy_lost_mwh"]
    for a, e in farm.energy.items():
        rows.append({"turbine": f"T-{a}", "actual MWh": round(e["actual"], 2), "expected MWh": round(e["expected"], 2),
                     "lost to injected anomalies MWh": round(lost[a], 2),
                     "performance %": round(100 * e["actual"] / e["expected"], 1) if e["expected"] > 0 else np.nan})
    t = pd.DataFrame(rows)
    tot = {"turbine": "FARM", "actual MWh": t["actual MWh"].sum(), "expected MWh": t["expected MWh"].sum(),
           "lost to injected anomalies MWh": t["lost to injected anomalies MWh"].sum()}
    tot["performance %"] = round(100 * tot["actual MWh"] / tot["expected MWh"], 1) if tot["expected MWh"] else np.nan
    return pd.concat([t, pd.DataFrame([tot])], ignore_index=True)


def injection_table(farm):
    if farm is None or not farm.injections:
        return None
    rows = []
    for f in farm.injections:
        exp_cls = f["expected_class"] or "(not a trained class)"
        rows.append({"#": f["id"], "turbine": f"T-{f['turbine']}", "anomaly": f["type"], "severity": f["severity"],
                     "start": pd.Timestamp(f["start"]).strftime("%d %b %H:%M"),
                     "RF class": f["rf_class"] or "-", "RF correct": (f["rf_class"] == f["expected_class"]) if f["rf_class"] else False,
                     "RF delay h": round((f["rf_detected"] - f["start"]).total_seconds() / 3600, 1) if f["rf_detected"] is not None else None,
                     "residual component": f["det_component"] or "-",
                     "residual correct": f["det_component"] == f["expected_component"],
                     "residual delay h": round((f["det_detected"] - f["start"]).total_seconds() / 3600, 1) if f["det_detected"] is not None else None,
                     "true class": exp_cls, "energy lost MWh": round(f["energy_lost_mwh"], 2), "tripped": f["tripped"],
                     "trip after h": round((f["trip_time"] - f["start"]).total_seconds() / 3600, 1) if f.get("trip_time") is not None else None})
    return pd.DataFrame(rows)


def render(st, go, farm=None, key="dash"):
    ui.title(st, "Results dashboard", "Energy, forecasting accuracy, anomaly detection metrics (real data) and the live simulation")
    # ---------------- energy
    st.subheader("Energy output and energy lost")
    et = energy_table(farm)
    if et is not None:
        f = et[et.turbine != "FARM"]
        ui.kpis(st, [("Farm energy (simulated session)", f"{et.iloc[-1]['actual MWh']:.1f} MWh", f"expected {et.iloc[-1]['expected MWh']:.1f} MWh"),
                     ("Lost to injected anomalies", f"{et.iloc[-1]['lost to injected anomalies MWh']:.2f} MWh", "expected minus actual while active"),
                     ("Farm performance", f"{et.iloc[-1]['performance %']:.1f} %", "actual / expected energy"),
                     ("Rated power used", f"{farm.rated_mw:.1f} MW", "per turbine (power data is scaled 0-1)")])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=f["turbine"], y=f["actual MWh"], name="actual", marker_color=ui.TEAL))
        fig.add_trace(go.Bar(x=f["turbine"], y=f["expected MWh"], name="expected (twin)", marker_color=ui.GREY))
        fig.add_trace(go.Bar(x=f["turbine"], y=f["lost to injected anomalies MWh"], name="lost to anomalies", marker_color=ui.RED))
        fig.update_layout(barmode="group")
        st.plotly_chart(ui.style(fig, "Energy per turbine (simulation session)", 330, "MWh"), use_container_width=True, key=f"{key}_energy")
        st.dataframe(et, hide_index=True, use_container_width=True)
    else:
        st.info("Start the live simulation (Wind farm page) to see energy figures here.")
    # ---------------- forecasting
    st.subheader("Forecasting accuracy (unseen real data)")
    fm = forecast_metrics()
    if fm:
        ui.kpis(st, [("MAE", f"{fm.get('MAE', float('nan')):.2f} %", "of rated power"),
                     ("RMSE", f"{fm.get('RMSE', float('nan')):.2f} %", "of rated power"),
                     ("R²", f"{fm.get('R2', float('nan')):.3f}", f"{fm.get('n', '')} test rows" if fm.get("n") else ""),
                     ("MAPE", f"{fm.get('MAPE', float('nan')):.1f} %", "rows above 10 % of rated")])
    # ---------------- anomaly metrics (real data)
    st.subheader("Anomaly detection on real data (Step 1, leave-one-turbine-out)")
    t = anomaly_tables()
    if t["before_after"] is not None:
        st.dataframe(t["before_after"], hide_index=True, use_container_width=True)
    if t["attempt2"] is not None:
        st.caption("Attempt 2 (low false alarms):")
        st.dataframe(t["attempt2"], hide_index=True, use_container_width=True)
    c1, c2 = st.columns([1, 1.15])
    with c1:
        if t["per_class"] is not None:
            st.caption("Random Forest (model C, used in this app) - per-class metrics, out-of-fold rows")
            st.dataframe(t["per_class"], use_container_width=True)
    with c2:
        cm = t["confusion"]
        if cm is not None:
            z = cm.to_numpy().astype(float)
            zn = z / np.clip(z.sum(axis=1, keepdims=True), 1, None)
            fig = go.Figure(go.Heatmap(z=zn, x=[c.replace("pred ", "") for c in cm.columns], y=[c.replace("true ", "") for c in cm.index],
                                       text=[[f"{int(v)}" for v in row] for row in z], texttemplate="%{text}", colorscale="Blues",
                                       zmin=0, zmax=1, showscale=False))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(ui.style(fig, "Confusion matrix - Random Forest (C)", 340), use_container_width=True, key=f"{key}_cm")
    st.caption("With 1-6 events per fault type, no model classifies the fault type reliably on turbines it has not seen "
               "(upper limit with settings tuned on all data: 3 of 12). Values near 100 % only appear with data leakage "
               "(logbook status flags + random split).")
    # ---------------- live simulation results
    st.subheader("Live simulation: injected anomalies and detection")
    it = injection_table(farm)
    if it is not None:
        st.dataframe(it, hide_index=True, use_container_width=True)
        fig = go.Figure(go.Bar(x=it["#"].astype(str) + " " + it["anomaly"], y=it["energy lost MWh"], marker_color=ui.RED))
        st.plotly_chart(ui.style(fig, "Energy lost per injected anomaly", 300, "MWh"), use_container_width=True, key=f"{key}_lost")
    else:
        st.info("No anomalies injected yet (Anomalies page).")
    if farm is not None and farm.events:
        ev = pd.DataFrame(farm.events)
        fig = go.Figure()
        for det, colr, y in (("Random Forest", ui.PURPLE, 1), ("Residual detector", ui.TEAL, 0)):
            e = ev[ev.detector == det]
            if len(e):
                fig.add_trace(go.Scatter(x=e["time"], y=[f"T-{a}" for a in e["turbine"]], mode="markers", name=det,
                                         marker=dict(size=11, color=colr, symbol="diamond" if y else "circle"),
                                         text=e["result"].astype(str) + "  (" + e["injected"].where(e["injected"] != "", "no injection") + ")",
                                         hovertemplate="%{x}<br>%{y}: %{text}<extra></extra>"))
        for f in farm.injections:
            fig.add_trace(go.Scatter(x=[f["start"], min(f["end"], farm.now)], y=[f"T-{f['turbine']}"] * 2, mode="lines",
                                     line=dict(color=ui.RED, width=10), opacity=0.25, name="injected", showlegend=False,
                                     hovertemplate=f"{f['type']}<extra></extra>"))
        fig.update_layout(hovermode="closest")
        st.plotly_chart(ui.style(fig, "Anomaly history (bars = injected, markers = detections)", 300), use_container_width=True, key=f"{key}_timeline")
    # ---------------- export
    st.subheader("Export")
    if st.button("📄 Build PDF report", key=f"{key}_pdf"):
        import export_pdf
        st.session_state[f"{key}_pdf_bytes"] = export_pdf.build(farm)
    if st.session_state.get(f"{key}_pdf_bytes"):
        st.download_button("⬇ Download results_dashboard.pdf", st.session_state[f"{key}_pdf_bytes"],
                           file_name="results_dashboard.pdf", mime="application/pdf", key=f"{key}_dl")
    st.caption("Every chart can also be saved as PNG with the camera icon in its top-right corner.")


def main():
    import streamlit as st
    import plotly.graph_objects as go
    st.set_page_config(page_title="Results dashboard", page_icon="📊", layout="wide")
    st.markdown(ui.CSS, unsafe_allow_html=True)
    render(st, go, st.session_state.get("farm"))


if __name__ == "__main__":
    main()
