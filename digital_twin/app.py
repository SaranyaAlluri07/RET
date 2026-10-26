"""
Wind Farm Digital Twin - main application
=========================================
    pip install -r requirements.txt
    streamlit run digital_twin/app.py

Pages
  🌐 Wind farm (live)     3D farm (click a turbine), live readings, actual vs expected power with error band
  ⚠️ Anomalies            current anomalies, anomaly injection (12 types), detection in real time, energy lost
  📈 Forecasting details  step-by-step explanation of the forecasting model with its results
  📊 Results dashboard    energy, forecast metrics, anomaly metrics, history, PDF export
  ❓ Help                 what each part does
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore")
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin_core import paths, ui, models_io                     # noqa: E402
from twin_core import simulator as sim                          # noqa: E402
from twin_core import prepare                                   # noqa: E402
sys.path.insert(0, paths.DASH_DIR)
import results_dashboard                                        # noqa: E402

FRAME_SECONDS = 1.5
PAGES = ["🌐 Wind farm (live)", "⚠️ Anomalies", "📈 Forecasting details", "📊 Results dashboard", "❓ Help"]
LIVE = PAGES[:2]
farm3d = components.declare_component("farm3d", path=os.path.join(HERE, "components", "farm3d"))


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner=False)
def load_static():
    return dict(pm=models_io.power_model(), rf=models_io.anomaly_classifier(), sigma=models_io.power_sigma())


def get_farm(rated_mw, scenario):
    farm = st.session_state.get("farm")
    if farm is None:
        missing = prepare.missing_models()
        if missing:
            if not paths.find_farm_dir():
                st.error("The trained turbine models are missing and the Wind Farm A dataset was not found. "
                         "Put the CARE_To_Compare folder into the data/ folder (see data/README.md) and restart.")
                st.stop()
            box = st.status(f"First start: preparing {len(missing)} simulated turbines from the real data "
                            f"(about 1-2 min each, only once) ...", expanded=True)
            prepare.prepare(missing, log=lambda s: box.write(s))
            box.update(label="Turbines prepared", state="complete")
        s = load_static()
        models = models_io.turbines()
        with st.spinner("Starting the wind farm simulation ..."):
            farm = sim.FarmSimulator(models, s["pm"], s["rf"], power_sigma=s["sigma"], scenario=scenario,
                                     seed=int(time.time()) % 1000, rated_mw=rated_mw, warmup_hours=30)
        st.session_state["farm"] = farm
        st.session_state.setdefault("selected", next(iter(models)))
    farm.rated_mw = rated_mw
    farm.set_scenario(scenario)
    return farm


# ------------------------------------------------------------------ helpers
def status_of(row, mode):
    if bool(row.get("tripped")):
        return "tripped"
    rf, det = bool(row.get("rf_alarm")), bool(row.get("det_alarm"))
    if mode.startswith("Random Forest"):
        fault, warn = rf, row.get("rf_class") != "normal"
    elif mode.startswith("Residual"):
        fault, warn = det, row.get("det_score", 0) > 0.6 * sim.DET_THRESHOLD
    else:
        fault, warn = rf or det, (row.get("rf_class") != "normal") or row.get("det_score", 0) > 0.6 * sim.DET_THRESHOLD
    return "fault" if fault else ("warning" if warn else "normal")


def cause_for(row):
    comp = row.get("det_component")
    for v in sim.ANOMALIES.values():
        if v["comp"] == comp:
            return v["cause"]
    return "-"


def readings_html(a, r, mode, rated):
    st_ = status_of(r, mode)
    col = ui.STATUS_COL[st_]
    rf = sim.CLASS_LABEL.get(r["rf_class"], r["rf_class"])
    rows = [("Wind speed", f"{r['wind']:.1f} m/s"), ("Wind direction", f"{r['wind_dir']:.0f}°"),
            ("Nacelle direction (yaw)", f"{r['nacelle_dir']:.0f}°"), ("Yaw error", f"{r['yaw_error']:+.1f}°"),
            ("Ambient temperature", f"{r['ambient']:.1f} °C"), ("Rotor speed", f"{r['rotor_rpm']:.1f} rpm"),
            ("Generator speed", f"{r['gen_rpm']:.0f} rpm"), ("Pitch angle", f"{r['pitch']:.1f}°"),
            ("Power (actual)", f"{max(0, r['power']) * rated:.2f} MW"), ("Power (forecast)", f"{max(0, r['expected']) * rated:.2f} MW"),
            ("Gearbox bearing / oil", f"{r['gearbox_bearing']:.1f} / {r['gearbox_oil']:.1f} °C"),
            ("Generator bearing DE / NDE", f"{r['gen_bearing_de']:.1f} / {r['gen_bearing_nde']:.1f} °C"),
            ("Stator winding (mean)", f"{r['stator']:.1f} °C"), ("Transformer (max phase)", f"{r['transformer']:.1f} °C"),
            ("Hydraulic oil", f"{r['hydraulic_oil']:.1f} °C"), ("Converter IGBT (max)", f"{r['converter']:.1f} °C"),
            ("Nacelle temperature", f"{r['nacelle_temp']:.1f} °C")]
    t = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return (f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div style="font-weight:800;font-size:18px">Turbine T-{a}</div>{ui.pill(st_.upper(), col)}</div>'
            f'<div class="k" style="margin:4px 0 8px">Random Forest: <b>{rf}</b> ({100 * r["rf_conf"]:.0f} %) · '
            f'residual check: <b>{r["det_component"]}</b> ({r["det_score"]:.2f} / {sim.DET_THRESHOLD})'
            f'{" · energy index " + format(r["perf_index"], ".2f") if not pd.isna(r.get("perf_index")) else ""}</div>'
            f'<table class="r">{t}</table></div>')


# ------------------------------------------------------------------ pages
def page_live(farm, mode):
    latest = farm.latest()
    rated = farm.rated_mw
    sel = st.session_state.get("selected", next(iter(latest)))
    total = sum(max(0, r["power"]) for r in latest.values()) * rated
    exp = sum(max(0, r["expected"]) for r in latest.values()) * rated
    n_fault = sum(status_of(r, mode) in ("fault", "tripped") for r in latest.values())
    act = [f for f in farm.injections if f["end"] > farm.now]
    ui.kpis(st, [("Simulated time", f"{pd.Timestamp(farm.now):%d %b %Y %H:%M}", f"weather: {farm.weather.scenario}"),
                 ("Farm power now", f"{total:.2f} MW", f"forecast {exp:.2f} MW"),
                 ("Farm energy (session)", f"{sum(e['actual'] for e in farm.energy.values()):.1f} MWh",
                  f"expected {sum(e['expected'] for e in farm.energy.values()):.1f} MWh"),
                 ("Turbines with a fault", f"{n_fault} / {len(latest)}", f"{len(act)} injected anomaly(ies) active")])
    c1, c2 = st.columns([2.1, 1])
    with c1:
        r0 = latest[sel]
        tb = [dict(id=int(a), label=str(a), x=paths.LAYOUT[a][0], y=paths.LAYOUT[a][1], yaw=float(r["nacelle_dir"]),
                   rpm=float(r["rotor_rpm"]), status=status_of(r, mode), tripped=bool(r["tripped"]),
                   power_pct=float(max(0, r["power"]) * 100), selected=(a == sel)) for a, r in latest.items()]
        val = farm3d(turbines=tb, wind_dir=float(r0["wind_dir"]), wind_speed=float(np.mean([r["wind"] for r in latest.values()])),
                     height=470, title=f"Wind Farm A · {len(tb)} turbines · {pd.Timestamp(farm.now):%d %b %Y %H:%M}",
                     key="farm3d", default=None)
        if val and val.get("t") != st.session_state.get("last_click"):
            st.session_state["last_click"] = val.get("t")
            st.session_state["selected"] = int(val["selected"])
            sel = int(val["selected"])
        pick = st.radio("Selected turbine", list(latest), index=list(latest).index(sel), horizontal=True,
                        format_func=lambda a: f"T-{a}", key=f"sel_radio_{sel}")
        if pick != sel:
            st.session_state["selected"] = sel = pick
    with c2:
        ui.html(st, readings_html(sel, latest[sel], mode, rated))
    d = farm.frame(sel).tail(48 * 6)
    k = rated
    fig = go.Figure()
    band = 1.28 * farm.turbines[sel].power_sigma * k
    fig.add_trace(go.Scatter(x=d["time_stamp"], y=(d["expected"] * k + band), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["time_stamp"], y=(d["expected"] * k - band).clip(lower=0), line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(47,128,237,0.15)", name="error band (80 %)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["time_stamp"], y=d["power"].clip(lower=0) * k, name="actual", line=dict(color=ui.TEAL, width=2)))
    fig.add_trace(go.Scatter(x=d["time_stamp"], y=d["expected"] * k, name="forecast (twin)", line=dict(color=ui.BLUE, width=2, dash="dash")))
    for f in farm.injections:
        if f["turbine"] == sel and f["end"] > d["time_stamp"].min():
            fig.add_vrect(x0=f["start"], x1=min(f["end"], farm.now), fillcolor=ui.RED, opacity=0.08, line_width=0,
                          annotation_text=f["type"], annotation_position="top left", annotation_font=dict(size=10, color=ui.RED))
    st.plotly_chart(ui.style(fig, f"T-{sel}: actual vs forecast power (last 48 h)", 330, "MW"), use_container_width=True, key="live_power")
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for c, n, colr in (("gearbox_bearing", "gearbox bearing", ui.BLUE), ("gen_bearing_de", "gen. bearing DE", ui.PURPLE),
                           ("transformer", "transformer", ui.AMBER), ("hydraulic_oil", "hydraulic oil", ui.RED), ("converter", "converter", ui.TEAL)):
            fig.add_trace(go.Scatter(x=d["time_stamp"], y=d[c], name=n, line=dict(color=colr, width=1.6)))
        st.plotly_chart(ui.style(fig, f"T-{sel}: component temperatures", 300, "°C"), use_container_width=True, key="live_temps")
    with c4:
        fig = go.Figure()
        for a in farm.turbines:
            dd = farm.frame(a).tail(48 * 6)
            fig.add_trace(go.Scatter(x=dd["time_stamp"], y=dd["power"].clip(lower=0) * k, name=f"T-{a}", stackgroup="farm",
                                     line=dict(width=0.5, color=ui.TURBINE_COL.get(a, ui.GREY))))
        st.plotly_chart(ui.style(fig, "Farm power (stacked)", 300, "MW"), use_container_width=True, key="live_farm")


def page_anomalies(farm, mode):
    ui.title(st, "Anomalies", "Inject an anomaly into a simulated turbine and watch the twin detect and classify it")
    with st.expander("💉 Anomaly injection", expanded=True):
        c1, c2, c3 = st.columns([1, 2, 1])
        t = c1.selectbox("Turbine", list(farm.turbines), format_func=lambda a: f"T-{a}")
        kind = c2.selectbox("Anomaly type (12)", list(sim.ANOMALIES))
        sev = c3.selectbox("Severity", list(sim.SEVERITY), index=1)
        c4, c5, c6 = st.columns([1, 1, 1])
        dev = c4.selectbox("How it develops", list(sim.DEVELOPMENT), index=1)
        dur = c5.slider("Duration (hours)", 6, 120, 48, 6)
        spec = sim.ANOMALIES[kind]
        c6.markdown(f"<div style='font-size:12.5px;color:{ui.MUTED};margin-top:6px'><b>Component:</b> {spec['comp']}<br>"
                    f"<b>Trained RF class:</b> {sim.CLASS_LABEL.get(spec['cls'], 'none (not in the data)')}<br>"
                    f"<b>Cause:</b> {spec['cause']}</div>", unsafe_allow_html=True)
        if st.button("Inject anomaly", type="primary"):
            farm.inject(t, kind, sev, dev, dur)
            st.success(f"Injected '{kind}' ({sev}, {dev.lower()}, {dur} h) into T-{t} at {pd.Timestamp(farm.now):%d %b %H:%M}. "
                       f"Press Play (sidebar) to let the simulation run.")
    latest = farm.latest()
    rows = []
    for a, r in latest.items():
        rows.append({"turbine": f"T-{a}", "status": status_of(r, mode).upper(),
                     "Random Forest class": sim.CLASS_LABEL.get(r["rf_class"], r["rf_class"]),
                     "RF confidence": f"{100 * r['rf_conf']:.0f} %", "RF alarm": bool(r["rf_alarm"]),
                     "residual component": r["det_component"], "residual score": round(float(r["det_score"]), 2),
                     "residual alarm": bool(r["det_alarm"]), "energy index": None if pd.isna(r.get("perf_index")) else round(float(r["perf_index"]), 2),
                     "timestamp": f"{pd.Timestamp(r['time_stamp']):%d %b %H:%M}", "likely cause": cause_for(r) if r["det_alarm"] else "-",
                     "injected": r["injected"] or "-"})
    st.subheader("Current state of every turbine")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.subheader("Injected anomalies: detection and energy lost")
    it = results_dashboard.injection_table(farm)
    if it is not None:
        st.dataframe(it, hide_index=True, use_container_width=True)
        active = [f for f in farm.injections if f["end"] > farm.now]
        if active:
            stop = st.selectbox("Stop an active anomaly (repair)", [f["id"] for f in active],
                                format_func=lambda i: next(f"#{f['id']} {f['type']} on T-{f['turbine']}" for f in active if f["id"] == i))
            if st.button("🔧 Repair now"):
                farm.stop(stop)
    else:
        st.info("No anomaly injected yet.")
    st.subheader("Detection log")
    if farm.events:
        ev = pd.DataFrame(farm.events).sort_values("time", ascending=False)
        ev["time"] = pd.to_datetime(ev["time"]).dt.strftime("%d %b %H:%M")
        ev["turbine"] = "T-" + ev["turbine"].astype(str)
        ev["confidence / score"] = ev["confidence"].round(2)
        ev["injected"] = ev["injected"].where(ev["injected"] != "", "none (false alarm?)")
        st.dataframe(ev[["time", "turbine", "detector", "result", "confidence / score", "injected", "cause"]],
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No detections yet.")
    sel = st.session_state.get("selected", next(iter(latest)))
    d = farm.frame(sel).tail(72 * 6)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["time_stamp"], y=d["det_score"], name="residual score", line=dict(color=ui.TEAL, width=2)))
    fig.add_trace(go.Scatter(x=d["time_stamp"], y=d["rf_conf"] * np.where(d["rf_class"] != "normal", 1, 0) * 5,
                             name="RF fault confidence ×5", line=dict(color=ui.PURPLE, width=1.5)))
    fig.add_hline(y=sim.DET_THRESHOLD, line=dict(color=ui.INK, dash="dash", width=1), annotation_text="residual alarm level")
    for f in farm.injections:
        if f["turbine"] == sel and f["end"] > d["time_stamp"].min():
            fig.add_vrect(x0=f["start"], x1=min(f["end"], farm.now), fillcolor=ui.RED, opacity=0.08, line_width=0)
    st.plotly_chart(ui.style(fig, f"T-{sel}: detector signals (last 72 h)", 300), use_container_width=True, key="anom_sig")


def page_forecasting():
    r = models_io.json_or_none(os.path.join(paths.FORECAST_DIR, "results", "forecasting_results.json"))
    m = models_io.json_or_none(os.path.join(paths.POWER_OUT, "metrics.json"))
    ui.title(st, "Forecasting details", "How the twin predicts power from the weather - step by step")
    st.markdown("### 1 · Data and preprocessing")
    st.markdown("""
* **Source:** CARE to Compare, Wind Farm A - 5 turbines, 22 datasets, 10-minute SCADA averages (≈1.2 million rows).
* **Signals used:** wind speed (`wind_speed_3_avg`), grid power (`power_30_avg`, scaled 0-1 by the data provider),
  ambient temperature (`sensor_0_avg`), wind direction (`sensor_1_avg`).
* **Cleaning:** only *normal operation* (status 0 / 2) - stops, curtailment and service removed;
  rows of overlapping datasets counted once; only the *Avg* signals (Min/Max/Std are unreliable in this dataset).
* **Split:** trained on the training periods, tested on the **unseen** prediction periods of the healthy datasets.""")
    st.markdown("### 2 · Features and their importance")
    st.markdown("""
* **Air density** ρ = p / (R·T) from the ambient temperature, and the **density-corrected wind** v·(ρ/1.225)^(1/3) (IEC 61400-12-1).
* v, v², v³ (power ∝ v³), gustiness (1-h std of wind), wind change, wind direction (sin/cos), time of day and season (sin/cos).""")
    fi = pd.Series((r or {}).get("feature_importance") or (m or {}).get("feature_importance") or {}).sort_values()
    if len(fi):
        fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=ui.TEAL))
        fig.update_xaxes(type="log")
        st.plotly_chart(ui.style(fig, "Permutation importance: drop in R² when the feature is shuffled (log scale)", 380), use_container_width=True, key="fi")
        st.caption("The density-corrected wind speed carries almost all the information - exactly what the physics (P = ½ρAv³Cp) says.")
    st.markdown("### 3 · Model choice and why")
    st.markdown("""
| Model | Why |
|---|---|
| **Physics power curve** (IEC 61400-12 binned curve on density-corrected wind) | transparent baseline every engineer trusts |
| **Gradient Boosting** (HistGradientBoostingRegressor, *used by the twin*) | learns the non-linear curve **and** small effects (density, gustiness, direction); fast; handles missing values; no scaling needed |

A neural network was not needed: with one dominant input the gain would be small and the model harder to explain.""")
    st.markdown("### 4 · Training and test results (unseen data)")
    if m:
        ml, ph = m["ml_gradient_boosting"], m["physics_power_curve"]
        mape = (r or {}).get("ml_gradient_boosting", {}).get("MAPE_%")
        mape_p = (r or {}).get("physics_power_curve", {}).get("MAPE_%")
        ui.kpis(st, [("MAE", f"{ml['nMAE_%']:.2f} %", f"of rated · physics {ph['nMAE_%']:.2f} %"),
                     ("RMSE", f"{ml['nRMSE_%']:.2f} %", f"of rated · physics {ph['nRMSE_%']:.2f} %"),
                     ("R²", f"{ml['R2']:.4f}", f"physics {ph['R2']:.4f}"),
                     ("MAPE", f"{mape:.1f} %" if mape else "run evaluate", f"rows > 10 % rated · physics {mape_p:.1f} %" if mape_p else "")])
        st.caption(f"MAE / RMSE / R² on {ml['n']:,} unseen rows (power_model.py). MAPE is only meaningful when the power "
                   "is not near zero, so it is computed on rows above 10 % of rated power "
                   f"({(r or {}).get('test_set', '')}).")
        per = pd.DataFrame(m.get("ml_per_turbine", {})).T
        if len(per):
            st.dataframe(per[["R2", "nRMSE_%", "nMAE_%", "n"]].round(4).rename_axis("turbine"), use_container_width=True)
    st.markdown("### 5 · Predicted vs actual, residuals and power curve")
    tp = os.path.join(paths.POWER_OUT, "test_predictions.csv")
    if os.path.isfile(tp):
        d = pd.read_csv(tp)
        d = d[(d["event_label"].astype(str).str.lower() == "normal") & d["status_type_id"].isin([0, 2])]
        d = d.sample(min(len(d), 6000), random_state=0)
        k = 100 / (m or {}).get("rated_power_estimate", 1.0)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scattergl(x=d["power"] * k, y=d["pred_ml"] * k, mode="markers", marker=dict(size=3, color=ui.BLUE, opacity=0.35), name="rows"))
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], line=dict(color=ui.INK, dash="dash"), name="perfect"))
            fig.update_layout(hovermode="closest")
            st.plotly_chart(ui.style(fig, "Predicted vs actual (% of rated)", 360, "predicted", "actual"), use_container_width=True, key="pva")
        with c2:
            res = (d["pred_ml"] - d["power"]) * k
            fig = go.Figure(go.Histogram(x=res, nbinsx=80, marker_color=ui.TEAL))
            st.plotly_chart(ui.style(fig, "Residuals (predicted - actual, % of rated)", 360), use_container_width=True, key="res")
        c3, c4 = st.columns(2)
        with c3:
            b = d.assign(bin=(d["wind"] // 1).clip(upper=24), r=(d["pred_ml"] - d["power"]) * k).groupby("bin")["r"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=b.mean().index, y=b.quantile(0.9), line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=b.mean().index, y=b.quantile(0.1), line=dict(width=0), fill="tonexty", fillcolor="rgba(47,128,237,0.15)", name="80 % of errors"))
            fig.add_trace(go.Scatter(x=b.mean().index, y=b.mean(), line=dict(color=ui.BLUE, width=2), name="mean error"))
            st.plotly_chart(ui.style(fig, "Residual vs wind speed", 330, "% of rated", "wind speed (m/s)"), use_container_width=True, key="rvw")
        with c4:
            g = d.assign(bin=(d["wind"] * 2).round() / 2).groupby("bin")
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=d["wind"], y=d["power"] * k, mode="markers", marker=dict(size=2, color=ui.GREY, opacity=0.3), name="measured"))
            fig.add_trace(go.Scatter(x=g["pred_phys"].mean().index, y=g["pred_phys"].mean() * k, line=dict(color=ui.AMBER, width=2.5), name="physics curve"))
            fig.add_trace(go.Scatter(x=g["pred_ml"].mean().index, y=g["pred_ml"].mean() * k, line=dict(color=ui.BLUE, width=2.5, dash="dash"), name="ML model"))
            fig.update_layout(hovermode="closest")
            st.plotly_chart(ui.style(fig, "Power curve (unseen data)", 330, "% of rated", "wind speed (m/s)"), use_container_width=True, key="pc")
    st.markdown("### 6 · How to explain it in the viva")
    st.markdown("""
* *"The twin knows what the turbine **should** produce for the current weather. Wind speed, corrected for air density,
  explains almost everything (R² ≈ 0.995); the ML model adds the small effects the power curve cannot."*
* *"The difference between actual and forecast power is a health signal: a sustained shortfall means
  under-performance (pitch, yaw, blades) - that is what the energy performance monitor in the twin uses."*
* Errors are largest between cut-in and rated wind (steep part of the curve), smallest at rated power.""")


def page_help():
    ui.title(st, "Help", "What each part of the digital twin does")
    st.markdown(open(os.path.join(paths.ROOT, "README.md"), encoding="utf-8").read() if os.path.isfile(os.path.join(paths.ROOT, "README.md")) else "")


# ------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Wind Farm Digital Twin", page_icon="🌬️", layout="wide")
    st.markdown(ui.CSS, unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='font-weight:800;font-size:19px;color:#0B3B5C'>🌬️ Wind Farm</div>"
                        f"<div style='font-weight:700;font-size:16px;color:{ui.TEAL};margin-bottom:10px'>Digital Twin</div>",
                        unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Simulation**")
    playing = st.sidebar.toggle("▶ Play", key="playing")
    speed = st.sidebar.select_slider("Speed (10-min steps per second)", [1, 3, 6, 12, 36], value=6)
    scenario = st.sidebar.selectbox("Weather", list(sim.SCENARIOS))
    rated = st.sidebar.number_input("Rated power per turbine (MW)", 0.5, 10.0, 2.0, 0.1,
                                    help="Power in the dataset is scaled 0-1; this converts it to MW / MWh")
    mode = st.sidebar.selectbox("Status colours from", ["Random Forest (Step 1 model C)", "Residual detector + energy monitor",
                                                        "Either detector"],
                                help="Which detector decides normal / warning / fault in the 3D view and the tables")
    if page in (PAGES[2], PAGES[4]) and "farm" not in st.session_state:
        farm = None
    else:
        farm = get_farm(rated, scenario)
    c1, c2 = st.sidebar.columns(2)
    if farm is not None and c1.button("⏭ +6 h"):
        farm.step(36)
    if c2.button("↺ Reset"):
        st.session_state.pop("farm", None)
        st.rerun()
    st.sidebar.caption("Data: CARE to Compare, Wind Farm A. Simulated turbines imitate the real assets 0, 10, 11, 13, 21.")

    live = page in LIVE and farm is not None

    def body():
        if playing and live:
            farm.step(speed)
        if page == PAGES[0]:
            page_live(farm, mode)
        elif page == PAGES[1]:
            page_anomalies(farm, mode)
        elif page == PAGES[2]:
            page_forecasting()
        elif page == PAGES[3]:
            results_dashboard.render(st, go, farm)
        else:
            page_help()

    frag = getattr(st, "fragment", None)
    if frag is not None:
        frag(run_every=FRAME_SECONDS if (playing and live) else None)(body)()
    else:
        body()
        if playing and live:
            time.sleep(FRAME_SECONDS)
            st.rerun()


if __name__ == "__main__":
    main()
