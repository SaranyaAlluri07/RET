"""
Builds USER_MANUAL.pdf (bundle root).
To add your own screenshots of the running app, save them as docs/screenshots/app_<anything>.png and run:
    python docs/make_manual.py
"""
import glob
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHOTS = os.path.join(HERE, "screenshots")
OUT = os.path.join(ROOT, "USER_MANUAL.pdf")
TEAL = colors.HexColor("#0E9F8E")
INK = colors.HexColor("#1F2D3D")
MUTED = colors.HexColor("#6B7C93")
LINE = colors.HexColor("#D9E3EC")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=INK, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=TEAL, spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("P", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.2, textColor=INK, alignment=TA_LEFT)
SMALL = ParagraphStyle("S", parent=P, fontSize=8.3, leading=11, textColor=MUTED)
CODE = ParagraphStyle("C", parent=P, fontName="Courier", fontSize=8.8, leading=11.5, backColor=colors.HexColor("#F2F6F9"),
                      borderPadding=5, leftIndent=4, spaceBefore=3, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=P, fontSize=8.4, leading=10.6)


def p(t):
    return Paragraph(t, P)


def bullets(items):
    return [Paragraph("&bull; " + i, P) for i in items]


def code(t):
    return Paragraph(t.replace("\n", "<br/>").replace(" ", "&nbsp;"), CODE)


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F5F2"))]
    t.setStyle(TableStyle(st))
    return t


def shot(name, caption, width=16.5):
    path = os.path.join(SHOTS, name)
    if not os.path.isfile(path):
        return []
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = width * cm
    return [KeepTogether([Image(path, width=w, height=w * ih / iw), Paragraph(caption, SMALL), Spacer(1, 8)])]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Wind Farm Digital Twin - User manual")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


def build():
    s = []
    s += [Spacer(1, 2.5 * cm), Paragraph("Wind Farm Digital Twin", ParagraphStyle("T", parent=H1, fontSize=28, leading=34)),
          Paragraph("User manual", ParagraphStyle("T2", parent=H1, fontSize=18, textColor=TEAL)), Spacer(1, 0.6 * cm),
          p("Power forecasting, anomaly detection and classification, a live 3D simulation of 5 turbines with anomaly "
            "injection, and a results dashboard - built on the CARE to Compare SCADA dataset (Wind Farm A)."), Spacer(1, 0.6 * cm)]
    s += shot("farm3d_live.png", "The 3D wind farm view of the app (rendered by the app's own 3D component from a live simulation): "
                                 "T-10 stopped by its protection after an injected severe gearbox fault, T-21 in fault with a yaw misalignment "
                                 "(rotor turned away from the wind).")
    s += [Spacer(1, 0.4 * cm), Paragraph("Contents", H2)]
    s += bullets(["1  What is in the bundle", "2  Installation", "3  Running each part", "4  The screens",
                  "5  Injecting anomalies", "6  The models and their real-data results", "7  Retraining the models",
                  "8  Troubleshooting", "9  Limitations"])
    s.append(PageBreak())

    s += [Paragraph("1  What is in the bundle", H1)]
    s.append(table([["Folder / file", "Contents"],
                    ["digital_twin/", "the app (app.py), simulator and live twin (twin_core/), 3D view (components/farm3d/), turbine models (models/, created on the first start)"],
                    ["forecasting_model/", "power_model.py (training), evaluate_forecasting.py (MAE, RMSE, R2, MAPE + figures), trained model in outputs_power_model/, results/"],
                    ["anomaly_detection_model/", "anomaly_classifier.py (Step 1 comparison), anomaly_classifier_v2.py (attempt 2), anomaly_classifier.joblib (Random Forest used by the app), per-class metrics and confusion matrices in results_attempt1/ and results_attempt2/, fault_detection*.py (normal-behaviour models)"],
                    ["dashboard/", "results_dashboard.py (also a page of the app), export_pdf.py, results_dashboard.pdf"],
                    ["data/", "processed/ (extracts used to train the classifier) and README.md (where to put the raw dataset)"],
                    ["requirements.txt, README.md", "packages and quick start"],
                    ["docs/", "this manual's source (make_manual.py) and screenshots"]], [4.3 * cm, 12.2 * cm]))
    s += [Paragraph("All paths are relative to the bundle folder, so the project runs from any location on Windows, macOS or Linux.", SMALL)]

    s += [Paragraph("2  Installation", H1), Paragraph("2.1  Python and packages", H2),
          p("Python 3.10 - 3.13 (Anaconda / Miniconda or python.org). No GPU or special hardware is needed. In a terminal "
            "(Windows: <b>Anaconda Prompt</b>), go to the bundle folder and install:"),
          code("cd path/to/RET_Wind_Farm_Twin\npip install -r requirements.txt"),
          p("With conda you can use your existing environment:"),
          code("conda activate ret\npip install -r requirements.txt"),
          Paragraph("2.2  The raw dataset (once)", H2),
          p("The first start builds the 5 simulated turbines from the real data. Extract the CARE to Compare zip "
            "(Zenodo record 14958989) so that a folder <b>Wind Farm A</b> (with event_info.csv and the dataset CSV files) is "
            "either inside <b>data/</b> or in the folder that <b>contains</b> the bundle (for example your old project folder). "
            "The turbines imitate the healthy datasets 24, 3, 25, 14 and 13 (assets 0, 10, 11, 13 and 21).")]

    s += [Paragraph("3  Running each part", H1)]
    s.append(table([["Part", "Command (from the bundle folder)", "Notes"],
                    ["Digital twin app", "streamlit run digital_twin/app.py", "opens http://localhost:8501; first start 5-8 min (turbines are built once)"],
                    ["Results dashboard alone", "streamlit run dashboard/results_dashboard.py", "static results, PDF export"],
                    ["PDF report", "python dashboard/export_pdf.py", "writes dashboard/results_dashboard.pdf"],
                    ["Forecast evaluation", "python forecasting_model/evaluate_forecasting.py", "MAE, RMSE, R2, MAPE, figures"],
                    ["Prepare turbines manually", "python digital_twin/twin_core/prepare.py", "same as the first start of the app"],
                    ["This manual", "python docs/make_manual.py", "adds docs/screenshots/app_*.png"]],
                   [3.6 * cm, 6.8 * cm, 6.1 * cm]))
    s += [p("Stop the app with <b>Ctrl + C</b> in the terminal.")]

    s += [PageBreak(), Paragraph("4  The screens", H1), Paragraph("4.1  Sidebar (always visible)", H2)]
    s += bullets(["<b>Navigation</b> - the five pages.", "<b>Play</b> - runs the simulation; every second (about) the farm advances.",
                  "<b>Speed</b> - 10-minute steps per second (1, 3, 6, 12 or 36 = 6 hours per second).",
                  "<b>Weather</b> - normal, windy, calm, storm (very high wind, cut-out), hot day, cold day.",
                  "<b>Rated power</b> - MW per turbine (the dataset power is scaled 0-1; used for MW and MWh).",
                  "<b>Status colours from</b> - which detector decides normal / warning / fault: the Random Forest (Step 1 model C, default), "
                  "the residual detector + energy monitor, or either.",
                  "<b>+6 h</b> jumps ahead, <b>Reset</b> starts a new simulation."])
    s += [Paragraph("4.2  Wind farm (live)", H2),
          p("Top: simulated time, farm power now (actual and forecast), energy of the session, number of turbines in fault. "
            "Left: the <b>3D view</b> - rotors spin at the simulated rotor speed, each nacelle points where the turbine is yawed, "
            "the compass shows where the wind comes from; the ring and lamp colour show the status (green normal, amber warning, "
            "red fault, dark red tripped). <b>Drag</b> to rotate, <b>wheel</b> to zoom, <b>click</b> a turbine to select it, "
            "<b>double-click</b> to reset the camera. Right: live readings of the selected turbine - wind, direction, yaw error, "
            "ambient, rotor / generator speed, pitch, actual and forecast power, gearbox, generator bearing, stator, transformer, "
            "hydraulic and converter temperatures, plus the Random Forest class and the residual check. Below: actual vs forecast "
            "power with the 80 % error band (injected anomalies shaded red), component temperatures, and the stacked farm power.")]
    s += shot("live_power.png", "Actual vs forecast power of the selected turbine with the error band (live chart; same data as in the app).")
    s += [Paragraph("4.3  Anomalies", H2),
          p("<b>Anomaly injection</b> panel (section 5). <b>Current state of every turbine</b>: status, Random Forest class and "
            "confidence, its alarm, the residual component and score, the energy performance index, timestamp, likely cause and the "
            "injected anomaly (if any). <b>Injected anomalies</b>: for each injection the class / component found, the detection "
            "delay, energy lost and whether the turbine tripped; an active one can be repaired. <b>Detection log</b>: every alarm "
            "with time, detector, result, confidence, the injection it belongs to (or <i>none - false alarm?</i>) and the likely cause. "
            "Chart: the detector signals of the selected turbine.")]
    s += shot("anomaly_signals.png", "Detector signal of a turbine with an injected severe gearbox fault: the twin's alarm comes about "
                                     "20 hours before the turbine's protection stops it.")
    s += [Paragraph("4.4  Forecasting details", H2),
          p("A step-by-step explanation for the presentation and viva: (1) data and preprocessing, (2) features and their "
            "permutation importance, (3) model choice and why, (4) training / test results - MAE, RMSE, R2 and MAPE with the "
            "physics power curve for comparison and per turbine, (5) predicted vs actual, residual distribution, residual vs wind "
            "speed, power curve, (6) how to explain it.")]
    s += shot("pred_vs_actual.png", "Predicted vs actual power on unseen data.", 8) + shot("power_curve.png", "Power curve: measured, physics curve and ML model.", 9)
    s += [Paragraph("4.5  Results dashboard", H2),
          p("Energy per turbine and for the farm (actual, expected, lost to injected anomalies), forecast metrics, the anomaly "
            "detection metrics on real data (before vs after table, per-class table and confusion matrix of the Random Forest, "
            "attempt 2), the injected anomalies with detection results, energy lost per anomaly, the anomaly history timeline, and "
            "<b>Build PDF report</b> (download). Every chart has a camera icon for a PNG.")]
    s += shot("confusion_C.png", "Confusion matrix of the Random Forest on real data (out-of-fold, leave-one-turbine-out).", 11)
    s += shot("dashboard_pdf-1.png", "First page of the PDF export.", 10)
    extra = sorted(glob.glob(os.path.join(SHOTS, "app_*.png")))
    if extra:
        s += [Paragraph("4.6  Screenshots of the running app", H2)]
        for f in extra:
            s += shot(os.path.basename(f), os.path.splitext(os.path.basename(f))[0].replace("app_", "").replace("_", " "))

    s += [PageBreak(), Paragraph("5  Injecting anomalies", H1),
          p("Page <b>Anomalies</b> -> panel <b>Anomaly injection</b>: choose the turbine, one of the 12 anomaly types, the severity, "
            "how it develops and the duration, then press <b>Inject anomaly</b> and <b>Play</b> (or +6 h). The simulated turbine's "
            "signals change; the twin does not know about the injection and has to find it."), Spacer(1, 4)]
    from twin_core_import import ANOMALIES, CLASS_LABEL                                   # noqa
    rows = [["Anomaly", "Component (residual)", "Random Forest class", "What changes"]]
    for k, v in ANOMALIES.items():
        what = []
        if v.get("temp"):
            what.append("temperature rise: " + ", ".join(t.replace("_avg", "").replace("sensor_", "s") for t in v["temp"]))
        if v.get("pitch"):
            what.append("pitch offset")
        if v.get("yaw"):
            what.append("yaw offset (power x cos^3)")
        if v.get("loss"):
            what.append("power loss")
        rows.append([k, v["comp"], CLASS_LABEL.get(v["cls"], "not a trained class"), "; ".join(what)])
    s.append(table(rows, [4.6 * cm, 3.3 * cm, 3.1 * cm, 5.5 * cm]))
    s += [Spacer(1, 6), table([["Setting", "Meaning"],
                                ["Severity", "Mild 5, Moderate 17, Severe 25 x the sensor's normal noise (temperatures); pitch 1.5 / 3 / 6 deg; yaw 8 / 15 / 30 deg; power loss 6 / 15 / 30 %. Temperature rises grow with the load (0.6 + 0.4 x load)."],
                                ["How it develops", "Sudden, or growing linearly over 1 day / 3 days"],
                                ["Duration", "6 - 120 h; afterwards the anomaly ends (repaired). 'Repair now' ends it at once."],
                                ["Protection trip", "a Severe temperature anomaly at full level for 6 h stops the turbine (rotor stops, 3D view dark red, energy lost); the twin pauses judging a stopped turbine"],
                                ["Energy lost", "sum of (forecast x the turbine's healthy performance level - actual power) while the anomaly is active, x rated MW x 1/6 h"]],
                               [3.3 * cm, 13.2 * cm])]
    s += [Paragraph("What to expect", H2)]
    s += bullets(["<b>Residual detector + energy monitor</b>: on the simulated turbines it found 11 of the 12 anomaly types with the "
                  "correct component (moderate / severe), typically 4 - 25 hours after the start, and gives an early warning before a "
                  "protection trip. Pitch misalignment is often missed (pitch is noisy). Power-loss anomalies need roughly > 15 - 20 % loss.",
                  "<b>Random Forest (Step 1 model C)</b>: as on the real data, it mostly answers 'hydraulic group' and also flags "
                  "healthy turbines; its class is shown but should not be trusted (section 6).",
                  "Synthetic anomalies show how the twin reacts. They are not used to claim accuracy."])

    s += [Paragraph("6  The models and their real-data results", H1), Paragraph("6.1  Forecasting (power from weather)", H2),
          p("Gradient Boosting on density-corrected wind, wind, gustiness, direction, temperature, air density, time of day and "
            "season; physics power curve (IEC 61400-12-1) as baseline. Unseen test data: <b>R2 0.995, RMSE 1.82 %, MAE 1.15 % of "
            "rated</b> (physics: R2 0.993, RMSE 2.12 %). MAPE (rows above 10 % of rated) from evaluate_forecasting.py.")]
    s += [Paragraph("6.2  Anomaly detection and classification (Step 1)", H2),
          p("12 recorded faults of 4 types (hydraulic 6, gearbox 3, generator bearing 2, transformer 1) and 10 healthy datasets. "
            "Leave-one-turbine-out cross-validation, running turbine only, logbook status flags not used as inputs (they mark the "
            "fault windows - using them gives a fake 100 %).")]
    s.append(table([["Approach", "Faults detected", "Correct type", "False alarms (healthy)"],
                    ["A  old model (threshold + diagnosis)", "1/12", "0/12", "0/10"],
                    ["B  Gradient Boosting", "9/12", "1/12", "7/10"],
                    ["C  Random Forest (used in the app)", "8/12", "1/12", "6/10"],
                    ["D  detector + fault-type classifier", "10/12", "3/12", "8/10"],
                    ["E  fault-component detector + physics rule", "2/12", "1/12", "1/10"]],
                   [7.3 * cm, 3 * cm, 2.7 * cm, 3.5 * cm]))
    s += [p("Upper limit: even with settings tuned on all data, at most 3 of 12 faults can be classified correctly - most faults "
            "do not rise above the normal variation of healthy turbines in 10-minute SCADA data, and there are only 1 - 6 events per "
            "type. The twin therefore shows the residual detector + 24-h energy performance monitor next to the Random Forest.")]

    s += [Paragraph("7  Retraining the models", H1)]
    s.append(table([["Model", "Command", "Time"],
                    ["Forecasting model", "python forecasting_model/power_model.py", "~2 min"],
                    ["Forecast evaluation", "python forecasting_model/evaluate_forecasting.py", "< 1 min"],
                    ["Normal-behaviour models (features)", "python anomaly_detection_model/fault_detection.py\npython anomaly_detection_model/fault_detection_v2.py", "15-20 min each"],
                    ["Data extracts for the classifier", "python data/processed/extract_signals.py", "~2 min"],
                    ["Random Forest - final model only", "python anomaly_detection_model/anomaly_classifier.py --final-only", "~1 min"],
                    ["Step 1 full comparison (CV)", "python anomaly_detection_model/anomaly_classifier.py", "~50 min"],
                    ["Attempt 2", "python anomaly_detection_model/anomaly_classifier_v2.py", "~5 min"],
                    ["Simulated turbines", "python digital_twin/twin_core/prepare.py", "1-2 min per turbine"]],
                   [4.6 * cm, 9.2 * cm, 2.7 * cm]))
    s += [p("If a saved model cannot be loaded with your scikit-learn version, the app rebuilds it automatically "
            "(Random Forest: final model only; turbines: prepare).")]

    s += [Paragraph("8  Troubleshooting", H1)]
    s.append(table([["Problem", "Fix"],
                    ["'streamlit' is not recognized", "the environment is not active: conda activate ret (Anaconda Prompt), or use python -m streamlit run digital_twin/app.py"],
                    ["ModuleNotFoundError", "pip install -r requirements.txt in the same environment"],
                    ["'Wind Farm A dataset was not found'", "put the CARE_To_Compare folder into data/ or next to the bundle (data/README.md)"],
                    ["First start takes long", "normal: 5 turbines are built from the real data once (5-8 min); later starts take seconds"],
                    ["Whole page reloads during Play / slow", "update Streamlit to 1.37 or newer; use speed 36 for fast progress; close other heavy programs"],
                    ["3D view is empty", "refresh the browser page (F5); the view needs a browser with canvas support (Chrome, Edge, Firefox)"],
                    ["Many sklearn UserWarning messages", "harmless; they are hidden in the scripts"],
                    ["Port 8501 in use", "streamlit run digital_twin/app.py --server.port 8502"]],
                   [5.2 * cm, 11.3 * cm]))
    s += [Paragraph("9  Limitations", H1)]
    s += bullets(["The 'live' data is simulated from models trained on the real turbines; the turbines are not connected to a real plant.",
                  "Fault types cannot be classified reliably from this data (section 6); accuracy is reported only on real data.",
                  "Power in the dataset is scaled 0-1; MW and MWh use the rated power set in the sidebar (default 2 MW).",
                  "The 3D view is a schematic visualisation (turbines drawn about twice their real size)."])
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Wind Farm Digital Twin - User manual", author="RET project")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("Saved", OUT)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(ROOT, "digital_twin"))
    import types
    from twin_core import simulator
    mod = types.ModuleType("twin_core_import")
    mod.ANOMALIES, mod.CLASS_LABEL = simulator.ANOMALIES, simulator.CLASS_LABEL
    sys.modules["twin_core_import"] = mod
    build()
