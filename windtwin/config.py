"""WindTwin configuration: dataset version, paths, signal map, subsystems and fixed design constants."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CARE_To_Compare"
FARM_DIR = DATA_DIR / "Wind Farm A"
DATASETS_DIR = FARM_DIR / "datasets"
EVENT_INFO = FARM_DIR / "event_info.csv"
FEATURE_DESC = FARM_DIR / "feature_description.csv"

ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
REPLAY_DIR = ARTIFACTS / "replay"
BENCH_DIR = ARTIFACTS / "benchmark"
FORECAST_DIR = ARTIFACTS / "forecast"
CACHE_DIR = ARTIFACTS / "cache"

LEGACY_DIR = ROOT / "anomaly_detection_model"
LEGACY_SCORES = LEGACY_DIR / "outputs_fault_detection_v2" / "scores"
LEGACY_POWER_MODEL = ROOT / "forecasting_model" / "power_model.py"

try:
    _v = json.loads((DATA_DIR / "VERSION.json").read_text())
except (OSError, ValueError):
    _v = {}
DATASET_RECORD = str(_v.get("record", "15846963"))
DATASET_DOI = _v.get("doi", "10.5281/zenodo.15846963")
DATASET_VERSION = _v.get("version", 6)
PIPELINE_VERSION = "windtwin-2.0.0"

STEPS_PER_HOUR = 6
STEP_MINUTES = 10
NORMAL_STATUS = (0, 2)                  # training filter only (Farm A status comes from the logbook)
CONTEXT_STEPS = 2 * 24 * STEPS_PER_HOUR  # monitoring starts 2 days before the prediction window
WARMUP_STEPS = 300                       # history rows for rolling features + flatline checks (>= 288)
VAL_FRACTION = 0.2                      # chronological: last 20 % of normal train rows = validation

# turbine assets and the healthy dataset each simulated turbine is built from
ASSETS = [0, 10, 11, 13, 21]
SIM_DATASETS = {0: 24, 10: 3, 11: 25, 13: 14, 21: 13}

# ------------------------------------------------------------------ raw signals (Avg only)
WIND, WIND_EST, POWER, AMBIENT = "wind_speed_3_avg", "wind_speed_4_avg", "power_30_avg", "sensor_0_avg"
NACELLE, GEN_RPM, ROTOR_RPM = "sensor_43_avg", "sensor_18_avg", "sensor_52_avg"
WIND_DIR, REL_DIR, NACELLE_DIR, PITCH_RAW = "sensor_1_avg", "sensor_2_avg", "sensor_42_avg", "sensor_5_avg"

# key -> (label, unit, component or None, kind). kind drives quality ranges and sigma floors.
SIGNALS = {
    AMBIENT: ("Ambient temperature", "°C", None, "temp"),
    WIND_DIR: ("Wind direction", "°", None, "angle"),
    REL_DIR: ("Wind relative direction (yaw error)", "°", None, "angle"),
    WIND: ("Wind speed", "m/s", None, "wind"),
    WIND_EST: ("Estimated wind speed", "m/s", None, "wind"),
    PITCH_RAW: ("Pitch angle (raw)", "°", None, "angle"),
    "sensor_6_avg": ("Hub controller temp", "°C", "Hub / pitch", "temp"),
    "sensor_7_avg": ("Top nacelle controller temp", "°C", "Nacelle cabinets", "temp"),
    "sensor_8_avg": ("Choke coils temp (VCS)", "°C", "Nacelle cabinets", "temp"),
    "sensor_9_avg": ("VCP-board temp", "°C", "Nacelle cabinets", "temp"),
    "sensor_10_avg": ("VCS cooling water temp", "°C", "Converter", "temp"),
    "sensor_11_avg": ("Gearbox HSS bearing temp", "°C", "Gearbox", "temp"),
    "sensor_12_avg": ("Gearbox oil temp", "°C", "Gearbox", "temp"),
    "sensor_13_avg": ("Gen. bearing DE temp", "°C", "Generator bearings", "temp"),
    "sensor_14_avg": ("Gen. bearing NDE temp", "°C", "Generator bearings", "temp"),
    "sensor_15_avg": ("Stator winding L1 temp", "°C", "Generator windings", "temp"),
    "sensor_16_avg": ("Stator winding L2 temp", "°C", "Generator windings", "temp"),
    "sensor_17_avg": ("Stator winding L3 temp", "°C", "Generator windings", "temp"),
    GEN_RPM: ("Generator speed", "rpm", None, "gen_rpm"),
    "sensor_19_avg": ("Split ring chamber temp", "°C", "Nacelle cabinets", "temp"),
    "sensor_20_avg": ("Busbar temp", "°C", "Nacelle cabinets", "temp"),
    "sensor_21_avg": ("Grid-side IGBT temp", "°C", "Converter", "temp"),
    "sensor_22_avg": ("Phase displacement", "-", None, "ratio"),
    "sensor_23_avg": ("Phase current L1", "A", "Electrical balance", "current"),
    "sensor_24_avg": ("Phase current L2", "A", "Electrical balance", "current"),
    "sensor_25_avg": ("Phase current L3", "A", "Electrical balance", "current"),
    "sensor_26_avg": ("Grid frequency", "Hz", None, "freq"),
    "power_29_avg": ("Possible grid active power", "pu", None, "power"),
    POWER: ("Grid power", "pu", "Power performance", "power"),
    "sensor_32_avg": ("Phase voltage L1", "V", "Electrical balance", "voltage"),
    "sensor_33_avg": ("Phase voltage L2", "V", "Electrical balance", "voltage"),
    "sensor_34_avg": ("Phase voltage L3", "V", "Electrical balance", "voltage"),
    "sensor_35_avg": ("Rotor-side IGBT L1 temp", "°C", "Converter", "temp"),
    "sensor_36_avg": ("Rotor-side IGBT L2 temp", "°C", "Converter", "temp"),
    "sensor_37_avg": ("Rotor-side IGBT L3 temp", "°C", "Converter", "temp"),
    "sensor_38_avg": ("HV transformer L1 temp", "°C", "Transformer", "temp"),
    "sensor_39_avg": ("HV transformer L2 temp", "°C", "Transformer", "temp"),
    "sensor_40_avg": ("HV transformer L3 temp", "°C", "Transformer", "temp"),
    "sensor_41_avg": ("Hydraulic oil temp", "°C", "Hydraulic group", "temp"),
    NACELLE_DIR: ("Nacelle direction", "°", None, "angle"),
    NACELLE: ("Nacelle temperature", "°C", None, "temp"),
    ROTOR_RPM: ("Rotor speed", "rpm", None, "rotor_rpm"),
    "sensor_53_avg": ("Nose cone temp", "°C", "Hub / pitch", "temp"),
}

# derived physics signals: key -> (label, unit, kind, raw sensors it is built from)
DERIVED = {
    "pitch": ("Pitch angle", "°", "pitch", [PITCH_RAW]),
    "gen_bearing_diff": ("Gen. bearing DE − NDE", "°C", "tdiff", ["sensor_13_avg", "sensor_14_avg"]),
    "gbx_bearing_minus_oil": ("Gearbox bearing − oil", "°C", "tdiff", ["sensor_11_avg", "sensor_12_avg"]),
    "hydraulic_minus_nacelle": ("Hydraulic oil − nacelle", "°C", "tdiff", ["sensor_41_avg", NACELLE]),
    "stator_spread": ("Stator phase spread", "°C", "tdiff", ["sensor_15_avg", "sensor_16_avg", "sensor_17_avg"]),
    "transformer_spread": ("Transformer phase spread", "°C", "tdiff", ["sensor_38_avg", "sensor_39_avg", "sensor_40_avg"]),
    "igbt_spread": ("Rotor-side IGBT phase spread", "°C", "tdiff", ["sensor_35_avg", "sensor_36_avg", "sensor_37_avg"]),
    "current_imbalance": ("Phase current imbalance", "-", "ratio", ["sensor_23_avg", "sensor_24_avg", "sensor_25_avg"]),
    "voltage_imbalance": ("Phase voltage imbalance", "-", "ratio", ["sensor_32_avg", "sensor_33_avg", "sensor_34_avg"]),
}

# ------------------------------------------------------------------ monitored components
SUBSYSTEMS = {
    "Gearbox": ["sensor_11_avg", "sensor_12_avg", "gbx_bearing_minus_oil"],
    "Generator bearings": ["sensor_13_avg", "sensor_14_avg", "gen_bearing_diff"],
    "Generator windings": ["sensor_15_avg", "sensor_16_avg", "sensor_17_avg", "stator_spread"],
    "Transformer": ["sensor_38_avg", "sensor_39_avg", "sensor_40_avg", "transformer_spread"],
    "Hydraulic group": ["sensor_41_avg", "hydraulic_minus_nacelle"],
    "Hub / pitch": ["sensor_6_avg", "sensor_53_avg", "pitch"],
    "Converter": ["sensor_21_avg", "sensor_35_avg", "sensor_36_avg", "sensor_37_avg", "igbt_spread", "sensor_10_avg"],
    "Nacelle cabinets": ["sensor_7_avg", "sensor_8_avg", "sensor_9_avg", "sensor_19_avg", "sensor_20_avg"],
    "Electrical balance": ["current_imbalance", "voltage_imbalance"],
    "Power performance": [POWER],
}
COMPONENTS = list(SUBSYSTEMS)
TARGETS = [t for ts in SUBSYSTEMS.values() for t in ts]
TARGET_COMPONENT = {t: c for c, ts in SUBSYSTEMS.items() for t in ts}
UNKNOWN = "Unknown — insufficient evidence"

# residual direction: +1 = higher than expected is bad, -1 = lower is bad, 0 = both directions
DIRECTION = {t: 1 for t in TARGETS}
DIRECTION.update({POWER: -1, "pitch": 0, "gen_bearing_diff": 0, "voltage_imbalance": 1})

# event description keyword -> components accepted as a correct diagnosis
FAULT_COMPONENTS = [
    ("generator bearing", ["Generator bearings"]),
    ("gearbox", ["Gearbox"]),
    ("transformer", ["Transformer"]),
    ("hydraulic", ["Hydraulic group", "Hub / pitch"]),   # hydraulic pitch system sits in the hub
]


def fault_components(description):
    d = str(description or "").lower()
    return next((c for k, c in FAULT_COMPONENTS if k in d), [])


def fault_class(description):
    """Primary component of a logged fault (first accepted component) or None."""
    c = fault_components(description)
    return c[0] if c else None


def label_of(key):
    if key in SIGNALS:
        return SIGNALS[key][0]
    if key in DERIVED:
        return DERIVED[key][0]
    return key


def unit_of(key):
    if key in SIGNALS:
        return SIGNALS[key][1]
    if key in DERIVED:
        return DERIVED[key][1]
    return ""


def kind_of(key):
    if key in SIGNALS:
        return SIGNALS[key][3]
    if key in DERIVED:
        return DERIVED[key][2]
    return "other"


def raw_sources(key):
    """Raw sensor columns a (possibly derived) signal depends on."""
    return DERIVED[key][3] if key in DERIVED else [key]


# ------------------------------------------------------------------ quality limits by kind
# (low, high) physically plausible range; max |step change| per 10 min; flatline steps
RANGES = {"temp": (-40, 160), "tdiff": (-60, 60), "wind": (0, 45), "power": (-0.15, 1.3),
          "gen_rpm": (-50, 2200), "rotor_rpm": (-2, 25), "pitch": (-15, 100), "angle": (-360, 720),
          "current": (-10, 3000), "voltage": (300, 480), "ratio": (-1, 5), "freq": (45, 55)}
MAX_JUMP = {"temp": 20, "wind": 20, "power": 1.2, "gen_rpm": 2000, "rotor_rpm": 20, "voltage": 60}
FLATLINE_STEPS = {"temp": 288, "wind": 18, "voltage": 36, "current": 36}  # 48 h / 3 h / 6 h

# sigma floors per kind (residual units), protects against over-confident z on quantized signals
SIGMA_FLOOR = {"temp": 0.5, "tdiff": 0.5, "power": 0.015, "pitch": 0.3, "ratio": 0.003}

# operating-state thresholds (power in p.u. of rated)
P_PRODUCTION = 0.02
P_STOPPED = 0.005
RPM_STOPPED = 1.0
