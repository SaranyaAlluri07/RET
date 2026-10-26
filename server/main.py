"""WindTwin API + web app.

    python -m uvicorn server.main:app --port 8000        (from the project folder)
    -> http://127.0.0.1:8000

Read-only endpoints serve the artifacts written by windtwin.benchmark / windtwin.replay / windtwin.forecast.
The /api/sim/* endpoints drive one synthetic simulation session (server/sim.py).
"""
import gzip
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
WEB = ROOT / "web"

app = FastAPI(title="WindTwin", default_response_class=ORJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=2048)
_lock = threading.Lock()
_sim = None


def _json(path: Path):
    if not path.is_file():
        raise HTTPException(404, f"{path.relative_to(ROOT)} not found - run the pipeline (see README)")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ read-only
@app.get("/api/meta")
def meta():
    from windtwin import config
    from server.sim import ASSETS
    return {"dataset": {"name": "CARE to Compare - Wind Farm A", "record": config.DATASET_RECORD,
                        "doi": config.DATASET_DOI, "version": config.DATASET_VERSION},
            "pipeline_version": config.PIPELINE_VERSION,
            "assets": [{"id": a, "label": f"T-{a:02d}", "x": x, "y": y} for a, (x, y) in ASSETS.items()],
            "modes": ["replay", "simulation"], "rated_mw_default": 2.0}


@app.get("/api/benchmark")
def benchmark():
    return _json(ART / "benchmark" / "benchmark.json")


@app.get("/api/benchmark/{method}")
def benchmark_method(method: str):
    if not method.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "bad method id")
    return _json(ART / "benchmark" / f"{method}.json")


@app.get("/api/forecast")
def forecast():
    return _json(ART / "forecast" / "forecast.json")


@app.get("/api/datasets")
def datasets():
    return _json(ART / "datasets.json")


@app.get("/api/replay/{ds}")
def replay(ds: int):
    return _json(ART / "replay" / f"{ds}.json.gz")


# ------------------------------------------------------------------ simulation
class ResetReq(BaseModel):
    seed: int | None = None
    weather: str | None = None


class StepReq(BaseModel):
    steps: int = 6


class InjectReq(BaseModel):
    turbine: int
    scenario: str
    severity: str = "moderate"
    ramp_h: float = 24
    duration_h: float = 48
    seed: int | None = None


class RepairReq(BaseModel):
    injection_id: int


def sim():
    global _sim
    if _sim is None:
        from server.sim import FarmSim
        _sim = FarmSim()
    return _sim


@app.get("/api/sim/state")
def sim_state():
    with _lock:
        return sim().state()


@app.post("/api/sim/reset")
def sim_reset(r: ResetReq):
    with _lock:
        return sim().reset(seed=r.seed, weather=r.weather)


@app.post("/api/sim/step")
def sim_step(r: StepReq):
    with _lock:
        return sim().step(max(1, min(r.steps, 144)))


@app.post("/api/sim/inject")
def sim_inject(r: InjectReq):
    with _lock:
        s = sim()
        try:
            inj = s.inject(r.turbine, r.scenario, r.severity, r.ramp_h, r.duration_h, r.seed)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))
        return {"injection": inj, "state": s.state()}


@app.post("/api/sim/repair")
def sim_repair(r: RepairReq):
    with _lock:
        return sim().repair(r.injection_id)


@app.get("/api/sim/scenarios")
def sim_scenarios():
    from windtwin import scenarios
    return scenarios.catalog()


@app.get("/api/sim/turbine/{tid}")
def sim_turbine(tid: int, hours: float = 48):
    with _lock:
        try:
            return sim().series(tid, hours)
        except KeyError:
            raise HTTPException(404, f"turbine {tid} not in the simulation")


@app.get("/api/sim/cases")
def sim_cases():
    with _lock:
        return sim().cases()


@app.get("/api/sim/injections")
def sim_injections():
    with _lock:
        return sim().injections_out()


# ------------------------------------------------------------------ web app
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
