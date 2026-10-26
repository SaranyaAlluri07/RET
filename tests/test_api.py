"""Tests for FastAPI backend endpoints (read-only and simulation)."""
import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_meta_endpoint(client):
    res = client.get("/api/meta")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"]["record"] == "15846963"
    assert data["dataset"]["version"] == 6
    assert len(data["assets"]) == 5
    assert "replay" in data["modes"]
    assert "simulation" in data["modes"]


def test_benchmark_endpoints(client):
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "methods" in data
    assert "selected" in data

    # Valid method
    selected = data["selected"]
    res_m = client.get(f"/api/benchmark/{selected}")
    assert res_m.status_code == 200
    data_m = res_m.json()
    assert "per_dataset" in data_m

    # Invalid method id format
    res_bad = client.get("/api/benchmark/bad;id!!")
    assert res_bad.status_code == 400

    # Non-existent method file
    res_404 = client.get("/api/benchmark/non_existent_method_xyz")
    assert res_404.status_code == 404


def test_datasets_endpoint(client):
    res = client.get("/api/datasets")
    assert res.status_code == 200
    datasets = res.json()
    assert len(datasets) == 22
    assert any(d["id"] == 0 for d in datasets)


def test_replay_endpoint(client):
    # Valid dataset
    res = client.get("/api/replay/0")
    assert res.status_code == 200
    rep = res.json()
    assert rep["asset"] == 0
    assert rep["source"] == "replay"
    assert len(rep["t"]) > 0

    # Missing dataset
    res_404 = client.get("/api/replay/999999")
    assert res_404.status_code == 404


def test_sim_scenarios_endpoint(client):
    res = client.get("/api/sim/scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) >= 17
    assert any(s["id"] == "gearbox_bearing" for s in scenarios)
    assert any(s["id"] == "hydraulic_overheating" for s in scenarios)


def test_sim_state_and_step(client):
    # Initial state
    res = client.get("/api/sim/state")
    assert res.status_code == 200
    st = res.json()
    assert st["source"] == "simulation"
    assert len(st["turbines"]) == 5
    assert "farm" in st

    # Single step
    res_step = client.post("/api/sim/step", json={"steps": 3})
    assert res_step.status_code == 200
    st_after = res_step.json()
    assert st_after["source"] == "simulation"

    # Step bounds check (steps capped between 1 and 144)
    res_cap = client.post("/api/sim/step", json={"steps": 999})
    assert res_cap.status_code == 200


def test_sim_inject_repair_lifecycle(client):
    # Reset first
    res_reset = client.post("/api/sim/reset", json={"seed": 42})
    assert res_reset.status_code == 200

    # Inject invalid turbine
    res_bad_t = client.post("/api/sim/inject", json={"turbine": 999, "scenario": "gearbox_bearing"})
    assert res_bad_t.status_code == 400

    # Inject invalid scenario
    res_bad_s = client.post("/api/sim/inject", json={"turbine": 0, "scenario": "invalid_scenario_xyz"})
    assert res_bad_s.status_code == 400

    # Inject invalid severity
    res_bad_sev = client.post("/api/sim/inject", json={"turbine": 0, "scenario": "gearbox_bearing", "severity": "catastrophic"})
    assert res_bad_sev.status_code == 400

    # Valid injection
    res_inj = client.post("/api/sim/inject", json={
        "turbine": 0,
        "scenario": "hydraulic_overheating",
        "severity": "severe",
        "ramp_h": 2.0,
        "duration_h": 24.0,
        "seed": 101,
    })
    assert res_inj.status_code == 200
    inj_data = res_inj.json()
    inj_id = inj_data["injection"]["id"]
    assert inj_data["injection"]["scenario"] == "hydraulic_overheating"
    assert inj_data["injection"]["active"] is True

    # Injections list
    res_list = client.get("/api/sim/injections")
    assert res_list.status_code == 200
    injections = res_list.json()
    assert any(i["id"] == inj_id for i in injections)

    # Step simulation through the injection
    client.post("/api/sim/step", json={"steps": 18})

    # Turbine series check
    res_series = client.get("/api/sim/turbine/0?hours=12")
    assert res_series.status_code == 200
    series_data = res_series.json()
    assert series_data["id"] == 0
    assert "signals" in series_data
    assert "components" in series_data

    # Series for unknown turbine
    res_no_t = client.get("/api/sim/turbine/999")
    assert res_no_t.status_code == 404

    # Repair injection
    res_repair = client.post("/api/sim/repair", json={"injection_id": inj_id})
    assert res_repair.status_code == 200
    st_rep = res_repair.json()
    assert "turbines" in st_rep

    # Check injection marked ended
    res_list_after = client.get("/api/sim/injections")
    inj_after = next(i for i in res_list_after.json() if i["id"] == inj_id)
    assert inj_after["active"] is False
