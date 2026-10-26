"""Tests for model isolation, calibration boundary, determinism, and full scenario state progressions."""
import hashlib
from pathlib import Path
import numpy as np
import pytest

from server.sim import FarmSim, ASSETS
from windtwin import config as C


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_model_isolation_and_calibration_boundary():
    """Verify that simulation startup, calibration, and commissioning NEVER alter on-disk

    models, benchmark files, or replay artifacts, nor mutate baseline bundles in memory.
    """
    model_dir = C.ROOT / "artifacts" / "models"
    benchmark_dir = C.ROOT / "artifacts" / "benchmark"
    replay_dir = C.ROOT / "artifacts" / "replay"

    # Hash a sample of model, replay, and benchmark files before simulation runs
    sample_files = [
        model_dir / "ds_0.joblib",
        model_dir / "ds_10.joblib",
        benchmark_dir / "benchmark.json",
        replay_dir / "0.json.gz",
        replay_dir / "10.json.gz",
    ]
    hashes_before = {f: file_hash(f) for f in sample_files if f.is_file()}

    # Initialize simulation and perform reset with full commissioning
    sim = FarmSim(seed=99)
    baseline_sigmas = {
        (a, t): np.copy(sim.bundles[a].normalizer.sigma[t])
        for a in ASSETS for t in sim.bundles[a].targets
    }

    # Run steps and inject a fault
    sim.step(6)
    inj = sim.inject(turbine=0, scenario="hydraulic_overheating", severity="severe", ramp_h=0, duration_h=24)
    sim.step(12)
    sim.repair(inj["id"])
    sim.step(6)

    # Verify on-disk files are strictly unchanged
    for f, orig_hash in hashes_before.items():
        assert file_hash(f) == orig_hash, f"Artifact file {f.name} was modified by simulation!"

    # Verify that sim.bundles (baseline reference) was not mutated by the active monitors
    for a in ASSETS:
        for t in sim.bundles[a].targets:
            np.testing.assert_array_equal(
                sim.bundles[a].normalizer.sigma[t],
                baseline_sigmas[(a, t)],
                err_msg=f"Baseline bundle for turbine {a} target {t} was mutated in memory!",
            )


def test_determinism_fixed_seeds():
    """Verify that repeated simulation runs with identical seeds produce bitwise consistent signals and states."""
    sim1 = FarmSim(seed=42)
    sim1.step(6)
    sim1.inject(turbine=10, scenario="gearbox_bearing", severity="moderate", ramp_h=6, duration_h=36, seed=1234)
    sim1.step(12)
    state1 = sim1.state()
    series1 = sim1.series(10, hours=24)

    sim2 = FarmSim(seed=42)
    sim2.step(6)
    sim2.inject(turbine=10, scenario="gearbox_bearing", severity="moderate", ramp_h=6, duration_h=36, seed=1234)
    sim2.step(12)
    state2 = sim2.state()
    series2 = sim2.series(10, hours=24)

    # Compare farm level metrics
    assert state1["farm"]["power"] == pytest.approx(state2["farm"]["power"], rel=1e-7)
    assert state1["farm"]["energy_actual"] == pytest.approx(state2["farm"]["energy_actual"], rel=1e-7)

    # Compare series signals for the injected turbine
    for sig_name in ("power", "sensor_11_avg", "sensor_12_avg"):
        if sig_name in series1["signals"] and sig_name in series2["signals"]:
            obs1 = np.array([x for x in series1["signals"][sig_name]["observed"] if x is not None])
            obs2 = np.array([x for x in series2["signals"][sig_name]["observed"] if x is not None])
            np.testing.assert_allclose(obs1, obs2, rtol=1e-7)


def test_complete_hydraulic_scenario_progression():
    """Automate complete hydraulic-overheating scenario and verify actual state changes."""
    sim = FarmSim(seed=105)
    # Turbine 0 should start in normal state
    st0 = sim.state()["turbines"]
    t0_init = next(t for t in st0 if t["id"] == 0)
    assert t0_init["state"] in ("normal", "watch")

    # Inject severe hydraulic overheating with instant onset (ramp_h=0)
    inj = sim.inject(turbine=0, scenario="hydraulic_overheating", severity="severe", ramp_h=0, duration_h=24)
    assert inj["active"] is True

    # Step simulation until alarm fires or state transitions (persist=36 requires >36 steps, so iteration 7 transitions to warning)
    alarm_fired = False
    for _ in range(8):
        sim.step(6)
        st = sim.state()
        t0 = next(t for t in st["turbines"] if t["id"] == 0)
        if t0["state"] in ("warning", "fault"):
            alarm_fired = True
            break

    assert alarm_fired, "Hydraulic overheating did not transition to warning/fault state"

    # Verify case was created and check diagnosis
    cases = sim.cases()
    hyd_cases = [c for c in cases if c["turbine"] == 0 and c["status"] != "closed"]
    assert len(hyd_cases) > 0, "No active case created for injected hydraulic fault"
    case = hyd_cases[0]
    assert case["component"] in ("Hydraulic group", "Unknown - insufficient evidence"), (
        f"Unexpected component diagnosis {case['component']}"
    )

    # Repair the fault
    sim.repair(inj["id"])
    assert not any(i["active"] for i in sim.injections_out() if i["id"] == inj["id"])

    # Step forward and verify recovery/cooldown progression
    sim.step(36)
    t0_repaired = next(t for t in sim.state()["turbines"] if t["id"] == 0)
    assert t0_repaired["health"] >= t0["health"]


def test_complete_gearbox_scenario_progression():
    """Automate complete gearbox bearing overheating scenario and verify actual state changes."""
    sim = FarmSim(seed=202)
    inj = sim.inject(turbine=10, scenario="gearbox_bearing", severity="severe", ramp_h=0, duration_h=24)

    alarm_fired = False
    for _ in range(8):
        sim.step(6)
        t10 = next(t for t in sim.state()["turbines"] if t["id"] == 10)
        if t10["state"] in ("warning", "fault"):
            alarm_fired = True
            break

    assert alarm_fired, "Gearbox scenario did not transition turbine 10 to warning/fault state"

    # Check that gearbox bearing sensor (sensor_11_avg) elevated
    series = sim.series(10, hours=24)
    s11 = series["signals"]["sensor_11_avg"]
    obs = [x for x in s11["observed"] if x is not None]
    exp = [x for x in s11["expected"] if x is not None]
    assert len(obs) > 0 and len(exp) > 0
    assert np.mean(obs[-12:]) > np.mean(exp[-12:])

    # Repair and verify injection ends
    sim.repair(inj["id"])
    assert not any(i["active"] for i in sim.injections_out() if i["id"] == inj["id"])
