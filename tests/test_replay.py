"""Tests for real-data SCADA replay loading, integrity, timestamps, and signals."""
import gzip
import json
import pytest

from windtwin import config as C


@pytest.mark.parametrize("ds_id", [0, 10, 22])
def test_replay_file_integrity(ds_id):
    path = C.ROOT / "artifacts" / "replay" / f"{ds_id}.json.gz"
    assert path.is_file(), f"Replay file for dataset {ds_id} not found"

    with gzip.open(path, "rt", encoding="utf-8") as f:
        rep = json.load(f)

    # Core metadata
    assert rep["source"] == "replay"
    assert "id" in rep
    assert "asset" in rep
    assert rep["id"] == ds_id

    # Timestamps monotonicity
    timestamps = rep["t"]
    assert len(timestamps) > 100
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1], f"Non-monotonic timestamp at index {i}"

    # Power signal
    assert "power" in rep
    assert len(rep["power"]) == len(timestamps)

    # Signals structure
    signals = rep["signals"]
    assert len(signals) > 0
    first_sig = next(iter(signals.values()))
    assert "observed" in first_sig
    assert "expected" in first_sig
    assert len(first_sig["observed"]) == len(timestamps)

    # Alarm and component states
    assert "state" in rep
    assert len(rep["state"]) == len(timestamps)
    assert set(rep["state"]).issubset({"normal", "watch", "warning", "fault", "unknown", "offline"})

    # Limits present
    assert "limits" in rep
    assert "z_band" in rep["limits"]

    # If anomaly dataset, verify event window annotations
    if rep.get("label") == "anomaly":
        assert "event_start" in rep
        assert "event_end" in rep
        assert rep["event_start"] < rep["event_end"]
