from __future__ import annotations

import pytest

from gameplayer.telemetry import TelemetryLogger, read_telemetry


def test_telemetry_log_and_read(tmp_path) -> None:
    log_path = tmp_path / "logs" / "telemetry.jsonl"
    logger = TelemetryLogger(log_path)

    logger.log("window", "inspect", status="ok", duration_ms=1.5, details={"wid": 156})
    logger.log("input", "tap", status="ok", details={"key": "w"})
    logger.log("window", "capture", status="ok", details={"size": [1280, 720]})

    all_records = read_telemetry(log_path)
    assert len(all_records) == 3
    assert all_records[0]["tool"] == "window"
    assert all_records[0]["details"] == {"wid": 156}

    window_records = read_telemetry(log_path, tool="window")
    assert len(window_records) == 2

    tailed = read_telemetry(log_path, limit=2)
    assert [r["action"] for r in tailed] == ["tap", "capture"]


def test_telemetry_timed_ok_error_and_timeout(tmp_path) -> None:
    log_path = tmp_path / "telemetry.jsonl"
    logger = TelemetryLogger(log_path)

    with logger.timed("input", "hold", {"keys": ["w"]}) as d:
        d["extra"] = 42

    with pytest.raises(RuntimeError):
        with logger.timed("window", "capture"):
            raise RuntimeError("boom")

    with pytest.raises(TimeoutError):
        with logger.timed("input", "run_sequence"):
            raise TimeoutError("too slow")

    records = read_telemetry(log_path)
    assert [r["status"] for r in records] == ["ok", "error", "timeout"]
    assert records[0]["details"] == {"keys": ["w"], "extra": 42}
    assert records[0]["duration_ms"] >= 0.0
    assert "RuntimeError: boom" in records[1]["error"]
    assert "TimeoutError: too slow" in records[2]["error"]


def test_telemetry_env_override(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "env_telemetry.jsonl"
    monkeypatch.setenv("GAMEPLAYER_TELEMETRY_PATH", str(custom))
    logger = TelemetryLogger()
    logger.log("test", "ping")
    assert custom.exists()
    assert len(read_telemetry(custom)) == 1
