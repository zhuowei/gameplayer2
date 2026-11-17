"""Structured JSONL telemetry logging for gameplayer tools."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

DEFAULT_TELEMETRY_PATH = Path("logs/telemetry.jsonl")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TelemetryLogger:
    """Thread-safe JSONL event logger under an ignored logs directory."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        env_path = os.environ.get("GAMEPLAYER_TELEMETRY_PATH")
        resolved = path if path is not None else (env_path or DEFAULT_TELEMETRY_PATH)
        self.path = Path(resolved)
        self._lock = threading.Lock()

    def log(
        self,
        tool: str,
        action: str,
        status: str = "ok",
        duration_ms: Optional[float] = None,
        details: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        """Append one structured telemetry event and return the recorded dict."""
        record: dict[str, Any] = {
            "ts": _utc_iso(),
            "monotonic_s": round(time.monotonic(), 6),
            "tool": tool,
            "action": action,
            "status": status,
        }
        if duration_ms is not None:
            record["duration_ms"] = round(float(duration_ms), 3)
        if details is not None:
            record["details"] = dict(details)
        if error is not None:
            record["error"] = str(error)

        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record

    @contextmanager
    def timed(
        self,
        tool: str,
        action: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[dict[str, Any]]:
        """Context manager that logs duration_ms and status ('ok' or 'error')."""
        mutable_details: dict[str, Any] = dict(details) if details else {}
        t0 = time.monotonic()
        try:
            yield mutable_details
        except BaseException as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            status = "timeout" if isinstance(exc, TimeoutError) else "error"
            self.log(
                tool=tool,
                action=action,
                status=status,
                duration_ms=elapsed_ms,
                details=mutable_details or None,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self.log(
                tool=tool,
                action=action,
                status="ok",
                duration_ms=elapsed_ms,
                details=mutable_details or None,
            )


_DEFAULT_LOGGER: Optional[TelemetryLogger] = None
_DEFAULT_LOGGER_LOCK = threading.Lock()


def get_default_logger() -> TelemetryLogger:
    """Return the process-wide default TelemetryLogger."""
    global _DEFAULT_LOGGER
    with _DEFAULT_LOGGER_LOCK:
        if _DEFAULT_LOGGER is None:
            _DEFAULT_LOGGER = TelemetryLogger()
        return _DEFAULT_LOGGER


def read_telemetry(
    path: Path | str = DEFAULT_TELEMETRY_PATH,
    tool: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Read JSONL telemetry records, optionally filtered by tool and tailed to limit."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if tool is not None and entry.get("tool") != tool:
                continue
            records.append(entry)
    if limit is not None and limit >= 0:
        return records[-limit:] if limit > 0 else []
    return records
