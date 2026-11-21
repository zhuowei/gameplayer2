"""Atomic machine-written live run scoreboard persisted to gitignored `logs/status.json`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_STATUS_PATH = Path("logs/status.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def resolve_status_path(path: Optional[Path | str] = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get("GAMEPLAYER_STATUS_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_STATUS_PATH


@dataclass(frozen=True)
class RunStatusSnapshot:
    mission: str
    checkpoint: str
    consecutive_failures: int = 0
    minutes_on_checkpoint: float = 0.0
    llm_turns_on_checkpoint: int = 0
    last_commit: str = ""
    red_reticle_rate: Optional[float] = None
    active_issues: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_status_snapshot(
    snapshot: RunStatusSnapshot,
    path: Optional[Path | str] = None,
) -> Path:
    """Atomically write `snapshot` to gitignored `logs/status.json` without dirtying git-tracked files."""
    target = resolve_status_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_status_snapshot(path: Optional[Path | str] = None) -> Optional[RunStatusSnapshot]:
    """Load latest `RunStatusSnapshot` from disk, or `None` if absent."""
    target = resolve_status_path(path)
    if not target.exists():
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    return RunStatusSnapshot(
        mission=str(data.get("mission", "")),
        checkpoint=str(data.get("checkpoint", "")),
        consecutive_failures=int(data.get("consecutive_failures", 0)),
        minutes_on_checkpoint=float(data.get("minutes_on_checkpoint", 0.0)),
        llm_turns_on_checkpoint=int(data.get("llm_turns_on_checkpoint", 0)),
        last_commit=str(data.get("last_commit", "")),
        red_reticle_rate=(
            float(data["red_reticle_rate"]) if data.get("red_reticle_rate") is not None else None
        ),
        active_issues=[str(x) for x in data.get("active_issues", ())],
        updated_at=str(data.get("updated_at", "")),
        extra=dict(data.get("extra", {})),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read or inspect gitignored live status snapshot.")
    parser.add_argument("--path", default=None, help="Path to status JSON (default: logs/status.json).")
    parser.add_argument("--show", action="store_true", help="Print current status snapshot JSON.")
    args = parser.parse_args(argv)

    snap = read_status_snapshot(args.path)
    if snap is None:
        print(json.dumps({"exists": False, "path": str(resolve_status_path(args.path))}))
        return 0
    print(json.dumps(snap.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
