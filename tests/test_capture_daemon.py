from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import pytest
from PIL import Image

from gameplayer.capture_daemon import (
    CaptureConfig,
    FFmpegSegmentSink,
    RollingCaptureDaemon,
    extract_clip_slice,
    list_completed_segments,
    prune_old_segments,
)
from gameplayer.telemetry import TelemetryLogger
from gameplayer.window import WindowBounds, WindowInfo


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.now += dt


class SyntheticWindowBackend:
    def __init__(self, fail_indices: Optional[set[int]] = None) -> None:
        self.calls = 0
        self.fail_indices = fail_indices or set()
        self.win = WindowInfo(
            window_id=156,
            owner="GTA5.exe",
            title="Grand Theft Auto V",
            pid=9999,
            bounds=WindowBounds(0, 0, 1280, 720),
            layer=0,
            is_onscreen=True,
            is_frontmost=False,
        )

    def list_windows(self) -> list[WindowInfo]:
        self.calls += 1
        if self.calls in self.fail_indices:
            return []
        return [self.win]

    def capture_window_image(self, window_id: int) -> Image.Image:
        color = ((self.calls * 37) % 255, (self.calls * 73) % 255, 128)
        return Image.new("RGB", (1280, 720), color)

    def focus_window(self, window: WindowInfo) -> bool:
        raise AssertionError("Capture daemon must never focus the game window")


class MockVideoSink:
    def __init__(self) -> None:
        self.opened: list[tuple[Path, int, int, float]] = []
        self.frame_counts: list[int] = []
        self._current_path: Optional[Path] = None
        self._current_frames = 0

    def open_segment(self, final_path: Path, width: int, height: int, fps: float) -> None:
        self.opened.append((final_path, width, height, fps))
        self._current_path = final_path
        self._current_frames = 0

    def write_frame(self, rgb_bytes: bytes) -> None:
        assert len(rgb_bytes) == 640 * 360 * 3
        self._current_frames += 1

    def close_segment(self) -> Optional[Path]:
        if self._current_path is None:
            return None
        path = self._current_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frames={self._current_frames}".encode("utf-8"))
        self.frame_counts.append(self._current_frames)
        self._current_path = None
        self._current_frames = 0
        return path


def test_rolling_capture_daemon_segments_pruning_and_status(tmp_path) -> None:
    clock = FakeClock()
    out_dir = tmp_path / "recordings"
    status_path = tmp_path / "logs" / "capture_daemon.json"
    cfg = CaptureConfig(
        output_dir=out_dir,
        status_path=status_path,
        width=640,
        fps=2.0,
        segment_seconds=2.0,  # 4 frames per segment
        max_segments=2,
    )
    backend = SyntheticWindowBackend(fail_indices={3})  # transient window miss on call 3
    sink = MockVideoSink()
    logger = TelemetryLogger(tmp_path / "logs" / "telemetry.jsonl")

    daemon = RollingCaptureDaemon(
        config=cfg,
        window_backend=backend,
        video_sink=sink,
        logger=logger,
        clock=clock,
        sleep_fn=clock.sleep,
    )
    completed = daemon.run(max_frames=10)

    # 10 frames at 4 frames/segment => 3 segments (4, 4, 2 frames), pruned to newest 2
    assert len(completed) == 3
    assert sink.frame_counts == [4, 4, 2]
    remaining = list_completed_segments(out_dir)
    assert len(remaining) == 2
    assert remaining == completed[-2:]

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["running"] is False
    assert status["frames_captured"] == 10
    assert status["resolution"] == [640, 360]
    assert len(status["completed_segments"]) == 2


def test_prune_old_segments(tmp_path) -> None:
    d = tmp_path / "rec"
    d.mkdir()
    for i in range(5):
        (d / f"seg_20260101T00000{i}Z_000{i}.mp4").write_bytes(b"x")
    (d / ".in_progress_seg_tmp.mp4").write_bytes(b"keep")

    removed = prune_old_segments(d, max_segments=2)
    assert len(removed) == 3
    remaining = list_completed_segments(d)
    assert [p.name for p in remaining] == [
        "seg_20260101T000003Z_0003.mp4",
        "seg_20260101T000004Z_0004.mp4",
    ]
    assert (d / ".in_progress_seg_tmp.mp4").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_segment_sink_and_slice_extraction_offline(tmp_path) -> None:
    clock = FakeClock()
    out_dir = tmp_path / "recordings"
    cfg = CaptureConfig(
        output_dir=out_dir,
        status_path=tmp_path / "logs" / "daemon.json",
        width=640,
        fps=2.0,
        segment_seconds=3.0,  # 6 frames = 3s video
        max_segments=5,
    )
    daemon = RollingCaptureDaemon(
        config=cfg,
        window_backend=SyntheticWindowBackend(),
        video_sink=FFmpegSegmentSink(),
        logger=TelemetryLogger(tmp_path / "logs" / "telemetry.jsonl"),
        clock=clock,
        sleep_fn=clock.sleep,
    )
    completed = daemon.run(max_frames=6)
    assert len(completed) == 1
    seg = completed[0]
    assert seg.exists() and seg.stat().st_size > 0
    assert not (out_dir / f".in_progress_{seg.name}").exists()

    clip = extract_clip_slice(seg, tmp_path / "clips" / "debug_slice.mp4", start_s=0.5, duration_s=1.5)
    assert clip.exists() and clip.stat().st_size > 0
