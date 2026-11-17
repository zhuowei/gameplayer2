"""Rolling downscaled 16:9 window video capture daemon and slice extractor for `watcher`."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from gameplayer.telemetry import TelemetryLogger, get_default_logger
from gameplayer.window import (
    QuartzWindowBackend,
    WindowBackend,
    WindowCaptureError,
    WindowNotFoundError,
    capture_game_window,
)


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path = Path("recordings")
    status_path: Path = Path("logs/capture_daemon.json")
    width: int = 640
    fps: float = 2.0
    segment_seconds: float = 180.0
    max_segments: int = 10
    ffmpeg_bin: str = "ffmpeg"

    @property
    def height(self) -> int:
        h = int(round(self.width * 9.0 / 16.0))
        return h if h % 2 == 0 else h + 1


class VideoSink(Protocol):
    """Abstract video segment writer for offline testing and ffmpeg encoding."""

    def open_segment(self, final_path: Path, width: int, height: int, fps: float) -> None:
        ...

    def write_frame(self, rgb_bytes: bytes) -> None:
        ...

    def close_segment(self) -> Optional[Path]:
        ...


class FFmpegSegmentSink:
    """Encodes raw RGB24 frames into an H.264 MP4 via ffmpeg and atomically promotes on close."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg") -> None:
        self._ffmpeg_bin = ffmpeg_bin
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._tmp_path: Optional[Path] = None
        self._final_path: Optional[Path] = None
        self._frames_written = 0

    def open_segment(self, final_path: Path, width: int, height: int, fps: float) -> None:
        if self._proc is not None:
            self.close_segment()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(f".in_progress_{final_path.name}")
        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._tmp_path = tmp_path
        self._final_path = final_path
        self._frames_written = 0

    def write_frame(self, rgb_bytes: bytes) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Video segment is not open")
        try:
            self._proc.stdin.write(rgb_bytes)
            self._frames_written += 1
        except BrokenPipeError as exc:
            stderr = self._proc.stderr.read().decode("utf-8", errors="replace") if self._proc.stderr else ""
            raise RuntimeError(f"ffmpeg pipe closed unexpectedly: {stderr}") from exc

    def close_segment(self) -> Optional[Path]:
        if self._proc is None:
            return None
        proc = self._proc
        tmp_path = self._tmp_path
        final_path = self._final_path
        frames_written = self._frames_written
        self._proc = None
        self._tmp_path = None
        self._final_path = None
        self._frames_written = 0

        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass
        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait(timeout=30)
        if rc != 0:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"ffmpeg exited with code {rc}: {stderr_bytes.decode('utf-8', errors='replace')}"
            )
        if frames_written == 0:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return None
        if tmp_path and final_path and tmp_path.exists():
            os.replace(tmp_path, final_path)
            return final_path
        return None


def list_completed_segments(output_dir: Path | str = Path("recordings")) -> list[Path]:
    """Return sorted completed segment paths (`seg_*.mp4`), excluding in-progress files."""
    directory = Path(output_dir)
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.name.startswith("seg_") and p.suffix == ".mp4"
    )


def prune_old_segments(output_dir: Path | str, max_segments: int) -> list[Path]:
    """Delete oldest completed segments beyond `max_segments`; return removed paths."""
    if max_segments < 1:
        raise ValueError(f"max_segments must be >= 1, got {max_segments}")
    segments = list_completed_segments(output_dir)
    removed: list[Path] = []
    excess = len(segments) - max_segments
    if excess > 0:
        for old in segments[:excess]:
            old.unlink(missing_ok=True)
            removed.append(old)
    return removed


def extract_clip_slice(
    input_video: Path | str,
    output_clip: Path | str,
    start_s: float = 0.0,
    duration_s: float = 12.0,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Extract a short 10-15s debug clip from a completed segment without re-reading raw video into chat."""
    in_path = Path(input_video)
    out_path = Path(output_clip)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")
    if start_s < 0 or duration_s <= 0:
        raise ValueError(f"Invalid slice range: start_s={start_s}, duration_s={duration_s}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(in_path),
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg slice failed ({res.returncode}): {res.stderr.strip()}")
    return out_path


class RollingCaptureDaemon:
    """Captures unobstructed 16:9 game frames at low FPS into rolling 3-minute MP4 segments."""

    def __init__(
        self,
        config: Optional[CaptureConfig] = None,
        window_backend: Optional[WindowBackend] = None,
        video_sink: Optional[VideoSink] = None,
        logger: Optional[TelemetryLogger] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or CaptureConfig()
        if self.config.width <= 0 or self.config.width % 2 != 0:
            raise ValueError(f"config.width must be a positive even integer, got {self.config.width}")
        if self.config.fps <= 0:
            raise ValueError(f"config.fps must be positive, got {self.config.fps}")
        if self.config.segment_seconds <= 0:
            raise ValueError(f"config.segment_seconds must be positive, got {self.config.segment_seconds}")

        self._window_backend = window_backend or QuartzWindowBackend()
        self._video_sink = video_sink or FFmpegSegmentSink(ffmpeg_bin=self.config.ffmpeg_bin)
        self._logger = logger or get_default_logger()
        self._clock = clock
        self._sleep = sleep_fn

        self._frames_captured = 0
        self._segment_seq = 0
        self._current_segment: Optional[Path] = None
        self._latest_completed: Optional[Path] = None
        self._last_error: Optional[str] = None
        self._last_frame_utc: Optional[str] = None
        self._segment_open = False

    def _next_segment_path(self) -> Path:
        self._segment_seq += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.config.output_dir / f"seg_{stamp}_{self._segment_seq:04d}.mp4"

    def _write_status(self, running: bool) -> None:
        self.config.status_path.parent.mkdir(parents=True, exist_ok=True)
        completed = [str(p) for p in list_completed_segments(self.config.output_dir)]
        payload: dict[str, Any] = {
            "pid": os.getpid(),
            "running": running,
            "fps": self.config.fps,
            "resolution": [self.config.width, self.config.height],
            "segment_seconds": self.config.segment_seconds,
            "max_segments": self.config.max_segments,
            "frames_captured": self._frames_captured,
            "current_segment": str(self._current_segment) if self._current_segment else None,
            "latest_completed_segment": str(self._latest_completed) if self._latest_completed else None,
            "completed_segments": completed,
            "last_frame_utc": self._last_frame_utc,
            "last_error": self._last_error,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp = self.config.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.config.status_path)

    def _open_new_segment(self) -> None:
        seg_path = self._next_segment_path()
        self._video_sink.open_segment(
            final_path=seg_path,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )
        self._current_segment = seg_path
        self._segment_open = True

    def _finalize_segment(self) -> Optional[Path]:
        if not self._segment_open:
            return None
        finished = self._video_sink.close_segment()
        self._segment_open = False
        self._current_segment = None
        if finished is not None:
            self._latest_completed = finished
            prune_old_segments(self.config.output_dir, self.config.max_segments)
            self._logger.log(
                tool="capture_daemon",
                action="segment_completed",
                status="ok",
                details={"path": str(finished)},
            )
        return finished

    def run(
        self,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> list[Path]:
        """Run the rolling capture loop until stopped or limits reached; return completed segments."""
        stop = stop_event or threading.Event()
        frame_interval = 1.0 / self.config.fps
        frames_per_segment = max(1, int(round(self.config.segment_seconds * self.config.fps)))
        frames_in_segment = 0
        start_time = self._clock()
        completed_in_run: list[Path] = []
        target_size = (self.config.width, self.config.height)

        self._logger.log(
            tool="capture_daemon",
            action="start",
            status="ok",
            details={
                "width": self.config.width,
                "height": self.config.height,
                "fps": self.config.fps,
                "segment_seconds": self.config.segment_seconds,
                "max_segments": self.config.max_segments,
            },
        )
        self._write_status(running=True)

        try:
            while not stop.is_set():
                if max_frames is not None and self._frames_captured >= max_frames:
                    break
                if max_seconds is not None and (self._clock() - start_time) >= max_seconds:
                    break

                tick_start = self._clock()
                try:
                    cap = capture_game_window(
                        backend=self._window_backend,
                        logger=self._logger,
                        target_width=self.config.width,
                        require_16_9=True,
                    )
                    frame_img = cap.image
                    if frame_img.size != target_size:
                        frame_img = frame_img.resize(target_size)
                    if not self._segment_open:
                        self._open_new_segment()
                    self._video_sink.write_frame(frame_img.convert("RGB").tobytes())
                    self._frames_captured += 1
                    frames_in_segment += 1
                    self._last_frame_utc = cap.timestamp_utc
                    self._last_error = None

                    if frames_in_segment >= frames_per_segment:
                        done = self._finalize_segment()
                        if done is not None:
                            completed_in_run.append(done)
                        frames_in_segment = 0

                    self._write_status(running=True)
                except (WindowNotFoundError, WindowCaptureError) as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._logger.log(
                        tool="capture_daemon",
                        action="frame_skipped",
                        status="error",
                        error=self._last_error,
                    )
                    self._write_status(running=True)

                if max_frames is not None and self._frames_captured >= max_frames:
                    break
                if max_seconds is not None and (self._clock() - start_time) >= max_seconds:
                    break

                elapsed = self._clock() - tick_start
                delay = max(0.0, frame_interval - elapsed)
                if delay > 0 and not stop.is_set():
                    self._sleep(delay)
        finally:
            done = self._finalize_segment()
            if done is not None:
                completed_in_run.append(done)
            self._write_status(running=False)
            self._logger.log(
                tool="capture_daemon",
                action="stop",
                status="ok",
                details={"frames_captured": self._frames_captured},
            )
        return completed_in_run


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rolling downscaled 16:9 game window capture daemon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run rolling segment capture daemon (does not focus game window)")
    p_run.add_argument("--output-dir", type=Path, default=Path("recordings"))
    p_run.add_argument("--status-path", type=Path, default=Path("logs/capture_daemon.json"))
    p_run.add_argument("--width", type=int, default=640)
    p_run.add_argument("--fps", type=float, default=2.0)
    p_run.add_argument("--segment-seconds", type=float, default=180.0)
    p_run.add_argument("--max-segments", type=int, default=10)
    p_run.add_argument("--max-frames", type=int, default=None)
    p_run.add_argument("--max-seconds", type=float, default=None)

    p_slice = sub.add_parser("slice", help="Extract a 10-15s debug clip from a segment")
    p_slice.add_argument("input_video", type=Path)
    p_slice.add_argument("output_clip", type=Path)
    p_slice.add_argument("--start", type=float, default=0.0)
    p_slice.add_argument("--duration", type=float, default=12.0)

    p_status = sub.add_parser("status", help="Show completed segments and daemon status JSON")
    p_status.add_argument("--output-dir", type=Path, default=Path("recordings"))
    p_status.add_argument("--status-path", type=Path, default=Path("logs/capture_daemon.json"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = CaptureConfig(
            output_dir=args.output_dir,
            status_path=args.status_path,
            width=args.width,
            fps=args.fps,
            segment_seconds=args.segment_seconds,
            max_segments=args.max_segments,
        )
        daemon = RollingCaptureDaemon(config=cfg)
        stop_ev = threading.Event()

        def _handle_sig(signum: int, frame: Any) -> None:
            stop_ev.set()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, _handle_sig)
            signal.signal(signal.SIGTERM, _handle_sig)

        daemon.run(max_frames=args.max_frames, max_seconds=args.max_seconds, stop_event=stop_ev)
        return 0

    if args.cmd == "slice":
        out = extract_clip_slice(
            input_video=args.input_video,
            output_clip=args.output_clip,
            start_s=args.start,
            duration_s=args.duration,
        )
        print(json.dumps({"slice": str(out), "start_s": args.start, "duration_s": args.duration}, indent=2))
        return 0

    if args.cmd == "status":
        segs = [str(p) for p in list_completed_segments(args.output_dir)]
        daemon_status = None
        if args.status_path.exists():
            daemon_status = json.loads(args.status_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "completed_segments": segs,
                    "latest_completed_segment": segs[-1] if segs else None,
                    "daemon_status": daemon_status,
                },
                indent=2,
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
