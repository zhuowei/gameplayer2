"""Offline `.mp4` clip replay benchmark for perception, optical flow, reticle, minimap, and posture."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from gameplayer.minimap import inspect_minimap
from gameplayer.perception import (
    CameraMotionTracker,
    HumanDetectorBackend,
    SilhouetteHumanDetectorBackend,
    VisionHumanDetectorBackend,
    classify_posture,
    select_visual_enemy_target,
)
from gameplayer.runner import detect_red_reticle


@dataclass(frozen=True)
class ClipBenchmarkResult:
    clip_path: str
    frames_evaluated: int
    fps: float
    detector_backend: str
    detector_ms_per_frame: float
    raw_human_detections: int
    visual_enemy_lock_frames: int
    red_reticle_frames: int
    visible_reticle_frames: int
    phantom_peek_frames: int
    minimap_visible_frames: int
    min_health: Optional[float]
    max_health: Optional[float]
    mean_flow_confidence: float
    yaw_stall_frames: int
    in_cover_frames: int
    standing_frames: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_clip_frames(
    clip_path: Path | str,
    out_dir: Path | str,
    fps: float = 2.0,
    max_frames: Optional[int] = None,
) -> list[Path]:
    """Decode frames from an `.mp4` clip offline via `ffmpeg`."""
    src = Path(clip_path)
    if not src.exists():
        raise FileNotFoundError(f"Clip not found: {src}")
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", f"fps={fps}"]
    if max_frames is not None and max_frames > 0:
        cmd.extend(["-frames:v", str(int(max_frames))])
    cmd.append(str(dst / "frame_%04d.png"))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [Path(p) for p in sorted(glob.glob(str(dst / "frame_*.png")))]


def benchmark_frames(
    frame_paths: Sequence[Path | str],
    clip_label: str = "frames",
    fps: float = 2.0,
    use_vision: bool = True,
    human_detector: Optional[HumanDetectorBackend] = None,
) -> ClipBenchmarkResult:
    """Evaluate perception, optical flow, reticle, minimap, and posture over extracted frames."""
    detector: HumanDetectorBackend
    if human_detector is not None:
        detector = human_detector
    elif use_vision:
        try:
            detector = VisionHumanDetectorBackend()
        except Exception:
            detector = SilhouetteHumanDetectorBackend()
    else:
        detector = SilhouetteHumanDetectorBackend()

    tracker = CameraMotionTracker()
    det_total_ms = 0.0
    raw_humans = 0
    visual_locks = 0
    red_frames = 0
    vis_frames = 0
    phantom_frames = 0
    mm_frames = 0
    health_vals: list[float] = []
    flow_confs: list[float] = []
    yaw_stalls = 0
    in_cover = 0
    standing = 0

    for idx, fpath in enumerate(frame_paths):
        img = Image.open(fpath).convert("RGB")
        mm = inspect_minimap(img)
        if mm.minimap_visible:
            mm_frames += 1
            health_vals.append(mm.health_ratio)

        ret = detect_red_reticle(img)
        if ret.is_red:
            red_frames += 1
        if ret.is_visible:
            vis_frames += 1
        else:
            phantom_frames += 1

        t0 = time.perf_counter()
        cands = detector.detect_humans(img, roi=(0.15, 0.20, 0.88, 0.76))
        det_total_ms += (time.perf_counter() - t0) * 1000.0
        raw_humans += len(cands)
        if select_visual_enemy_target(cands, mm, preferred_enemy=mm.nearest_enemy) is not None:
            visual_locks += 1

        motion = tracker.update(img)
        if idx > 0 and motion.confidence > 0.0:
            flow_confs.append(motion.confidence)
            if motion.confidence >= 0.25 and abs(motion.delta_yaw_px) < 2.0:
                yaw_stalls += 1

        posture = classify_posture(img)
        if posture.state == "in_cover":
            in_cover += 1
        elif posture.state == "standing":
            standing += 1

    n = len(frame_paths)
    return ClipBenchmarkResult(
        clip_path=str(clip_label),
        frames_evaluated=n,
        fps=float(fps),
        detector_backend=type(detector).__name__,
        detector_ms_per_frame=round(det_total_ms / max(1, n), 2),
        raw_human_detections=raw_humans,
        visual_enemy_lock_frames=visual_locks,
        red_reticle_frames=red_frames,
        visible_reticle_frames=vis_frames,
        phantom_peek_frames=phantom_frames,
        minimap_visible_frames=mm_frames,
        min_health=round(min(health_vals), 3) if health_vals else None,
        max_health=round(max(health_vals), 3) if health_vals else None,
        mean_flow_confidence=round(sum(flow_confs) / len(flow_confs), 3) if flow_confs else 0.0,
        yaw_stall_frames=yaw_stalls,
        in_cover_frames=in_cover,
        standing_frames=standing,
    )


def benchmark_clip(
    clip_path: Path | str,
    fps: float = 2.0,
    max_frames: Optional[int] = None,
    use_vision: bool = True,
    human_detector: Optional[HumanDetectorBackend] = None,
) -> ClipBenchmarkResult:
    """Extract frames from `clip_path` into a temp directory and evaluate offline."""
    with tempfile.TemporaryDirectory(prefix="gp_bench_") as tmp_dir:
        frames = extract_clip_frames(clip_path, tmp_dir, fps=fps, max_frames=max_frames)
        return benchmark_frames(
            frames,
            clip_label=str(clip_path),
            fps=fps,
            use_vision=use_vision,
            human_detector=human_detector,
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline .mp4 clip benchmark for gameplayer perception.")
    parser.add_argument("clips", nargs="+", help="One or more .mp4 clip paths.")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate (default: 2.0).")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames per clip.")
    parser.add_argument("--no-vision", action="store_true", help="Use pure-numpy silhouette detector only.")
    args = parser.parse_args(argv)

    results = [
        benchmark_clip(
            clip,
            fps=args.fps,
            max_frames=args.max_frames,
            use_vision=not args.no_vision,
        ).to_dict()
        for clip in args.clips
    ]
    print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
