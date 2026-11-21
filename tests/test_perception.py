from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gameplayer.minimap import inspect_minimap
from gameplayer.perception import (
    CameraMotionTracker,
    ScreenTarget,
    SilhouetteHumanDetectorBackend,
    VisionHumanDetectorBackend,
    select_visual_enemy_target,
)
from tests.test_minimap import make_synthetic_minimap_frame


def _draw_person_scene(
    person_cx: float,
    person_cy: float,
    *,
    width: int = 640,
    height: int = 360,
    include_foreground_player: bool = False,
    include_ally: Optional[tuple[float, float]] = None,
) -> Image.Image:
    del include_ally
    img = Image.new("RGB", (width, height), (140, 155, 170))
    draw = ImageDraw.Draw(img)
    # Horizon and skyline texture
    hy = height // 2
    draw.rectangle((0, hy, width, height), fill=(195, 205, 215))
    for x in range(40, width, 90):
        draw.rectangle((x, hy - 50, x + 12, hy), fill=(70, 80, 90))

    if include_foreground_player:
        # Trevor foreground silhouette in lower-left (must be filtered out)
        draw.ellipse((int(width * 0.28), int(height * 0.44), int(width * 0.36), int(height * 0.56)), fill=(35, 35, 38))
        draw.rectangle((int(width * 0.23), int(height * 0.56), int(width * 0.40), int(height * 0.98)), fill=(65, 85, 72))

    px = int(round(person_cx * width))
    py = int(round(person_cy * height))
    draw.ellipse((px - 7, py - 24, px + 7, py - 10), fill=(25, 28, 34))
    draw.rectangle((px - 11, py - 10, px + 11, py + 14), fill=(20, 24, 32))
    draw.rectangle((px - 8, py + 14, px + 8, py + 28), fill=(35, 40, 48))
    return img


def test_silhouette_detector_and_target_selector_filter_player_and_ally() -> None:
    mm_frame = make_synthetic_minimap_frame(
        health_ratio=0.90,
        red_points=[(0.62, 0.48)],   # bearing ~ +0.13 -> expected_x ~ 0.78
        blue_points=[(0.36, 0.52)],  # bearing ~ -0.17 -> expected_x ~ 0.13
    )
    mm = inspect_minimap(mm_frame)
    scene = _draw_person_scene(0.68, 0.46, include_foreground_player=True)
    # Add an ally figure on the left near blue bearing
    draw = ImageDraw.Draw(scene)
    ax, ay = int(0.14 * 640), int(0.48 * 360)
    draw.ellipse((ax - 7, ay - 22, ax + 7, ay - 8), fill=(30, 35, 45))
    draw.rectangle((ax - 11, ay - 8, ax + 11, ay + 24), fill=(30, 45, 70))

    cands = SilhouetteHumanDetectorBackend().detect_humans(scene)
    assert len(cands) >= 1

    # Also explicitly test player-foreground and blue-ally filtering via synthetic ScreenTarget list
    synthetic_cands = [
        ScreenTarget(bbox=(0.21, 0.44, 0.19, 0.54), aim_point=(0.305, 0.62), confidence=0.95),  # Trevor foreground
        ScreenTarget(bbox=(0.11, 0.42, 0.05, 0.14), aim_point=(0.135, 0.47), confidence=0.90),  # Blue ally
        ScreenTarget(bbox=(0.66, 0.40, 0.04, 0.12), aim_point=(0.680, 0.44), confidence=0.72),  # Hostile enemy
    ]
    selected = select_visual_enemy_target(synthetic_cands, mm, preferred_enemy=mm.nearest_enemy)
    assert selected is not None
    assert abs(selected.target.aim_point[0] - 0.68) < 1e-6
    assert selected.dx > 0
    assert selected.dy < 0


def test_camera_motion_tracker_measures_yaw_pitch_and_zero_clamp() -> None:
    scene = _draw_person_scene(0.55, 0.48, width=1280, height=720)
    arr = np.asarray(scene, dtype=np.uint8)
    # Camera pans right (+24 px) and pitches up (-12 px) -> world shifts (-24, +12)
    shifted = Image.fromarray(np.roll(arr, shift=(12, -24), axis=(0, 1)))

    tracker = CameraMotionTracker()
    m0 = tracker.update(scene)
    assert m0.confidence == 0.0

    m1 = tracker.update(shifted)
    assert m1.confidence >= 0.60
    assert abs(m1.delta_yaw_px - 24.0) <= 2.5
    assert abs(m1.delta_pitch_px - (-12.0)) <= 2.5

    # Clamped / dropped input: identical frame gives ~0 delta with high confidence
    m2 = tracker.update(shifted)
    assert m2.confidence >= 0.80
    assert abs(m2.delta_yaw_px) <= 0.5
    assert abs(m2.delta_pitch_px) <= 0.5


def test_vision_human_detector_runs_on_frame_without_error() -> None:
    backend = VisionHumanDetectorBackend()
    sample_path = Path("/Users/zhuowei/Documents/proj/gameplayer/captures/p3_a21_run2_end.png")
    if sample_path.exists():
        img = Image.open(sample_path).convert("RGB")
        targets = backend.detect_humans(img, roi=(0.15, 0.20, 0.88, 0.76))
        assert isinstance(targets, list)
        assert len(targets) >= 1
    else:
        img = _draw_person_scene(0.60, 0.48, width=640, height=360)
        targets = backend.detect_humans(img)
        assert isinstance(targets, list)


def test_select_visual_enemy_rejects_trevor_pillar_and_out_of_cone_candidates() -> None:
    mm_frame = make_synthetic_minimap_frame(
        health_ratio=0.95,
        red_points=[(0.54, 0.44)],  # bearing ~ +0.045 -> pref_x ~ 0.60
    )
    mm = inspect_minimap(mm_frame)
    cands = [
        # Trevor third-person ADS shoulder clipped by ROI bottom=0.76 -> bottom=0.752
        ScreenTarget(bbox=(0.31, 0.45, 0.15, 0.302), aim_point=(0.385, 0.56), confidence=0.95, source="vision_human"),
        # Wide silhouette pillar/wall artifact
        ScreenTarget(bbox=(0.56, 0.30, 0.16, 0.25), aim_point=(0.64, 0.39), confidence=0.90, source="silhouette"),
        # Left-pillar edge silhouette (x < 0.20 while enemy is ahead)
        ScreenTarget(bbox=(0.06, 0.25, 0.04, 0.12), aim_point=(0.08, 0.29), confidence=0.88, source="silhouette"),
        # Out-of-cone target (> 0.24 from expected_x ~ 0.60)
        ScreenTarget(bbox=(0.88, 0.40, 0.03, 0.10), aim_point=(0.895, 0.44), confidence=0.92, source="vision_human"),
        # Valid silhouette enemy within ±0.24 cone
        ScreenTarget(bbox=(0.62, 0.42, 0.03, 0.10), aim_point=(0.635, 0.46), confidence=0.68, source="silhouette"),
    ]
    selected = select_visual_enemy_target(cands, mm, preferred_enemy=mm.nearest_enemy)
    assert selected is not None
    assert abs(selected.target.aim_point[0] - 0.635) < 1e-6
    assert selected.target.source == "silhouette"


def test_select_visual_enemy_rejects_closer_occluding_blue_ally_and_floor_boxes() -> None:
    # Cop far ahead at (0.50, 0.30) (~0.61 dist), closer blue ally Brad at (0.50, 0.69) (~0.10 dist, same bearing!)
    mm_frame = make_synthetic_minimap_frame(
        health_ratio=0.95,
        red_points=[(0.50, 0.30)],
        blue_points=[(0.50, 0.69)],
    )
    mm = inspect_minimap(mm_frame)
    cands = [
        # Brad foreground occluding along same bearing (h=0.169, aim_y=0.544)
        ScreenTarget(bbox=(0.452, 0.485, 0.061, 0.169), aim_point=(0.483, 0.544), confidence=0.88, source="vision_human"),
        # Floor / crouch artifact below combat horizon (aim_y=0.614 > 0.57)
        ScreenTarget(bbox=(0.50, 0.57, 0.04, 0.12), aim_point=(0.52, 0.614), confidence=0.85, source="vision_human"),
        # Real distant cop visible above/beside Brad
        ScreenTarget(bbox=(0.54, 0.41, 0.03, 0.09), aim_point=(0.555, 0.442), confidence=0.62, source="vision_human"),
    ]
    selected = select_visual_enemy_target(cands, mm, preferred_enemy=mm.nearest_enemy)
    assert selected is not None
    assert abs(selected.target.aim_point[0] - 0.555) < 1e-6
