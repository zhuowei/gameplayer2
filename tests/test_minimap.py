from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from gameplayer.minimap import (
    ARMOR_BAR_BOX,
    HEALTH_BAR_BOX,
    MINIMAP_RADAR_BOX,
    PLAYER_ANCHOR_REL,
    SPECIAL_BAR_BOX,
    inspect_minimap,
    main as minimap_cli_main,
)
from gameplayer.telemetry import TelemetryLogger


def make_synthetic_minimap_frame(
    health_ratio: float = 1.0,
    armor_ratio: float = 0.5,
    special_ratio: float = 0.0,
    yellow_points: list[tuple[float, float]] = (),
    purple_points: list[tuple[float, float]] = (),
    red_points: list[tuple[float, float]] = (),
    blue_points: list[tuple[float, float]] = (),
    route_yellow_rel: list[tuple[float, float]] = (),
    minimap_hidden: bool = False,
) -> Image.Image:
    """Create a 1280x720 16:9 synthetic frame with calibrated GTA 5 minimap & bars."""
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (30, 35, 42))
    if minimap_hidden:
        return img

    draw = ImageDraw.Draw(img)
    rx0, ry0, rx1, ry1 = [
        int(round(MINIMAP_RADAR_BOX[0] * w)),
        int(round(MINIMAP_RADAR_BOX[1] * h)),
        int(round(MINIMAP_RADAR_BOX[2] * w)),
        int(round(MINIMAP_RADAR_BOX[3] * h)),
    ]
    draw.rectangle((rx0, ry0, rx1, ry1), fill=(72, 68, 70))

    # Health bar
    hx0, hy0, hx1, hy1 = [
        int(round(HEALTH_BAR_BOX[0] * w)),
        int(round(HEALTH_BAR_BOX[1] * h)),
        int(round(HEALTH_BAR_BOX[2] * w)),
        int(round(HEALTH_BAR_BOX[3] * h)),
    ]
    draw.rectangle((hx0, hy0, hx1, hy1), fill=(35, 45, 35))
    if health_ratio > 0:
        fill_x = hx0 + max(1, int(round((hx1 - hx0) * min(1.0, health_ratio))))
        color = (70, 175, 85) if health_ratio > 0.25 else (210, 55, 50)
        draw.rectangle((hx0, hy0, fill_x, hy1), fill=color)

    # Armor bar
    ax0, ay0, ax1, ay1 = [
        int(round(ARMOR_BAR_BOX[0] * w)),
        int(round(ARMOR_BAR_BOX[1] * h)),
        int(round(ARMOR_BAR_BOX[2] * w)),
        int(round(ARMOR_BAR_BOX[3] * h)),
    ]
    draw.rectangle((ax0, ay0, ax1, ay1), fill=(35, 40, 50))
    if armor_ratio > 0:
        fill_x = ax0 + max(1, int(round((ax1 - ax0) * min(1.0, armor_ratio))))
        draw.rectangle((ax0, ay0, fill_x, ay1), fill=(65, 145, 220))

    # Special bar
    sx0, sy0, sx1, sy1 = [
        int(round(SPECIAL_BAR_BOX[0] * w)),
        int(round(SPECIAL_BAR_BOX[1] * h)),
        int(round(SPECIAL_BAR_BOX[2] * w)),
        int(round(SPECIAL_BAR_BOX[3] * h)),
    ]
    draw.rectangle((sx0, sy0, sx1, sy1), fill=(24, 23, 16))
    if special_ratio > 0:
        fill_x = sx0 + max(1, int(round((sx1 - sx0) * min(1.0, special_ratio))))
        draw.rectangle((sx0, sy0, fill_x, sy1), fill=(155, 130, 45))

    rw = rx1 - rx0
    rh = ry1 - ry0

    if len(route_yellow_rel) >= 2:
        pts = [(rx0 + px * rw, ry0 + py * rh) for px, py in route_yellow_rel]
        draw.line(pts, fill=(240, 200, 40), width=4)

    for pts, color in (
        (yellow_points, (240, 205, 45)),
        (purple_points, (185, 80, 225)),
        (red_points, (225, 45, 45)),
        (blue_points, (70, 145, 235)),
    ):
        for px, py in pts:
            cx, cy = rx0 + px * rw, ry0 + py * rh
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)

    return img


def test_minimap_health_armor_and_blips(tmp_path) -> None:
    # Player anchor is at (0.50, 0.77).
    # Yellow objective to the front-right, red enemy ahead-left, blue ally behind-left.
    frame = make_synthetic_minimap_frame(
        health_ratio=0.75,
        armor_ratio=0.50,
        special_ratio=0.80,
        yellow_points=[(0.75, 0.40)],
        red_points=[(0.30, 0.45)],
        blue_points=[(0.35, 0.88)],
    )
    obs = inspect_minimap(frame, logger=TelemetryLogger(tmp_path / "t.jsonl"))
    assert obs.minimap_visible is True
    assert 0.70 <= obs.health_ratio <= 0.80
    assert 0.44 <= obs.armor_ratio <= 0.56
    assert 0.74 <= obs.special_ratio <= 0.86
    assert obs.primary_objective is not None
    assert obs.primary_objective.kind == "yellow"
    assert obs.primary_objective.bearing > 0.1
    assert obs.nearest_enemy is not None
    assert obs.nearest_enemy.kind == "red"
    assert obs.nearest_enemy.bearing < -0.05
    assert len(obs.targets["blue"]) == 1


def test_minimap_purple_waypoint_and_low_health_red_bar(tmp_path) -> None:
    frame = make_synthetic_minimap_frame(
        health_ratio=0.20,
        armor_ratio=0.0,
        purple_points=[(0.50, 0.25)],
    )
    obs = inspect_minimap(frame, logger=TelemetryLogger(tmp_path / "t.jsonl"))
    assert obs.minimap_visible is True
    assert 0.15 <= obs.health_ratio <= 0.25
    assert obs.armor_ratio == 0.0
    assert obs.primary_objective is not None
    assert obs.primary_objective.kind == "purple"
    assert abs(obs.primary_objective.bearing) < 0.05
    assert obs.purple_route_bearing is not None


def test_minimap_yellow_gps_route_bearing(tmp_path) -> None:
    frame = make_synthetic_minimap_frame(
        health_ratio=1.0,
        armor_ratio=0.0,
        route_yellow_rel=[(0.50, 0.75), (0.65, 0.55), (0.80, 0.35)],
    )
    obs = inspect_minimap(frame, logger=TelemetryLogger(tmp_path / "t.jsonl"))
    assert obs.yellow_route_bearing is not None
    assert obs.yellow_route_bearing > 0.1


def test_minimap_hidden_during_cutscene_or_menu(tmp_path) -> None:
    frame = make_synthetic_minimap_frame(minimap_hidden=True)
    obs = inspect_minimap(frame, logger=TelemetryLogger(tmp_path / "t.jsonl"))
    assert obs.minimap_visible is False
    assert obs.health_ratio == 0.0


def test_minimap_cli_inspect_frame(tmp_path, capsys) -> None:
    frame = make_synthetic_minimap_frame(health_ratio=0.9, yellow_points=[(0.50, 0.30)])
    p = tmp_path / "mm.png"
    frame.save(p)
    rc = minimap_cli_main(["--inspect-frame", str(p)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["minimap_visible"] is True
    assert payload["primary_objective"]["kind"] == "yellow"


def test_minimap_rejects_magenta_police_siren_flash_and_detects_true_enemy(tmp_path) -> None:
    from PIL import ImageDraw

    frame = make_synthetic_minimap_frame(health_ratio=1.0, red_points=[(0.50, 0.40)])
    # Paint a magenta-red siren flash (191, 51, 71) near bottom of radar (observed in 5-star chase)
    w, h = frame.size
    rx0 = int(round(MINIMAP_RADAR_BOX[0] * w))
    ry0 = int(round(MINIMAP_RADAR_BOX[1] * h))
    rw = int(round((MINIMAP_RADAR_BOX[2] - MINIMAP_RADAR_BOX[0]) * w))
    rh = int(round((MINIMAP_RADAR_BOX[3] - MINIMAP_RADAR_BOX[1]) * h))
    draw = ImageDraw.Draw(frame)
    sx, sy = rx0 + int(0.48 * rw), ry0 + int(0.92 * rh)
    draw.rectangle((sx - 2, sy - 2, sx + 2, sy + 2), fill=(191, 51, 71))

    obs = inspect_minimap(frame, logger=TelemetryLogger(tmp_path / "t.jsonl"))
    assert len(obs.targets["red"]) == 1
    assert obs.nearest_enemy is not None
    assert abs(obs.nearest_enemy.bearing) < 0.05


def test_minimap_calibrated_bars_and_blue_siren_wash_filtering(tmp_path) -> None:
    # 1) Depleted green bar (50% bright green fill, 50% empty dark-green track),
    #    empty slate-blue armor track, and locked Prologue ochre special bar.
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (30, 35, 42))
    draw = ImageDraw.Draw(img)

    rx0, ry0, rx1, ry1 = [int(round(MINIMAP_RADAR_BOX[i] * (w if i % 2 == 0 else h))) for i in range(4)]
    draw.rectangle((rx0, ry0, rx1, ry1), fill=(72, 68, 70))

    hx0, hy0, hx1, hy1 = [int(round(HEALTH_BAR_BOX[i] * (w if i % 2 == 0 else h))) for i in range(4)]
    mid_x = hx0 + (hx1 - hx0) // 2
    draw.rectangle((hx0, hy0, mid_x, hy1), fill=(78, 145, 78))
    draw.rectangle((mid_x + 1, hy0, hx1, hy1), fill=(42, 88, 42))  # empty dark-green track

    ax0, ay0, ax1, ay1 = [int(round(ARMOR_BAR_BOX[i] * (w if i % 2 == 0 else h))) for i in range(4)]
    draw.rectangle((ax0, ay0, ax1, ay1), fill=(33, 76, 100))  # empty slate-blue armor track

    sx0, sy0, sx1, sy1 = [int(round(SPECIAL_BAR_BOX[i] * (w if i % 2 == 0 else h))) for i in range(4)]
    draw.rectangle((sx0, sy0, sx1, sy1), fill=(80, 64, 22))  # locked Prologue ochre special bar

    # Paint a large 20x15 blue siren road wash and a single 6x6 real ally blip
    draw.rectangle((rx0 + 20, ry0 + 20, rx0 + 39, ry0 + 34), fill=(105, 142, 160))
    draw.rectangle((rx0 + 90, ry0 + 60, rx0 + 95, ry0 + 65), fill=(93, 182, 229))

    obs = inspect_minimap(img, logger=TelemetryLogger(tmp_path / "t1.jsonl"))
    assert 0.45 <= obs.health_ratio <= 0.55
    assert obs.armor_ratio == 0.0
    assert obs.special_ratio == 0.0
    assert len(obs.targets["blue"]) == 1

    # 2) Red-vignette green bar & blue-siren green bar (70% full) must not drop out to 0.0
    for fill_rgb, empty_rgb in [
        ((127, 135, 80), (38, 87, 46)),
        ((76, 114, 86), (40, 86, 58)),
    ]:
        img2 = img.copy()
        d2 = ImageDraw.Draw(img2)
        split_x = hx0 + int(round((hx1 - hx0) * 0.70))
        d2.rectangle((hx0, hy0, split_x, hy1), fill=fill_rgb)
        d2.rectangle((split_x + 1, hy0, hx1, hy1), fill=empty_rgb)
        obs2 = inspect_minimap(img2, logger=TelemetryLogger(tmp_path / "t2.jsonl"))
        assert 0.64 <= obs2.health_ratio <= 0.76
