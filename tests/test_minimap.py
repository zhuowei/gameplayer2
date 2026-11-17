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
