"""Bottom-left 16:9 minimap health/armor bar and blip/GPS route extraction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from gameplayer.telemetry import TelemetryLogger, get_default_logger
from gameplayer.window import normalize_to_16_9

NormalizedBox = tuple[float, float, float, float]

# Calibrated from 1280x720 16:9 GTA 5 gameplay captures.
MINIMAP_RADAR_BOX: NormalizedBox = (0.0156, 0.805, 0.1563, 0.966)
HEALTH_BAR_BOX: NormalizedBox = (0.0156, 0.970, 0.0855, 0.981)
ARMOR_BAR_BOX: NormalizedBox = (0.0875, 0.970, 0.1200, 0.981)
SPECIAL_BAR_BOX: NormalizedBox = (0.1225, 0.970, 0.1558, 0.981)
PLAYER_ANCHOR_REL: tuple[float, float] = (0.50, 0.77)


@dataclass(frozen=True)
class RadarTarget:
    kind: str  # "yellow", "purple", "red", "blue"
    bearing: float  # -1.0 (left/behind) .. 0.0 (ahead) .. +1.0 (right/behind)
    distance: float  # relative to player-to-top-of-radar (~0.0..1.3)
    pixel_count: int
    centroid_rel: tuple[float, float]  # (x_rel, y_rel) inside MINIMAP_RADAR_BOX

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "bearing": round(float(self.bearing), 4),
            "distance": round(float(self.distance), 4),
            "pixel_count": self.pixel_count,
            "centroid_rel": [round(float(self.centroid_rel[0]), 4), round(float(self.centroid_rel[1]), 4)],
        }


@dataclass(frozen=True)
class MinimapObservation:
    image_size: tuple[int, int]
    minimap_visible: bool
    health_ratio: float
    armor_ratio: float
    targets: dict[str, tuple[RadarTarget, ...]]
    primary_objective: Optional[RadarTarget]
    nearest_enemy: Optional[RadarTarget]
    yellow_route_bearing: Optional[float]
    purple_route_bearing: Optional[float]
    special_ratio: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "image_size": list(self.image_size),
            "minimap_visible": self.minimap_visible,
            "health_ratio": round(float(self.health_ratio), 4),
            "armor_ratio": round(float(self.armor_ratio), 4),
            "special_ratio": round(float(self.special_ratio), 4),
            "primary_objective": self.primary_objective.to_dict() if self.primary_objective else None,
            "nearest_enemy": self.nearest_enemy.to_dict() if self.nearest_enemy else None,
            "yellow_route_bearing": (
                round(float(self.yellow_route_bearing), 4) if self.yellow_route_bearing is not None else None
            ),
            "purple_route_bearing": (
                round(float(self.purple_route_bearing), 4) if self.purple_route_bearing is not None else None
            ),
            "targets": {k: [t.to_dict() for t in v] for k, v in self.targets.items()},
        }


def _crop_box(arr: np.ndarray, box: NormalizedBox) -> np.ndarray:
    h, w, _ = arr.shape
    x0 = max(0, min(w - 1, int(round(box[0] * w))))
    y0 = max(0, min(h - 1, int(round(box[1] * h))))
    x1 = max(x0 + 1, min(w, int(round(box[2] * w))))
    y1 = max(y0 + 1, min(h, int(round(box[3] * h))))
    return arr[y0:y1, x0:x1]


def _bar_interior(bar_rgb: np.ndarray) -> np.ndarray:
    return bar_rgb[1:-1] if bar_rgb.shape[0] >= 4 else bar_rgb


def _bar_fill_ratio(active_mask: np.ndarray) -> float:
    if active_mask.size == 0 or active_mask.shape[1] == 0:
        return 0.0
    col_active = np.mean(active_mask, axis=0) >= 0.35
    if int(np.count_nonzero(col_active)) < 2:
        return 0.0
    return float(np.mean(col_active))


def _extract_health_ratio(bar_rgb: np.ndarray) -> float:
    interior = _bar_interior(bar_rgb)
    r = interior[:, :, 0].astype(np.int16)
    g = interior[:, :, 1].astype(np.int16)
    b = interior[:, :, 2].astype(np.int16)
    # Healthy green bar (tolerates blue siren & red damage vignette bleed) OR pulsing low-health red bar
    green_mask = (g >= 110) & ((g - r) >= 5) & ((g - b) >= 15)
    low_red_mask = (r >= 120) & (g <= 60) & ((r - g) >= 85) & ((r - b) >= 75)
    return _bar_fill_ratio(green_mask | low_red_mask)


def _extract_armor_ratio(bar_rgb: np.ndarray) -> float:
    interior = _bar_interior(bar_rgb)
    r = interior[:, :, 0].astype(np.int16)
    g = interior[:, :, 1].astype(np.int16)
    b = interior[:, :, 2].astype(np.int16)
    blue_mask = (b >= 150) & (g >= 110) & ((b - r) >= 65)
    return _bar_fill_ratio(blue_mask)


def _extract_special_ratio(bar_rgb: np.ndarray) -> float:
    interior = _bar_interior(bar_rgb)
    r = interior[:, :, 0].astype(np.int16)
    g = interior[:, :, 1].astype(np.int16)
    b = interior[:, :, 2].astype(np.int16)
    yellow_mask = (r >= 145) & (g >= 110) & ((r - b) >= 70) & ((g - b) >= 45)
    return _bar_fill_ratio(yellow_mask)


def _color_masks(radar_rgb: np.ndarray) -> dict[str, np.ndarray]:
    r = radar_rgb[:, :, 0].astype(np.int16)
    g = radar_rgb[:, :, 1].astype(np.int16)
    b = radar_rgb[:, :, 2].astype(np.int16)

    return {
        "yellow": (r >= 175) & (g >= 145) & (b <= 110) & ((r - b) >= 65) & ((g - b) >= 45),
        "purple": (r >= 120) & (b >= 140) & (g <= 125) & ((b - g) >= 35) & ((r - g) >= 25),
        "red": (
            (r >= 175)
            & (g <= 80)
            & (b <= 80)
            & ((r - g) >= 95)
            & ((r - b) >= 95)
            & (np.abs(g - b) <= 12)
        ),
        "blue": (b >= 180) & (r <= 115) & (g >= 125) & ((b - r) >= 75) & ((b - g) >= 28),
    }


def _bearing_and_distance(cx: float, cy: float, width: int, height: int) -> tuple[float, float]:
    ax = width * PLAYER_ANCHOR_REL[0]
    ay = height * PLAYER_ANCHOR_REL[1]
    dx = cx - ax
    dy = ay - cy  # positive = ahead (upward on minimap)
    theta = math.atan2(dx, dy)
    bearing = max(-1.0, min(1.0, theta / math.pi))
    scale = max(1.0, ay)
    distance = math.hypot(dx, dy) / scale
    return bearing, distance


def _extract_components(
    mask: np.ndarray,
    kind: str,
    min_pixels: int = 4,
) -> tuple[RadarTarget, ...]:
    h, w = mask.shape
    pts = np.argwhere(mask)
    if len(pts) < min_pixels:
        return ()

    scale = (h * w) / (180.0 * 116.0)
    max_pixels = max(120, int(round(120.0 * scale))) if kind in ("red", "blue") else None

    active = set((int(y), int(x)) for y, x in pts)
    visited: set[tuple[int, int]] = set()
    targets: list[RadarTarget] = []

    step = 1 if kind in ("red", "blue") else 2
    for start in active:
        if start in visited:
            continue
        queue: deque[tuple[int, int]] = deque([start])
        visited.add(start)
        comp: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            comp.append((cy, cx))
            for ny in range(cy - step, cy + step + 1):
                for nx in range(cx - step, cx + step + 1):
                    nb = (ny, nx)
                    if nb in active and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

        if len(comp) < min_pixels:
            continue
        if max_pixels is not None and len(comp) > max_pixels:
            continue
        mean_y = sum(p[0] for p in comp) / len(comp)
        mean_x = sum(p[1] for p in comp) / len(comp)
        bearing, distance = _bearing_and_distance(mean_x, mean_y, w, h)
        targets.append(
            RadarTarget(
                kind=kind,
                bearing=bearing,
                distance=distance,
                pixel_count=len(comp),
                centroid_rel=(mean_x / max(1, w), mean_y / max(1, h)),
            )
        )

    targets.sort(key=lambda t: (t.distance, -t.pixel_count))
    return tuple(targets)


def _route_lookahead_bearing(mask: np.ndarray) -> Optional[float]:
    """Estimate near-ahead GPS route bearing from route pixels in the forward/mid zone."""
    h, w = mask.shape
    pts = np.argwhere(mask)
    if len(pts) < 6:
        return None

    ax = w * PLAYER_ANCHOR_REL[0]
    ay = h * PLAYER_ANCHOR_REL[1]
    ys = pts[:, 0].astype(np.float64)
    xs = pts[:, 1].astype(np.float64)
    dists = np.hypot(xs - ax, ay - ys) / max(1.0, ay)

    # Prefer lookahead band [0.10..0.55] ahead of the player so bends steer smoothly.
    band = (dists >= 0.10) & (dists <= 0.55)
    if np.count_nonzero(band) < 4:
        band = dists >= 0.06
    if np.count_nonzero(band) == 0:
        return None

    bx = float(np.mean(xs[band]))
    by = float(np.mean(ys[band]))
    bearing, _ = _bearing_and_distance(bx, by, w, h)
    return bearing


def inspect_minimap(
    image_or_path: Image.Image | Path | str,
    logger: Optional[TelemetryLogger] = None,
    crop_titlebar: bool = True,
) -> MinimapObservation:
    """Extract health/armor bar ratios and minimap blips/GPS bearings from a 16:9 frame."""
    active_logger = logger or get_default_logger()
    with active_logger.timed("minimap", "inspect") as details:
        if isinstance(image_or_path, (Path, str)):
            img = Image.open(image_or_path)
        else:
            img = image_or_path

        norm_img, _ = normalize_to_16_9(img.convert("RGB"), crop_titlebar=crop_titlebar)
        arr = np.asarray(norm_img, dtype=np.uint8)

        health_bar = _crop_box(arr, HEALTH_BAR_BOX)
        armor_bar = _crop_box(arr, ARMOR_BAR_BOX)
        special_bar = _crop_box(arr, SPECIAL_BAR_BOX)
        radar_rgb = _crop_box(arr, MINIMAP_RADAR_BOX)

        health_ratio = _extract_health_ratio(health_bar)
        armor_ratio = _extract_armor_ratio(armor_bar)
        special_ratio = _extract_special_ratio(special_bar)
        minimap_visible = (health_ratio > 0.03) or (armor_ratio > 0.03)

        masks = _color_masks(radar_rgb)
        targets = {kind: _extract_components(mask, kind=kind) for kind, mask in masks.items()}

        yellow_targets = targets["yellow"]
        purple_targets = targets["purple"]
        red_targets = targets["red"]

        primary_objective: Optional[RadarTarget] = None
        if yellow_targets:
            primary_objective = yellow_targets[0]
        elif purple_targets:
            primary_objective = purple_targets[0]

        nearest_enemy = red_targets[0] if red_targets else None
        yellow_route_bearing = _route_lookahead_bearing(masks["yellow"])
        purple_route_bearing = _route_lookahead_bearing(masks["purple"])

        obs = MinimapObservation(
            image_size=norm_img.size,
            minimap_visible=minimap_visible,
            health_ratio=health_ratio,
            armor_ratio=armor_ratio,
            special_ratio=special_ratio,
            targets=targets,
            primary_objective=primary_objective,
            nearest_enemy=nearest_enemy,
            yellow_route_bearing=yellow_route_bearing,
            purple_route_bearing=purple_route_bearing,
        )
        details.update(
            {
                "minimap_visible": minimap_visible,
                "health_ratio": round(health_ratio, 3),
                "armor_ratio": round(armor_ratio, 3),
                "special_ratio": round(special_ratio, 3),
                "enemy_count": len(red_targets),
                "yellow_count": len(yellow_targets),
            }
        )
        return obs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline 16:9 GTA 5 minimap extractor")
    parser.add_argument("--inspect-frame", type=Path, required=True, help="Path to a 16:9 PNG frame")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    obs = inspect_minimap(args.inspect_frame)
    print(json.dumps(obs.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
