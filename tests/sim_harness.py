"""Advancing-clock & lightweight 2.5D kinematic test simulator for closed-loop ActionRunner rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image, ImageDraw

from gameplayer.window import WindowBounds, WindowInfo
from tests.test_minimap import make_synthetic_minimap_frame


@dataclass
class BoxOccluder:
    """Axis-aligned 2D/3D obstacle blocking movement and line-of-sight."""

    x0: float
    y0: float
    x1: float
    y1: float
    height_m: float = 2.0
    phantom_peek: bool = False  # True simulates wide pillar hiding the center reticle dot

    def contains(self, x: float, y: float, pad: float = 0.0) -> bool:
        return (self.x0 - pad) <= x <= (self.x1 + pad) and (self.y0 - pad) <= y <= (self.y1 + pad)

    def intersects_segment(self, ax: float, ay: float, bx: float, by: float) -> Optional[float]:
        """Return parametric t in [0, 1] of first intersection with segment A->B, or None."""
        dx = bx - ax
        dy = by - ay
        tmin = 0.0
        tmax = 1.0
        for p, q in (
            (-dx, ax - self.x0),
            (dx, self.x1 - ax),
            (-dy, ay - self.y0),
            (dy, self.y1 - ay),
        ):
            if abs(p) < 1e-9:
                if q < 0:
                    return None
            else:
                r = q / p
                if p < 0:
                    if r > tmax:
                        return None
                    if r > tmin:
                        tmin = r
                else:
                    if r < tmin:
                        return None
                    if r < tmax:
                        tmax = r
        return tmin if tmin <= tmax else None


@dataclass
class SimEnemy:
    """Simulated hostile NPC with torso/head bounding box and HP."""

    x: float
    y: float
    z_center: float = 1.15
    half_width_m: float = 0.42
    half_height_m: float = 0.65
    hp: float = 100.0
    dps: float = 0.0

    @property
    def alive(self) -> bool:
        return self.hp > 0.0


@dataclass
class SimAlly:
    x: float
    y: float


class SimClock:
    """Advancing monotonic clock that steps kinematic physics during every sleep."""

    def __init__(self, initial_time: float = 100.0, on_advance: Optional[Callable[[float], None]] = None) -> None:
        self.now = float(initial_time)
        self.on_advance = on_advance

    def __call__(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        step = max(0.0, float(dt))
        if step <= 0.0:
            return
        self.now += step
        if self.on_advance is not None:
            self.on_advance(step)


@dataclass
class KinematicWorld:
    """Offline 2.5D GTA V combat & camera simulator for closed-loop ActionRunner tests."""

    player_x: float = 0.0
    player_y: float = 0.0
    eye_z: float = 1.45
    yaw_deg: float = 0.0      # 0 = +Y ahead; + = turned right
    pitch_deg: float = 0.0    # 0 = horizon; + = looking down; - = looking up
    in_cover: bool = False
    health: float = 0.90
    armor: float = 0.50
    special: float = 0.0

    # Calibrated GTA V physics parameters
    ads_deg_per_px: float = 0.10
    hipfire_deg_per_px: float = 0.28       # ~2.8x ADS sensitivity
    recoil_px_per_round: float = 0.75      # upward camera kick per round fired
    health_regen_cap: float = 0.48         # Prologue passive health regen cap
    health_regen_rate_per_s: float = 0.08
    walk_speed_mps: float = 3.2
    sprint_speed_mps: float = 5.4
    fov_h_deg: float = 56.0
    fov_v_deg: float = 33.0
    minimap_range_m: float = 35.0

    # Failure-mode knobs
    cover_yaw_clamp_deg: Optional[float] = None
    drop_mouse_deltas: bool = False
    drop_after_right_release_s: float = 0.0

    enemies: list[SimEnemy] = field(default_factory=list)
    allies: list[SimAlly] = field(default_factory=list)
    occluders: list[BoxOccluder] = field(default_factory=list)
    yellow_target: Optional[tuple[float, float]] = None

    held_keys: set[str] = field(default_factory=set)
    held_buttons: set[str] = field(default_factory=set)
    _last_right_up_t: float = -1e9
    _clock_ref: Optional[SimClock] = None
    shots_landed: int = 0
    rounds_fired: int = 0
    max_abs_pitch_deg: float = 0.0

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        # Move player along held WASD keys with occluder collision sliding.
        fwd = (1.0 if "w" in self.held_keys else 0.0) - (1.0 if "s" in self.held_keys else 0.0)
        strafe = (1.0 if "d" in self.held_keys else 0.0) - (1.0 if "a" in self.held_keys else 0.0)
        if fwd != 0.0 or strafe != 0.0:
            speed = self.sprint_speed_mps if ("lshift" in self.held_keys and fwd > 0) else self.walk_speed_mps
            mag = math.hypot(fwd, strafe)
            fwd /= mag
            strafe /= mag
            rad = math.radians(self.yaw_deg)
            vx = (fwd * math.sin(rad) + strafe * math.cos(rad)) * speed * dt
            vy = (fwd * math.cos(rad) - strafe * math.sin(rad)) * speed * dt
            nx = self.player_x + vx
            ny = self.player_y + vy
            if not any(occ.contains(nx, self.player_y, pad=0.35) for occ in self.occluders):
                self.player_x = nx
            if not any(occ.contains(self.player_x, ny, pad=0.35) for occ in self.occluders):
                self.player_y = ny

        # Passive health regen in cover / out of fire, strictly capped at `health_regen_cap` (0.48).
        is_ducked = self.in_cover and ("right" not in self.held_buttons)
        if is_ducked or not any(e.alive and e.dps > 0.0 for e in self.enemies):
            if 0.0 < self.health < self.health_regen_cap:
                self.health = min(self.health_regen_cap, self.health + self.health_regen_rate_per_s * dt)

        # Enemy return fire when player is exposed (not ducked behind cover).
        if not is_ducked:
            for enemy in self.enemies:
                if enemy.alive and enemy.dps > 0.0 and self._has_los_to_enemy(enemy):
                    self.health = max(0.0, self.health - enemy.dps * dt)

    def _has_los_to_enemy(self, enemy: SimEnemy) -> bool:
        for occ in self.occluders:
            t = occ.intersects_segment(self.player_x, self.player_y, enemy.x, enemy.y)
            if t is not None and 0.0 < t < 1.0:
                return False
        return True

    def _world_to_bearing_dist(self, tx: float, ty: float) -> tuple[float, float]:
        dx = tx - self.player_x
        dy = ty - self.player_y
        world_ang = math.degrees(math.atan2(dx, dy))
        rel_ang = ((world_ang - self.yaw_deg + 180.0) % 360.0) - 180.0
        bearing = max(-1.0, min(1.0, rel_ang / 180.0))
        dist = min(0.95, math.hypot(dx, dy) / self.minimap_range_m)
        return bearing, dist

    def _bearing_dist_to_minimap_pt(self, bearing: float, dist: float) -> tuple[float, float]:
        theta = bearing * math.pi
        # PLAYER_ANCHOR_REL = (0.50, 0.77)
        px = 0.50 + dist * 0.77 * math.sin(theta)
        py = 0.77 - dist * 0.77 * math.cos(theta)
        return max(0.06, min(0.94, px)), max(0.06, min(0.94, py))

    def _ray_status(self) -> tuple[bool, bool, Optional[SimEnemy]]:
        """Return `(reticle_visible, reticle_is_red, locked_enemy)`."""
        rad_yaw = math.radians(self.yaw_deg)
        ray_far_x = self.player_x + math.sin(rad_yaw) * 40.0
        ray_far_y = self.player_y + math.cos(rad_yaw) * 40.0

        nearest_occ_t = 1.0
        nearest_occ: Optional[BoxOccluder] = None
        for occ in self.occluders:
            t = occ.intersects_segment(self.player_x, self.player_y, ray_far_x, ray_far_y)
            if t is not None and t < nearest_occ_t:
                nearest_occ_t = t
                nearest_occ = occ

        occ_dist = nearest_occ_t * 40.0
        if nearest_occ is not None and nearest_occ.phantom_peek and occ_dist < 3.0:
            return False, False, None

        best_enemy: Optional[SimEnemy] = None
        best_dist = 1e9
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dx = enemy.x - self.player_x
            dy = enemy.y - self.player_y
            d = math.hypot(dx, dy)
            if d < 0.2 or d >= occ_dist:
                continue
            yaw_to = math.degrees(math.atan2(dx, dy))
            dyaw = ((yaw_to - self.yaw_deg + 180.0) % 360.0) - 180.0
            half_w_deg = math.degrees(math.atan2(enemy.half_width_m, d))
            if abs(dyaw) > half_w_deg:
                continue
            dz = enemy.z_center - self.eye_z
            pitch_to = -math.degrees(math.atan2(dz, d))
            half_h_deg = math.degrees(math.atan2(enemy.half_height_m, d))
            if abs(self.pitch_deg - pitch_to) <= half_h_deg and d < best_dist:
                best_dist = d
                best_enemy = enemy

        return True, (best_enemy is not None), best_enemy

    def apply_mouse_delta(self, dx: int, dy: int, held_buttons: frozenset[str]) -> None:
        if self.drop_mouse_deltas:
            return
        now_t = self._clock_ref() if self._clock_ref is not None else 0.0
        if self.drop_after_right_release_s > 0.0 and (now_t - self._last_right_up_t) < self.drop_after_right_release_s:
            return
        sens = self.ads_deg_per_px if "right" in held_buttons else self.hipfire_deg_per_px
        new_yaw = self.yaw_deg + dx * sens
        if self.in_cover and "right" in held_buttons and self.cover_yaw_clamp_deg is not None:
            new_yaw = max(-self.cover_yaw_clamp_deg, min(self.cover_yaw_clamp_deg, new_yaw))
        self.yaw_deg = new_yaw
        self.pitch_deg = max(-60.0, min(60.0, self.pitch_deg + dy * sens))
        self.max_abs_pitch_deg = max(self.max_abs_pitch_deg, abs(self.pitch_deg))

    def apply_fire(self) -> None:
        self.rounds_fired += 1
        # Apply calibrated +0.75 px/round upward recoil before checking shot or after shot.
        _, is_red, target = self._ray_status()
        if is_red and target is not None:
            target.hp -= 35.0
            self.shots_landed += 1
        self.pitch_deg = max(-60.0, self.pitch_deg - self.recoil_px_per_round * self.ads_deg_per_px)
        self.max_abs_pitch_deg = max(self.max_abs_pitch_deg, abs(self.pitch_deg))

    def render_frame(self) -> Image.Image:
        red_pts = []
        for e in self.enemies:
            if e.alive:
                b, d = self._world_to_bearing_dist(e.x, e.y)
                red_pts.append(self._bearing_dist_to_minimap_pt(b, d))
        blue_pts = []
        for a in self.allies:
            b, d = self._world_to_bearing_dist(a.x, a.y)
            blue_pts.append(self._bearing_dist_to_minimap_pt(b, d))
        yellow_pts = []
        if self.yellow_target is not None:
            b, d = self._world_to_bearing_dist(self.yellow_target[0], self.yellow_target[1])
            yellow_pts.append(self._bearing_dist_to_minimap_pt(b, d))

        base = make_synthetic_minimap_frame(
            health_ratio=self.health,
            armor_ratio=self.armor,
            special_ratio=self.special,
            yellow_points=yellow_pts,
            red_points=red_pts,
            blue_points=blue_pts,
        )
        # Preserve bottom-left minimap crop, draw 2.5D scene elsewhere.
        w, h = base.size
        minimap_crop = base.crop((0, int(h * 0.72), int(w * 0.22), h))

        draw = ImageDraw.Draw(base)
        horizon_y = int(round(h * (0.50 - self.pitch_deg / self.fov_v_deg)))
        # Sky & snow ground
        draw.rectangle((0, 0, w, max(0, min(h, horizon_y))), fill=(95, 112, 134))
        draw.rectangle((0, max(0, min(h, horizon_y)), w, h), fill=(195, 206, 218))
        # World-anchored snow tracks in [11.5..34m] so every walking step produces >4.0 road motion inside y=[0.15..0.72]
        rad = math.radians(self.yaw_deg)
        along = self.player_x * math.sin(rad) + self.player_y * math.cos(rad)
        phase = along % 2.2
        for band_i in range(9):
            dist_m = 11.5 + band_i * 2.2 + (2.2 - phase)
            gy = int(round(horizon_y + (math.degrees(math.atan2(self.eye_z, dist_m)) / self.fov_v_deg) * h))
            if horizon_y + 2 < gy < h - 10:
                half_span = max(80, int(round(w * 0.45)))
                band_h = max(4, int(round(105.0 / dist_m)))
                draw.rectangle((w // 2 - half_span, gy - band_h, w // 2 + half_span, gy + band_h), fill=(78, 92, 112))

        # Background skyline posts shifting smoothly with yaw/pitch for optical-flow tracking.
        for wx in range(-180, 195, 13):
            rel_yaw = ((wx - self.yaw_deg + 180.0) % 360.0) - 180.0
            if abs(rel_yaw) <= self.fov_h_deg * 0.7:
                sx = int(round((0.5 + rel_yaw / self.fov_h_deg) * w))
                post_h = 38 + ((abs(wx) * 37 + 11) % 92)
                post_w = 5 + ((abs(wx) * 13 + 3) % 11)
                shade = 40 + ((abs(wx) * 23) % 35)
                draw.rectangle((sx - post_w, horizon_y - post_h, sx + post_w, horizon_y), fill=(shade, shade + 8, shade + 16))

        # Project occluders and alive enemies sorted far-to-near.
        items: list[tuple[float, str, object]] = []
        for e in self.enemies:
            if e.alive:
                d = math.hypot(e.x - self.player_x, e.y - self.player_y)
                items.append((d, "enemy", e))
        for occ in self.occluders:
            ocx = 0.5 * (occ.x0 + occ.x1)
            ocy = 0.5 * (occ.y0 + occ.y1)
            d = math.hypot(ocx - self.player_x, ocy - self.player_y)
            items.append((d, "occ", occ))
        items.sort(key=lambda it: it[0], reverse=True)

        for d, kind, obj in items:
            if d < 0.25:
                continue
            if kind == "enemy":
                e: SimEnemy = obj  # type: ignore[assignment]
                if not self._has_los_to_enemy(e):
                    continue
                yaw_to = math.degrees(math.atan2(e.x - self.player_x, e.y - self.player_y))
                dyaw = ((yaw_to - self.yaw_deg + 180.0) % 360.0) - 180.0
                if abs(dyaw) > self.fov_h_deg * 0.6:
                    continue
                sx = int(round((0.5 + dyaw / self.fov_h_deg) * w))
                pitch_to = -math.degrees(math.atan2(e.z_center - self.eye_z, d))
                sy = int(round((0.5 + (pitch_to - self.pitch_deg) / self.fov_v_deg) * h))
                hw = max(8, int(round(math.degrees(math.atan2(e.half_width_m, d)) / self.fov_h_deg * w)))
                hh = max(16, int(round(math.degrees(math.atan2(e.half_height_m, d)) / self.fov_v_deg * h)))
                # Draw human-shaped silhouette (head + narrower waist + dark shoulders/torso)
                head_r = max(4, hw // 2)
                draw.ellipse((sx - head_r, sy - hh, sx + head_r, sy - hh + 2 * head_r), fill=(32, 34, 40))
                draw.rectangle((sx - hw, sy - hh + 2 * head_r, sx + hw, sy + hh // 2), fill=(26, 30, 38))
                draw.rectangle((sx - int(hw * 0.7), sy + hh // 2, sx + int(hw * 0.7), sy + hh), fill=(42, 46, 52))
            else:
                occ: BoxOccluder = obj  # type: ignore[assignment]
                corners = [(occ.x0, occ.y0), (occ.x1, occ.y0), (occ.x1, occ.y1), (occ.x0, occ.y1)]
                angles = []
                for cx, cy in corners:
                    ang = ((math.degrees(math.atan2(cx - self.player_x, cy - self.player_y)) - self.yaw_deg + 180.0) % 360.0) - 180.0
                    angles.append(ang)
                min_a, max_a = min(angles), max(angles)
                if max_a < -self.fov_h_deg * 0.6 or min_a > self.fov_h_deg * 0.6 or (max_a - min_a) > 150.0:
                    continue
                x0 = int(round((0.5 + min_a / self.fov_h_deg) * w))
                x1 = int(round((0.5 + max_a / self.fov_h_deg) * w))
                top_y = int(round(horizon_y - (math.degrees(math.atan2(occ.height_m - self.eye_z, max(0.5, d))) / self.fov_v_deg) * h))
                draw.rectangle((x0, max(0, top_y), x1, min(h, horizon_y + 120)), fill=(82, 84, 88), outline=(55, 57, 62))

        # Paste calibrated minimap back in bottom-left
        base.paste(minimap_crop, (0, int(h * 0.72)))

        # Center reticle dot
        ret_vis, ret_red, _ = self._ray_status()
        if ret_vis:
            cx, cy = w // 2, h // 2
            color = (245, 30, 30) if ret_red else (224, 224, 224)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)
        return base


class SimWindowBackend:
    """WindowBackend serving rendered frames from `KinematicWorld`."""

    def __init__(self, world: KinematicWorld) -> None:
        self.world = world
        self.calls = 0
        self._win = WindowInfo(
            window_id=999,
            owner="GTA5.exe",
            title="Grand Theft Auto V",
            pid=9999,
            bounds=WindowBounds(0, 0, 1280, 720),
            layer=0,
            is_onscreen=True,
            is_frontmost=True,
        )

    def list_windows(self) -> list[WindowInfo]:
        return [self._win]

    def capture_window_image(self, window_id: int) -> Image.Image:
        self.calls += 1
        return self.world.render_frame()

    def focus_window(self, window: WindowInfo) -> bool:
        raise AssertionError("Offline simulator must never call focus_window")


class SimInputBackend:
    """InputBackend forwarding events directly into `KinematicWorld`."""

    def __init__(self, world: KinematicWorld, clock: SimClock) -> None:
        self.world = world
        self.clock = clock
        self.world._clock_ref = clock
        self.events: list[tuple[str, object]] = []

    def key_down(self, key: str) -> None:
        self.world.held_keys.add(key)
        self.events.append(("key_down", key))
        if key == "q":
            self.world.in_cover = not self.world.in_cover

    def key_up(self, key: str) -> None:
        self.world.held_keys.discard(key)
        self.events.append(("key_up", key))

    def mouse_down(self, button: str) -> None:
        self.world.held_buttons.add(button)
        self.events.append(("mouse_down", button))
        if button == "left":
            self.world.apply_fire()

    def mouse_up(self, button: str) -> None:
        self.world.held_buttons.discard(button)
        self.events.append(("mouse_up", button))
        if button == "right":
            self.world._last_right_up_t = self.clock()

    def mouse_move_delta(
        self,
        dx: int,
        dy: int,
        held_buttons: frozenset[str] = frozenset(),
    ) -> None:
        self.events.append(("mouse_delta", (dx, dy, sorted(held_buttons))))
        self.world.apply_mouse_delta(dx, dy, held_buttons)
