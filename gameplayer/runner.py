"""Fast closed-loop real-time action runner (~8-12 Hz) with automatic safety stops."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from gameplayer.hud import HudObservation, OCRBackend, inspect_frame
from gameplayer.input import InputController
from gameplayer.minimap import MinimapObservation, inspect_minimap
from gameplayer.telemetry import TelemetryLogger, get_default_logger
from gameplayer.window import CaptureResult, WindowBackend, capture_game_window


DEFAULT_RETICLE_SWEEP: tuple[tuple[int, int], ...] = (
    (0, 0),
    (22, 0),
    (26, 0),
    (-70, 0),
    (-26, 0),
    (48, 20),
    (24, 0),
    (26, 0),
    (-74, 0),
    (-26, 0),
    (50, 18),
    (24, 0),
    (-48, 0),
    (24, -52),
    (24, 0),
    (-48, 0),
    (24, 14),
)


@dataclass(frozen=True)
class ReticleStatus:
    is_red: bool
    center_red_pixels: int
    outer_red_fraction: float
    contrast_score: float


def detect_red_reticle(image: Image.Image) -> ReticleStatus:
    """Detect GTA 5's hostile red reticle dot at screen center (0.5, 0.5) while rejecting ambient red walls."""
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    h, w, _ = arr.shape
    cx, cy = w // 2, h // 2
    inner_rx = max(3, int(round(w * 0.0035)))
    inner_ry = max(3, int(round(h * 0.006)))
    outer_rx = max(18, int(round(w * 0.032)))
    outer_ry = max(18, int(round(h * 0.055)))

    patch = arr[
        max(0, cy - outer_ry) : min(h, cy + outer_ry + 1),
        max(0, cx - outer_rx) : min(w, cx + outer_rx + 1),
    ]
    r = patch[:, :, 0]
    g = patch[:, :, 1]
    b = patch[:, :, 2]

    red_excess = r - np.maximum(g, b)
    red_mask = (r >= 165) & (g <= 95) & (b <= 95) & (red_excess >= 90)

    py, px = patch.shape[0] // 2, patch.shape[1] // 2
    inner_slice = (
        slice(max(0, py - inner_ry), min(patch.shape[0], py + inner_ry + 1)),
        slice(max(0, px - inner_rx), min(patch.shape[1], px + inner_rx + 1)),
    )
    inner_mask = red_mask[inner_slice]
    inner_pixels = int(np.count_nonzero(inner_mask))

    outer_mask = np.ones_like(red_mask, dtype=bool)
    outer_mask[inner_slice] = False
    outer_total = max(1, int(np.count_nonzero(outer_mask)))
    outer_red_pixels = int(np.count_nonzero(red_mask & outer_mask))
    outer_red_fraction = outer_red_pixels / outer_total

    if inner_pixels > 0:
        contrast_score = float(np.mean(red_excess[inner_slice][inner_mask])) - float(np.mean(red_excess[outer_mask]))
    else:
        contrast_score = 0.0

    is_red = inner_pixels >= 2 and contrast_score >= 40.0 and outer_red_fraction < 0.25
    return ReticleStatus(
        is_red=is_red,
        center_red_pixels=inner_pixels,
        outer_red_fraction=outer_red_fraction,
        contrast_score=contrast_score,
    )


@dataclass(frozen=True)
class LoopConfig:
    max_duration_s: float = 8.0
    hz: float = 10.0
    hud_check_interval_s: float = 0.25
    stop_on_banner: bool = True
    stop_on_help_change: bool = True
    stop_on_subtitle_change: bool = False
    health_drop_threshold: Optional[float] = 0.15


def _road_gray_sample(image: Image.Image) -> np.ndarray:
    """Return downsampled grayscale upper/road region for fast frame-motion detection."""
    w, h = image.size
    crop = image.crop(
        (
            int(round(w * 0.15)),
            int(round(h * 0.15)),
            int(round(w * 0.85)),
            int(round(h * 0.72)),
        )
    )
    small = crop.convert("L").resize((80, 45), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.int16)


def _road_motion_score(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    return float(np.mean(np.abs(cur_gray - prev_gray)))


@dataclass(frozen=True)
class FrameState:
    tick: int
    elapsed_s: float
    capture: CaptureResult
    minimap: MinimapObservation
    hud: Optional[HudObservation]


@dataclass(frozen=True)
class LoopResult:
    stop_reason: str
    ticks: int
    elapsed_s: float
    banner: Optional[str]
    last_hud: Optional[HudObservation]
    last_minimap: Optional[MinimapObservation]
    actions_sent: int

    def to_dict(self) -> dict[str, object]:
        return {
            "stop_reason": self.stop_reason,
            "ticks": self.ticks,
            "elapsed_s": round(float(self.elapsed_s), 3),
            "banner": self.banner,
            "actions_sent": self.actions_sent,
            "last_hud": self.last_hud.to_dict() if self.last_hud else None,
            "last_minimap": self.last_minimap.to_dict() if self.last_minimap else None,
        }


class ActionRunner:
    """Executes closed-loop gameplay skills using capture + minimap + HUD OCR + fail-safe inputs."""

    def __init__(
        self,
        input_controller: InputController,
        window_backend: Optional[WindowBackend] = None,
        ocr_backend: Optional[OCRBackend] = None,
        logger: Optional[TelemetryLogger] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.input = input_controller
        self._window_backend = window_backend
        self._ocr_backend = ocr_backend
        self._logger = logger or get_default_logger()
        self._clock = clock
        self._sleep = sleep_fn

    def _run_loop(
        self,
        mode_name: str,
        config: LoopConfig,
        on_tick: Callable[[FrameState], Optional[str]],
        initial_keys: Iterable[str] = (),
        initial_buttons: Iterable[str] = (),
    ) -> LoopResult:
        if config.hz <= 0:
            raise ValueError(f"config.hz must be positive, got {config.hz}")
        if config.max_duration_s <= 0:
            raise ValueError(f"config.max_duration_s must be positive, got {config.max_duration_s}")

        tick_period = 1.0 / config.hz
        start_t = self._clock()
        last_hud_t = -1e9
        baseline_help: Optional[str] = None
        baseline_subtitle: Optional[str] = None
        baseline_health: Optional[float] = None
        last_hud: Optional[HudObservation] = None
        last_minimap: Optional[MinimapObservation] = None
        stop_reason = "timeout"
        ticks = 0
        actions_sent = 0

        with self._logger.timed("runner", mode_name, {"hz": config.hz, "max_duration_s": config.max_duration_s}) as details:
            try:
                for key in initial_keys:
                    self.input.key_down(key)
                    actions_sent += 1
                for btn in initial_buttons:
                    self.input.mouse_down(btn)
                    actions_sent += 1

                while True:
                    now = self._clock()
                    elapsed = now - start_t
                    if elapsed >= config.max_duration_s:
                        stop_reason = "timeout"
                        break

                    tick_start = now
                    cap = capture_game_window(
                        backend=self._window_backend,
                        logger=self._logger,
                        require_16_9=True,
                    )
                    mm = inspect_minimap(cap.image, logger=self._logger)
                    last_minimap = mm

                    if mm.minimap_visible and config.health_drop_threshold is not None:
                        if baseline_health is None:
                            baseline_health = mm.health_ratio
                        elif (baseline_health - mm.health_ratio) >= config.health_drop_threshold:
                            stop_reason = "health_drop"
                            ticks += 1
                            break
                        elif mm.health_ratio > baseline_health:
                            baseline_health = mm.health_ratio

                    hud: Optional[HudObservation] = None
                    if (now - last_hud_t) >= config.hud_check_interval_s:
                        hud = inspect_frame(cap.image, ocr_backend=self._ocr_backend, logger=self._logger)
                        last_hud = hud
                        last_hud_t = now

                        if config.stop_on_banner and hud.banner is not None:
                            stop_reason = f"banner:{hud.banner}"
                            ticks += 1
                            break

                        if baseline_help is None:
                            baseline_help = hud.help_prompt
                        elif (
                            config.stop_on_help_change
                            and hud.help_prompt
                            and hud.help_prompt != baseline_help
                        ):
                            stop_reason = "help_changed"
                            ticks += 1
                            break

                        if baseline_subtitle is None:
                            baseline_subtitle = hud.subtitle_or_objective
                        elif (
                            config.stop_on_subtitle_change
                            and hud.subtitle_or_objective
                            and hud.subtitle_or_objective != baseline_subtitle
                        ):
                            stop_reason = "subtitle_changed"
                            ticks += 1
                            break

                    state = FrameState(
                        tick=ticks,
                        elapsed_s=elapsed,
                        capture=cap,
                        minimap=mm,
                        hud=hud,
                    )
                    custom_stop = on_tick(state)
                    ticks += 1
                    if custom_stop is not None:
                        stop_reason = custom_stop
                        break

                    tick_elapsed = self._clock() - tick_start
                    remaining_budget = config.max_duration_s - (self._clock() - start_t)
                    sleep_s = min(max(0.0, tick_period - tick_elapsed), max(0.0, remaining_budget))
                    if sleep_s > 0:
                        self._sleep(sleep_s)
            finally:
                self.input.release_all(reason=f"runner:{stop_reason}", suppress_errors=True)

            total_elapsed = self._clock() - start_t
            details.update(
                {
                    "stop_reason": stop_reason,
                    "ticks": ticks,
                    "elapsed_s": round(total_elapsed, 3),
                }
            )
            return LoopResult(
                stop_reason=stop_reason,
                ticks=ticks,
                elapsed_s=total_elapsed,
                banner=last_hud.banner if last_hud else None,
                last_hud=last_hud,
                last_minimap=last_minimap,
                actions_sent=actions_sent,
            )

    def run_until_event(
        self,
        hold_keys: Iterable[str] = (),
        hold_buttons: Iterable[str] = (),
        predicate: Optional[Callable[[FrameState], Optional[str]]] = None,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Hold keys/buttons while monitoring at ~8-12 Hz until an event, predicate, or timeout."""
        cfg = config or LoopConfig()

        def _on_tick(state: FrameState) -> Optional[str]:
            if predicate is not None:
                return predicate(state)
            return None

        return self._run_loop(
            mode_name="run_until_event",
            config=cfg,
            on_tick=_on_tick,
            initial_keys=hold_keys,
            initial_buttons=hold_buttons,
        )

    def follow_route_on_foot(
        self,
        sprint: bool = False,
        target_color: str = "yellow",
        arrive_distance: float = 0.08,
        steer_gain: float = 160.0,
        deadband: float = 0.05,
        stop_on_subtitle_keywords: Sequence[str] = ("return to", "wait for", "stay with", "follow"),
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Walk/sprint toward minimap route/objective with crew-tether subtitle keyword stop."""
        cfg = config or LoopConfig(max_duration_s=10.0, hz=10.0)
        keys = ("w", "lshift") if sprint else ("w",)
        norm_keywords = tuple(kw.strip().lower() for kw in stop_on_subtitle_keywords if kw.strip())

        def _on_tick(state: FrameState) -> Optional[str]:
            if state.hud is not None and norm_keywords:
                combined_hud = f"{state.hud.subtitle_or_objective} {state.hud.help_prompt}".lower()
                for kw in norm_keywords:
                    if kw in combined_hud:
                        return f"subtitle_keyword:{kw}"

            mm = state.minimap
            targets = mm.targets.get(target_color, ())
            if targets:
                target = targets[0]
            elif target_color in ("yellow", "purple"):
                target = mm.primary_objective
            else:
                target = None

            if target is not None and target.distance <= arrive_distance:
                return "target_reached"

            bearing: Optional[float] = None
            if target_color == "yellow" and mm.yellow_route_bearing is not None:
                bearing = mm.yellow_route_bearing
            elif target_color == "purple" and mm.purple_route_bearing is not None:
                bearing = mm.purple_route_bearing
            elif target is not None:
                bearing = target.bearing

            if bearing is not None and abs(bearing) > deadband:
                dx = int(round(max(-240.0, min(240.0, bearing * steer_gain))))
                if dx != 0:
                    self._steer_camera(dx, 0)
            return None

        return self._run_loop(
            mode_name="follow_route_on_foot",
            config=cfg,
            on_tick=_on_tick,
            initial_keys=keys,
        )

    def follow_blue_ally(
        self,
        sprint: bool = False,
        arrive_distance: float = 0.14,
        steer_gain: float = 160.0,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Walk/jog back to nearest blue ally minimap blip (e.g. Michael & Brad crew tether)."""
        return self.follow_route_on_foot(
            sprint=sprint,
            target_color="blue",
            arrive_distance=arrive_distance,
            steer_gain=steer_gain,
            stop_on_subtitle_keywords=(),
            config=config,
        )

    def follow_gps_drive(
        self,
        route_color: str = "yellow",
        throttle_key: str = "w",
        brake_key: str = "s",
        steer_left_key: str = "a",
        steer_right_key: str = "d",
        arrive_distance: float = 0.07,
        steer_deadband: float = 0.08,
        sharp_turn_threshold: float = 0.45,
        enable_stuck_recovery: bool = True,
        stuck_window_s: float = 1.5,
        stuck_motion_threshold: float = 3.0,
        reverse_duration_s: float = 0.8,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Drive along minimap GPS route with steering, sharp-turn braking, and stuck-car reverse recovery."""
        cfg = config or LoopConfig(max_duration_s=12.0, hz=10.0)
        prev_gray: Optional[np.ndarray] = None
        last_motion_check_t: Optional[float] = None
        low_motion_elapsed_s = 0.0
        last_bearing = 0.0
        recovery_count = 0

        def _set_key(key: str, want_down: bool) -> None:
            is_down = key in self.input.held_keys
            if want_down and not is_down:
                self.input.key_down(key)
            elif not want_down and is_down:
                self.input.key_up(key)

        def _perform_stuck_recovery() -> None:
            nonlocal recovery_count, low_motion_elapsed_s, prev_gray
            recovery_count += 1
            counter_steer = steer_right_key if (last_bearing < 0 or recovery_count % 2 == 0) else steer_left_key
            _set_key(throttle_key, False)
            _set_key(steer_left_key, False)
            _set_key(steer_right_key, False)
            _set_key(brake_key, True)
            _set_key(counter_steer, True)
            self._logger.log(
                tool="runner",
                action="stuck_recovery",
                status="ok",
                details={"attempt": recovery_count, "counter_steer": counter_steer, "duration_s": reverse_duration_s},
            )
            step = 0.1
            rem = max(0.0, reverse_duration_s)
            while rem > 0:
                dt = min(step, rem)
                self._sleep(dt)
                rem -= dt
            _set_key(counter_steer, False)
            _set_key(brake_key, False)
            low_motion_elapsed_s = 0.0
            prev_gray = None

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal prev_gray, last_motion_check_t, low_motion_elapsed_s, last_bearing
            mm = state.minimap
            targets = mm.targets.get(route_color, ())
            target = targets[0] if targets else mm.primary_objective
            if target is not None and target.distance <= arrive_distance:
                return "target_reached"

            bearing: Optional[float] = None
            if route_color == "yellow" and mm.yellow_route_bearing is not None:
                bearing = mm.yellow_route_bearing
            elif route_color == "purple" and mm.purple_route_bearing is not None:
                bearing = mm.purple_route_bearing
            elif target is not None:
                bearing = target.bearing

            if bearing is not None:
                last_bearing = bearing

            sharp = bearing is not None and abs(bearing) >= sharp_turn_threshold
            if enable_stuck_recovery and not sharp:
                cur_gray = _road_gray_sample(state.capture.image)
                if prev_gray is not None and last_motion_check_t is not None:
                    dt = max(0.0, state.elapsed_s - last_motion_check_t)
                    motion = _road_motion_score(prev_gray, cur_gray)
                    if motion < stuck_motion_threshold:
                        low_motion_elapsed_s += dt
                    else:
                        low_motion_elapsed_s = 0.0
                prev_gray = cur_gray
                last_motion_check_t = state.elapsed_s

                if low_motion_elapsed_s >= stuck_window_s and (cfg.max_duration_s - state.elapsed_s) > 0.25:
                    _perform_stuck_recovery()
                    _set_key(throttle_key, True)
                    return None
            else:
                low_motion_elapsed_s = 0.0
                prev_gray = None
                last_motion_check_t = state.elapsed_s

            if bearing is None:
                _set_key(steer_left_key, False)
                _set_key(steer_right_key, False)
                _set_key(brake_key, False)
                _set_key(throttle_key, True)
                return None

            _set_key(steer_left_key, bearing < -steer_deadband)
            _set_key(steer_right_key, bearing > steer_deadband)
            _set_key(throttle_key, not sharp)
            _set_key(brake_key, sharp)
            return None

        return self._run_loop(
            mode_name="follow_gps_drive",
            config=cfg,
            on_tick=_on_tick,
            initial_keys=(throttle_key,),
        )

    def _tap_key_preserving_held(self, key: str, hold_s: float = 0.08) -> None:
        self.input.key_down(key)
        try:
            self._sleep(hold_s)
        finally:
            self.input.key_up(key)

    def _steer_camera(
        self,
        dx: int,
        dy: int,
        release_aim_button: Optional[str] = None,
    ) -> None:
        """Emit smooth ~15ms/step mouse deltas, temporarily releasing aim so free-look MouseMoved turns reliably."""
        mag = max(abs(dx), abs(dy))
        if mag == 0:
            return
        repress_aim = False
        if release_aim_button and release_aim_button in self.input.held_buttons:
            self.input.mouse_up(release_aim_button)
            repress_aim = True
        try:
            if mag > 20:
                steps = max(3, min(8, mag // 18))
                dur = max(0.05, steps * 0.015)
                self.input.mouse_delta(dx, dy, steps=steps, duration_s=dur)
            else:
                self.input.mouse_delta(dx, dy, steps=2, duration_s=0.03)
        finally:
            if repress_aim and release_aim_button:
                self.input.mouse_down(release_aim_button)

    def _fire_bursts(self, fire_button: str, bursts: int = 1, hold_s: float = 0.03, gap_s: float = 0.03) -> None:
        for idx in range(max(1, bursts)):
            self.input.mouse_down(fire_button)
            try:
                self._sleep(hold_s)
            finally:
                self.input.mouse_up(fire_button)
            if idx + 1 < bursts and gap_s > 0:
                self._sleep(gap_s)

    def switch_character(
        self,
        character: str,
        method: str = "hotkey",
        wheel_hold_s: float = 0.25,
        timeout_s: Optional[float] = None,
    ) -> None:
        """Switch character (`michael`, `franklin`, `trevor`, `online`) via F5-F8 hotkey or lalt+wheel."""
        self.input.switch_character(
            character=character,
            method=method,
            wheel_hold_s=wheel_hold_s,
            timeout_s=timeout_s,
        )

    def combat_burst_scan(
        self,
        aim_button: str = "right",
        fire_button: str = "left",
        steer_gain: float = 220.0,
        align_threshold: float = 0.05,
        scan_step_dx: int = 85,
        empty_scans_to_stop: Optional[int] = 10,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Track nearest red minimap enemy, pan aim toward bearing, and fire bursts when aligned."""
        cfg = config or LoopConfig(max_duration_s=8.0, hz=10.0)
        empty_ticks = 0

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal empty_ticks
            enemy = state.minimap.nearest_enemy
            if enemy is None:
                empty_ticks += 1
                if empty_scans_to_stop is not None and empty_ticks >= empty_scans_to_stop:
                    return "enemies_cleared"
                self.input.mouse_delta(scan_step_dx, 0, steps=1, duration_s=0.0)
                return None

            empty_ticks = 0
            if abs(enemy.bearing) > align_threshold:
                dx = int(round(max(-240.0, min(240.0, enemy.bearing * steer_gain))))
                if dx != 0:
                    self._steer_camera(dx, 0, release_aim_button=aim_button)
            else:
                self._fire_bursts(fire_button, bursts=1, hold_s=0.03)
            return None

        return self._run_loop(
            mode_name="combat_burst_scan",
            config=cfg,
            on_tick=_on_tick,
            initial_buttons=(aim_button,),
        )

    def aim_until_red_reticle(
        self,
        aim_button: str = "right",
        fire_button: str = "left",
        fire_on_lock: bool = True,
        bursts: int = 2,
        stop_on_shot: bool = True,
        max_shots: int = 1,
        steer_gain: float = 220.0,
        bearing_deadband: float = 0.10,
        aim_settle_s: float = 0.18,
        scan_pattern: Sequence[tuple[int, int]] = DEFAULT_RETICLE_SWEEP,
        empty_scans_to_stop: Optional[int] = None,
        fallback_fire_on_aligned: bool = False,
        fallback_fire_interval_ticks: int = 6,
        reacquire_aim_interval_ticks: Optional[int] = None,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Hold aim, sweep/steer until center reticle turns hostile red, and fire precision bursts.

        When `fallback_fire_on_aligned=True` (used by `shoot_enemies`), if an enemy is aligned on
        the minimap (`|bearing| <= bearing_deadband`) but occluded behind distant cover so the reticle
        stays white, fires periodic suppressive bursts and optionally re-taps aim to pop out/snap.
        """
        cfg = config or LoopConfig(max_duration_s=6.0, hz=12.0, hud_check_interval_s=0.25)
        sweep_idx = 0
        shots_fired = 0
        no_enemy_ticks = 0
        aligned_without_lock_ticks = 0
        consecutive_steer_ticks = 0
        last_steer_bearing: Optional[float] = None

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal sweep_idx, shots_fired, no_enemy_ticks, aligned_without_lock_ticks, consecutive_steer_ticks, last_steer_bearing
            if state.tick == 0 and aim_settle_s > 0:
                self._sleep(aim_settle_s)
                return None

            reticle = detect_red_reticle(state.capture.image)
            if reticle.is_red:
                no_enemy_ticks = 0
                aligned_without_lock_ticks = 0
                consecutive_steer_ticks = 0
                if fire_on_lock:
                    self._fire_bursts(fire_button, bursts=bursts, hold_s=0.03, gap_s=0.03)
                    shots_fired += 1
                    if stop_on_shot or shots_fired >= max_shots:
                        return "reticle_shot"
                else:
                    return "reticle_locked"
                return None

            enemy = state.minimap.nearest_enemy
            if enemy is not None and abs(enemy.bearing) > bearing_deadband:
                no_enemy_ticks = 0
                if last_steer_bearing is not None and abs(enemy.bearing - last_steer_bearing) < 0.03:
                    consecutive_steer_ticks += 1
                else:
                    consecutive_steer_ticks = 1
                last_steer_bearing = enemy.bearing

                if consecutive_steer_ticks >= 10:
                    return "steer_stalled"

                if consecutive_steer_ticks < 4 or abs(enemy.bearing) > 0.35:
                    aligned_without_lock_ticks = 0
                    dx = int(round(max(-220.0, min(220.0, enemy.bearing * steer_gain))))
                    if dx != 0:
                        self._steer_camera(dx, 0, release_aim_button=aim_button if abs(dx) >= 60 else None)
                        if abs(dx) >= 30:
                            sweep_idx = 0
                    return None
            else:
                consecutive_steer_ticks = 0
                last_steer_bearing = None

            if enemy is None:
                no_enemy_ticks += 1
                aligned_without_lock_ticks = 0
                if empty_scans_to_stop is not None and no_enemy_ticks >= empty_scans_to_stop:
                    return "enemies_cleared"
            else:
                no_enemy_ticks = 0
                aligned_without_lock_ticks += 1
                if (
                    fallback_fire_on_aligned
                    and fire_on_lock
                    and fallback_fire_interval_ticks > 0
                    and aligned_without_lock_ticks % fallback_fire_interval_ticks == 0
                ):
                    self._fire_bursts(fire_button, bursts=max(1, bursts), hold_s=0.03, gap_s=0.03)
                    shots_fired += 1
                    if shots_fired >= max_shots:
                        return "suppressive_shots_done"

                if (
                    reacquire_aim_interval_ticks is not None
                    and reacquire_aim_interval_ticks > 0
                    and aligned_without_lock_ticks % reacquire_aim_interval_ticks == 0
                ):
                    self.input.mouse_up(aim_button)
                    self._sleep(0.08)
                    self.input.mouse_down(aim_button)
                    if aim_settle_s > 0:
                        self._sleep(aim_settle_s)
                    return None

            if scan_pattern:
                dx, dy = scan_pattern[sweep_idx % len(scan_pattern)]
                sweep_idx += 1
                if dx != 0 or dy != 0:
                    self.input.mouse_delta(dx, dy, steps=2, duration_s=0.03)
            return None

        return self._run_loop(
            mode_name="aim_until_red_reticle",
            config=cfg,
            on_tick=_on_tick,
            initial_buttons=(aim_button,),
        )

    def aim_and_shoot(
        self,
        bursts: int = 2,
        stop_on_shot: bool = True,
        max_shots: int = 1,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Alias for `aim_until_red_reticle` with fire-on-lock enabled."""
        return self.aim_until_red_reticle(
            fire_on_lock=True,
            bursts=bursts,
            stop_on_shot=stop_on_shot,
            max_shots=max_shots,
            config=config,
        )

    def hostage_shot(
        self,
        bursts: int = 2,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Precision single-target reticle sweep & shot (e.g. Prologue hostage rescue)."""
        cfg = config or LoopConfig(max_duration_s=5.0, hz=12.0, hud_check_interval_s=0.25)
        return self.aim_until_red_reticle(
            fire_on_lock=True,
            bursts=bursts,
            stop_on_shot=True,
            max_shots=1,
            fallback_fire_on_aligned=False,
            config=cfg,
        )

    def shoot_enemies(
        self,
        max_shots: int = 8,
        empty_scans_to_stop: int = 12,
        fallback_fire_on_aligned: bool = True,
        use_special: bool = False,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Multi-target combat loop: red-reticle precision shots + aligned suppressive bursts."""
        cfg = config or LoopConfig(max_duration_s=10.0, hz=10.0, hud_check_interval_s=0.25)
        if use_special:
            self._tap_key_preserving_held("capslock", hold_s=0.08)
        return self.aim_until_red_reticle(
            fire_on_lock=True,
            bursts=2,
            stop_on_shot=False,
            max_shots=max_shots,
            empty_scans_to_stop=empty_scans_to_stop,
            fallback_fire_on_aligned=fallback_fire_on_aligned,
            fallback_fire_interval_ticks=4,
            reacquire_aim_interval_ticks=18,
            config=cfg,
        )

    def cover_peek_combat(
        self,
        enter_cover: bool = True,
        use_special: bool = True,
        min_peek_health: float = 0.45,
        rapid_drop_duck_threshold: float = 0.20,
        heal_wait_s: float = 3.5,
        duck_heal_s: Optional[float] = None,
        exit_on_health_drop: bool = False,
        peek_burst_s: float = 2.0,
        max_shots: Optional[int] = None,
        stop_when_no_enemies: bool = True,
        no_enemies_clear_s: float = 1.5,
        empty_scans_to_stop: Optional[int] = None,
        aim_button: str = "right",
        fire_button: str = "left",
        steer_gain: float = 220.0,
        bearing_deadband: float = 0.10,
        aim_settle_s: float = 0.18,
        scan_pattern: Sequence[tuple[int, int]] = DEFAULT_RETICLE_SWEEP,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Cover-peek combat loop with in-loop heal & resume, special ability, and early enemy-clear exit.

        Note: `q` is a toggle in GTA 5. Pass `enter_cover=False` if already crouched in cover so
        this loop does not accidentally stand out of cover.
        """
        eff_heal_wait_s = heal_wait_s if duck_heal_s is None else float(duck_heal_s)
        base_cfg = config or LoopConfig(
            max_duration_s=12.0,
            hz=10.0,
            hud_check_interval_s=0.25,
            health_drop_threshold=None,
        )
        cfg = LoopConfig(
            max_duration_s=base_cfg.max_duration_s,
            hz=base_cfg.hz,
            hud_check_interval_s=base_cfg.hud_check_interval_s,
            stop_on_banner=base_cfg.stop_on_banner,
            stop_on_help_change=base_cfg.stop_on_help_change,
            stop_on_subtitle_change=base_cfg.stop_on_subtitle_change,
            health_drop_threshold=(
                (base_cfg.health_drop_threshold if base_cfg.health_drop_threshold is not None else rapid_drop_duck_threshold)
                if exit_on_health_drop
                else None
            ),
        )
        if enter_cover:
            self._tap_key_preserving_held("q", hold_s=0.08)

        special_triggered = False
        is_peeking = False
        peek_start_t = 0.0
        duck_until_t = 0.0
        duck_logged = False
        peak_health: Optional[float] = None
        sweep_idx = 0
        shots_fired = 0
        no_enemy_ticks = 0
        no_enemy_since_t: Optional[float] = None
        aligned_ticks = 0
        consecutive_steer_ticks = 0
        last_steer_bearing: Optional[float] = None

        def _set_peek(want_peek: bool, now_t: float) -> None:
            nonlocal is_peeking, peek_start_t
            if want_peek and not is_peeking:
                self.input.mouse_down(aim_button)
                is_peeking = True
                peek_start_t = now_t
                if aim_settle_s > 0:
                    self._sleep(aim_settle_s)
            elif not want_peek and is_peeking:
                self.input.mouse_up(aim_button)
                is_peeking = False

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal special_triggered, duck_until_t, duck_logged, peak_health, sweep_idx, shots_fired, no_enemy_ticks, no_enemy_since_t, aligned_ticks, consecutive_steer_ticks, last_steer_bearing
            mm = state.minimap
            now_t = state.elapsed_s

            if use_special and not special_triggered and mm.minimap_visible and mm.special_ratio >= 0.20:
                self._tap_key_preserving_held("capslock", hold_s=0.08)
                special_triggered = True

            # Check in-loop heal/duck trigger (low HP or rapid drop when HP < 0.70 / low armor)
            if mm.minimap_visible:
                if peak_health is None or mm.health_ratio > peak_health:
                    peak_health = mm.health_ratio
                rapid_drop = (peak_health - mm.health_ratio) >= rapid_drop_duck_threshold and (
                    mm.health_ratio < 0.70 or mm.armor_ratio <= 0.05
                )
                if mm.health_ratio < min_peek_health or rapid_drop:
                    _set_peek(False, now_t)
                    duck_until_t = max(duck_until_t, now_t + eff_heal_wait_s)
                    peak_health = mm.health_ratio
                    if not duck_logged:
                        duck_logged = True
                        self._logger.log(
                            tool="runner",
                            action="cover_duck",
                            status="ok",
                            details={"health_ratio": round(mm.health_ratio, 3), "heal_wait_s": eff_heal_wait_s},
                        )
                    return None

            if now_t < duck_until_t:
                _set_peek(False, now_t)
                return None

            duck_logged = False

            # Limit continuous exposure during peek bursts
            if is_peeking and (now_t - peek_start_t) >= peek_burst_s:
                _set_peek(False, now_t)
                duck_until_t = now_t + 0.35
                return None

            _set_peek(True, now_t)

            reticle = detect_red_reticle(state.capture.image)
            if reticle.is_red:
                no_enemy_ticks = 0
                no_enemy_since_t = None
                aligned_ticks = 0
                consecutive_steer_ticks = 0
                self._fire_bursts(fire_button, bursts=3, hold_s=0.03, gap_s=0.025)
                shots_fired += 1
                if max_shots is not None and shots_fired >= max_shots:
                    return "reticle_shots_done"
                return None

            enemy = mm.nearest_enemy
            if enemy is not None and abs(enemy.bearing) > bearing_deadband:
                no_enemy_ticks = 0
                no_enemy_since_t = None
                if last_steer_bearing is not None and abs(enemy.bearing - last_steer_bearing) < 0.03:
                    consecutive_steer_ticks += 1
                else:
                    consecutive_steer_ticks = 1
                last_steer_bearing = enemy.bearing

                # Only treat as cover-clamped when target stays stuck genuinely outside cover FOV (> 0.45).
                if abs(enemy.bearing) > 0.45 and consecutive_steer_ticks >= 8:
                    self._logger.log(
                        tool="runner",
                        action="cover_clamped",
                        status="ok",
                        details={"bearing": round(enemy.bearing, 3)},
                    )
                    return "cover_clamped"

                # Keep aim held in cover (releasing right pulls back into cover and resets cover yaw).
                if consecutive_steer_ticks < 3 or abs(enemy.bearing) > 0.35:
                    aligned_ticks = 0
                    dx = int(round(max(-220.0, min(220.0, enemy.bearing * steer_gain))))
                    if dx != 0:
                        self._steer_camera(dx, 0)
                        if abs(dx) >= 30:
                            sweep_idx = 0
                    return None
                if consecutive_steer_ticks % 5 == 0 and abs(enemy.bearing) <= 0.45:
                    dx = int(round(max(-90.0, min(90.0, enemy.bearing * 110.0))))
                    if dx != 0:
                        self._steer_camera(dx, 0)
            else:
                consecutive_steer_ticks = 0
                last_steer_bearing = None

            if enemy is None:
                no_enemy_ticks += 1
                if no_enemy_since_t is None:
                    no_enemy_since_t = now_t
                aligned_ticks = 0
                if stop_when_no_enemies:
                    if empty_scans_to_stop is not None and no_enemy_ticks >= empty_scans_to_stop:
                        return "enemies_cleared"
                    if empty_scans_to_stop is None and (now_t - no_enemy_since_t) >= no_enemies_clear_s:
                        return "enemies_cleared"
            else:
                no_enemy_ticks = 0
                no_enemy_since_t = None
                aligned_ticks += 1
                if aligned_ticks % 4 == 0:
                    self._fire_bursts(fire_button, bursts=2, hold_s=0.03, gap_s=0.03)
                    shots_fired += 1
                    if max_shots is not None and shots_fired >= max_shots:
                        return "suppressive_shots_done"

            if scan_pattern:
                dx, dy = scan_pattern[sweep_idx % len(scan_pattern)]
                sweep_idx += 1
                if dx != 0 or dy != 0:
                    self.input.mouse_delta(dx, dy, steps=2, duration_s=0.03)
            return None

        return self._run_loop(
            mode_name="cover_peek_combat",
            config=cfg,
            on_tick=_on_tick,
        )
