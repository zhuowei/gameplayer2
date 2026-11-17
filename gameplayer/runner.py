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
from gameplayer.window import CaptureResult, WindowBackend, WindowInfo, capture_game_window, find_game_window


DEFAULT_RETICLE_SWEEP: tuple[tuple[int, int], ...] = (
    (0, 0),
    (18, 0),
    (18, 0),
    (-54, 0),
    (-18, 0),
    (36, -22),
    (18, 0),
    (18, 0),
    (-54, 0),
    (-18, 0),
    (36, -26),
    (18, 0),
    (-36, 0),
    (18, -26),
    (24, 0),
    (-48, 0),
    (24, 84),
    (24, 0),
    (-48, 0),
    (24, -10),
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

    def _snap_cover_on_health_drop(self, mode_name: str) -> None:
        self._tap_key_preserving_held("q", hold_s=0.08)
        self._logger.log(
            tool="runner",
            action="cover_on_health_drop",
            status="ok",
            details={"mode": mode_name},
        )

    def _run_loop(
        self,
        mode_name: str,
        config: LoopConfig,
        on_tick: Callable[[FrameState], Optional[str]],
        initial_keys: Iterable[str] = (),
        initial_buttons: Iterable[str] = (),
        on_health_drop: Optional[Callable[[], None]] = None,
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
                    if ticks == 0 and hasattr(self.input.backend, "set_anchor_bounds"):
                        wb = cap.window.bounds
                        if wb.width > 0 and wb.height > 0:
                            self.input.backend.set_anchor_bounds(wb.x, wb.y, wb.width, wb.height)
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
                if stop_reason == "health_drop" and on_health_drop is not None:
                    on_health_drop()

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

    def activate_wine_capture(
        self,
        button: str = "right",
        hold_s: float = 0.04,
        timeout_s: Optional[float] = None,
    ) -> WindowInfo:
        """Center cursor inside the GTA 5 window and click `button` so Wine captures keyboard/mouse input.

        WARNING: Avoid `button="right"` inside the Pause Menu (`right` backs out of menus).
        """
        win = find_game_window(backend=self._window_backend)
        wb = win.bounds
        self.input.click_window_center(
            window_bounds=(wb.x, wb.y, wb.width, wb.height),
            button=button,
            hold_s=hold_s,
            timeout_s=timeout_s,
        )
        self._logger.log(
            tool="runner",
            action="activate_wine_capture",
            status="ok",
            details={"window_id": win.window_id, "button": button, "bounds": [wb.x, wb.y, wb.width, wb.height]},
        )
        return win

    def _sleep_chunked(self, duration_s: float, step: float = 0.1) -> None:
        rem = max(0.0, float(duration_s))
        while rem > 0:
            dt = min(step, rem)
            self._sleep(dt)
            rem -= dt

    def _perform_on_foot_sidestep(
        self,
        attempt: int,
        last_bearing: float,
        duration_s: float,
        vault: bool,
    ) -> None:
        if last_bearing < -0.05:
            sidestep_key = "a"
        elif last_bearing > 0.05:
            sidestep_key = "d"
        else:
            sidestep_key = "d" if attempt % 2 == 1 else "a"
        self._logger.log(
            tool="runner",
            action="on_foot_stuck_recovery",
            status="ok",
            details={
                "attempt": attempt,
                "sidestep_key": sidestep_key,
                "duration_s": duration_s,
                "vault": vault,
            },
        )
        self.input.key_down(sidestep_key)
        try:
            if vault:
                self._tap_key_preserving_held("space", hold_s=0.06)
            self._sleep_chunked(duration_s)
        finally:
            self.input.key_up(sidestep_key)

    def _perform_on_foot_backpedal(self, attempt: int, duration_s: float = 0.4) -> None:
        """Briefly backpedal diagonally away from close exposed fire."""
        strafe_key = "a" if attempt % 2 == 0 else "d"
        self._logger.log(
            tool="runner",
            action="on_foot_backpedal",
            status="ok",
            details={
                "attempt": attempt,
                "strafe_key": strafe_key,
                "duration_s": duration_s,
            },
        )
        self.input.key_down("s")
        try:
            self.input.key_down(strafe_key)
            try:
                self._sleep_chunked(duration_s)
            finally:
                self.input.key_up(strafe_key)
        finally:
            self.input.key_up("s")

    @staticmethod
    def _resolve_route_target_and_bearing(
        mm: MinimapObservation,
        target_color: str,
    ):
        targets = mm.targets.get(target_color, ())
        if targets:
            target = targets[0]
        elif target_color in ("yellow", "purple"):
            target = mm.primary_objective
        else:
            target = None

        bearing: Optional[float] = None
        if target_color == "yellow" and mm.yellow_route_bearing is not None:
            bearing = mm.yellow_route_bearing
        elif target_color == "purple" and mm.purple_route_bearing is not None:
            bearing = mm.purple_route_bearing
        elif target is not None:
            bearing = target.bearing
        return target, bearing

    def follow_route_on_foot(
        self,
        sprint: bool = False,
        target_color: str = "yellow",
        arrive_distance: float = 0.08,
        steer_gain: float = 160.0,
        deadband: float = 0.05,
        stop_on_subtitle_keywords: Sequence[str] = ("return to", "wait for", "stay with", "follow"),
        enable_stuck_recovery: bool = True,
        stuck_window_s: float = 1.5,
        stuck_motion_threshold: float = 2.5,
        unstuck_duration_s: float = 0.55,
        vault_on_stuck: bool = True,
        cover_on_health_drop: bool = True,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Walk/sprint toward minimap route/objective with crew-tether subtitle stop and obstacle recovery."""
        cfg = config or LoopConfig(max_duration_s=10.0, hz=10.0)
        keys = ("w", "lshift") if sprint else ("w",)
        norm_keywords = tuple(kw.strip().lower() for kw in stop_on_subtitle_keywords if kw.strip())
        prev_gray: Optional[np.ndarray] = None
        last_motion_check_t: Optional[float] = None
        low_motion_elapsed_s = 0.0
        last_bearing = 0.0
        recovery_count = 0

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal prev_gray, last_motion_check_t, low_motion_elapsed_s, last_bearing, recovery_count
            if state.hud is not None and norm_keywords:
                combined_hud = f"{state.hud.subtitle_or_objective} {state.hud.help_prompt}".lower()
                for kw in norm_keywords:
                    if kw in combined_hud:
                        return f"subtitle_keyword:{kw}"

            mm = state.minimap
            target, bearing = self._resolve_route_target_and_bearing(mm, target_color)
            if target is not None and target.distance <= arrive_distance:
                return "target_reached"

            if bearing is not None:
                last_bearing = bearing

            if enable_stuck_recovery:
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
                    recovery_count += 1
                    allow_vault = vault_on_stuck and (mm.nearest_enemy is None) and (
                        not mm.minimap_visible or mm.health_ratio >= 0.65
                    )
                    self._perform_on_foot_sidestep(recovery_count, last_bearing, unstuck_duration_s, allow_vault)
                    low_motion_elapsed_s = 0.0
                    prev_gray = None
                    last_motion_check_t = None
                    return None

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
            on_health_drop=(lambda: self._snap_cover_on_health_drop("follow_route_on_foot")) if cover_on_health_drop else None,
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

    def advance_under_fire(
        self,
        sprint: bool = False,
        target_color: str = "yellow",
        arrive_distance: float = 0.08,
        ally_tether_distance: float = 0.28,
        move_phase_s: float = 2.2,
        engage_phase_s: float = 3.2,
        engage_distance: float = 0.65,
        close_enemy_hold_distance: float = 0.28,
        min_engage_health: float = 0.25,
        recover_health: float = 0.42,
        recover_target_health: float = 0.45,
        recover_wait_s: float = 2.5,
        use_special: bool = True,
        recoil_comp_dy: int = 1,
        aim_settle_s: float = 0.15,
        exposed_recover_drop_threshold: float = 0.04,
        steer_gain: float = 160.0,
        engage_steer_gain: float = 200.0,
        deadband: float = 0.06,
        bearing_deadband: float = 0.025,
        engage_align_bearing: float = 0.06,
        stop_on_subtitle_keywords: Sequence[str] = ("get in", "enter", "vehicle", "car"),
        tether_subtitle_keywords: Sequence[str] = ("return to", "wait for", "stay with", "follow"),
        enable_stuck_recovery: bool = True,
        stuck_window_s: float = 0.9,
        stuck_motion_threshold: float = 2.5,
        unstuck_duration_s: float = 0.55,
        vault_on_stuck: bool = True,
        cover_on_timeout: bool = True,
        scan_pattern: Sequence[tuple[int, int]] = DEFAULT_RETICLE_SWEEP,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Closed-loop shoot-and-move advance (up to ~30s) alternating move, engage, and cover-recover phases."""
        base_cfg = config or LoopConfig(
            max_duration_s=30.0,
            hz=10.0,
            hud_check_interval_s=0.25,
            stop_on_banner=True,
            stop_on_help_change=False,
            stop_on_subtitle_change=False,
            health_drop_threshold=None,
        )
        cfg = LoopConfig(
            max_duration_s=base_cfg.max_duration_s,
            hz=base_cfg.hz,
            hud_check_interval_s=base_cfg.hud_check_interval_s,
            stop_on_banner=base_cfg.stop_on_banner,
            stop_on_help_change=False,
            stop_on_subtitle_change=False,
            health_drop_threshold=None,
        )
        norm_stop_kws = tuple(kw.strip().lower() for kw in stop_on_subtitle_keywords if kw.strip())
        norm_tether_kws = tuple(kw.strip().lower() for kw in tether_subtitle_keywords if kw.strip())

        phase = "move"
        phase_end_t = move_phase_s
        min_move_until_t = 0.0
        engage_cooldown_until_t = 0.0
        recover_cooldown_until_t = 0.0
        tether_active_until_t = 0.0
        recover_start_health: Optional[float] = None
        special_triggered = False

        prev_gray: Optional[np.ndarray] = None
        last_motion_check_t: Optional[float] = None
        low_motion_elapsed_s = 0.0
        last_bearing = 0.0
        recovery_count = 0

        sweep_idx = 0
        sweep_offset_x = 0
        sweep_offset_y = 0
        aligned_ticks = 0

        def _set_key(key: str, want_down: bool) -> None:
            is_down = key in self.input.held_keys
            if want_down and not is_down:
                self.input.key_down(key)
            elif not want_down and is_down:
                self.input.key_up(key)

        def _set_movement(want_move: bool, use_sprint: bool = False) -> None:
            _set_key("w", want_move)
            _set_key("lshift", want_move and use_sprint)

        def _recenter_sweep() -> None:
            nonlocal sweep_idx, sweep_offset_x, sweep_offset_y
            if sweep_offset_x != 0 or sweep_offset_y != 0:
                self.input.mouse_delta(-sweep_offset_x, -sweep_offset_y, steps=2, duration_s=0.03)
                sweep_offset_x = 0
                sweep_offset_y = 0
            sweep_idx = 0

        def _reset_motion_tracker() -> None:
            nonlocal prev_gray, last_motion_check_t, low_motion_elapsed_s
            prev_gray = None
            last_motion_check_t = None
            low_motion_elapsed_s = 0.0

        def _set_aim(want_aim: bool) -> None:
            is_aiming = "right" in self.input.held_buttons
            if want_aim and not is_aiming:
                self.input.mouse_down("right")
                if aim_settle_s > 0:
                    self._sleep(aim_settle_s)
            elif not want_aim and is_aiming:
                _recenter_sweep()
                self.input.mouse_up("right")

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal phase, phase_end_t, min_move_until_t, engage_cooldown_until_t, recover_cooldown_until_t, tether_active_until_t, recover_start_health, special_triggered, prev_gray, last_motion_check_t, low_motion_elapsed_s, last_bearing, recovery_count, sweep_idx, sweep_offset_x, sweep_offset_y, aligned_ticks
            now_t = state.elapsed_s
            mm = state.minimap

            if state.hud is not None:
                combined_hud = f"{state.hud.subtitle_or_objective} {state.hud.help_prompt}".lower()
                for kw in norm_stop_kws:
                    if kw in combined_hud:
                        _set_aim(False)
                        _set_movement(False)
                        return f"objective_prompt:{kw}"
                for kw in norm_tether_kws:
                    if kw in combined_hud:
                        tether_active_until_t = max(tether_active_until_t, now_t + 2.5)
                        break

            if use_special and not special_triggered and mm.minimap_visible and mm.special_ratio >= 0.20:
                if mm.nearest_enemy is not None or mm.health_ratio < 0.85:
                    self._tap_key_preserving_held("capslock", hold_s=0.08)
                    special_triggered = True

            target, route_bearing = self._resolve_route_target_and_bearing(mm, target_color)
            if target is not None and target.distance <= arrive_distance:
                _set_aim(False)
                _set_movement(False)
                return "target_reached"

            blue_targets = mm.targets.get("blue", ())
            blue_ally = blue_targets[0] if blue_targets else None
            enemy = mm.nearest_enemy

            def _resume_move_from_recover(*, tether: bool = False, cooldown_s: float = 0.0) -> None:
                nonlocal phase, phase_end_t, min_move_until_t, engage_cooldown_until_t
                nonlocal recover_cooldown_until_t, recover_start_health, tether_active_until_t
                phase = "move"
                if tether:
                    tether_active_until_t = max(tether_active_until_t, now_t + 2.5)
                phase_end_t = now_t + move_phase_s
                min_move_until_t = now_t + move_phase_s
                engage_cooldown_until_t = now_t + move_phase_s
                if cooldown_s > 0.0:
                    recover_cooldown_until_t = now_t + cooldown_s
                recover_start_health = None
                _reset_motion_tracker()

            def _enter_engage_phase(*, cooldown_s: float = 0.0) -> None:
                nonlocal phase, phase_end_t, aligned_ticks, sweep_idx, sweep_offset_x, sweep_offset_y
                nonlocal recover_cooldown_until_t, recover_start_health
                _set_movement(False)
                phase = "engage"
                phase_end_t = now_t + engage_phase_s
                aligned_ticks = 0
                sweep_idx = 0
                sweep_offset_x = 0
                sweep_offset_y = 0
                if cooldown_s > 0.0:
                    recover_cooldown_until_t = now_t + cooldown_s
                recover_start_health = None
                _reset_motion_tracker()
                _set_aim(True)

            # Low-health cover/recover phase (critical < 0.25 HP bypasses recover cooldown)
            if (
                phase != "recover"
                and mm.minimap_visible
                and (
                    mm.health_ratio < 0.25
                    or (now_t >= recover_cooldown_until_t and mm.health_ratio < recover_health)
                )
            ):
                _set_aim(False)
                _set_movement(False)
                if mm.health_ratio < 0.25 and (blue_ally is None or blue_ally.distance <= 0.14):
                    self._tap_key_preserving_held("q", hold_s=0.08)
                    self._logger.log(
                        tool="runner",
                        action="advance_exposed",
                        status="ok",
                        details={"health_ratio": round(mm.health_ratio, 3)},
                    )
                    return "low_health"
                if (
                    blue_ally is not None
                    and blue_ally.distance > 0.14
                    and (
                        mm.health_ratio < 0.25
                        or (enemy is not None and enemy.distance < 0.45 and abs(blue_ally.bearing) > 0.35)
                    )
                ):
                    _resume_move_from_recover(tether=True, cooldown_s=1.5)
                    self._logger.log(
                        tool="runner",
                        action="advance_retreat",
                        status="ok",
                        details={
                            "health_ratio": round(mm.health_ratio, 3),
                            "ally_distance": round(blue_ally.distance, 3),
                            "enemy_distance": round(enemy.distance, 3) if enemy is not None else None,
                        },
                    )
                else:
                    phase = "recover"
                    phase_end_t = now_t + recover_wait_s
                    recover_cooldown_until_t = phase_end_t + 3.0
                    recover_start_health = mm.health_ratio
                    _reset_motion_tracker()
                    self._tap_key_preserving_held("q", hold_s=0.08)
                    self._logger.log(
                        tool="runner",
                        action="advance_recover",
                        status="ok",
                        details={"health_ratio": round(mm.health_ratio, 3), "wait_s": recover_wait_s},
                    )
                    return None

            if phase == "recover":
                exposed_drop = (
                    mm.minimap_visible
                    and recover_start_health is not None
                    and (recover_start_health - mm.health_ratio) >= exposed_recover_drop_threshold
                )
                critical_hp = mm.minimap_visible and mm.health_ratio < 0.25
                if exposed_drop or critical_hp:
                    if blue_ally is not None and blue_ally.distance > 0.14:
                        _resume_move_from_recover(tether=True, cooldown_s=1.5)
                        self._logger.log(
                            tool="runner",
                            action="advance_retreat",
                            status="ok",
                            details={"health_ratio": round(mm.health_ratio, 3), "ally_distance": round(blue_ally.distance, 3)},
                        )
                    elif critical_hp:
                        _set_aim(False)
                        _set_movement(False)
                        self._logger.log(
                            tool="runner",
                            action="advance_exposed",
                            status="ok",
                            details={"health_ratio": round(mm.health_ratio, 3)},
                        )
                        return "low_health"
                    elif enemy is not None and enemy.distance < 0.45:
                        recovery_count += 1
                        self._perform_on_foot_backpedal(attempt=recovery_count, duration_s=0.4)
                        if enemy.distance <= engage_distance:
                            _enter_engage_phase(cooldown_s=2.0)
                        else:
                            _resume_move_from_recover(cooldown_s=1.5)
                if phase == "recover":
                    if now_t < phase_end_t and (not mm.minimap_visible or mm.health_ratio < recover_target_health):
                        return None
                    _resume_move_from_recover()

            need_ally_tether = (now_t < tether_active_until_t) or (
                blue_ally is not None and blue_ally.distance > ally_tether_distance
            )
            if need_ally_tether and blue_ally is not None:
                nav_bearing = blue_ally.bearing
            elif route_bearing is not None:
                nav_bearing = route_bearing
            elif blue_ally is not None and blue_ally.distance > 0.14:
                nav_bearing = blue_ally.bearing
            elif enemy is not None and enemy.distance > 0.35:
                nav_bearing = enemy.bearing
            else:
                nav_bearing = None

            can_engage = (
                enemy is not None
                and enemy.distance <= engage_distance
                and (not mm.minimap_visible or mm.health_ratio >= min_engage_health)
                and not need_ally_tether
            )
            hold_engage_threat = (
                enemy is not None
                and (
                    enemy.distance <= close_enemy_hold_distance
                    or (len(mm.targets["red"]) >= 3 and enemy.distance <= 0.40)
                )
                and (not mm.minimap_visible or mm.health_ratio >= min_engage_health)
                and not need_ally_tether
            )

            if phase == "move" and can_engage and (
                hold_engage_threat or (now_t >= min_move_until_t and now_t >= engage_cooldown_until_t)
            ):
                _enter_engage_phase()
            elif phase == "engage" and (
                enemy is None or enemy.distance > engage_distance or need_ally_tether or now_t >= phase_end_t
            ):
                if phase == "engage" and hold_engage_threat and now_t >= phase_end_t:
                    phase_end_t = now_t + 1.5
                else:
                    _set_aim(False)
                    phase = "move"
                    phase_end_t = now_t + move_phase_s
                    min_move_until_t = now_t + move_phase_s
                    engage_cooldown_until_t = now_t + move_phase_s
                    _reset_motion_tracker()

            if phase == "engage":
                _set_movement(False)
                _set_aim(True)
                reticle = detect_red_reticle(state.capture.image)
                if reticle.is_red:
                    aligned_ticks = 0
                    self._fire_bursts("left", bursts=3, hold_s=0.03, gap_s=0.025, reason="red_reticle")
                    if recoil_comp_dy > 0:
                        self.input.mouse_delta(0, int(recoil_comp_dy), steps=1, duration_s=0.01)
                    return None

                if enemy is not None:
                    if abs(enemy.bearing) > bearing_deadband:
                        dx = self._ads_steer_dx(enemy.bearing, steer_gain=engage_steer_gain)
                        if dx != 0:
                            self._steer_camera(dx, 0)
                    if abs(enemy.bearing) <= engage_align_bearing:
                        aligned_ticks += 1
                        if aligned_ticks % 5 == 0:
                            self._fire_bursts("left", bursts=2, hold_s=0.03, gap_s=0.03, reason="suppressive")
                            if recoil_comp_dy > 0:
                                self.input.mouse_delta(0, int(recoil_comp_dy), steps=1, duration_s=0.01)
                    else:
                        aligned_ticks = 0

                if scan_pattern:
                    dx, dy = scan_pattern[sweep_idx % len(scan_pattern)]
                    sweep_idx += 1
                    if dx != 0 or dy != 0:
                        self.input.mouse_delta(dx, dy, steps=2, duration_s=0.03)
                        sweep_offset_x += dx
                        sweep_offset_y += dy
                return None

            # Move phase
            _set_aim(False)
            retreating = now_t < tether_active_until_t
            _set_movement(True, use_sprint=(retreating or (sprint and not need_ally_tether)))
            if nav_bearing is not None:
                last_bearing = nav_bearing

            if enable_stuck_recovery and (not mm.minimap_visible or mm.health_ratio >= recover_health):
                cur_gray = _road_gray_sample(state.capture.image)
                if prev_gray is not None and last_motion_check_t is not None:
                    dt = max(0.0, now_t - last_motion_check_t)
                    motion = _road_motion_score(prev_gray, cur_gray)
                    if motion < stuck_motion_threshold:
                        low_motion_elapsed_s += dt
                    else:
                        low_motion_elapsed_s = 0.0
                prev_gray = cur_gray
                last_motion_check_t = now_t

                if low_motion_elapsed_s >= stuck_window_s and (cfg.max_duration_s - now_t) > 0.25:
                    recovery_count += 1
                    allow_vault = vault_on_stuck and (enemy is None) and (not mm.minimap_visible or mm.health_ratio >= 0.65)
                    self._perform_on_foot_sidestep(recovery_count, last_bearing, unstuck_duration_s, allow_vault)
                    _reset_motion_tracker()
                    return None

            if nav_bearing is not None and abs(nav_bearing) > deadband:
                dx = int(round(max(-240.0, min(240.0, nav_bearing * steer_gain))))
                if dx != 0:
                    self._steer_camera(dx, 0)
            return None

        try:
            result = self._run_loop(
                mode_name="advance_under_fire",
                config=cfg,
                on_tick=_on_tick,
            )
        finally:
            _recenter_sweep()
        if cover_on_timeout and result.stop_reason == "timeout" and phase != "recover":
            self._tap_key_preserving_held("q", hold_s=0.08)
            self._logger.log(
                tool="runner",
                action="advance_timeout_cover",
                status="ok",
                details={
                    "health_ratio": round(result.last_minimap.health_ratio, 3) if result.last_minimap else None,
                    "phase": phase,
                },
            )
        return result

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
            self._sleep_chunked(reverse_duration_s)
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

    @staticmethod
    def _ads_steer_dx(
        bearing: float,
        steer_gain: float = 200.0,
        min_step_px: int = 6,
        max_step_px: int = 65,
    ) -> int:
        if bearing == 0.0:
            return 0
        raw = bearing * steer_gain
        if abs(bearing) <= 0.35:
            mag = max(min_step_px, min(max_step_px, int(round(abs(raw)))))
            return mag if bearing > 0 else -mag
        return int(round(max(-220.0, min(220.0, raw))))

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

    def _fire_bursts(
        self,
        fire_button: str,
        bursts: int = 1,
        hold_s: float = 0.03,
        gap_s: float = 0.03,
        reason: Optional[str] = None,
    ) -> None:
        count = max(1, bursts)
        for idx in range(count):
            self.input.mouse_down(fire_button)
            try:
                self._sleep(hold_s)
            finally:
                self.input.mouse_up(fire_button)
            if idx + 1 < count and gap_s > 0:
                self._sleep(gap_s)
        self._logger.log(
            tool="runner",
            action="fire_burst",
            status="ok",
            details={"button": fire_button, "bursts": count, "reason": reason or "burst"},
        )

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
        steer_gain: float = 200.0,
        bearing_deadband: float = 0.025,
        aim_settle_s: float = 0.18,
        scan_pattern: Sequence[tuple[int, int]] = DEFAULT_RETICLE_SWEEP,
        empty_scans_to_stop: Optional[int] = None,
        fallback_fire_on_aligned: bool = False,
        fallback_fire_interval_ticks: int = 6,
        fallback_fire_max_bearing: float = 0.08,
        reacquire_aim_interval_ticks: Optional[int] = None,
        cover_on_health_drop: bool = False,
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
                    dx = self._ads_steer_dx(enemy.bearing, steer_gain=steer_gain)
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
            elif abs(enemy.bearing) <= fallback_fire_max_bearing:
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
            else:
                no_enemy_ticks = 0
                aligned_without_lock_ticks = 0

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
            on_health_drop=(lambda: self._snap_cover_on_health_drop("aim_until_red_reticle")) if cover_on_health_drop else None,
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
        cover_on_health_drop: bool = True,
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
            cover_on_health_drop=cover_on_health_drop,
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
        steer_gain: float = 200.0,
        bearing_deadband: float = 0.025,
        suppressive_max_bearing: float = 0.12,
        aim_settle_s: float = 0.18,
        recoil_comp_dy: int = 1,
        exposed_duck_drop_threshold: float = 0.12,
        flank_recovery_min_health: float = 0.55,
        max_unclamp_attempts: int = 1,
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
        duck_reason_heal = False
        duck_start_health: Optional[float] = None
        peak_health: Optional[float] = None
        sweep_idx = 0
        sweep_offset_x = 0
        sweep_offset_y = 0
        shots_fired = 0
        no_enemy_ticks = 0
        no_enemy_since_t: Optional[float] = None
        aligned_ticks = 0
        steer_cadence_ticks = 0
        large_steer_ticks = 0
        clamped_ticks = 0
        unclamp_attempts = 0
        flank_stage2_done = False
        toggled_cover_for_flank = False
        pre_flank_bearing: Optional[float] = None
        last_clamped_bearing: Optional[float] = None

        def _recenter_sweep() -> None:
            nonlocal sweep_idx, sweep_offset_x, sweep_offset_y
            if sweep_offset_x != 0 or sweep_offset_y != 0:
                self.input.mouse_delta(-sweep_offset_x, -sweep_offset_y, steps=2, duration_s=0.03)
                sweep_offset_x = 0
                sweep_offset_y = 0
            sweep_idx = 0

        def _set_peek(want_peek: bool, now_t: float) -> None:
            nonlocal is_peeking, peek_start_t
            if want_peek and not is_peeking:
                self.input.mouse_down(aim_button)
                is_peeking = True
                peek_start_t = now_t
                if aim_settle_s > 0:
                    self._sleep(aim_settle_s)
            elif not want_peek and is_peeking:
                _recenter_sweep()
                self.input.mouse_up(aim_button)
                is_peeking = False

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal special_triggered, duck_until_t, duck_logged, duck_reason_heal, duck_start_health, peak_health, sweep_idx, sweep_offset_x, sweep_offset_y, shots_fired, no_enemy_ticks, no_enemy_since_t, aligned_ticks, steer_cadence_ticks, large_steer_ticks, clamped_ticks, unclamp_attempts, flank_stage2_done, toggled_cover_for_flank, pre_flank_bearing, last_clamped_bearing
            mm = state.minimap
            now_t = state.elapsed_s

            if use_special and not special_triggered and mm.minimap_visible and mm.special_ratio >= 0.20:
                self._tap_key_preserving_held("capslock", hold_s=0.08)
                special_triggered = True

            # Check in-loop heal/duck trigger (low HP or rapid drop when HP < 0.70 / low armor)
            if mm.minimap_visible:
                if (
                    duck_reason_heal
                    and now_t < duck_until_t
                    and duck_start_health is not None
                    and (duck_start_health - mm.health_ratio) >= exposed_duck_drop_threshold
                ):
                    self._logger.log(
                        tool="runner",
                        action="cover_exposed",
                        status="ok",
                        details={
                            "duck_start_health": round(duck_start_health, 3),
                            "health_ratio": round(mm.health_ratio, 3),
                        },
                    )
                    _set_peek(False, now_t)
                    return "exposed_in_cover"

                if peak_health is None or mm.health_ratio > peak_health:
                    peak_health = mm.health_ratio
                rapid_drop = (peak_health - mm.health_ratio) >= rapid_drop_duck_threshold and (
                    mm.health_ratio < 0.70 or mm.armor_ratio <= 0.05
                )
                if mm.health_ratio < min_peek_health or rapid_drop:
                    _set_peek(False, now_t)
                    duck_until_t = max(duck_until_t, now_t + eff_heal_wait_s)
                    if not duck_reason_heal:
                        duck_reason_heal = True
                        duck_start_health = mm.health_ratio
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
            duck_reason_heal = False
            duck_start_health = None

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
                steer_cadence_ticks = 0
                large_steer_ticks = 0
                clamped_ticks = 0
                last_clamped_bearing = None
                self._fire_bursts(fire_button, bursts=3, hold_s=0.03, gap_s=0.025, reason="red_reticle")
                if recoil_comp_dy > 0:
                    self.input.mouse_delta(0, int(recoil_comp_dy), steps=1, duration_s=0.01)
                shots_fired += 1
                if max_shots is not None and shots_fired >= max_shots:
                    _recenter_sweep()
                    return "reticle_shots_done"
                return None

            enemy = mm.nearest_enemy
            if enemy is None:
                steer_cadence_ticks = 0
                large_steer_ticks = 0
                clamped_ticks = 0
                last_clamped_bearing = None
                no_enemy_ticks += 1
                if no_enemy_since_t is None:
                    no_enemy_since_t = now_t
                aligned_ticks = 0
                if stop_when_no_enemies:
                    if empty_scans_to_stop is not None and no_enemy_ticks >= empty_scans_to_stop:
                        _recenter_sweep()
                        return "enemies_cleared"
                    if empty_scans_to_stop is None and (now_t - no_enemy_since_t) >= no_enemies_clear_s:
                        _recenter_sweep()
                        return "enemies_cleared"
            else:
                no_enemy_ticks = 0
                no_enemy_since_t = None

                # Flanking / cover-clamped enemy (> 0.45): 2-stage recovery (free-look+sidestep, then q cover-break + aim turn) if healthy.
                if abs(enemy.bearing) > 0.45:
                    if (
                        last_clamped_bearing is not None
                        and (enemy.bearing * last_clamped_bearing) > 0
                        and abs(enemy.bearing - last_clamped_bearing) < 0.06
                    ):
                        clamped_ticks += 1
                    else:
                        clamped_ticks = 1
                    last_clamped_bearing = enemy.bearing

                    healthy_for_flank = not mm.minimap_visible or mm.health_ratio >= flank_recovery_min_health
                    if clamped_ticks >= 5 and unclamp_attempts < max_unclamp_attempts and healthy_for_flank:
                        _set_peek(False, now_t)
                        dx = int(round(max(-320.0, min(320.0, enemy.bearing * 450.0))))
                        if dx != 0:
                            self._steer_camera(dx, 0)
                        sidestep_key = "a" if enemy.bearing < 0 else "d"
                        self._tap_key_preserving_held(sidestep_key, hold_s=0.16)
                        pre_flank_bearing = enemy.bearing
                        unclamp_attempts += 1
                        clamped_ticks = 0
                        last_clamped_bearing = None
                        aligned_ticks = 0
                        self._logger.log(
                            tool="runner",
                            action="cover_flank_recovery",
                            status="ok",
                            details={
                                "stage": 1,
                                "attempt": unclamp_attempts,
                                "bearing": round(enemy.bearing, 3),
                                "sidestep_key": sidestep_key,
                                "dx": dx,
                            },
                        )
                        return None

                    if (
                        unclamp_attempts > 0
                        and not flank_stage2_done
                        and healthy_for_flank
                        and pre_flank_bearing is not None
                        and (abs(pre_flank_bearing) - abs(enemy.bearing)) < 0.12
                    ):
                        flank_stage2_done = True
                        self._tap_key_preserving_held("q", hold_s=0.08)
                        toggled_cover_for_flank = True
                        _set_peek(True, now_t)
                        dx = int(round(max(-320.0, min(320.0, enemy.bearing * 450.0))))
                        if dx != 0:
                            self._steer_camera(dx, 0)
                        clamped_ticks = 0
                        last_clamped_bearing = None
                        aligned_ticks = 0
                        self._logger.log(
                            tool="runner",
                            action="cover_flank_recovery",
                            status="ok",
                            details={
                                "stage": 2,
                                "attempt": unclamp_attempts,
                                "bearing": round(enemy.bearing, 3),
                                "dx": dx,
                            },
                        )
                        return None

                    abort_threshold = 2 if flank_stage2_done else (3 if unclamp_attempts > 0 else 8)
                    if clamped_ticks >= abort_threshold:
                        self._logger.log(
                            tool="runner",
                            action="cover_clamped",
                            status="ok",
                            details={"bearing": round(enemy.bearing, 3)},
                        )
                        _set_peek(False, now_t)
                        if toggled_cover_for_flank:
                            self._tap_key_preserving_held("q", hold_s=0.08)
                            toggled_cover_for_flank = False
                        return "cover_clamped"
                else:
                    clamped_ticks = 0
                    last_clamped_bearing = None
                    toggled_cover_for_flank = False

                # Keep aim held in cover (releasing right pulls back into cover and resets cover yaw).
                if abs(enemy.bearing) > bearing_deadband:
                    steer_cadence_ticks += 1
                    if abs(enemy.bearing) > 0.35:
                        large_steer_ticks += 1
                        if large_steer_ticks <= 2 or large_steer_ticks % 3 == 0:
                            dx = int(round(max(-220.0, min(220.0, enemy.bearing * steer_gain))))
                            if dx != 0:
                                if abs(dx) >= 60:
                                    _recenter_sweep()
                                self._steer_camera(dx, 0)
                        if large_steer_ticks <= 2:
                            aligned_ticks = 0
                            return None
                    else:
                        large_steer_ticks = 0
                        # In-FOV yaw tracking: nudge smoothly every other tick without resetting sweep/fire.
                        if steer_cadence_ticks % 2 == 1:
                            dx = self._ads_steer_dx(enemy.bearing, steer_gain=steer_gain)
                            if dx != 0:
                                self._steer_camera(dx, 0)
                else:
                    steer_cadence_ticks = 0
                    large_steer_ticks = 0

                if abs(enemy.bearing) <= suppressive_max_bearing:
                    aligned_ticks += 1
                    if aligned_ticks % 3 == 0:
                        self._fire_bursts(fire_button, bursts=2, hold_s=0.03, gap_s=0.03, reason="suppressive")
                        if recoil_comp_dy > 0:
                            self.input.mouse_delta(0, int(recoil_comp_dy), steps=1, duration_s=0.01)
                        shots_fired += 1
                        if max_shots is not None and shots_fired >= max_shots:
                            _recenter_sweep()
                            return "suppressive_shots_done"
                else:
                    aligned_ticks = 0

            if scan_pattern:
                dx, dy = scan_pattern[sweep_idx % len(scan_pattern)]
                sweep_idx += 1
                if dx != 0 or dy != 0:
                    self.input.mouse_delta(dx, dy, steps=2, duration_s=0.03)
                    sweep_offset_x += dx
                    sweep_offset_y += dy
            return None

        try:
            return self._run_loop(
                mode_name="cover_peek_combat",
                config=cfg,
                on_tick=_on_tick,
            )
        finally:
            _recenter_sweep()
