"""Fast closed-loop real-time action runner (~8-12 Hz) with automatic safety stops."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from gameplayer.hud import HudObservation, OCRBackend, inspect_frame
from gameplayer.input import InputController
from gameplayer.minimap import MinimapObservation, inspect_minimap
from gameplayer.telemetry import TelemetryLogger, get_default_logger
from gameplayer.window import CaptureResult, WindowBackend, capture_game_window


@dataclass(frozen=True)
class LoopConfig:
    max_duration_s: float = 8.0
    hz: float = 10.0
    hud_check_interval_s: float = 0.25
    stop_on_banner: bool = True
    stop_on_help_change: bool = True
    stop_on_subtitle_change: bool = False
    health_drop_threshold: float = 0.15


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

                    if mm.minimap_visible:
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
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Walk/sprint toward minimap route/objective using proportional mouse camera steering."""
        cfg = config or LoopConfig(max_duration_s=10.0, hz=10.0)
        keys = ("w", "lshift") if sprint else ("w",)

        def _on_tick(state: FrameState) -> Optional[str]:
            mm = state.minimap
            targets = mm.targets.get(target_color, ())
            target = targets[0] if targets else mm.primary_objective
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
                    self.input.mouse_delta(dx, 0, steps=1, duration_s=0.0)
            return None

        return self._run_loop(
            mode_name="follow_route_on_foot",
            config=cfg,
            on_tick=_on_tick,
            initial_keys=keys,
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
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Drive along minimap GPS route by modulating throttle, steering (A/D), and sharp-turn brake."""
        cfg = config or LoopConfig(max_duration_s=12.0, hz=10.0)

        def _set_key(key: str, want_down: bool) -> None:
            is_down = key in self.input.held_keys
            if want_down and not is_down:
                self.input.key_down(key)
            elif not want_down and is_down:
                self.input.key_up(key)

        def _on_tick(state: FrameState) -> Optional[str]:
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

            if bearing is None:
                _set_key(steer_left_key, False)
                _set_key(steer_right_key, False)
                _set_key(brake_key, False)
                _set_key(throttle_key, True)
                return None

            _set_key(steer_left_key, bearing < -steer_deadband)
            _set_key(steer_right_key, bearing > steer_deadband)
            sharp = abs(bearing) >= sharp_turn_threshold
            _set_key(throttle_key, not sharp)
            _set_key(brake_key, sharp)
            return None

        return self._run_loop(
            mode_name="follow_gps_drive",
            config=cfg,
            on_tick=_on_tick,
            initial_keys=(throttle_key,),
        )

    def combat_burst_scan(
        self,
        aim_button: str = "right",
        fire_button: str = "left",
        steer_gain: float = 180.0,
        align_threshold: float = 0.12,
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
                    self.input.mouse_delta(dx, 0, steps=1, duration_s=0.0)
            else:
                self.input.mouse_down(fire_button)
                try:
                    self._sleep(0.03)
                finally:
                    self.input.mouse_up(fire_button)
            return None

        return self._run_loop(
            mode_name="combat_burst_scan",
            config=cfg,
            on_tick=_on_tick,
            initial_buttons=(aim_button,),
        )
