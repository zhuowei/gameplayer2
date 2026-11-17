from __future__ import annotations

from typing import Callable

import pytest
from PIL import Image, ImageDraw

from gameplayer.hud import OcrLine
from gameplayer.input import InputController
from gameplayer.runner import ActionRunner, LoopConfig, detect_red_reticle
from gameplayer.telemetry import TelemetryLogger
from gameplayer.window import WindowBounds, WindowInfo
from tests.test_minimap import make_synthetic_minimap_frame


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.now += dt


class ReplayWindowBackend:
    """Serves a scripted sequence of frames per capture call."""

    def __init__(self, frames_fn: Callable[[int], Image.Image]) -> None:
        self._frames_fn = frames_fn
        self.calls = 0
        self._win = WindowInfo(
            window_id=156,
            owner="GTA5.exe",
            title="Grand Theft Auto V",
            pid=1167,
            bounds=WindowBounds(0, 0, 1280, 720),
            layer=0,
            is_onscreen=True,
            is_frontmost=True,
        )

    def list_windows(self) -> list[WindowInfo]:
        return [self._win]

    def capture_window_image(self, window_id: int) -> Image.Image:
        img = self._frames_fn(self.calls)
        self.calls += 1
        return img

    def focus_window(self, window: WindowInfo) -> bool:
        raise AssertionError("Runner must never call focus_window")


class RecordingInputBackend:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.currently_held_keys: set[str] = set()
        self.currently_held_buttons: set[str] = set()

    def key_down(self, key: str) -> None:
        self.currently_held_keys.add(key)
        self.events.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.currently_held_keys.discard(key)
        self.events.append(("key_up", key))

    def mouse_down(self, button: str) -> None:
        self.currently_held_buttons.add(button)
        self.events.append(("mouse_down", button))

    def mouse_up(self, button: str) -> None:
        self.currently_held_buttons.discard(button)
        self.events.append(("mouse_up", button))

    def mouse_move_delta(
        self,
        dx: int,
        dy: int,
        held_buttons: frozenset[str] = frozenset(),
    ) -> None:
        self.events.append(("mouse_delta", (dx, dy, sorted(held_buttons))))


class ScriptedHudOCR:
    """Returns scripted OCR lines per `inspect_frame` call (4 regions per call)."""

    def __init__(self, step_fn: Callable[[int], dict[str, list[OcrLine]]]) -> None:
        self._step_fn = step_fn
        self._call = 0
        self._region_order = ("top_left_help", "top_right_status", "center_banner", "bottom_subtitle")

    def recognize_lines(self, image: Image.Image) -> list[OcrLine]:
        inspect_idx = self._call // 4
        region_name = self._region_order[self._call % 4]
        self._call += 1
        return self._step_fn(inspect_idx).get(region_name, [])


def test_runner_stops_early_on_wasted_banner_and_releases_keys(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)
    win_backend = ReplayWindowBackend(lambda idx: make_synthetic_minimap_frame(health_ratio=1.0))
    ocr = ScriptedHudOCR(
        lambda idx: {"center_banner": [OcrLine("WASTED", 0.99, (0.2, 0.4, 0.6, 0.2))]} if idx >= 2 else {}
    )

    runner = ActionRunner(
        input_controller=ctrl,
        window_backend=win_backend,
        ocr_backend=ocr,
        logger=TelemetryLogger(tmp_path / "t.jsonl"),
        clock=clock,
        sleep_fn=clock.sleep,
    )
    res = runner.run_until_event(
        hold_keys=["w", "lshift"],
        config=LoopConfig(max_duration_s=5.0, hz=10.0, hud_check_interval_s=0.0),
    )

    assert res.stop_reason == "banner:WASTED"
    assert res.banner == "WASTED"
    assert res.ticks == 3
    assert backend.currently_held_keys == set()
    assert backend.currently_held_buttons == set()


def test_runner_stops_on_help_prompt_change_and_health_drop(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)
    win_backend = ReplayWindowBackend(lambda idx: make_synthetic_minimap_frame(health_ratio=1.0))
    ocr = ScriptedHudOCR(
        lambda idx: {"top_left_help": [OcrLine("Press E to detonate.", 0.95, (0.1, 0.1, 0.7, 0.2))]} if idx >= 1 else {}
    )
    runner = ActionRunner(ctrl, win_backend, ocr, TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)
    res_help = runner.run_until_event(hold_keys=["w"], config=LoopConfig(max_duration_s=3.0, hz=10.0, hud_check_interval_s=0.0))
    assert res_help.stop_reason == "help_changed"
    assert backend.currently_held_keys == set()

    # Rapid health drop
    clock2 = FakeClock()
    backend2 = RecordingInputBackend()
    ctrl2 = InputController(backend=backend2, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock2, sleep_fn=clock2.sleep, register_signals=False)
    win_backend2 = ReplayWindowBackend(
        lambda idx: make_synthetic_minimap_frame(health_ratio=1.0 if idx < 2 else 0.55)
    )
    runner2 = ActionRunner(ctrl2, win_backend2, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock2, clock2.sleep)
    res_hp = runner2.run_until_event(hold_keys=["w"], config=LoopConfig(max_duration_s=3.0, hz=10.0, hud_check_interval_s=0.0))
    assert res_hp.stop_reason == "health_drop"
    assert backend2.currently_held_keys == set()


def test_follow_route_on_foot_steers_and_stops_on_target_reached(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)

    # Objective starts front-right (0.78, 0.45), then straight ahead (0.50, 0.45), then arrived (0.50, 0.74)
    positions = [(0.78, 0.45), (0.50, 0.45), (0.50, 0.74)]
    win_backend = ReplayWindowBackend(
        lambda idx: make_synthetic_minimap_frame(yellow_points=[positions[min(idx, len(positions) - 1)]])
    )
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.follow_route_on_foot(sprint=True, config=LoopConfig(max_duration_s=4.0, hz=10.0, hud_check_interval_s=0.0))
    assert res.stop_reason == "target_reached"
    assert backend.currently_held_keys == set()
    deltas = [ev[1][0] for ev in backend.events if ev[0] == "mouse_delta"]
    assert any(dx > 0 for dx in deltas)


def test_follow_gps_drive_steers_brakes_and_releases_all(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)

    # Tick 0: sharp right turn -> press D + S, release W
    # Tick 1: slight left -> press A + W, release D + S
    # Tick 2: target reached -> stop & release all
    frames = [
        make_synthetic_minimap_frame(yellow_points=[(0.90, 0.80)]),
        make_synthetic_minimap_frame(yellow_points=[(0.30, 0.45)]),
        make_synthetic_minimap_frame(yellow_points=[(0.50, 0.75)]),
    ]
    win_backend = ReplayWindowBackend(lambda idx: frames[min(idx, len(frames) - 1)])
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.follow_gps_drive(config=LoopConfig(max_duration_s=4.0, hz=10.0, hud_check_interval_s=0.0))
    assert res.stop_reason == "target_reached"
    assert ("key_down", "d") in backend.events
    assert ("key_down", "s") in backend.events
    assert ("key_down", "a") in backend.events
    assert backend.currently_held_keys == set()


def test_combat_burst_scan_keeps_aim_held_while_firing_and_stops_when_clear(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)

    # Tick 0: enemy to right -> pans right
    # Tick 1: enemy aligned straight ahead -> fires left click while keeping right held
    # Tick 2..3: no enemies -> stops after empty_scans_to_stop=2
    frames = [
        make_synthetic_minimap_frame(red_points=[(0.80, 0.45)]),
        make_synthetic_minimap_frame(red_points=[(0.50, 0.40)]),
        make_synthetic_minimap_frame(red_points=[]),
        make_synthetic_minimap_frame(red_points=[]),
    ]
    win_backend = ReplayWindowBackend(lambda idx: frames[min(idx, len(frames) - 1)])
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.combat_burst_scan(
        empty_scans_to_stop=2,
        config=LoopConfig(max_duration_s=5.0, hz=10.0, hud_check_interval_s=0.0),
    )
    assert res.stop_reason == "enemies_cleared"
    assert ("mouse_down", "right") in backend.events
    assert ("mouse_down", "left") in backend.events
    assert ("mouse_up", "left") in backend.events
    # Ensure right-click aim was only released at the very end, not dropped mid-burst
    right_ups = [i for i, ev in enumerate(backend.events) if ev == ("mouse_up", "right")]
    left_up_idx = backend.events.index(("mouse_up", "left"))
    assert len(right_ups) == 1 and right_ups[0] > left_up_idx
    assert backend.currently_held_buttons == set()


def _with_center_reticle(base: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    out = base.copy()
    draw = ImageDraw.Draw(out)
    cx, cy = out.width // 2, out.height // 2
    draw.rectangle((cx - 45, cy - 45, cx + 45, cy + 45), fill=(215, 215, 215))
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
    return out


def test_detect_red_reticle_distinguishes_hostile_dot_from_white_and_red_wall() -> None:
    base = Image.new("RGB", (1280, 720), (40, 40, 45))
    red_reticle = _with_center_reticle(base, (240, 35, 35))
    white_reticle = _with_center_reticle(base, (250, 250, 250))
    uniform_red_wall = Image.new("RGB", (1280, 720), (210, 30, 30))

    # Tiny 3x3 red Simple reticle dot over dim red-lit wall (real GTA 5 Prologue scenario)
    dim_red_wall = Image.new("RGB", (1280, 720), (78, 21, 12))
    draw = ImageDraw.Draw(dim_red_wall)
    cx, cy = 640, 360
    draw.rectangle((cx - 1, cy - 1, cx + 1, cy + 1), fill=(205, 38, 35))

    assert detect_red_reticle(red_reticle).is_red is True
    assert detect_red_reticle(dim_red_wall).is_red is True
    assert detect_red_reticle(white_reticle).is_red is False
    assert detect_red_reticle(uniform_red_wall).is_red is False


def test_hostage_shot_sweeps_then_fires_on_red_reticle(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)

    base = make_synthetic_minimap_frame(health_ratio=1.0)
    white_frame = _with_center_reticle(base, (245, 245, 245))
    red_frame = _with_center_reticle(base, (245, 30, 30))

    win_backend = ReplayWindowBackend(lambda idx: red_frame if idx >= 2 else white_frame)
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.hostage_shot(bursts=2, config=LoopConfig(max_duration_s=4.0, hz=12.0, hud_check_interval_s=0.0))
    assert res.stop_reason == "reticle_shot"
    assert backend.events.count(("mouse_down", "left")) == 2
    assert backend.events.count(("mouse_up", "left")) == 2
    assert backend.currently_held_buttons == set()


def test_shoot_enemies_steers_and_confirms_red_reticle(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)

    f0 = _with_center_reticle(make_synthetic_minimap_frame(red_points=[(0.82, 0.45)]), (245, 245, 245))
    f1 = _with_center_reticle(make_synthetic_minimap_frame(red_points=[(0.50, 0.45)]), (245, 30, 30))
    f2 = _with_center_reticle(make_synthetic_minimap_frame(red_points=[]), (245, 245, 245))
    frames = [f0, f1, f2, f2]

    win_backend = ReplayWindowBackend(lambda idx: frames[min(idx, len(frames) - 1)])
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.shoot_enemies(
        max_shots=3,
        empty_scans_to_stop=2,
        config=LoopConfig(max_duration_s=5.0, hz=10.0, hud_check_interval_s=0.0),
    )
    assert res.stop_reason == "enemies_cleared"
    assert ("mouse_down", "left") in backend.events
    assert backend.currently_held_buttons == set()


def test_shoot_enemies_fires_suppressive_bursts_when_aligned_without_red_reticle(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)

    # Distant enemy aligned on minimap (-0.05 bearing), white reticle (no red tint at 35m behind car)
    white_aligned = _with_center_reticle(make_synthetic_minimap_frame(red_points=[(0.48, 0.35)]), (245, 245, 245))
    win_backend = ReplayWindowBackend(lambda idx: white_aligned)
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.shoot_enemies(
        max_shots=1,
        empty_scans_to_stop=5,
        config=LoopConfig(max_duration_s=3.0, hz=10.0, hud_check_interval_s=0.0),
    )
    assert res.stop_reason == "suppressive_shots_done"
    assert ("mouse_down", "left") in backend.events
    assert backend.currently_held_buttons == set()


def test_cover_peek_combat_enters_cover_activates_special_and_ducks_on_low_health(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)

    f_healthy_enemy = _with_center_reticle(
        make_synthetic_minimap_frame(health_ratio=0.80, special_ratio=0.75, red_points=[(0.50, 0.40)]),
        (245, 30, 30),
    )
    f_low_hp = make_synthetic_minimap_frame(health_ratio=0.30, special_ratio=0.50, red_points=[(0.50, 0.40)])
    f_healed_clear = make_synthetic_minimap_frame(health_ratio=0.60, special_ratio=0.40, red_points=[])
    frames = [f_healthy_enemy, f_low_hp, f_low_hp, f_healed_clear, f_healed_clear]

    win_backend = ReplayWindowBackend(lambda idx: frames[min(idx, len(frames) - 1)])
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.cover_peek_combat(
        enter_cover=True,
        use_special=True,
        min_peek_health=0.45,
        duck_heal_s=0.25,
        max_shots=5,
        empty_scans_to_stop=2,
        config=LoopConfig(max_duration_s=5.0, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )
    assert res.stop_reason == "enemies_cleared"
    assert ("key_down", "q") in backend.events
    assert ("key_up", "q") in backend.events
    assert ("key_down", "capslock") in backend.events
    assert ("key_up", "capslock") in backend.events
    assert ("mouse_down", "right") in backend.events
    # Must duck (release right mouse) when health drops below min_peek_health
    assert backend.events.count(("mouse_up", "right")) >= 2
    assert backend.currently_held_buttons == set()
    assert backend.currently_held_keys == set()


def test_cover_peek_combat_in_loop_heal_resumes_then_clears_without_toggling_q_when_false(tmp_path) -> None:
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)

    # Rapid health drop from 0.90 -> 0.50 (above 0.45, but rapid drop >= 0.20) triggers in-loop duck/heal,
    # even when caller passes LoopConfig(health_drop_threshold=0.20) with exit_on_health_drop=False,
    # then resumes peeking and stops early via no_enemies_clear_s=0.25.
    f_high = _with_center_reticle(make_synthetic_minimap_frame(health_ratio=0.90, red_points=[(0.50, 0.40)]), (245, 30, 30))
    f_drop = make_synthetic_minimap_frame(health_ratio=0.50, red_points=[(0.50, 0.40)])
    f_clear = make_synthetic_minimap_frame(health_ratio=0.55, red_points=[])
    frames = [f_high, f_drop, f_drop, f_drop, f_clear, f_clear, f_clear, f_clear]

    win_backend = ReplayWindowBackend(lambda idx: frames[min(idx, len(frames) - 1)])
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.cover_peek_combat(
        enter_cover=False,
        use_special=False,
        min_peek_health=0.45,
        rapid_drop_duck_threshold=0.20,
        heal_wait_s=0.25,
        exit_on_health_drop=False,
        stop_when_no_enemies=True,
        no_enemies_clear_s=0.25,
        config=LoopConfig(max_duration_s=5.0, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=0.20),
    )
    assert res.stop_reason == "enemies_cleared"
    assert ("key_down", "q") not in backend.events
    # Duck released right-click on drop, then resumed peeking (2nd mouse_down right) before clearing
    assert backend.events.count(("mouse_down", "right")) >= 2
    assert backend.currently_held_buttons == set()


def test_follow_gps_drive_stuck_recovery_and_moving_no_false_trigger(tmp_path) -> None:
    # 1) Stuck scenario: identical frames for > 1.5s triggers reverse ('s') + counter-steer
    clock = FakeClock()
    backend = RecordingInputBackend()
    ctrl = InputController(backend=backend, logger=TelemetryLogger(tmp_path / "t.jsonl"), clock=clock, sleep_fn=clock.sleep, register_signals=False)
    stuck_frame = make_synthetic_minimap_frame(yellow_points=[(0.55, 0.35)])
    win_backend = ReplayWindowBackend(lambda idx: stuck_frame)
    runner = ActionRunner(ctrl, win_backend, ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t.jsonl"), clock, clock.sleep)

    res = runner.follow_gps_drive(
        stuck_window_s=1.0,
        reverse_duration_s=0.4,
        config=LoopConfig(max_duration_s=2.5, hz=10.0, hud_check_interval_s=0.0),
    )
    assert res.stop_reason == "timeout"
    assert ("key_down", "s") in backend.events
    assert ("key_up", "s") in backend.events
    assert backend.currently_held_keys == set()

    # 2) Moving scenario: high visual change in road band does NOT trigger reverse ('s')
    clock2 = FakeClock()
    backend2 = RecordingInputBackend()
    ctrl2 = InputController(backend=backend2, logger=TelemetryLogger(tmp_path / "t2.jsonl"), clock=clock2, sleep_fn=clock2.sleep, register_signals=False)

    def _moving_frame(idx: int) -> Image.Image:
        img = make_synthetic_minimap_frame(yellow_points=[(0.50, 0.35)])
        draw = ImageDraw.Draw(img)
        shade = 220 if (idx % 2 == 0) else 30
        draw.rectangle((300, 150, 900, 450), fill=(shade, shade, shade))
        return img

    runner2 = ActionRunner(ctrl2, ReplayWindowBackend(_moving_frame), ScriptedHudOCR(lambda idx: {}), TelemetryLogger(tmp_path / "t2.jsonl"), clock2, clock2.sleep)
    runner2.follow_gps_drive(
        stuck_window_s=0.6,
        reverse_duration_s=0.4,
        config=LoopConfig(max_duration_s=1.5, hz=10.0, hud_check_interval_s=0.0),
    )
    assert ("key_down", "s") not in backend2.events
    assert backend2.currently_held_keys == set()
