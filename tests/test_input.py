from __future__ import annotations

import signal
from typing import Optional

import pytest

import gameplayer.input as gp_input
from gameplayer.input import (
    DryRunInputBackend,
    InputController,
    InputTimeoutError,
    _release_all_active_controllers,
    main as input_cli_main,
)
from gameplayer.telemetry import TelemetryLogger, read_telemetry


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.now += dt


class FaultyReleaseBackend:
    def __init__(self, fail_key_up: Optional[str] = None) -> None:
        self.events: list[tuple[str, object]] = []
        self.fail_key_up = fail_key_up

    def key_down(self, key: str) -> None:
        self.events.append(("key_down", key))

    def key_up(self, key: str) -> None:
        if key == self.fail_key_up:
            raise RuntimeError(f"failed releasing {key}")
        self.events.append(("key_up", key))

    def mouse_down(self, button: str) -> None:
        self.events.append(("mouse_down", button))

    def mouse_up(self, button: str) -> None:
        self.events.append(("mouse_up", button))

    def mouse_move_delta(
        self,
        dx: int,
        dy: int,
        held_buttons: frozenset[str] = frozenset(),
    ) -> None:
        self.events.append(("mouse_delta", (dx, dy, sorted(held_buttons))))


def test_tap_hold_click_and_mouse_delta_with_fake_clock(tmp_path) -> None:
    clock = FakeClock()
    backend = FaultyReleaseBackend()
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")

    with InputController(
        backend=backend,
        logger=logger,
        clock=clock,
        sleep_fn=clock.sleep,
        register_signals=False,
    ) as ctrl:
        ctrl.tap("w", duration_s=0.05)
        ctrl.click("left", duration_s=0.05)
        ctrl.hold(keys=["w", "shift"], buttons=["right"], duration_s=0.2)
        ctrl.mouse_delta(100, -50, steps=4, duration_s=0.08)

        assert ctrl.held_keys == frozenset()
        assert ctrl.held_buttons == frozenset()

    assert ("key_down", "w") in backend.events
    assert ("key_up", "w") in backend.events
    assert ("key_down", "lshift") in backend.events
    assert ("key_up", "lshift") in backend.events
    assert ("mouse_down", "right") in backend.events
    assert ("mouse_up", "right") in backend.events

    deltas = [ev[1] for ev in backend.events if ev[0] == "mouse_delta"]
    assert len(deltas) == 4
    assert sum(d[0] for d in deltas) == 100
    assert sum(d[1] for d in deltas) == -50


def test_context_exit_releases_all_on_exception(tmp_path) -> None:
    backend = FaultyReleaseBackend()
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")

    with pytest.raises(ValueError):
        with InputController(backend=backend, logger=logger, register_signals=False) as ctrl:
            ctrl.key_down("w")
            ctrl.key_down("lshift")
            ctrl.mouse_down("right")
            assert ctrl.held_keys == frozenset({"w", "lshift"})
            assert ctrl.held_buttons == frozenset({"right"})
            raise ValueError("simulated crash")

    assert ctrl.held_keys == frozenset()
    assert ctrl.held_buttons == frozenset()
    assert ("key_up", "w") in backend.events
    assert ("key_up", "lshift") in backend.events
    assert ("mouse_up", "right") in backend.events


def test_run_sequence_releases_all_on_invalid_step_and_tracks_held_drag(tmp_path) -> None:
    clock = FakeClock()
    backend = FaultyReleaseBackend()
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")
    ctrl = InputController(
        backend=backend,
        logger=logger,
        clock=clock,
        sleep_fn=clock.sleep,
        register_signals=False,
    )

    executed = ctrl.run_sequence(
        [
            {"action": "mouse_down", "button": "right"},
            {"action": "mouse_delta", "dx": 20, "dy": 10, "steps": 2, "duration_s": 0.02},
            {"action": "mouse_up", "button": "right"},
        ]
    )
    assert executed == 3
    deltas = [ev[1] for ev in backend.events if ev[0] == "mouse_delta"]
    assert all(d[2] == ["right"] for d in deltas)

    with pytest.raises(ValueError):
        ctrl.run_sequence(
            [
                {"action": "key_down", "key": "w"},
                {"action": "mouse_down", "button": "left"},
                {"action": "bogus_action"},
            ]
        )

    assert ctrl.held_keys == frozenset()
    assert ctrl.held_buttons == frozenset()
    assert ("key_up", "w") in backend.events
    assert ("mouse_up", "left") in backend.events


def test_release_all_continues_when_one_key_release_fails(tmp_path) -> None:
    backend = FaultyReleaseBackend(fail_key_up="w")
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")
    ctrl = InputController(backend=backend, logger=logger, register_signals=False)
    ctrl.key_down("w")
    ctrl.key_down("d")
    ctrl.mouse_down("left")

    with pytest.raises(RuntimeError):
        ctrl.release_all(reason="test_partial_fail", suppress_errors=False)

    assert ctrl.held_keys == frozenset()
    assert ctrl.held_buttons == frozenset()
    assert ("key_up", "d") in backend.events
    assert ("mouse_up", "left") in backend.events


def test_hard_timeout_watchdog_releases_all_held_inputs(tmp_path) -> None:
    backend = FaultyReleaseBackend()
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")
    ctrl = InputController(backend=backend, logger=logger, register_signals=False)

    with pytest.raises(InputTimeoutError):
        ctrl.hold(keys=["w", "space"], buttons=["left"], duration_s=2.0, timeout_s=0.05)

    assert ctrl.held_keys == frozenset()
    assert ctrl.held_buttons == frozenset()
    assert ("key_up", "w") in backend.events
    assert ("key_up", "space") in backend.events
    assert ("mouse_up", "left") in backend.events

    records = read_telemetry(tmp_path / "telemetry.jsonl", tool="input")
    assert any(
        r["action"] == "release_all" and "timeout_watchdog" in str(r.get("details", {}).get("reason", ""))
        for r in records
    )


def test_atexit_and_signal_cleanup_release_active_controllers(tmp_path, monkeypatch) -> None:
    backend = FaultyReleaseBackend()
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")
    ctrl = InputController(backend=backend, logger=logger, register_signals=False)
    ctrl.key_down("a")
    ctrl.mouse_down("right")

    _release_all_active_controllers("atexit")
    assert ctrl.held_keys == frozenset()
    assert ctrl.held_buttons == frozenset()

    ctrl.key_down("s")
    prev_called: list[int] = []
    monkeypatch.setitem(gp_input._PREV_SIGNAL_HANDLERS, signal.SIGTERM, lambda sig, frm: prev_called.append(sig))
    gp_input._signal_handler(signal.SIGTERM, None)
    assert prev_called == [signal.SIGTERM]
    assert ctrl.held_keys == frozenset()
    assert ("key_up", "s") in backend.events


def test_cli_dry_run_mode(tmp_path, capsys) -> None:
    seq_file = tmp_path / "seq.json"
    seq_file.write_text('[{"action": "tap", "key": "e", "duration_s": 0.01}]', encoding="utf-8")
    rc = input_cli_main(["--dry-run", "sequence", str(seq_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "key_down" in out and "key_up" in out
    assert isinstance(DryRunInputBackend(), DryRunInputBackend)


def test_required_macos_virtual_keycodes_match_hitoolbox() -> None:
    expected = {
        "f5": 0x60,
        "f6": 0x61,
        "f7": 0x62,
        "f8": 0x64,
        "tab": 0x30,
        "q": 0x0C,
        "r": 0x0F,
        "f": 0x03,
        "e": 0x0E,
        "g": 0x05,
        "c": 0x08,
        "v": 0x09,
        "lalt": 0x3A,
        "lctrl": 0x3B,
        "lshift": 0x38,
        "capslock": 0x39,
        "space": 0x31,
        "enter": 0x24,
        "escape": 0x35,
        "left": 0x7B,
        "right": 0x7C,
        "down": 0x7D,
        "up": 0x7E,
    }
    for name, code in expected.items():
        assert gp_input.KEYCODES[name] == code


def test_switch_character_hotkey_wheel_and_cli(tmp_path, capsys) -> None:
    clock = FakeClock()
    backend = FaultyReleaseBackend()
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")
    with InputController(
        backend=backend,
        logger=logger,
        clock=clock,
        sleep_fn=clock.sleep,
        register_signals=False,
    ) as ctrl:
        ctrl.switch_character("trevor", method="hotkey")
        ctrl.switch_character("michael", method="hotkey")
        ctrl.switch_character("franklin", method="hotkey")
        assert ("key_down", "f7") in backend.events
        assert ("key_up", "f7") in backend.events
        assert ("key_down", "f5") in backend.events
        assert ("key_down", "f6") in backend.events

        backend.events.clear()
        ctrl.switch_character("trevor", method="wheel")
        assert backend.events[0] == ("key_down", "lalt")
        assert backend.events[-1] == ("key_up", "lalt")
        deltas = [ev[1][0] for ev in backend.events if ev[0] == "mouse_delta"]
        assert sum(deltas) == 60
        assert ctrl.held_keys == frozenset()

        backend.events.clear()
        ctrl.switch_character("trevor", method="wheel_hold")
        assert backend.events == [("key_down", "lalt"), ("key_up", "lalt")]
        assert ctrl.held_keys == frozenset()

    rc = input_cli_main(["--dry-run", "switch-character", "trevor", "--method", "wheel"])
    assert rc == 0
    assert "lalt" in capsys.readouterr().out


def test_cgevent_backend_warps_cursor_back_to_center_when_offscreen(monkeypatch) -> None:
    warped: list[tuple[float, float]] = []

    class _Pt:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    class _Rect:
        def __init__(self) -> None:
            self.origin = _Pt(0.0, 0.0)
            self.size = type("Size", (), {"width": 1920.0, "height": 1080.0})()

    class _FakeQuartz:
        kCGEventSourceStateHIDSystemState = 0
        kCGEventSuppressionStateSuppressionInterval = 0
        kCGEventSuppressionStateRemoteMouseDrag = 1
        kCGEventFilterMaskPermitAllEvents = 0x7
        kCGEventMouseMoved = 5
        kCGEventRightMouseDragged = 7
        kCGMouseButtonLeft = 0
        kCGMouseButtonRight = 1
        kCGMouseEventDeltaX = 1
        kCGMouseEventDeltaY = 2
        kCGHIDEventTap = 0
        kCGEventFlagMaskShift = 1
        kCGEventFlagMaskControl = 2
        kCGEventFlagMaskAlternate = 4
        kCGEventFlagMaskCommand = 8
        kCGEventFlagMaskNumericPad = 16
        kCGEventFlagMaskSecondaryFn = 32

        def __init__(self) -> None:
            self.suppression_interval = None
            self.filters: list[tuple[int, int]] = []

        def CGEventSourceCreate(self, state):
            return object()

        def CGEventSourceSetLocalEventsSuppressionInterval(self, src, interval):
            self.suppression_interval = interval

        def CGEventSourceSetLocalEventsFilterDuringSuppressionState(self, src, mask, state):
            self.filters.append((mask, state))

        def CGEventCreate(self, src):
            return object()

        def CGEventGetLocation(self, ev):
            return _Pt(22032.0, 540.0)

        def CGDisplayBounds(self, did):
            return _Rect()

        def CGMainDisplayID(self):
            return 0

        def CGWarpMouseCursorPosition(self, pos):
            warped.append(pos)

        def CGEventCreateMouseEvent(self, src, ev_type, pos, btn):
            return {"type": ev_type, "pos": pos, "btn": btn}

        def CGEventCreateKeyboardEvent(self, src, keycode, is_down):
            return {"kind": "key", "keycode": keycode, "down": is_down}

        def CGEventSetIntegerValueField(self, ev, field, val):
            ev[field] = val

        def CGEventSetFlags(self, ev, flags):
            ev["flags"] = flags

        def CGEventPost(self, tap, ev):
            self.last_posted = ev

    fake_q = _FakeQuartz()
    import sys
    monkeypatch.setitem(sys.modules, "Quartz", fake_q)
    backend = gp_input.CGEventInputBackend()
    backend.mouse_move_delta(-67, 0, held_buttons=frozenset({"right"}))
    assert fake_q.suppression_interval == 0.0
    assert warped == [(960.0, 540.0)]
    assert fake_q.last_posted["pos"] == (893.0, 540.0)
    assert fake_q.last_posted[fake_q.kCGMouseEventDeltaX] == -67

    # Window-anchored bounds warp to the window center instead of the display center
    warped.clear()
    backend.set_anchor_bounds(100.0, 200.0, 1280.0, 720.0)
    backend.mouse_move_delta(15, -5, held_buttons=frozenset())
    assert warped == [(740.0, 560.0)]
    assert fake_q.last_posted["pos"] == (755.0, 555.0)


def test_input_controller_click_window_center_anchors_warps_and_releases(tmp_path) -> None:
    class _AnchorRecordingBackend(FaultyReleaseBackend):
        def __init__(self) -> None:
            super().__init__()
            self.anchor_bounds: list[tuple[float, float, float, float]] = []
            self.warp_calls = 0

        def set_anchor_bounds(self, x: float, y: float, width: float, height: float) -> None:
            self.anchor_bounds.append((x, y, width, height))

        def warp_to_center(self) -> tuple[float, float]:
            self.warp_calls += 1
            x, y, w, h = self.anchor_bounds[-1]
            return (x + w * 0.5, y + h * 0.5)

    clock = FakeClock()
    backend = _AnchorRecordingBackend()
    ctrl = InputController(
        backend=backend,
        logger=TelemetryLogger(tmp_path / "t.jsonl"),
        clock=clock,
        sleep_fn=clock.sleep,
        register_signals=False,
    )
    ctrl.click_window_center(window_bounds=(320.0, 180.0, 1280.0, 720.0), button="right", hold_s=0.04)
    assert backend.anchor_bounds == [(320.0, 180.0, 1280.0, 720.0)]
    assert backend.warp_calls == 1
    assert ("mouse_down", "right") in backend.events
    assert ("mouse_up", "right") in backend.events
    assert ctrl.held_buttons == frozenset()


def test_cgevent_backend_preserves_arrow_and_fn_key_flags(monkeypatch) -> None:
    class _FlagFakeQuartz:
        kCGEventSourceStateHIDSystemState = 1
        kCGHIDEventTap = 0
        kCGEventFlagMaskShift = 1 << 17
        kCGEventFlagMaskControl = 1 << 18
        kCGEventFlagMaskAlternate = 1 << 19
        kCGEventFlagMaskCommand = 1 << 20
        kCGEventFlagMaskNumericPad = 1 << 21
        kCGEventFlagMaskSecondaryFn = 1 << 23

        def __init__(self) -> None:
            self.posted: list[dict[str, object]] = []

        def CGEventSourceCreate(self, state):
            return object()

        def CGEventSourceSetLocalEventsSuppressionInterval(self, src, interval):
            pass

        def CGEventCreateKeyboardEvent(self, src, keycode, is_down):
            return {"keycode": keycode, "down": is_down, "flags": 0}

        def CGEventSetFlags(self, ev, flags):
            ev["flags"] = flags

        def CGEventPost(self, tap, ev):
            self.posted.append(ev)

    fake_q = _FlagFakeQuartz()
    import sys
    monkeypatch.setitem(sys.modules, "Quartz", fake_q)
    backend = gp_input.CGEventInputBackend()
    backend.key_down("up")
    backend.key_up("up")
    backend.key_down("f5")
    backend.key_down("w")

    expected_arrow = fake_q.kCGEventFlagMaskNumericPad | fake_q.kCGEventFlagMaskSecondaryFn
    assert fake_q.posted[0]["keycode"] == gp_input.KEYCODES["up"]
    assert fake_q.posted[0]["flags"] == expected_arrow
    assert fake_q.posted[1]["flags"] == expected_arrow
    assert fake_q.posted[2]["flags"] == fake_q.kCGEventFlagMaskSecondaryFn
    assert fake_q.posted[3]["flags"] == 0
