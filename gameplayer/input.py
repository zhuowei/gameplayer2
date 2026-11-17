"""Fail-safe synthetic OS input module using macOS Quartz CGEvents.

Safety contract:
- Every held key and mouse button is tracked in-memory and automatically released
  on normal exit, exception, context exit, interpreter exit (atexit), OS signal
  (SIGINT/SIGTERM), or hard timeout watchdog expiration.
- CLI and live backends require explicit opt-in (`--live` or instantiating
  `CGEventInputBackend`) so offline tests never touch the live game window.
"""

from __future__ import annotations

import argparse
import atexit
import json
import signal
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, Sequence

from gameplayer.telemetry import TelemetryLogger, get_default_logger

# macOS virtual keycodes (Carbon / HIToolbox Events.h US ANSI layout)
KEYCODES: dict[str, int] = {
    "a": 0x00,
    "s": 0x01,
    "d": 0x02,
    "f": 0x03,
    "h": 0x04,
    "g": 0x05,
    "z": 0x06,
    "x": 0x07,
    "c": 0x08,
    "v": 0x09,
    "b": 0x0B,
    "q": 0x0C,
    "w": 0x0D,
    "e": 0x0E,
    "r": 0x0F,
    "y": 0x10,
    "t": 0x11,
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "6": 0x16,
    "5": 0x17,
    "equal": 0x18,
    "9": 0x19,
    "7": 0x1A,
    "minus": 0x1B,
    "8": 0x1C,
    "0": 0x1D,
    "right_bracket": 0x1E,
    "o": 0x1F,
    "u": 0x20,
    "left_bracket": 0x21,
    "i": 0x22,
    "p": 0x23,
    "enter": 0x24,
    "return": 0x24,
    "l": 0x25,
    "j": 0x26,
    "quote": 0x27,
    "k": 0x28,
    "semicolon": 0x29,
    "backslash": 0x2A,
    "comma": 0x2B,
    "slash": 0x2C,
    "n": 0x2D,
    "m": 0x2E,
    "period": 0x2F,
    "tab": 0x30,
    "space": 0x31,
    "grave": 0x32,
    "backspace": 0x33,
    "escape": 0x35,
    "esc": 0x35,
    "lcmd": 0x37,
    "lshift": 0x38,
    "shift": 0x38,
    "capslock": 0x39,
    "lalt": 0x3A,
    "alt": 0x3A,
    "option": 0x3A,
    "lctrl": 0x3B,
    "ctrl": 0x3B,
    "rshift": 0x3C,
    "ralt": 0x3D,
    "rctrl": 0x3E,
    "f17": 0x40,
    "numpad_decimal": 0x41,
    "numpad_multiply": 0x43,
    "numpad_plus": 0x45,
    "numpad_clear": 0x47,
    "numpad_divide": 0x4B,
    "numpad_enter": 0x4C,
    "numpad_minus": 0x4E,
    "numpad_equal": 0x51,
    "numpad0": 0x52,
    "numpad1": 0x53,
    "numpad2": 0x54,
    "numpad3": 0x55,
    "numpad4": 0x56,
    "numpad5": 0x57,
    "numpad6": 0x58,
    "numpad7": 0x59,
    "numpad8": 0x5B,
    "numpad9": 0x5C,
    "f5": 0x60,
    "f6": 0x61,
    "f7": 0x62,
    "f3": 0x63,
    "f8": 0x64,
    "f9": 0x65,
    "f11": 0x67,
    "f10": 0x6D,
    "f12": 0x6F,
    "home": 0x73,
    "page_up": 0x74,
    "delete": 0x75,
    "f4": 0x76,
    "end": 0x77,
    "f2": 0x78,
    "page_down": 0x79,
    "f1": 0x7A,
    "left": 0x7B,
    "right": 0x7C,
    "down": 0x7D,
    "up": 0x7E,
}

CANONICAL_KEY_ALIASES: dict[str, str] = {
    "return": "enter",
    "esc": "escape",
    "shift": "lshift",
    "alt": "lalt",
    "option": "lalt",
    "ctrl": "lctrl",
}

VALID_MOUSE_BUTTONS = frozenset({"left", "right", "middle"})


class InputTimeoutError(TimeoutError):
    """Raised when an input action exceeds its hard timeout watchdog."""


class InputBackend(Protocol):
    """Abstract OS input backend so all safety/control logic can be tested offline."""

    def key_down(self, key: str) -> None:
        ...

    def key_up(self, key: str) -> None:
        ...

    def mouse_down(self, button: str) -> None:
        ...

    def mouse_up(self, button: str) -> None:
        ...

    def mouse_move_delta(
        self,
        dx: int,
        dy: int,
        held_buttons: frozenset[str] = frozenset(),
    ) -> None:
        ...


def normalize_key_name(key: str) -> str:
    norm = key.strip().lower()
    norm = CANONICAL_KEY_ALIASES.get(norm, norm)
    if norm not in KEYCODES:
        raise ValueError(f"Unknown key name: {key!r}")
    return norm


def normalize_button_name(button: str) -> str:
    norm = button.strip().lower()
    if norm not in VALID_MOUSE_BUTTONS:
        raise ValueError(f"Unknown mouse button: {button!r}. Expected one of {sorted(VALID_MOUSE_BUTTONS)}")
    return norm


class CGEventInputBackend:
    """macOS Quartz CGEvent HID input backend. ONLY `player` may use this on the live game."""

    def __init__(self, target_pid: Optional[int] = None) -> None:
        import Quartz  # noqa: F401

        self._quartz = Quartz
        self._target_pid = target_pid

    def _post(self, event: Any) -> None:
        q = self._quartz
        if self._target_pid is not None:
            q.CGEventPostToPid(int(self._target_pid), event)
        else:
            q.CGEventPost(q.kCGHIDEventTap, event)

    def _cursor_pos(self) -> Any:
        q = self._quartz
        probe = q.CGEventCreate(None)
        return q.CGEventGetLocation(probe)

    def key_down(self, key: str) -> None:
        norm = normalize_key_name(key)
        q = self._quartz
        ev = q.CGEventCreateKeyboardEvent(None, KEYCODES[norm], True)
        self._post(ev)

    def key_up(self, key: str) -> None:
        norm = normalize_key_name(key)
        q = self._quartz
        ev = q.CGEventCreateKeyboardEvent(None, KEYCODES[norm], False)
        self._post(ev)

    def _mouse_event_type_and_btn(self, button: str, down: bool) -> tuple[int, int]:
        q = self._quartz
        if button == "left":
            return (q.kCGEventLeftMouseDown if down else q.kCGEventLeftMouseUp, q.kCGMouseButtonLeft)
        if button == "right":
            return (q.kCGEventRightMouseDown if down else q.kCGEventRightMouseUp, q.kCGMouseButtonRight)
        return (q.kCGEventOtherMouseDown if down else q.kCGEventOtherMouseUp, q.kCGMouseButtonCenter)

    def mouse_down(self, button: str) -> None:
        norm = normalize_button_name(button)
        q = self._quartz
        ev_type, cg_btn = self._mouse_event_type_and_btn(norm, down=True)
        ev = q.CGEventCreateMouseEvent(None, ev_type, self._cursor_pos(), cg_btn)
        self._post(ev)

    def mouse_up(self, button: str) -> None:
        norm = normalize_button_name(button)
        q = self._quartz
        ev_type, cg_btn = self._mouse_event_type_and_btn(norm, down=False)
        ev = q.CGEventCreateMouseEvent(None, ev_type, self._cursor_pos(), cg_btn)
        self._post(ev)

    def mouse_move_delta(
        self,
        dx: int,
        dy: int,
        held_buttons: frozenset[str] = frozenset(),
    ) -> None:
        q = self._quartz
        pos = self._cursor_pos()
        new_pos = (pos.x + dx, pos.y + dy)
        if "left" in held_buttons:
            ev_type, cg_btn = q.kCGEventLeftMouseDragged, q.kCGMouseButtonLeft
        elif "right" in held_buttons:
            ev_type, cg_btn = q.kCGEventRightMouseDragged, q.kCGMouseButtonRight
        elif "middle" in held_buttons:
            ev_type, cg_btn = q.kCGEventOtherMouseDragged, q.kCGMouseButtonCenter
        else:
            ev_type, cg_btn = q.kCGEventMouseMoved, q.kCGMouseButtonLeft
        ev = q.CGEventCreateMouseEvent(None, ev_type, new_pos, cg_btn)
        q.CGEventSetIntegerValueField(ev, q.kCGMouseEventDeltaX, int(dx))
        q.CGEventSetIntegerValueField(ev, q.kCGMouseEventDeltaY, int(dy))
        self._post(ev)


_ACTIVE_CONTROLLERS: weakref.WeakSet["InputController"] = weakref.WeakSet()
_REGISTRY_LOCK = threading.Lock()
_SIGNAL_HANDLERS_INSTALLED = False
_PREV_SIGNAL_HANDLERS: dict[int, Any] = {}


def _release_all_active_controllers(reason: str) -> None:
    with _REGISTRY_LOCK:
        controllers = list(_ACTIVE_CONTROLLERS)
    for ctrl in controllers:
        ctrl.release_all(reason=reason, suppress_errors=True)


def _signal_handler(signum: int, frame: Any) -> None:
    _release_all_active_controllers(reason=f"signal_{signum}")
    prev = _PREV_SIGNAL_HANDLERS.get(signum)
    if callable(prev):
        prev(signum, frame)
    elif prev == signal.SIG_DFL:
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)


def install_signal_handlers() -> None:
    """Install SIGINT/SIGTERM cleanup handlers (main thread only, idempotent)."""
    global _SIGNAL_HANDLERS_INSTALLED
    if threading.current_thread() is not threading.main_thread():
        return
    with _REGISTRY_LOCK:
        if _SIGNAL_HANDLERS_INSTALLED:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                _PREV_SIGNAL_HANDLERS[sig] = signal.getsignal(sig)
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):
                pass
        _SIGNAL_HANDLERS_INSTALLED = True


atexit.register(_release_all_active_controllers, "atexit")


class InputController:
    """Fail-safe input controller guaranteeing release of all held keys/buttons."""

    def __init__(
        self,
        backend: InputBackend,
        logger: Optional[TelemetryLogger] = None,
        default_timeout_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        register_signals: bool = True,
    ) -> None:
        self.backend = backend
        self.logger = logger or get_default_logger()
        self.default_timeout_s = float(default_timeout_s)
        self._clock = clock
        self._sleep = sleep_fn
        self._lock = threading.RLock()
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        self._closed = False

        with _REGISTRY_LOCK:
            _ACTIVE_CONTROLLERS.add(self)
        if register_signals:
            install_signal_handlers()

    @property
    def held_keys(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._held_keys)

    @property
    def held_buttons(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._held_buttons)

    def __enter__(self) -> "InputController":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        reason = f"exception:{exc_type.__name__}" if exc_type is not None else "context_exit"
        self.release_all(reason=reason, suppress_errors=True)

    def close(self) -> dict[str, list[str]]:
        self._closed = True
        return self.release_all(reason="close", suppress_errors=True)

    def key_down(self, key: str) -> None:
        norm = normalize_key_name(key)
        with self._lock:
            if self._closed:
                raise RuntimeError("InputController is closed")
            self.backend.key_down(norm)
            self._held_keys.add(norm)

    def key_up(self, key: str) -> None:
        norm = normalize_key_name(key)
        with self._lock:
            try:
                self.backend.key_up(norm)
            finally:
                self._held_keys.discard(norm)

    def mouse_down(self, button: str) -> None:
        norm = normalize_button_name(button)
        with self._lock:
            if self._closed:
                raise RuntimeError("InputController is closed")
            self.backend.mouse_down(norm)
            self._held_buttons.add(norm)

    def mouse_up(self, button: str) -> None:
        norm = normalize_button_name(button)
        with self._lock:
            try:
                self.backend.mouse_up(norm)
            finally:
                self._held_buttons.discard(norm)

    def release_all(
        self,
        reason: str = "explicit",
        suppress_errors: bool = False,
    ) -> dict[str, list[str]]:
        """Release EVERY held key and mouse button, continuing even if an individual release errors."""
        with self._lock:
            keys_to_release = sorted(self._held_keys)
            buttons_to_release = sorted(self._held_buttons)
            if not keys_to_release and not buttons_to_release:
                return {"keys": [], "buttons": []}

            errors: list[Exception] = []
            for key in keys_to_release:
                try:
                    self.backend.key_up(key)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    self._held_keys.discard(key)

            for btn in buttons_to_release:
                try:
                    self.backend.mouse_up(btn)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    self._held_buttons.discard(btn)

            self.logger.log(
                tool="input",
                action="release_all",
                status="error" if errors else "ok",
                details={
                    "reason": reason,
                    "keys": keys_to_release,
                    "buttons": buttons_to_release,
                },
                error="; ".join(f"{type(e).__name__}: {e}" for e in errors) if errors else None,
            )

            if errors and not suppress_errors:
                raise errors[0]
            return {"keys": keys_to_release, "buttons": buttons_to_release}

    @contextmanager
    def _watchdog(self, timeout_s: Optional[float], action_name: str) -> Iterator[threading.Event]:
        effective_timeout = self.default_timeout_s if timeout_s is None else float(timeout_s)
        if effective_timeout <= 0:
            raise ValueError(f"timeout_s must be positive, got {effective_timeout}")
        timed_out = threading.Event()

        def _on_timeout() -> None:
            timed_out.set()
            self.release_all(reason=f"timeout_watchdog:{action_name}", suppress_errors=True)

        timer = threading.Timer(effective_timeout, _on_timeout)
        timer.daemon = True
        timer.start()
        try:
            yield timed_out
        finally:
            timer.cancel()

    def _sleep_checked(
        self,
        duration_s: float,
        abort_event: threading.Event,
        action_name: str,
        timeout_s: float,
    ) -> None:
        if duration_s < 0:
            raise ValueError(f"duration_s must be >= 0, got {duration_s}")
        if duration_s == 0:
            if abort_event.is_set():
                raise InputTimeoutError(f"Action {action_name!r} exceeded hard timeout {timeout_s:.3f}s")
            return
        deadline = self._clock() + duration_s
        while True:
            if abort_event.is_set():
                raise InputTimeoutError(f"Action {action_name!r} exceeded hard timeout {timeout_s:.3f}s")
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            self._sleep(min(0.02, remaining))
            if abort_event.is_set():
                raise InputTimeoutError(f"Action {action_name!r} exceeded hard timeout {timeout_s:.3f}s")

    def hold(
        self,
        keys: Iterable[str] = (),
        buttons: Iterable[str] = (),
        duration_s: float = 0.1,
        timeout_s: Optional[float] = None,
    ) -> None:
        """Press and hold keys/buttons for `duration_s`, releasing all on exit/error/timeout."""
        norm_keys = [normalize_key_name(k) for k in keys]
        norm_buttons = [normalize_button_name(b) for b in buttons]
        eff_timeout = self.default_timeout_s if timeout_s is None else float(timeout_s)

        with self.logger.timed(
            "input",
            "hold",
            {"keys": norm_keys, "buttons": norm_buttons, "duration_s": duration_s},
        ):
            with self._watchdog(eff_timeout, "hold") as abort_event:
                try:
                    for k in norm_keys:
                        self.key_down(k)
                    for b in norm_buttons:
                        self.mouse_down(b)
                    self._sleep_checked(duration_s, abort_event, "hold", eff_timeout)
                finally:
                    self.release_all(reason="hold_end", suppress_errors=True)

    def tap(self, key: str, duration_s: float = 0.06, timeout_s: Optional[float] = None) -> None:
        """Press and release a single key."""
        self.hold(keys=[key], duration_s=duration_s, timeout_s=timeout_s)

    def click(self, button: str = "left", duration_s: float = 0.06, timeout_s: Optional[float] = None) -> None:
        """Press and release a single mouse button."""
        self.hold(buttons=[button], duration_s=duration_s, timeout_s=timeout_s)

    def _emit_mouse_delta_steps(
        self,
        dx: int,
        dy: int,
        steps: int,
        duration_s: float,
        abort_event: threading.Event,
        action_name: str,
        timeout_s: float,
    ) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        rem_x = 0.0
        rem_y = 0.0
        step_sleep = duration_s / steps if duration_s > 0 else 0.0
        for _ in range(steps):
            if abort_event.is_set():
                raise InputTimeoutError(f"Action {action_name!r} exceeded hard timeout {timeout_s:.3f}s")
            rem_x += dx / steps
            rem_y += dy / steps
            ix = int(round(rem_x))
            iy = int(round(rem_y))
            rem_x -= ix
            rem_y -= iy
            self.backend.mouse_move_delta(ix, iy, held_buttons=self.held_buttons)
            if step_sleep > 0:
                self._sleep_checked(step_sleep, abort_event, action_name, timeout_s)

    def mouse_delta(
        self,
        dx: int,
        dy: int,
        steps: int = 1,
        duration_s: float = 0.0,
        timeout_s: Optional[float] = None,
    ) -> None:
        """Move the mouse by relative delta (dx, dy) across `steps` increments."""
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        eff_timeout = self.default_timeout_s if timeout_s is None else float(timeout_s)
        with self.logger.timed(
            "input",
            "mouse_delta",
            {"dx": int(dx), "dy": int(dy), "steps": steps, "duration_s": duration_s},
        ):
            with self._watchdog(eff_timeout, "mouse_delta") as abort_event:
                self._emit_mouse_delta_steps(
                    dx=dx,
                    dy=dy,
                    steps=steps,
                    duration_s=duration_s,
                    abort_event=abort_event,
                    action_name="mouse_delta",
                    timeout_s=eff_timeout,
                )

    def run_sequence(
        self,
        steps: Sequence[dict[str, Any]],
        timeout_s: Optional[float] = None,
    ) -> int:
        """Execute a multi-step input sequence with guaranteed release_all on exit/error/timeout."""
        eff_timeout = self.default_timeout_s if timeout_s is None else float(timeout_s)
        executed = 0
        with self.logger.timed("input", "run_sequence", {"step_count": len(steps)}) as details:
            with self._watchdog(eff_timeout, "run_sequence") as abort_event:
                try:
                    for step in steps:
                        if abort_event.is_set():
                            raise InputTimeoutError(
                                f"Action 'run_sequence' exceeded hard timeout {eff_timeout:.3f}s"
                            )
                        action = str(step.get("action", "")).lower()
                        if action == "hold":
                            for k in step.get("keys", ()):
                                self.key_down(k)
                            for b in step.get("buttons", ()):
                                self.mouse_down(b)
                            self._sleep_checked(
                                float(step.get("duration_s", 0.1)),
                                abort_event,
                                "run_sequence:hold",
                                eff_timeout,
                            )
                            for k in step.get("keys", ()):
                                self.key_up(k)
                            for b in step.get("buttons", ()):
                                self.mouse_up(b)
                        elif action == "tap":
                            k = str(step["key"])
                            self.key_down(k)
                            self._sleep_checked(
                                float(step.get("duration_s", 0.06)),
                                abort_event,
                                "run_sequence:tap",
                                eff_timeout,
                            )
                            self.key_up(k)
                        elif action == "click":
                            b = str(step.get("button", "left"))
                            self.mouse_down(b)
                            self._sleep_checked(
                                float(step.get("duration_s", 0.06)),
                                abort_event,
                                "run_sequence:click",
                                eff_timeout,
                            )
                            self.mouse_up(b)
                        elif action == "key_down":
                            self.key_down(str(step["key"]))
                        elif action == "key_up":
                            self.key_up(str(step["key"]))
                        elif action == "mouse_down":
                            self.mouse_down(str(step["button"]))
                        elif action == "mouse_up":
                            self.mouse_up(str(step["button"]))
                        elif action == "mouse_delta":
                            self._emit_mouse_delta_steps(
                                dx=int(step.get("dx", 0)),
                                dy=int(step.get("dy", 0)),
                                steps=int(step.get("steps", 1)),
                                duration_s=float(step.get("duration_s", 0.0)),
                                abort_event=abort_event,
                                action_name="run_sequence:mouse_delta",
                                timeout_s=eff_timeout,
                            )
                        elif action == "wait":
                            self._sleep_checked(
                                float(step.get("duration_s", 0.1)),
                                abort_event,
                                "run_sequence:wait",
                                eff_timeout,
                            )
                        else:
                            raise ValueError(f"Unsupported sequence action: {action!r}")
                        executed += 1
                    details["executed_steps"] = executed
                finally:
                    self.release_all(reason="run_sequence_end", suppress_errors=True)
        return executed


class DryRunInputBackend:
    """In-memory logging backend for safe CLI dry-runs."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def key_down(self, key: str) -> None:
        self.events.append(("key_down", key))

    def key_up(self, key: str) -> None:
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-safe synthetic OS input CLI")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Execute against DryRunInputBackend (safe offline)")
    mode.add_argument("--live", action="store_true", help="Send live CGEvents (ONLY `player` may use this)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Hard watchdog timeout in seconds")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tap = sub.add_parser("tap", help="Tap a single key")
    p_tap.add_argument("key")
    p_tap.add_argument("--duration", type=float, default=0.06)

    p_hold = sub.add_parser("hold", help="Hold keys/buttons for duration")
    p_hold.add_argument("--keys", nargs="*", default=[])
    p_hold.add_argument("--buttons", nargs="*", default=[])
    p_hold.add_argument("--duration", type=float, default=0.25)

    p_mouse = sub.add_parser("mouse-delta", help="Relative mouse move/look")
    p_mouse.add_argument("dx", type=int)
    p_mouse.add_argument("dy", type=int)
    p_mouse.add_argument("--steps", type=int, default=1)
    p_mouse.add_argument("--duration", type=float, default=0.0)

    p_seq = sub.add_parser("sequence", help="Run a JSON sequence file")
    p_seq.add_argument("json_file", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    backend: InputBackend = CGEventInputBackend() if args.live else DryRunInputBackend()

    with InputController(backend=backend, default_timeout_s=args.timeout) as ctrl:
        if args.cmd == "tap":
            ctrl.tap(args.key, duration_s=args.duration, timeout_s=args.timeout)
        elif args.cmd == "hold":
            ctrl.hold(keys=args.keys, buttons=args.buttons, duration_s=args.duration, timeout_s=args.timeout)
        elif args.cmd == "mouse-delta":
            ctrl.mouse_delta(args.dx, args.dy, steps=args.steps, duration_s=args.duration, timeout_s=args.timeout)
        elif args.cmd == "sequence":
            steps = json.loads(args.json_file.read_text(encoding="utf-8"))
            ctrl.run_sequence(steps, timeout_s=args.timeout)

    if isinstance(backend, DryRunInputBackend):
        print(json.dumps({"dry_run_events": backend.events}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
