from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from gameplayer.telemetry import TelemetryLogger, read_telemetry
from gameplayer.window import (
    QuartzWindowBackend,
    WindowBounds,
    WindowCaptureError,
    WindowInfo,
    WindowNotFoundError,
    capture_game_window,
    find_game_window,
    focus_game_window,
    is_valid_16_9,
    normalize_to_16_9,
)


class MockWindowBackend:
    def __init__(self, windows: list[WindowInfo], images: dict[int, Image.Image], focus_succeeds: bool = True) -> None:
        self._windows = windows
        self._images = images
        self._focus_succeeds = focus_succeeds
        self.focused_windows: list[int] = []

    def list_windows(self) -> list[WindowInfo]:
        return list(self._windows)

    def capture_window_image(self, window_id: int) -> Image.Image:
        return self._images[window_id]

    def focus_window(self, window: WindowInfo) -> bool:
        self.focused_windows.append(window.window_id)
        if not self._focus_succeeds:
            return False
        self._windows = [
            WindowInfo(
                window_id=w.window_id,
                owner=w.owner,
                title=w.title,
                pid=w.pid,
                bounds=w.bounds,
                layer=w.layer,
                is_onscreen=w.is_onscreen,
                is_frontmost=(w.window_id == window.window_id),
                obstructed_by=() if w.window_id == window.window_id else w.obstructed_by,
            )
            for w in self._windows
        ]
        return True


def _make_win(
    wid: int,
    owner: str,
    title: str,
    width: int = 1280,
    height: int = 720,
    layer: int = 0,
    onscreen: bool = True,
    frontmost: bool = False,
) -> WindowInfo:
    return WindowInfo(
        window_id=wid,
        owner=owner,
        title=title,
        pid=1000 + wid,
        bounds=WindowBounds(0.0, 0.0, float(width), float(height)),
        layer=layer,
        is_onscreen=onscreen,
        is_frontmost=frontmost,
    )


def test_find_game_window_scoring_and_filtering() -> None:
    windows = [
        _make_win(1, "Antigravity", "Editor", 1280, 720),
        _make_win(2, "GTA5.exe", "Grand Theft Auto V", 100, 100),  # too small
        _make_win(3, "GTA5.exe", "Grand Theft Auto V", 1280, 720, layer=1),  # non-zero layer
        _make_win(4, "wine64", "Grand Theft Auto Helper", 640, 360),
        _make_win(5, "GTA5.exe", "Grand Theft Auto V", 1280, 720, frontmost=True),
    ]
    backend = MockWindowBackend(windows, {})
    found = find_game_window(backend=backend)
    assert found.window_id == 5


def test_find_game_window_raises_when_missing() -> None:
    backend = MockWindowBackend([_make_win(1, "Finder", "Desktop", 1920, 1080)], {})
    with pytest.raises(WindowNotFoundError):
        find_game_window(backend=backend)


def test_normalize_to_16_9_crops_titlebar_from_top() -> None:
    # Top 28px red (titlebar), bottom 720px blue (game content)
    img = Image.new("RGB", (1280, 748), (0, 0, 255))
    titlebar = Image.new("RGB", (1280, 28), (255, 0, 0))
    img.paste(titlebar, (0, 0))

    norm, cropped = normalize_to_16_9(img, crop_titlebar=True)
    assert cropped is True
    assert norm.size == (1280, 720)
    assert norm.getpixel((10, 0)) == (0, 0, 255)


def test_normalize_to_16_9_crops_ultrawide_horizontally() -> None:
    img = Image.new("RGB", (1920, 720), (10, 200, 30))
    norm, cropped = normalize_to_16_9(img)
    assert cropped is True
    assert norm.size == (1280, 720)
    assert is_valid_16_9(*norm.size)


def test_capture_game_window_downscale_numpy_and_telemetry(tmp_path) -> None:
    win = _make_win(156, "GTA5.exe", "Grand Theft Auto V", 1280, 720, frontmost=False)
    img = Image.new("RGB", (1280, 720), (12, 34, 56))
    backend = MockWindowBackend([win], {156: img})
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")

    res = capture_game_window(backend=backend, logger=logger, target_width=640)
    assert res.size == (640, 360)
    assert res.raw_size == (1280, 720)
    assert res.was_cropped is False
    arr = res.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (360, 640, 3)
    assert arr.dtype == np.uint8
    assert tuple(arr[0, 0]) == (12, 34, 56)

    out_png = res.save(tmp_path / "captures" / "frame.png")
    assert out_png.exists()

    records = read_telemetry(tmp_path / "telemetry.jsonl", tool="window")
    assert len(records) == 1
    assert records[0]["action"] == "capture"
    assert records[0]["details"]["size"] == [640, 360]


def test_focus_game_window_requires_explicit_opt_in_and_verifies(tmp_path) -> None:
    win = _make_win(156, "GTA5.exe", "Grand Theft Auto V", frontmost=False)
    backend = MockWindowBackend([win], {})
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")

    with pytest.raises(PermissionError):
        focus_game_window(backend=backend, logger=logger)
    assert backend.focused_windows == []

    focused = focus_game_window(backend=backend, logger=logger, allow_focus=True)
    assert focused.window_id == 156
    assert focused.is_frontmost is True
    assert backend.focused_windows == [156]

    failing_backend = MockWindowBackend([win], {}, focus_succeeds=False)
    with pytest.raises(WindowCaptureError):
        focus_game_window(backend=failing_backend, logger=logger, allow_focus=True)


def test_window_bounds_intersects_and_to_dict() -> None:
    a = WindowBounds(0, 0, 1920, 971)
    b = WindowBounds(320, 180, 1280, 720)
    c = WindowBounds(1700, 0, 200, 200)
    assert a.intersects(b) is True
    assert b.intersects(c) is False
    assert b.center == (960.0, 540.0)
    win = WindowInfo(156, "GTA5.exe", "Grand Theft Auto V", 1167, b, obstructed_by=("OBS Studio(pid=1074,wid=95)",))
    assert win.to_dict()["obstructed_by"] == ["OBS Studio(pid=1074,wid=95)"]


def test_focus_game_window_warps_and_optionally_clicks_window_center(tmp_path) -> None:
    class _CenterAwareBackend(MockWindowBackend):
        def __init__(self, windows: list[WindowInfo]) -> None:
            super().__init__(windows, {})
            self.warped: list[tuple[float, float]] = []
            self.clicked: list[tuple[tuple[float, float], str, float]] = []

        def warp_cursor_to_window_center(self, window: WindowInfo) -> tuple[float, float]:
            self.warped.append(window.bounds.center)
            return window.bounds.center

        def click_window_center(
            self, window: WindowInfo, button: str = "right", hold_s: float = 0.04
        ) -> tuple[float, float]:
            self.clicked.append((window.bounds.center, button, hold_s))
            return window.bounds.center

    win = WindowInfo(
        window_id=156,
        owner="GTA5.exe",
        title="Grand Theft Auto V",
        pid=1167,
        bounds=WindowBounds(100.0, 200.0, 1280.0, 720.0),
        is_frontmost=False,
    )
    backend = _CenterAwareBackend([win])
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")

    focus_game_window(backend=backend, logger=logger, allow_focus=True)
    assert backend.warped == [(740.0, 560.0)]
    assert backend.clicked == []

    focus_game_window(
        backend=backend,
        logger=logger,
        allow_focus=True,
        click_to_activate=True,
        activate_button="right",
    )
    assert backend.clicked == [((740.0, 560.0), "right", 0.04)]
    recs = read_telemetry(tmp_path / "telemetry.jsonl", tool="window")
    assert recs[-1]["details"]["clicked_to_activate"] == "right"
    assert recs[-1]["details"]["cursor_warped_to"] == [740.0, 560.0]


def test_focus_game_window_cycles_app_focus_before_activate_and_click(tmp_path) -> None:
    class _CycleAwareBackend(MockWindowBackend):
        def __init__(self, windows: list[WindowInfo], yield_pid: int | None = 505) -> None:
            super().__init__(windows, {})
            self._yield_pid = yield_pid
            self.events: list[tuple[str, object]] = []

        def yield_app_focus(self, exclude_pid: int, settle_s: float = 0.25) -> int | None:
            self.events.append(("yield", (exclude_pid, settle_s)))
            self._windows = [
                WindowInfo(
                    window_id=w.window_id,
                    owner=w.owner,
                    title=w.title,
                    pid=w.pid,
                    bounds=w.bounds,
                    layer=w.layer,
                    is_onscreen=w.is_onscreen,
                    is_frontmost=False,
                    obstructed_by=w.obstructed_by,
                )
                for w in self._windows
            ]
            return self._yield_pid

        def focus_window(self, window: WindowInfo) -> bool:
            self.events.append(("focus", window.window_id))
            return super().focus_window(window)

        def click_window_center(
            self, window: WindowInfo, button: str = "right", hold_s: float = 0.04
        ) -> tuple[float, float]:
            self.events.append(("click", (window.bounds.center, button, hold_s)))
            return window.bounds.center

    win = WindowInfo(
        window_id=135,
        owner="GTA5.exe",
        title="Grand Theft Auto V",
        pid=1212,
        bounds=WindowBounds(320.0, 180.0, 1280.0, 720.0),
        is_frontmost=True,
    )
    backend = _CycleAwareBackend([win], yield_pid=505)
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")

    focused = focus_game_window(
        backend=backend,
        logger=logger,
        allow_focus=True,
        cycle_app_focus=True,
        yield_settle_s=0.3,
        click_to_activate=True,
        activate_button="right",
    )
    assert focused.is_frontmost is True
    assert backend.events == [
        ("yield", (1212, 0.3)),
        ("focus", 135),
        ("click", ((960.0, 540.0), "right", 0.04)),
    ]
    recs = read_telemetry(tmp_path / "telemetry.jsonl", tool="window")
    assert recs[-1]["details"]["cycled_app_focus"] is True
    assert recs[-1]["details"]["yielded_to_pid"] == 505

    no_target_backend = _CycleAwareBackend([win], yield_pid=None)
    focus_game_window(
        backend=no_target_backend,
        logger=logger,
        allow_focus=True,
        cycle_app_focus=True,
    )
    recs = read_telemetry(tmp_path / "telemetry.jsonl", tool="window")
    assert recs[-1]["details"]["cycled_app_focus"] is False
    assert "yielded_to_pid" not in recs[-1]["details"]

    plain_backend = MockWindowBackend([win], {})
    focus_game_window(
        backend=plain_backend,
        logger=logger,
        allow_focus=True,
        cycle_app_focus=True,
    )
    recs = read_telemetry(tmp_path / "telemetry.jsonl", tool="window")
    assert recs[-1]["details"]["cycled_app_focus"] is False


def test_quartz_yield_app_focus_prefers_finder_with_mocked_appkit() -> None:
    class _FakeApp:
        def __init__(self, pid: int, bundle_id: str, policy: int = 0) -> None:
            self._pid = pid
            self._bundle_id = bundle_id
            self._policy = policy
            self.activated_options: list[int] = []

        def processIdentifier(self) -> int:
            return self._pid

        def bundleIdentifier(self) -> str:
            return self._bundle_id

        def activationPolicy(self) -> int:
            return self._policy

        def activateWithOptions_(self, opts: int) -> bool:
            self.activated_options.append(opts)
            return True

    class _FakeWorkspace:
        def __init__(self, apps: list[_FakeApp]) -> None:
            self._apps = apps

        def runningApplications(self) -> list[_FakeApp]:
            return self._apps

    # Note: QuartzWindowBackend reads NSWorkspace.sharedWorkspace()
    gta = _FakeApp(1212, "com.codeweavers.CrossOver", policy=0)
    other = _FakeApp(2020, "com.example.Other", policy=0)
    finder = _FakeApp(505, "com.apple.finder", policy=0)

    backend = QuartzWindowBackend.__new__(QuartzWindowBackend)
    pumped: list[float] = []
    backend._pump_runloop = lambda s=0.05: pumped.append(s)  # type: ignore[method-assign]

    class _NSWorkspaceCls:
        def __init__(self, ws: _FakeWorkspace) -> None:
            self._ws = ws

        def sharedWorkspace(self) -> _FakeWorkspace:
            return self._ws

    class _AppKitNamespace:
        NSApplicationActivationPolicyRegular = 0

        def __init__(self, apps: list[_FakeApp]) -> None:
            self.NSWorkspace = _NSWorkspaceCls(_FakeWorkspace(apps))

    backend._appkit = _AppKitNamespace([gta, other, finder])
    assert backend.yield_app_focus(1212, settle_s=0.2) == 505
    assert finder.activated_options == [2]
    assert gta.activated_options == []
    assert other.activated_options == []
    assert pumped == [0.2]

    backend._appkit = _AppKitNamespace([gta])
    assert backend.yield_app_focus(1212, settle_s=0.2) is None
