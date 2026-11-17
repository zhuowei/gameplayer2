from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from gameplayer.telemetry import TelemetryLogger, read_telemetry
from gameplayer.window import (
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
    win = WindowInfo(156, "GTA5.exe", "Grand Theft Auto V", 1167, b, obstructed_by=("OBS Studio(pid=1074,wid=95)",))
    assert win.to_dict()["obstructed_by"] == ["OBS Studio(pid=1074,wid=95)"]
