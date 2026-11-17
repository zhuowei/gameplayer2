"""Black-box window discovery and clean unobstructed 16:9 window capture."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from gameplayer.telemetry import TelemetryLogger, get_default_logger

TARGET_ASPECT = 16.0 / 9.0
DEFAULT_OWNER_PATTERNS = ("GTA5.exe", "Grand Theft Auto")
DEFAULT_TITLE_PATTERNS = ("Grand Theft Auto V", "Grand Theft Auto", "GTA5")


class WindowNotFoundError(RuntimeError):
    """Raised when the target game window cannot be located."""


class WindowCaptureError(RuntimeError):
    """Raised when capturing or focusing a window fails."""


@dataclass(frozen=True)
class WindowBounds:
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def intersects(self, other: "WindowBounds") -> bool:
        return (
            self.x < (other.x + other.width)
            and (self.x + self.width) > other.x
            and self.y < (other.y + other.height)
            and (self.y + self.height) > other.y
        )


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    owner: str
    title: str
    pid: int
    bounds: WindowBounds
    layer: int = 0
    is_onscreen: bool = True
    is_frontmost: bool = False
    obstructed_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "owner": self.owner,
            "title": self.title,
            "pid": self.pid,
            "bounds": {
                "x": self.bounds.x,
                "y": self.bounds.y,
                "width": self.bounds.width,
                "height": self.bounds.height,
            },
            "layer": self.layer,
            "is_onscreen": self.is_onscreen,
            "is_frontmost": self.is_frontmost,
            "obstructed_by": list(self.obstructed_by),
        }


@dataclass(frozen=True)
class CaptureResult:
    image: Image.Image
    window: WindowInfo
    raw_size: tuple[int, int]
    size: tuple[int, int]
    was_cropped: bool
    aspect_ratio: float
    timestamp_utc: str

    def to_numpy(self) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB numpy array."""
        return np.asarray(self.image, dtype=np.uint8)

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(out)
        return out


class WindowBackend(Protocol):
    """Abstract window discovery/capture backend for testability."""

    def list_windows(self) -> list[WindowInfo]:
        ...

    def capture_window_image(self, window_id: int) -> Image.Image:
        ...

    def focus_window(self, window: WindowInfo) -> bool:
        ...


def is_valid_16_9(width: int, height: int, tolerance: float = 0.02) -> bool:
    """Return True if width/height is within `tolerance` of 16:9."""
    if width <= 0 or height <= 0:
        return False
    return abs((width / height) - TARGET_ASPECT) <= tolerance


def normalize_to_16_9(
    image: Image.Image,
    crop_titlebar: bool = True,
    tolerance: float = 0.02,
) -> tuple[Image.Image, bool]:
    """Crop an image to 16:9 if its aspect ratio differs beyond `tolerance`.

    When taller than 16:9 and `crop_titlebar=True`, excess pixels are cropped
    from the top (standard macOS window title bar location). When wider than
    16:9, excess pixels are cropped symmetrically left/right.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    if is_valid_16_9(width, height, tolerance=tolerance):
        return image, False

    aspect = width / height
    if aspect < TARGET_ASPECT:
        target_h = max(1, int(round(width * 9.0 / 16.0)))
        if crop_titlebar:
            top = max(0, height - target_h)
            box = (0, top, width, height)
        else:
            top = max(0, (height - target_h) // 2)
            box = (0, top, width, top + target_h)
    else:
        target_w = max(1, int(round(height * 16.0 / 9.0)))
        left = max(0, (width - target_w) // 2)
        box = (left, 0, left + target_w, height)

    return image.crop(box), True


class _CarbonPSN(ctypes.Structure):
    _fields_ = [("highLongOfPSN", ctypes.c_uint32), ("lowLongOfPSN", ctypes.c_uint32)]


class QuartzWindowBackend:
    """macOS CoreGraphics/AppKit/Accessibility black-box window discovery & capture backend."""

    def __init__(self) -> None:
        import AppKit  # noqa: F401
        import ApplicationServices  # noqa: F401
        import Foundation  # noqa: F401
        import Quartz  # noqa: F401

        self._appkit = AppKit
        self._appserv = ApplicationServices
        self._foundation = Foundation
        self._quartz = Quartz

    def _pump_runloop(self, seconds: float = 0.05) -> None:
        until = self._foundation.NSDate.dateWithTimeIntervalSinceNow_(seconds)
        self._foundation.NSRunLoop.currentRunLoop().runUntilDate_(until)

    def _frontmost_pid(self) -> Optional[int]:
        self._pump_runloop(0.01)
        app = self._appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        return int(app.processIdentifier())

    def list_windows(self) -> list[WindowInfo]:
        q = self._quartz
        opts = q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements
        raw_list = q.CGWindowListCopyWindowInfo(opts, q.kCGNullWindowID) or []
        front_pid = self._frontmost_pid()

        parsed: list[tuple[int, str, str, int, WindowBounds, int, bool]] = []
        for entry in raw_list:
            b = entry.get("kCGWindowBounds") or {}
            bounds = WindowBounds(
                x=float(b.get("X", 0.0)),
                y=float(b.get("Y", 0.0)),
                width=float(b.get("Width", 0.0)),
                height=float(b.get("Height", 0.0)),
            )
            parsed.append(
                (
                    int(entry.get("kCGWindowNumber", 0)),
                    str(entry.get("kCGWindowOwnerName", "") or ""),
                    str(entry.get("kCGWindowName", "") or ""),
                    int(entry.get("kCGWindowOwnerPID", 0)),
                    bounds,
                    int(entry.get("kCGWindowLayer", 0)),
                    bool(entry.get("kCGWindowIsOnscreen", True)),
                )
            )

        results: list[WindowInfo] = []
        for idx, (wid, owner, title, pid, bounds, layer, onscreen) in enumerate(parsed):
            obstructors: list[str] = []
            if layer == 0 and onscreen and bounds.area > 0:
                for prev_wid, prev_owner, _, prev_pid, prev_bounds, prev_layer, prev_on in parsed[:idx]:
                    if prev_layer == 0 and prev_on and prev_pid != pid and prev_bounds.intersects(bounds):
                        label = f"{prev_owner}(pid={prev_pid},wid={prev_wid})"
                        if label not in obstructors:
                            obstructors.append(label)
            results.append(
                WindowInfo(
                    window_id=wid,
                    owner=owner,
                    title=title,
                    pid=pid,
                    bounds=bounds,
                    layer=layer,
                    is_onscreen=onscreen,
                    is_frontmost=(front_pid is not None and pid == front_pid),
                    obstructed_by=tuple(obstructors),
                )
            )
        return results

    def capture_window_image(self, window_id: int) -> Image.Image:
        q = self._quartz
        cg_img = q.CGWindowListCreateImage(
            q.CGRectNull,
            q.kCGWindowListOptionIncludingWindow,
            int(window_id),
            q.kCGWindowImageBoundsIgnoreFraming | q.kCGWindowImageNominalResolution,
        )
        if cg_img is None:
            raise WindowCaptureError(
                f"CGWindowListCreateImage returned None for window_id={window_id}. "
                "Check Screen Recording permission and that the window exists."
            )
        width = int(q.CGImageGetWidth(cg_img))
        height = int(q.CGImageGetHeight(cg_img))
        bytes_per_row = int(q.CGImageGetBytesPerRow(cg_img))
        if width <= 0 or height <= 0:
            raise WindowCaptureError(f"Captured empty image {width}x{height} for window_id={window_id}")

        provider = q.CGImageGetDataProvider(cg_img)
        raw_data = bytes(q.CGDataProviderCopyData(provider))
        bitmap_info = int(q.CGImageGetBitmapInfo(cg_img))
        byte_order = bitmap_info & q.kCGBitmapByteOrderMask
        alpha_info = bitmap_info & q.kCGBitmapAlphaInfoMask

        is_little_endian = byte_order in (q.kCGBitmapByteOrder32Little, q.kCGBitmapByteOrderDefault)
        alpha_first = alpha_info in (
            q.kCGImageAlphaPremultipliedFirst,
            q.kCGImageAlphaFirst,
            q.kCGImageAlphaNoneSkipFirst,
        )
        if is_little_endian and alpha_first:
            raw_mode = "BGRA"
        elif not is_little_endian and not alpha_first:
            raw_mode = "RGBA"
        elif is_little_endian and not alpha_first:
            raw_mode = "ABGR"
        else:
            raw_mode = "ARGB"

        rgba = Image.frombytes("RGBA", (width, height), raw_data, "raw", raw_mode, bytes_per_row, 1)
        return rgba.convert("RGB")

    def _is_window_active_and_unobstructed(self, window: WindowInfo) -> bool:
        for w in self.list_windows():
            if w.window_id == window.window_id:
                return w.is_frontmost and not w.obstructed_by
        return False

    def _activate_via_ax(self, window: WindowInfo) -> None:
        ax = self._appserv
        ax_app = ax.AXUIElementCreateApplication(int(window.pid))
        err, ax_wins = ax.AXUIElementCopyAttributeValue(ax_app, "AXWindows", None)
        if err == 0 and ax_wins:
            target_ax_win = ax_wins[0]
            for cand in ax_wins:
                _, t = ax.AXUIElementCopyAttributeValue(cand, "AXTitle", None)
                if t and str(t) == window.title:
                    target_ax_win = cand
                    break
            ax.AXUIElementSetAttributeValue(target_ax_win, "AXMain", True)
            ax.AXUIElementSetAttributeValue(target_ax_win, "AXFocused", True)
            ax.AXUIElementPerformAction(target_ax_win, "AXRaise")
        ax.AXUIElementSetAttributeValue(ax_app, "AXFrontmost", True)

    def _activate_via_carbon(self, pid: int) -> None:
        his = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        psn = _CarbonPSN()
        if his.GetProcessForPID(int(pid), ctypes.byref(psn)) == 0:
            his.SetFrontProcess(ctypes.byref(psn))

    def _activate_via_system_events(self, pid: int) -> None:
        script = (
            f'tell application "System Events" to set frontmost of '
            f'first process whose unix id is {int(pid)} to true'
        )
        subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3.0)

    def focus_window(self, window: WindowInfo, timeout_s: float = 1.5) -> bool:
        """Activate and raise the target window, verifying frontmost & unobstructed state.

        STRICT OWNERSHIP: ONLY `player` may invoke this on the live game window.
        """
        pid = int(window.pid)
        app = self._appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is None:
            return False

        # 1. Accessibility raise + frontmost (works well for Wine/unbundled apps when trusted)
        try:
            self._activate_via_ax(window)
        except Exception:
            pass

        # 2. Carbon SetFrontProcess + Cocoa activation
        try:
            self._activate_via_carbon(pid)
        except Exception:
            pass
        app.activateWithOptions_(1 << 1)

        deadline = time.monotonic() + max(0.2, float(timeout_s))
        tried_system_events = False
        while time.monotonic() < deadline:
            self._pump_runloop(0.08)
            if self._is_window_active_and_unobstructed(window):
                return True
            if not tried_system_events:
                tried_system_events = True
                try:
                    self._activate_via_system_events(pid)
                except Exception:
                    pass
                try:
                    self._activate_via_ax(window)
                except Exception:
                    pass

        return self._is_window_active_and_unobstructed(window)


def _score_candidate(
    window: WindowInfo,
    owner_patterns: Sequence[str],
    title_patterns: Sequence[str],
    min_width: int,
    min_height: int,
) -> Optional[tuple[int, float]]:
    if window.layer != 0 or not window.is_onscreen:
        return None
    if window.bounds.width < min_width or window.bounds.height < min_height:
        return None

    owner_l = window.owner.lower()
    title_l = window.title.lower()
    owner_hit = any(p.lower() in owner_l for p in owner_patterns)
    title_hit = any(p.lower() in title_l for p in title_patterns)
    if not owner_hit and not title_hit:
        return None

    score = (2 if title_hit else 0) + (2 if owner_hit else 0)
    if window.title.strip().lower() == "grand theft auto v":
        score += 3
    if window.owner.strip().lower() == "gta5.exe":
        score += 3
    return (score, window.bounds.area)


def find_game_window(
    backend: Optional[WindowBackend] = None,
    owner_patterns: Sequence[str] = DEFAULT_OWNER_PATTERNS,
    title_patterns: Sequence[str] = DEFAULT_TITLE_PATTERNS,
    min_width: int = 320,
    min_height: int = 180,
) -> WindowInfo:
    """Locate the running GTA 5 window purely via OS window metadata."""
    active_backend = backend or QuartzWindowBackend()
    best_win: Optional[WindowInfo] = None
    best_key: tuple[int, float] = (-1, -1.0)
    for win in active_backend.list_windows():
        key = _score_candidate(win, owner_patterns, title_patterns, min_width, min_height)
        if key is not None and key > best_key:
            best_key = key
            best_win = win
    if best_win is None:
        raise WindowNotFoundError(
            f"No matching game window found (owners={list(owner_patterns)}, titles={list(title_patterns)})"
        )
    return best_win


def capture_game_window(
    backend: Optional[WindowBackend] = None,
    logger: Optional[TelemetryLogger] = None,
    owner_patterns: Sequence[str] = DEFAULT_OWNER_PATTERNS,
    title_patterns: Sequence[str] = DEFAULT_TITLE_PATTERNS,
    target_width: Optional[int] = None,
    crop_titlebar: bool = True,
    require_16_9: bool = True,
    aspect_tolerance: float = 0.02,
) -> CaptureResult:
    """Discover and capture a clean unobstructed 16:9 image of the game window."""
    active_backend = backend or QuartzWindowBackend()
    active_logger = logger or get_default_logger()

    with active_logger.timed("window", "capture") as details:
        win = find_game_window(
            backend=active_backend,
            owner_patterns=owner_patterns,
            title_patterns=title_patterns,
        )
        raw_img = active_backend.capture_window_image(win.window_id)
        raw_size = raw_img.size
        img, was_cropped = normalize_to_16_9(
            raw_img,
            crop_titlebar=crop_titlebar,
            tolerance=aspect_tolerance,
        )
        if target_width is not None:
            if target_width <= 0:
                raise ValueError(f"target_width must be positive, got {target_width}")
            target_height = max(1, int(round(target_width * 9.0 / 16.0)))
            if img.size != (target_width, target_height):
                img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

        w, h = img.size
        if require_16_9 and not is_valid_16_9(w, h, tolerance=aspect_tolerance):
            raise WindowCaptureError(f"Captured frame {w}x{h} is not 16:9 within tolerance={aspect_tolerance}")

        aspect = round(w / h, 6)
        details.update(
            {
                "window_id": win.window_id,
                "owner": win.owner,
                "raw_size": list(raw_size),
                "size": [w, h],
                "was_cropped": was_cropped,
                "is_frontmost": win.is_frontmost,
            }
        )
        return CaptureResult(
            image=img,
            window=win,
            raw_size=raw_size,
            size=(w, h),
            was_cropped=was_cropped,
            aspect_ratio=aspect,
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )


def focus_game_window(
    backend: Optional[WindowBackend] = None,
    logger: Optional[TelemetryLogger] = None,
    *,
    allow_focus: bool = False,
    owner_patterns: Sequence[str] = DEFAULT_OWNER_PATTERNS,
    title_patterns: Sequence[str] = DEFAULT_TITLE_PATTERNS,
) -> WindowInfo:
    """Focus the game window and verify it is frontmost. ONLY `player` may call with allow_focus=True."""
    if not allow_focus:
        raise PermissionError(
            "Focusing the live game window requires allow_focus=True and is restricted exclusively to `player`."
        )
    active_backend = backend or QuartzWindowBackend()
    active_logger = logger or get_default_logger()
    with active_logger.timed("window", "focus") as details:
        win = find_game_window(
            backend=active_backend,
            owner_patterns=owner_patterns,
            title_patterns=title_patterns,
        )
        ok = active_backend.focus_window(win)
        refreshed = find_game_window(
            backend=active_backend,
            owner_patterns=owner_patterns,
            title_patterns=title_patterns,
        )
        details.update(
            {
                "window_id": refreshed.window_id,
                "pid": refreshed.pid,
                "activated": ok,
                "is_frontmost": refreshed.is_frontmost,
                "obstructed_by": list(refreshed.obstructed_by),
            }
        )
        if not ok:
            raise WindowCaptureError(
                f"Failed to bring window_id={refreshed.window_id} (pid={refreshed.pid}) to front "
                f"(is_frontmost={refreshed.is_frontmost}, obstructed_by={list(refreshed.obstructed_by)})"
            )
        return refreshed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Black-box GTA 5 window discovery and 16:9 capture")
    parser.add_argument("--inspect", action="store_true", help="Print discovered window metadata as JSON (read-only)")
    parser.add_argument("--capture", type=Path, help="Capture clean 16:9 PNG to this path (does not focus window)")
    parser.add_argument("--width", type=int, default=None, help="Optional 16:9 downscale width (e.g. 640 or 1280)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.inspect and not args.capture:
        parser.print_help()
        return 2

    if args.inspect:
        win = find_game_window()
        print(json.dumps(win.to_dict(), indent=2))

    if args.capture:
        res = capture_game_window(target_width=args.width)
        out = res.save(args.capture)
        print(
            json.dumps(
                {
                    "saved": str(out),
                    "size": list(res.size),
                    "raw_size": list(res.raw_size),
                    "was_cropped": res.was_cropped,
                    "aspect_ratio": res.aspect_ratio,
                    "window_id": res.window.window_id,
                    "is_frontmost": res.window.is_frontmost,
                    "obstructed_by": list(res.window.obstructed_by),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
