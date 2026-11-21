"""Fast black-box screen-space perception: human/enemy targeting and optical-flow camera tracking."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from gameplayer.minimap import MinimapObservation, RadarTarget


@dataclass(frozen=True)
class ScreenTarget:
    """Detected person/enemy bounding box in normalized full-frame coordinates (top-left origin)."""

    bbox: tuple[float, float, float, float]  # (x, y, w, h)
    aim_point: tuple[float, float]           # (ax, ay) upper-torso/chest target
    confidence: float
    source: str = "vision_human"


@dataclass(frozen=True)
class CameraMotion:
    """Measured frame-to-frame camera pixel delta estimated via background phase correlation."""

    delta_yaw_px: float    # + = camera panned right (world moved left)
    delta_pitch_px: float  # + = camera pitched down (world moved up)
    confidence: float      # 0.0..1.0 normalized correlation peak


@dataclass(frozen=True)
class VisualSteerTarget:
    """Selected enemy target and proportional ADS steering delta toward `aim_point`."""

    target: ScreenTarget
    dx: int
    dy: int
    error_x: float
    error_y: float


class HumanDetectorBackend(Protocol):
    """Protocol for screen-space person/silhouette detection."""

    def detect_humans(
        self,
        image: Image.Image,
        roi: Optional[tuple[float, float, float, float]] = None,
    ) -> list[ScreenTarget]:
        ...


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _clamp_roi(roi: Optional[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if roi is None:
        return (0.15, 0.18, 0.88, 0.78)
    x0, y0, x1, y1 = roi
    x0 = _clamp(float(x0), 0.0, 0.95)
    y0 = _clamp(float(y0), 0.0, 0.95)
    x1 = _clamp(float(x1), x0 + 0.05, 1.0)
    y1 = _clamp(float(y1), y0 + 0.05, 1.0)
    return (x0, y0, x1, y1)


class VisionHumanDetectorBackend:
    """macOS Vision (`VNDetectHumanRectanglesRequest`) upper-body detector on upscaled horizon ROIs."""

    def __init__(
        self,
        upper_body_only: bool = True,
        min_crop_height: int = 360,
        min_confidence: float = 0.45,
    ) -> None:
        import Foundation  # noqa: F401
        import Vision  # noqa: F401

        self._foundation = Foundation
        self._vision = Vision
        self.upper_body_only = upper_body_only
        self.min_crop_height = min_crop_height
        self.min_confidence = min_confidence

    def detect_humans(
        self,
        image: Image.Image,
        roi: Optional[tuple[float, float, float, float]] = None,
    ) -> list[ScreenTarget]:
        rx0, ry0, rx1, ry1 = _clamp_roi(roi)
        rgb = image.convert("RGB")
        w, h = rgb.size
        px0 = max(0, int(round(rx0 * w)))
        py0 = max(0, int(round(ry0 * h)))
        px1 = max(px0 + 4, min(w, int(round(rx1 * w))))
        py1 = max(py0 + 4, min(h, int(round(ry1 * h))))
        crop = rgb.crop((px0, py0, px1, py1))

        if self.min_crop_height > 0 and crop.height < self.min_crop_height:
            scale = max(2, int(round(self.min_crop_height / max(1, crop.height))))
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        crop.save(buf, format="BMP")
        raw = buf.getvalue()
        ns_data = self._foundation.NSData.dataWithBytes_length_(raw, len(raw))

        req = self._vision.VNDetectHumanRectanglesRequest.alloc().init()
        if hasattr(req, "setUpperBodyOnly_"):
            req.setUpperBodyOnly_(bool(self.upper_body_only))

        handler = self._vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        ok, _err = handler.performRequests_error_([req], None)
        if not ok:
            return []

        roi_w = (px1 - px0) / max(1, w)
        roi_h = (py1 - py0) / max(1, h)
        roi_x0 = px0 / max(1, w)
        roi_y0 = py0 / max(1, h)

        targets: list[ScreenTarget] = []
        for obs in req.results() or ():
            conf = float(obs.confidence())
            if conf < self.min_confidence:
                continue
            bb = obs.boundingBox()
            bx = float(bb.origin.x)
            bw = float(bb.size.width)
            bh = float(bb.size.height)
            # Vision uses bottom-left origin; convert to top-left full-frame normalized coords.
            by = max(0.0, 1.0 - float(bb.origin.y) - bh)
            fx = _clamp(roi_x0 + bx * roi_w, 0.0, 1.0)
            fy = _clamp(roi_y0 + by * roi_h, 0.0, 1.0)
            fw = _clamp(bw * roi_w, 0.005, 1.0 - fx)
            fh = _clamp(bh * roi_h, 0.005, 1.0 - fy)
            aim_x = fx + 0.5 * fw
            aim_y = fy + (0.35 if self.upper_body_only else 0.28) * fh
            targets.append(
                ScreenTarget(
                    bbox=(fx, fy, fw, fh),
                    aim_point=(aim_x, aim_y),
                    confidence=conf,
                    source="vision_human",
                )
            )
        targets.sort(key=lambda t: t.confidence, reverse=True)
        return targets


class SilhouetteHumanDetectorBackend:
    """Fast pure-numpy upper-body silhouette detector for offline/synthetic frames and Vision fallback."""

    def __init__(
        self,
        contrast_threshold: float = 28.0,
        min_height: float = 0.035,
        max_height: float = 0.28,
        max_width: float = 0.12,
        min_aspect_ratio: float = 1.15,
    ) -> None:
        self.contrast_threshold = contrast_threshold
        self.min_height = min_height
        self.max_height = max_height
        self.max_width = max_width
        self.min_aspect_ratio = min_aspect_ratio

    def detect_humans(
        self,
        image: Image.Image,
        roi: Optional[tuple[float, float, float, float]] = None,
    ) -> list[ScreenTarget]:
        rx0, ry0, rx1, ry1 = _clamp_roi(roi)
        rgb = image.convert("RGB")
        w, h = rgb.size
        # Downsample full image to 320x180 for fast connected-component extraction.
        gw, gh = 320, 180
        small = np.asarray(rgb.resize((gw, gh), Image.Resampling.BILINEAR), dtype=np.float32)
        lum = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]

        ix0 = max(0, int(round(rx0 * gw)))
        iy0 = max(0, int(round(ry0 * gh)))
        ix1 = max(ix0 + 4, min(gw, int(round(rx1 * gw))))
        iy1 = max(iy0 + 4, min(gh, int(round(ry1 * gh))))
        patch = lum[iy0:iy1, ix0:ix1]
        if patch.size == 0 or float(np.std(patch)) < 4.0:
            return []

        # Compare each column against row-wise horizon median to isolate upright figures from sky/ground bands.
        row_med = np.median(patch, axis=1, keepdims=True)
        diff = np.abs(patch - row_med)
        mask = diff >= self.contrast_threshold
        if int(np.count_nonzero(mask)) < 8:
            return []

        # Simple 4-connected BFS on the small grid patch.
        ph, pw = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        targets: list[ScreenTarget] = []
        ys, xs = np.nonzero(mask)
        for sy, sx in zip(ys.tolist(), xs.tolist()):
            if visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            min_x = max_x = sx
            min_y = max_y = sy
            area = 0
            strength_sum = 0.0
            while stack:
                cy, cx = stack.pop()
                area += 1
                strength_sum += float(diff[cy, cx])
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < ph and 0 <= nx < pw and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if area < 8:
                continue
            px_w = max_x - min_x + 1
            px_h = max_y - min_y + 1
            bw = px_w / gw
            bh = px_h / gh
            if bh < self.min_height or bh > self.max_height or bw > self.max_width:
                continue
            if (px_h / max(1, px_w)) < self.min_aspect_ratio:
                continue
            fill = area / max(1, px_w * px_h)
            if fill < 0.35:
                continue
            if px_w >= 6 and px_h >= 6:
                sub_patch = patch[min_y + 2 : max_y - 1, min_x + 2 : max_x - 1]
                interior_std = float(np.std(sub_patch)) if sub_patch.size > 0 else 0.0
                if fill > 0.90 and interior_std < 5.0:
                    continue

            fx = (ix0 + min_x) / gw
            fy = (iy0 + min_y) / gh
            mean_contrast = strength_sum / area
            conf = _clamp(0.40 + 0.35 * min(1.0, mean_contrast / 70.0) + 0.15 * fill, 0.0, 0.95)
            targets.append(
                ScreenTarget(
                    bbox=(fx, fy, bw, bh),
                    aim_point=(fx + 0.5 * bw, fy + 0.35 * bh),
                    confidence=conf,
                    source="silhouette",
                )
            )

        targets.sort(key=lambda t: t.confidence, reverse=True)
        return targets


def _is_player_foreground_box(target: ScreenTarget) -> bool:
    """Filter out Trevor's third-person body/shoulder on the lower-left foreground."""
    x, y, w, h = target.bbox
    ax, _ay = target.aim_point
    bottom = y + h
    if ax < 0.46 and bottom >= 0.76 and h >= 0.22:
        return True
    if x < 0.43 and h >= 0.36 and bottom >= 0.80:
        return True
    return False


def expected_screen_x_from_bearing(bearing: float, bearing_scale: float = 2.2) -> float:
    """Project minimap bearing (-1..+1) into approximate 16:9 ADS normalized screen X."""
    return _clamp(0.5 + float(bearing) * bearing_scale, 0.02, 0.98)


def select_visual_enemy_target(
    candidates: Sequence[ScreenTarget],
    minimap: MinimapObservation,
    preferred_enemy: Optional[RadarTarget] = None,
    visual_gain_x: float = 190.0,
    visual_gain_y: float = 170.0,
    aim_deadband: float = 0.015,
    max_step_x: int = 55,
    max_step_y: int = 32,
) -> Optional[VisualSteerTarget]:
    """Filter player/ally boxes, match against minimap enemy bearing, and compute ADS `(dx, dy)`."""
    if not candidates:
        return None

    red_xs = [
        expected_screen_x_from_bearing(t.bearing)
        for t in minimap.targets.get("red", ())
        if abs(t.bearing) <= 0.45
    ]
    blue_xs = [
        expected_screen_x_from_bearing(t.bearing)
        for t in minimap.targets.get("blue", ())
        if abs(t.bearing) <= 0.45
    ]
    pref_x = (
        expected_screen_x_from_bearing(preferred_enemy.bearing)
        if preferred_enemy is not None and abs(preferred_enemy.bearing) <= 0.45
        else None
    )

    best: Optional[ScreenTarget] = None
    best_score = -1e9
    for cand in candidates:
        if _is_player_foreground_box(cand):
            continue
        ax, ay = cand.aim_point
        if ax < 0.06 or ax > 0.94 or ay < 0.18 or ay > 0.80:
            continue

        # Reject detections tightly aligned with a blue ally when no red enemy is nearby.
        if blue_xs:
            min_blue = min(abs(ax - bx) for bx in blue_xs)
            min_red = min((abs(ax - rx) for rx in red_xs), default=1.0)
            if min_blue < 0.09 and min_red > 0.16:
                continue

        score = cand.confidence
        if pref_x is not None:
            dist_pref = abs(ax - pref_x)
            if dist_pref > 0.38:
                continue
            score -= dist_pref * 1.1
        elif red_xs:
            min_red = min(abs(ax - rx) for rx in red_xs)
            score -= min_red * 0.8
        # Prefer detections near the combat horizon band (ay ~ 0.46)
        score -= abs(ay - 0.46) * 0.4

        if score > best_score:
            best_score = score
            best = cand

    if best is None:
        return None

    err_x = best.aim_point[0] - 0.5
    err_y = best.aim_point[1] - 0.5
    dx = 0 if abs(err_x) <= aim_deadband else int(round(_clamp(err_x * visual_gain_x, -max_step_x, max_step_x)))
    dy = 0 if abs(err_y) <= aim_deadband else int(round(_clamp(err_y * visual_gain_y, -max_step_y, max_step_y)))
    return VisualSteerTarget(
        target=best,
        dx=dx,
        dy=dy,
        error_x=err_x,
        error_y=err_y,
    )


class CameraMotionTracker:
    """Fast background-ROI 2D FFT phase-correlation tracker for camera `(Δyaw_px, Δpitch_px)`."""

    def __init__(
        self,
        roi: tuple[float, float, float, float] = (0.32, 0.14, 0.94, 0.58),
        grid_size: tuple[int, int] = (192, 80),  # (width, height)
        min_texture_std: float = 1.5,
    ) -> None:
        self.roi = roi
        self.grid_size = grid_size
        self.min_texture_std = min_texture_std
        self._prev_fft: Optional[np.ndarray] = None
        self._shape: Optional[tuple[int, int]] = None
        self._scale_xy: tuple[float, float] = (1.0, 1.0)
        self._window: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_fft = None
        self._shape = None

    def _prepare_patch(
        self,
        image: Image.Image,
    ) -> tuple[Optional[np.ndarray], tuple[int, int], tuple[float, float]]:
        w, h = image.size
        x0 = max(0, int(round(self.roi[0] * w)))
        y0 = max(0, int(round(self.roi[1] * h)))
        x1 = max(x0 + 16, min(w, int(round(self.roi[2] * w))))
        y1 = max(y0 + 16, min(h, int(round(self.roi[3] * h))))
        gw, gh = self.grid_size
        crop = image.crop((x0, y0, x1, y1)).convert("L").resize((gw, gh), Image.Resampling.BILINEAR)
        g = np.asarray(crop, dtype=np.float32)

        # 4-neighbor high-pass removes global brightness/siren flashes while locking onto edges.
        hp = g[1:-1, 1:-1] - 0.25 * (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        th, tw = hp.shape
        scale_xy = ((x1 - x0) / float(gw), (y1 - y0) / float(gh))
        if float(np.std(hp)) < self.min_texture_std:
            return None, (th, tw), scale_xy

        hp = hp - float(np.mean(hp))
        if self._window is None or self._window.shape != (th, tw):
            self._window = np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32)
        return np.fft.rfft2(hp * self._window), (th, tw), scale_xy

    @staticmethod
    def _subpixel_offset(idx: int, size: int, line: np.ndarray) -> float:
        xm = (idx - 1) % size
        xp = (idx + 1) % size
        denom = 2.0 * float(line[xm] - 2.0 * line[idx] + line[xp])
        if abs(denom) > 1e-6:
            delta = float(line[xm] - line[xp]) / denom
            delta = _clamp(delta, -0.5, 0.5)
        else:
            delta = 0.0
        val = float(idx) + delta
        if val > size / 2.0:
            val -= float(size)
        return val

    def update(self, image: Image.Image) -> CameraMotion:
        cur_fft, shape, scale_xy = self._prepare_patch(image)
        if cur_fft is None:
            self._prev_fft = None
            self._shape = None
            return CameraMotion(delta_yaw_px=0.0, delta_pitch_px=0.0, confidence=0.0)

        if self._prev_fft is None or self._shape != shape or self._scale_xy != scale_xy:
            self._prev_fft = cur_fft
            self._shape = shape
            self._scale_xy = scale_xy
            return CameraMotion(delta_yaw_px=0.0, delta_pitch_px=0.0, confidence=0.0)

        cross = cur_fft * np.conj(self._prev_fft)
        mag = np.abs(cross)
        norm_cross = np.divide(cross, mag, out=np.zeros_like(cross), where=(mag > 1e-6))
        cc = np.fft.irfft2(norm_cross, s=shape)

        iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        peak = float(cc[iy, ix])
        confidence = _clamp(peak * 1.4, 0.0, 1.0)
        img_dy = self._subpixel_offset(int(iy), shape[0], cc[:, ix])
        img_dx = self._subpixel_offset(int(ix), shape[1], cc[iy, :])

        self._prev_fft = cur_fft
        self._shape = shape
        self._scale_xy = scale_xy

        # World shifts opposite camera motion: camera right (+) -> image shifts left (-).
        return CameraMotion(
            delta_yaw_px=-img_dx * scale_xy[0],
            delta_pitch_px=-img_dy * scale_xy[1],
            confidence=confidence,
        )


@dataclass(frozen=True)
class PostureStatus:
    """Visual third-person posture / cover classification (`in_cover` vs `standing` vs `unknown`)."""

    state: str                   # "in_cover" | "standing" | "unknown"
    confidence: float            # 0.0..1.0
    cover_wall_side: Optional[str] = None  # "left" | "right" | None
    left_vertical_ratio: float = 0.0
    right_vertical_ratio: float = 0.0
    head_top_y: Optional[float] = None


def classify_posture(
    image: Image.Image,
    human_detector: Optional[HumanDetectorBackend] = None,
) -> PostureStatus:
    """Classify GTA 5 third-person posture (`in_cover` vs `standing`) from vertical cover continuity & silhouette."""
    gray = np.asarray(image.convert("L").resize((320, 180), Image.Resampling.BILINEAR), dtype=np.float32)
    if float(np.std(gray)) < 4.0:
        return PostureStatus(state="unknown", confidence=0.0)

    def _zone_metrics(x0_rel: float, x1_rel: float) -> tuple[float, float, float]:
        x0 = max(0, int(round(x0_rel * 320)))
        x1 = min(320, int(round(x1_rel * 320)))
        patch = gray[27:153, x0:x1]
        up = gray[27:76, x0:x1]
        lo = gray[99:153, x0:x1]
        cont = abs(float(np.mean(up)) - float(np.mean(lo)))
        gx = float(np.mean(np.abs(np.diff(patch, axis=1))))
        gy = float(np.mean(np.abs(np.diff(patch, axis=0))))
        return gx / (gy + 0.1), cont, gx + gy

    l_ratio, l_cont, l_grad = _zone_metrics(0.18, 0.36)
    r_ratio, r_cont, r_grad = _zone_metrics(0.48, 0.66)
    if l_grad < 0.8 and r_grad < 0.8:
        return PostureStatus(state="unknown", confidence=0.0)

    head_top_y: Optional[float] = None
    if human_detector is not None:
        fg = [
            t for t in human_detector.detect_humans(image, roi=(0.18, 0.35, 0.50, 0.99))
            if t.bbox[3] >= 0.24 and (t.bbox[1] + t.bbox[3]) >= 0.76
        ]
        if fg:
            head_top_y = min(t.bbox[1] for t in fg)

    left_wall = l_ratio >= 1.15 and l_cont <= 14.0
    right_wall = r_ratio >= 1.15 and r_cont <= 14.0
    if left_wall or right_wall:
        wall_side = "left" if (left_wall and (not right_wall or l_ratio >= r_ratio)) else "right"
        peak_ratio = max(l_ratio if left_wall else 0.0, r_ratio if right_wall else 0.0)
        conf = _clamp(0.62 + 0.25 * min(1.0, (peak_ratio - 1.15) / 0.25), 0.0, 0.95)
        return PostureStatus(
            state="in_cover",
            confidence=conf,
            cover_wall_side=wall_side,
            left_vertical_ratio=l_ratio,
            right_vertical_ratio=r_ratio,
            head_top_y=head_top_y,
        )

    if head_top_y is not None and head_top_y >= 0.52:
        return PostureStatus(
            state="in_cover",
            confidence=0.68,
            cover_wall_side=None,
            left_vertical_ratio=l_ratio,
            right_vertical_ratio=r_ratio,
            head_top_y=head_top_y,
        )

    if l_ratio < 1.04 and r_ratio < 1.04:
        margin = max(0.0, 1.04 - max(l_ratio, r_ratio))
        conf = _clamp(0.62 + 0.25 * min(1.0, margin / 0.20), 0.0, 0.92)
        return PostureStatus(
            state="standing",
            confidence=conf,
            cover_wall_side=None,
            left_vertical_ratio=l_ratio,
            right_vertical_ratio=r_ratio,
            head_top_y=head_top_y,
        )

    return PostureStatus(
        state="unknown",
        confidence=0.35,
        cover_wall_side=None,
        left_vertical_ratio=l_ratio,
        right_vertical_ratio=r_ratio,
        head_top_y=head_top_y,
    )
