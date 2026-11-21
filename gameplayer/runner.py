"""Fast closed-loop real-time action runner (~8-12 Hz) with automatic safety stops."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from gameplayer.hud import HudObservation, OCRBackend, inspect_frame
from gameplayer.input import InputController
from gameplayer.minimap import MinimapObservation, RadarTarget, inspect_minimap
from gameplayer.perception import (
    CameraMotion,
    CameraMotionTracker,
    HumanDetectorBackend,
    PostureStatus,
    ScreenTarget,
    SilhouetteHumanDetectorBackend,
    VisionHumanDetectorBackend,
    VisualSteerTarget,
    classify_posture,
    expected_screen_x_from_bearing,
    select_visual_enemy_target,
)
from gameplayer.telemetry import TelemetryLogger, get_default_logger
from gameplayer.window import CaptureResult, WindowBackend, WindowInfo, capture_game_window, find_game_window


# Horizon-centered bidirectional sweep for GTA 5 ADS/cover-peek: scans horizon -> car-window/torso -> crouched/car-door (+22/+44) -> standing/walk-lift (-24/-42) with exact zero net drift.
DEFAULT_RETICLE_SWEEP: tuple[tuple[int, int], ...] = (
    (18, 0),     # cum=(+18,   0) horizon right
    (-36, 0),    # cum=(-18,   0) horizon left
    (18, 22),    # cum=(  0, +22) mid-range torso / car-window center
    (20, 0),     # cum=(+20, +22) torso right
    (-40, 0),    # cum=(-20, +22) torso left
    (20, 22),    # cum=(  0, +44) crouched / car-door / downhill center
    (24, 0),     # cum=(+24, +44) crouched right
    (-48, 0),    # cum=(-24, +44) crouched left
    (24, -68),   # cum=(  0, -24) standing head / slight walk-tilt lift
    (20, 0),     # cum=(+20, -24) head right
    (-40, 0),    # cum=(-20, -24) head left
    (20, -18),   # cum=(  0, -42) fresh walk->ADS upward lift
    (22, 0),     # cum=(+22, -42) elevated right
    (-22, 42),   # cum=(  0,   0) exact reset
)


@dataclass(frozen=True)
class ReticleStatus:
    is_red: bool
    center_red_pixels: int
    outer_red_fraction: float
    contrast_score: float
    is_visible: bool = True


def detect_red_reticle(image: Image.Image) -> ReticleStatus:
    """Detect GTA 5's center reticle visibility and hostile red tint at screen center (0.5, 0.5)."""
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

    lum = (r + g + b) // 3
    max_c = np.maximum(r, np.maximum(g, b))
    min_c = np.minimum(r, np.minimum(g, b))
    outer_med = float(np.median(lum[outer_mask]))
    outer_std = float(np.std(lum[outer_mask]))
    white_dot = (
        (min_c[inner_slice] >= 170)
        & ((max_c[inner_slice] - min_c[inner_slice]) <= 28)
        & ((lum[inner_slice] - outer_med) >= 18.0)
    )
    bright_glare = outer_med >= 185.0 and int(np.max(min_c[inner_slice])) >= 200
    is_visible = bool(is_red or np.any(white_dot) or bright_glare or outer_std < 1.5)

    return ReticleStatus(
        is_red=is_red,
        center_red_pixels=inner_pixels,
        outer_red_fraction=outer_red_fraction,
        contrast_score=contrast_score,
        is_visible=is_visible,
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
    stop_on_focus_loss: bool = True


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


class _CameraFeedback:
    """Tracks optical-flow camera response to detect cover yaw clamping, Wine drops, and recoil pitch drift."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.tracker = CameraMotionTracker() if enabled else None
        self.last_cmd_dx: int = 0
        self.last_cmd_dy: int = 0
        self.fired_last_tick: bool = False
        self.yaw_stall_ticks: int = 0
        self.recoil_drift_y: float = 0.0
        self.visual_steer_count: int = 0

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        self.last_cmd_dx = 0
        self.last_cmd_dy = 0
        self.fired_last_tick = False
        self.yaw_stall_ticks = 0
        self.recoil_drift_y = 0.0

    def observe(
        self,
        image: Image.Image,
        mode: str,
        logger: TelemetryLogger,
        input_ctrl: InputController,
        aim_button: str = "right",
        allow_recoil_comp: bool = True,
    ) -> tuple[Optional[CameraMotion], bool]:
        if not self.enabled or self.tracker is None:
            self.last_cmd_dx = 0
            self.last_cmd_dy = 0
            self.fired_last_tick = False
            return None, False

        motion = self.tracker.update(image)
        if motion.confidence >= 0.25 and abs(self.last_cmd_dx) >= 18:
            if abs(motion.delta_yaw_px) < 3.0:
                self.yaw_stall_ticks += 1
                if self.yaw_stall_ticks == 4:
                    logger.log(
                        tool="runner",
                        action="camera_yaw_stalled",
                        status="ok",
                        details={
                            "mode": mode,
                            "commanded_dx": self.last_cmd_dx,
                            "measured_yaw_px": round(motion.delta_yaw_px, 2),
                            "confidence": round(motion.confidence, 3),
                        },
                    )
            else:
                self.yaw_stall_ticks = 0
        elif abs(self.last_cmd_dx) >= 18:
            self.yaw_stall_ticks = 0

        if allow_recoil_comp and self.fired_last_tick and motion.confidence >= 0.25 and self.last_cmd_dy == 0:
            self.recoil_drift_y = min(0.0, self.recoil_drift_y + motion.delta_pitch_px)
            if self.recoil_drift_y <= -6.0 and aim_button in input_ctrl.held_buttons:
                comp = min(10, int(round(-self.recoil_drift_y * 0.5)))
                if comp > 0:
                    input_ctrl.mouse_delta(0, comp, steps=1, duration_s=0.01)
                    self.recoil_drift_y = min(0.0, self.recoil_drift_y + comp)
                    logger.log(
                        tool="runner",
                        action="camera_recoil_comp",
                        status="ok",
                        details={
                            "mode": mode,
                            "comp_dy": comp,
                            "residual_pitch_px": round(self.recoil_drift_y, 2),
                        },
                    )

        self.last_cmd_dx = 0
        self.last_cmd_dy = 0
        self.fired_last_tick = False
        return motion, self.yaw_stall_ticks >= 4


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
        human_detector: Optional[HumanDetectorBackend] = None,
        enable_visual_targeting: bool = True,
        enable_camera_motion_tracking: bool = True,
    ) -> None:
        self.input = input_controller
        self._window_backend = window_backend
        self._ocr_backend = ocr_backend
        self._logger = logger or get_default_logger()
        self._clock = clock
        self._sleep = sleep_fn
        self._human_detector = human_detector
        self._enable_visual_targeting = enable_visual_targeting
        self._enable_camera_motion_tracking = enable_camera_motion_tracking
        self._silhouette_fallback = SilhouetteHumanDetectorBackend()
        self._in_cover_hint: Optional[bool] = None
        self._pitch_down_sign: Optional[int] = None
        self._pitch_px_per_dy: Optional[float] = None
        self._last_flank_streak: int = 0
        self._last_flank_base_side: str = "d"

    def _get_human_detector(self) -> Optional[HumanDetectorBackend]:
        if not self._enable_visual_targeting:
            return None
        if self._human_detector is not None:
            return self._human_detector
        if self._window_backend is None:
            try:
                self._human_detector = VisionHumanDetectorBackend()
            except Exception:
                self._human_detector = self._silhouette_fallback
        else:
            self._human_detector = self._silhouette_fallback
        return self._human_detector

    _VISUAL_MICRO_DITHER_STEPS: tuple[tuple[int, int], ...] = ((6, -4), (-12, 0), (6, 8), (0, -4))

    def _detect_visual_target(
        self,
        image: Image.Image,
        minimap: MinimapObservation,
        preferred_enemy: Optional[RadarTarget],
    ) -> Optional[VisualSteerTarget]:
        detector = self._get_human_detector()
        if detector is None:
            return None

        has_fov_enemy = preferred_enemy is not None and abs(preferred_enemy.bearing) <= 0.45
        cx = expected_screen_x_from_bearing(preferred_enemy.bearing) if has_fov_enemy else 0.5
        sil_roi: tuple[float, float, float, float] = (
            max(0.05, cx - 0.14),
            0.20,
            min(0.95, cx + 0.14),
            0.76,
        )

        def _valid_silhouette(cands: Sequence[ScreenTarget]) -> list[ScreenTarget]:
            out: list[ScreenTarget] = []
            bearing = preferred_enemy.bearing if preferred_enemy is not None else 0.0
            for c in cands:
                bx, _by, bw, bh = c.bbox
                if bw >= 0.14 or bh >= 0.32:
                    continue
                if bx < 0.20 and (preferred_enemy is None or bearing >= -0.35):
                    continue
                out.append(c)
            return out

        if isinstance(detector, SilhouetteHumanDetectorBackend):
            if preferred_enemy is None or abs(preferred_enemy.bearing) > 0.35:
                return None
            cands = _valid_silhouette(detector.detect_humans(image, roi=sil_roi))
            return select_visual_enemy_target(cands, minimap, preferred_enemy=preferred_enemy)

        if preferred_enemy is not None and abs(preferred_enemy.bearing) <= 0.35:
            roi: tuple[float, float, float, float] = (
                max(0.05, cx - 0.24),
                0.20,
                min(0.95, cx + 0.24),
                0.76,
            )
        else:
            roi = (0.15, 0.20, 0.88, 0.76)

        candidates = detector.detect_humans(image, roi=roi)
        selected = select_visual_enemy_target(candidates, minimap, preferred_enemy=preferred_enemy)
        if selected is None and isinstance(detector, VisionHumanDetectorBackend):
            hood_roi: tuple[float, float, float, float] = (0.22, 0.25, 0.88, 0.58)
            if roi != hood_roi:
                hood_cands = detector.detect_humans(image, roi=hood_roi)
                selected = select_visual_enemy_target(hood_cands, minimap, preferred_enemy=preferred_enemy)
        return selected

    def _handle_visual_aligned_tick(
        self,
        *,
        fire_button: str,
        aligned_ticks: int,
        recoil_comp_dy: int,
        cam_fb: _CameraFeedbackState,
    ) -> bool:
        """Fire immediately when visually aligned (tick 1, then every 4th tick) or micro-dither for red_reticle."""
        if aligned_ticks == 1 or aligned_ticks % 4 == 0:
            self._fire_bursts(fire_button, bursts=2, hold_s=0.03, gap_s=0.03, reason="suppressive")
            cam_fb.fired_last_tick = True
            if recoil_comp_dy > 0:
                sup_scale = 1 if aligned_ticks == 1 else 2
                comp_dy = int(recoil_comp_dy) * sup_scale
                self.input.mouse_delta(0, comp_dy, steps=1, duration_s=0.01)
                cam_fb.last_cmd_dy = comp_dy
            return True
        if aligned_ticks >= 2:
            dx, dy = self._VISUAL_MICRO_DITHER_STEPS[(aligned_ticks - 2) % len(self._VISUAL_MICRO_DITHER_STEPS)]
            self.input.mouse_delta(dx, dy, steps=1, duration_s=0.01)
            cam_fb.last_cmd_dx += dx
            cam_fb.last_cmd_dy += dy
        return False

    def _snap_cover_on_health_drop(self, mode_name: str) -> None:
        self._tap_key_preserving_held("q", hold_s=0.08)
        self._in_cover_hint = True
        self._logger.log(
            tool="runner",
            action="cover_on_health_drop",
            status="ok",
            details={"mode": mode_name},
        )

    def classify_current_posture(self) -> tuple[PostureStatus, CaptureResult]:
        """Capture current frame and classify third-person posture (`in_cover` vs `standing`)."""
        cap = capture_game_window(
            backend=self._window_backend,
            logger=self._logger,
            require_16_9=True,
        )
        posture = classify_posture(cap.image, human_detector=self._get_human_detector())
        return posture, cap

    def enter_cover(self, force: bool = False, settle_s: float = 0.25) -> PostureStatus:
        """Idempotently enter cover (`q` toggle): skips `q` when already classified `in_cover`."""
        posture, before_cap = self.classify_current_posture()
        hint_hold_cover = (
            self._in_cover_hint is True
            and (
                posture.state != "standing"
                or posture.head_top_y is None
                or posture.head_top_y >= 0.45
            )
        )
        if not force and (
            (posture.state == "in_cover" and posture.confidence >= 0.60)
            or hint_hold_cover
        ):
            self._in_cover_hint = True
            self._logger.log(
                tool="runner",
                action="enter_cover_noop",
                status="ok",
                details={"state": posture.state, "confidence": round(posture.confidence, 3)},
            )
            return posture

        self._tap_key_preserving_held("q", hold_s=0.08)
        if settle_s > 0:
            self._sleep(settle_s)
        after_posture, after_cap = self.classify_current_posture()
        # Check foreground pixel shift in player region [0.22..0.52, 0.42..0.95] as confirmation
        b_fg = _road_gray_sample(before_cap.image)
        a_fg = _road_gray_sample(after_cap.image)
        shift_score = _road_motion_score(b_fg, a_fg)
        self._in_cover_hint = (after_posture.state == "in_cover") or (shift_score >= 10.0)
        self._logger.log(
            tool="runner",
            action="enter_cover",
            status="ok",
            details={
                "before_state": posture.state,
                "after_state": after_posture.state,
                "shift_score": round(shift_score, 2),
                "in_cover_hint": self._in_cover_hint,
            },
        )
        return after_posture

    def exit_cover(
        self,
        min_health: float = 0.42,
        force: bool = False,
        settle_s: float = 0.25,
    ) -> tuple[bool, PostureStatus]:
        """Idempotently exit cover (`q` toggle), refusing to stand if `health_ratio < min_health`."""
        posture, before_cap = self.classify_current_posture()
        mm = inspect_minimap(before_cap.image, logger=self._logger)
        if not force and mm.minimap_visible and mm.health_ratio < min_health:
            self._logger.log(
                tool="runner",
                action="exit_cover_blocked_low_health",
                status="ok",
                details={"health_ratio": round(mm.health_ratio, 3), "min_health": min_health},
            )
            return False, posture

        if not force and (
            (posture.state == "standing" and posture.confidence >= 0.60)
            or (posture.state == "unknown" and self._in_cover_hint is False)
        ):
            self._in_cover_hint = False
            self._logger.log(
                tool="runner",
                action="exit_cover_noop",
                status="ok",
                details={"state": posture.state, "confidence": round(posture.confidence, 3)},
            )
            return True, posture

        self._tap_key_preserving_held("q", hold_s=0.08)
        if settle_s > 0:
            self._sleep(settle_s)
        after_posture, _ = self.classify_current_posture()
        self._in_cover_hint = False
        self._logger.log(
            tool="runner",
            action="exit_cover",
            status="ok",
            details={"before_state": posture.state, "after_state": after_posture.state},
        )
        return True, after_posture

    def calibrate_controls(
        self,
        probe_dx: int = 20,
        aim_settle_s: float = 0.18,
        min_response_px: float = 3.0,
        save_path: Optional[Path | str] = Path("logs/calibration.json"),
    ) -> dict[str, object]:
        """Run a <=5s startup input, yaw ADS/hipfire gain, and vertical pitch-sign self-calibration."""
        start_t = self._clock()
        tracker = CameraMotionTracker()
        hipfire_yaw_px = 0.0
        hipfire_conf = 0.0
        ads_yaw_px = 0.0
        ads_conf = 0.0
        pitch_down_sign: Optional[int] = None
        pitch_px_per_dy = 0.0
        pitch_conf = 0.0
        minimap_visible = False

        try:
            c0 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            if hasattr(self.input.backend, "set_anchor_bounds"):
                wb = c0.window.bounds
                if wb.width > 0 and wb.height > 0:
                    self.input.backend.set_anchor_bounds(wb.x, wb.y, wb.width, wb.height)
            mm0 = inspect_minimap(c0.image, logger=self._logger)
            minimap_visible = mm0.minimap_visible
            tracker.update(c0.image)

            # 1) Hipfire yaw probe
            self.input.mouse_delta(probe_dx, 0, steps=4, duration_s=0.06)
            self._sleep(0.09)
            c1 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            m_hip = tracker.update(c1.image)
            hipfire_yaw_px = float(m_hip.delta_yaw_px)
            hipfire_conf = float(m_hip.confidence)
            self.input.mouse_delta(-probe_dx, 0, steps=4, duration_s=0.06)
            self._sleep(0.06)

            # 1b) Vertical pitch-sign probe (delta_pitch_px > 0 means camera pitched DOWN).
            probe_dy = max(24, abs(probe_dx))
            c_v0 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            tracker.update(c_v0.image)
            self.input.mouse_delta(0, probe_dy, steps=4, duration_s=0.06)
            self._sleep(0.09)
            c_v1 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            m_vp = tracker.update(c_v1.image)
            self.input.mouse_delta(0, -probe_dy, steps=4, duration_s=0.06)
            self._sleep(0.06)
            vp_dp = float(m_vp.delta_pitch_px)
            pitch_conf = float(m_vp.confidence)
            if abs(vp_dp) >= min_response_px:
                pitch_down_sign = 1 if vp_dp > 0 else -1
                pitch_px_per_dy = abs(vp_dp) / max(1, probe_dy)
            else:
                c_v2 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
                tracker.update(c_v2.image)
                self.input.mouse_delta(0, -probe_dy, steps=4, duration_s=0.06)
                self._sleep(0.09)
                c_v3 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
                m_vn = tracker.update(c_v3.image)
                self.input.mouse_delta(0, probe_dy, steps=4, duration_s=0.06)
                self._sleep(0.06)
                vn_dp = float(m_vn.delta_pitch_px)
                if abs(vn_dp) >= min_response_px:
                    pitch_down_sign = -1 if vn_dp > 0 else 1
                    pitch_px_per_dy = abs(vn_dp) / max(1, probe_dy)
                    pitch_conf = float(m_vn.confidence)

            # 2) ADS yaw probe
            self.input.mouse_down("right")
            if aim_settle_s > 0:
                self._sleep(aim_settle_s)
            c2 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            tracker.update(c2.image)
            self.input.mouse_delta(probe_dx, 0, steps=4, duration_s=0.06)
            self._sleep(0.09)
            c3 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            m_ads = tracker.update(c3.image)
            ads_yaw_px = float(m_ads.delta_yaw_px)
            ads_conf = float(m_ads.confidence)
            self.input.mouse_delta(-probe_dx, 0, steps=4, duration_s=0.06)
            self._sleep(0.02)
            self.input.mouse_up("right")
        finally:
            self.input.release_all(reason="calibrate_controls", suppress_errors=True)

        if pitch_down_sign is not None:
            self._pitch_down_sign = pitch_down_sign
        if pitch_px_per_dy > 0.0:
            self._pitch_px_per_dy = pitch_px_per_dy
        hipfire_gain = abs(hipfire_yaw_px) / max(1, abs(probe_dx))
        ads_gain = abs(ads_yaw_px) / max(1, abs(probe_dx))
        ads_hipfire_ratio = (ads_gain / hipfire_gain) if hipfire_gain > 1e-4 else 0.0
        mouse_responsive = (
            (abs(hipfire_yaw_px) >= min_response_px)
            or (abs(ads_yaw_px) >= min_response_px)
            or (pitch_down_sign is not None and pitch_px_per_dy > 0.0)
        )
        elapsed_s = max(0.0, self._clock() - start_t)
        report: dict[str, object] = {
            "elapsed_s": round(elapsed_s, 3),
            "probe_dx": probe_dx,
            "minimap_visible": minimap_visible,
            "mouse_responsive": mouse_responsive,
            "hipfire_yaw_px": round(hipfire_yaw_px, 2),
            "hipfire_confidence": round(hipfire_conf, 3),
            "hipfire_px_per_dx": round(hipfire_gain, 3),
            "ads_yaw_px": round(ads_yaw_px, 2),
            "ads_confidence": round(ads_conf, 3),
            "ads_px_per_dx": round(ads_gain, 3),
            "ads_to_hipfire_ratio": round(ads_hipfire_ratio, 3),
            "pitch_down_sign": pitch_down_sign,
            "pitch_px_per_dy": round(pitch_px_per_dy, 3),
            "pitch_confidence": round(pitch_conf, 3),
            "warning": None if mouse_responsive else "mouse_unresponsive_check_raw_input_or_focus",
        }
        self._logger.log(tool="runner", action="calibrate_controls", status="ok", details=report)
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
            tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, out)
        return report

    def level_camera_pitch(
        self,
        probe_dy: Optional[int] = None,
        recovery_dy: int = 200,
        settle_s: float = 0.09,
        min_move_px: float = 4.0,
        max_passes: int = 2,
    ) -> dict[str, object]:
        """Sign-agnostically detect ceiling/floor pitch clamping via optical flow and re-level toward horizon."""
        if probe_dy is not None:
            probe = max(12, abs(int(probe_dy)))
        elif self._pitch_px_per_dy is not None and self._pitch_px_per_dy > 0.05:
            probe = max(24, min(75, int(round(10.0 / self._pitch_px_per_dy))))
        else:
            probe = 24
        rec = max(probe, abs(int(recovery_dy)))
        total_applied_dy = 0
        final_status = "horizon_ok"
        last_pos_dp = 0.0
        last_neg_dp = 0.0

        for _ in range(max(1, int(max_passes))):
            tracker = CameraMotionTracker()
            c0 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            tracker.update(c0.image)

            # Probe +probe and return
            self.input.mouse_delta(0, probe, steps=3, duration_s=0.05)
            if settle_s > 0:
                self._sleep(settle_s)
            c_pos = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            m_pos = tracker.update(c_pos.image)
            dp_pos = float(m_pos.delta_pitch_px)

            self.input.mouse_delta(0, -probe, steps=3, duration_s=0.05)
            if settle_s > 0:
                self._sleep(settle_s)
            c_mid = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            tracker.update(c_mid.image)

            # Probe -probe and return
            self.input.mouse_delta(0, -probe, steps=3, duration_s=0.05)
            if settle_s > 0:
                self._sleep(settle_s)
            c_neg = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            m_neg = tracker.update(c_neg.image)
            dp_neg = float(m_neg.delta_pitch_px)

            self.input.mouse_delta(0, probe, steps=3, duration_s=0.05)
            if settle_s > 0:
                self._sleep(settle_s)

            last_pos_dp = dp_pos
            last_neg_dp = dp_neg

            if abs(dp_pos) < min_move_px and abs(dp_neg) < min_move_px:
                if total_applied_dy == 0:
                    final_status = "unresponsive"
                break

            pos_clamped = (
                m_pos.confidence >= 0.40
                and abs(dp_neg) >= min_move_px * 1.5
                and abs(dp_pos) < max(min_move_px, 0.45 * abs(dp_neg))
            )
            neg_clamped = (
                m_neg.confidence >= 0.40
                and abs(dp_pos) >= min_move_px * 1.5
                and abs(dp_neg) < max(min_move_px, 0.45 * abs(dp_pos))
            )

            # delta_pitch_px > 0 means camera pitched DOWN; < 0 means pitched UP.
            if pos_clamped:
                # +probe hit limit; -probe moved freely -> move in -probe direction.
                self._pitch_down_sign = -1 if dp_neg > 0 else 1
                self.input.mouse_delta(0, -rec, steps=6, duration_s=0.09)
                if settle_s > 0:
                    self._sleep(settle_s)
                total_applied_dy -= rec
                final_status = "recovered_ceiling_clamp" if dp_neg > 0 else "recovered_floor_clamp"
            elif neg_clamped:
                # -probe hit limit; +probe moved freely -> move in +probe direction.
                self._pitch_down_sign = 1 if dp_pos > 0 else -1
                self.input.mouse_delta(0, rec, steps=6, duration_s=0.09)
                if settle_s > 0:
                    self._sleep(settle_s)
                total_applied_dy += rec
                final_status = "recovered_ceiling_clamp" if dp_pos > 0 else "recovered_floor_clamp"
            else:
                if abs(dp_pos) >= abs(dp_neg) and abs(dp_pos) >= min_move_px:
                    self._pitch_down_sign = 1 if dp_pos > 0 else -1
                elif abs(dp_neg) >= min_move_px:
                    self._pitch_down_sign = -1 if dp_neg > 0 else 1
                if total_applied_dy == 0:
                    final_status = "horizon_ok"
                break

        report: dict[str, object] = {
            "status": final_status,
            "applied_dy": total_applied_dy,
            "pitch_down_sign": self._pitch_down_sign,
            "pos_probe_pitch_px": round(last_pos_dp, 2),
            "neg_probe_pitch_px": round(last_neg_dp, 2),
        }
        self._logger.log(tool="runner", action="level_camera_pitch", status="ok", details=report)
        return report

    def _run_loop(
        self,
        mode_name: str,
        config: LoopConfig,
        on_tick: Callable[[FrameState], Optional[str]],
        initial_keys: Iterable[str] = (),
        initial_buttons: Iterable[str] = (),
        on_health_drop: Optional[Callable[[], None]] = None,
        on_before_release: Optional[Callable[[], None]] = None,
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
        last_logged_help = ""
        last_logged_subtitle = ""
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
                    if config.stop_on_focus_loss and not cap.window.is_frontmost:
                        self._logger.log(
                            tool="runner",
                            action="focus_lost",
                            status="ok",
                            details={"mode": mode_name, "obstructed_by": list(cap.window.obstructed_by)},
                        )
                        stop_reason = "focus_lost"
                        ticks += 1
                        break
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

                        def _meaningful_hud_text(text: str) -> str:
                            cleaned = text.strip()
                            words = [w for w in cleaned.split() if any(ch.isalpha() for ch in w)]
                            if len(cleaned) >= 6 and len(words) >= 2 and sum(1 for ch in cleaned if ch.isalnum()) >= 5:
                                return cleaned
                            return ""

                        help_text = _meaningful_hud_text(hud.help_prompt)
                        sub_text = _meaningful_hud_text(hud.subtitle_or_objective)
                        help_changed = bool(help_text) and help_text != last_logged_help
                        sub_changed = bool(sub_text) and sub_text != last_logged_subtitle
                        if help_changed or sub_changed:
                            prompt_details: dict[str, Any] = {}
                            if help_changed:
                                last_logged_help = help_text
                                prompt_details["help_prompt"] = help_text[:160]
                            if sub_changed:
                                last_logged_subtitle = sub_text
                                prompt_details["subtitle_or_objective"] = sub_text[:160]
                            self._logger.log(
                                tool="runner",
                                action="hud_prompt",
                                status="ok",
                                details=prompt_details,
                            )

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
                if on_before_release is not None:
                    try:
                        on_before_release()
                    except Exception:
                        pass
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
        steer_dx: int = 0,
        back_up_s: float = 0.0,
    ) -> None:
        # Commit 2 steps to one side before flipping so narrow doorways aren't oscillated in place.
        primary = "a" if last_bearing < -0.05 else "d"
        secondary = "d" if primary == "a" else "a"
        sidestep_key = primary if ((attempt - 1) // 2) % 2 == 0 else secondary
        self._logger.log(
            tool="runner",
            action="on_foot_stuck_recovery",
            status="ok",
            details={
                "attempt": attempt,
                "sidestep_key": sidestep_key,
                "duration_s": duration_s,
                "vault": vault,
                "steer_dx": steer_dx,
                "back_up_s": back_up_s,
            },
        )
        # Release held forward movement ("w"/"lshift") during sidestep so Trevor doesn't keep pressing into the obstacle.
        held_forward = [k for k in ("lshift", "w") if k in self.input.held_keys]
        for k in held_forward:
            self.input.key_up(k)
        try:
            eff_back_s = back_up_s if back_up_s > 0.0 else (min(0.2, duration_s * 0.5) if attempt >= 2 else 0.0)
            if eff_back_s > 0.0:
                self.input.key_down("s")
                try:
                    self._sleep_chunked(eff_back_s)
                finally:
                    self.input.key_up("s")
            if steer_dx != 0:
                self._steer_camera(steer_dx, 0)
            self.input.key_down(sidestep_key)
            try:
                if vault:
                    self._tap_key_preserving_held("space", hold_s=0.06)
                self._sleep_chunked(duration_s)
            finally:
                self.input.key_up(sidestep_key)
        finally:
            for k in reversed(held_forward):
                self.input.key_down(k)

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
    def _select_cycle_target(red_targets: Sequence[Any], target_idx: int) -> Any:
        if not red_targets:
            return None
        in_fov = [t for t in red_targets if abs(t.bearing) <= 0.45]
        pool = in_fov if in_fov else red_targets
        return pool[target_idx % len(pool)]

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
        stop_on_subtitle_keywords: Sequence[str] = ("return to", "wait for", "stay with", "follow", "get in cover"),
        enable_stuck_recovery: bool = True,
        stuck_window_s: float = 0.9,
        stuck_motion_threshold: float = 2.5,
        unstuck_duration_s: float = 0.55,
        vault_on_stuck: bool = True,
        cover_on_health_drop: bool = True,
        level_pitch: bool = False,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Walk/sprint toward minimap route/objective with crew-tether subtitle stop and obstacle recovery."""
        cfg = config or LoopConfig(max_duration_s=10.0, hz=10.0)
        self._in_cover_hint = False
        if level_pitch:
            self.level_camera_pitch()
        keys = ("w", "lshift") if sprint else ("w",)
        norm_keywords = tuple(kw.strip().lower() for kw in stop_on_subtitle_keywords if kw.strip())
        prev_gray: Optional[np.ndarray] = None
        last_motion_check_t: Optional[float] = None
        low_motion_elapsed_s = 0.0
        free_motion_elapsed_s = 0.0
        min_target_dist: Optional[float] = None
        target_stall_elapsed_s = 0.0
        last_dist_check_t: Optional[float] = None
        last_bearing = 0.0
        recovery_count = 0
        stuck_anchor_bearing: Optional[float] = None
        bypass_bearing_until_t = 0.0

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal prev_gray, last_motion_check_t, low_motion_elapsed_s, free_motion_elapsed_s, min_target_dist, target_stall_elapsed_s, last_dist_check_t, last_bearing, recovery_count, stuck_anchor_bearing, bypass_bearing_until_t
            mm = state.minimap
            target, bearing = self._resolve_route_target_and_bearing(mm, target_color)
            if state.hud is not None and norm_keywords:
                combined_hud = f"{state.hud.subtitle_or_objective} {state.hud.help_prompt}".lower()
                for kw in norm_keywords:
                    if kw in combined_hud:
                        if kw == "get in cover" and target is not None and target.distance > max(arrive_distance * 2.0, 0.16):
                            continue
                        return f"subtitle_keyword:{kw}"

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
                        free_motion_elapsed_s = 0.0
                    else:
                        low_motion_elapsed_s = 0.0
                        free_motion_elapsed_s += dt
                        if free_motion_elapsed_s >= 1.5:
                            recovery_count = 0
                            stuck_anchor_bearing = None
                prev_gray = cur_gray
                last_motion_check_t = state.elapsed_s

                if target is not None:
                    if min_target_dist is None or target.distance < (min_target_dist - 0.015):
                        min_target_dist = target.distance
                        target_stall_elapsed_s = 0.0
                    elif last_dist_check_t is not None:
                        target_stall_elapsed_s += max(0.0, state.elapsed_s - last_dist_check_t)
                    last_dist_check_t = state.elapsed_s
                else:
                    min_target_dist = None
                    target_stall_elapsed_s = 0.0
                    last_dist_check_t = None

                dist_stall_window_s = max(stuck_window_s * 2.2, 2.2)
                is_stuck = (low_motion_elapsed_s >= stuck_window_s) or (target_stall_elapsed_s >= dist_stall_window_s)
                if is_stuck and (cfg.max_duration_s - state.elapsed_s) > 0.25:
                    recovery_count += 1
                    if recovery_count == 1 or stuck_anchor_bearing is None:
                        stuck_anchor_bearing = last_bearing
                    off_axis = abs(last_bearing) > 0.35
                    allow_vault = (
                        vault_on_stuck
                        and (not off_axis)
                        and (recovery_count == 1)
                        and (mm.nearest_enemy is None)
                        and (not mm.minimap_visible or mm.health_ratio >= 0.65)
                    )
                    steer_dx = (
                        int(round(max(-320.0, min(320.0, last_bearing * max(steer_gain * 1.5, 240.0)))))
                        if off_axis
                        else 0
                    )
                    eff_unstuck_s = (
                        unstuck_duration_s
                        if off_axis
                        else min(unstuck_duration_s, 0.28 if (recovery_count % 2 == 1) else 0.42)
                    )
                    back_up_s = 0.25
                    tick_clock_0 = self._clock()
                    self._perform_on_foot_sidestep(
                        recovery_count,
                        last_bearing if off_axis else stuck_anchor_bearing,
                        eff_unstuck_s,
                        allow_vault,
                        steer_dx=steer_dx,
                        back_up_s=back_up_s,
                    )
                    if not off_axis:
                        # Hold wall-parallel heading for 1.1s after sidestep to clear narrow doorways before re-steering.
                        post_sidestep_t = state.elapsed_s + max(0.0, self._clock() - tick_clock_0)
                        bypass_bearing_until_t = post_sidestep_t + 1.1
                    low_motion_elapsed_s = 0.0
                    prev_gray = None
                    last_motion_check_t = None
                    min_target_dist = None
                    target_stall_elapsed_s = 0.0
                    last_dist_check_t = None
                    return None

            close_range_orbit = (
                target is not None
                and bearing is not None
                and target.distance <= max(arrive_distance * 3.0, 0.20)
                and abs(bearing) >= 0.28
            )
            sharp_off_axis = bearing is not None and abs(bearing) >= 0.55
            should_walk = (state.elapsed_s < bypass_bearing_until_t) or (
                not close_range_orbit and not sharp_off_axis
            )
            if should_walk:
                if "w" not in self.input.held_keys:
                    self.input.key_down("w")
                if sprint and "lshift" not in self.input.held_keys:
                    self.input.key_down("lshift")
            else:
                if "lshift" in self.input.held_keys:
                    self.input.key_up("lshift")
                if "w" in self.input.held_keys:
                    self.input.key_up("w")

            if (
                bearing is not None
                and abs(bearing) > deadband
                and state.elapsed_s >= bypass_bearing_until_t
            ):
                eff_gain = (
                    max(steer_gain, 240.0)
                    if (abs(bearing) > 0.35 and target_stall_elapsed_s >= 0.4)
                    else steer_gain
                )
                dx = int(round(max(-260.0, min(260.0, bearing * eff_gain))))
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

    def approach_and_enter_cover(
        self,
        target_color: str = "yellow",
        arrive_distance: float = 0.085,
        max_duration_s: float = 22.0,
        level_pitch: bool = True,
        probe_step_s: float = 0.30,
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Walk to cover objective (`vault_on_stuck=False`), enter cover (`q`), and probe faces until confirmed."""
        cfg = config or LoopConfig(max_duration_s=max_duration_s, hz=10.0)
        start_t = self._clock()
        self._logger.log(
            tool="runner",
            action="approach_and_enter_cover_start",
            status="ok",
            details={"target_color": target_color, "arrive_distance": arrive_distance, "max_duration_s": cfg.max_duration_s},
        )
        if level_pitch:
            self.level_camera_pitch()

        approach_budget = max(2.0, min(cfg.max_duration_s * 0.65, 14.0))
        approach_cfg = LoopConfig(
            max_duration_s=approach_budget,
            hz=cfg.hz,
            hud_check_interval_s=cfg.hud_check_interval_s,
            health_drop_threshold=cfg.health_drop_threshold,
            stop_on_banner=cfg.stop_on_banner,
        )
        approach_res = self.follow_route_on_foot(
            sprint=False,
            target_color=target_color,
            arrive_distance=arrive_distance,
            stop_on_subtitle_keywords=("return to", "wait for", "stay with", "follow"),
            vault_on_stuck=False,
            cover_on_health_drop=True,
            level_pitch=False,
            config=approach_cfg,
        )
        if approach_res.banner is not None or approach_res.stop_reason.startswith("banner:"):
            return approach_res

        last_hud = approach_res.last_hud
        last_mm = approach_res.last_minimap
        ticks = approach_res.ticks
        stop_reason = "timeout"

        def _is_cover_confirmed(mm: MinimapObservation, posture: PostureStatus) -> bool:
            target, _ = self._resolve_route_target_and_bearing(mm, target_color)
            if target is None or target.distance > max(arrive_distance * 2.5, 0.22):
                return True
            if len(mm.targets.get("red", ())) > 0 and target is None:
                return True
            if target.distance <= 0.08:
                if posture.state == "in_cover" and posture.confidence >= 0.60:
                    return True
            return False

        cover_slide_idx = 0
        probe_idx = 0
        orbit_pref_dir: Optional[str] = None
        try:
            # Align toward off-axis cover blip before first `q` attempt so Trevor faces the cover object.
            if last_mm is not None:
                t_init, b_init = self._resolve_route_target_and_bearing(last_mm, target_color)
                if t_init is not None and b_init is not None and abs(b_init) > 0.18:
                    dx_init = int(round(max(-240.0, min(240.0, b_init * 180.0))))
                    if dx_init != 0:
                        self._steer_camera(dx_init, 0)
                    self.input.key_down("w")
                    try:
                        self._sleep_chunked(min(0.20, max(0.05, cfg.max_duration_s - (self._clock() - start_t) - 0.5)))
                    finally:
                        self.input.key_up("w")

            while (self._clock() - start_t) < cfg.max_duration_s:
                if self._in_cover_hint is not True:
                    posture = self.enter_cover(force=False, settle_s=0.22)
                else:
                    posture, _ = self.classify_current_posture()
                cap = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
                mm = inspect_minimap(cap.image, logger=self._logger)
                hud = inspect_frame(cap.image, ocr_backend=self._ocr_backend, logger=self._logger)
                last_mm = mm
                last_hud = hud
                ticks += 1

                if cfg.stop_on_focus_loss and not cap.window.is_frontmost:
                    self._logger.log(
                        tool="runner",
                        action="focus_lost",
                        status="ok",
                        details={"mode": "approach_and_enter_cover", "obstructed_by": list(cap.window.obstructed_by)},
                    )
                    stop_reason = "focus_lost"
                    break
                if cfg.stop_on_banner and hud.banner is not None:
                    stop_reason = f"banner:{hud.banner}"
                    break
                if _is_cover_confirmed(mm, posture):
                    stop_reason = "cover_entered"
                    break

                if (self._clock() - start_t) >= cfg.max_duration_s - 0.35:
                    break

                rem_t = max(0.05, cfg.max_duration_s - (self._clock() - start_t) - 0.3)
                cur_target, _ = self._resolve_route_target_and_bearing(mm, target_color)
                snapped_cover = (
                    (self._in_cover_hint is True)
                    or (posture.state == "in_cover" and posture.confidence >= 0.60)
                )
                if snapped_cover and cur_target is not None and cur_target.distance > 0.08:
                    # Snapped to a wrong prop (e.g. side pillar) while objective remains > 0.08 away: exit & advance.
                    self._logger.log(
                        tool="runner",
                        action="wrong_prop_cover",
                        status="ok",
                        details={"distance": round(cur_target.distance, 4), "bearing": round(cur_target.bearing, 3)},
                    )
                    self.exit_cover(force=True, settle_s=0.15)
                    cover_slide_idx = 0
                    adv_budget = max(0.5, min(6.0, cfg.max_duration_s - (self._clock() - start_t) - 0.3))
                    adv_res = self.follow_route_on_foot(
                        sprint=False,
                        target_color=target_color,
                        arrive_distance=min(arrive_distance, 0.075),
                        stop_on_subtitle_keywords=("return to", "wait for", "stay with", "follow"),
                        vault_on_stuck=False,
                        cover_on_health_drop=True,
                        level_pitch=False,
                        config=LoopConfig(
                            max_duration_s=adv_budget,
                            hz=cfg.hz,
                            hud_check_interval_s=cfg.hud_check_interval_s,
                            health_drop_threshold=cfg.health_drop_threshold,
                            stop_on_banner=cfg.stop_on_banner,
                            stop_on_focus_loss=cfg.stop_on_focus_loss,
                        ),
                    )
                    ticks += adv_res.ticks
                    if adv_res.stop_reason in ("focus_lost",) or adv_res.stop_reason.startswith("banner:"):
                        last_mm = adv_res.last_minimap
                        last_hud = adv_res.last_hud
                        stop_reason = adv_res.stop_reason
                        break
                    continue

                if self._in_cover_hint is True and cover_slide_idx < 2:
                    # Already latched in cover: slide laterally along cover without untoggling `q`.
                    slide_key = "a" if cover_slide_idx == 0 else "d"
                    cover_slide_idx += 1
                    self.input.key_down(slide_key)
                    try:
                        self._sleep_chunked(min(probe_step_s * 1.2, rem_t))
                    finally:
                        self.input.key_up(slide_key)
                    continue

                if self._in_cover_hint is True:
                    # Two cover slides didn't clear objective; explicitly exit cover and reposition toward blip.
                    self.exit_cover(force=True, settle_s=0.15)
                    cover_slide_idx = 0

                if probe_idx < 2:
                    _, bearing = self._resolve_route_target_and_bearing(mm, target_color)
                    if bearing is not None and abs(bearing) > 0.15:
                        dx = int(round(max(-220.0, min(220.0, bearing * 180.0))))
                        if dx != 0:
                            self._steer_camera(dx, 0)
                    if probe_idx == 0:
                        step_key, step_dur = "w", probe_step_s
                    else:
                        step_key = "a" if (bearing is not None and bearing < 0.0) else "d"
                        step_dur = probe_step_s
                    probe_idx += 1
                    self.input.key_down(step_key)
                    try:
                        self._sleep_chunked(min(step_dur, rem_t))
                    finally:
                        self.input.key_up(step_key)
                else:
                    # Local face blocked (e.g. prop leaning on crate): commit 2 orbit steps per side before flipping.
                    if orbit_pref_dir is None:
                        _, b_now = self._resolve_route_target_and_bearing(mm, target_color)
                        orbit_pref_dir = "a" if (b_now is not None and b_now < 0.0) else "d"
                    alt_dir = "d" if orbit_pref_dir == "a" else "a"
                    orbit_dir = orbit_pref_dir if (((probe_idx - 2) // 2) % 2 == 0) else alt_dir
                    probe_idx += 1
                    for o_key, o_dur in (("s", 0.30), (orbit_dir, 0.55)):
                        rem_now = max(0.0, cfg.max_duration_s - (self._clock() - start_t) - 0.25)
                        if rem_now <= 0.05:
                            break
                        self.input.key_down(o_key)
                        try:
                            self._sleep_chunked(min(o_dur, rem_now))
                        finally:
                            self.input.key_up(o_key)
                    cap_mid = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
                    mm_mid = inspect_minimap(cap_mid.image, logger=self._logger)
                    _, bearing_mid = self._resolve_route_target_and_bearing(mm_mid, target_color)
                    if bearing_mid is not None and abs(bearing_mid) > 0.10:
                        dx_mid = int(round(max(-240.0, min(240.0, bearing_mid * 180.0))))
                        if dx_mid != 0:
                            self._steer_camera(dx_mid, 0)
                    rem_fwd = max(0.0, cfg.max_duration_s - (self._clock() - start_t) - 0.25)
                    if rem_fwd > 0.05:
                        self.input.key_down("w")
                        try:
                            self._sleep_chunked(min(0.75, rem_fwd))
                        finally:
                            self.input.key_up("w")
        finally:
            self.input.release_all(reason=f"runner:approach_and_enter_cover:{stop_reason}", suppress_errors=True)

        total_elapsed = max(0.0, self._clock() - start_t)
        self._logger.log(
            tool="runner",
            action="approach_and_enter_cover",
            status="ok",
            details={"stop_reason": stop_reason, "ticks": ticks, "elapsed_s": round(total_elapsed, 3)},
        )
        return LoopResult(
            stop_reason=stop_reason,
            ticks=ticks,
            elapsed_s=total_elapsed,
            banner=last_hud.banner if last_hud else None,
            last_hud=last_hud,
            last_minimap=last_mm,
            actions_sent=ticks,
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
        recover_wait_s: float = 4.0,
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
        cover_on_low_health: bool = True,
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
        engage_start_t = 0.0
        flank_unblock_until_t = 0.0
        recover_cooldown_until_t = 0.0
        tether_active_until_t = 0.0
        recover_start_health: Optional[float] = None
        retreat_start_health: Optional[float] = None
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
        target_idx = 0
        no_lock_ticks = 0
        engage_red_hits = 0
        blocked_flank_streak = self._last_flank_streak
        visual_suppressed_until_t = 0.0
        cam_fb = _CameraFeedback(enabled=self._enable_camera_motion_tracking)

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
                if "right" in self.input.held_buttons:
                    self._sleep(0.02)
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
                cam_fb.reset()
            elif not want_aim and is_aiming:
                _recenter_sweep()
                self.input.mouse_up("right")
                cam_fb.reset()

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal phase, phase_end_t, min_move_until_t, engage_cooldown_until_t, engage_start_t, flank_unblock_until_t, recover_cooldown_until_t, tether_active_until_t, recover_start_health, retreat_start_health, special_triggered, prev_gray, last_motion_check_t, low_motion_elapsed_s, last_bearing, recovery_count, sweep_idx, sweep_offset_x, sweep_offset_y, aligned_ticks, target_idx, no_lock_ticks, engage_red_hits, blocked_flank_streak, visual_suppressed_until_t
            now_t = state.elapsed_s
            tick_clock_0 = self._clock()

            def _elapsed_now() -> float:
                return now_t + max(0.0, self._clock() - tick_clock_0)

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
            nearest_enemy = mm.nearest_enemy
            point_blank_enemy = nearest_enemy is not None and nearest_enemy.distance < 0.18
            enemy = nearest_enemy if point_blank_enemy else self._select_cycle_target(mm.targets["red"], target_idx)

            def _resume_move_from_recover(*, tether: bool = False, cooldown_s: float = 0.0) -> None:
                nonlocal phase, phase_end_t, min_move_until_t, engage_cooldown_until_t
                nonlocal recover_cooldown_until_t, recover_start_health, retreat_start_health, tether_active_until_t
                phase = "move"
                if tether:
                    tether_active_until_t = max(tether_active_until_t, now_t + 2.5)
                    retreat_start_health = mm.health_ratio
                else:
                    retreat_start_health = None
                phase_end_t = now_t + move_phase_s
                min_move_until_t = now_t + move_phase_s
                engage_cooldown_until_t = now_t + move_phase_s
                if cooldown_s > 0.0:
                    recover_cooldown_until_t = now_t + cooldown_s
                recover_start_health = None
                _reset_motion_tracker()

            def _enter_engage_phase(*, cooldown_s: float = 0.0) -> None:
                nonlocal phase, phase_end_t, engage_start_t, aligned_ticks, no_lock_ticks, engage_red_hits, sweep_idx, sweep_offset_x, sweep_offset_y
                nonlocal recover_cooldown_until_t, recover_start_health
                _set_movement(False)
                _set_aim(True)
                cur_t = _elapsed_now()
                phase = "engage"
                engage_start_t = cur_t
                phase_end_t = cur_t + engage_phase_s
                aligned_ticks = 0
                no_lock_ticks = 0
                engage_red_hits = 0
                sweep_idx = 0
                sweep_offset_x = 0
                sweep_offset_y = 0
                if cooldown_s > 0.0:
                    recover_cooldown_until_t = cur_t + cooldown_s
                recover_start_health = None
                _reset_motion_tracker()
                cam_fb.reset()

            def _abort_low_health(*, tap_cover: bool) -> str:
                _set_aim(False)
                _set_movement(False)
                if tap_cover:
                    self._tap_key_preserving_held("q", hold_s=0.08)
                    self._in_cover_hint = True
                self._logger.log(
                    tool="runner",
                    action="advance_exposed",
                    status="ok",
                    details={"health_ratio": round(mm.health_ratio, 3), "tap_cover": bool(tap_cover)},
                )
                return "low_health"

            can_retreat_to_ally = (
                (not point_blank_enemy)
                and (now_t >= tether_active_until_t)
                and (blue_ally is not None and blue_ally.distance > 0.20)
            )

            # Low-health cover/recover phase (critical < 0.25 HP bypasses recover cooldown)
            mid_retreat = now_t < tether_active_until_t
            retreat_drop = (
                mid_retreat
                and retreat_start_health is not None
                and (retreat_start_health - mm.health_ratio) >= exposed_recover_drop_threshold
            )
            if mid_retreat and mm.minimap_visible and mm.health_ratio < 0.25 and (point_blank_enemy or retreat_drop):
                return _abort_low_health(tap_cover=cover_on_low_health)
            if (
                phase != "recover"
                and (not mid_retreat)
                and mm.minimap_visible
                and (
                    mm.health_ratio < 0.25
                    or (now_t >= recover_cooldown_until_t and mm.health_ratio < recover_health)
                )
            ):
                _set_aim(False)
                _set_movement(False)
                if mm.health_ratio < 0.25:
                    if can_retreat_to_ally and (enemy is None or enemy.distance >= 0.35) and now_t >= recover_cooldown_until_t:
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
                        return _abort_low_health(tap_cover=cover_on_low_health)
                elif point_blank_enemy and enemy.distance <= engage_distance:
                    _enter_engage_phase(cooldown_s=1.5)
                elif (
                    (not point_blank_enemy)
                    and blue_ally is not None
                    and blue_ally.distance > 0.14
                    and enemy is not None
                    and enemy.distance < 0.45
                    and abs(blue_ally.bearing) > 0.35
                ):
                    _resume_move_from_recover(tether=True, cooldown_s=1.5)
                    self._logger.log(
                        tool="runner",
                        action="advance_retreat",
                        status="ok",
                        details={
                            "health_ratio": round(mm.health_ratio, 3),
                            "ally_distance": round(blue_ally.distance, 3),
                            "enemy_distance": round(enemy.distance, 3),
                        },
                    )
                else:
                    phase = "recover"
                    phase_end_t = now_t + recover_wait_s
                    recover_cooldown_until_t = phase_end_t + 3.0
                    recover_start_health = mm.health_ratio
                    _reset_motion_tracker()
                    self._tap_key_preserving_held("q", hold_s=0.08)
                    self._in_cover_hint = True
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
                    if critical_hp:
                        if can_retreat_to_ally and (enemy is None or enemy.distance >= 0.35):
                            _resume_move_from_recover(tether=True, cooldown_s=1.5)
                            self._logger.log(
                                tool="runner",
                                action="advance_retreat",
                                status="ok",
                                details={"health_ratio": round(mm.health_ratio, 3), "ally_distance": round(blue_ally.distance, 3)},
                            )
                        else:
                            return _abort_low_health(tap_cover=False)
                    elif (not point_blank_enemy) and blue_ally is not None and blue_ally.distance > 0.14:
                        _resume_move_from_recover(tether=True, cooldown_s=1.5)
                        self._logger.log(
                            tool="runner",
                            action="advance_retreat",
                            status="ok",
                            details={"health_ratio": round(mm.health_ratio, 3), "ally_distance": round(blue_ally.distance, 3)},
                        )
                    elif enemy is not None and enemy.distance < 0.45:
                        recovery_count += 1
                        self._perform_on_foot_backpedal(attempt=recovery_count, duration_s=0.4)
                        if enemy.distance <= engage_distance:
                            _enter_engage_phase(cooldown_s=2.0)
                        else:
                            _resume_move_from_recover(cooldown_s=1.5)
                if phase == "recover":
                    actively_healing = (
                        mm.minimap_visible
                        and mm.health_ratio < recover_target_health
                        and recover_start_health is not None
                        and mm.health_ratio > recover_start_health
                    )
                    max_heal_until_t = phase_end_t + 1.5 if actively_healing else phase_end_t
                    if now_t < max_heal_until_t and (not mm.minimap_visible or mm.health_ratio < recover_target_health):
                        return None
                    _resume_move_from_recover()

            need_ally_tether = (
                (not point_blank_enemy)
                and (
                    (now_t < tether_active_until_t)
                    or (blue_ally is not None and blue_ally.distance > ally_tether_distance)
                )
            )
            if need_ally_tether and blue_ally is not None:
                nav_bearing = blue_ally.bearing
            elif route_bearing is not None:
                nav_bearing = route_bearing
            elif blue_ally is not None and blue_ally.distance > 0.14 and abs(blue_ally.bearing) <= 0.45:
                nav_bearing = blue_ally.bearing
            elif enemy is not None and enemy.distance > 0.22:
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

            max_engage_hold_s = max(engage_phase_s * 2.0, 6.5)
            if phase == "move" and can_engage and (
                (point_blank_enemy and now_t >= flank_unblock_until_t)
                or (
                    now_t >= engage_cooldown_until_t
                    and (hold_engage_threat or now_t >= min_move_until_t)
                )
            ):
                _enter_engage_phase()
            elif phase == "engage" and (
                enemy is None or enemy.distance > engage_distance or need_ally_tether or now_t >= phase_end_t
            ):
                if (
                    phase == "engage"
                    and hold_engage_threat
                    and engage_red_hits > 0
                    and now_t >= phase_end_t
                    and (now_t - engage_start_t) < max_engage_hold_s
                ):
                    phase_end_t = min(now_t + 1.5, engage_start_t + max_engage_hold_s)
                else:
                    _set_aim(False)
                    if (
                        now_t >= phase_end_t
                        and (not need_ally_tether)
                        and route_bearing is None
                        and enemy is not None
                        and enemy.distance <= 0.32
                        and (len(mm.targets["red"]) >= 2 or engage_red_hits == 0)
                    ):
                        target_idx += 1
                        no_lock_ticks = 0
                        if engage_red_hits > 0:
                            blocked_flank_streak = 0
                            self._last_flank_streak = 0
                            self._last_flank_base_side = "d"
                            recovery_count += 1
                            self._perform_on_foot_sidestep(recovery_count, enemy.bearing, 0.35, False)
                            _enter_engage_phase()
                            return None
                        # Zero red_reticle locks across engage window: pick away-from-enemy side initially and flip polarity every 2 streaks.
                        blocked_flank_streak += 1
                        self._last_flank_streak = blocked_flank_streak
                        if blocked_flank_streak == 1:
                            self._last_flank_base_side = "a" if enemy.bearing > 0.20 else "d"
                        flip_side = ((blocked_flank_streak - 1) // 2) % 2 == 1
                        opp_side = "a" if self._last_flank_base_side == "d" else "d"
                        flank_key = opp_side if flip_side else self._last_flank_base_side
                        self._logger.log(
                            tool="runner",
                            action="advance_flank_unblock",
                            status="ok",
                            details={"streak": blocked_flank_streak, "flank_key": flank_key, "enemy_distance": round(enemy.distance, 3)},
                        )
                        self.input.key_down("s")
                        try:
                            self._sleep_chunked(0.20)
                        finally:
                            self.input.key_up("s")
                        self.input.key_down(flank_key)
                        try:
                            self._sleep_chunked(0.85)
                        finally:
                            self.input.key_up(flank_key)
                        if abs(enemy.bearing) > 0.30:
                            self._steer_camera(-45 if flank_key == "a" else 45, 0)
                        post_flank_t = _elapsed_now()
                        flank_move_s = min(move_phase_s, 0.85)
                        flank_unblock_until_t = post_flank_t + flank_move_s
                        phase = "move"
                        phase_end_t = flank_unblock_until_t
                        min_move_until_t = flank_unblock_until_t
                        engage_cooldown_until_t = flank_unblock_until_t
                        _reset_motion_tracker()
                        cam_fb.reset()
                        return None
                    phase = "move"
                    phase_end_t = now_t + move_phase_s
                    min_move_until_t = now_t + move_phase_s
                    engage_cooldown_until_t = now_t + move_phase_s
                    _reset_motion_tracker()
                    cam_fb.reset()

            if phase == "engage":
                _set_movement(False)
                _set_aim(True)
                cam_fb.observe(
                    state.capture.image,
                    mode="advance_under_fire",
                    logger=self._logger,
                    input_ctrl=self.input,
                    aim_button="right",
                )
                reticle = detect_red_reticle(state.capture.image)
                if reticle.is_red:
                    aligned_ticks = 0
                    no_lock_ticks = 0
                    engage_red_hits += 1
                    blocked_flank_streak = 0
                    self._last_flank_streak = 0
                    self._last_flank_base_side = "d"
                    visual_suppressed_until_t = 0.0
                    self._fire_bursts("left", bursts=3, hold_s=0.03, gap_s=0.025, reason="red_reticle")
                    cam_fb.fired_last_tick = True
                    if recoil_comp_dy > 0:
                        self.input.mouse_delta(0, int(recoil_comp_dy) * 2, steps=1, duration_s=0.01)
                        cam_fb.last_cmd_dy = int(recoil_comp_dy) * 2
                    return None

                if reticle.is_visible and now_t >= visual_suppressed_until_t and (enemy is None or abs(enemy.bearing) <= 0.45):
                    visual = self._detect_visual_target(state.capture.image, mm, enemy)
                else:
                    visual = None

                if visual is not None:
                    cam_fb.visual_steer_count += 1
                    if cam_fb.visual_steer_count == 1 or cam_fb.visual_steer_count % 5 == 0:
                        self._logger.log(
                            tool="runner",
                            action="visual_target_steer",
                            status="ok",
                            details={
                                "mode": "advance_under_fire",
                                "aim_point": [round(visual.target.aim_point[0], 3), round(visual.target.aim_point[1], 3)],
                                "dx": visual.dx,
                                "dy": visual.dy,
                                "source": visual.target.source,
                                "confidence": round(visual.target.confidence, 3),
                            },
                        )
                    if visual.dx != 0 or visual.dy != 0:
                        self._steer_camera(visual.dx, visual.dy)
                        cam_fb.last_cmd_dx = visual.dx
                        cam_fb.last_cmd_dy = visual.dy
                    no_lock_ticks += 1
                    if no_lock_ticks >= 16:
                        no_lock_ticks = 0
                        aligned_ticks = 0
                        target_idx += 1
                        _recenter_sweep()
                        return None
                    if abs(visual.error_x) <= 0.045 and abs(visual.error_y) <= 0.045:
                        aligned_ticks += 1
                        if aligned_ticks >= 5:
                            # Visual box stayed aligned across 2 suppressive bursts with 0 red_reticle (ally/occluder): yield to sweep.
                            visual_suppressed_until_t = now_t + 1.8
                            aligned_ticks = 0
                            no_lock_ticks = 0
                            target_idx += 1
                            _recenter_sweep()
                            return None
                        self._handle_visual_aligned_tick(
                            fire_button="left",
                            aligned_ticks=aligned_ticks,
                            recoil_comp_dy=recoil_comp_dy,
                            cam_fb=cam_fb,
                        )
                    else:
                        aligned_ticks = 0
                    return None

                if enemy is not None:
                    if abs(enemy.bearing) > bearing_deadband:
                        dx = self._ads_steer_dx(enemy.bearing, steer_gain=engage_steer_gain)
                        if dx != 0:
                            self._steer_camera(dx, 0)
                            cam_fb.last_cmd_dx = dx
                    no_lock_ticks += 1
                    if no_lock_ticks >= 16:
                        no_lock_ticks = 0
                        target_idx += 1
                        _recenter_sweep()
                        return None
                    if abs(enemy.bearing) <= engage_align_bearing:
                        aligned_ticks += 1
                        if engage_red_hits == 0:
                            # Blind probe without Vision or red_reticle hits: conserve ammo with a single 1-burst probe at tick 8.
                            if no_lock_ticks == 8:
                                self._fire_bursts("left", bursts=1, hold_s=0.03, gap_s=0.03, reason="suppressive")
                                cam_fb.fired_last_tick = True
                                if recoil_comp_dy > 0:
                                    comp_dy = int(recoil_comp_dy)
                                    self.input.mouse_delta(0, comp_dy, steps=1, duration_s=0.01)
                                    cam_fb.last_cmd_dy = comp_dy
                        elif no_lock_ticks in (6, 12):
                            self._fire_bursts("left", bursts=2, hold_s=0.03, gap_s=0.03, reason="suppressive")
                            cam_fb.fired_last_tick = True
                            if recoil_comp_dy > 0:
                                sup_scale = 1 if no_lock_ticks == 6 else 2
                                comp_dy = int(recoil_comp_dy) * sup_scale
                                self.input.mouse_delta(0, comp_dy, steps=1, duration_s=0.01)
                                cam_fb.last_cmd_dy = comp_dy
                    else:
                        aligned_ticks = 0

                if scan_pattern:
                    dx, dy = scan_pattern[sweep_idx % len(scan_pattern)]
                    sweep_idx += 1
                    if dx != 0 or dy != 0:
                        self.input.mouse_delta(dx, dy, steps=2, duration_s=0.03)
                        sweep_offset_x += dx
                        sweep_offset_y += dy
                        cam_fb.last_cmd_dx += dx
                        cam_fb.last_cmd_dy += dy
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

        result = self._run_loop(
            mode_name="advance_under_fire",
            config=cfg,
            on_tick=_on_tick,
            on_before_release=_recenter_sweep,
        )
        if cover_on_timeout and result.stop_reason == "timeout" and phase != "recover":
            self._tap_key_preserving_held("q", hold_s=0.08)
            self._in_cover_hint = True
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

    def enter_nearest_vehicle(
        self,
        *,
        target_color: str = "blue",
        arrive_distance: float = 0.06,
        approach_duration_s: float = 12.0,
        enter_key: str = "f",
        settle_s: float = 3.8,
        enter_prompt_keywords: Sequence[str] = ("get in", "enter", "vehicle", "car"),
        post_drive_keywords: Sequence[str] = ("get to", "drive", "lose", "escape", "follow", "cops", "police"),
        config: Optional[LoopConfig] = None,
    ) -> LoopResult:
        """Approach nearest vehicle blip without overshooting, tap `enter_key`, and verify boarding before driving."""
        start_t = self._clock()
        cfg = config or LoopConfig(max_duration_s=approach_duration_s + settle_s + 2.0, hz=10.0)
        enter_kws = tuple(k.strip().lower() for k in enter_prompt_keywords if k.strip())
        drive_kws = tuple(k.strip().lower() for k in post_drive_keywords if k.strip())
        ticks = 0
        actions_sent = 0
        last_hud: Optional[HudObservation] = None
        last_mm: Optional[MinimapObservation] = None
        stop_reason = "vehicle_entry_unconfirmed"
        banner: Optional[str] = None

        try:
            cap0 = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            mm0 = inspect_minimap(cap0.image, logger=self._logger)
            hud0 = inspect_frame(cap0.image, ocr_backend=self._ocr_backend, logger=self._logger)
            last_mm, last_hud = mm0, hud0
            ticks += 1

            if cfg.stop_on_focus_loss and not cap0.window.is_frontmost:
                return LoopResult("focus_lost", ticks, self._clock() - start_t, None, last_hud, last_mm, actions_sent)
            if cfg.stop_on_banner and hud0.banner is not None:
                return LoopResult(f"banner:{hud0.banner}", ticks, self._clock() - start_t, hud0.banner, last_hud, last_mm, actions_sent)

            t0, _ = self._resolve_route_target_and_bearing(mm0, target_color)
            if t0 is not None and t0.distance > arrive_distance:
                app_res = self.follow_route_on_foot(
                    sprint=False,
                    target_color=target_color,
                    arrive_distance=arrive_distance,
                    stop_on_subtitle_keywords=enter_kws,
                    enable_stuck_recovery=True,
                    vault_on_stuck=False,
                    cover_on_health_drop=False,
                    level_pitch=False,
                    config=LoopConfig(
                        max_duration_s=min(approach_duration_s, max(1.0, cfg.max_duration_s - settle_s - 0.5)),
                        hz=cfg.hz,
                        hud_check_interval_s=cfg.hud_check_interval_s,
                        stop_on_banner=cfg.stop_on_banner,
                        stop_on_focus_loss=cfg.stop_on_focus_loss,
                        health_drop_threshold=None,
                    ),
                )
                ticks += app_res.ticks
                actions_sent += app_res.actions_sent
                if app_res.last_minimap is not None:
                    last_mm = app_res.last_minimap
                if app_res.last_hud is not None:
                    last_hud = app_res.last_hud
                if app_res.stop_reason.startswith("banner:") or app_res.stop_reason == "focus_lost":
                    return LoopResult(app_res.stop_reason, ticks, self._clock() - start_t, app_res.banner, last_hud, last_mm, actions_sent)

            pre_cap = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
            pre_mm = inspect_minimap(pre_cap.image, logger=self._logger)
            pre_hud = inspect_frame(pre_cap.image, ocr_backend=self._ocr_backend, logger=self._logger)
            last_mm, last_hud = pre_mm, pre_hud
            pre_gray = _road_gray_sample(pre_cap.image)
            pre_text = f"{pre_hud.subtitle_or_objective} {pre_hud.help_prompt}".lower()
            pre_had_enter = any(k in pre_text for k in enter_kws)
            pre_target, _ = self._resolve_route_target_and_bearing(pre_mm, target_color)
            pre_yellow_route = pre_mm.yellow_route_bearing

            self._tap_key_preserving_held(enter_key, hold_s=0.08)
            actions_sent += 1

            poll_start = self._clock()
            max_settle_t = max(0.25, min(settle_s, cfg.max_duration_s - (poll_start - start_t)))
            while (self._clock() - poll_start) < max_settle_t:
                self._sleep(min(0.25, max(0.05, max_settle_t - (self._clock() - poll_start))))
                cur_cap = capture_game_window(backend=self._window_backend, logger=self._logger, require_16_9=True)
                cur_mm = inspect_minimap(cur_cap.image, logger=self._logger)
                cur_hud = inspect_frame(cur_cap.image, ocr_backend=self._ocr_backend, logger=self._logger)
                last_mm, last_hud = cur_mm, cur_hud
                ticks += 1

                if cfg.stop_on_focus_loss and not cur_cap.window.is_frontmost:
                    stop_reason = "focus_lost"
                    break
                if cfg.stop_on_banner and cur_hud.banner is not None:
                    banner = cur_hud.banner
                    stop_reason = f"banner:{cur_hud.banner}"
                    break

                cur_text = f"{cur_hud.subtitle_or_objective} {cur_hud.help_prompt}".lower()
                cur_has_enter = any(k in cur_text for k in enter_kws)
                cur_has_drive = any(k in cur_text for k in drive_kws) and not cur_has_enter
                cur_target, _ = self._resolve_route_target_and_bearing(cur_mm, target_color)
                shift_score = _road_motion_score(pre_gray, _road_gray_sample(cur_cap.image))

                near_for_shift = pre_target is None or pre_target.distance <= max(arrive_distance * 2.5, 0.16)
                close_for_drive = pre_target is None or pre_target.distance <= 0.14
                if (
                    (pre_had_enter and not cur_has_enter)
                    or (cur_has_drive and close_for_drive)
                    or (pre_yellow_route is None and cur_mm.yellow_route_bearing is not None and close_for_drive)
                    or (pre_target is not None and cur_target is None and (self._clock() - poll_start) >= 0.75)
                    or (shift_score >= 12.0 and near_for_shift and not cur_has_enter and (self._clock() - poll_start) >= 0.75)
                ):
                    stop_reason = "entered_vehicle"
                    break

            self._logger.log(
                tool="runner",
                action="enter_nearest_vehicle",
                status="ok" if stop_reason == "entered_vehicle" else "warn",
                details={
                    "stop_reason": stop_reason,
                    "target_color": target_color,
                    "pre_dist": round(pre_target.distance, 3) if pre_target is not None else None,
                },
            )
            return LoopResult(stop_reason, ticks, self._clock() - start_t, banner, last_hud, last_mm, actions_sent)
        finally:
            self.input.release_all()

    def follow_gps_drive(
        self,
        route_color: str = "yellow",
        throttle_key: str = "w",
        brake_key: str = "s",
        steer_left_key: str = "a",
        steer_right_key: str = "d",
        arrive_distance: float = 0.07,
        steer_deadband: float = 0.08,
        mild_steer_threshold: float = 0.24,
        mild_steer_pulse_s: float = 0.045,
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
        high_bearing_elapsed_s = 0.0
        prev_best_abs_bearing: Optional[float] = None
        min_target_dist: Optional[float] = None
        target_stall_elapsed_s = 0.0
        sharp_brake_until_t = 0.0
        sharp_brake_cooldown_until_t = 0.0
        last_bearing = 0.0
        recovery_count = 0

        def _set_key(key: str, want_down: bool) -> None:
            is_down = key in self.input.held_keys
            if want_down and not is_down:
                self.input.key_down(key)
            elif not want_down and is_down:
                self.input.key_up(key)

        def _perform_stuck_recovery(reason: str = "low_motion") -> None:
            nonlocal recovery_count, low_motion_elapsed_s, high_bearing_elapsed_s, prev_best_abs_bearing, target_stall_elapsed_s, min_target_dist, prev_gray, last_motion_check_t, sharp_brake_until_t, sharp_brake_cooldown_until_t
            recovery_count += 1
            primary_counter = steer_left_key if last_bearing >= 0 else steer_right_key
            alt_counter = steer_right_key if primary_counter == steer_left_key else steer_left_key
            counter_steer = primary_counter if ((recovery_count - 1) // 2) % 2 == 0 else alt_counter
            _set_key(throttle_key, False)
            _set_key(steer_left_key, False)
            _set_key(steer_right_key, False)
            _set_key(brake_key, True)
            _set_key(counter_steer, True)
            self._logger.log(
                tool="runner",
                action="stuck_recovery",
                status="ok",
                details={
                    "attempt": recovery_count,
                    "counter_steer": counter_steer,
                    "duration_s": reverse_duration_s,
                    "reason": reason,
                },
            )
            try:
                self._sleep_chunked(reverse_duration_s)
            finally:
                _set_key(counter_steer, False)
                _set_key(brake_key, False)
            fwd_steer_s = min(0.45, max(0.0, reverse_duration_s * 0.6))
            if fwd_steer_s > 0.0:
                _set_key(throttle_key, True)
                _set_key(alt_counter, True)
                try:
                    self._sleep_chunked(fwd_steer_s)
                finally:
                    _set_key(alt_counter, False)
            low_motion_elapsed_s = 0.0
            high_bearing_elapsed_s = 0.0
            prev_best_abs_bearing = None
            target_stall_elapsed_s = 0.0
            min_target_dist = None
            prev_gray = None
            last_motion_check_t = None
            sharp_brake_until_t = 0.0
            sharp_brake_cooldown_until_t = 0.0

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal prev_gray, last_motion_check_t, low_motion_elapsed_s, high_bearing_elapsed_s, prev_best_abs_bearing, min_target_dist, target_stall_elapsed_s, sharp_brake_until_t, sharp_brake_cooldown_until_t, last_bearing
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
            if enable_stuck_recovery:
                cur_gray = _road_gray_sample(state.capture.image)
                dt = max(0.0, state.elapsed_s - last_motion_check_t) if last_motion_check_t is not None else 0.0
                if prev_gray is not None and last_motion_check_t is not None:
                    motion = _road_motion_score(prev_gray, cur_gray)
                    if motion < stuck_motion_threshold:
                        low_motion_elapsed_s += dt
                    else:
                        low_motion_elapsed_s = 0.0
                prev_gray = cur_gray
                last_motion_check_t = state.elapsed_s

                if bearing is not None and abs(bearing) >= 0.58:
                    abs_b = abs(bearing)
                    if prev_best_abs_bearing is None:
                        prev_best_abs_bearing = abs_b
                        high_bearing_elapsed_s += dt
                    elif abs_b < prev_best_abs_bearing - 0.08:
                        prev_best_abs_bearing = abs_b
                        high_bearing_elapsed_s = 0.0
                    else:
                        high_bearing_elapsed_s += dt
                else:
                    high_bearing_elapsed_s = 0.0
                    prev_best_abs_bearing = None

                if len(targets) == 1 and target is not None and target.distance > max(arrive_distance * 2.0, 0.18):
                    if min_target_dist is None or target.distance < (min_target_dist - 0.015):
                        min_target_dist = target.distance
                        target_stall_elapsed_s = 0.0
                    else:
                        target_stall_elapsed_s += dt
                else:
                    min_target_dist = None
                    target_stall_elapsed_s = 0.0

                stall_reason: Optional[str] = None
                if low_motion_elapsed_s >= stuck_window_s:
                    stall_reason = "low_motion"
                elif high_bearing_elapsed_s >= max(stuck_window_s * 2.0, 3.0):
                    stall_reason = "high_bearing_stall"
                elif target_stall_elapsed_s >= max(stuck_window_s * 4.5, 4.0):
                    stall_reason = "target_dist_stall"

                if stall_reason is not None and (cfg.max_duration_s - state.elapsed_s) > 0.25:
                    _perform_stuck_recovery(reason=stall_reason)
                    _set_key(throttle_key, True)
                    return None
            else:
                low_motion_elapsed_s = 0.0
                high_bearing_elapsed_s = 0.0
                prev_best_abs_bearing = None
                target_stall_elapsed_s = 0.0
                min_target_dist = None
                prev_gray = None
                last_motion_check_t = state.elapsed_s

            if bearing is None:
                _set_key(steer_left_key, False)
                _set_key(steer_right_key, False)
                _set_key(brake_key, False)
                _set_key(throttle_key, True)
                return None

            if sharp and state.elapsed_s >= sharp_brake_cooldown_until_t:
                sharp_brake_until_t = state.elapsed_s + 0.25
                sharp_brake_cooldown_until_t = state.elapsed_s + 1.6
            braking = sharp and (state.elapsed_s < sharp_brake_until_t)
            _set_key(throttle_key, not braking)
            _set_key(brake_key, braking)
            if abs(bearing) < steer_deadband:
                _set_key(steer_left_key, False)
                _set_key(steer_right_key, False)
            elif abs(bearing) < mild_steer_threshold:
                steer_key = steer_left_key if bearing < 0 else steer_right_key
                other_key = steer_right_key if bearing < 0 else steer_left_key
                _set_key(other_key, False)
                _set_key(steer_key, False)
                self._tap_key_preserving_held(steer_key, hold_s=mild_steer_pulse_s)
            else:
                _set_key(steer_left_key, bearing < 0)
                _set_key(steer_right_key, bearing > 0)
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
        recoil_comp_dy: int = 1,
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
                    fire_bursts = max(1, bursts)
                    self._fire_bursts(fire_button, bursts=bursts, hold_s=0.03, gap_s=0.03)
                    if recoil_comp_dy > 0:
                        self.input.mouse_delta(0, int(recoil_comp_dy) * max(1, fire_bursts - 1), steps=1, duration_s=0.01)
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
                    fire_bursts = max(1, bursts)
                    self._fire_bursts(fire_button, bursts=fire_bursts, hold_s=0.03, gap_s=0.03)
                    if recoil_comp_dy > 0:
                        self.input.mouse_delta(0, int(recoil_comp_dy) * max(1, fire_bursts - 1), steps=1, duration_s=0.01)
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
        peek_burst_s: float = 4.5,
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
        exposed_duck_drop_threshold: float = 0.05,
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
            self.enter_cover(force=False)
        elif self._in_cover_hint is None:
            self._in_cover_hint = True

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
        target_idx = 0
        no_lock_ticks = 0
        steer_cadence_ticks = 0
        large_steer_ticks = 0
        clamped_ticks = 0
        unclamp_attempts = 0
        flank_stage2_done = False
        toggled_cover_for_flank = False
        pre_flank_bearing: Optional[float] = None
        last_clamped_bearing: Optional[float] = None
        no_reticle_ticks = 0
        edge_shift_attempts = 0
        suppressive_no_red_count = 0
        visual_suppressed_until_t = 0.0
        cam_fb = _CameraFeedback(enabled=self._enable_camera_motion_tracking)

        def _recenter_sweep() -> None:
            nonlocal sweep_idx, sweep_offset_x, sweep_offset_y
            if sweep_offset_x != 0 or sweep_offset_y != 0:
                self.input.mouse_delta(-sweep_offset_x, -sweep_offset_y, steps=2, duration_s=0.03)
                sweep_offset_x = 0
                sweep_offset_y = 0
                if aim_button in self.input.held_buttons:
                    self._sleep(0.02)
            sweep_idx = 0

        def _set_peek(want_peek: bool, now_t: float) -> None:
            nonlocal is_peeking, peek_start_t, no_reticle_ticks
            if want_peek and not is_peeking:
                self.input.mouse_down(aim_button)
                is_peeking = True
                peek_start_t = now_t
                if aim_settle_s > 0:
                    self._sleep(aim_settle_s)
                cam_fb.reset()
            elif not want_peek and is_peeking:
                _recenter_sweep()
                self.input.mouse_up(aim_button)
                is_peeking = False
                no_reticle_ticks = 0
                cam_fb.reset()

        def _handle_blocked_cover(now_t: float, enemy: RadarTarget, block_reason: str) -> Optional[str]:
            nonlocal edge_shift_attempts, no_reticle_ticks, no_lock_ticks, aligned_ticks, suppressive_no_red_count, toggled_cover_for_flank
            if edge_shift_attempts >= 2:
                self._logger.log(
                    tool="runner",
                    action="cover_blocked",
                    status="ok",
                    details={
                        "edge_shift_attempts": edge_shift_attempts,
                        "bearing": round(enemy.bearing, 3),
                        "reason": block_reason,
                    },
                )
                _set_peek(False, now_t)
                if toggled_cover_for_flank:
                    self._tap_key_preserving_held("q", hold_s=0.08)
                    toggled_cover_for_flank = False
                return "cover_blocked"
            edge_shift_attempts += 1
            preferred_key = "a" if enemy.bearing < 0 else "d"
            sidestep_key = preferred_key if edge_shift_attempts == 1 else ("d" if preferred_key == "a" else "a")
            _set_peek(False, now_t)
            self._tap_key_preserving_held(sidestep_key, hold_s=0.30)
            no_reticle_ticks = 0
            no_lock_ticks = 0
            aligned_ticks = 0
            suppressive_no_red_count = 0
            self._logger.log(
                tool="runner",
                action="cover_edge_shift",
                status="ok",
                details={
                    "attempt": edge_shift_attempts,
                    "sidestep_key": sidestep_key,
                    "bearing": round(enemy.bearing, 3),
                    "reason": block_reason,
                },
            )
            _set_peek(True, now_t)
            return None

        def _on_tick(state: FrameState) -> Optional[str]:
            nonlocal special_triggered, duck_until_t, duck_logged, duck_reason_heal, duck_start_health, peak_health, sweep_idx, sweep_offset_x, sweep_offset_y, shots_fired, no_enemy_ticks, no_enemy_since_t, aligned_ticks, target_idx, no_lock_ticks, steer_cadence_ticks, large_steer_ticks, clamped_ticks, unclamp_attempts, flank_stage2_done, toggled_cover_for_flank, pre_flank_bearing, last_clamped_bearing, no_reticle_ticks, edge_shift_attempts, suppressive_no_red_count, visual_suppressed_until_t
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
                        duck_posture = classify_posture(state.capture.image, human_detector=self._get_human_detector())
                        if (not self._in_cover_hint) or (
                            duck_posture.state == "standing"
                            and duck_posture.head_top_y is not None
                            and duck_posture.head_top_y < 0.45
                        ):
                            self.enter_cover(force=False, settle_s=0.15)
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
            _, yaw_stalled = cam_fb.observe(
                state.capture.image,
                mode="cover_peek_combat",
                logger=self._logger,
                input_ctrl=self.input,
                aim_button=aim_button,
            )

            reticle = detect_red_reticle(state.capture.image)
            if reticle.is_red:
                no_enemy_ticks = 0
                no_enemy_since_t = None
                no_reticle_ticks = 0
                edge_shift_attempts = 0
                suppressive_no_red_count = 0
                aligned_ticks = 0
                no_lock_ticks = 0
                visual_suppressed_until_t = 0.0
                steer_cadence_ticks = 0
                large_steer_ticks = 0
                clamped_ticks = 0
                last_clamped_bearing = None
                self._fire_bursts(fire_button, bursts=3, hold_s=0.03, gap_s=0.025, reason="red_reticle")
                cam_fb.fired_last_tick = True
                if recoil_comp_dy > 0:
                    self.input.mouse_delta(0, int(recoil_comp_dy) * 2, steps=1, duration_s=0.01)
                    cam_fb.last_cmd_dy = int(recoil_comp_dy) * 2
                shots_fired += 1
                if max_shots is not None and shots_fired >= max_shots:
                    _recenter_sweep()
                    return "reticle_shots_done"
                return None

            enemy = self._select_cycle_target(mm.targets["red"], target_idx)
            if enemy is None:
                steer_cadence_ticks = 0
                large_steer_ticks = 0
                clamped_ticks = 0
                last_clamped_bearing = None
                no_enemy_ticks += 1
                if no_enemy_since_t is None:
                    no_enemy_since_t = now_t
                aligned_ticks = 0
                no_lock_ticks = 0
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
                if reticle.is_visible:
                    no_reticle_ticks = 0
                else:
                    no_reticle_ticks += 1
                    if no_reticle_ticks >= 6:
                        res = _handle_blocked_cover(now_t, enemy, "no_reticle")
                        return res

                # Flanking / cover-clamped enemy (> 0.45) or sustained optical-flow yaw stall (> 0.25):
                # 2-stage recovery (free-look+sidestep, then q cover-break + aim turn) if healthy.
                if abs(enemy.bearing) > 0.45 or (yaw_stalled and abs(enemy.bearing) > 0.25):
                    if (
                        last_clamped_bearing is not None
                        and (enemy.bearing * last_clamped_bearing) > 0
                        and abs(enemy.bearing - last_clamped_bearing) < 0.06
                    ):
                        clamped_ticks += 1
                    else:
                        clamped_ticks = 1
                    if yaw_stalled and abs(enemy.bearing) > 0.25:
                        clamped_ticks = max(clamped_ticks, 5)
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

                if reticle.is_visible and now_t >= visual_suppressed_until_t and abs(enemy.bearing) <= 0.45:
                    visual = self._detect_visual_target(state.capture.image, mm, enemy)
                else:
                    visual = None

                if visual is not None:
                    cam_fb.visual_steer_count += 1
                    if cam_fb.visual_steer_count == 1 or cam_fb.visual_steer_count % 5 == 0:
                        self._logger.log(
                            tool="runner",
                            action="visual_target_steer",
                            status="ok",
                            details={
                                "mode": "cover_peek_combat",
                                "aim_point": [round(visual.target.aim_point[0], 3), round(visual.target.aim_point[1], 3)],
                                "dx": visual.dx,
                                "dy": visual.dy,
                                "source": visual.target.source,
                                "confidence": round(visual.target.confidence, 3),
                            },
                        )
                    if visual.dx != 0 or visual.dy != 0:
                        self._steer_camera(visual.dx, visual.dy)
                        cam_fb.last_cmd_dx = visual.dx
                        cam_fb.last_cmd_dy = visual.dy
                    no_lock_ticks += 1
                    if no_lock_ticks >= 16:
                        no_lock_ticks = 0
                        aligned_ticks = 0
                        target_idx += 1
                        _recenter_sweep()
                        return None
                    if abs(visual.error_x) <= 0.045 and abs(visual.error_y) <= 0.045:
                        aligned_ticks += 1
                        if aligned_ticks >= 5:
                            visual_suppressed_until_t = now_t + 1.8
                            aligned_ticks = 0
                            no_lock_ticks = 0
                            target_idx += 1
                            _recenter_sweep()
                            return None
                        if self._handle_visual_aligned_tick(
                            fire_button=fire_button,
                            aligned_ticks=aligned_ticks,
                            recoil_comp_dy=recoil_comp_dy,
                            cam_fb=cam_fb,
                        ):
                            shots_fired += 1
                            suppressive_no_red_count += 1
                            if max_shots is not None and shots_fired >= max_shots:
                                _recenter_sweep()
                                return "suppressive_shots_done"
                            if suppressive_no_red_count >= 6:
                                return _handle_blocked_cover(now_t, enemy, "zero_red_suppressive")
                    else:
                        aligned_ticks = 0
                    return None

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
                                cam_fb.last_cmd_dx = dx
                        if large_steer_ticks <= 2:
                            aligned_ticks = 0
                            return None
                    else:
                        large_steer_ticks = 0
                        # Steer every tick when > 0.08 off-target so cover yaw locks quickly; damp <= 0.08 to every other tick.
                        if abs(enemy.bearing) > 0.08 or steer_cadence_ticks % 2 == 1:
                            dx = self._ads_steer_dx(enemy.bearing, steer_gain=steer_gain)
                            if dx != 0:
                                self._steer_camera(dx, 0)
                                cam_fb.last_cmd_dx = dx
                else:
                    steer_cadence_ticks = 0
                    large_steer_ticks = 0

                if reticle.is_visible:
                    if abs(enemy.bearing) <= 0.16 or steer_cadence_ticks >= 8:
                        no_lock_ticks += 1
                    if no_lock_ticks >= 16:
                        no_lock_ticks = 0
                        target_idx += 1
                        _recenter_sweep()
                        return None

                    if abs(enemy.bearing) <= suppressive_max_bearing:
                        aligned_ticks += 1
                        if no_lock_ticks in (6, 12):
                            self._fire_bursts(fire_button, bursts=1, hold_s=0.03, gap_s=0.03, reason="suppressive")
                            cam_fb.fired_last_tick = True
                            if recoil_comp_dy > 0:
                                sup_scale = 1 if no_lock_ticks == 6 else 2
                                comp_dy = int(recoil_comp_dy) * sup_scale
                                self.input.mouse_delta(0, comp_dy, steps=1, duration_s=0.01)
                                cam_fb.last_cmd_dy = comp_dy
                            shots_fired += 1
                            suppressive_no_red_count += 1
                            if max_shots is not None and shots_fired >= max_shots:
                                _recenter_sweep()
                                return "suppressive_shots_done"
                            if suppressive_no_red_count >= 6:
                                return _handle_blocked_cover(now_t, enemy, "zero_red_suppressive")
                    else:
                        aligned_ticks = 0
                else:
                    aligned_ticks = 0

            if scan_pattern:
                dx, dy = scan_pattern[sweep_idx % len(scan_pattern)]
                sweep_idx += 1
                if dx != 0 or dy != 0:
                    self.input.mouse_delta(dx, dy, steps=2, duration_s=0.03)
                    sweep_offset_x += dx
                    sweep_offset_y += dy
                    cam_fb.last_cmd_dx += dx
                    cam_fb.last_cmd_dy += dy
            return None

        res = self._run_loop(
            mode_name="cover_peek_combat",
            config=cfg,
            on_tick=_on_tick,
            on_before_release=_recenter_sweep,
        )
        if (
            res.stop_reason == "timeout"
            and res.last_minimap is not None
            and res.last_minimap.minimap_visible
            and res.last_minimap.health_ratio < min_peek_health
        ):
            rem_heal_s = min(3.5, max(0.0, duck_until_t - res.elapsed_s))
            if rem_heal_s > 0:
                self._sleep_chunked(rem_heal_s)
                cap = capture_game_window(
                    backend=self._window_backend,
                    logger=self._logger,
                    require_16_9=True,
                )
                mm_after = inspect_minimap(cap.image, logger=self._logger)
                res = LoopResult(
                    stop_reason=res.stop_reason,
                    ticks=res.ticks,
                    elapsed_s=res.elapsed_s + rem_heal_s,
                    banner=res.banner,
                    last_hud=res.last_hud,
                    last_minimap=mm_after,
                    actions_sent=res.actions_sent,
                )
            if res.last_minimap is not None and res.last_minimap.minimap_visible and res.last_minimap.health_ratio < 0.30:
                self._logger.log(
                    tool="runner",
                    action="cover_low_health_exit",
                    status="ok",
                    details={"health_ratio": round(res.last_minimap.health_ratio, 3)},
                )
                res = LoopResult(
                    stop_reason="low_health",
                    ticks=res.ticks,
                    elapsed_s=res.elapsed_s,
                    banner=res.banner,
                    last_hud=res.last_hud,
                    last_minimap=res.last_minimap,
                    actions_sent=res.actions_sent,
                )
        return res
