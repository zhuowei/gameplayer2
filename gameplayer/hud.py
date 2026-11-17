"""HUD region cropping and native macOS Vision OCR for clean 16:9 game frames."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

from PIL import Image

from gameplayer.telemetry import TelemetryLogger, get_default_logger
from gameplayer.window import normalize_to_16_9

NormalizedBox = tuple[float, float, float, float]

# Normalized (x0, y0, x1, y1) regions for a 16:9 GTA 5 frame.
HUD_REGIONS: dict[str, NormalizedBox] = {
    "top_left_help": (0.015, 0.02, 0.36, 0.28),
    "top_right_status": (0.68, 0.02, 0.985, 0.26),
    "center_banner": (0.15, 0.28, 0.85, 0.72),
    "bottom_subtitle": (0.10, 0.76, 0.90, 0.96),
}

BANNER_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("WASTED", ("WASTED",)),
    ("BUSTED", ("BUSTED",)),
    ("MISSION_FAILED", ("MISSION FAILED", "MISSION FAIL")),
    ("MISSION_PASSED", ("MISSION PASSED",)),
)


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x, y, w, h) normalized top-left origin

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 4) for v in self.bbox],
        }


@dataclass(frozen=True)
class RegionOcrResult:
    name: str
    box_px: tuple[int, int, int, int]
    lines: tuple[OcrLine, ...]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "box_px": list(self.box_px),
            "text": self.text,
            "lines": [line.to_dict() for line in self.lines],
        }


@dataclass(frozen=True)
class HudObservation:
    image_size: tuple[int, int]
    was_cropped_to_16_9: bool
    banner: Optional[str]
    subtitle_or_objective: str
    help_prompt: str
    top_right_status: str
    center_text: str
    regions: dict[str, RegionOcrResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "image_size": list(self.image_size),
            "was_cropped_to_16_9": self.was_cropped_to_16_9,
            "banner": self.banner,
            "subtitle_or_objective": self.subtitle_or_objective,
            "help_prompt": self.help_prompt,
            "top_right_status": self.top_right_status,
            "center_text": self.center_text,
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
        }


class OCRBackend(Protocol):
    """Abstract OCR backend so HUD parsing can be tested deterministically offline."""

    def recognize_lines(self, image: Image.Image) -> list[OcrLine]:
        ...


class VisionOCRBackend:
    """macOS native Vision (VNRecognizeTextRequest) fast OCR backend."""

    def __init__(
        self,
        uses_language_correction: bool = True,
        min_upscale_height: int = 300,
        min_confidence: float = 0.0,
    ) -> None:
        import Foundation  # noqa: F401
        import Vision  # noqa: F401

        self._foundation = Foundation
        self._vision = Vision
        self.uses_language_correction = uses_language_correction
        self.min_upscale_height = min_upscale_height
        self.min_confidence = min_confidence

    def recognize_lines(self, image: Image.Image) -> list[OcrLine]:
        work = image.convert("RGB")
        if self.min_upscale_height > 0 and work.height < self.min_upscale_height:
            scale = max(2, int(round(self.min_upscale_height / max(1, work.height))))
            work = work.resize((work.width * scale, work.height * scale), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        work.save(buf, format="BMP")
        raw = buf.getvalue()
        ns_data = self._foundation.NSData.dataWithBytes_length_(raw, len(raw))

        req = self._vision.VNRecognizeTextRequest.alloc().init()
        # Use Fast level: Accurate level can stall in headless macOS background sessions.
        req.setRecognitionLevel_(self._vision.VNRequestTextRecognitionLevelFast)
        req.setUsesLanguageCorrection_(bool(self.uses_language_correction))

        handler = self._vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(f"Vision OCR failed: {err}")

        lines: list[OcrLine] = []
        for obs in req.results() or ():
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            cand = cands[0]
            conf = float(cand.confidence())
            if conf < self.min_confidence:
                continue
            text = str(cand.string()).strip()
            if not text:
                continue
            bb = obs.boundingBox()
            # Vision coordinates use bottom-left origin; convert to top-left origin.
            x = float(bb.origin.x)
            w = float(bb.size.width)
            h = float(bb.size.height)
            y = max(0.0, 1.0 - float(bb.origin.y) - h)
            lines.append(OcrLine(text=text, confidence=conf, bbox=(x, y, w, h)))

        lines.sort(key=lambda item: (round(item.bbox[1], 3), item.bbox[0]))
        return lines


def crop_hud_regions(
    image: Image.Image,
    regions: Mapping[str, NormalizedBox] = HUD_REGIONS,
    crop_titlebar: bool = True,
) -> tuple[Image.Image, bool, dict[str, tuple[tuple[int, int, int, int], Image.Image]]]:
    """Normalize image to 16:9 and crop each configured HUD region."""
    norm_img, was_cropped = normalize_to_16_9(image.convert("RGB"), crop_titlebar=crop_titlebar)
    w, h = norm_img.size
    crops: dict[str, tuple[tuple[int, int, int, int], Image.Image]] = {}
    for name, (x0, y0, x1, y1) in regions.items():
        px0 = max(0, min(w - 1, int(round(x0 * w))))
        py0 = max(0, min(h - 1, int(round(y0 * h))))
        px1 = max(px0 + 1, min(w, int(round(x1 * w))))
        py1 = max(py0 + 1, min(h, int(round(y1 * h))))
        box = (px0, py0, px1, py1)
        crops[name] = (box, norm_img.crop(box))
    return norm_img, was_cropped, crops


def _clean_alpha(text: str) -> str:
    return re.sub(r"[^A-Z ]+", " ", text.upper()).strip()


def classify_center_banner(lines: Sequence[OcrLine]) -> Optional[str]:
    """Identify GTA 5 center status banners (`WASTED`, `BUSTED`, `MISSION_FAILED`, `MISSION_PASSED`)."""
    candidates = [_clean_alpha(line.text) for line in lines if line.text.strip()]
    joined = _clean_alpha(" ".join(line.text for line in lines))
    if joined:
        candidates.append(joined)

    for cand in candidates:
        if not cand:
            continue
        for canonical, keywords in BANNER_PATTERNS:
            for kw in keywords:
                if kw in cand:
                    return canonical
                # Allow minor single-character OCR glitch on bold stylized banner fonts
                for word_or_span in (cand, *cand.split()):
                    if len(kw) >= 6 and SequenceMatcher(None, word_or_span, kw).ratio() >= 0.83:
                        return canonical
    return None


def inspect_frame(
    image_or_path: Image.Image | Path | str,
    ocr_backend: Optional[OCRBackend] = None,
    logger: Optional[TelemetryLogger] = None,
    regions: Mapping[str, NormalizedBox] = HUD_REGIONS,
    crop_titlebar: bool = True,
) -> HudObservation:
    """Extract HUD region crops and OCR text from a frame or PNG path (100% offline)."""
    active_backend = ocr_backend or VisionOCRBackend()
    active_logger = logger or get_default_logger()

    with active_logger.timed("hud", "inspect_frame") as details:
        if isinstance(image_or_path, (Path, str)):
            img = Image.open(image_or_path)
        else:
            img = image_or_path

        norm_img, was_cropped, cropped = crop_hud_regions(
            img,
            regions=regions,
            crop_titlebar=crop_titlebar,
        )

        region_results: dict[str, RegionOcrResult] = {}
        for name, (box_px, crop_img) in cropped.items():
            lines = tuple(active_backend.recognize_lines(crop_img))
            region_results[name] = RegionOcrResult(name=name, box_px=box_px, lines=lines)

        center_res = region_results.get("center_banner")
        banner = classify_center_banner(center_res.lines if center_res else ())

        obs = HudObservation(
            image_size=norm_img.size,
            was_cropped_to_16_9=was_cropped,
            banner=banner,
            subtitle_or_objective=region_results["bottom_subtitle"].text if "bottom_subtitle" in region_results else "",
            help_prompt=region_results["top_left_help"].text if "top_left_help" in region_results else "",
            top_right_status=region_results["top_right_status"].text if "top_right_status" in region_results else "",
            center_text=center_res.text if center_res else "",
            regions=region_results,
        )
        details.update(
            {
                "size": list(obs.image_size),
                "banner": obs.banner,
                "has_subtitle": bool(obs.subtitle_or_objective),
                "has_help": bool(obs.help_prompt),
            }
        )
        return obs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline 16:9 HUD region crop & Vision OCR inspector")
    parser.add_argument("--inspect-frame", type=Path, required=True, help="Path to a 16:9 PNG frame to inspect")
    parser.add_argument("--save-crops", type=Path, default=None, help="Optional directory to save region crop PNGs")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.save_crops is not None:
        args.save_crops.mkdir(parents=True, exist_ok=True)
        _, _, crops = crop_hud_regions(Image.open(args.inspect_frame))
        for name, (_, crop_img) in crops.items():
            crop_img.save(args.save_crops / f"{name}.png")

    obs = inspect_frame(args.inspect_frame)
    print(json.dumps(obs.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
