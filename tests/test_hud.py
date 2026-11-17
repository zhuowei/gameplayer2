from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest
from PIL import Image, ImageDraw, ImageFont

from gameplayer.hud import (
    HUD_REGIONS,
    OcrLine,
    VisionOCRBackend,
    classify_center_banner,
    crop_hud_regions,
    inspect_frame,
    main as hud_cli_main,
)
from gameplayer.telemetry import TelemetryLogger


class MockOCRBackend:
    """Returns canned OCR lines keyed by approximate region dimensions."""

    def __init__(self, responses: list[list[OcrLine]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def recognize_lines(self, image: Image.Image) -> list[OcrLine]:
        idx = self.calls
        self.calls += 1
        return self._responses[idx] if idx < len(self._responses) else []


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _make_synthetic_hud_frame(
    center_banner: Optional[str] = None,
    subtitle: Optional[str] = None,
    help_text: Optional[str] = None,
    top_right: Optional[str] = None,
) -> Image.Image:
    img = Image.new("RGB", (1280, 720), (18, 22, 28))
    draw = ImageDraw.Draw(img)

    if help_text:
        draw.rectangle((25, 25, 440, 140), fill=(10, 10, 10))
        draw.text((40, 45), help_text, fill=(245, 245, 245), font=_load_font(26))

    if top_right:
        draw.text((980, 40), top_right, fill=(240, 240, 240), font=_load_font(28))

    if center_banner:
        draw.text((420, 310), center_banner, fill=(235, 70, 70), font=_load_font(64))

    if subtitle:
        draw.rectangle((180, 595, 1100, 675), fill=(8, 8, 8))
        draw.text((310, 618), subtitle, fill=(250, 225, 90), font=_load_font(30))

    return img


def test_crop_hud_regions_normalizes_titlebar_and_all_regions() -> None:
    raw = Image.new("RGB", (1280, 748), (0, 0, 0))
    norm_img, was_cropped, crops = crop_hud_regions(raw)
    assert was_cropped is True
    assert norm_img.size == (1280, 720)
    assert set(crops.keys()) == set(HUD_REGIONS.keys())
    for box, crop in crops.values():
        assert crop.size == (box[2] - box[0], box[3] - box[1])


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("WASTED", "WASTED"),
        ("w4sted", "WASTED"),
        ("BUSTED", "BUSTED"),
        ("MISSION FAILED", "MISSION_FAILED"),
        ("MISSION PASSED", "MISSION_PASSED"),
        ("NORTH YANKTON", None),
    ],
)
def test_classify_center_banner(raw_text: str, expected: Optional[str]) -> None:
    lines = [OcrLine(text=raw_text, confidence=0.9, bbox=(0.2, 0.4, 0.6, 0.2))]
    assert classify_center_banner(lines) == expected


def test_inspect_frame_with_mock_ocr_and_telemetry(tmp_path) -> None:
    frame = _make_synthetic_hud_frame()
    mock_ocr = MockOCRBackend(
        responses=[
            [OcrLine("Press E to take cover.", 0.92, (0.1, 0.1, 0.6, 0.2))],
            [OcrLine("MICHAEL $1500", 0.88, (0.2, 0.1, 0.5, 0.2))],
            [OcrLine("MISSION FAILED", 0.95, (0.2, 0.3, 0.6, 0.3))],
            [OcrLine("Go to the guard room.", 0.94, (0.2, 0.4, 0.6, 0.2))],
        ]
    )
    logger = TelemetryLogger(tmp_path / "telemetry.jsonl")
    obs = inspect_frame(frame, ocr_backend=mock_ocr, logger=logger)

    assert obs.image_size == (1280, 720)
    assert obs.banner == "MISSION_FAILED"
    assert obs.help_prompt == "Press E to take cover."
    assert obs.top_right_status == "MICHAEL $1500"
    assert obs.subtitle_or_objective == "Go to the guard room."
    assert obs.to_dict()["banner"] == "MISSION_FAILED"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Vision OCR only")
def test_vision_ocr_on_synthetic_hud_frame_and_cli(tmp_path, capsys) -> None:
    frame = _make_synthetic_hud_frame(
        center_banner="WASTED",
        subtitle="Hold up the hostages.",
        help_text="Press E to interact.",
        top_right="MICHAEL",
    )
    png_path = tmp_path / "synthetic_hud.png"
    frame.save(png_path)

    obs = inspect_frame(png_path, ocr_backend=VisionOCRBackend(), logger=TelemetryLogger(tmp_path / "t.jsonl"))
    assert obs.banner == "WASTED"
    assert "hostages" in obs.subtitle_or_objective.lower()
    assert "interact" in obs.help_prompt.lower()
    assert "michael" in obs.top_right_status.lower()

    rc = hud_cli_main(["--inspect-frame", str(png_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["banner"] == "WASTED"
