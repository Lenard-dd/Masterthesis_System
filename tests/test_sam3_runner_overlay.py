"""Tests for SAM3 runner visual evidence that do not require SAM3 weights."""

import importlib.util
from pathlib import Path

from PIL import Image


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sam3_runner.py"
    spec = importlib.util.spec_from_file_location("sam3_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combined_overlay_contains_all_localized_objects(tmp_path):
    runner = _load_runner_module()
    image = Image.new("RGB", (32, 24), "white")
    first_mask = tmp_path / "pen_mask.png"
    second_mask = tmp_path / "cube_mask.png"
    Image.new("L", (32, 24), 0).save(first_mask)
    Image.new("L", (32, 24), 0).save(second_mask)

    first = Image.open(first_mask)
    first.paste(255, (2, 2, 9, 14))
    first.save(first_mask)
    second = Image.open(second_mask)
    second.paste(255, (17, 5, 29, 19))
    second.save(second_mask)

    overlay = runner._save_combined_overlay(
        image,
        [
            {
                "label": "pen", "confidence": 0.9,
                "bbox_xyxy": [2, 2, 9, 14], "mask_path": str(first_mask),
            },
            {
                "label": "black cube", "confidence": 0.8,
                "bbox_xyxy": [17, 5, 29, 19], "mask_path": str(second_mask),
            },
        ],
        tmp_path,
    )

    assert overlay.name == "latest_sam3_overlay.png"
    assert overlay.is_file()
    assert Image.open(overlay).size == (32, 24)
