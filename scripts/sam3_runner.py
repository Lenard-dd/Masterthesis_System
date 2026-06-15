#!/usr/bin/env python3
"""SAM3 single-image runner for use inside the ``sam3`` conda environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def make_result(
    status: str,
    message: str,
    detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "detections": detections or [],
    }


def run_sam3_inference(
    image_path: Path,
    query: str,
    min_confidence: float,
    output_dir: Path,
    device: str,
    checkpoint: str | None,
    model_config: str | None,
) -> dict[str, Any]:
    """Load SAM3 and run inference.

    SAM3-specific imports intentionally live in this function so importing this
    runner in tests or tools does not require SAM3 in the main ROS environment.
    """
    try:
        import sam3  # type: ignore[import-not-found]
    except ImportError:
        return make_result(
            "NOT_AVAILABLE",
            "SAM3 is not installed or not importable in this conda environment.",
        )

    _ = (sam3, image_path, query, min_confidence, output_dir, device, checkpoint, model_config)

    # TODO: Adapt this block to the exact SAM3 Python API installed in the
    # conda environment. The expected normalized detections are:
    # {
    #     "label": query,
    #     "confidence": 0.85,
    #     "bbox_xyxy": [x1, y1, x2, y2],
    #     "mask_path": "data/perception_outputs/example_mask.png",
    #     "overlay_path": "data/perception_outputs/latest_overlay.png",
    #     "metadata": {},
    # }
    return make_result(
        "NOT_AVAILABLE",
        "SAM3 is importable, but the installed Python API is not recognized by this runner yet.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 on a single image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--output-dir", default="data/perception_outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not image_path.exists():
        print(
            json.dumps(
                make_result("FAILED", f"Image does not exist: {image_path}"),
                ensure_ascii=True,
            )
        )
        return 0

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = run_sam3_inference(
            image_path=image_path,
            query=args.query,
            min_confidence=args.min_confidence,
            output_dir=output_dir,
            device=args.device,
            checkpoint=args.checkpoint,
            model_config=args.model_config,
        )
    except Exception as exc:  # noqa: BLE001 - runner must report JSON, not crash.
        print(f"SAM3 runner failed: {exc}", file=sys.stderr)
        result = make_result("FAILED", f"SAM3 runner failed: {exc}")

    detections = result.get("detections", [])
    if result.get("status") == "SUCCESS" and not detections:
        result = make_result("NOT_FOUND", "SAM3 returned no detections.")

    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
