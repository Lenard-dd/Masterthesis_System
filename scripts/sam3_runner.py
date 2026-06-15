#!/usr/bin/env python3
"""SAM3 single-image runner for use inside the ``sam3`` conda environment."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError:
        return make_result(
            "NOT_AVAILABLE",
            "SAM3 is not installed or not importable in this conda environment.",
        )

    _ = model_config
    resolved_device = _resolve_device(device, torch)
    model = build_sam3_image_model(
        device=resolved_device,
        checkpoint_path=checkpoint or None,
        load_from_HF=not bool(checkpoint),
    )
    processor = Sam3Processor(model)

    image = Image.open(image_path).convert("RGB")
    autocast_context = _autocast_context(resolved_device, torch)
    with torch.inference_mode(), autocast_context:
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=query)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")
    if masks is None or boxes is None or scores is None:
        return make_result(
            "FAILED",
            "SAM3 output did not contain masks, boxes, and scores.",
        )

    detections = _normalize_sam3_outputs(
        image=image,
        query=query,
        masks=masks,
        boxes=boxes,
        scores=scores,
        min_confidence=min_confidence,
        output_dir=output_dir,
    )
    if not detections:
        return make_result("NOT_FOUND", f"No SAM3 detections for query: {query}")

    return make_result("SUCCESS", f"Detected {len(detections)} object(s).", detections)


def _resolve_device(device: str, torch_module) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _autocast_context(device: str, torch_module):
    if device.startswith("cuda"):
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    return nullcontext()


def _normalize_sam3_outputs(
    image,
    query: str,
    masks,
    boxes,
    scores,
    min_confidence: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    detections: list[dict[str, Any]] = []
    overlay = image.convert("RGBA")
    overlay_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)

    mask_tensor = masks.detach().cpu()
    box_tensor = boxes.detach().cpu()
    score_tensor = scores.detach().cpu()

    if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
        mask_tensor = mask_tensor[:, 0]

    safe_query = _safe_stem(query)
    for index, score in enumerate(score_tensor.tolist(), start=1):
        confidence = float(score)
        if confidence < min_confidence:
            continue

        mask = mask_tensor[index - 1] > 0
        box = [float(value) for value in box_tensor[index - 1].tolist()]
        mask_path = output_dir / f"{safe_query}_{index}_mask.png"
        _save_mask(mask, mask_path)

        mask_image = Image.open(mask_path).convert("L")
        color = _color_for_index(index)
        color_layer = Image.new("RGBA", image.size, color)
        overlay_layer.alpha_composite(
            Image.composite(color_layer, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask_image)
        )
        draw.rectangle(box, outline=color[:3] + (255,), width=3)

        detections.append(
            {
                "label": query,
                "confidence": confidence,
                "bbox_xyxy": box,
                "mask_path": str(mask_path),
                "overlay_path": str(output_dir / "latest_overlay.png"),
                "metadata": {},
            }
        )

    if detections:
        overlay = Image.alpha_composite(overlay, overlay_layer)
        overlay.save(output_dir / "latest_overlay.png")

    return detections


def _save_mask(mask, path: Path) -> None:
    import torch
    from PIL import Image

    mask_uint8 = mask.to(dtype=torch.uint8) * 255
    Image.fromarray(mask_uint8.numpy()).save(path)


def _safe_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    return safe or "object"


def _color_for_index(index: int) -> tuple[int, int, int, int]:
    colors = [
        (255, 70, 70, 95),
        (70, 160, 255, 95),
        (80, 220, 130, 95),
        (255, 205, 70, 95),
    ]
    return colors[(index - 1) % len(colors)]


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
