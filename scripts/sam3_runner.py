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


class Sam3Runtime:
    """One loaded SAM3 model that can process several text prompts per image."""

    def __init__(self, *, device: str, checkpoint: str | None, model_config: str | None):
        # SAM3-specific imports stay here so importing this runner in the ROS
        # environment does not require SAM3 or torch.
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        _ = model_config
        self.torch = torch
        self.device = _resolve_device(device, torch)
        model = build_sam3_image_model(
            device=self.device,
            checkpoint_path=checkpoint or None,
            load_from_HF=not bool(checkpoint),
        )
        self.processor = Sam3Processor(model, device=self.device)

    def infer_many(
        self,
        image_path: Path,
        queries: list[str],
        min_confidence: float,
        output_dir: Path,
    ) -> dict[str, Any]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # The vision backbone runs exactly once for this RGB image. Each query
        # then supplies only fresh text features and grounding heads.
        with self.torch.inference_mode(), _autocast_context(self.device, self.torch):
            inference_state = self.processor.set_image(image)
            query_results = []
            for query in queries:
                self.processor.reset_all_prompts(inference_state)
                output = self.processor.set_text_prompt(state=inference_state, prompt=query)
                detections = _detections_from_output(
                    query=query,
                    output=output,
                    min_confidence=min_confidence,
                    output_dir=output_dir,
                )
                query_results.append({
                    "query": query,
                    "status": "SUCCESS" if detections else "NOT_FOUND",
                    "message": (
                        f"Detected {len(detections)} object(s)." if detections
                        else f"No SAM3 detections for query: {query}"
                    ),
                    "detections": detections,
                })

        all_detections = [
            detection
            for item in query_results
            for detection in item["detections"]
        ]
        if all_detections:
            # One common overlay makes multi-object results easy to inspect.
            # Masks remain separate files because RGB-D pose estimation uses
            # each object's mask independently.
            overlay_path = _save_combined_overlay(image, all_detections, output_dir)
            for detection in all_detections:
                detection["overlay_path"] = str(overlay_path)

        status = "SUCCESS" if any(item["status"] == "SUCCESS" for item in query_results) else "NOT_FOUND"
        return {
            "status": status,
            "message": f"Processed {len(queries)} query(ies) using one SAM3 image encoding.",
            "query_results": query_results,
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
    """Compatibility one-shot entry point used by existing diagnostics."""
    try:
        runtime = Sam3Runtime(device=device, checkpoint=checkpoint, model_config=model_config)
    except ImportError:
        return make_result("NOT_AVAILABLE", "SAM3 is not installed or not importable in this conda environment.")
    result = runtime.infer_many(image_path, [query], min_confidence, output_dir)
    query_result = result["query_results"][0]
    return make_result(query_result["status"], query_result["message"], query_result["detections"])


def _detections_from_output(
    *, query: str, output: dict[str, Any], min_confidence: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    masks, boxes, scores = output.get("masks"), output.get("boxes"), output.get("scores")
    if masks is None or boxes is None or scores is None:
        return []
    return _normalize_sam3_outputs(
        query=query, masks=masks, boxes=boxes, scores=scores,
        min_confidence=min_confidence, output_dir=output_dir,
    )


def _resolve_device(device: str, torch_module) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _autocast_context(device: str, torch_module):
    if device.startswith("cuda"):
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    return nullcontext()


def _normalize_sam3_outputs(
    query: str,
    masks,
    boxes,
    scores,
    min_confidence: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []

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

        detections.append(
            {
                "label": query,
                "confidence": confidence,
                "bbox_xyxy": box,
                "mask_path": str(mask_path),
                "overlay_path": "",
                "metadata": {},
            }
        )

    return detections


def _save_combined_overlay(image, detections: list[dict[str, Any]], output_dir: Path) -> Path:
    """Draw every localized object into one human-readable SAM3 evidence image."""
    from PIL import Image, ImageDraw

    overlay = image.convert("RGBA")
    overlay_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)
    transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for index, detection in enumerate(detections, start=1):
        color = _color_for_index(index)
        mask_image = Image.open(str(detection["mask_path"])).convert("L")
        color_layer = Image.new("RGBA", image.size, color)
        overlay_layer.alpha_composite(Image.composite(color_layer, transparent, mask_image))
        box = detection["bbox_xyxy"]
        draw.rectangle(box, outline=color[:3] + (255,), width=3)
        label = f"{detection['label']} {float(detection['confidence']):.0%}"
        text_y = max(0, int(box[1]) - 15)
        draw.text((int(box[0]) + 2, text_y), label, fill=color[:3] + (255,))

    result = Image.alpha_composite(overlay, overlay_layer)
    path = output_dir / "latest_sam3_overlay.png"
    result.save(path)
    return path


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
    parser.add_argument("--image")
    parser.add_argument("--query")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--output-dir", default="data/perception_outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config")
    parser.add_argument("--serve", action="store_true", help="Read multi-query JSONL requests from stdin.")
    return parser.parse_args()


def serve(args: argparse.Namespace) -> int:
    """Run a JSONL worker; model weights stay on GPU between requests."""
    try:
        runtime = Sam3Runtime(device=args.device, checkpoint=args.checkpoint, model_config=args.model_config)
    except ImportError:
        print(json.dumps({"id": None, **make_result("NOT_AVAILABLE", "SAM3 is not installed or importable.")}), flush=True)
        return 1
    except Exception as exc:  # noqa: BLE001 - report startup failure through JSONL.
        print(json.dumps({"id": None, **make_result("FAILED", f"SAM3 worker could not start: {exc}")}), flush=True)
        return 1
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            image_path = Path(str(request.get("image", "")))
            queries = [query for query in request.get("queries", []) if isinstance(query, str) and query.strip()]
            output_dir = Path(str(request.get("output_dir", args.output_dir)))
            if not image_path.exists():
                result = {"status": "FAILED", "message": f"Image does not exist: {image_path}", "query_results": []}
            elif not queries:
                result = {"status": "FAILED", "message": "No non-empty queries supplied.", "query_results": []}
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                result = runtime.infer_many(
                    image_path, queries, float(request.get("min_confidence", args.min_confidence)), output_dir,
                )
        except Exception as exc:  # noqa: BLE001 - JSONL worker must stay alive.
            result = {"status": "FAILED", "message": f"SAM3 worker request failed: {exc}", "query_results": []}
        print(json.dumps({"id": request_id, **result}, ensure_ascii=True), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.serve:
        return serve(args)
    if not args.image or not args.query:
        print(json.dumps(make_result("FAILED", "--image and --query are required outside --serve.")))
        return 2
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
