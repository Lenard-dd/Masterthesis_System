"""Bind a SAM3-localized paper to a human-readable region label.

The vision-language model is deliberately restricted to reading a short label
from already localized paper candidates.  SAM3/RGB-D remains the source of
truth for masks, object poses, and all downstream geometry.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_skill_robot.agent.llm_client import OpenAILLMClient, load_llm_config
from llm_skill_robot.perception.scene_model import SceneModel


DEFAULT_VISUAL_QUERY = "white rectangular paper"


class TextGenerationClient(Protocol):
    def generate_text(self, messages: list[dict[str, Any]]) -> str:
        """Return the model's textual response."""


class RegionReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., min_length=1, max_length=40)
    label: str = Field(..., min_length=1, max_length=16)
    readable: bool


class RegionReadingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readings: list[RegionReading] = Field(default_factory=list, max_length=20)


class LabeledRegionRecognitionError(RuntimeError):
    """Raised when candidate labels cannot be read safely."""


def labeled_region_letter(query: str) -> str | None:
    """Return the requested single-character region label, if present."""
    normalized = " ".join(str(query).strip().split())
    patterns = (
        r"^(?:(?:the|a)\s+)?(?:area|region|zone)\s+([a-z0-9])$",
        r"^(?:(?:the|a)\s+)?(?:placement\s+)?(?:area|region|zone)\s+"
        r"(?:label(?:led)?\s+)?([a-z0-9])$",
        r"^(?:white\s+)?(?:rectangular\s+)?(?:paper|card|sheet)\s+"
        r"(?:marked|labelled|labeled|printed)\s+(?:with\s+)?(?:letter\s+)?([a-z0-9])$",
        r"^(?:区域|位置)\s*([a-z0-9])$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


class LabeledRegionVisionClient:
    """Read one printed label from each SAM3 paper candidate crop."""

    SYSTEM_PROMPT = (
        "You read large printed labels from paper-region candidate crops in a robot "
        "workspace. Return only one JSON object matching the requested schema. Each "
        "image is paired with a candidate_id. Read only a clearly visible single "
        "uppercase Latin letter or digit printed on the paper. Do not infer a label "
        "from the candidate_id, image order, task context, or requested target. If the "
        "mark is missing, ambiguous, occluded, or unreadable, return label UNKNOWN and "
        "readable false. Do not return coordinates or robot instructions."
    )

    def __init__(
        self,
        llm_client: TextGenerationClient,
        *,
        image_detail: str = "high",
        crop_padding_px: int = 8,
    ) -> None:
        if image_detail not in {"low", "high", "auto"}:
            raise ValueError("image_detail must be one of: low, high, auto.")
        if crop_padding_px < 0:
            raise ValueError("crop_padding_px must be non-negative.")
        self.llm_client = llm_client
        self.image_detail = image_detail
        self.crop_padding_px = int(crop_padding_px)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "LabeledRegionVisionClient":
        config = config or load_llm_config()
        llm_config = config.get("llm", {}) if isinstance(config, dict) else {}
        region_config = config.get("labeled_region_recognition", {}) if isinstance(config, dict) else {}
        return cls(
            OpenAILLMClient(
                model=region_config.get("model") or llm_config.get("model"),
                temperature=float(region_config.get("temperature", 0.0)),
                max_retries=int(llm_config.get("max_retries", 2)),
                timeout_sec=float(region_config.get("timeout_sec", llm_config.get("timeout_sec", 30.0))),
            ),
            image_detail=str(region_config.get("image_detail", "high")),
            crop_padding_px=int(region_config.get("crop_padding_px", 8)),
        )

    def read_labels(
        self,
        image_path: str | Path,
        candidates: list[dict[str, Any]],
    ) -> dict[str, str]:
        path = Path(image_path)
        if not path.is_file():
            raise LabeledRegionRecognitionError(f"RGB image does not exist: {path}")
        if not candidates:
            return {}
        messages, candidate_ids = self._build_messages(path, candidates)
        response = self.llm_client.generate_text(messages)
        try:
            parsed = RegionReadingResponse.model_validate_json(self._strip_fence(response))
        except (ValidationError, ValueError) as exc:
            raise LabeledRegionRecognitionError(
                f"Region-label model returned invalid JSON: {exc}"
            ) from exc

        readings: dict[str, str] = {}
        for reading in parsed.readings:
            if reading.candidate_id not in candidate_ids:
                raise LabeledRegionRecognitionError(
                    f"Region-label model invented candidate {reading.candidate_id!r}."
                )
            if reading.candidate_id in readings:
                raise LabeledRegionRecognitionError(
                    f"Region-label model repeated candidate {reading.candidate_id!r}."
                )
            label = reading.label.strip().upper()
            if not reading.readable:
                label = "UNKNOWN"
            elif not re.fullmatch(r"[A-Z0-9]", label):
                raise LabeledRegionRecognitionError(
                    f"Candidate {reading.candidate_id!r} did not contain one valid label."
                )
            readings[reading.candidate_id] = label
        for candidate_id in candidate_ids:
            readings.setdefault(candidate_id, "UNKNOWN")
        return readings

    def _build_messages(
        self,
        image_path: Path,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise LabeledRegionRecognitionError(
                "Pillow is required to crop labeled paper candidates."
            ) from exc

        candidate_ids: set[str] = set()
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps({
                "task": "Read the visible printed label in every candidate paper crop.",
                "schema": RegionReadingResponse.model_json_schema(),
                "rules": [
                    "Return exactly one reading for every supplied candidate_id.",
                    "A readable label is exactly one uppercase Latin letter or digit.",
                    "Use UNKNOWN with readable=false whenever the mark is not clear.",
                ],
            }, ensure_ascii=False),
        }]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            for index, candidate in enumerate(candidates, start=1):
                candidate_id = str(candidate.get("candidate_id") or f"paper_{index}")
                if candidate_id in candidate_ids:
                    raise LabeledRegionRecognitionError(
                        f"Duplicate region candidate ID {candidate_id!r}."
                    )
                box = candidate.get("bbox_xyxy")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    raise LabeledRegionRecognitionError(
                        f"Candidate {candidate_id!r} has no valid bounding box."
                    )
                left, top, right, bottom = (int(round(float(value))) for value in box)
                padding = self.crop_padding_px
                left, top = max(0, left - padding), max(0, top - padding)
                right, bottom = min(width, right + padding), min(height, bottom + padding)
                if right <= left or bottom <= top:
                    raise LabeledRegionRecognitionError(
                        f"Candidate {candidate_id!r} has an empty crop."
                    )
                crop = rgb.crop((left, top, right, bottom))
                buffer = BytesIO()
                crop.save(buffer, format="PNG")
                data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                content.extend([
                    {"type": "text", "text": f"candidate_id: {candidate_id}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": self.image_detail},
                    },
                ])
                candidate_ids.add(candidate_id)
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ], candidate_ids

    @staticmethod
    def _strip_fence(value: str) -> str:
        text = (value or "").strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()


class LabeledRegionResolver:
    """Relabel exactly one localized paper for each requested region label."""

    def __init__(
        self,
        vision_client: LabeledRegionVisionClient,
        *,
        visual_query: str = DEFAULT_VISUAL_QUERY,
    ) -> None:
        self.vision_client = vision_client
        self.visual_query = " ".join(str(visual_query).strip().split())
        if not self.visual_query:
            raise ValueError("visual_query must not be empty.")

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "LabeledRegionResolver":
        config = config or load_llm_config()
        region_config = config.get("labeled_region_recognition", {}) if isinstance(config, dict) else {}
        return cls(
            LabeledRegionVisionClient.from_config(config),
            visual_query=str(region_config.get("visual_query", DEFAULT_VISUAL_QUERY)),
        )

    def resolve(
        self,
        scene: SceneModel,
        image_path: str | Path,
        requested_queries: list[str],
    ) -> tuple[SceneModel, dict[str, Any]]:
        requested = {
            query: letter
            for query in requested_queries
            if (letter := labeled_region_letter(query)) is not None
        }
        papers = [
            item for item in scene.objects
            if str(item.metadata.get("query", item.label)) == self.visual_query
        ]
        candidates = []
        candidate_objects = {}
        for index, paper in enumerate(papers, start=1):
            bbox = paper.metadata.get("bbox_xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            candidate_id = f"paper_{index}"
            candidates.append({"candidate_id": candidate_id, "bbox_xyxy": list(bbox)})
            candidate_objects[candidate_id] = paper
        if not candidates:
            return scene, {
                "status": "NOT_FOUND",
                "message": f"SAM3 found no usable {self.visual_query!r} candidate for labeled-region reading.",
                "resolved_queries": [],
            }

        readings = self.vision_client.read_labels(image_path, candidates)
        resolved_objects = {item.object_id: item for item in scene.objects}
        resolved_queries: list[str] = []
        ambiguous_queries: list[str] = []
        for query, requested_letter in requested.items():
            matches = [
                candidate_objects[candidate_id]
                for candidate_id, label in readings.items()
                if label == requested_letter and candidate_id in candidate_objects
            ]
            if len(matches) != 1:
                ambiguous_queries.append(query)
                continue
            paper = matches[0]
            metadata = {
                **paper.metadata,
                "visual_label": self.visual_query,
                "region_type": "labeled_paper",
                "region_label": requested_letter,
                "requested_region_query": query,
                "label_reader": "vision_llm",
            }
            resolved_objects[paper.object_id] = paper.model_copy(update={
                "label": query,
                "source": "d435i_sam3+vision_llm",
                "metadata": metadata,
            })
            resolved_queries.append(query)

        resolved_scene = scene.model_copy(update={
            "objects": [resolved_objects[item.object_id] for item in scene.objects],
            "source": "d435i_sam3+vision_llm",
        })
        if ambiguous_queries:
            return resolved_scene, {
                "status": "NOT_FOUND",
                "message": (
                    "Vision-language label reading did not produce exactly one match for: "
                    + ", ".join(ambiguous_queries)
                ),
                "resolved_queries": resolved_queries,
                "ambiguous_queries": ambiguous_queries,
                "readings": readings,
            }
        return resolved_scene, {
            "status": "SUCCESS",
            "message": "Bound labeled paper region(s): " + ", ".join(resolved_queries),
            "resolved_queries": resolved_queries,
            "readings": readings,
        }
