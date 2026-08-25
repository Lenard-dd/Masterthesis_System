"""Vision-language scene description for read-only scene scanning.

This module deliberately separates semantic scene description from geometric
perception.  A vision-language model may suggest *candidate* objects, but it
does not create tracked objects, masks, poses, or robot actions.  SAM3 and
RGB-D remain the source of truth for those later stages.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_skill_robot.agent.llm_client import OpenAILLMClient, load_llm_config


class TextGenerationClient(Protocol):
    """Minimal provider-neutral dependency used by scene understanding."""

    def generate_text(self, messages: list[dict[str, Any]]) -> str:
        """Return the model's textual response."""


class SceneCandidate(BaseModel):
    """An unverified object candidate suggested by a vision-language model."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=80)
    description: str = Field(..., min_length=1, max_length=120)
    manipulable: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class SceneDescription(BaseModel):
    """Schema returned from the semantic, non-geometric scene scan."""

    model_config = ConfigDict(extra="forbid")

    scene_type: str = Field(..., min_length=1, max_length=80)
    summary: str = Field(..., min_length=1, max_length=220)
    candidate_objects: list[SceneCandidate] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list, max_length=8)


class SceneUnderstandingError(RuntimeError):
    """Raised when a vision-language response cannot be used safely."""


class SceneUnderstandingClient:
    """Ask a vision-capable language model for a bounded scene summary."""

    SYSTEM_PROMPT = (
        "You describe one robot camera image for a read-only tabletop scene scan. "
        "Return only one JSON object matching the requested schema. Do not use "
        "Markdown. Do not output coordinates, masks, object IDs, robot actions, "
        "grasp instructions, or claims about objects that are not visibly supported. "
        "Candidate queries must be short English noun phrases suitable for later "
        'segmentation, such as "black cube" or "blue bottle". Candidates are '
        "unverified suggestions, not detections. Keep the summary to two short "
        "sentences and each candidate description to one short phrase."
    )

    def __init__(
        self,
        llm_client: TextGenerationClient,
        *,
        image_detail: str = "low",
        max_candidates: int = 8,
    ):
        if image_detail not in {"low", "high", "auto"}:
            raise ValueError("image_detail must be one of: low, high, auto.")
        self.llm_client = llm_client
        self.image_detail = image_detail
        self.max_candidates = max_candidates

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "SceneUnderstandingClient":
        config = config or load_llm_config()
        llm_config = config.get("llm", {}) if isinstance(config, dict) else {}
        scene_config = config.get("scene_understanding", {}) if isinstance(config, dict) else {}
        client = OpenAILLMClient(
            model=scene_config.get("model") or llm_config.get("model"),
            temperature=float(scene_config.get("temperature", 0.0)),
            max_retries=int(llm_config.get("max_retries", 2)),
            timeout_sec=float(scene_config.get("timeout_sec", llm_config.get("timeout_sec", 30.0))),
        )
        return cls(
            client,
            image_detail=str(scene_config.get("image_detail", "low")),
            max_candidates=int(scene_config.get("max_candidates", 8)),
        )

    def describe_image(self, image_path: str | Path, *, max_candidates: int | None = None) -> SceneDescription:
        path = Path(image_path)
        if not path.is_file():
            raise SceneUnderstandingError(f"RGB image does not exist: {path}")
        requested_limit = self.max_candidates if max_candidates is None else max_candidates
        if not 1 <= requested_limit <= 12:
            raise ValueError("max_candidates must be between 1 and 12.")

        response = self.llm_client.generate_text(
            self._build_messages(path, requested_limit)
        )
        try:
            description = SceneDescription.model_validate_json(self._strip_fence(response))
        except (ValidationError, ValueError) as exc:
            raise SceneUnderstandingError(
                f"Scene model did not return valid SceneDescription JSON: {exc}"
            ) from exc
        if len(description.candidate_objects) > requested_limit:
            raise SceneUnderstandingError(
                f"Scene model returned {len(description.candidate_objects)} candidates; "
                f"the requested maximum is {requested_limit}."
            )
        return description

    def _build_messages(self, image_path: Path, max_candidates: int) -> list[dict[str, Any]]:
        image_data_url = self._image_data_url(image_path)
        schema = SceneDescription.model_json_schema()
        instruction = {
            "task": "Describe the visible environment and propose only visually supported candidate objects for later verification.",
            "maximum_candidate_objects": max_candidates,
            "scene_description_schema": schema,
            "rules": [
                "Use scene_type 'tabletop_workspace' for a normal robot tabletop scene when appropriate.",
                "Set manipulable=true only for visible, plausibly graspable objects.",
                "Put edges, clutter, occlusion, or uncertain areas in hazards.",
                "Do not infer exact sizes, 3D positions, or unseen objects.",
            ],
        }
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(instruction, ensure_ascii=False)},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": self.image_detail},
                    },
                ],
            },
        ]

    @staticmethod
    def _image_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise SceneUnderstandingError(
                f"Unsupported RGB image type for scene description: {image_path.suffix or 'unknown'}"
            )
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

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


class SceneDescriptionSkillBackend:
    """Skill-facing bridge that captures RGB and returns an unverified summary."""

    def __init__(
        self,
        perception_pipeline,
        scene_client: SceneUnderstandingClient | None = None,
    ):
        self.perception_pipeline = perception_pipeline
        self.scene_client = scene_client

    def describe_scene(self, max_candidates: int = 8, **_kwargs) -> dict[str, Any]:
        camera_backend = getattr(self.perception_pipeline, "backend", None)
        capture = getattr(camera_backend, "capture_rgb_image", None)
        if not callable(capture):
            return self._result(
                "NOT_AVAILABLE",
                "Scene description requires a perception backend that can capture an RGB image.",
            )
        capture_result = capture()
        if not isinstance(capture_result, dict) or capture_result.get("status") != "SUCCESS":
            return self._result(
                str(capture_result.get("status", "FAILED")) if isinstance(capture_result, dict) else "FAILED",
                str(capture_result.get("message", "Unable to capture an RGB image.")) if isinstance(capture_result, dict) else "Unable to capture an RGB image.",
            )
        try:
            client = self.scene_client or SceneUnderstandingClient.from_config()
            description = client.describe_image(
                capture_result["path"], max_candidates=max_candidates
            )
        except Exception as exc:  # Skill boundary must return a structured failure.
            return self._result("FAILED", f"describe_scene failed: {exc}")

        return {
            "success": True,
            "status": "SUCCESS",
            "message": "Scene description completed. Candidate objects are unverified; use SAM3/RGB-D before manipulation.",
            "scene_description": description.model_dump(),
            "image_path": capture_result["path"],
        }

    @staticmethod
    def _result(status: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "status": status,
            "message": message,
            "scene_description": None,
        }
