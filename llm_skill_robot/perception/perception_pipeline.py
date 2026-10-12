from typing import Any

from llm_skill_robot.agent.llm_client import load_llm_config
from llm_skill_robot.perception.labeled_region_recognition import (
    DEFAULT_VISUAL_QUERY,
    LabeledRegionResolver,
    labeled_region_letter,
)
from llm_skill_robot.perception.scene_model import DetectionResult, SceneModel
from llm_skill_robot.perception.unavailable_perception_backend import (
    UnavailablePerceptionBackend,
)


class PerceptionPipeline:
    """Route perception skill calls to a backend and retain the latest scene."""

    def __init__(
        self,
        backend=None,
        labeled_region_resolver: LabeledRegionResolver | None = None,
        labeled_region_config: dict[str, Any] | None = None,
    ):
        self.backend = backend or UnavailablePerceptionBackend()
        self.latest_scene = SceneModel()
        llm_config = load_llm_config()
        configured = (
            labeled_region_config
            if labeled_region_config is not None
            else llm_config.get("labeled_region_recognition", {})
        )
        self._labeled_region_config = dict(configured or {})
        self._labeled_region_resolver = labeled_region_resolver

    def detect_object(
        self,
        query: str,
        min_confidence: float = 0.5,
        require_pose: bool = True,
    ) -> dict:
        result = self.backend.detect_object(
            query=query,
            min_confidence=min_confidence,
            require_pose=require_pose,
        )
        if not isinstance(result, DetectionResult):
            result = DetectionResult.model_validate(result)

        if result.scene is not None:
            self.latest_scene = result.scene

        return {
            "success": result.status == "SUCCESS",
            "status": result.status,
            "message": result.message,
            "objects": [scene_object.model_dump() for scene_object in result.objects],
            "scene": self.latest_scene.model_dump(),
        }

    def get_latest_scene(self) -> SceneModel:
        return self.latest_scene

    def detect_objects(
        self,
        queries: list[str],
        min_confidence: float = 0.5,
        require_pose: bool = True,
    ) -> dict:
        requested_queries = list(dict.fromkeys(
            query.strip() for query in queries if isinstance(query, str) and query.strip()
        ))
        requested_region_queries = [
            query for query in requested_queries if labeled_region_letter(query) is not None
        ]
        region_enabled = bool(self._labeled_region_config.get("enabled", True))
        region_queries = requested_region_queries if region_enabled else []
        visual_query = str(
            self._labeled_region_config.get("visual_query", DEFAULT_VISUAL_QUERY)
        ).strip() or DEFAULT_VISUAL_QUERY
        detector_queries = list(dict.fromkeys(
            visual_query if query in region_queries else query
            for query in requested_queries
        ))
        result = self.backend.detect_objects(
            queries=detector_queries,
            min_confidence=min_confidence,
            require_pose=require_pose,
        )
        scene_data = result.get("scene") if isinstance(result, dict) else None
        if scene_data is not None:
            self.latest_scene = SceneModel.model_validate(scene_data)
        if region_queries:
            result = self._resolve_labeled_regions(
                result, requested_queries, region_queries, visual_query,
            )
        return {
            **result,
            "success": result.get("status") == "SUCCESS",
            "objects": result.get("objects", []),
            "scene": self.latest_scene.model_dump(),
            "query_results": result.get("query_results", []),
        }

    def _resolve_labeled_regions(
        self,
        result: dict[str, Any],
        requested_queries: list[str],
        region_queries: list[str],
        visual_query: str,
    ) -> dict[str, Any]:
        """Bind requested area labels while preserving SAM3/RGB-D geometry."""
        updated = dict(result)
        if str(updated.get("status")) in {"FAILED", "NOT_AVAILABLE"}:
            return updated
        image_path = updated.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            updated.update({
                "status": "FAILED",
                "message": "Labeled-region recognition requires the RGB image used by SAM3.",
            })
            return updated
        try:
            resolver = self._labeled_region_resolver
            if resolver is None:
                resolver = LabeledRegionResolver.from_config()
                self._labeled_region_resolver = resolver
            # An injected resolver owns its query setting. Configuration and
            # detector routing must agree so candidate association stays exact.
            if resolver.visual_query != visual_query:
                raise ValueError(
                    "Labeled-region visual query differs between resolver and pipeline."
                )
            scene, resolution = resolver.resolve(
                self.latest_scene, image_path, region_queries,
            )
        except Exception as exc:
            updated.update({
                "status": "FAILED",
                "message": f"Labeled-region recognition failed: {exc}",
                "labeled_region_resolution": {
                    "status": "FAILED", "message": str(exc),
                },
            })
            return updated

        self.latest_scene = scene
        if hasattr(self.backend, "latest_scene"):
            self.backend.latest_scene = scene
        resolved_queries = set(resolution.get("resolved_queries", []))
        raw_query_results = updated.get("query_results", [])
        rewritten_query_results = []
        for query in requested_queries:
            if query in region_queries:
                succeeded = query in resolved_queries
                rewritten_query_results.append({
                    "query": query,
                    "status": "SUCCESS" if succeeded else str(resolution.get("status", "NOT_FOUND")),
                    "message": (
                        f"Bound {query!r} using SAM3 paper geometry and vision-language label reading."
                        if succeeded else str(resolution.get("message", "Labeled region was not resolved."))
                    ),
                    "object_count": 1 if succeeded else 0,
                })
                continue
            match = next(
                (item for item in raw_query_results if item.get("query") == query),
                None,
            )
            rewritten_query_results.append(match or {
                "query": query, "status": "NOT_FOUND", "message": "No result.", "object_count": 0,
            })

        updated.update({
            "objects": [item.model_dump() for item in scene.objects],
            "scene": scene.model_dump(),
            "query_results": rewritten_query_results,
            "labeled_region_resolution": resolution,
            "message": f"{updated.get('message', '')} {resolution.get('message', '')}".strip(),
        })
        if str(resolution.get("status")) != "SUCCESS":
            updated["status"] = str(resolution.get("status", "NOT_FOUND"))
        return updated
