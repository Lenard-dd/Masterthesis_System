from abc import ABC, abstractmethod

from llm_skill_robot.perception.scene_model import DetectionResult, SceneModel


class BasePerceptionBackend(ABC):
    """Interface for future RGB-D and segmentation-backed perception."""

    @abstractmethod
    def get_rgb_frame(self) -> dict:
        """Return the latest RGB frame payload or a status dictionary."""

    @abstractmethod
    def get_depth_frame(self) -> dict:
        """Return the latest depth frame payload or a status dictionary."""

    @abstractmethod
    def get_camera_info(self) -> dict:
        """Return camera calibration/intrinsic data or a status dictionary."""

    @abstractmethod
    def detect_object(
        self,
        query: str,
        min_confidence: float = 0.5,
        require_pose: bool = True,
    ) -> DetectionResult:
        """Detect objects matching a natural-language or class-label query."""

    def detect_objects(
        self,
        queries: list[str],
        min_confidence: float = 0.5,
        require_pose: bool = True,
    ) -> dict:
        """Fallback multi-query interface for backends without shared capture.

        A live RGB-D backend overrides this method to use one synchronized
        observation for every requested object.
        """
        results = [
            self.detect_object(query, min_confidence=min_confidence, require_pose=require_pose)
            for query in queries
        ]
        objects = [obj for result in results for obj in result.objects]
        scene = results[-1].scene if results else self.get_scene()
        return {
            "status": "SUCCESS" if objects else (results[-1].status if results else "NOT_FOUND"),
            "message": f"Localized {len(objects)} object(s) for {len(queries)} query(ies).",
            "objects": [obj.model_dump() for obj in objects],
            "scene": scene.model_dump() if scene is not None else self.get_scene().model_dump(),
            "query_results": [
                {"query": result.query, "status": result.status, "message": result.message,
                 "object_count": len(result.objects)}
                for result in results
            ],
        }

    @abstractmethod
    def get_scene(self) -> SceneModel:
        """Return the latest scene model."""
