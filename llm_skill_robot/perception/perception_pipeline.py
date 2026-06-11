from llm_skill_robot.perception.scene_model import DetectionResult, SceneModel
from llm_skill_robot.perception.unavailable_perception_backend import (
    UnavailablePerceptionBackend,
)


class PerceptionPipeline:
    """Route perception skill calls to a backend and retain the latest scene."""

    def __init__(self, backend=None):
        self.backend = backend or UnavailablePerceptionBackend()
        self.latest_scene = SceneModel()

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
