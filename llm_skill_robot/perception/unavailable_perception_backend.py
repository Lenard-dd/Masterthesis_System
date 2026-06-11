from llm_skill_robot.perception.perception_backend_base import BasePerceptionBackend
from llm_skill_robot.perception.scene_model import DetectionResult, SceneModel


NOT_AVAILABLE_MESSAGE = (
    "Perception backend is not connected. D435i/SAM3 is not available yet."
)


class UnavailablePerceptionBackend(BasePerceptionBackend):
    """Perception backend used before real D435i/SAM3 integration exists."""

    def get_rgb_frame(self) -> dict:
        return {
            "status": "NOT_AVAILABLE",
            "message": NOT_AVAILABLE_MESSAGE,
            "frame": None,
        }

    def get_depth_frame(self) -> dict:
        return {
            "status": "NOT_AVAILABLE",
            "message": NOT_AVAILABLE_MESSAGE,
            "frame": None,
        }

    def get_camera_info(self) -> dict:
        return {
            "status": "NOT_AVAILABLE",
            "message": NOT_AVAILABLE_MESSAGE,
            "camera_info": None,
        }

    def detect_object(
        self,
        query: str,
        min_confidence: float = 0.5,
        require_pose: bool = True,
    ) -> DetectionResult:
        scene = self.get_scene()
        return DetectionResult(
            query=query,
            status="NOT_AVAILABLE",
            objects=[],
            message=NOT_AVAILABLE_MESSAGE,
            scene=scene,
        )

    def get_scene(self) -> SceneModel:
        return SceneModel(source="unavailable_perception_backend")
