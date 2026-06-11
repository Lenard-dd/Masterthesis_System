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

    @abstractmethod
    def get_scene(self) -> SceneModel:
        """Return the latest scene model."""
