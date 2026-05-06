from .scene_model import SceneModel, ObjectDetection


class MockPerception:
    """Mock perception system for testing."""

    def get_scene_model(self) -> SceneModel:
        """Return a mock scene model."""
        # Mock some objects
        objects = [
            ObjectDetection(
                object_id="cube_1",
                position={"x": 0.1, "y": 0.2, "z": 0.0},
                confidence=0.95
            ),
            ObjectDetection(
                object_id="sphere_1",
                position={"x": -0.1, "y": 0.1, "z": 0.0},
                confidence=0.88
            )
        ]
        return SceneModel(objects=objects)