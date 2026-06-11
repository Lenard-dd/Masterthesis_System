from .scene_model import SceneModel


class MockPerception:
    """Placeholder perception system for tests that do not require objects."""

    def get_scene_model(self) -> SceneModel:
        """Return an empty scene model without fabricated objects."""
        return SceneModel(source="mock_perception")
