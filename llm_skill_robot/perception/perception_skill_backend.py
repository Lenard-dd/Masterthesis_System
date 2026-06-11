from llm_skill_robot.perception.perception_pipeline import PerceptionPipeline


class PerceptionSkillBackend:
    """Skill-facing wrapper around the perception pipeline."""

    def __init__(self, pipeline: PerceptionPipeline | None = None):
        self.pipeline = pipeline or PerceptionPipeline()

    def detect_object(
        self,
        query: str,
        min_confidence: float = 0.5,
        require_pose: bool = True,
        **kwargs,
    ) -> dict:
        return self.pipeline.detect_object(
            query=query,
            min_confidence=min_confidence,
            require_pose=require_pose,
        )
