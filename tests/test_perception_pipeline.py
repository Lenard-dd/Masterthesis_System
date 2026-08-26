from llm_skill_robot.perception.perception_pipeline import PerceptionPipeline
from llm_skill_robot.perception.perception_skill_backend import PerceptionSkillBackend
from llm_skill_robot.perception.scene_model import SceneModel


def test_default_pipeline_returns_not_available():
    pipeline = PerceptionPipeline()

    result = pipeline.detect_object("red cube")

    assert result["success"] is False
    assert result["status"] == "NOT_AVAILABLE"
    assert "not connected" in result["message"]
    assert result["objects"] == []
    assert result["scene"]["objects"] == []


def test_latest_scene_remains_empty_when_no_detection_is_available():
    pipeline = PerceptionPipeline()

    pipeline.detect_object("red cube")
    scene = pipeline.get_latest_scene()

    assert isinstance(scene, SceneModel)
    assert scene.objects == []


def test_detect_object_returns_success_false_when_backend_is_unavailable():
    skill_backend = PerceptionSkillBackend()

    result = skill_backend.detect_object(
        query="red cube",
        min_confidence=0.6,
        require_pose=True,
    )

    assert result["success"] is False
    assert result["status"] == "NOT_AVAILABLE"


def test_detect_objects_returns_one_scene_and_per_query_statuses_when_unavailable():
    result = PerceptionPipeline().detect_objects(["black cube", "red cube"])

    assert result["success"] is False
    assert result["status"] == "NOT_AVAILABLE"
    assert result["objects"] == []
    assert [item["query"] for item in result["query_results"]] == ["black cube", "red cube"]
