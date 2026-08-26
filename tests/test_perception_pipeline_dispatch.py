from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.plan import PlanStep
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.perception.perception_pipeline import PerceptionPipeline
from llm_skill_robot.perception.scene_model import (
    DetectionResult,
    ObjectPose,
    SceneModel,
    SceneObject,
)
from llm_skill_robot.ros_nl_rviz_sim_demo import (
    _audit_perception_result,
    _verify_object_step_for_execution,
    execute_non_motion_skill,
)


class _SceneDescriptionBackend:
    def describe_scene(self, max_candidates=8):
        return {
            "success": True,
            "status": "SUCCESS",
            "message": "semantic scene description",
            "scene_description": {
                "scene_type": "tabletop_workspace",
                "summary": "A cube is visible.",
                "candidate_objects": [],
                "hazards": [],
            },
            "max_candidates": max_candidates,
        }


def test_describe_scene_dispatches_without_creating_geometric_scene():
    context = {}
    step = PlanStep(
        step_id="1",
        skill_id="describe_scene",
        description="Describe scene",
        parameters={"max_candidates": 4},
    )

    result = execute_non_motion_skill(
        step,
        gripper_backend=None,
        context=context,
        scene_description_backend=_SceneDescriptionBackend(),
    )

    assert result["success"] is True
    assert result["status"] == "SUCCESS"
    assert context["unverified_scene_description"]["scene_type"] == "tabletop_workspace"
    assert "scene" not in context


def test_detect_object_step_dispatches_to_default_unavailable_backend():
    context = {}
    step = PlanStep(
        step_id="1",
        skill_id="detect_object",
        description="Detect red cube",
        parameters={"query": "red cube"},
    )

    result = execute_non_motion_skill(
        step,
        gripper_backend=None,
        context=context,
    )

    assert result["success"] is False
    assert result["skill_id"] == "detect_object"
    assert result["status"] == "NOT_AVAILABLE"
    assert "D435i/SAM3 is not available yet" in result["message"]
    assert result["objects"] == []
    assert context["perception_available"] is False
    assert context["scene"].objects == []


class _MultiObjectBackend:
    def detect_objects(self, queries, min_confidence=0.5, require_pose=True):
        objects = [
            SceneObject(
                object_id=f"{query}_1", label=query, confidence=0.9,
                pose_available=True,
                pose=ObjectPose(frame="base_link", x=index, y=0.0, z=0.1),
                source="d435i_sam3",
            )
            for index, query in enumerate(queries)
        ]
        scene = SceneModel(objects=objects, frame="base_link", source="d435i_sam3")
        return {
            "status": "SUCCESS",
            "message": "localized",
            "objects": [item.model_dump() for item in objects],
            "scene": scene.model_dump(),
            "query_results": [{"query": query, "status": "SUCCESS", "object_count": 1} for query in queries],
        }


def test_detect_objects_dispatches_one_multi_object_result_into_context():
    context = {}
    step = PlanStep(
        step_id="1",
        skill_id="detect_objects",
        description="Localize three cubes",
        parameters={"queries": ["black cube", "red cube", "white cube"]},
    )

    result = execute_non_motion_skill(
        step,
        gripper_backend=None,
        context=context,
        perception_pipeline=PerceptionPipeline(_MultiObjectBackend()),
    )

    assert result["success"] is True
    assert len(result["objects"]) == 3
    assert len(context["localized_objects"]) == 3
    assert context["scene"].frame == "base_link"


def test_pick_object_after_unavailable_detection_is_blocked():
    context = {}
    detect_step = PlanStep(
        step_id="1",
        skill_id="detect_object",
        description="Detect red cube",
        parameters={"query": "red cube"},
    )
    execute_non_motion_skill(detect_step, gripper_backend=None, context=context)

    pick_step = PlanStep(
        step_id="2",
        skill_id="pick_object",
        description="Pick red cube",
        parameters={"query": "red cube"},
    )

    result = _verify_object_step_for_execution(
        step=pick_step,
        skill_library=SkillLibrary(),
        execution_context=context,
    )

    assert result is not None
    assert result["success"] is False
    assert result["skill_id"] == "pick_object"
    assert "no detected scene object" in result["message"]


def test_perception_audit_event_records_status_and_object_count(tmp_path):
    log_file = tmp_path / "audit.log"
    audit_logger = AuditLogger(log_file=str(log_file))
    step = PlanStep(
        step_id="1",
        skill_id="detect_object",
        description="Detect red cube",
        parameters={"query": "red cube"},
    )
    result = execute_non_motion_skill(step, gripper_backend=None, context={})

    _audit_perception_result(audit_logger, step, result)

    log_text = log_file.read_text()
    assert "perception_skill_executed" in log_text
    assert "NOT_AVAILABLE" in log_text
    assert "objects_count" in log_text


class _FakePerceptionBackend:
    def __init__(self, result):
        self.result = result

    def detect_object(self, query, min_confidence=0.5, require_pose=True):
        return self.result


class _RaisingPerceptionBackend:
    def detect_object(self, query, min_confidence=0.5, require_pose=True):
        raise RuntimeError("camera pipeline unavailable")


def test_detect_object_updates_scene_model_from_pipeline():
    scene_object = SceneObject(
        object_id="red cubes_1",
        label="red cubes",
        confidence=0.95,
        pose_available=True,
        pose=ObjectPose(frame="base_link", x=0.1, y=0.2, z=0.3),
        graspable=True,
        source="d435i_sam3",
    )
    scene = SceneModel(objects=[scene_object], frame="base_link", source="d435i_sam3")
    pipeline = PerceptionPipeline(
        _FakePerceptionBackend(
            DetectionResult(
                query="red cubes",
                status="SUCCESS",
                objects=[scene_object],
                message="Detected 1 object(s).",
                scene=scene,
            )
        )
    )
    context = {
        "require_base_link_pose_for_pick": True,
        "target_frame": "base_link",
    }
    step = PlanStep(
        step_id="1",
        skill_id="detect_object",
        description="Detect red cubes",
        parameters={"query": "red cubes"},
    )

    result = execute_non_motion_skill(
        step,
        gripper_backend=None,
        context=context,
        perception_pipeline=pipeline,
    )

    assert result["success"] is True
    assert result["status"] == "SUCCESS"
    assert context["scene"].objects[0].object_id == "red cubes_1"
    assert context["scene"].objects[0].pose.frame == "base_link"
    assert context["perception_available"] is True


def test_detect_object_failure_does_not_crash_plan_pipeline():
    pipeline = PerceptionPipeline(_RaisingPerceptionBackend())
    context = {}
    step = PlanStep(
        step_id="1",
        skill_id="detect_object",
        description="Detect red cube",
        parameters={"query": "red cube"},
    )

    result = execute_non_motion_skill(
        step,
        gripper_backend=None,
        context=context,
        perception_pipeline=pipeline,
    )

    assert result["success"] is False
    assert result["status"] == "FAILED"
    assert "camera pipeline unavailable" in result["message"]
    assert context["scene"].objects == []


class _ReadyRGBDCamera:
    def __init__(self):
        self.latest_depth_msg = {"image": [[1000]], "encoding": "16UC1"}
        self.latest_camera_info_msg = {"k": [100.0, 0, 0, 0, 100.0, 0, 0, 0, 1]}
        self.config = {"depth_frame": "camera_color_optical_frame"}
        self.node = None

    def _get_latest_depth_image(self):
        return self.latest_depth_msg


class _PointCloudBuilder:
    def build(self, scene_object, depth_frame, camera_info, frame=None):
        assert scene_object["object_id"] == "cube_1"
        assert depth_frame is not None
        assert camera_info is not None
        return {
            "success": True,
            "status": "SUCCESS",
            "message": "point cloud built",
            "point_cloud_path": "/tmp/cube_1.ply",
            "frame": frame,
        }


def test_build_object_point_cloud_is_dispatched_after_detection():
    pipeline = PerceptionPipeline()
    pipeline.backend = _ReadyRGBDCamera()
    context = {
        "selected_object": {"object_id": "cube_1", "confidence": 0.9},
        "resolved_object_id": "cube_1",
    }
    step = PlanStep(
        step_id="2",
        skill_id="build_object_point_cloud",
        description="Build point cloud",
        parameters={"object_id": "<resolved>"},
    )

    result = execute_non_motion_skill(
        step,
        gripper_backend=None,
        context=context,
        perception_pipeline=pipeline,
        point_cloud_builder=_PointCloudBuilder(),
    )

    assert result["success"] is True
    assert result["output"]["point_cloud_path"] == "/tmp/cube_1.ply"
    assert context["point_cloud"]["status"] == "SUCCESS"
