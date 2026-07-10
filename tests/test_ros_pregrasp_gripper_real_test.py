import importlib

from llm_skill_robot.ros_pregrasp_gripper_real_test import (
    PregraspGripperRealTestController,
)
from llm_skill_robot.safety.hardware_mode import load_hardware_mode
from llm_skill_robot.safety.real_arm_safety import load_real_arm_safety


class FakeCameraBackend:
    config = {"depth_frame": "base_link"}
    latest_camera_info_msg = object()

    def _get_latest_depth_image(self):
        return object()


class FakePerceptionPipeline:
    backend = FakeCameraBackend()

    def detect_object(self, query, require_pose=False):
        return {
            "success": True,
            "objects": [
                {"object_id": "cube_1", "label": query, "confidence": 0.9}
            ],
        }


class FakePointCloudBuilder:
    def build(self, *args, **kwargs):
        return {
            "success": True,
            "point_cloud_path": "/tmp/fake_object.ply",
            "frame": "base_link",
            "centroid_m": {"x": 0.4, "y": 0.0, "z": 0.25},
        }


def _candidate(width=0.06):
    data = {
        "candidate_id": "g1",
        "valid": True,
        "score": 0.9,
        "pose_frame": "base_link",
        "pose": {
            "frame": "base_link",
            "position": {"x": 0.4, "y": 0.0, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "approach_vector": [0.0, 0.0, 1.0],
    }
    if width is not None:
        data["gripper_width_m"] = width
    return data


class FakeGraspPlanner:
    def __init__(self, candidate=None):
        self.candidate = candidate if candidate is not None else _candidate()

    def generate_candidates(self, point_cloud_path):
        candidates = self.candidate if isinstance(self.candidate, list) else [self.candidate]
        return {"success": True, "candidates": candidates}

    def select_best_candidate(self, candidates):
        return candidates[0] if candidates else None


class FakeMoveItBackend:
    def __init__(self):
        self.plan_calls = []
        self.execute_calls = []
        self.approach_calls = []

    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        return {
            "success": True,
            "plan_id": "plan_pregrasp",
            "summary": {
                "plan_id": "plan_pregrasp",
                "max_joint_delta_rad": 0.1,
                "total_joint_delta_rad": 0.2,
            },
        }

    def execute_cached_plan_real(self, plan_id, final_confirmed=False):
        self.execute_calls.append((plan_id, final_confirmed))
        return {"success": True, "message": "executed"}

    def approach_grasp(self, *args, **kwargs):
        self.approach_calls.append((args, kwargs))
        raise AssertionError("approach_grasp must not be called")


class FakeGripperBackend:
    def __init__(self):
        self.open_calls = []
        self.width_calls = []
        self.close_calls = []

    def open_gripper(self, **kwargs):
        self.open_calls.append(kwargs)
        return {"success": True, "message": "opened"}

    def set_gripper_width(self, **kwargs):
        self.width_calls.append(kwargs)
        return {"success": True, "message": "width set"}

    def close_gripper(self, **kwargs):
        self.close_calls.append(kwargs)
        raise AssertionError("close_gripper must not be called")


def _controller(candidate=None):
    moveit = FakeMoveItBackend()
    gripper = FakeGripperBackend()
    controller = PregraspGripperRealTestController(
        hardware_mode=load_hardware_mode(),
        real_arm_safety=load_real_arm_safety(),
        perception_pipeline=FakePerceptionPipeline(),
        point_cloud_builder=FakePointCloudBuilder(),
        grasp_planner=FakeGraspPlanner(candidate),
        moveit_backend=moveit,
        gripper_backend=gripper,
        grasping_config={
            "plan_only_preview": {
                "pregrasp_distance_m": 0.1,
                "velocity_scale": 0.05,
                "acceleration_scale": 0.05,
            }
        },
        gripper_config={"default_speed_mps": 0.02, "default_force_N": 10.0},
    )
    return controller, moveit, gripper


def test_module_imports_without_ros_initialization():
    module = importlib.import_module("llm_skill_robot.ros_pregrasp_gripper_real_test")

    assert callable(module.main)


def test_missing_candidate_width_stops():
    controller, moveit, gripper = _controller(_candidate(width=None))

    result = controller.run_once("cube")

    assert result.status == "BLOCKED"
    assert "gripper_width_m" in result.message
    assert moveit.plan_calls == []
    assert gripper.open_calls == []


def test_unsafe_width_rejected():
    controller, moveit, gripper = _controller(_candidate(width=0.02))

    result = controller.run_once("cube")

    assert result.status == "BLOCKED"
    assert "safe_close_width_m" in result.message
    assert moveit.plan_calls == []
    assert gripper.width_calls == []


def test_bad_high_score_candidate_is_skipped_for_safe_pregrasp_width():
    bad = _candidate(width=0.08)
    bad["candidate_id"] = "bad_high_score"
    bad["score"] = 0.99
    bad["pose"]["position"]["z"] = -0.3
    good = _candidate(width=0.07)
    good["candidate_id"] = "good_lower_score"
    good["score"] = 0.8
    controller, moveit, gripper = _controller([bad, good])

    result = controller.run_once(
        "cube",
        open_confirmation="NO",
        execute_confirmation="EXECUTE",
        set_width_confirmation="YES",
    )

    assert result.success is True
    assert result.width_m == 0.07
    assert moveit.execute_calls == [("plan_pregrasp", True)]
    assert gripper.width_calls[0]["width_m"] == 0.07


def test_no_close_gripper_call():
    controller, _moveit, gripper = _controller()

    controller.run_once(
        "cube",
        open_confirmation="YES",
        execute_confirmation="EXECUTE",
        set_width_confirmation="YES",
    )

    assert gripper.close_calls == []


def test_selected_candidate_prints_axes(capsys):
    from llm_skill_robot.ros_pregrasp_gripper_real_test import print_selected_candidate

    print_selected_candidate(_candidate())

    output = capsys.readouterr().out
    assert "approach_vector:" in output
    assert "closing_axis:" in output


def test_no_approach_grasp_call():
    controller, moveit, _gripper = _controller()

    controller.run_once(
        "cube",
        execute_confirmation="EXECUTE",
        set_width_confirmation="YES",
    )

    assert moveit.approach_calls == []


def test_arm_execution_requires_execute():
    controller, moveit, gripper = _controller()

    result = controller.run_once(
        "cube",
        open_confirmation="YES",
        execute_confirmation="NO",
        set_width_confirmation="YES",
    )

    assert result.status == "CANCELLED"
    assert moveit.execute_calls == []
    assert gripper.width_calls == []


def test_gripper_commands_require_yes():
    controller, moveit, gripper = _controller()

    result = controller.run_once(
        "cube",
        open_confirmation="NO",
        execute_confirmation="EXECUTE",
        set_width_confirmation="NO",
    )

    assert result.status == "ARM_ONLY"
    assert moveit.execute_calls == [("plan_pregrasp", True)]
    assert gripper.open_calls == []
    assert gripper.width_calls == []
