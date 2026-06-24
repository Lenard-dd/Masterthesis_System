import importlib

from llm_skill_robot.ros_hw_perception_gripper_demo import (
    ARM_BLOCK_MESSAGE,
    HardwareSafeDemoController,
)
from llm_skill_robot.safety.hardware_mode import load_hardware_mode


CANDIDATE = {
    "candidate_id": "g1",
    "score": 0.95,
    "gripper_width_m": 0.06,
    "pose": {
        "frame": "base_link",
        "position": {"x": 0.4, "y": 0.0, "z": 0.3},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    },
}


class FakeCameraBackend:
    config = {"depth_frame": "camera_color_optical_frame"}
    latest_camera_info_msg = object()

    def _get_latest_depth_image(self):
        return object()


class FakePerceptionPipeline:
    def __init__(self):
        self.backend = FakeCameraBackend()

    def detect_object(self, query, require_pose=False):
        return {
            "success": True,
            "status": "SUCCESS",
            "objects": [
                {"object_id": "cube_1", "label": query, "confidence": 0.9}
            ],
        }


class FakePointCloudBuilder:
    def __init__(self):
        self.calls = 0

    def build(self, *args, **kwargs):
        self.calls += 1
        return {
            "success": True,
            "status": "SUCCESS",
            "point_cloud_path": "/tmp/fake_object.ply",
            "num_points": 100,
            "frame": "base_link",
        }


class FakeGraspPlanner:
    def __init__(self):
        self.generate_calls = 0

    def generate_candidates(self, point_cloud_path):
        self.generate_calls += 1
        return {"success": True, "status": "SUCCESS", "candidates": [CANDIDATE]}

    def select_best_candidate(self, candidates):
        return max(candidates, key=lambda item: item["score"]) if candidates else None


class FakeGripperBackend:
    def __init__(self):
        self.commands = []

    def set_gripper_width(self, width_m, speed_mps, force_N, confirmed=False):
        self.commands.append((width_m, speed_mps, force_N, confirmed))
        return {
            "success": True,
            "status": "SENT",
            "message": "fake sent",
            "output": {"command_sent": True},
        }

    def get_gripper_state(self):
        return {"success": True, "output": {"feedback_available": False}}


class FakeMoveItBackend:
    execution_mode = "plan_only"

    def __init__(self):
        self.plan_calls = []
        self.execution_calls = []

    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        return {
            "success": True,
            "plan_id": f"plan_{len(self.plan_calls)}",
            "summary": {
                "success": True,
                "num_trajectory_points": 0,
                "trajectory_preview": {"dry_run": True},
            },
        }

    def get_plan_end_state(self, plan_id):
        return None

    def execute_cached_plan_simulated(self, *args, **kwargs):
        self.execution_calls.append((args, kwargs))
        raise AssertionError("unified demo attempted arm execution")


class FakeValidator:
    def validate_motion_plan_summary(self, summary):
        return {"decision": "ALLOW", "reasons": [], "warnings": []}


def _controller():
    return HardwareSafeDemoController(
        hardware_mode=load_hardware_mode(),
        perception_pipeline=FakePerceptionPipeline(),
        point_cloud_builder=FakePointCloudBuilder(),
        grasp_planner=FakeGraspPlanner(),
        gripper_backend=FakeGripperBackend(),
        moveit_backend=FakeMoveItBackend(),
        trajectory_validator=FakeValidator(),
        trajectory_visualizer=None,
        gripper_config={"default_speed_mps": 0.02, "default_force_N": 10.0},
    )


def test_module_imports_without_ros_initialization():
    module = importlib.import_module("llm_skill_robot.ros_hw_perception_gripper_demo")
    assert callable(module.main)


def test_state_machine_blocks_out_of_order_steps():
    controller = _controller()

    assert controller.build_object_point_cloud()["success"] is False
    assert controller.generate_grasp_candidates()["success"] is False
    assert controller.review_best_candidate("YES")["success"] is False
    assert controller.test_real_gripper_width("YES")["success"] is False
    assert controller.preview_moveit_plan_only()["success"] is False
    assert controller.point_cloud_builder.calls == 0
    assert controller.grasp_planner.generate_calls == 0
    assert controller.gripper_backend.commands == []
    assert controller.moveit_backend.plan_calls == []


def test_gripper_requires_exact_yes_confirmation():
    controller = _controller()
    assert controller.detect_object("cube")["success"] is True
    assert controller.build_object_point_cloud()["success"] is True
    assert controller.generate_grasp_candidates()["success"] is True
    assert controller.review_best_candidate("YES")["success"] is True

    result = controller.test_real_gripper_width("yes")

    assert result["status"] == "CANCELLED"
    assert controller.gripper_backend.commands == []


def test_exact_yes_can_reach_fake_gripper_only_after_review():
    controller = _controller()
    controller.detect_object("cube")
    controller.build_object_point_cloud()
    controller.generate_grasp_candidates()
    controller.review_best_candidate("YES")

    result = controller.test_real_gripper_width("YES")

    assert result["success"] is True
    assert controller.gripper_backend.commands == [(0.06, 0.02, 10.0, True)]


def test_plan_preview_and_arm_block_never_call_execution():
    controller = _controller()
    controller.detect_object("cube")
    controller.build_object_point_cloud()
    controller.generate_grasp_candidates()
    controller.review_best_candidate("YES")

    preview = controller.preview_moveit_plan_only()
    blocked = controller.request_arm_execution()

    assert preview["success"] is True
    assert len(controller.moveit_backend.plan_calls) == 3
    assert blocked["message"] == ARM_BLOCK_MESSAGE
    assert controller.moveit_backend.execution_calls == []
