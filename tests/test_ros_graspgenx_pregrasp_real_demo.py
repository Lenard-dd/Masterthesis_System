import importlib
from dataclasses import replace

from llm_skill_robot.ros_graspgenx_pregrasp_real_demo import (
    PregraspRealDemoController,
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


class FakeGraspPlanner:
    def generate_candidates(self, point_cloud_path):
        return {
            "success": True,
            "candidates": [
                {
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
            ],
        }

    def select_best_candidate(self, candidates):
        return candidates[0]


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
        self.close_calls = []

    def open_gripper(self, **kwargs):
        self.open_calls.append(kwargs)
        return {"success": True, "message": "opened"}

    def close_gripper(self, **kwargs):
        self.close_calls.append(kwargs)
        raise AssertionError("close_gripper must not be called")


def _controller():
    hardware_mode = load_hardware_mode()
    safety = load_real_arm_safety()
    safety = replace(
        safety,
        gripper={**safety.gripper, "allow_open_gripper": True},
    )
    moveit = FakeMoveItBackend()
    gripper = FakeGripperBackend()
    controller = PregraspRealDemoController(
        hardware_mode=hardware_mode,
        real_arm_safety=safety,
        perception_pipeline=FakePerceptionPipeline(),
        point_cloud_builder=FakePointCloudBuilder(),
        grasp_planner=FakeGraspPlanner(),
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
    module = importlib.import_module("llm_skill_robot.ros_graspgenx_pregrasp_real_demo")

    assert callable(module.main)


def test_fake_pipeline_can_reach_pregrasp_planning_stage():
    controller, moveit, gripper = _controller()

    result = controller.run_once(
        "cube",
        open_gripper_confirmation="",
        execute_confirmation="no",
    )

    assert result.status == "CANCELLED"
    assert len(moveit.plan_calls) == 1
    assert moveit.execute_calls == []
    assert gripper.close_calls == []


def test_execution_blocked_without_execute():
    controller, moveit, _gripper = _controller()

    result = controller.run_once("cube", execute_confirmation="YES")

    assert result.status == "CANCELLED"
    assert moveit.execute_calls == []


def test_execute_reaches_pregrasp_only_with_execute():
    controller, moveit, _gripper = _controller()

    result = controller.run_once("cube", execute_confirmation="EXECUTE")

    assert result.success is True
    assert moveit.execute_calls == [("plan_pregrasp", True)]


def test_approach_grasp_is_never_called():
    controller, moveit, _gripper = _controller()

    controller.run_once("cube", execute_confirmation="EXECUTE")

    assert moveit.approach_calls == []


def test_close_gripper_is_never_called():
    controller, _moveit, gripper = _controller()

    controller.run_once(
        "cube",
        open_gripper_confirmation="YES",
        execute_confirmation="EXECUTE",
    )

    assert len(gripper.open_calls) == 1
    assert gripper.close_calls == []
