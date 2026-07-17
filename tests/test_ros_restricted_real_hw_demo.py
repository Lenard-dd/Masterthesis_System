import importlib

from llm_skill_robot.ros_restricted_real_hw_demo import (
    RestrictedRealHardwareDemoController,
    grasp_preference_score,
    menu_text,
)
from llm_skill_robot.safety.hardware_mode import load_hardware_mode
from llm_skill_robot.safety.real_arm_safety import load_real_arm_safety
from llm_skill_robot.safety.tabletop_safety import TabletopSafety


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
            "point_cloud_path": "/tmp/restricted_demo_object.ply",
            "frame": "base_link",
            "centroid_m": {"x": 0.4, "y": 0.0, "z": 0.25},
        }


def _candidate(
    width=0.07,
    candidate_id="g1",
    score=0.9,
    approach_vector=None,
    closing_axis=None,
):
    return {
        "candidate_id": candidate_id,
        "valid": True,
        "score": score,
        "pose_frame": "base_link",
        "pose": {
            "frame": "base_link",
            "position": {"x": 0.4, "y": 0.0, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "approach_vector": approach_vector or [0.0, 0.0, 1.0],
        "closing_axis": closing_axis or [1.0, 0.0, 0.0],
        "gripper_width_m": width,
    }


class FakeGraspPlanner:
    def __init__(self, candidates=None):
        self.candidates = candidates

    def generate_candidates(self, point_cloud_path):
        return {"success": True, "candidates": self.candidates or [_candidate()]}


class FakeMoveItBackend:
    def __init__(self):
        self.plan_calls = []
        self.execute_calls = []

    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        return {
            "success": True,
            "plan_id": "plan_pregrasp",
            "summary": {
                "plan_id": "plan_pregrasp",
                "success": True,
                "num_trajectory_points": 2,
                "max_joint_delta_rad": 0.1,
                "total_joint_delta_rad": 0.2,
                "max_cartesian_step_m": 0.02,
                "max_velocity_abs": 0.05,
                "duration_sec": 1.0,
            },
        }

    def plan_to_named_target(self, target_name, **kwargs):
        self.plan_calls.append({"target_name": target_name, **kwargs})
        return {
            "success": True,
            "plan_id": f"plan_{target_name}",
            "summary": {
                "plan_id": f"plan_{target_name}",
                "success": True,
                "num_trajectory_points": 2,
                "max_joint_delta_rad": 0.1,
                "total_joint_delta_rad": 0.2,
                "max_cartesian_step_m": 0.02,
                "max_velocity_abs": 0.05,
                "duration_sec": 1.0,
            },
        }

    def execute_cached_plan_real(self, plan_id, final_confirmed=False):
        self.execute_calls.append((plan_id, final_confirmed))
        return {"success": True, "message": "executed", "output": {}}


class FakeGripperBackend:
    def __init__(self):
        self.open_calls = []
        self.width_calls = []
        self.close_calls = []

    def open_gripper(self, **kwargs):
        self.open_calls.append(kwargs)
        return {"success": True, "message": "opened", "output": {"command_sent": True}}

    def set_gripper_width(self, **kwargs):
        self.width_calls.append(kwargs)
        return {"success": True, "message": "width set", "output": {"command_sent": True}}

    def get_gripper_state(self):
        return {"success": True, "message": "state ok", "output": {"current_width_m": 0.08}}

    def close_for_grasp(self, **kwargs):
        self.close_calls.append(kwargs)
        return {"success": True, "message": "closed", "output": {"command_sent": True}}


def _controller(candidates=None, grasp_preferences=None):
    moveit = FakeMoveItBackend()
    gripper = FakeGripperBackend()
    candidate_screening = {
        "grasp_preferences": grasp_preferences
        if grasp_preferences is not None
        else {
            "enabled": True,
            "prefer_top_down": True,
            "max_tilt_deg": 30.0,
            "prefer_face_aligned_yaw": True,
            "max_face_yaw_deviation_deg": 25.0,
        }
    }
    controller = RestrictedRealHardwareDemoController(
        hardware_mode=load_hardware_mode(),
        real_arm_safety=load_real_arm_safety(),
        perception_pipeline=FakePerceptionPipeline(),
        point_cloud_builder=FakePointCloudBuilder(),
        grasp_planner=FakeGraspPlanner(candidates),
        moveit_backend=moveit,
        gripper_backend=gripper,
        grasping_config={
            "plan_only_preview": {
                "pregrasp_distance_m": 0.05,
                "velocity_scale": 0.05,
                "acceleration_scale": 0.05,
                "moveit": {
                    "pipeline_id": "pilz_industrial_motion_planner",
                    "planner_id": "PTP",
                    "stage_planner_ids": {"grasp": "LIN", "retreat": "LIN"},
                },
                "candidate_screening": candidate_screening,
            }
        },
        gripper_config={"default_speed_mps": 0.02, "default_force_N": 10.0},
    )
    return controller, moveit, gripper


def _prepare_reviewed_controller():
    controller, moveit, gripper = _controller()
    assert controller.detect_object("red cube").success
    assert controller.build_object_point_cloud().success
    assert controller.generate_grasp_candidates().success
    assert controller.review_selected_grasp_candidate().success
    return controller, moveit, gripper


def test_module_imports_without_ros_initialization():
    module = importlib.import_module("llm_skill_robot.ros_restricted_real_hw_demo")

    assert callable(module.main)


def test_state_machine_blocks_out_of_order_steps():
    controller, _moveit, _gripper = _controller()

    assert controller.build_object_point_cloud().status == "BLOCKED"
    assert controller.generate_grasp_candidates().status == "BLOCKED"
    assert controller.review_selected_grasp_candidate().status == "BLOCKED"
    assert controller.set_real_gripper_to_candidate_width("YES").status == "BLOCKED"


def test_menu_includes_supervised_grasp_contact_and_retreat_items():
    lowered = menu_text().lower()

    for term in ("approach grasp", "close real gripper", "retreat with grasped object"):
        assert term in lowered


def test_pregrasp_requires_grasp_plan():
    controller, moveit, _gripper = _controller()

    result = controller.move_to_pregrasp("EXECUTE")

    assert result.status == "BLOCKED"
    assert "GraspPlan" in result.message
    assert moveit.execute_calls == []


def test_real_arm_confirmation_required():
    controller, moveit, _gripper = _prepare_reviewed_controller()

    result = controller.move_to_pregrasp("NO")

    assert result.status == "CANCELLED"
    assert moveit.execute_calls == []

    result = controller.move_to_pregrasp("EXECUTE")

    assert result.status == "SUCCESS"
    assert moveit.execute_calls == [("plan_pregrasp", True)]


def test_real_gripper_confirmation_required():
    controller, _moveit, gripper = _prepare_reviewed_controller()

    result = controller.open_real_gripper("NO")

    assert result.status == "CANCELLED"
    assert gripper.open_calls == []

    result = controller.open_real_gripper("YES")

    assert result.status == "SUCCESS"
    assert len(gripper.open_calls) == 1

    result = controller.set_real_gripper_to_candidate_width("NO")

    assert result.status == "CANCELLED"
    assert gripper.width_calls == []

    result = controller.set_real_gripper_to_candidate_width("YES")

    assert result.status == "SUCCESS"
    assert gripper.width_calls[0]["width_m"] == 0.07


def test_contact_close_requires_reaching_grasp_pose_and_yes_confirmation():
    controller, _moveit, gripper = _prepare_reviewed_controller()

    assert controller.close_real_gripper_for_grasp("YES").status == "BLOCKED"
    controller.state.at_grasp_pose = True
    assert controller.close_real_gripper_for_grasp("NO").status == "CANCELLED"
    assert gripper.close_calls == []
    assert controller.close_real_gripper_for_grasp("YES").status == "SUCCESS"
    assert controller.state.gripper_closed is True


def test_approach_close_and_retreat_are_single_confirmed_steps():
    controller, moveit, _gripper = _prepare_reviewed_controller()
    controller.state.grasp_plan = {
        "pregrasp_pose": _candidate()["pose"],
        "grasp_pose": {
            **_candidate()["pose"],
            "position": {"x": 0.4, "y": 0.0, "z": 0.25},
        },
        "retreat_pose": {
            **_candidate()["pose"],
            "position": {"x": 0.4, "y": 0.0, "z": 0.30},
        },
        "grasp_score": 0.9,
        "candidate_id": "g1",
    }
    assert controller.move_to_pregrasp("EXECUTE").success
    assert controller.approach_grasp("EXECUTE", trajectory_review_approved=True).success
    assert controller.close_real_gripper_for_grasp("YES").success
    assert controller.retreat_grasp("EXECUTE", trajectory_review_approved=True).success
    assert moveit.execute_calls == [
        ("plan_pregrasp", True),
        ("plan_pregrasp", True),
        ("plan_pregrasp", True),
    ]


def test_approach_and_retreat_use_configured_pilz_lin_stage_settings():
    controller, _moveit, _gripper = _prepare_reviewed_controller()

    assert controller.grasp_motion.planning_kwargs_by_skill["approach_grasp"].items() >= {
        "pipeline_id": "pilz_industrial_motion_planner",
        "planner_id": "LIN",
    }.items()
    assert controller.grasp_motion.planning_kwargs_by_skill["retreat_grasp"] == {
        "pipeline_id": "pilz_industrial_motion_planner",
        "planner_id": "LIN",
    }


def test_tabletop_backoff_moves_grasp_and_retreat_toward_pregrasp():
    controller, _moveit, _gripper = _controller()
    controller.tabletop_safety = TabletopSafety(grasp_contact_backoff_m=0.02)
    controller.state.point_cloud = {"frame": "base_link"}

    plan = controller._build_full_grasp_plan(_candidate())

    assert round(plan["pregrasp_pose"]["position"]["z"], 6) == 0.25
    assert round(plan["grasp_pose"]["position"]["z"], 6) == 0.28
    assert round(plan["retreat_pose"]["position"]["z"], 6) == 0.38


def test_top_down_face_aligned_candidate_is_preferred_over_higher_score_diagonal():
    diagonal = _candidate(
        candidate_id="edge_high_score",
        score=0.99,
        approach_vector=[0.0, 0.0, 1.0],
        closing_axis=[1.0, 1.0, 0.0],
    )
    face_aligned = _candidate(
        candidate_id="face_lower_score",
        score=0.80,
        approach_vector=[0.0, 0.0, 1.0],
        closing_axis=[1.0, 0.0, 0.0],
    )
    controller, _moveit, _gripper = _controller([diagonal, face_aligned])
    assert controller.detect_object("red cube").success
    assert controller.build_object_point_cloud().success
    assert controller.generate_grasp_candidates().success

    result = controller.review_selected_grasp_candidate()

    assert result.success is True
    assert controller.state_summary()["selected_candidate_id"] == "face_lower_score"


def test_grasp_preference_rejects_excessive_tilt():
    tilted = _candidate(
        candidate_id="tilted",
        approach_vector=[1.0, 0.0, 0.0],
        closing_axis=[1.0, 0.0, 0.0],
    )

    result = grasp_preference_score(
        tilted,
        {
            "prefer_top_down": True,
            "max_tilt_deg": 30.0,
            "prefer_face_aligned_yaw": False,
        },
    )

    assert result["allowed"] is False
    assert "top_down tilt" in result["reason"]


def test_grasp_preference_rejects_edge_facing_yaw():
    edge_facing = _candidate(
        candidate_id="edge",
        approach_vector=[0.0, 0.0, 1.0],
        closing_axis=[1.0, 1.0, 0.0],
    )

    result = grasp_preference_score(
        edge_facing,
        {
            "prefer_top_down": False,
            "prefer_face_aligned_yaw": True,
            "max_face_yaw_deviation_deg": 25.0,
        },
    )

    assert result["allowed"] is False
    assert "face yaw deviation" in result["reason"]


def test_grasp_preference_uses_object_reference_axes_for_rotated_cube():
    face_aligned_to_world_but_edge_aligned_to_object = _candidate(
        candidate_id="world_axis_edge",
        approach_vector=[0.0, 0.0, 1.0],
        closing_axis=[1.0, 0.0, 0.0],
    )
    object_axis = [0.70710678, 0.70710678, 0.0]

    result = grasp_preference_score(
        face_aligned_to_world_but_edge_aligned_to_object,
        {
            "prefer_top_down": False,
            "prefer_face_aligned_yaw": True,
            "max_face_yaw_deviation_deg": 25.0,
            "face_reference_axes": [
                object_axis,
                [-object_axis[1], object_axis[0], 0.0],
            ],
        },
    )

    assert result["allowed"] is False
    assert "face yaw deviation" in result["reason"]

    corrected = _candidate(
        candidate_id="object_face",
        approach_vector=[0.0, 0.0, 1.0],
        closing_axis=object_axis,
    )
    result = grasp_preference_score(
        corrected,
        {
            "prefer_top_down": False,
            "prefer_face_aligned_yaw": True,
            "max_face_yaw_deviation_deg": 25.0,
            "face_reference_axes": [
                object_axis,
                [-object_axis[1], object_axis[0], 0.0],
            ],
        },
    )

    assert result["allowed"] is True


def test_current_gripper_yaw_preference_prefers_minimal_rotation():
    turn_90_high_score = _candidate(
        candidate_id="turn_90_high_score",
        score=0.99,
        approach_vector=[0.0, 0.0, 1.0],
        closing_axis=[0.0, 1.0, 0.0],
    )
    current_yaw_low_score = _candidate(
        candidate_id="current_yaw_low_score",
        score=0.75,
        approach_vector=[0.0, 0.0, 1.0],
        closing_axis=[1.0, 0.0, 0.0],
    )
    controller, _moveit, _gripper = _controller(
        [turn_90_high_score, current_yaw_low_score],
        grasp_preferences={
            "enabled": True,
            "prefer_top_down": True,
            "prefer_face_aligned_yaw": False,
            "prefer_current_gripper_yaw": True,
            "current_gripper_closing_axis": [1.0, 0.0, 0.0],
            "current_yaw_weight": 2.0,
            "max_preferred_yaw_change_deg": 90.0,
        },
    )
    assert controller.detect_object("red cube").success
    assert controller.build_object_point_cloud().success
    assert controller.generate_grasp_candidates().success

    result = controller.review_selected_grasp_candidate()

    assert result.success is True
    assert controller.state_summary()["selected_candidate_id"] == "current_yaw_low_score"


def test_current_gripper_yaw_preference_scores_yaw_change():
    aligned = _candidate(closing_axis=[1.0, 0.0, 0.0])
    rotated = _candidate(closing_axis=[0.0, 1.0, 0.0])
    config = {
        "prefer_top_down": False,
        "prefer_face_aligned_yaw": False,
        "prefer_current_gripper_yaw": True,
        "current_gripper_closing_axis": [1.0, 0.0, 0.0],
        "max_preferred_yaw_change_deg": 90.0,
    }

    aligned_score = grasp_preference_score(aligned, config)
    rotated_score = grasp_preference_score(rotated, config)

    assert aligned_score["score"] > rotated_score["score"]
    assert "current gripper yaw change 0.0 deg" in aligned_score["reason"]
    assert "current gripper yaw change 90.0 deg" in rotated_score["reason"]
