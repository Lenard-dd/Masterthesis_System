import copy
import importlib
import math
from dataclasses import replace

import pytest

from llm_skill_robot.ros_graspgenx_plan_only_demo import (
    apply_grasp_frame_to_ee_transform,
    attempt_real_arm_execution,
    assert_plan_only_mode,
    camera_message_status,
    create_preview_poses,
    describe_grasp_frame_to_ee_transform,
    diagnose_pregrasp_planning,
    localize_graspgenx_candidate,
    moveit_planning_kwargs,
    moveit_stage_planning_kwargs,
    plan_pose_sequence,
    planning_variant_kwargs,
    prepare_candidate_preview,
    screen_grasp_candidates_for_plan_only,
    transform_pose_sequence_to_target,
    wait_for_camera_messages,
)
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.safety.hardware_mode import load_hardware_mode


FAKE_CANDIDATE = {
    "candidate_id": "grasp_1",
    "score": 0.93,
    "gripper_width_m": 0.06,
    "pose": {
        "frame": "base_link",
        "position": {"x": 0.4, "y": 0.1, "z": 0.3},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    },
}


class AllowValidator:
    def validate_motion_plan_summary(self, summary):
        return {"decision": "ALLOW", "reasons": [], "warnings": []}


class FakePlanOnlyBackend:
    def __init__(self):
        self.plan_calls = []
        self.execution_calls = []

    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        plan_id = f"plan_{len(self.plan_calls)}"
        return {
            "success": True,
            "message": "fake plan only",
            "plan_id": plan_id,
            "summary": {
                "success": True,
                "num_trajectory_points": 2,
                "trajectory_preview": {},
            },
        }

    def get_plan_end_state(self, plan_id):
        return {"fake_start_after": plan_id}

    def execute_cached_plan_simulated(self, *args, **kwargs):
        self.execution_calls.append((args, kwargs))
        raise AssertionError("HW4 attempted trajectory execution")


class FakeReachabilityBackend(FakePlanOnlyBackend):
    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        reachable = float(kwargs["position"]["x"]) >= 0.5
        plan_id = f"screen_plan_{len(self.plan_calls)}"
        return {
            "success": reachable,
            "message": "reachable" if reachable else "unreachable",
            "plan_id": plan_id,
            "summary": {
                "success": reachable,
                "num_trajectory_points": 4 if reachable else 0,
                "max_joint_delta_rad": 0.2 if reachable else None,
                "trajectory_preview": {},
            },
        }


def test_sparse_candidate_keeps_full_orientation_even_when_relaxation_is_configured():
    backend = FakePlanOnlyBackend()
    candidate = copy.deepcopy(FAKE_CANDIDATE)
    candidate["metadata"] = {"generation_branch": "obb_sparse"}
    result = screen_grasp_candidates_for_plan_only(
        candidates=[candidate], point_cloud={},
        preview_config={"candidate_screening": {
            "variants": ["relaxed_orientation", "position_only"],
            "full_orientation_tolerance_rad": 0.05,
        }},
        target_frame="base_link", transformer=None, backend=backend,
        trajectory_validator=AllowValidator(),
    )
    assert result["variant"] == "full_pose"
    assert len(backend.plan_calls) == 3
    assert all(call["orientation_tolerance"] == 0.05 for call in backend.plan_calls)


def test_sparse_symmetry_uses_nearest_pose_without_planning_opposite_fallback():
    backend = FakePlanOnlyBackend()
    candidate = copy.deepcopy(FAKE_CANDIDATE)
    candidate.update({
        "closing_axis": [-1.0, 0.0, 0.0],
        "approach_vector": [0.0, 0.0, 1.0],
        "metadata": {"generation_branch": "obb_sparse"},
    })
    candidate["pose"]["orientation"] = {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}

    result = screen_grasp_candidates_for_plan_only(
        candidates=[candidate], point_cloud={"frame": "base_link"},
        preview_config={"candidate_screening": {
            "variants": ["full_pose"],
            "grasp_preferences": {
                "current_gripper_closing_axis": [1.0, 0.0, 0.0],
                "parallel_jaw_symmetry_enabled": True,
            },
        }},
        target_frame="base_link", transformer=None, backend=backend,
        trajectory_validator=AllowValidator(),
    )

    assert result["selected"]["closing_axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert result["selected"]["metadata"]["parallel_jaw_symmetry_role"] == "preferred"
    assert len(backend.plan_calls) == 3


class FakeSequenceReachabilityBackend(FakePlanOnlyBackend):
    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        is_bad_grasp = (
            kwargs.get("skill_id") == "plan_grasp_pose"
            and float(kwargs["position"]["y"]) > 0.5
        )
        success = not is_bad_grasp
        plan_id = f"sequence_plan_{len(self.plan_calls)}"
        return {
            "success": success,
            "message": "fake sequence plan" if success else "fake grasp IK failure",
            "plan_id": plan_id,
            "summary": {
                "success": success,
                "num_trajectory_points": 4 if success else 0,
                "max_joint_delta_rad": 0.2 if success else None,
                "trajectory_preview": {},
            },
        }


class FakeWristBranchBackend(FakePlanOnlyBackend):
    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        # The canonical pose deliberately represents a poor IK branch here;
        # its 180-degree equivalent represents the short wrist solution.
        wrist_delta = (
            3.0 if abs(float(kwargs["orientation"].get("w", 1.0))) > 0.5 else 0.1
        )
        plan_id = f"wrist_plan_{len(self.plan_calls)}"
        return {
            "success": True,
            "message": "fake wrist branch",
            "plan_id": plan_id,
            "summary": {
                "success": True,
                "num_trajectory_points": 2,
                "max_joint_delta_rad": wrist_delta,
                "joint_names": ["wrist_3_joint"],
                "trajectory_preview": {
                    "first_point_positions": [0.0],
                    "last_point_positions": [wrist_delta],
                },
            },
        }


def test_large_wrist_rotation_plans_and_selects_symmetry_fallback():
    backend = FakeWristBranchBackend()
    candidate = copy.deepcopy(FAKE_CANDIDATE)
    candidate.update({
        "closing_axis": [-1.0, 0.0, 0.0],
        "approach_vector": [0.0, 0.0, 1.0],
        "metadata": {
            "generation_branch": "diffusion",
            "gripper_name": "robotiq_2f_140",
        },
    })
    candidate["pose"]["orientation"] = {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}

    result = screen_grasp_candidates_for_plan_only(
        candidates=[candidate], point_cloud={"frame": "base_link"},
        preview_config={"candidate_screening": {
            "variants": ["full_pose"],
            "grasp_preferences": {
                "current_gripper_closing_axis": [1.0, 0.0, 0.0],
                "parallel_jaw_symmetry_enabled": True,
                "symmetry_replan_wrist_threshold_rad": 2.0,
            },
        }},
        target_frame="base_link", transformer=None, backend=backend,
        trajectory_validator=AllowValidator(),
    )

    assert result["selected"]["candidate_id"] == "grasp_1__sym180"
    assert result["attempt"]["wrist_3_joint_delta_rad"] == pytest.approx(0.3)
    assert len(backend.plan_calls) == 6


class FakeCameraBackend:
    config = {"wait_timeout_sec": 0.1}

    def __init__(self, rgb=None, depth=None, camera_info=None):
        self.latest_rgb_msg = rgb
        self.latest_depth_msg = depth
        self.latest_camera_info_msg = camera_info


class FakeRclpy:
    def __init__(self):
        self.spin_calls = 0

    def spin_once(self, node, timeout_sec=0.0):
        self.spin_calls += 1


class FakePoseTransformer:
    available = True

    def transform_pose_dict(self, pose):
        transformed = dict(pose)
        transformed["frame"] = "base_link"
        transformed["position"] = {
            "x": pose["position"]["x"] + 1.0,
            "y": pose["position"]["y"] + 2.0,
            "z": pose["position"]["z"] + 3.0,
        }
        return transformed


def test_module_imports_without_ros_initialization():
    module = importlib.import_module("llm_skill_robot.ros_graspgenx_plan_only_demo")
    assert callable(module.main)


def test_fake_planner_reaches_all_plan_only_stages():
    backend = FakePlanOnlyBackend()
    poses = create_preview_poses(FAKE_CANDIDATE)

    result = plan_pose_sequence(backend, poses, AllowValidator())

    assert result["success"] is True
    assert [stage["stage"] for stage in result["stages"]] == [
        "pregrasp",
        "grasp",
        "retreat",
    ]
    assert len(backend.plan_calls) == 3
    assert backend.plan_calls[0]["start_state"] is None
    assert backend.plan_calls[1]["start_state"] == {"fake_start_after": "plan_1"}


def test_plan_pose_sequence_forwards_selected_planning_variant_kwargs():
    backend = FakePlanOnlyBackend()
    poses = create_preview_poses(FAKE_CANDIDATE)

    result = plan_pose_sequence(
        backend,
        poses,
        AllowValidator(),
        plan_kwargs={"use_orientation_constraint": False, "position_tolerance": 0.02},
    )

    assert result["success"] is True
    assert backend.plan_calls[0]["use_orientation_constraint"] is False
    assert backend.plan_calls[0]["position_tolerance"] == 0.02


def test_plan_pose_sequence_applies_stage_planner_kwargs():
    backend = FakePlanOnlyBackend()
    poses = create_preview_poses(FAKE_CANDIDATE)

    result = plan_pose_sequence(
        backend,
        poses,
        AllowValidator(),
        plan_kwargs={"pipeline_id": "pilz_industrial_motion_planner"},
        stage_plan_kwargs={
            "pregrasp": {"planner_id": "PTP"},
            "grasp": {"planner_id": "LIN"},
            "retreat": {"planner_id": "LIN"},
        },
    )

    assert result["success"] is True
    assert backend.plan_calls[0]["pipeline_id"] == "pilz_industrial_motion_planner"
    assert backend.plan_calls[0]["planner_id"] == "PTP"
    assert backend.plan_calls[1]["planner_id"] == "LIN"
    assert backend.plan_calls[2]["planner_id"] == "LIN"


def test_plan_only_sequence_never_calls_execution_function():
    backend = FakePlanOnlyBackend()

    result = plan_pose_sequence(
        backend, create_preview_poses(FAKE_CANDIDATE), AllowValidator()
    )

    assert result["success"] is True
    assert backend.execution_calls == []


def test_plan_only_candidate_screening_selects_reachable_candidate():
    high_score_unreachable = {
        **FAKE_CANDIDATE,
        "candidate_id": "high_score_bad",
        "score": 0.99,
        "pose": {
            **FAKE_CANDIDATE["pose"],
            "position": {"x": 0.4, "y": 0.1, "z": 0.3},
        },
    }
    lower_score_reachable = {
        **FAKE_CANDIDATE,
        "candidate_id": "lower_score_good",
        "score": 0.75,
        "pose": {
            **FAKE_CANDIDATE["pose"],
            "position": {"x": 0.6, "y": 0.1, "z": 0.3},
        },
    }
    backend = FakeReachabilityBackend()

    result = screen_grasp_candidates_for_plan_only(
        [high_score_unreachable, lower_score_reachable],
        point_cloud={"frame": "base_link"},
        preview_config={
            "candidate_screening": {
                "enabled": True,
                "max_candidates": 2,
                "variants": ["full_pose"],
            }
        },
        target_frame="base_link",
        transformer=None,
        backend=backend,
        trajectory_validator=TrajectoryValidator(
            {"trajectory_limits": {"max_joint_delta_rad": 1.0}}
        ),
    )

    assert result["selected"]["candidate_id"] == "lower_score_good"
    assert result["variant"] == "full_pose"
    assert result["poses"]["pregrasp"]["position"]["x"] == pytest.approx(0.6)
    assert backend.execution_calls == []


def test_candidate_screening_rejects_candidate_when_grasp_stage_fails():
    high_score_bad_grasp = {
        **FAKE_CANDIDATE,
        "candidate_id": "high_score_bad_grasp",
        "score": 0.99,
        "pose": {
            **FAKE_CANDIDATE["pose"],
            "position": {"x": 0.6, "y": 0.8, "z": 0.3},
        },
    }
    lower_score_full_sequence = {
        **FAKE_CANDIDATE,
        "candidate_id": "lower_score_full_sequence",
        "score": 0.75,
        "pose": {
            **FAKE_CANDIDATE["pose"],
            "position": {"x": 0.6, "y": 0.1, "z": 0.3},
        },
    }
    backend = FakeSequenceReachabilityBackend()

    result = screen_grasp_candidates_for_plan_only(
        [high_score_bad_grasp, lower_score_full_sequence],
        point_cloud={"frame": "base_link"},
        preview_config={
            "moveit": {
                "pipeline_id": "pilz_industrial_motion_planner",
                "planner_id": "PTP",
                "stage_planner_ids": {
                    "pregrasp": "PTP",
                    "grasp": "LIN",
                    "retreat": "LIN",
                },
            },
            "candidate_screening": {
                "enabled": True,
                "max_candidates": 2,
                "variants": ["full_pose"],
            },
        },
        target_frame="base_link",
        transformer=None,
        backend=backend,
        trajectory_validator=TrajectoryValidator(
            {"trajectory_limits": {"max_joint_delta_rad": 1.0}}
        ),
    )

    assert result["selected"]["candidate_id"] == "lower_score_full_sequence"
    failed_attempts = [
        attempt
        for attempt in result["attempts"]
        if attempt["candidate_id"] == "high_score_bad_grasp"
    ]
    assert failed_attempts[0]["failed_stage"] == "grasp"
    assert backend.plan_calls[0]["planner_id"] == "PTP"
    assert backend.plan_calls[1]["planner_id"] == "LIN"


def test_candidate_screening_adjusts_poses_before_moveit_planning():
    backend = FakePlanOnlyBackend()
    validated = {}

    def adjust(poses):
        adjusted = copy.deepcopy(poses)
        adjusted["grasp"]["position"]["x"] = 0.77
        return adjusted

    def validate(poses):
        validated.update(poses)
        return {"allowed": True, "checks": {"grasp": {"allowed": True}}}

    result = screen_grasp_candidates_for_plan_only(
        [FAKE_CANDIDATE],
        point_cloud={"frame": "base_link"},
        preview_config={"candidate_screening": {"variants": ["full_pose"]}},
        target_frame="base_link",
        transformer=None,
        backend=backend,
        trajectory_validator=AllowValidator(),
        pose_adjuster=adjust,
        pose_validator=validate,
    )

    grasp_call = next(
        call for call in backend.plan_calls if call["skill_id"] == "plan_grasp_pose"
    )
    assert grasp_call["position"]["x"] == pytest.approx(0.77)
    assert validated["grasp"]["position"]["x"] == pytest.approx(0.77)
    assert result["attempt"]["tabletop_safety"]["allowed"] is True


def test_candidate_screening_blocks_unsafe_pose_before_moveit_planning():
    backend = FakePlanOnlyBackend()

    result = screen_grasp_candidates_for_plan_only(
        [FAKE_CANDIDATE],
        point_cloud={"frame": "base_link"},
        preview_config={"candidate_screening": {"variants": ["full_pose"]}},
        target_frame="base_link",
        transformer=None,
        backend=backend,
        trajectory_validator=AllowValidator(),
        pose_validator=lambda _poses: {
            "allowed": False,
            "reason": "gripper contact point is too close to the table",
        },
    )

    assert result["selected"] is None
    assert backend.plan_calls == []
    assert result["attempts"][0]["variant"] == "tabletop_safety"
    assert result["attempts"][0]["validation"]["decision"] == "BLOCK"


def test_prepare_candidate_preview_applies_tf_and_ee_transform():
    camera_candidate = {
        **FAKE_CANDIDATE,
        "pose": {
            **FAKE_CANDIDATE["pose"],
            "frame": "camera_color_optical_frame",
        },
    }

    poses = prepare_candidate_preview(
        camera_candidate,
        point_cloud={"frame": "camera_color_optical_frame"},
        preview_config={
            "grasp_frame_to_ee_frame": {
                "enabled": True,
                "translation_m": [0.1, 0.0, 0.0],
                "rotation_rpy_rad": [0.0, 0.0, 0.0],
            }
        },
        target_frame="base_link",
        transformer=FakePoseTransformer(),
    )

    assert poses["grasp"]["frame"] == "base_link"
    assert poses["grasp"]["position"]["x"] == pytest.approx(1.5)


def test_prepare_candidate_preview_can_retreat_in_target_frame():
    poses = prepare_candidate_preview(
        FAKE_CANDIDATE,
        point_cloud={"frame": "base_link"},
        preview_config={
            "retreat_distance_m": 0.10,
            "retreat_vector_frame": "target_frame",
            "retreat_vector": [0.0, 0.0, 1.0],
        },
        target_frame="base_link",
        transformer=None,
    )

    assert poses["retreat"]["frame"] == "base_link"
    assert poses["retreat"]["position"]["x"] == pytest.approx(
        poses["grasp"]["position"]["x"]
    )
    assert poses["retreat"]["position"]["y"] == pytest.approx(
        poses["grasp"]["position"]["y"]
    )
    assert poses["retreat"]["position"]["z"] == pytest.approx(
        poses["grasp"]["position"]["z"] + 0.10
    )


def test_planning_variant_kwargs_supports_position_only():
    kwargs = planning_variant_kwargs(
        "position_only",
        {"position_tolerance_m": 0.03},
    )

    assert kwargs == {
        "position_tolerance": 0.03,
        "use_orientation_constraint": False,
    }


def test_moveit_planning_kwargs_reads_preview_config():
    kwargs = moveit_planning_kwargs(
        {
            "moveit": {
                "pipeline_id": "pilz_industrial_motion_planner",
                "planner_id": "RRTConnectkConfigDefault",
                "num_planning_attempts": 8,
                "allowed_planning_time_sec": 7.5,
                "service_timeout_sec": 12.0,
            }
        }
    )

    assert kwargs == {
        "pipeline_id": "pilz_industrial_motion_planner",
        "planner_id": "RRTConnectkConfigDefault",
        "num_planning_attempts": 8,
        "allowed_planning_time": 7.5,
        "service_timeout_sec": 12.0,
    }


def test_moveit_stage_planning_kwargs_overrides_pilz_stage_planners():
    preview_config = {
        "moveit": {
            "pipeline_id": "pilz_industrial_motion_planner",
            "planner_id": "PTP",
            "stage_planner_ids": {
                "pregrasp": "PTP",
                "grasp": "LIN",
                "retreat": "LIN",
            },
        }
    }

    assert moveit_stage_planning_kwargs(preview_config, "pregrasp") == {
        "pipeline_id": "pilz_industrial_motion_planner",
        "planner_id": "PTP",
    }
    assert moveit_stage_planning_kwargs(preview_config, "grasp") == {
        "pipeline_id": "pilz_industrial_motion_planner",
        "planner_id": "LIN",
    }


def test_pregrasp_diagnostics_runs_full_relaxed_and_position_only_plans():
    backend = FakePlanOnlyBackend()
    pose = create_preview_poses(FAKE_CANDIDATE)["pregrasp"]

    result = diagnose_pregrasp_planning(backend, pose, AllowValidator())

    assert [item["variant"] for item in result] == [
        "full_pose",
        "relaxed_orientation",
        "position_only",
    ]
    assert backend.plan_calls[0]["orientation_tolerance"] == 0.05
    assert backend.plan_calls[1]["orientation_tolerance"] == 0.75
    assert backend.plan_calls[2]["use_orientation_constraint"] is False
    assert backend.execution_calls == []


def test_real_arm_execution_blocked_even_after_confirmation():
    with pytest.raises(PermissionError, match="HW4 never executes real arm trajectories"):
        attempt_real_arm_execution(load_hardware_mode(), confirmed=True)


def test_plan_only_preview_is_allowed_in_full_grasp_supervised_mode():
    assert assert_plan_only_mode(load_hardware_mode()) is None


def test_plan_only_preview_is_blocked_when_plan_only_is_disabled():
    mode = load_hardware_mode()
    disabled = replace(mode, arm=replace(mode.arm, allow_plan_only=False))

    with pytest.raises(PermissionError, match="Plan-only arm requests are disabled"):
        assert_plan_only_mode(disabled)


def test_preview_pose_offsets_do_not_modify_grasp_pose():
    poses = create_preview_poses(FAKE_CANDIDATE)

    assert poses["grasp"]["position"] == {"x": 0.4, "y": 0.1, "z": 0.3}
    assert poses["pregrasp"]["position"]["z"] == pytest.approx(0.2)
    assert poses["retreat"]["position"]["z"] == pytest.approx(0.4)


def test_camera_message_status_reports_cached_d435i_messages():
    status = camera_message_status(
        FakeCameraBackend(rgb=object(), depth=object(), camera_info=object())
    )

    assert status == {
        "rgb_received": True,
        "depth_received": True,
        "camera_info_received": True,
    }


def test_wait_for_camera_messages_returns_immediately_when_ready():
    fake_rclpy = FakeRclpy()

    status = wait_for_camera_messages(
        FakeCameraBackend(rgb=object(), depth=object(), camera_info=object()),
        node=object(),
        rclpy_module=fake_rclpy,
    )

    assert status["rgb_received"] is True
    assert status["depth_received"] is True
    assert status["camera_info_received"] is True
    assert fake_rclpy.spin_calls == 0


def test_localize_graspgenx_candidate_adds_point_cloud_centroid():
    candidate = {
        "candidate_id": "g1",
        "pose_frame": "object_point_cloud",
        "transform": [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
        "pose": {
            "frame": "object_point_cloud",
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    point_cloud = {
        "frame": "camera_color_optical_frame",
        "centroid_m": {"x": 0.4, "y": -0.1, "z": 0.7},
    }

    localized = localize_graspgenx_candidate(candidate, point_cloud)

    assert localized["pose_frame"] == "camera_color_optical_frame"
    assert localized["pose"]["frame"] == "camera_color_optical_frame"
    assert localized["pose"]["position"] == pytest.approx(
        {"x": 0.5, "y": 0.1, "z": 1.0}
    )
    assert localized["transform"][0][3] == pytest.approx(0.5)
    assert localized["metadata"]["point_cloud_centroid_m"] == point_cloud["centroid_m"]


def test_transform_pose_sequence_to_target_uses_tf_transformer():
    poses = {
        "pregrasp": {
            "frame": "camera_color_optical_frame",
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    transformed = transform_pose_sequence_to_target(
        poses, FakePoseTransformer(), "base_link"
    )

    assert transformed["pregrasp"]["frame"] == "base_link"
    assert transformed["pregrasp"]["position"] == pytest.approx(
        {"x": 1.1, "y": 2.2, "z": 3.3}
    )


def test_disabled_grasp_frame_to_ee_transform_leaves_poses_unchanged():
    poses = create_preview_poses(FAKE_CANDIDATE)

    transformed = apply_grasp_frame_to_ee_transform(poses, {"enabled": False})

    assert transformed is poses
    assert "disabled" in describe_grasp_frame_to_ee_transform({"enabled": False})


def test_grasp_frame_to_ee_transform_applies_local_translation():
    poses = {
        "grasp": {
            "frame": "base_link",
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    transformed = apply_grasp_frame_to_ee_transform(
        poses,
        {
            "enabled": True,
            "translation_m": [0.1, -0.2, 0.3],
            "rotation_rpy_rad": [0.0, 0.0, 0.0],
        },
    )

    assert transformed["grasp"]["position"] == pytest.approx(
        {"x": 1.1, "y": 1.8, "z": 3.3}
    )
    assert transformed["grasp"]["orientation"] == pytest.approx(
        {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    )


def test_grasp_frame_to_ee_transform_translation_is_in_grasp_frame():
    yaw_90 = math.sqrt(0.5)
    poses = {
        "grasp": {
            "frame": "base_link",
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": yaw_90, "w": yaw_90},
        }
    }

    transformed = apply_grasp_frame_to_ee_transform(
        poses,
        {
            "enabled": True,
            "translation_m": [0.1, 0.0, 0.0],
            "rotation_rpy_rad": [0.0, 0.0, 0.0],
        },
    )

    assert transformed["grasp"]["position"] == pytest.approx(
        {"x": 0.0, "y": 0.1, "z": 0.0},
        abs=1e-8,
    )


def test_grasp_frame_to_ee_transform_composes_orientation():
    poses = {
        "grasp": {
            "frame": "base_link",
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    transformed = apply_grasp_frame_to_ee_transform(
        poses,
        {
            "enabled": True,
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_rpy_rad": [0.0, 0.0, math.pi / 2.0],
        },
    )

    assert transformed["grasp"]["orientation"] == pytest.approx(
        {"x": 0.0, "y": 0.0, "z": math.sqrt(0.5), "w": math.sqrt(0.5)}
    )
