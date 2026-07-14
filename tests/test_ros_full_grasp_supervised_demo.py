import importlib
from types import SimpleNamespace

from llm_skill_robot.grasping.full_grasp_state_machine import FullGraspStateMachine
from llm_skill_robot.grasping.grasp_verifier import GraspVerifier
from llm_skill_robot.ros_full_grasp_supervised_demo import (
    FullGraspSupervisedDemoController,
    print_full_grasp_report,
)
from llm_skill_robot.safety.hardware_mode import load_hardware_mode
from llm_skill_robot.safety.real_arm_safety import load_real_arm_safety


def _pose(z):
    return {
        "frame": "base_link",
        "position": {"x": 0.4, "y": 0.0, "z": z},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _pose_builder(*_args):
    return {"pregrasp": _pose(0.36), "grasp": _pose(0.31), "retreat": _pose(0.40)}


class FakePerception:
    class Backend:
        config = {"depth_frame": "base_link"}
        latest_camera_info_msg = object()

        @staticmethod
        def _get_latest_depth_image():
            return object()

    def __init__(self, recheck_objects=None):
        self.backend = self.Backend()
        self.calls = 0
        self.recheck_objects = recheck_objects

    def detect_object(self, _query, require_pose=False):
        self.calls += 1
        if self.calls == 1:
            return {"success": True, "objects": [{"object_id": "cube_1", "confidence": 0.9}]}
        return {"success": True, "objects": self.recheck_objects}


class FakePointCloudBuilder:
    def build(self, *_args, **_kwargs):
        return {"success": True, "point_cloud_path": "/tmp/cube_1.ply", "frame": "base_link"}


class FakePlanner:
    def __init__(self, score=0.9):
        self.score = score

    def generate_candidates(self, _path):
        return {"success": True, "candidates": [{"candidate_id": "g1", "score": self.score, "valid": True}]}


class FakeMotion:
    def __init__(self):
        self.calls = []
        self.planning_options = {}

    def set_planning_kwargs(self, skill_id, options):
        self.planning_options[skill_id] = options

    def move_to_observe(self, confirmation):
        self.calls.append(("move_to_observe", confirmation))
        return {"success": True, "message": "observe complete"}

    def move_to_pregrasp(self, object_id, plans, confirmation):
        self.calls.append(("move_to_pregrasp", confirmation))
        return {"success": True, "message": "pregrasp complete"}

    def approach_grasp(self, object_id, plans, confirmation):
        self.calls.append(("approach_grasp", confirmation))
        return {"success": True, "message": "approach complete"}

    def retreat_grasp(self, object_id, plans, confirmation):
        self.calls.append(("retreat_grasp", confirmation))
        return {"success": True, "message": "retreat complete"}


class FakeGripper:
    def __init__(self):
        self.open_calls = 0
        self.close_calls = 0

    def open_gripper(self, confirmed=False):
        self.open_calls += 1
        return {"success": True, "message": "opened"}

    def close_for_grasp(self, confirmation):
        self.close_calls += 1
        return {"success": True, "message": "closed", "output": {"command_sent": True, "feedback_available": False}}

    def get_gripper_state(self):
        return {"success": False, "output": {"command_sent": True, "feedback_available": False}}


def _controller(responses, *, score=0.9, recheck_objects=None):
    values = iter(responses)
    return FullGraspSupervisedDemoController(
        hardware_mode=load_hardware_mode(),
        real_arm_safety=load_real_arm_safety(),
        perception_pipeline=FakePerception(recheck_objects),
        point_cloud_builder=FakePointCloudBuilder(),
        grasp_planner=FakePlanner(score),
        motion_backend=FakeMotion(),
        gripper_backend=FakeGripper(),
        grasp_verifier=GraspVerifier(),
        input_fn=lambda _prompt: next(values),
        pose_builder=_pose_builder,
    )


def test_module_imports_without_ros_initialization():
    assert callable(importlib.import_module("llm_skill_robot.ros_full_grasp_supervised_demo").main)


def test_fake_successful_sequence_reaches_done_with_honest_uncertain_verification():
    controller = _controller(["EXECUTE", "YES", "YES", "EXECUTE", "EXECUTE", "YES", "EXECUTE"])

    report = controller.run("cube")

    assert report["status"] == FullGraspStateMachine.DONE
    assert report["results"]["verify_grasp"]["status"] == "UNCERTAIN"


def test_invalid_transition_stops_the_controller():
    controller = _controller([])

    allowed = controller._transition(FullGraspStateMachine.DONE, "invalid")

    assert allowed is False
    assert controller.state.current_state() == FullGraspStateMachine.ABORTED


def test_missing_execute_blocks_before_any_arm_motion():
    controller = _controller(["EXECUTE", "YES", "YES", "NO"])

    report = controller.run("cube")

    assert report["status"] == FullGraspStateMachine.ABORTED
    assert controller.motion_backend.calls == [("move_to_observe", "EXECUTE")]


def test_missing_yes_blocks_contact_close():
    controller = _controller(["EXECUTE", "YES", "YES", "EXECUTE", "EXECUTE", "NO"])

    report = controller.run("cube")

    assert report["status"] == FullGraspStateMachine.ABORTED
    assert controller.gripper_backend.close_calls == 0


def test_low_grasp_score_stops_before_hardware_actions():
    controller = _controller(["EXECUTE"], score=0.59)

    report = controller.run("cube")

    assert report["status"] == FullGraspStateMachine.ABORTED
    assert controller.gripper_backend.open_calls == 0
    assert controller.motion_backend.calls == [("move_to_observe", "EXECUTE")]


def test_missing_observe_confirmation_blocks_detection_and_all_motion():
    controller = _controller(["NO"])

    report = controller.run("cube")

    assert report["status"] == FullGraspStateMachine.ABORTED
    assert controller.motion_backend.calls == []
    assert controller.perception_pipeline.calls == 0


def test_abort_report_includes_the_reason_for_operator_diagnosis():
    controller = _controller([])

    report = controller.run("")

    assert report["status"] == FullGraspStateMachine.ABORTED
    assert report["abort_reason"] == "Object query must not be empty."


def test_compact_operator_report_does_not_dump_raw_detection_payload(capsys):
    print_full_grasp_report(
        {
            "status": "ABORTED",
            "abort_reason": "move_to_pregrasp failed: safety limit.",
            "object_id": "cube_1",
            "candidate_id": "g1",
            "candidate_score": 0.8,
            "results": {
                "detect_object": {"objects": [{"large": "raw payload"}]},
                "move_to_pregrasp": {"success": False, "message": "safety limit."},
            },
            "audit_log_path": "audit.log",
        }
    )

    output = capsys.readouterr().out
    assert "Stopped safely: move_to_pregrasp failed: safety limit." in output
    assert "move_to_pregrasp: BLOCKED/FAILED — safety limit." in output
    assert "raw payload" not in output


def test_enabled_screening_reuses_restricted_pregrasp_selector(monkeypatch):
    controller = _controller([])
    controller.grasping_config = {
        "plan_only_preview": {
            "candidate_screening": {"enabled": True, "variants": ["full_pose"]}
        }
    }
    controller.motion_backend.moveit_backend = object()
    candidates = [
        {"candidate_id": "unsafe_high_score", "score": 0.95, "valid": True},
        {"candidate_id": "safe_lower_score", "score": 0.80, "valid": True},
    ]

    def select_safe_candidate(_self, screened_candidates, _point_cloud, **kwargs):
        assert kwargs["preserve_candidate_order"] is True
        candidate = screened_candidates[0]
        if candidate["candidate_id"] == "unsafe_high_score":
            return None
        return SimpleNamespace(
            candidate=candidate,
            variant="full_pose",
            attempts=[{"candidate_id": "unsafe_high_score", "accepted": False}],
        )

    monkeypatch.setattr(
        "llm_skill_robot.ros_full_grasp_supervised_demo.PregraspRealDemoController.select_restricted_pregrasp_plan",
        select_safe_candidate,
    )
    monkeypatch.setattr(
        controller,
        "_screen_grasp_and_retreat_targets",
        lambda *_args: {"allowed": True, "attempts": []},
    )

    screened = controller._screen_pregrasp_candidates(candidates, {"point_cloud_path": "/tmp/cube.ply"})

    assert screened[0]["candidate_id"] == "safe_lower_score"
    assert screened[1] == "full_pose"


def test_full_candidate_screening_rejects_grasp_target_without_ik():
    class NoIkMoveIt:
        def plan_to_pose(self, **_kwargs):
            return {"success": False, "message": "NO_IK_SOLUTION (-31)."}

    controller = _controller([])
    result = controller._screen_grasp_and_retreat_targets(
        {"candidate_id": "no_ik", "score": 0.9},
        {"point_cloud_path": "/tmp/cube.ply"},
        NoIkMoveIt(),
        "full_pose",
    )

    assert result["allowed"] is False
    assert "NO_IK_SOLUTION" in result["attempts"][0]["reason"]
