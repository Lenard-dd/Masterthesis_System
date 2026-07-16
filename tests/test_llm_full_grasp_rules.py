import json
from pathlib import Path

import pytest

from llm_skill_robot.agent.llm_plan_generator import LLMPlanGenerator
from llm_skill_robot.agent.mock_plan_generator import PlanGenerationError
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.execution_manager import ExecutionManager
from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import VerificationDecision, VerificationResult
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.grasping.full_grasp_state_machine import FullGraspStateMachine
from llm_skill_robot.safety.hardware_mode import load_hardware_mode


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SEQUENCE = [
    "detect_object",
    "build_object_point_cloud",
    "generate_grasp_pose",
    "review_grasp_candidate",
    "open_gripper",
    "move_to_pregrasp",
    "approach_grasp",
    "close_gripper",
    "retreat_grasp",
    "verify_grasp",
]


def _full_mode():
    return load_hardware_mode(str(ROOT / "config" / "hardware_mode.yaml"))


def _full_plan() -> RobotPlan:
    parameters = [
        {"query": "red cube"},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>"},
        {},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>", "during_contact": True},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>"},
    ]
    return RobotPlan(
        task="pick the red cube",
        steps=[
            PlanStep(
                step_id=str(index),
                skill_id=skill_id,
                description=skill_id,
                parameters=parameters[index - 1],
            )
            for index, skill_id in enumerate(REQUIRED_SEQUENCE, start=1)
        ],
    )


class FakeLLMClient:
    def __init__(self, plan: RobotPlan):
        self.response = plan.model_dump_json()
        self.messages = []

    def generate_text(self, messages):
        self.messages.append(messages)
        return self.response


def test_llm_full_grasp_plan_contains_required_sequence(tmp_path):
    plan = _full_plan()
    client = FakeLLMClient(plan)
    logger = AuditLogger(str(tmp_path / "audit.log"), hardware_mode=_full_mode())
    library = SkillLibrary()
    generator = LLMPlanGenerator(
        client, hardware_mode=_full_mode(), audit_logger=logger
    )

    generated = generator.generate_plan(
        "Pick the red cube",
        [library.get_skill(skill_id) for skill_id in library.list_skills()],
    )

    assert [step.skill_id for step in generated.steps] == REQUIRED_SEQUENCE
    assert generated.steps[7].parameters["during_contact"] is True
    prompt = client.messages[0][1]["content"]
    assert "STATE_MACHINE_AND_PER_STEP_CONFIRMATION_ONLY" in prompt
    assert "batch_confirmation" in prompt
    entries = [json.loads(line) for line in (tmp_path / "audit.log").read_text().splitlines()]
    assert any(entry["event"] == "llm_full_grasp_plan_generated" for entry in entries)


def test_rviz_detection_policy_adds_observe_to_full_grasp_plan():
    generator = LLMPlanGenerator(
        FakeLLMClient(_full_plan()),
        hardware_mode=_full_mode(),
        require_observe_before_detection=True,
    )

    generated = generator.generate_plan(
        "Pick the red cube",
        [SkillLibrary().get_skill(skill_id) for skill_id in SkillLibrary().list_skills()],
    )

    assert generated.steps[0].skill_id == "move_to_named_target"
    assert generated.steps[0].parameters["target_name"] == "observe"
    assert [step.skill_id for step in generated.steps[1:]] == REQUIRED_SEQUENCE


def test_llm_direct_pick_is_blocked_in_full_mode():
    plan = RobotPlan(
        task="pick directly",
        steps=[PlanStep(
            step_id="1", skill_id="pick_object", description="direct pick",
            parameters={"object_id": "cube_1"},
        )],
    )
    generator = LLMPlanGenerator(FakeLLMClient(plan), hardware_mode=_full_mode())

    with pytest.raises(PlanGenerationError, match="hardware-mode-blocked.*pick_object"):
        generator.generate_plan("pick cube", SkillLibrary().list_skills())


def test_close_gripper_before_approach_is_blocked(tmp_path):
    logger = AuditLogger(str(tmp_path / "audit.log"), hardware_mode=_full_mode())
    verifier = PlanVerifier(
        SkillLibrary(), hardware_mode=_full_mode(), audit_logger=logger
    )
    plan = RobotPlan(task="early close", steps=[PlanStep(
        step_id="1", skill_id="close_gripper", description="contact close",
        parameters={"object_id": "cube_1", "during_contact": True},
    )])

    result = verifier.verify_plan(plan, context={"gripper_confirmations": {"1": "YES"}})

    assert result.decision == VerificationDecision.BLOCK
    assert any("AT_GRASP_POSE" in reason for reason in result.reasons)
    entries = [json.loads(line) for line in (tmp_path / "audit.log").read_text().splitlines()]
    assert any(entry["event"] == "llm_step_blocked" for entry in entries)


def test_retreat_before_close_is_blocked():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=_full_mode())
    plan = RobotPlan(task="early retreat", steps=[PlanStep(
        step_id="1", skill_id="retreat_grasp", description="retreat",
        parameters={"object_id": "cube_1"},
    )])

    result = verifier.verify_plan(
        plan,
        context={
            "arm_execution_mode": "full_grasp_supervised",
            "grasp_candidate_approved": True,
            "full_grasp_state": FullGraspStateMachine.AT_GRASP_POSE,
        },
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("GRIPPER_CLOSED" in reason for reason in result.reasons)


def test_batch_execution_request_is_blocked():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=_full_mode())
    plan = RobotPlan(task="batch", steps=[PlanStep(
        step_id="1", skill_id="approach_grasp", description="execute all steps",
        parameters={"object_id": "cube_1", "execute_all": True},
    )])

    result = verifier.verify_plan(plan)

    assert result.decision == VerificationDecision.BLOCK
    assert any("Batch execution" in reason for reason in result.reasons)


def test_raw_command_is_blocked():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=_full_mode())
    plan = RobotPlan(task="raw", steps=[PlanStep(
        step_id="1", skill_id="detect_object", description="raw command",
        parameters={"query": "cube", "ros_command": "ros2 topic pub /arm ..."},
    )])

    result = verifier.verify_plan(plan)

    assert result.decision == VerificationDecision.BLOCK
    assert any("Raw ROS" in reason for reason in result.reasons)


def test_detection_is_blocked_when_demo_requires_observe_precondition():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=_full_mode())
    plan = RobotPlan(task="detect", steps=[PlanStep(
        step_id="1", skill_id="detect_object", description="detect cube",
        parameters={"query": "cube"},
    )])

    result = verifier.verify_plan(
        plan,
        context={
            "skip_perception_scene_checks": True,
            "require_observe_before_detection": True,
        },
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("target_name=observe" in reason for reason in result.reasons)


class FullBackend:
    execution_mode = "full_grasp_supervised"

    def __init__(self):
        self.calls = []

    def _ok(self, name, **kwargs):
        self.calls.append((name, kwargs.get("confirmation")))
        return {"success": True, "message": f"{name} ok", "output": {}}

    def detect_object(self, **kwargs):
        result = self._ok("detect_object", **kwargs)
        result["output"] = {"object_id": "red_cube_1"}
        return result
    def build_object_point_cloud(self, **kwargs): return self._ok("build_object_point_cloud", **kwargs)
    def generate_grasp_pose(self, **kwargs): return self._ok("generate_grasp_pose", **kwargs)
    def open_gripper(self, **kwargs): return self._ok("open_gripper", **kwargs)
    def move_to_pregrasp(self, **kwargs): return self._ok("move_to_pregrasp", **kwargs)
    def approach_grasp(self, **kwargs): return self._ok("approach_grasp", **kwargs)
    def close_for_grasp(self, **kwargs): return self._ok("close_for_grasp", **kwargs)
    def retreat_grasp(self, **kwargs): return self._ok("retreat_grasp", **kwargs)
    def verify_grasp(self, **kwargs): return self._ok("verify_grasp", **kwargs)


def test_full_execution_uses_state_machine_and_separate_confirmations(tmp_path):
    backend = FullBackend()
    machine = FullGraspStateMachine()
    requested_prompts = []

    def confirmation(prompt):
        requested_prompts.append(prompt)
        return "EXECUTE" if "EXECUTE" in prompt else "YES"

    logger = AuditLogger(str(tmp_path / "audit.log"), hardware_mode=_full_mode())
    manager = ExecutionManager(
        SkillLibrary(), backend, logger, hardware_mode=_full_mode(),
        full_grasp_state_machine=machine, confirmation_provider=confirmation,
    )
    context = {
        "arm_execution_mode": "full_grasp_supervised",
        "selected_grasp_candidate": {"candidate_id": "g1", "score": 0.9},
        "grasp_candidate_confirmation": "YES",
        "grasp_plan": {"candidate_id": "g1"},
    }

    report = manager.execute_plan(
        _full_plan(),
        VerificationResult(decision=VerificationDecision.ALLOW),
        "approved",
        context=context,
    )

    assert report.status == "SUCCESS"
    assert machine.current_state() == FullGraspStateMachine.DONE
    assert len(requested_prompts) == 5
    assert [phrase for _, phrase in backend.calls if phrase] == [
        "YES", "EXECUTE", "EXECUTE", "YES", "EXECUTE"
    ]
    entries = [json.loads(line) for line in (tmp_path / "audit.log").read_text().splitlines()]
    assert any(entry["event"] == "full_grasp_state_transition" for entry in entries)
    assert any(entry["event"] == "llm_step_verified" for entry in entries)
    assert sum(entry["event"] == "human_confirmation_result" for entry in entries) == 5
