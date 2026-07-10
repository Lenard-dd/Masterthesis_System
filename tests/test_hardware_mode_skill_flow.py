import json

from llm_skill_robot.agent.llm_plan_generator import LLMPlanGenerator
from llm_skill_robot.agent.mock_plan_generator import PlanGenerationError
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.execution_manager import ExecutionManager
from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import VerificationDecision, VerificationResult
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.robot.mock_robot_backend import MockRobotBackend
from llm_skill_robot.safety.hardware_mode import load_hardware_mode


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def generate_text(self, messages):
        self.messages.append(messages)
        return self.response


def _safe_pick_response():
    steps = [
        ("move_to_named_target", {"target": "observe"}),
        ("detect_object", {"query": "red cube"}),
        ("build_object_point_cloud", {"object_id": "<resolved>"}),
        ("generate_grasp_pose", {"object_id": "<resolved>"}),
        ("review_grasp_candidate", {"object_id": "<resolved>"}),
        ("open_gripper", {}),
        ("move_to_pregrasp", {"object_id": "<resolved>"}),
        ("test_gripper_width", {"object_id": "<resolved>"}),
    ]
    return json.dumps(
        {
            "task": "pick the red cube safely",
            "steps": [
                {
                    "step_id": index,
                    "skill_id": skill_id,
                    "description": skill_id,
                    "parameters": parameters,
                }
                for index, (skill_id, parameters) in enumerate(steps, start=1)
            ],
        }
    )


def test_llm_pick_plan_in_hardware_mode_contains_only_safe_stages():
    library = SkillLibrary()
    client = FakeLLMClient(_safe_pick_response())
    generator = LLMPlanGenerator(client, hardware_mode=load_hardware_mode())

    plan = generator.generate_plan(
        "pick the red cube",
        [library.get_skill(skill_id) for skill_id in library.list_skills()],
    )

    skill_ids = [step.skill_id for step in plan.steps]
    assert "pick_object" not in skill_ids
    assert "execute_trajectory" not in skill_ids
    assert "approach_grasp" not in skill_ids
    assert "lift_object" not in skill_ids
    assert skill_ids[-3:] == ["open_gripper", "move_to_pregrasp", "test_gripper_width"]
    prompt = client.messages[0][1]["content"]
    assert "real_camera_real_gripper_restricted_arm" in prompt
    assert "Do not generate pick_object in restricted-arm hardware mode" in prompt
    assert "Do not generate approach_grasp" in prompt


def test_llm_direct_pick_output_is_rejected_even_if_prompt_is_ignored():
    response = json.dumps(
        {
            "task": "unsafe direct pick",
            "steps": [
                {
                    "step_id": 1,
                    "skill_id": "pick_object",
                    "description": "direct pick",
                    "parameters": {"object_id": "cube_1"},
                }
            ],
        }
    )
    library = SkillLibrary()
    generator = LLMPlanGenerator(
        FakeLLMClient(response), hardware_mode=load_hardware_mode()
    )

    try:
        generator.generate_plan(
            "pick cube",
            [library.get_skill(skill_id) for skill_id in library.list_skills()],
        )
    except PlanGenerationError as exc:
        assert "hardware-mode-blocked skill_id: pick_object" in str(exc)
    else:
        raise AssertionError("unsafe direct-pick LLM output was accepted")


def test_llm_approach_grasp_output_is_rejected_in_restricted_mode():
    response = json.dumps(
        {
            "task": "unsafe approach",
            "steps": [
                {
                    "step_id": 1,
                    "skill_id": "approach_grasp",
                    "description": "approach the grasp",
                    "parameters": {"object_id": "cube_1"},
                }
            ],
        }
    )
    library = SkillLibrary()
    generator = LLMPlanGenerator(
        FakeLLMClient(response), hardware_mode=load_hardware_mode()
    )

    try:
        generator.generate_plan(
            "pick cube",
            [library.get_skill(skill_id) for skill_id in library.list_skills()],
        )
    except PlanGenerationError as exc:
        assert "hardware-mode-blocked skill_id: approach_grasp" in str(exc)
    else:
        raise AssertionError("unsafe approach_grasp LLM output was accepted")


def test_real_arm_execution_request_is_blocked():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="unsafe real motion",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_named_target",
                description="unsafe",
                parameters={"target": "observe", "execution_mode": "real_robot"},
            )
        ],
    )

    result = verifier.verify_plan(plan)

    assert result.decision == VerificationDecision.BLOCK
    assert any("Real arm execution request is blocked" in reason for reason in result.reasons)


def test_set_gripper_width_over_hardware_limit_is_blocked():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="unsafe width",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="set_gripper_width",
                description="too wide",
                parameters={"width_m": 0.20, "speed_mps": 0.02, "force_N": 10.0},
            )
        ],
    )

    result = verifier.verify_plan(
        plan,
        context={
            "gripper_confirmation": "YES",
            "grasp_candidate_approved": True,
        },
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("exceeds maximum" in reason for reason in result.reasons)


def test_real_gripper_command_without_exact_confirmation_is_blocked():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="width without confirmation",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="set_gripper_width",
                description="width",
                parameters={"width_m": 0.05},
            )
        ],
    )

    result = verifier.verify_plan(
        plan,
        context={"gripper_confirmation": "yes", "grasp_candidate_approved": True},
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("exact trusted YES" in reason for reason in result.reasons)


def test_pick_object_direct_execution_is_blocked():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="direct pick",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="pick_object",
                description="direct hardware pick",
                parameters={"object_id": "cube_1"},
            )
        ],
    )

    result = verifier.verify_plan(plan)

    assert result.decision == VerificationDecision.BLOCK
    assert any("pick_object direct execution is blocked" in reason for reason in result.reasons)


def test_approach_grasp_is_blocked_in_restricted_mode():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="unsafe approach",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="approach_grasp",
                description="approach",
                parameters={"object_id": "cube_1"},
            )
        ],
    )

    result = verifier.verify_plan(
        plan,
        context={
            "grasp_candidate_approved": True,
            "arm_execution_mode": "restricted_real_motion",
        },
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("approach_grasp is blocked" in reason for reason in result.reasons)


def test_lift_object_is_blocked_in_restricted_mode_even_if_unknown_skill():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="unsafe lift",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="lift_object",
                description="lift",
                parameters={"object_id": "cube_1"},
            )
        ],
    )

    result = verifier.verify_plan(plan)

    assert result.decision == VerificationDecision.BLOCK
    assert any("lift_object is blocked" in reason for reason in result.reasons)


def test_close_gripper_during_contact_is_blocked():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="unsafe close",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="close_gripper",
                description="close during contact",
                parameters={"during_contact": True},
            )
        ],
    )

    result = verifier.verify_plan(
        plan,
        context={
            "gripper_confirmations": {"1": "YES"},
            "grasp_candidate_approved": True,
        },
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("close_gripper during contact" in reason for reason in result.reasons)


def test_move_to_pregrasp_requires_grasp_plan_in_restricted_mode():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="pregrasp",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_pregrasp",
                description="pregrasp",
                parameters={"object_id": "cube_1"},
            )
        ],
    )

    result = verifier.verify_plan(
        plan,
        context={
            "arm_execution_mode": "restricted_real_motion",
            "grasp_candidate_approved": True,
        },
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("valid GraspPlan" in reason for reason in result.reasons)


def test_move_to_pregrasp_allowed_with_valid_grasp_plan_and_summary():
    verifier = PlanVerifier(SkillLibrary(), hardware_mode=load_hardware_mode())
    plan = RobotPlan(
        task="pregrasp",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_pregrasp",
                description="pregrasp",
                parameters={"object_id": "cube_1"},
            )
        ],
    )

    result = verifier.verify_plan(
        plan,
        context={
            "arm_execution_mode": "restricted_real_motion",
            "grasp_candidate_approved": True,
            "arm_confirmation_required": True,
            "grasp_plan": {"pregrasp_pose": {"frame": "base_link"}},
            "pregrasp_motion_summary": {
                "max_joint_delta_rad": 0.01,
                "total_joint_delta_rad": 0.02,
                "max_cartesian_step_m": 0.01,
                "velocity_scale": 0.01,
                "acceleration_scale": 0.01,
            },
        },
    )

    assert result.decision == VerificationDecision.ALLOW


def test_execution_manager_blocks_direct_pick_before_backend_call():
    backend = MockRobotBackend()
    backend.open_gripper()
    manager = ExecutionManager(
        SkillLibrary(), backend, hardware_mode=load_hardware_mode()
    )
    plan = RobotPlan(
        task="direct pick",
        steps=[PlanStep(step_id="1", skill_id="pick_object", description="pick", parameters={"object_id": "cube_1"})],
    )

    report = manager.execute_plan(
        plan,
        VerificationResult(decision=VerificationDecision.ALLOW),
        "approved",
    )

    assert report.status == "FAILED"
    assert backend.held_object is None


class RestrictedArmBackend:
    execution_mode = "restricted_real_motion"

    def __init__(self):
        self.pregrasp_calls = 0

    def plan_to_pregrasp(self, **kwargs):
        self.pregrasp_calls += 1
        return {
            "success": True,
            "message": "pregrasp executed",
            "output": {"target": kwargs.get("object_id")},
        }


def test_restricted_real_arm_step_requires_per_step_execute_confirmation(tmp_path):
    backend = RestrictedArmBackend()
    logger = AuditLogger(str(tmp_path / "audit.log"), hardware_mode=load_hardware_mode())
    manager = ExecutionManager(
        SkillLibrary(), backend, logger, hardware_mode=load_hardware_mode()
    )
    plan = RobotPlan(
        task="pregrasp",
        steps=[
            PlanStep(
                step_id="7",
                skill_id="move_to_pregrasp",
                description="pregrasp",
                parameters={"object_id": "cube_1"},
            )
        ],
    )

    report = manager.execute_plan(
        plan,
        VerificationResult(decision=VerificationDecision.ALLOW),
        "approved",
        context={
            "arm_execution_mode": "restricted_real_motion",
            "arm_confirmation": "EXECUTE",
            "grasp_candidate_approved": True,
        },
    )

    assert report.status == "FAILED"
    assert backend.pregrasp_calls == 0
    assert "Exact EXECUTE" in report.step_results[0].message

    report = manager.execute_plan(
        plan,
        VerificationResult(decision=VerificationDecision.ALLOW),
        "approved",
        context={
            "arm_execution_mode": "restricted_real_motion",
            "arm_confirmations": {"7": "EXECUTE"},
            "grasp_candidate_approved": True,
        },
    )

    assert report.status == "SUCCESS"
    assert backend.pregrasp_calls == 1
    entries = [json.loads(line) for line in (tmp_path / "audit.log").read_text().splitlines()]
    assert any(entry["event"] == "restricted_real_motion_mode_active" for entry in entries)
    assert any(entry["event"] == "real_arm_motion_executed" for entry in entries)


def test_restricted_real_gripper_requires_per_step_yes_confirmation():
    backend = MockRobotBackend()
    manager = ExecutionManager(
        SkillLibrary(), backend, hardware_mode=load_hardware_mode()
    )
    plan = RobotPlan(
        task="open",
        steps=[
            PlanStep(
                step_id="6",
                skill_id="open_gripper",
                description="open",
                parameters={},
            )
        ],
    )

    report = manager.execute_plan(
        plan,
        VerificationResult(decision=VerificationDecision.ALLOW),
        "approved",
        context={"gripper_confirmation": "YES"},
    )

    assert report.status == "FAILED"
    assert "Exact YES" in report.step_results[0].message

    report = manager.execute_plan(
        plan,
        VerificationResult(decision=VerificationDecision.ALLOW),
        "approved",
        context={"gripper_confirmations": {"6": "YES"}},
    )

    assert report.status == "SUCCESS"


def test_audit_records_hardware_mode(tmp_path):
    log_file = tmp_path / "audit.log"
    logger = AuditLogger(str(log_file), hardware_mode=load_hardware_mode())

    logger.log_hardware_safety_event(
        "real_arm_execution_blocked", real_arm_execution_blocked=True
    )

    entry = json.loads(log_file.read_text().splitlines()[0])
    assert entry["hardware_mode"] == "real_camera_real_gripper_restricted_arm"
    assert entry["real_arm_execution_blocked"] is True
    assert "real_camera_used" in entry
    assert "real_gripper_command_requested" in entry
    assert "real_gripper_command_sent" in entry
    assert "grasp_candidate_reviewed" in entry


def test_grasp_candidate_review_requires_exact_yes_and_records_review(tmp_path):
    logger = AuditLogger(str(tmp_path / "audit.log"), hardware_mode=load_hardware_mode())
    manager = ExecutionManager(
        SkillLibrary(), MockRobotBackend(), logger, hardware_mode=load_hardware_mode()
    )
    plan = RobotPlan(
        task="review",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="review_grasp_candidate",
                description="review",
                parameters={"object_id": "cube_1"},
            )
        ],
    )
    context = {
        "selected_grasp_candidate": {"candidate_id": "g1", "score": 0.9},
        "grasp_candidate_confirmation": "YES",
    }

    report = manager.execute_plan(
        plan,
        VerificationResult(decision=VerificationDecision.ALLOW),
        "approved",
        context=context,
    )

    assert report.status == "SUCCESS"
    assert context["grasp_candidate_approved"] is True
    entries = [json.loads(line) for line in (tmp_path / "audit.log").read_text().splitlines()]
    assert any(entry["grasp_candidate_reviewed"] is True for entry in entries)


def test_hw5_staged_skills_are_registered():
    library = SkillLibrary()
    for skill_id in (
        "build_object_point_cloud",
        "generate_grasp_pose",
        "review_grasp_candidate",
        "test_gripper_width",
        "preview_grasp_motion",
        "move_to_pregrasp",
        "approach_grasp",
        "retreat_grasp",
    ):
        assert library.get_skill(skill_id) is not None
