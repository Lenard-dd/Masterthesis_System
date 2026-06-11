from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.plan import PlanStep
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.ros_nl_rviz_sim_demo import (
    _audit_perception_result,
    _verify_object_step_for_execution,
    execute_non_motion_skill,
)


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
