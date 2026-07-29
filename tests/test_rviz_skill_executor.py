from llm_skill_robot.agent.rviz_skill_executor import RvizSkillExecutor
from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.core.schemas import ValidationDecision, ValidationResult, VerificationDecision, VerificationResult


def test_rviz_executor_runs_one_step_and_preserves_context():
    seen = []
    def execute(step, context):
        seen.append((step.skill_id, dict(context)))
        context["object_id"] = "cube_1"
        return {"success": True, "message": "RViz step completed.", "output": {"object_id": "cube_1"}}
    executor = RvizSkillExecutor(execute, {"mode": "rviz_sim"})
    plan = RobotPlan(task="detect", steps=[PlanStep(step_id="1", skill_id="detect_object", description="detect", parameters={})])
    report = executor.execute_plan(plan, VerificationResult(decision=VerificationDecision.ALLOW), ValidationResult(decision=ValidationDecision.APPROVE), {"session": "x"})
    assert report.status == "SUCCESS"
    assert seen[0][1]["session"] == "x"
    assert executor.context["object_id"] == "cube_1"


def test_rviz_executor_refuses_multi_step_plan():
    executor = RvizSkillExecutor(lambda *_: {"success": True, "message": "no"})
    plan = RobotPlan(task="bad", steps=[
        PlanStep(step_id="1", skill_id="a", description="a"),
        PlanStep(step_id="2", skill_id="b", description="b"),
    ])
    report = executor.execute_plan(plan, VerificationResult(decision=VerificationDecision.ALLOW), "approved")
    assert report.status == "FAILED"
