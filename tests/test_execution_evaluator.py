from llm_skill_robot.core.execution_evaluator import ExecutionEvaluator
from llm_skill_robot.core.plan import RobotPlan, PlanStep
from llm_skill_robot.core.schemas import ExecutionReport, StepExecutionResult


def test_successful_report_task_success_true():
    evaluator = ExecutionEvaluator()
    plan = RobotPlan(
        task="Test",
        steps=[PlanStep(step_id="1", skill_id="open_gripper", description="Open", parameters={})]
    )
    report = ExecutionReport(
        task="Test",
        status="SUCCESS",
        step_results=[
            StepExecutionResult(step_id=1, skill_id="open_gripper", status="SUCCESS", message="Done", output={})
        ],
        summary="All good."
    )

    result = evaluator.evaluate(plan, report)
    assert result["task_success"] is True
    assert result["num_steps"] == 1
    assert result["num_successful_steps"] == 1
    assert result["failed_steps"] == []
    assert "succeeded" in result["summary"]


def test_failed_report_task_success_false():
    evaluator = ExecutionEvaluator()
    plan = RobotPlan(
        task="Test",
        steps=[
            PlanStep(step_id="1", skill_id="open_gripper", description="Open", parameters={}),
            PlanStep(step_id="2", skill_id="pick_object", description="Pick", parameters={"object_id": "cube_1"}),
        ]
    )
    report = ExecutionReport(
        task="Test",
        status="FAILED",
        step_results=[
            StepExecutionResult(step_id=1, skill_id="open_gripper", status="SUCCESS", message="Done", output={}),
            StepExecutionResult(step_id=2, skill_id="pick_object", status="FAILED", message="Missing object.", output={}),
        ],
        summary="One failure."
    )

    result = evaluator.evaluate(plan, report)
    assert result["task_success"] is False
    assert result["num_steps"] == 2
    assert result["num_successful_steps"] == 1
    assert len(result["failed_steps"]) == 1
    assert result["failed_steps"][0]["step_id"] == 2
    assert result["failed_steps"][0]["skill_id"] == "pick_object"
    assert "failed" in result["summary"]


def test_failed_steps_are_reported_correctly():
    evaluator = ExecutionEvaluator()
    plan = RobotPlan(
        task="Test",
        steps=[
            PlanStep(step_id="1", skill_id="open_gripper", description="Open", parameters={}),
            PlanStep(step_id="2", skill_id="pick_object", description="Pick", parameters={"object_id": "cube_1"}),
        ]
    )
    report = ExecutionReport(
        task="Test",
        status="FAILED",
        step_results=[
            StepExecutionResult(step_id=1, skill_id="open_gripper", status="FAILED", message="Open failed.", output={}),
            StepExecutionResult(step_id=2, skill_id="pick_object", status="SKIPPED", message="Skipped after failure.", output={}),
        ],
        summary="First step failed."
    )

    result = evaluator.evaluate(plan, report)
    assert result["task_success"] is False
    assert len(result["failed_steps"]) == 1
    assert result["failed_steps"][0] == {
        "step_id": 1,
        "skill_id": "open_gripper",
        "message": "Open failed.",
    }
