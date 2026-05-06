import pytest
from llm_skill_robot.core.schemas import ExecutionReport, StepExecutionResult


def test_step_execution_result_valid():
    result = StepExecutionResult(
        step_id=1,
        skill_id="move_to_named_target",
        status="SUCCESS",
        message="Reached target successfully.",
        output={"duration": 3.4}
    )

    assert result.step_id == 1
    assert result.skill_id == "move_to_named_target"
    assert result.status == "SUCCESS"
    assert result.output["duration"] == 3.4


def test_execution_report_valid():
    report = ExecutionReport(
        task="Example task",
        status="SUCCESS",
        step_results=[
            StepExecutionResult(
                step_id=1,
                skill_id="open_gripper",
                status="SUCCESS",
                message="Gripper opened.",
                output={}
            )
        ],
        summary="All steps completed successfully."
    )

    assert report.task == "Example task"
    assert report.status == "SUCCESS"
    assert len(report.step_results) == 1
    assert report.summary.startswith("All steps")


def test_step_execution_result_invalid_status():
    with pytest.raises(ValueError):
        StepExecutionResult(
            step_id=2,
            skill_id="close_gripper",
            status="DONE",
            message="Invalid status value.",
            output={}
        )


def test_execution_report_invalid_status():
    with pytest.raises(ValueError):
        ExecutionReport(
            task="Example task",
            status="PARTIAL",
            step_results=[],
            summary="Invalid report status."
        )
