import json

from llm_skill_robot.agent.agent_controller import AgentController
from llm_skill_robot.agent.agent_runtime import AgentRuntime
from llm_skill_robot.agent.agent_session import AgentSessionRunner
from llm_skill_robot.agent.agent_state import AgentState, AgentStatus
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.schemas import (
    ExecutionReport,
    StepExecutionResult,
    VerificationDecision,
    VerificationResult,
)


class SequencedLLMClient:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def generate_text(self, _messages):
        return json.dumps(self.decisions.pop(0))


class AllowingVerifier:
    def verify_plan(self, _plan, context=None):
        return VerificationResult(decision=VerificationDecision.ALLOW)


class SessionExecutor:
    def __init__(self, fail_all=False):
        self.tool_names = []
        self.fail_all = fail_all

    def execute_plan(self, plan, _verification, _human, context=None):
        step = plan.steps[0]
        self.tool_names.append(step.skill_id)
        success = not self.fail_all
        output = {}
        if step.skill_id == "detect_object":
            output = {"object_id": "cube_1"}
        if step.skill_id == "generate_grasp_pose":
            output = {"candidate_id": "grasp_1"}
        return ExecutionReport(
            task=plan.task,
            status="SUCCESS" if success else "FAILED",
            summary="done",
            step_results=[StepExecutionResult(
                step_id=int(step.step_id),
                skill_id=step.skill_id,
                status="SUCCESS" if success else "FAILED",
                message="ok" if success else "failed",
                output=output,
            )],
        )


def _runtime(decisions, executor):
    return AgentRuntime(
        AgentController(SequencedLLMClient(decisions)),
        AllowingVerifier(),
        executor,
        human_gate=lambda _plan, _proposal, _state: True,
    )


def test_session_mixes_primitive_tool_and_composite_skill(tmp_path):
    decisions = [
        {
            "kind": "tool_call", "message": "Move to observe.",
            "tool_call": {"tool_name": "move_to_named_target", "arguments": {"target_name": "observe"}},
        },
        {
            "kind": "composite_skill_call", "message": "Pick the object.",
            "composite_skill_call": {"skill_name": "safe_pick_object", "arguments": {"object_query": "red cube"}},
        },
        {"kind": "complete", "message": "Task completed."},
    ]
    executor = SessionExecutor()
    logger = AuditLogger(str(tmp_path / "agent.jsonl"))
    runner = AgentSessionRunner(
        _runtime(decisions, executor),
        composite_human_gate=lambda _skill, _state: True,
        audit_logger=logger,
    )
    state = AgentState(user_goal="observe, pick red cube, then finish")

    result = runner.run(state)

    assert result.status == AgentStatus.COMPLETED
    assert result.turns == 3
    assert result.composite_results[0].status.value == "completed"
    assert state.selected_object_id == "cube_1"
    assert state.selected_grasp_candidate_id == "grasp_1"
    assert executor.tool_names[0] == "move_to_named_target"
    assert "close_gripper" in executor.tool_names
    entries = (tmp_path / "agent.jsonl").read_text()
    assert "agent_decision" in entries
    assert "composite_skill_finished" in entries


def test_session_stops_after_configured_consecutive_failures():
    decisions = [{
        "kind": "tool_call", "message": "Observe.",
        "tool_call": {"tool_name": "detect_object", "arguments": {"query": "red cube"}},
    }]
    state = AgentState(user_goal="find red cube")
    result = AgentSessionRunner(_runtime(decisions, SessionExecutor(fail_all=True))).run(
        state, max_consecutive_failures=1
    )

    assert result.status == AgentStatus.BLOCKED
    assert "Maximum consecutive failures" in result.termination_reason
    assert state.total_failures == 1


def test_session_rejects_composite_skill_without_task_level_human_gate():
    decisions = [{
        "kind": "composite_skill_call", "message": "Pick.",
        "composite_skill_call": {"skill_name": "safe_pick_object", "arguments": {"object_query": "red cube"}},
    }]
    executor = SessionExecutor()
    result = AgentSessionRunner(_runtime(decisions, executor)).run(
        AgentState(user_goal="pick red cube"), max_consecutive_failures=1
    )

    assert result.status == AgentStatus.BLOCKED
    assert result.composite_results[0].status.value == "aborted"
    assert executor.tool_names == []
