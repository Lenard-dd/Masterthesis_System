import json

from llm_skill_robot.agent.agent_controller import AgentController
from llm_skill_robot.agent.agent_runtime import AgentRuntime
from llm_skill_robot.agent.agent_state import AgentState
from llm_skill_robot.core.schemas import (
    ExecutionReport,
    StepExecutionResult,
    ValidationDecision,
    VerificationDecision,
    VerificationResult,
)


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    def generate_text(self, _messages):
        return self.response


class AllowingVerifier:
    def __init__(self):
        self.calls = []

    def verify_plan(self, plan, context=None):
        self.calls.append((plan, context))
        return VerificationResult(decision=VerificationDecision.ALLOW)


class BlockingVerifier:
    def verify_plan(self, _plan, context=None):
        return VerificationResult(
            decision=VerificationDecision.BLOCK, reasons=["Scene is unavailable."]
        )


class FakeExecutionManager:
    def __init__(self):
        self.calls = []

    def execute_plan(self, plan, verification, human_decision, context=None):
        self.calls.append((plan, verification, human_decision, context))
        step = plan.steps[0]
        return ExecutionReport(
            task=plan.task,
            status="SUCCESS",
            summary="Executed one safe step.",
            step_results=[
                StepExecutionResult(
                    step_id=int(step.step_id),
                    skill_id=step.skill_id,
                    status="SUCCESS",
                    message="Observation completed.",
                    output={"object_id": "cube_1"},
                )
            ],
        )


def _controller_for(tool_name, arguments):
    return AgentController(FakeLLMClient(json.dumps({
        "kind": "tool_call",
        "message": f"Use {tool_name}.",
        "tool_call": {"tool_name": tool_name, "arguments": arguments},
    })))


def test_runtime_verifies_executes_and_feeds_back_after_task_intent_approval():
    executor = FakeExecutionManager()
    runtime = AgentRuntime(
        _controller_for("detect_object", {"query": "red cube"}),
        AllowingVerifier(),
        executor,
        human_gate=lambda _plan, _proposal, _state: True,
    )
    state = AgentState(user_goal="find red cube")

    turn = runtime.run_turn(state)

    assert len(executor.calls) == 1
    assert turn.human_decision.decision == ValidationDecision.APPROVE
    assert turn.tool_result.success is True
    assert state.context["last_tool_result"]["output"]["object_id"] == "cube_1"
    assert state.context["task_intent_approved"] is True


def test_runtime_never_executes_a_verifier_blocked_tool():
    executor = FakeExecutionManager()
    runtime = AgentRuntime(
        _controller_for("detect_object", {"query": "red cube"}),
        BlockingVerifier(),
        executor,
    )

    turn = runtime.run_turn(AgentState(user_goal="find red cube"))

    assert executor.calls == []
    assert turn.tool_result.success is False
    assert turn.tool_result.output["reasons"] == ["Scene is unavailable."]


def test_runtime_rejects_high_risk_tool_when_no_human_gate_is_configured():
    executor = FakeExecutionManager()
    runtime = AgentRuntime(
        _controller_for("open_gripper", {}), AllowingVerifier(), executor
    )

    turn = runtime.run_turn(AgentState(user_goal="open gripper"))

    assert executor.calls == []
    assert turn.human_decision.decision == ValidationDecision.REJECT
    assert turn.tool_result.success is False


def test_runtime_passes_explicit_human_approval_to_execution_manager():
    executor = FakeExecutionManager()
    runtime = AgentRuntime(
        _controller_for("open_gripper", {}),
        AllowingVerifier(),
        executor,
        human_gate=lambda _plan, _proposal, _state: True,
    )

    turn = runtime.run_turn(AgentState(user_goal="open gripper"))

    assert len(executor.calls) == 1
    assert turn.human_decision.decision == ValidationDecision.APPROVE
