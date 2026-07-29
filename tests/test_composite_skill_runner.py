from llm_skill_robot.agent.agent_controller import AgentController
from llm_skill_robot.agent.agent_runtime import AgentRuntime
from llm_skill_robot.agent.agent_state import AgentState
from llm_skill_robot.agent.composite_skills import (
    CompositeSkillRunner,
    CompositeSkillStatus,
    SafePickObjectSkill,
)
from llm_skill_robot.core.schemas import (
    ExecutionReport,
    StepExecutionResult,
    ValidationDecision,
    VerificationDecision,
    VerificationResult,
)


class NoLLMClient:
    def generate_text(self, _messages):
        raise AssertionError("Composite Skill calls must not ask the LLM for each stage.")


class AllowingVerifier:
    def verify_plan(self, _plan, context=None):
        return VerificationResult(decision=VerificationDecision.ALLOW)


class WorkflowExecutor:
    def __init__(self, fail_tool=None):
        self.fail_tool = fail_tool
        self.tool_names = []

    def execute_plan(self, plan, _verification, _human_decision, context=None):
        step = plan.steps[0]
        self.tool_names.append(step.skill_id)
        success = step.skill_id != self.fail_tool
        output = {}
        if step.skill_id == "detect_object":
            output = {"object_id": "cube_1"}
        elif step.skill_id == "generate_grasp_pose":
            output = {"candidate_id": "grasp_1"}
        return ExecutionReport(
            task=plan.task,
            status="SUCCESS" if success else "FAILED",
            summary="stage complete",
            step_results=[StepExecutionResult(
                step_id=int(step.step_id),
                skill_id=step.skill_id,
                status="SUCCESS" if success else "FAILED",
                message="ok" if success else "simulated failure",
                output=output,
            )],
        )


def _runner(executor):
    runtime = AgentRuntime(
        AgentController(NoLLMClient()),
        AllowingVerifier(),
        executor,
        human_gate=lambda _plan, _proposal, _state: ValidationDecision.APPROVE.value,
    )
    return CompositeSkillRunner(runtime)


def test_runner_executes_each_safe_pick_stage_through_agent_runtime():
    executor = WorkflowExecutor()
    skill = SafePickObjectSkill()
    skill.start({"object_query": "red cube"})
    state = AgentState(user_goal="pick the red cube")

    result = _runner(executor).run_until_terminal(skill, state)

    assert result.status == CompositeSkillStatus.COMPLETED
    assert executor.tool_names == [
        "move_to_named_target", "detect_object", "build_object_point_cloud",
        "generate_grasp_pose", "review_grasp_candidate", "open_gripper",
        "move_to_pregrasp", "approach_grasp", "close_gripper", "retreat_grasp",
        "verify_grasp",
    ]
    assert [call.tool_name for call in state.proposed_calls] == executor.tool_names
    assert len(state.tool_results) == len(executor.tool_names)


def test_runner_stops_after_a_primitive_tool_failure():
    executor = WorkflowExecutor(fail_tool="generate_grasp_pose")
    skill = SafePickObjectSkill()
    skill.start({"object_query": "red cube"})

    result = _runner(executor).run_until_terminal(
        skill, AgentState(user_goal="pick the red cube")
    )

    assert result.status == CompositeSkillStatus.FAILED
    assert executor.tool_names[-1] == "generate_grasp_pose"
    assert "move_to_pregrasp" not in executor.tool_names
