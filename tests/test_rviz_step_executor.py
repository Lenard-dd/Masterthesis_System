from llm_skill_robot.agent.rviz_execution_context import RvizExecutionContext
from llm_skill_robot.agent.rviz_step_executor import RvizStepExecutor
from llm_skill_robot.core.plan import PlanStep


def _context():
    return RvizExecutionContext(*(object() for _ in range(8)), execution_config={}, safety_config={})


def test_step_executor_dispatches_motion_and_persists_output():
    called = []
    executor = RvizStepExecutor(
        _context(),
        non_motion_executor=lambda step, state: called.append(("non", step.skill_id)) or {"success": True, "message": "ok", "output": {"object_id": "cube_1"}},
        motion_executor=lambda step, ctx: called.append(("motion", step.skill_id)) or {"success": True, "message": "ok", "output": {}},
    )
    executor.execute_step(PlanStep(step_id="1", skill_id="detect_object", description="", parameters={}), {})
    executor.execute_step(PlanStep(step_id="2", skill_id="move_to_named_target", description="", parameters={}), {})
    assert called == [("non", "detect_object"), ("motion", "move_to_named_target")]
    assert executor.context.runtime_state["object_id"] == "cube_1"
