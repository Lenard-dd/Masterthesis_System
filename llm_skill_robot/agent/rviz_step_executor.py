"""Stateful dispatch of Primitive Tools onto existing RViz execution helpers."""

from __future__ import annotations

from typing import Any, Callable

from llm_skill_robot.agent.rviz_execution_context import RvizExecutionContext


class RvizStepExecutor:
    """Dispatch one reviewed Tool while retaining shared RViz session state."""

    MOTION_SKILLS = {"move_to_named_target", "move_to_pregrasp", "approach_grasp", "retreat_grasp"}

    def __init__(
        self,
        context: RvizExecutionContext,
        *,
        non_motion_executor: Callable[[Any, dict[str, Any]], dict[str, Any]],
        motion_executor: Callable[[Any, RvizExecutionContext], dict[str, Any]],
        candidate_reviewer: Callable[[Any, RvizExecutionContext], dict[str, Any]] | None = None,
    ):
        self.context = context
        self.non_motion_executor = non_motion_executor
        self.motion_executor = motion_executor
        self.candidate_reviewer = candidate_reviewer

    def execute_step(self, step, agent_context: dict[str, Any]) -> dict[str, Any]:
        runtime_state = self.context.merge_agent_context(agent_context)
        if step.skill_id == "review_grasp_candidate" and self.candidate_reviewer:
            result = self.candidate_reviewer(step, self.context)
        elif step.skill_id in self.MOTION_SKILLS:
            result = self.motion_executor(step, self.context)
        else:
            result = self.non_motion_executor(step, runtime_state)
        if not isinstance(result, dict):
            return {"success": False, "message": "RViz step executor returned invalid result.", "output": {}}
        output = result.get("output", {})
        if isinstance(output, dict):
            self.context.runtime_state.update({
                key: value for key, value in output.items()
                if key in {"object_id", "point_cloud", "point_cloud_path", "candidate_id", "candidate", "grasp_candidates"}
            })
        return result
