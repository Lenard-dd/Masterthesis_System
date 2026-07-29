from llm_skill_robot.agent.rviz_execution_context import RvizExecutionContext


def test_rviz_context_preserves_runtime_state_across_agent_steps():
    context = RvizExecutionContext(
        node=object(), motion_backend=object(), perception_pipeline=object(),
        point_cloud_builder=object(), grasp_planner=object(), gripper_backend=object(),
        trajectory_validator=object(), trajectory_reviewer=object(),
        execution_config={"mode": "rviz_sim"}, safety_config={},
    )
    context.merge_agent_context({"resolved_object_id": "cube_1"})
    state = context.merge_agent_context({"grasp_candidate_approved": True})
    assert state["resolved_object_id"] == "cube_1"
    assert state["grasp_candidate_approved"] is True
