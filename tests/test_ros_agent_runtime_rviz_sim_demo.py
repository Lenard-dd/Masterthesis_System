from llm_skill_robot.ros_agent_runtime_rviz_sim_demo import RVIZ_AGENT_WARNING


def test_rviz_agent_entrypoint_is_explicitly_simulation_only():
    assert "SIMULATION ONLY" in RVIZ_AGENT_WARNING
