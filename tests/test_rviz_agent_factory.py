import pytest
from llm_skill_robot.agent.rviz_agent_factory import validate_rviz_execution_config


def test_rviz_factory_rejects_non_simulated_or_real_enabled_config():
    with pytest.raises(ValueError, match="rviz_sim"):
        validate_rviz_execution_config({"mode": "plan_only"})
    with pytest.raises(ValueError, match="real robot"):
        validate_rviz_execution_config({"mode": "rviz_sim", "allow_real_robot_execution": True})
    validate_rviz_execution_config({"mode": "rviz_sim", "allow_real_robot_execution": False})
