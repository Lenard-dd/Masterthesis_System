import pytest
from llm_skill_robot.agent.rviz_agent_factory import (
    grasp_motion_policy,
    validate_rviz_execution_config,
)


def test_rviz_factory_rejects_non_simulated_or_real_enabled_config():
    with pytest.raises(ValueError, match="rviz_sim"):
        validate_rviz_execution_config({"mode": "plan_only"})
    with pytest.raises(ValueError, match="real robot"):
        validate_rviz_execution_config({"mode": "rviz_sim", "allow_real_robot_execution": True})
    validate_rviz_execution_config({"mode": "rviz_sim", "allow_real_robot_execution": False})


def test_grasp_motion_policy_matches_supervised_pilz_stage_configuration():
    policy = grasp_motion_policy(
        {
            "plan_only_preview": {
                "velocity_scale": 0.05,
                "acceleration_scale": 0.05,
                "moveit": {
                    "pipeline_id": "pilz_industrial_motion_planner",
                    "stage_planner_ids": {"pregrasp": "PTP", "grasp": "LIN", "retreat": "LIN"},
                },
            }
        },
        max_velocity_scale=0.1,
        max_acceleration_scale=0.1,
    )
    assert policy.kwargs_for("move_to_pregrasp")["planner_id"] == "PTP"
    assert policy.kwargs_for("approach_grasp")["planner_id"] == "LIN"
    assert policy.kwargs_for("retreat_grasp")["pipeline_id"] == "pilz_industrial_motion_planner"
