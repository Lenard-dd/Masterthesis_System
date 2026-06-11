from llm_skill_robot.core.plan import PlanStep
from llm_skill_robot.robot.robotiq_2f140_sim_backend import Robotiq2F140SimBackend
from llm_skill_robot.ros_nl_rviz_sim_demo import execute_non_motion_skill


def test_open_gripper_step_dispatches_to_sim_backend():
    backend = Robotiq2F140SimBackend()
    backend.close_gripper()
    step = PlanStep(
        step_id="1",
        skill_id="open_gripper",
        description="Open simulated gripper",
        parameters={},
    )

    result = execute_non_motion_skill(step, backend, context={})

    assert result["success"] is True
    assert result["skill_id"] == "open_gripper"
    assert result["output"]["current_width_m"] == 0.14
    assert result["output"]["is_open"] is True
    assert result["output"]["mode"] == "rviz_sim"
    assert result["output"]["real_gripper_enabled"] is False


def test_close_gripper_step_dispatches_to_sim_backend():
    backend = Robotiq2F140SimBackend()
    step = PlanStep(
        step_id="2",
        skill_id="close_gripper",
        description="Close simulated gripper",
        parameters={},
    )

    result = execute_non_motion_skill(step, backend, context={})

    assert result["success"] is True
    assert result["skill_id"] == "close_gripper"
    assert result["output"]["current_width_m"] == 0.0
    assert result["output"]["is_open"] is False
    assert result["output"]["real_gripper_enabled"] is False


def test_invalid_gripper_width_fails_clearly():
    backend = Robotiq2F140SimBackend()
    step = PlanStep(
        step_id="3",
        skill_id="set_gripper_width",
        description="Set invalid simulated gripper width",
        parameters={"width_m": 0.20},
    )

    result = execute_non_motion_skill(step, backend, context={})

    assert result["success"] is False
    assert "outside allowed range" in result["message"]
    assert result["output"]["current_width_m"] == 0.14
    assert result["output"]["real_gripper_enabled"] is False


def test_real_gripper_enabled_remains_false_for_state_query():
    backend = Robotiq2F140SimBackend()
    step = PlanStep(
        step_id="4",
        skill_id="get_gripper_state",
        description="Read simulated gripper state",
        parameters={},
    )

    result = execute_non_motion_skill(step, backend, context={})

    assert result["success"] is True
    assert result["output"]["mode"] == "rviz_sim"
    assert result["output"]["real_gripper_enabled"] is False
    assert backend.real_gripper_enabled is False
