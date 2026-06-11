from llm_skill_robot.robot.robotiq_2f140_sim_backend import Robotiq2F140SimBackend
from llm_skill_robot.robot.robotiq_2f140_joint_state_sim import Robotiq2F140JointStateSim


def test_initial_state_is_default_open():
    backend = Robotiq2F140SimBackend()

    assert backend.current_width_m == 0.14
    assert backend.is_open is True
    assert backend.real_gripper_enabled is False


def test_open_gripper_sets_width_to_max():
    backend = Robotiq2F140SimBackend()
    backend.close_gripper()

    result = backend.open_gripper()

    assert result["success"] is True
    assert result["output"]["current_width_m"] == 0.14
    assert result["output"]["is_open"] is True
    assert result["output"]["mode"] == "rviz_sim"
    assert result["output"]["real_gripper_enabled"] is False


def test_close_gripper_sets_width_to_min():
    backend = Robotiq2F140SimBackend()

    result = backend.close_gripper()

    assert result["success"] is True
    assert result["output"]["current_width_m"] == 0.0
    assert result["output"]["is_open"] is False


def test_set_gripper_width_midpoint_succeeds():
    backend = Robotiq2F140SimBackend()

    result = backend.set_gripper_width(0.07)

    assert result["success"] is True
    assert result["output"]["current_width_m"] == 0.07
    assert result["output"]["is_open"] is False
    assert backend.last_command["command"] == "set_gripper_width"


def test_set_gripper_width_below_min_fails():
    backend = Robotiq2F140SimBackend()

    result = backend.set_gripper_width(-0.01)

    assert result["success"] is False
    assert "outside allowed range" in result["message"]
    assert result["output"]["current_width_m"] == 0.14


def test_set_gripper_width_above_max_fails():
    backend = Robotiq2F140SimBackend()

    result = backend.set_gripper_width(0.20)

    assert result["success"] is False
    assert "outside allowed range" in result["message"]
    assert result["output"]["current_width_m"] == 0.14


def test_get_gripper_state_returns_current_state():
    backend = Robotiq2F140SimBackend()
    backend.set_gripper_width(0.07)

    result = backend.get_gripper_state()

    assert result["success"] is True
    assert result["output"]["current_width_m"] == 0.07
    assert result["output"]["is_open"] is False
    assert result["output"]["mode"] == "rviz_sim"


def test_backend_includes_visual_joint_state_when_simulator_is_provided():
    config = {
        "gripper_joint_mapping": {
            "enabled": True,
            "joint_names": ["robotiq_finger_joint"],
            "open_positions": {"robotiq_finger_joint": 0.0},
            "closed_positions": {"robotiq_finger_joint": 0.7},
            "max_width_m": 0.14,
            "min_width_m": 0.0,
        }
    }
    joint_state_sim = Robotiq2F140JointStateSim(node=None, config=config)
    backend = Robotiq2F140SimBackend(joint_state_sim=joint_state_sim)

    result = backend.set_gripper_width(0.07)

    assert result["success"] is True
    assert result["output"]["joint_positions"]["robotiq_finger_joint"] == 0.35
    assert result["output"]["real_gripper_enabled"] is False
