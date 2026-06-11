from copy import deepcopy

import pytest
import yaml

from llm_skill_robot.robot.robotiq_2f140_joint_state_sim import Robotiq2F140JointStateSim
from llm_skill_robot.utils import get_config_dir


def _load_config():
    with open(get_config_dir() / "gripper_joint_mapping.yaml", "r") as f:
        return yaml.safe_load(f)


def test_invalid_width_below_min_fails():
    sim = Robotiq2F140JointStateSim(node=None, config=_load_config())

    result = sim.set_width(-0.01)

    assert result["success"] is False
    assert "below minimum" in result["message"]
    assert result["output"]["real_gripper_enabled"] is False


def test_invalid_width_above_max_fails():
    sim = Robotiq2F140JointStateSim(node=None, config=_load_config())

    result = sim.set_width(0.20)

    assert result["success"] is False
    assert "above maximum" in result["message"]
    assert result["output"]["real_gripper_enabled"] is False


def test_get_state_returns_current_width_and_joint_positions():
    sim = Robotiq2F140JointStateSim(node=None, config=_load_config())
    sim.set_width(0.07)

    result = sim.get_state()

    assert result["success"] is True
    assert result["output"]["current_width_m"] == pytest.approx(0.07)
    assert result["output"]["joint_positions"]
    assert result["output"]["mode"] == "rviz_sim"
    assert result["output"]["real_gripper_enabled"] is False


def test_missing_joint_names_fail_clearly():
    config = deepcopy(_load_config())
    config["gripper_joint_mapping"]["joint_names"] = []
    sim = Robotiq2F140JointStateSim(node=None, config=config)

    result = sim.set_width(0.07)

    assert result["success"] is False
    assert "No gripper joint names configured" in result["message"]


def test_missing_open_position_fails_clearly():
    config = deepcopy(_load_config())
    joint_name = config["gripper_joint_mapping"]["joint_names"][0]
    del config["gripper_joint_mapping"]["open_positions"][joint_name]
    sim = Robotiq2F140JointStateSim(node=None, config=config)

    result = sim.set_width(0.07)

    assert result["success"] is False
    assert "Missing open_positions" in result["message"]


def test_real_gripper_enabled_is_never_true():
    sim = Robotiq2F140JointStateSim(node=None, config=_load_config())

    assert sim.real_gripper_enabled is False
    assert sim.close()["output"]["real_gripper_enabled"] is False
    assert sim.open()["output"]["real_gripper_enabled"] is False
