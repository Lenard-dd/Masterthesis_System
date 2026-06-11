import pytest
import yaml

from llm_skill_robot.robot.robotiq_2f140_joint_state_sim import Robotiq2F140JointStateSim
from llm_skill_robot.utils import get_config_dir


EXPECTED_GRIPPER_JOINTS = [
    "robotiq_finger_joint",
]


def _load_config():
    with open(get_config_dir() / "gripper_joint_mapping.yaml", "r") as f:
        return yaml.safe_load(f)


def test_config_loads_active_finger_joint_name():
    mapping = _load_config()["gripper_joint_mapping"]

    assert mapping["joint_names"] == EXPECTED_GRIPPER_JOINTS
    assert mapping["enabled"] is True
    assert mapping["mode"] == "rviz_sim"
    assert mapping["real_gripper_enabled"] is False


def test_open_width_maps_active_joint_to_zero():
    sim = Robotiq2F140JointStateSim(node=None, config=_load_config())

    positions = sim.width_to_positions(0.14)

    assert set(positions) == set(EXPECTED_GRIPPER_JOINTS)
    assert all(value == pytest.approx(0.0) for value in positions.values())


def test_closed_width_maps_active_joint_to_point_seven():
    sim = Robotiq2F140JointStateSim(node=None, config=_load_config())

    positions = sim.width_to_positions(0.0)

    assert set(positions) == set(EXPECTED_GRIPPER_JOINTS)
    assert all(value == pytest.approx(0.7) for value in positions.values())


def test_half_width_maps_active_joint_halfway():
    sim = Robotiq2F140JointStateSim(node=None, config=_load_config())

    positions = sim.width_to_positions(0.07)

    assert set(positions) == set(EXPECTED_GRIPPER_JOINTS)
    assert all(value == pytest.approx(0.35) for value in positions.values())
