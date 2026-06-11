import pytest
import yaml

from llm_skill_robot.robot.robotiq_2f140_joint_state_sim import (
    Robotiq2F140JointStateSim,
)
from llm_skill_robot.utils import get_config_dir


EXPECTED_GRIPPER_JOINTS = [
    "robotiq_finger_joint",
]


def _load_mapping_config():
    with open(get_config_dir() / "gripper_joint_mapping.yaml", "r") as f:
        return yaml.safe_load(f)


def test_gripper_joint_mapping_loads_active_finger_joint_name():
    config = _load_mapping_config()
    mapping = config["gripper_joint_mapping"]

    assert mapping["enabled"] is True
    assert mapping["mode"] == "rviz_sim"
    assert mapping["real_gripper_enabled"] is False
    assert mapping["publish_topic"] == "/joint_states"
    assert mapping["publish_rate_hz"] == 20.0
    assert mapping["joint_names"] == EXPECTED_GRIPPER_JOINTS


def test_open_width_maps_active_joint_to_open_position():
    config = _load_mapping_config()
    sim = Robotiq2F140JointStateSim(node=None, config=config)

    positions = sim.width_to_positions(width_m=0.14, max_width_m=0.14)

    assert set(positions) == set(EXPECTED_GRIPPER_JOINTS)
    assert all(position == pytest.approx(0.0) for position in positions.values())


def test_closed_width_maps_active_joint_to_closed_position():
    config = _load_mapping_config()
    sim = Robotiq2F140JointStateSim(node=None, config=config)

    positions = sim.width_to_positions(width_m=0.0, max_width_m=0.14)

    assert set(positions) == set(EXPECTED_GRIPPER_JOINTS)
    assert all(position == pytest.approx(0.7) for position in positions.values())


def test_half_width_maps_active_joint_halfway():
    config = _load_mapping_config()
    sim = Robotiq2F140JointStateSim(node=None, config=config)

    positions = sim.width_to_positions(width_m=0.07, max_width_m=0.14)

    assert set(positions) == set(EXPECTED_GRIPPER_JOINTS)
    assert all(position == pytest.approx(0.35) for position in positions.values())
