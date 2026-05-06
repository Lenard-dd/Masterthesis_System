import pytest
from llm_skill_robot.robot.mock_robot_backend import MockRobotBackend


def test_move_to_named_target_known():
    backend = MockRobotBackend()
    result = backend.move_to_named_target("home")
    assert result["success"] is True
    assert backend.current_named_target == "home"
    assert result["output"]["target"] == "home"


def test_move_to_named_target_unknown():
    backend = MockRobotBackend()
    result = backend.move_to_named_target("unknown")
    assert result["success"] is False
    assert backend.current_named_target is None
    assert "Unknown named target" in result["message"]


def test_pick_object_requires_gripper_open():
    backend = MockRobotBackend()
    result = backend.pick_object("cube_1")
    assert result["success"] is False
    assert backend.held_object is None
    assert backend.gripper_open is False

    open_result = backend.open_gripper()
    assert open_result["success"] is True

    pick_result = backend.pick_object("cube_1")
    assert pick_result["success"] is True
    assert backend.held_object == "cube_1"
    assert backend.gripper_open is False


def test_place_object_fails_when_no_object_held():
    backend = MockRobotBackend()
    result = backend.place_object("cube_1")
    assert result["success"] is False
    assert backend.held_object is None


def test_pick_then_place_succeeds():
    backend = MockRobotBackend()
    backend.open_gripper()
    pick_result = backend.pick_object("cube_1")
    assert pick_result["success"] is True

    place_result = backend.place_object("cube_1")
    assert place_result["success"] is True
    assert backend.held_object is None
    assert backend.gripper_open is True
