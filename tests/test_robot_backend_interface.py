from llm_skill_robot.robot.base_robot_backend import BaseRobotBackend
from llm_skill_robot.robot.mock_robot_backend import MockRobotBackend


def test_base_robot_backend_interface_exists():
    """Test that BaseRobotBackend has all required methods."""
    required_methods = [
        "get_robot_state",
        "list_named_targets",
        "move_to_named_target",
        "open_gripper",
        "close_gripper",
        "pick_object",
        "place_object",
        "stop_motion",
    ]
    
    for method_name in required_methods:
        assert hasattr(BaseRobotBackend, method_name), f"BaseRobotBackend missing method: {method_name}"


def test_mock_backend_implements_all_methods():
    """Test that MockRobotBackend implements all BaseRobotBackend methods."""
    backend = MockRobotBackend()
    
    assert callable(backend.get_robot_state)
    assert callable(backend.list_named_targets)
    assert callable(backend.move_to_named_target)
    assert callable(backend.open_gripper)
    assert callable(backend.close_gripper)
    assert callable(backend.pick_object)
    assert callable(backend.place_object)
    assert callable(backend.stop_motion)


def test_mock_backend_get_robot_state():
    """Test get_robot_state returns the correct state."""
    backend = MockRobotBackend()
    state = backend.get_robot_state()
    
    assert "current_named_target" in state
    assert "gripper_open" in state
    assert "held_object" in state
    assert state["gripper_open"] is False
    assert state["held_object"] is None


def test_mock_backend_list_named_targets():
    """Test list_named_targets returns available targets."""
    backend = MockRobotBackend()
    targets = backend.list_named_targets()
    
    assert isinstance(targets, list)
    assert "home" in targets
    assert "observe" in targets
    assert "pickup" not in targets
    assert "place" not in targets


def test_mock_backend_stop_motion():
    """Test stop_motion returns success."""
    backend = MockRobotBackend()
    result = backend.stop_motion()
    
    assert result["success"] is True
    assert "message" in result
    assert "output" in result
