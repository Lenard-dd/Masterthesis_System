import pytest
from llm_skill_robot.robot.base_robot_backend import BaseRobotBackend
from llm_skill_robot.robot.base_ros_backend import BaseROSBackend
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
    assert "pickup" in targets
    assert "place" in targets


def test_mock_backend_stop_motion():
    """Test stop_motion returns success."""
    backend = MockRobotBackend()
    result = backend.stop_motion()
    
    assert result["success"] is True
    assert "message" in result
    assert "output" in result


def test_base_ros_backend_initialization():
    """Test BaseROSBackend initialization without ROS running."""
    backend = BaseROSBackend(robot_name="test_robot")
    
    assert backend.robot_name == "test_robot"
    assert backend._is_shutdown is False
    state = backend.get_robot_state()
    assert state["robot_name"] == "test_robot"
    assert state["is_shutdown"] is False


def test_base_ros_backend_list_named_targets():
    """Test BaseROSBackend provides default target list."""
    backend = BaseROSBackend()
    targets = backend.list_named_targets()
    
    assert "home" in targets
    assert "pickup" in targets
    assert "place" in targets


def test_base_ros_backend_not_implemented_methods():
    """Test that BaseROSBackend methods raise NotImplementedError."""
    backend = BaseROSBackend()
    
    with pytest.raises(NotImplementedError):
        backend.move_to_named_target("home")
    
    with pytest.raises(NotImplementedError):
        backend.open_gripper()
    
    with pytest.raises(NotImplementedError):
        backend.close_gripper()
    
    with pytest.raises(NotImplementedError):
        backend.pick_object("obj")
    
    with pytest.raises(NotImplementedError):
        backend.place_object("obj")
    
    with pytest.raises(NotImplementedError):
        backend.stop_motion()
