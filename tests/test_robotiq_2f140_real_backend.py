import importlib
import sys
import types

import pytest

from llm_skill_robot.robot.robotiq_2f140_real_backend import (
    DRIVER_NOT_CONFIGURED_MESSAGE,
    Robotiq2F140RealBackend,
)


def _config(interface_type="auto"):
    return {
        "gripper": {
            "default_open_width_m": 0.14,
            "safe_close_width_m": 0.04,
            "default_speed_mps": 0.02,
            "default_force_N": 10.0,
            "real_gripper_enabled": True,
            "driver": {
                "interface_type": interface_type,
                "command_topic": "",
                "state_topic": "",
                "action_name": "",
                "service_name": "",
                "width_unit": "meters",
            },
        }
    }


def test_close_gripper_default_uses_safe_close_width():
    result = Robotiq2F140RealBackend(config=_config("dry_run")).close_gripper(
        confirmed=True
    )

    assert result["status"] == "DRY_RUN"
    assert result["output"]["command_preview"]["width_m"] == 0.04
    assert result["output"]["command_sent"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width_m": -0.01}, "below minimum"),
        ({"width_m": 0.15}, "exceeds maximum"),
        ({"width_m": 0.04, "speed_mps": 0.04}, "speed_mps"),
        ({"width_m": 0.04, "force_N": 21.0}, "force_N"),
    ],
)
def test_unsafe_commands_are_rejected_before_driver(kwargs, message):
    backend = Robotiq2F140RealBackend(config=_config("dry_run"))

    result = backend.set_gripper_width(confirmed=True, **kwargs)

    assert result["success"] is False
    assert result["status"] == "REJECTED"
    assert message in result["message"]
    assert backend.last_command is None


def test_human_confirmation_is_required_before_driver():
    result = Robotiq2F140RealBackend(config=_config("dry_run")).open_gripper()

    assert result["success"] is False
    assert result["status"] == "REJECTED"
    assert "Human confirmation is required" in result["message"]


def test_auto_interface_returns_not_available():
    result = Robotiq2F140RealBackend(config=_config("auto")).open_gripper(
        confirmed=True
    )

    assert result["success"] is False
    assert result["status"] == "NOT_AVAILABLE"
    assert result["message"] == DRIVER_NOT_CONFIGURED_MESSAGE
    assert result["output"]["command_sent"] is False


def test_partial_config_still_uses_hardware_mode_enablement():
    backend = Robotiq2F140RealBackend(
        config={"driver": {"interface_type": "auto"}}
    )

    result = backend.open_gripper(confirmed=True)

    assert result["status"] == "NOT_AVAILABLE"
    assert result["message"] == DRIVER_NOT_CONFIGURED_MESSAGE


def test_dry_run_does_not_send_real_command():
    class NodeThatMustNotBeUsed:
        def create_publisher(self, *args, **kwargs):
            raise AssertionError("dry_run attempted to create a ROS publisher")

    backend = Robotiq2F140RealBackend(
        node=NodeThatMustNotBeUsed(), config=_config("dry_run")
    )
    result = backend.set_gripper_width(0.07, confirmed=True)

    assert result["success"] is False
    assert result["status"] == "DRY_RUN"
    assert result["output"]["command_sent"] is False
    assert backend.last_command is None


def test_get_state_without_feedback_topic_reports_unavailable():
    result = Robotiq2F140RealBackend(config=_config()).get_gripper_state()

    assert result["success"] is False
    assert result["output"]["mode"] == "real_hardware"
    assert result["output"]["feedback_available"] is False
    assert result["output"]["real_gripper_enabled"] is True


def test_modules_import_without_ros_graph():
    assert importlib.import_module(
        "llm_skill_robot.robot.robotiq_2f140_real_backend"
    ).Robotiq2F140RealBackend
    assert callable(
        importlib.import_module("llm_skill_robot.ros_check_real_gripper").main
    )


def _install_fake_gripper_ros(monkeypatch):
    class FakeFloat32:
        def __init__(self, data=0.0):
            self.data = data

    class FakeFloat64:
        def __init__(self, data=0.0):
            self.data = data

    class FakeJointState:
        def __init__(self, name=None, position=None):
            self.name = list(name or [])
            self.position = list(position or [])

    class FakeGoal:
        def __init__(self):
            self.command = types.SimpleNamespace(position=0.0, max_effort=0.0)

    class FakeGripperCommand:
        Goal = FakeGoal

    class FakeFuture:
        def __init__(self, value):
            self.value = value

        def done(self):
            return True

        def result(self):
            return self.value

    action_result = types.SimpleNamespace(
        position=0.405,
        effort=7.5,
        stalled=False,
        reached_goal=True,
    )
    wrapped_result = types.SimpleNamespace(status=4, result=action_result)

    class FakeGoalHandle:
        accepted = True

        def get_result_async(self):
            return FakeFuture(wrapped_result)

    class FakeActionClient:
        last_instance = None

        def __init__(self, node, action_type, action_name):
            self.node = node
            self.action_type = action_type
            self.action_name = action_name
            self.goal = None
            FakeActionClient.last_instance = self

        def wait_for_server(self, timeout_sec):
            return True

        def send_goal_async(self, goal):
            self.goal = goal
            return FakeFuture(FakeGoalHandle())

    rclpy = types.ModuleType("rclpy")
    rclpy.spin_until_future_complete = lambda *args, **kwargs: None
    rclpy.spin_once = lambda *args, **kwargs: None
    rclpy_action = types.ModuleType("rclpy.action")
    rclpy_action.ActionClient = FakeActionClient
    control_msgs = types.ModuleType("control_msgs")
    control_msgs_action = types.ModuleType("control_msgs.action")
    control_msgs_action.GripperCommand = FakeGripperCommand
    action_msgs = types.ModuleType("action_msgs")
    action_msgs_msg = types.ModuleType("action_msgs.msg")
    action_msgs_msg.GoalStatus = types.SimpleNamespace(STATUS_SUCCEEDED=4)
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float32 = FakeFloat32
    std_msgs_msg.Float64 = FakeFloat64

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = FakeJointState

    for name, module in {
        "rclpy": rclpy,
        "rclpy.action": rclpy_action,
        "control_msgs": control_msgs,
        "control_msgs.action": control_msgs_action,
        "action_msgs": action_msgs,
        "action_msgs.msg": action_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return FakeFloat32, FakeFloat64, FakeJointState, FakeActionClient


def _action_config():
    config = _config("action_gripper_command")
    config["gripper"]["driver"].update(
        {
            "state_topic": "/gripper_state",
            "state_message_type": "std_msgs/msg/Float32",
            "action_name": "/gripper_controller/gripper_cmd",
            "controller_speed_mps": 0.02,
            "action_position_unit": "joint_radians",
            "state_unit": "joint_radians",
            "open_joint_position": 0.0,
            "closed_joint_position": 0.7,
            "server_wait_timeout_sec": 0.1,
            "command_timeout_sec": 0.1,
            "state_wait_timeout_sec": 0.0,
        }
    )
    return config


class FakeRosNode:
    def __init__(self):
        self.state_callback = None

    def create_subscription(self, message_type, topic, callback, qos):
        self.state_callback = callback
        return object()


def test_gripper_command_action_sends_validated_width_and_force(monkeypatch):
    _, _, _, action_client_type = _install_fake_gripper_ros(monkeypatch)
    node = FakeRosNode()
    backend = Robotiq2F140RealBackend(node=node, config=_action_config())

    result = backend.set_gripper_width(
        0.06, speed_mps=0.02, force_N=10.0, confirmed=True
    )

    assert result["success"] is True
    assert result["status"] == "SUCCEEDED"
    assert result["output"]["command_sent"] is True
    assert result["output"]["current_width_m"] == pytest.approx(0.059)
    goal = action_client_type.last_instance.goal
    assert goal.command.position == pytest.approx(0.4)
    assert goal.command.max_effort == 10.0


def test_float32_state_topic_updates_real_gripper_state(monkeypatch):
    float32_type, *_ = _install_fake_gripper_ros(monkeypatch)
    node = FakeRosNode()
    backend = Robotiq2F140RealBackend(node=node, config=_action_config())
    node.state_callback(float32_type(0.345))

    result = backend.get_gripper_state()

    assert result["success"] is True
    assert result["status"] == "OK"
    assert result["output"]["feedback_available"] is True
    assert result["output"]["current_width_m"] == pytest.approx(0.071)
    assert result["output"]["state_topic"] == "/gripper_state"


def test_float64_state_topic_updates_real_gripper_state(monkeypatch):
    _, float64_type, *_ = _install_fake_gripper_ros(monkeypatch)
    node = FakeRosNode()
    action_config = _action_config()
    action_config["gripper"]["driver"]["state_message_type"] = "std_msgs/msg/Float64"
    backend = Robotiq2F140RealBackend(node=node, config=action_config)
    node.state_callback(float64_type(0.345))

    result = backend.get_gripper_state()

    assert result["success"] is True
    assert result["status"] == "OK"
    assert result["output"]["feedback_available"] is True
    assert result["output"]["current_width_m"] == pytest.approx(0.071)
    assert result["output"]["state_topic"] == "/gripper_state"


def test_joint_state_topic_updates_real_gripper_state_with_named_joints(monkeypatch):
    _, _, joint_state_type, *_ = _install_fake_gripper_ros(monkeypatch)
    node = FakeRosNode()
    action_config = _action_config()
    action_config["gripper"]["driver"].update(
        {
            "state_message_type": "sensor_msgs/msg/JointState",
            "state_joint_names": ["finger_joint_1", "finger_joint_2"],
        }
    )
    backend = Robotiq2F140RealBackend(node=node, config=action_config)
    node.state_callback(joint_state_type(
        name=["finger_joint_1", "finger_joint_2"],
        position=[0.345, 0.345],
    ))

    result = backend.get_gripper_state()

    assert result["success"] is True
    assert result["status"] == "OK"
    assert result["output"]["feedback_available"] is True
    assert result["output"]["current_width_m"] == pytest.approx(0.071)
    assert result["output"]["state_topic"] == "/gripper_state"
