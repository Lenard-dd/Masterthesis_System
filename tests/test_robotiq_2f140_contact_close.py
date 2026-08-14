from dataclasses import replace

from llm_skill_robot.robot.robotiq_2f140_real_backend import Robotiq2F140RealBackend
from llm_skill_robot.safety.hardware_mode import load_hardware_mode
from llm_skill_robot.safety.real_arm_safety import load_real_arm_safety


def _config():
    return {
        "gripper": {
            "safe_close_width_m": 0.04,
            "default_speed_mps": 0.02,
            "default_force_N": 10.0,
            "real_gripper_enabled": True,
            "driver": {"interface_type": "dry_run", "width_unit": "meters"},
        }
    }


def _backend(**kwargs):
    return Robotiq2F140RealBackend(
        config=_config(),
        confirmation_provider=lambda _prompt: "YES",
        **kwargs,
    )


def _successful_driver(**command):
    return {
        "success": True,
        "status": "SENT",
        "message": "mock command sent",
        "output": {"command_sent": True, "command_preview": command},
    }


def test_contact_close_is_blocked_when_hardware_mode_disables_it():
    mode = load_hardware_mode()
    mode = replace(mode, gripper=replace(mode.gripper, allow_contact_close=False))

    result = _backend(hardware_mode=mode).close_for_grasp(confirmation="YES")

    assert result["success"] is False
    assert "disabled by hardware_mode" in result["message"]


def test_contact_close_is_blocked_when_real_arm_safety_disables_it():
    safety = load_real_arm_safety()
    safety = replace(safety, grasp={**safety.grasp, "allow_contact_close": False})

    result = _backend(real_arm_safety=safety).close_for_grasp(confirmation="YES")

    assert result["success"] is False
    assert "disabled by grasp safety" in result["message"]


def test_contact_close_rejects_force_above_limit_before_driver_command():
    backend = _backend()
    backend._send_driver_command = lambda **_command: (_ for _ in ()).throw(
        AssertionError("unsafe contact close reached driver")
    )

    result = backend.close_for_grasp(force_N=20.1, confirmation="YES")

    assert result["success"] is False
    assert "force_N" in result["message"]


def test_contact_close_rejects_speed_above_limit_before_driver_command():
    backend = _backend()
    backend._send_driver_command = lambda **_command: (_ for _ in ()).throw(
        AssertionError("unsafe contact close reached driver")
    )

    result = backend.close_for_grasp(speed_mps=0.031, confirmation="YES")

    assert result["success"] is False
    assert "speed_mps" in result["message"]


def test_contact_close_does_not_send_without_exact_yes():
    backend = _backend()
    backend._send_driver_command = lambda **_command: (_ for _ in ()).throw(
        AssertionError("unconfirmed contact close reached driver")
    )

    result = backend.close_for_grasp(confirmation="yes")

    assert result["success"] is False
    assert "Exact YES" in result["message"]


def test_contact_close_yes_sends_through_mocked_driver():
    backend = _backend()
    sent = []

    def driver(**command):
        sent.append(command)
        return _successful_driver(**command)

    backend._send_driver_command = driver
    result = backend.close_for_grasp(confirmation="YES")

    assert result["success"] is True
    assert sent == [{"width_m": 0.02, "speed_mps": 0.02, "force_N": 10.0}]
    assert result["output"]["command_sent"] is True
    assert result["output"]["mode"] == "real_grasp_contact_close"


def test_contact_close_uses_configured_native_close_service():
    backend = _backend()
    backend.driver_config["contact_close_interface_type"] = "service_close_gripper"
    sent = []
    backend._send_contact_close_service = lambda command: (
        sent.append(command) or _successful_driver(**command)
    )
    backend._send_driver_command = lambda **_command: (_ for _ in ()).throw(
        AssertionError("contact close should not use the generic action adapter")
    )

    result = backend.close_for_grasp(confirmation="YES")

    assert result["success"] is True
    assert sent == [{"width_m": 0.02, "speed_mps": 0.02, "force_N": 10.0}]


def test_contact_close_preserves_native_object_contact_as_hold_evidence():
    backend = _backend()
    backend.driver_config["contact_close_interface_type"] = "service_close_gripper"
    backend._send_contact_close_service = lambda command: {
        "success": True,
        "status": "SUCCEEDED",
        "message": "contact detected",
        "output": {
            "command_sent": True,
            "validated_command": command,
            "object_contact_detected": True,
            "error_code": 2,
        },
    }

    result = backend.close_for_grasp(confirmation="YES")

    assert result["success"] is True
    assert result["output"]["object_contact_detected"] is True
    assert result["output"]["object_may_be_held"] is True


def test_contact_close_reports_unavailable_feedback_honestly():
    backend = _backend()
    backend._send_driver_command = _successful_driver

    result = backend.close_for_grasp(confirmation="YES")

    assert result["success"] is True
    assert result["output"]["feedback_available"] is False
    assert result["output"]["current_width_m"] is None
    assert result["output"]["object_may_be_held"] is None
