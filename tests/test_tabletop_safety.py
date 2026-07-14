import pytest

from llm_skill_robot.safety.tabletop_safety import TabletopSafety


def _pose(z: float) -> dict:
    return {
        "frame": "base_link",
        "position": {"x": 0.4, "y": 0.0, "z": z},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _safety() -> TabletopSafety:
    return TabletopSafety.from_config(
        {
            "tabletop_safety": {
                "enabled": True,
                "frame": "base_link",
                "plane": {
                    "normal": {"x": 0, "y": 0, "z": 1},
                    "d": 0,
                },
                "collision_margin_m": 0.012,
                "clearance": {
                    "pregrasp_m": 0.04,
                    "grasp_m": 0.025,
                    "retreat_m": 0.04,
                },
                "close_sweep_compensation_m": 0.01,
                "max_approach_tilt_deg": 20,
            }
        }
    )


def test_rejects_grasp_target_without_close_sweep_clearance():
    result = _safety().validate_grasp_plan(
        {
            "pregrasp_pose": _pose(0.10),
            "grasp_pose": _pose(0.040),
            "retreat_pose": _pose(0.10),
        }
    )

    assert result["allowed"] is False
    assert "grasp gripper contact-point plane clearance" in result["reason"]


def test_accepts_clear_vertical_approach():
    result = _safety().validate_grasp_plan(
        {
            "pregrasp_pose": _pose(0.12),
            "grasp_pose": _pose(0.06),
            "retreat_pose": _pose(0.12),
        }
    )

    assert result["allowed"] is True
    assert result["checks"]["approach"]["tilt_deg"] == 0.0


def test_rejects_lateral_approach_near_table():
    plan = {
        "pregrasp_pose": _pose(0.12),
        "grasp_pose": _pose(0.06),
        "retreat_pose": _pose(0.12),
    }
    plan["pregrasp_pose"]["position"]["x"] = 0.50

    result = _safety().validate_grasp_plan(plan)

    assert result["allowed"] is False
    assert "Approach tilt" in result["reason"]


def test_gripper_contact_point_uses_pose_orientation_and_local_offset():
    safety = TabletopSafety.from_config(
        {
            "tabletop_safety": {
                "plane": {"normal": {"x": 0, "y": 0, "z": 1}, "d": 0},
                "tool0_to_grasp_contact_m": [0.0, 0.0, 0.199],
            }
        }
    )
    downward_pose = _pose(0.25)
    # 180 degrees about X maps tool0-local +Z toward base_link -Z.
    downward_pose["orientation"] = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}

    point = safety.grasp_contact_point(downward_pose)

    assert point["z"] == pytest.approx(0.051)


def test_contact_backoff_moves_grasp_and_retreat_toward_pregrasp_only():
    safety = TabletopSafety.from_config(
        {
            "tabletop_safety": {
                "plane": {"normal": {"x": 0, "y": 0, "z": 1}, "d": 0},
                "grasp_contact_backoff_m": 0.005,
            }
        }
    )
    poses = {
        "pregrasp": _pose(0.20),
        "grasp": _pose(0.10),
        "retreat": _pose(0.25),
    }

    adjusted = safety.apply_grasp_contact_backoff(poses)

    assert adjusted["pregrasp"]["position"]["z"] == pytest.approx(0.20)
    assert adjusted["grasp"]["position"]["z"] == pytest.approx(0.105)
    assert adjusted["retreat"]["position"]["z"] == pytest.approx(0.255)


def test_finger_close_sweep_is_checked_along_approach_axis():
    safety = TabletopSafety.from_config(
        {
            "tabletop_safety": {
                "enabled": True,
                "frame": "base_link",
                "plane": {"normal": {"x": 0, "y": 0, "z": 1}, "d": 0},
                "collision_margin_m": 0.005,
                "clearance": {"grasp_m": 0.005},
                "finger_axial_close_sweep_m": 0.025,
            }
        }
    )
    plan = {
        "pregrasp_pose": _pose(0.10),
        "grasp_pose": _pose(0.03),
    }

    result = safety.validate_grasp_plan(plan)

    assert result["allowed"] is False
    assert "Closed-finger sweep endpoint" in result["reason"]
    sweep = result["checks"]["finger_close_sweep"]
    assert sweep["measured_clearance_m"] == pytest.approx(0.005)
