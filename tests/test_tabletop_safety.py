from llm_skill_robot.safety.tabletop_safety import TabletopSafety


def _pose(z: float) -> dict:
    return {
        "frame": "base_link",
        "position": {"x": 0.4, "y": 0.0, "z": z},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _safety() -> TabletopSafety:
    return TabletopSafety.from_config({
        "tabletop_safety": {
            "enabled": True,
            "frame": "base_link",
            "plane": {"normal": {"x": 0, "y": 0, "z": 1}, "d": 0},
            "collision_margin_m": 0.012,
            "clearance": {"pregrasp_m": 0.04, "grasp_m": 0.025, "retreat_m": 0.04},
            "close_sweep_compensation_m": 0.01,
            "max_approach_tilt_deg": 20,
            "robotiq_2f_140_finger_envelope": {
                "enabled": True,
                "tool0_to_gripper_base_m": [0, 0, 0.004],
                "joint_min_rad": 0.0,
                "joint_max_rad": 0.7,
                "joint_samples": 15,
                "extra_clearance_m": 0.005,
            },
        }
    })


def test_rejects_grasp_target_without_close_sweep_clearance():
    result = _safety().validate_grasp_plan({
        "pregrasp_pose": _pose(0.10), "grasp_pose": _pose(0.040), "retreat_pose": _pose(0.10),
    })

    assert result["allowed"] is False
    assert "grasp TCP plane clearance" in result["reason"]


def test_accepts_clear_vertical_approach():
    result = _safety().validate_grasp_plan({
        "pregrasp_pose": _pose(0.30), "grasp_pose": _pose(0.25), "retreat_pose": _pose(0.30),
    })

    assert result["allowed"] is True
    assert result["checks"]["approach"]["tilt_deg"] == 0.0


def test_rejects_tcp_that_is_clear_but_finger_pad_sweep_is_not():
    safety = _safety()
    # Find a target where the TCP passes its 47 mm rule, while the actual
    # finger-pad proxy is too close to the table.  This is the failure mode
    # that a TCP-only check could not catch.
    pose = _pose(0.047)
    # A rotated wrist can place a pad substantially below tool0 even though
    # the tool0 target itself passes the TCP-only clearance rule.
    pose["orientation"] = {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}

    result = safety.validate_pose(pose, "grasp")

    assert result["allowed"] is False
    assert "Robotiq finger envelope" in result["reason"]


def test_rejects_lateral_approach_near_table():
    plan = {"pregrasp_pose": _pose(0.12), "grasp_pose": _pose(0.06), "retreat_pose": _pose(0.12)}
    plan["pregrasp_pose"]["position"]["x"] = 0.50

    result = _safety().validate_grasp_plan(plan)

    assert result["allowed"] is False
    assert "Approach tilt" in result["reason"]
