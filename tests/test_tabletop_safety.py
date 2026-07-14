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
    assert "grasp TCP plane clearance" in result["reason"]


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
