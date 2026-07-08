from llm_skill_robot.safety.real_arm_safety import (
    is_real_motion_skill_allowed,
    load_real_arm_safety,
    require_execute_confirmation,
    validate_motion_summary,
    validate_named_target,
)


def test_real_arm_safety_config_loads():
    config = load_real_arm_safety()

    assert config.enabled is True
    assert config.execution_mode == "restricted_real_motion"
    assert config.limits.max_cartesian_step_m > 0.0


def test_move_to_pregrasp_allowed():
    result = is_real_motion_skill_allowed("move_to_pregrasp", load_real_arm_safety())

    assert result["allowed"] is True


def test_approach_grasp_blocked():
    result = is_real_motion_skill_allowed("approach_grasp", load_real_arm_safety())

    assert result["allowed"] is False
    assert "blocked" in result["reason"]


def test_pick_object_blocked():
    result = is_real_motion_skill_allowed("pick_object", load_real_arm_safety())

    assert result["allowed"] is False
    assert "blocked" in result["reason"]


def test_safe_home_allowed():
    result = validate_named_target("safe_home", load_real_arm_safety())

    assert result["allowed"] is True


def test_unknown_named_target_blocked():
    result = validate_named_target("home", load_real_arm_safety())

    assert result["allowed"] is False
    assert "not allowed" in result["reason"]


def test_excessive_joint_delta_blocked():
    config = load_real_arm_safety()
    result = validate_motion_summary(
        {"max_joint_delta_rad": config.limits.max_joint_delta_rad + 0.01},
        config,
    )

    assert result["allowed"] is False
    assert "max_joint_delta_rad" in result["reason"]


def test_excessive_cartesian_step_blocked():
    config = load_real_arm_safety()
    result = validate_motion_summary(
        {"max_cartesian_step_m": config.limits.max_cartesian_step_m + 0.01},
        config,
    )

    assert result["allowed"] is False
    assert "max_cartesian_step_m" in result["reason"]


def test_confirmation_accepts_only_execute():
    assert require_execute_confirmation("EXECUTE") is True
    assert require_execute_confirmation("execute") is False
    assert require_execute_confirmation("EXECUTE ") is False
    assert require_execute_confirmation("YES") is False
