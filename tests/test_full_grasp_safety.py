"""Safety gates for supervised, step-by-step real grasp execution."""

from dataclasses import replace

from llm_skill_robot.safety.real_arm_safety import (
    is_real_motion_skill_allowed,
    load_real_arm_safety,
    validate_approach_distance,
    validate_contact_close_allowed,
    validate_full_grasp_step,
    validate_grasp_score,
    validate_lift_distance,
)


def test_full_grasp_supervised_mode_loads():
    config = load_real_arm_safety()

    assert config.execution_mode == "full_grasp_supervised"
    assert config.grasp["require_final_confirmation_each_step"] is True


def test_approach_grasp_is_allowed_with_safe_distance():
    config = load_real_arm_safety()

    assert validate_approach_distance(0.06, config)["allowed"] is True
    result = validate_full_grasp_step("approach_grasp", {"distance_m": 0.06}, config)
    assert result["allowed"] is True
    assert result["requires_confirmation"] is True


def test_approach_grasp_allows_floating_point_representation_of_exact_limit():
    result = validate_approach_distance(0.060000000000000004, load_real_arm_safety())

    assert result["allowed"] is True


def test_approach_grasp_is_blocked_when_distance_is_too_long():
    result = validate_approach_distance(0.061, load_real_arm_safety())

    assert result["allowed"] is False
    assert "exceeds limit" in result["reason"]


def test_lift_object_is_blocked_when_lift_is_too_high():
    result = validate_lift_distance(0.101, load_real_arm_safety())

    assert result["allowed"] is False
    assert "exceeds limit" in result["reason"]


def test_low_grasp_score_is_blocked():
    result = validate_grasp_score(0.59, load_real_arm_safety())

    assert result["allowed"] is False
    assert "below minimum" in result["reason"]


def test_contact_close_requires_the_explicit_configuration_gate():
    config = load_real_arm_safety()

    assert validate_contact_close_allowed(config)["allowed"] is True
    blocked = replace(config, grasp={**config.grasp, "allow_contact_close": False})
    assert validate_contact_close_allowed(blocked)["allowed"] is False


def test_direct_pick_object_is_blocked():
    result = is_real_motion_skill_allowed("pick_object_direct", load_real_arm_safety())

    assert result["allowed"] is False


def test_unconfirmed_pick_sequence_is_blocked():
    result = is_real_motion_skill_allowed(
        "execute_pick_sequence_without_confirmation", load_real_arm_safety()
    )

    assert result["allowed"] is False
