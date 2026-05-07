from llm_skill_robot.core.execution_mode import (
    ExecutionModeConfig,
    is_execution_allowed,
    load_execution_config,
)


def test_dry_run_allowed():
    config = ExecutionModeConfig().model_dump()

    assert is_execution_allowed("dry_run", config) is True


def test_plan_only_allowed():
    config = ExecutionModeConfig().model_dump()

    assert is_execution_allowed("plan_only", config) is True


def test_rviz_sim_allowed_if_configured():
    config = ExecutionModeConfig(
        allowed_execution_modes=["dry_run", "plan_only", "rviz_sim"]
    ).model_dump()

    assert is_execution_allowed("rviz_sim", config) is True


def test_real_robot_blocked_by_default():
    config = ExecutionModeConfig().model_dump()

    assert is_execution_allowed("real_robot", config) is False


def test_load_execution_config_from_project_config():
    config = load_execution_config()

    assert config.mode == "plan_only"
    assert config.allow_real_robot_execution is False
    assert config.require_human_confirmation is True
    assert config.require_final_confirmation_before_execution is True
    assert "rviz_sim" in config.allowed_execution_modes
    assert "real_robot" not in config.allowed_execution_modes
