"""Execution mode configuration and validation helpers."""

from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from pydantic import BaseModel, Field

from llm_skill_robot.utils import get_config_dir

ExecutionMode = Literal["dry_run", "plan_only", "rviz_sim", "real_robot"]


class ExecutionModeConfig(BaseModel):
    """Validated execution mode settings."""

    mode: ExecutionMode = "plan_only"
    allow_real_robot_execution: bool = False
    require_human_confirmation: bool = True
    require_trajectory_review: bool = True
    require_final_confirmation_before_execution: bool = True
    allowed_execution_modes: List[ExecutionMode] = Field(
        default_factory=lambda: ["dry_run", "plan_only", "rviz_sim"]
    )


def load_execution_config(config_dir: str = None) -> ExecutionModeConfig:
    """Load and validate execution mode configuration."""
    root = Path(config_dir) if config_dir is not None else get_config_dir()
    config_file = root / "execution.yaml"
    if not config_file.exists():
        return ExecutionModeConfig()

    with open(config_file, "r") as f:
        data = yaml.safe_load(f) or {}

    execution_data = data.get("execution", data)
    return ExecutionModeConfig(**execution_data)


def is_execution_allowed(mode: str, config: Dict[str, Any]) -> bool:
    """Return whether an execution mode is allowed by configuration."""
    if mode == "dry_run":
        return True
    if mode == "plan_only":
        return True
    if mode == "rviz_sim":
        return mode in config.get("allowed_execution_modes", [])
    if mode == "real_robot":
        return bool(config.get("allow_real_robot_execution", False))
    return False
