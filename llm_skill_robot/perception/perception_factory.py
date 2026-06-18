"""Factory helpers for configuring perception backends from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_skill_robot.perception.d435i_backend import D435iBackend
from llm_skill_robot.perception.perception_pipeline import PerceptionPipeline
from llm_skill_robot.perception.sam3_subprocess_adapter import SAM3SubprocessAdapter
from llm_skill_robot.utils import get_config_dir


def load_perception_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config/perception.yaml, returning an empty dict if it is absent."""
    path = Path(config_path) if config_path is not None else get_config_dir() / "perception.yaml"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def create_perception_pipeline(
    node=None,
    config: dict[str, Any] | None = None,
) -> PerceptionPipeline:
    """Create the configured perception pipeline without importing SAM3 directly."""
    full_config = config if config is not None else load_perception_config()
    perception_config = full_config.get("perception", {}) if isinstance(full_config, dict) else {}

    if perception_config.get("mode") != "d435i_sam3":
        return PerceptionPipeline()

    detector_adapter = None
    if perception_config.get("detector") == "sam3_subprocess":
        detector_adapter = SAM3SubprocessAdapter(full_config.get("sam3_subprocess", {}))

    backend = D435iBackend(
        node=node,
        config=full_config,
        detector_adapter=detector_adapter,
    )
    return PerceptionPipeline(backend)


def perception_context_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return verification context flags derived from perception config."""
    perception_config = {}
    if isinstance(config, dict):
        perception_config = config.get("perception", {}) or {}

    return {
        "require_base_link_pose_for_pick": bool(
            perception_config.get("require_base_link_pose_for_pick", False)
        ),
        "target_frame": perception_config.get("target_frame", "base_link"),
    }
