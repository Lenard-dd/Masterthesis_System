"""Persistent resources and trusted state for an RViz Agent session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RvizExecutionContext:
    """Keep RViz/fake-hardware resources alive across individual Tool calls."""

    node: Any
    motion_backend: Any
    perception_pipeline: Any
    point_cloud_builder: Any
    grasp_planner: Any
    gripper_backend: Any
    trajectory_validator: Any
    trajectory_reviewer: Any
    execution_config: dict[str, Any]
    safety_config: dict[str, Any]
    runtime_state: dict[str, Any] = field(default_factory=dict)

    def merge_agent_context(self, agent_context: dict[str, Any]) -> dict[str, Any]:
        """Merge trusted Agent observations into the persistent RViz context."""
        self.runtime_state.update(agent_context)
        return self.runtime_state
