"""Grasp candidate generation, planning, and supervised execution helpers."""

from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter

__all__ = [
    "FullGraspStateMachine",
    "GraspExecutionBackend",
    "GraspVerifier",
    "GraspGenXGraspPlanner",
    "GraspGenXSubprocessAdapter",
]


def __getattr__(name: str):
    """Lazily load executable helpers to keep ``python -m`` deterministic."""
    if name == "FullGraspStateMachine":
        from .full_grasp_state_machine import FullGraspStateMachine

        return FullGraspStateMachine
    if name == "GraspExecutionBackend":
        from .grasp_execution_backend import GraspExecutionBackend

        return GraspExecutionBackend
    if name == "GraspVerifier":
        from .grasp_verifier import GraspVerifier

        return GraspVerifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
