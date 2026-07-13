"""Step-scoped state control for supervised real grasp execution.

The state machine deliberately has no operation that advances more than one
state.  A caller must perform and confirm each real motion or gripper action
before recording its corresponding transition.
"""

from __future__ import annotations

from typing import Any


class FullGraspStateMachine:
    """Guard the required order of a complete, supervised grasp sequence."""

    INIT = "INIT"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    POINT_CLOUD_BUILT = "POINT_CLOUD_BUILT"
    GRASP_GENERATED = "GRASP_GENERATED"
    PREGRASP_PLANNED = "PREGRASP_PLANNED"
    AT_PREGRASP = "AT_PREGRASP"
    APPROACH_PLANNED = "APPROACH_PLANNED"
    AT_GRASP_POSE = "AT_GRASP_POSE"
    GRIPPER_CLOSED = "GRIPPER_CLOSED"
    RETREAT_PLANNED = "RETREAT_PLANNED"
    RETREATED = "RETREATED"
    GRASP_VERIFIED = "GRASP_VERIFIED"
    DONE = "DONE"
    ABORTED = "ABORTED"

    _ALLOWED_TRANSITIONS = {
        INIT: OBJECT_DETECTED,
        OBJECT_DETECTED: POINT_CLOUD_BUILT,
        POINT_CLOUD_BUILT: GRASP_GENERATED,
        GRASP_GENERATED: PREGRASP_PLANNED,
        PREGRASP_PLANNED: AT_PREGRASP,
        AT_PREGRASP: APPROACH_PLANNED,
        APPROACH_PLANNED: AT_GRASP_POSE,
        AT_GRASP_POSE: GRIPPER_CLOSED,
        GRIPPER_CLOSED: RETREAT_PLANNED,
        RETREAT_PLANNED: RETREATED,
        RETREATED: GRASP_VERIFIED,
        GRASP_VERIFIED: DONE,
    }
    _CONTEXT_FIELDS = frozenset(
        {
            "selected_object_id",
            "selected_grasp_candidate_id",
            "selected_grasp_score",
            "point_cloud_path",
            "grasp_plan_available",
            "pregrasp_planned",
            "approach_planned",
            "retreat_planned",
            "gripper_closed",
            "grasp_verified",
        }
    )
    _STATE_CONTEXT_UPDATES = {
        GRASP_GENERATED: {"grasp_plan_available": True},
        PREGRASP_PLANNED: {"pregrasp_planned": True},
        APPROACH_PLANNED: {"approach_planned": True},
        GRIPPER_CLOSED: {"gripper_closed": True},
        RETREAT_PLANNED: {"retreat_planned": True},
        GRASP_VERIFIED: {"grasp_verified": True},
    }

    def __init__(self) -> None:
        self._state = self.INIT
        self.selected_object_id: str | None = None
        self.selected_grasp_candidate_id: str | None = None
        self.selected_grasp_score: float | None = None
        self.point_cloud_path: str | None = None
        self.grasp_plan_available = False
        self.pregrasp_planned = False
        self.approach_planned = False
        self.retreat_planned = False
        self.gripper_closed = False
        self.grasp_verified = False

    def current_state(self) -> str:
        """Return the current sequence state."""
        return self._state

    def can_transition(self, next_state: str) -> bool:
        """Return whether exactly one normal transition may reach *next_state*."""
        return self._ALLOWED_TRANSITIONS.get(self._state) == next_state

    def transition(self, next_state: str, reason: str | None = None) -> dict[str, Any]:
        """Advance exactly one allowed state, leaving state unchanged if rejected."""
        if next_state == self.ABORTED:
            return self.abort(reason or "Aborted by transition request.")
        if not self.can_transition(next_state):
            return self._result(
                allowed=False,
                reason=(
                    f"Cannot transition from {self._state} to {next_state}; "
                    "the requested grasp step is out of order."
                ),
            )

        previous_state = self._state
        self._state = next_state
        for field, value in self._STATE_CONTEXT_UPDATES.get(next_state, {}).items():
            setattr(self, field, value)
        return self._result(
            allowed=True,
            reason=reason or f"Transitioned from {previous_state} to {next_state}.",
            previous_state=previous_state,
        )

    def abort(self, reason: str) -> dict[str, Any]:
        """Abort safely from any state, retaining context for diagnostics."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("An abort reason is required.")
        previous_state = self._state
        self._state = self.ABORTED
        return self._result(
            allowed=True,
            reason=reason,
            previous_state=previous_state,
        )

    def require_state(self, expected_state: str) -> dict[str, Any]:
        """Report whether the caller is at the one state required for an action."""
        if self._state == expected_state:
            return self._result(
                allowed=True, reason=f"Current state is {expected_state}."
            )
        return self._result(
            allowed=False,
            reason=f"Current state is {self._state}; expected {expected_state}.",
        )

    def update_context(self, **updates: Any) -> dict[str, Any]:
        """Record perception/planning data without advancing the state machine."""
        unknown = set(updates).difference(self._CONTEXT_FIELDS)
        if unknown:
            raise ValueError(f"Unknown full-grasp context fields: {', '.join(sorted(unknown))}.")
        for field, value in updates.items():
            setattr(self, field, value)
        return self._result(allowed=True, reason="Full-grasp context updated.")

    def context(self) -> dict[str, Any]:
        """Return a copy of the state-machine context for demos and executors."""
        return {field: getattr(self, field) for field in sorted(self._CONTEXT_FIELDS)}

    def _result(
        self,
        *,
        allowed: bool,
        reason: str,
        previous_state: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "allowed": allowed,
            "state": self._state,
            "reason": reason,
            "context": self.context(),
        }
        if previous_state is not None:
            result["previous_state"] = previous_state
        return result
