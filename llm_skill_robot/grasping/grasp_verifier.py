"""Conservative post-retreat grasp verification from independent observations."""

from __future__ import annotations

from typing import Any


class GraspVerifier:
    """Assess evidence without ever claiming a grasp is definitely successful."""

    def verify_grasp(
        self,
        object_id: str,
        gripper_state: dict | None = None,
        perception_result: dict | None = None,
    ) -> dict:
        """Return LIKELY_SUCCESS, LIKELY_FAILED, or UNCERTAIN.

        A gripper that stopped before its commanded closed width can indicate
        contact, but it is insufficient on its own.  It must be corroborated
        by a successful perception re-check that no longer finds the object at
        its original table location.
        """
        signals = self._signals(object_id, gripper_state, perception_result)
        if signals["object_may_be_held"] is True and signals["object_visible"] is False:
            return {
                "status": "LIKELY_SUCCESS",
                "message": (
                    "Gripper feedback suggests contact and the object is no longer "
                    "visible at its original table location."
                ),
                "signals": signals,
            }
        if signals["fully_closed"] is True and signals["object_visible"] is True:
            return {
                "status": "LIKELY_FAILED",
                "message": (
                    "The gripper reached its fully closed width while the object "
                    "remains visible at the original table location."
                ),
                "signals": signals,
            }
        return {
            "status": "UNCERTAIN",
            "message": self._uncertain_message(signals),
            "signals": signals,
        }

    def _signals(
        self,
        object_id: str,
        gripper_state: dict | None,
        perception_result: dict | None,
    ) -> dict[str, Any]:
        gripper = self._output_mapping(gripper_state)
        feedback_available = gripper.get("feedback_available") is True
        current_width = self._finite_float_or_none(gripper.get("current_width_m"))
        command_sent = gripper.get("command_sent") is True
        object_may_be_held = (
            gripper.get("object_may_be_held")
            if feedback_available and isinstance(gripper.get("object_may_be_held"), bool)
            else None
        )
        fully_closed = (
            feedback_available
            and command_sent
            and current_width is not None
            and current_width <= 1e-4
        )
        perception = self._perception_visibility(object_id, perception_result)
        return {
            "command_sent": command_sent,
            "feedback_available": feedback_available,
            "current_width_m": current_width,
            "object_may_be_held": object_may_be_held,
            "fully_closed": fully_closed,
            "perception_available": perception["available"],
            "object_visible": perception["visible"],
            "object_near_gripper": perception["near_gripper"],
        }

    @staticmethod
    def _output_mapping(state: dict | None) -> dict[str, Any]:
        if not isinstance(state, dict):
            return {}
        output = state.get("output")
        return output if isinstance(output, dict) else state

    def _perception_visibility(
        self, object_id: str, result: dict | None
    ) -> dict[str, bool | None]:
        if not isinstance(result, dict):
            return {"available": False, "visible": None, "near_gripper": None}

        if result.get("success") is False:
            return {"available": False, "visible": None, "near_gripper": None}
        near_gripper = result.get("object_near_gripper")
        if not isinstance(near_gripper, bool):
            near_gripper = None
        for key in ("object_still_visible", "object_visible_at_original_pose"):
            if isinstance(result.get(key), bool):
                return {
                    "available": True,
                    "visible": result[key],
                    "near_gripper": near_gripper,
                }
        if isinstance(result.get("object_no_longer_visible"), bool):
            return {
                "available": True,
                "visible": not result["object_no_longer_visible"],
                "near_gripper": near_gripper,
            }

        objects = result.get("objects")
        if isinstance(objects, list):
            return {
                "available": True,
                "visible": any(self._object_id(item) == object_id for item in objects),
                "near_gripper": near_gripper,
            }
        return {"available": False, "visible": None, "near_gripper": near_gripper}

    @staticmethod
    def _object_id(value: Any) -> str | None:
        if isinstance(value, dict):
            raw = value.get("object_id") or value.get("id")
        else:
            raw = getattr(value, "object_id", None) or getattr(value, "id", None)
        return str(raw) if raw is not None else None

    @staticmethod
    def _finite_float_or_none(value: Any) -> float | None:
        import math

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _uncertain_message(signals: dict[str, Any]) -> str:
        if not signals["feedback_available"] and not signals["perception_available"]:
            return "No useful gripper feedback or perception re-check is available."
        if signals["object_may_be_held"] is True:
            return "Gripper feedback suggests contact, but perception does not confirm object removal."
        if signals["object_visible"] is False:
            return "The object is not visible, but gripper feedback does not corroborate contact."
        return "Available evidence is insufficient to verify the grasp conservatively."
