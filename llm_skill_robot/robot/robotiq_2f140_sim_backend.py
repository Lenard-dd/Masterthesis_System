from __future__ import annotations

from typing import Any


class Robotiq2F140SimBackend:
    """Simulation-only Robotiq 2F-140 backend for planning and logging.

    This backend never connects to a real gripper driver and never publishes
    hardware commands. It only tracks an in-memory gripper state.
    """

    def __init__(self, config: dict | None = None, joint_state_sim=None):
        gripper_config = (config or {}).get("gripper", config or {})

        self.min_width_m = float(gripper_config.get("min_width_m", 0.0))
        self.max_width_m = float(gripper_config.get("max_width_m", 0.14))
        self.default_open_width_m = float(
            gripper_config.get("default_open_width_m", self.max_width_m)
        )
        self.default_close_width_m = float(
            gripper_config.get("default_close_width_m", self.min_width_m)
        )
        self.default_speed_mps = float(gripper_config.get("default_speed_mps", 0.03))
        self.default_force_N = float(gripper_config.get("default_force_N", 20.0))

        self.real_gripper_enabled = False
        self.current_width_m = self._normalize_width(self.default_open_width_m)
        self.is_open = self._is_width_open(self.current_width_m)
        self.last_command: dict[str, Any] | None = None
        self.joint_state_sim = joint_state_sim

    def open_gripper(
        self,
        width_m: float | None = None,
        speed_mps: float | None = None,
        force_N: float | None = None,
        **kwargs: Any,
    ) -> dict:
        target_width = self.default_open_width_m if width_m is None else width_m
        return self.set_gripper_width(
            width_m=target_width,
            speed_mps=speed_mps,
            force_N=force_N,
            command="open_gripper",
            **kwargs,
        )

    def close_gripper(
        self,
        width_m: float | None = None,
        speed_mps: float | None = None,
        force_N: float | None = None,
        **kwargs: Any,
    ) -> dict:
        target_width = self.default_close_width_m if width_m is None else width_m
        return self.set_gripper_width(
            width_m=target_width,
            speed_mps=speed_mps,
            force_N=force_N,
            command="close_gripper",
            **kwargs,
        )

    def set_gripper_width(
        self,
        width_m: float,
        speed_mps: float | None = None,
        force_N: float | None = None,
        **kwargs: Any,
    ) -> dict:
        command = kwargs.pop("command", "set_gripper_width")

        try:
            normalized_width = self._normalize_width(width_m)
        except (TypeError, ValueError) as exc:
            return self._result(
                success=False,
                message=str(exc),
            )

        resolved_speed = self.default_speed_mps if speed_mps is None else float(speed_mps)
        resolved_force = self.default_force_N if force_N is None else float(force_N)

        self.current_width_m = normalized_width
        self.is_open = self._is_width_open(normalized_width)
        self.last_command = {
            "command": command,
            "width_m": normalized_width,
            "speed_mps": resolved_speed,
            "force_N": resolved_force,
            "mode": "rviz_sim",
            "real_gripper_enabled": self.real_gripper_enabled,
        }

        visual_result = None
        if self.joint_state_sim is not None:
            if command == "open_gripper":
                visual_result = self.joint_state_sim.open()
            elif command == "close_gripper":
                visual_result = self.joint_state_sim.close()
            else:
                visual_result = self.joint_state_sim.set_width(normalized_width)
            if not visual_result.get("success", False):
                return self._result(
                    success=False,
                    message=f"Gripper state updated, but RViz joint-state update failed: {visual_result.get('message')}",
                    visual_result=visual_result,
                )

        return self._result(
            success=True,
            message=(
                f"Simulated Robotiq 2F-140 {command} to "
                f"{normalized_width:.3f} m."
            ),
            visual_result=visual_result,
        )

    def get_gripper_state(self, **kwargs: Any) -> dict:
        visual_result = None
        if self.joint_state_sim is not None:
            visual_result = self.joint_state_sim.get_state()
        return self._result(
            success=True,
            message="Simulated Robotiq 2F-140 gripper state observed.",
            visual_result=visual_result,
        )

    def _normalize_width(self, width_m: float) -> float:
        width = float(width_m)
        if width < self.min_width_m or width > self.max_width_m:
            raise ValueError(
                f"width_m {width:.3f} outside allowed range "
                f"[{self.min_width_m:.3f}, {self.max_width_m:.3f}]"
            )
        return width

    def _is_width_open(self, width_m: float) -> bool:
        tolerance = max(1e-6, self.max_width_m * 1e-3)
        return abs(width_m - self.max_width_m) <= tolerance

    def _result(self, success: bool, message: str, visual_result: dict | None = None) -> dict:
        output = {
            "current_width_m": self.current_width_m,
            "is_open": self.is_open,
            "mode": "rviz_sim",
            "real_gripper_enabled": self.real_gripper_enabled,
        }
        if visual_result is not None:
            output["visual_joint_state"] = visual_result.get("output", {})
            joint_positions = visual_result.get("output", {}).get("joint_positions")
            if joint_positions is not None:
                output["joint_positions"] = joint_positions
        return {
            "success": success,
            "message": message,
            "output": output,
        }
