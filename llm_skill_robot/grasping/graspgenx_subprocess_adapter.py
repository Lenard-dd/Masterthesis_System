"""Configurable JSON subprocess adapter for an isolated GraspGenX environment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from llm_skill_robot.utils import resolve_project_path


class GraspGenXSubprocessAdapter:
    DEFAULT_CONFIG = {
        "enabled": True,
        "conda_env": "graspgenx",
        "conda_executable": "conda",
        "runner_script": "scripts/graspgenx_zmq_runner.py",
        "timeout_sec": 120.0,
        "output_dir": "data/grasping/graspgenx_outputs",
        "server_host": "localhost",
        "server_port": 5556,
        "gripper_name": "robotiq_2f_140",
        "num_grasps": 200,
        "grasp_threshold": -1.0,
        "topk_num_grasps": 100,
        "timeout_ms": 60000,
        "estimate_gripper_width": True,
        "width_axis_index": 1,
        "width_clearance_m": 0.01,
        "strategy": "diffusion",
        "sparse_quota": 24,
        "moe_num_yaws": 36,
        "moe_z_offsets_cm": "0",
    }

    def __init__(self, config: dict | None = None):
        config = config or {}
        if "graspgenx_subprocess" in config:
            config = config.get("graspgenx_subprocess") or {}
        self.config = {**self.DEFAULT_CONFIG, **config}

    def generate_candidates(self, point_cloud_path: str, *, strategy: str | None = None) -> dict:
        if not self.config.get("enabled", True):
            return self._result("NOT_AVAILABLE", "GraspGenX subprocess is disabled.")
        runner = self._runner_path()
        if runner is None or not runner.is_file():
            return self._result(
                "NOT_AVAILABLE", "GraspGenX runner_script is not configured."
            )
        if not Path(point_cloud_path).is_file():
            return self._result("FAILED", f"Point cloud does not exist: {point_cloud_path}")

        command = self._build_command(point_cloud_path, runner, strategy=strategy)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(self.config["timeout_sec"]),
            )
        except subprocess.TimeoutExpired:
            return self._result("FAILED", "GraspGenX subprocess timed out.")
        except OSError as exc:
            return self._result("NOT_AVAILABLE", f"Could not start GraspGenX: {exc}")
        if completed.returncode != 0:
            return self._result(
                "FAILED",
                f"GraspGenX subprocess exited with code {completed.returncode}: {(completed.stderr or '').strip()}",
            )
        try:
            value = self._parse_runner_json(completed.stdout)
        except json.JSONDecodeError:
            detail = self._truncate((completed.stdout or "").strip())
            stderr = self._truncate((completed.stderr or "").strip())
            suffix = f" stdout={detail!r} stderr={stderr!r}" if detail or stderr else ""
            return self._result("FAILED", f"GraspGenX runner did not return valid JSON.{suffix}")
        return self._normalize(value)

    def _build_command(self, point_cloud_path: str, runner: Path | None = None, *, strategy: str | None = None) -> list[str]:
        runner = runner or self._runner_path()
        command = [
            str(self.config["conda_executable"]), "run", "-n",
            str(self.config["conda_env"]), "python", str(runner),
            "--point-cloud", point_cloud_path,
            "--output-dir", str(self.config["output_dir"]),
            "--host", str(self.config["server_host"]),
            "--port", str(int(self.config["server_port"])),
            "--gripper-name", str(self.config["gripper_name"]),
            "--num-grasps", str(int(self.config["num_grasps"])),
            "--grasp-threshold", str(float(self.config["grasp_threshold"])),
            "--topk-num-grasps", str(int(self.config["topk_num_grasps"])),
            "--timeout-ms", str(int(self.config["timeout_ms"])),
            "--width-axis-index", str(int(self.config["width_axis_index"])),
            "--width-clearance-m", str(float(self.config["width_clearance_m"])),
        ]
        chosen_strategy = str(strategy or self.config.get("strategy", "diffusion"))
        if chosen_strategy != "diffusion":
            command.extend([
                "--strategy", chosen_strategy,
                "--sparse-quota", str(int(self.config["sparse_quota"])),
                "--moe-num-yaws", str(int(self.config["moe_num_yaws"])),
                "--moe-z-offsets-cm=" + str(self.config["moe_z_offsets_cm"]),
            ])
        if bool(self.config.get("estimate_gripper_width", True)):
            command.append("--estimate-gripper-width")
        else:
            command.append("--no-estimate-gripper-width")
        default_width = self.config.get("default_gripper_width_m")
        if default_width is not None:
            command.extend(["--default-gripper-width-m", str(float(default_width))])
        return command

    def _runner_path(self) -> Path | None:
        configured = str(self.config.get("runner_script") or "").strip()
        if not configured:
            return None
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
        if path.exists():
            return path
        return resolve_project_path(path)

    @classmethod
    def _normalize(cls, value: Any) -> dict:
        if not isinstance(value, dict):
            return cls._result("FAILED", "GraspGenX runner JSON must be an object.")
        candidates = value.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        status = str(value.get("status", "SUCCESS" if candidates else "NOT_FOUND"))
        return {
            "success": status == "SUCCESS" and bool(candidates),
            "status": status,
            "message": str(value.get("message", "")),
            "candidates": candidates,
            "metadata": value.get("metadata", {}) if isinstance(value.get("metadata", {}), dict) else {},
        }

    @staticmethod
    def _parse_runner_json(stdout: str) -> Any:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as original_exc:
            for line in reversed((stdout or "").splitlines()):
                stripped = line.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    return json.loads(stripped)
            raise original_exc

    @staticmethod
    def _truncate(value: str, limit: int = 1000) -> str:
        if len(value) <= limit:
            return value
        return f"{value[:limit]}... <truncated {len(value) - limit} chars>"

    @staticmethod
    def _result(status: str, message: str) -> dict:
        return {"success": False, "status": status, "message": message, "candidates": []}
