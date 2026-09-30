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
        "width_axis_index": 0,
        "width_clearance_m": 0.01,
        "strategy": "diffusion",
        "sparse_quota": 24,
        "moe_num_yaws": 36,
        "moe_z_offsets_cm": "0",
        "scene_collision_enabled": True,
        "scene_collision_threshold_m": 0.006,
        "scene_collision_samples": 2500,
        "scene_collision_batch_size": 4,
        "scene_collision_closed_finger_enabled": True,
        "scene_collision_closed_finger_samples": 3500,
        "scene_collision_closed_finger_threshold_m": 0.008,
        "scene_collision_closed_finger_backoff_m": 0.040,
        "environment_constrained_generation": {
            "enabled": True,
            "min_backoff_m": 0.025,
            "max_backoff_m": 0.045,
            "backoff_step_m": 0.0025,
            "min_target_contact_points": 12,
            "min_target_contact_fraction": 0.20,
            "pad_contact_depth_m": 0.045,
            "support_plane_enabled": True,
            "support_plane_margin_m": 0.012,
            "support_plane_clearance_m": 0.010,
            "support_plane_inlier_threshold_m": 0.003,
            "support_plane_max_tilt_deg": 20.0,
            "support_plane_min_inliers": 24,
            "pregrasp_distance_m": 0.060,
            "retreat_distance_m": 0.100,
            "path_samples": 5,
            "path_surface_samples": 1800,
            "yaw_bin_deg": 15.0,
            "max_per_yaw_bin": 2,
        },
        "scene_collision_fallback_to_object_only": False,
    }

    def __init__(self, config: dict | None = None):
        config = config or {}
        if "graspgenx_subprocess" in config:
            config = config.get("graspgenx_subprocess") or {}
        self.config = {**self.DEFAULT_CONFIG, **config}

    def generate_candidates(
        self, point_cloud_path: str, *, strategy: str | None = None,
        scene_point_cloud_path: str | None = None,
    ) -> dict:
        if not self.config.get("enabled", True):
            return self._result("NOT_AVAILABLE", "GraspGenX subprocess is disabled.")
        runner = self._runner_path()
        if runner is None or not runner.is_file():
            return self._result(
                "NOT_AVAILABLE", "GraspGenX runner_script is not configured."
            )
        if not Path(point_cloud_path).is_file():
            return self._result("FAILED", f"Point cloud does not exist: {point_cloud_path}")

        if scene_point_cloud_path and not Path(scene_point_cloud_path).is_file():
            return self._result(
                "FAILED", f"Scene point cloud does not exist: {scene_point_cloud_path}"
            )
        command = self._build_command(
            point_cloud_path, runner, strategy=strategy,
            scene_point_cloud_path=scene_point_cloud_path,
        )
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

    def _build_command(
        self, point_cloud_path: str, runner: Path | None = None, *,
        strategy: str | None = None, scene_point_cloud_path: str | None = None,
    ) -> list[str]:
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
        if bool(self.config.get("scene_collision_enabled", True)) and scene_point_cloud_path:
            constrained = self.config.get("environment_constrained_generation", {}) or {}
            command.extend([
                "--scene-point-cloud", scene_point_cloud_path,
                "--collision-threshold-m", str(float(self.config["scene_collision_threshold_m"])),
                "--collision-samples", str(int(self.config["scene_collision_samples"])),
                "--collision-batch-size", str(int(self.config["scene_collision_batch_size"])),
                "--collision-closed-finger-samples", str(int(self.config["scene_collision_closed_finger_samples"])),
                "--collision-closed-finger-threshold-m", str(float(self.config["scene_collision_closed_finger_threshold_m"])),
                "--collision-closed-finger-backoff-m", str(float(self.config["scene_collision_closed_finger_backoff_m"])),
                "--constraint-min-backoff-m", str(float(constrained.get("min_backoff_m", 0.025))),
                "--constraint-max-backoff-m", str(float(constrained.get("max_backoff_m", 0.045))),
                "--constraint-backoff-step-m", str(float(constrained.get("backoff_step_m", 0.0025))),
                "--constraint-min-target-contact-points", str(int(constrained.get("min_target_contact_points", 12))),
                "--constraint-min-target-contact-fraction", str(float(constrained.get("min_target_contact_fraction", 0.20))),
                "--constraint-pad-contact-depth-m", str(float(constrained.get("pad_contact_depth_m", 0.045))),
                "--constraint-support-plane-margin-m", str(float(constrained.get("support_plane_margin_m", 0.012))),
                "--constraint-support-plane-clearance-m", str(float(constrained.get("support_plane_clearance_m", 0.010))),
                "--constraint-support-plane-inlier-threshold-m", str(float(constrained.get("support_plane_inlier_threshold_m", 0.003))),
                "--constraint-support-plane-max-tilt-deg", str(float(constrained.get("support_plane_max_tilt_deg", 20.0))),
                "--constraint-support-plane-min-inliers", str(int(constrained.get("support_plane_min_inliers", 24))),
                "--constraint-pregrasp-distance-m", str(float(constrained.get("pregrasp_distance_m", 0.060))),
                "--constraint-retreat-distance-m", str(float(constrained.get("retreat_distance_m", 0.100))),
                "--constraint-path-samples", str(int(constrained.get("path_samples", 5))),
                "--constraint-path-surface-samples", str(int(constrained.get("path_surface_samples", 1800))),
                "--constraint-yaw-bin-deg", str(float(constrained.get("yaw_bin_deg", 15.0))),
                "--constraint-max-per-yaw-bin", str(int(constrained.get("max_per_yaw_bin", 2))),
            ])
            command.append(
                "--environment-constrained-generation"
                if bool(constrained.get("enabled", True))
                else "--no-environment-constrained-generation"
            )
            command.append(
                "--constraint-support-plane-enabled"
                if bool(constrained.get("support_plane_enabled", True))
                else "--no-constraint-support-plane-enabled"
            )
            command.append(
                "--collision-closed-finger-enabled"
                if bool(self.config.get("scene_collision_closed_finger_enabled", True))
                else "--no-collision-closed-finger-enabled"
            )
            if bool(self.config.get("scene_collision_fallback_to_object_only", False)):
                command.append("--collision-fallback-to-object-only")
            else:
                command.append("--no-collision-fallback-to-object-only")
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
