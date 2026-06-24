"""Configurable JSON subprocess adapter for an isolated GraspGenX environment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


class GraspGenXSubprocessAdapter:
    DEFAULT_CONFIG = {
        "enabled": True,
        "conda_env": "graspgenx",
        "conda_executable": "conda",
        "runner_script": "",
        "timeout_sec": 120.0,
        "output_dir": "data/grasping/graspgenx_outputs",
    }

    def __init__(self, config: dict | None = None):
        config = config or {}
        if "graspgenx_subprocess" in config:
            config = config.get("graspgenx_subprocess") or {}
        self.config = {**self.DEFAULT_CONFIG, **config}

    def generate_candidates(self, point_cloud_path: str) -> dict:
        if not self.config.get("enabled", True):
            return self._result("NOT_AVAILABLE", "GraspGenX subprocess is disabled.")
        runner = self._runner_path()
        if runner is None or not runner.is_file():
            return self._result(
                "NOT_AVAILABLE", "GraspGenX runner_script is not configured."
            )
        if not Path(point_cloud_path).is_file():
            return self._result("FAILED", f"Point cloud does not exist: {point_cloud_path}")

        command = [
            str(self.config["conda_executable"]), "run", "-n",
            str(self.config["conda_env"]), "python", str(runner),
            "--point-cloud", point_cloud_path,
            "--output-dir", str(self.config["output_dir"]),
        ]
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
            value = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return self._result("FAILED", "GraspGenX runner did not return valid JSON.")
        return self._normalize(value)

    def _runner_path(self) -> Path | None:
        configured = str(self.config.get("runner_script") or "").strip()
        if not configured:
            return None
        path = Path(configured).expanduser()
        if path.is_absolute() or path.exists():
            return path
        return Path(__file__).resolve().parents[2] / path

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
        }

    @staticmethod
    def _result(status: str, message: str) -> dict:
        return {"success": False, "status": status, "message": message, "candidates": []}
