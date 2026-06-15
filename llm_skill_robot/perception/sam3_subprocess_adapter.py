"""Subprocess adapter for SAM3 inference in an isolated conda environment."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class SAM3SubprocessAdapter:
    """Call SAM3 through ``conda run`` and exchange JSON with the runner."""

    DEFAULT_CONFIG: dict[str, Any] = {
        "enabled": True,
        "conda_env": "sam3",
        "conda_executable": "conda",
        "runner_script": "scripts/sam3_runner.py",
        "timeout_sec": 60.0,
        "output_dir": "data/perception_outputs",
        "device": "auto",
        "checkpoint_path": "",
        "model_config": "",
    }

    def __init__(self, config: dict | None = None):
        config = config or {}
        if "sam3_subprocess" in config:
            config = config["sam3_subprocess"] or {}
        self.config = {**self.DEFAULT_CONFIG, **config}

    def is_available(self) -> bool:
        """Return whether the subprocess entry point can be invoked."""
        if not self.config.get("enabled", True):
            return False

        runner_script = Path(str(self.config["runner_script"]))
        if not runner_script.exists():
            return False

        try:
            completed = subprocess.run(
                [str(self.config["conda_executable"]), "--version"],
                capture_output=True,
                text=True,
                timeout=min(float(self.config["timeout_sec"]), 10.0),
            )
        except (OSError, subprocess.SubprocessError):
            return False

        return completed.returncode == 0

    def detect(
        self,
        rgb_image_path: str,
        query: str,
        min_confidence: float = 0.5,
    ) -> dict:
        """Run SAM3 inference for one RGB image and return the runner JSON."""
        if not self.config.get("enabled", True):
            return self._result(
                "NOT_AVAILABLE",
                "SAM3 subprocess detector is disabled.",
            )

        runner_script = Path(str(self.config["runner_script"]))
        if not runner_script.exists():
            return self._result(
                "NOT_AVAILABLE",
                f"SAM3 runner script does not exist: {runner_script}",
            )

        command = self._build_command(rgb_image_path, query, min_confidence)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(self.config["timeout_sec"]),
            )
        except subprocess.TimeoutExpired:
            return self._result(
                "FAILED",
                f"SAM3 subprocess timed out after {self.config['timeout_sec']} seconds.",
            )
        except OSError as exc:
            return self._result(
                "NOT_AVAILABLE",
                f"Could not start SAM3 subprocess: {exc}",
            )

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            message = f"SAM3 subprocess exited with code {completed.returncode}."
            if stderr:
                message = f"{message} stderr: {stderr}"
            return self._result("FAILED", message)

        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            stdout = self._summarize(completed.stdout)
            stderr = self._summarize(completed.stderr)
            return self._result(
                "FAILED",
                f"SAM3 runner did not return valid JSON. stdout: {stdout} stderr: {stderr}",
            )

        return self._normalize_result(parsed)

    def _build_command(
        self,
        rgb_image_path: str,
        query: str,
        min_confidence: float,
    ) -> list[str]:
        command = [
            str(self.config["conda_executable"]),
            "run",
            "-n",
            str(self.config["conda_env"]),
            "python",
            str(self.config["runner_script"]),
            "--image",
            rgb_image_path,
            "--query",
            query,
            "--min-confidence",
            str(min_confidence),
            "--output-dir",
            str(self.config["output_dir"]),
        ]

        checkpoint_path = str(self.config.get("checkpoint_path") or "")
        if checkpoint_path:
            command.extend(["--checkpoint", checkpoint_path])

        model_config = str(self.config.get("model_config") or "")
        if model_config:
            command.extend(["--model-config", model_config])

        device = str(self.config.get("device") or "")
        if device:
            command.extend(["--device", device])

        return command

    @staticmethod
    def _result(status: str, message: str, detections: list[dict] | None = None) -> dict:
        return {
            "status": status,
            "message": message,
            "detections": detections or [],
        }

    @classmethod
    def _normalize_result(cls, value: Any) -> dict:
        if not isinstance(value, dict):
            return cls._result("FAILED", "SAM3 runner JSON must be an object.")

        status = value.get("status", "FAILED")
        if status not in {"SUCCESS", "NOT_FOUND", "NOT_AVAILABLE", "FAILED"}:
            status = "FAILED"

        detections = value.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        return {
            "status": status,
            "message": str(value.get("message", "")),
            "detections": detections,
        }

    @staticmethod
    def _summarize(value: str | None, limit: int = 500) -> str:
        text = (value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."
