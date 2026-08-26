"""Subprocess adapter for SAM3 inference in an isolated conda environment."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from llm_skill_robot.utils import resolve_project_path


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
        "pytorch_cuda_alloc_conf": "",
        "persistent_worker": True,
    }

    def __init__(self, config: dict | None = None):
        config = config or {}
        if "sam3_subprocess" in config:
            config = config["sam3_subprocess"] or {}
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.config["runner_script"] = str(
            self._resolve_runner_script(str(self.config["runner_script"]))
        )
        self._worker: subprocess.Popen | None = None
        self._worker_responses: queue.Queue[str] = queue.Queue()
        self._worker_lock = threading.Lock()
        self._worker_request_id = 0

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
        env = self._build_environment()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(self.config["timeout_sec"]),
                env=env,
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

    def detect_many(
        self,
        rgb_image_path: str,
        queries: list[str],
        min_confidence: float = 0.5,
    ) -> dict:
        """Run several text queries with one persistent SAM3 image session.

        The worker loads model weights once and executes ``set_image`` once
        per call. Individual prompts still receive independent masks and
        confidence scores.
        """
        normalized_queries = list(
            dict.fromkeys(
                query.strip()
                for query in queries
                if isinstance(query, str) and query.strip()
            )
        )
        if not normalized_queries:
            return {
                "status": "FAILED",
                "message": "No non-empty SAM3 queries supplied.",
                "query_results": [],
            }
        if not self.config.get("persistent_worker", True):
            return self._detect_many_one_shot(rgb_image_path, normalized_queries, min_confidence)
        if not self.config.get("enabled", True):
            return {
                "status": "NOT_AVAILABLE",
                "message": "SAM3 subprocess detector is disabled.",
                "query_results": [],
            }
        if not Path(str(self.config["runner_script"])).exists():
            return {
                "status": "NOT_AVAILABLE",
                "message": "SAM3 runner script does not exist.",
                "query_results": [],
            }

        with self._worker_lock:
            try:
                self._ensure_worker()
                self._worker_request_id += 1
                request_id = self._worker_request_id
                assert self._worker is not None and self._worker.stdin is not None
                self._worker.stdin.write(json.dumps({
                    "id": request_id,
                    "image": rgb_image_path,
                    "queries": normalized_queries,
                    "min_confidence": min_confidence,
                    "output_dir": str(self.config["output_dir"]),
                }) + "\n")
                self._worker.stdin.flush()
                response = self._read_worker_response(request_id)
            except (OSError, RuntimeError, queue.Empty) as exc:
                self.close()
                return {
                    "status": "FAILED",
                    "message": f"SAM3 persistent worker failed: {exc}",
                    "query_results": [],
                }
        return self._normalize_multi_result(response)

    def close(self) -> None:
        """Stop the optional worker and release its GPU/model memory."""
        worker, self._worker = self._worker, None
        if worker is None:
            return
        try:
            if worker.stdin is not None:
                worker.stdin.close()
            worker.terminate()
            worker.wait(timeout=3.0)
        except (OSError, subprocess.SubprocessError):
            try:
                worker.kill()
            except OSError:
                pass

    def _detect_many_one_shot(
        self,
        image_path: str,
        queries: list[str],
        confidence: float,
    ) -> dict:
        results = [
            {"query": query, **self.detect(image_path, query, confidence)}
            for query in queries
        ]
        status = (
            "SUCCESS"
            if any(item["status"] == "SUCCESS" for item in results)
            else results[-1]["status"]
        )
        return {
            "status": status,
            "message": "SAM3 queries completed.",
            "query_results": results,
        }

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        self.close()
        self._worker_responses = queue.Queue()
        self._worker = subprocess.Popen(
            self._build_worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Responses are JSONL on stdout.  Suppress library diagnostics so
            # their pipe cannot fill and block a long-lived worker.
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=self._build_environment(),
        )
        assert self._worker.stdout is not None
        threading.Thread(
            target=self._collect_worker_stdout,
            args=(self._worker.stdout,), daemon=True,
        ).start()

    def _collect_worker_stdout(self, stream) -> None:
        for line in stream:
            self._worker_responses.put(line)

    def _read_worker_response(self, request_id: int) -> dict:
        deadline = time.monotonic() + float(self.config["timeout_sec"])
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty()
            try:
                line = self._worker_responses.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                if self._worker is not None and self._worker.poll() is not None:
                    raise RuntimeError("SAM3 persistent worker exited before replying.")
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate library startup logs emitted on stdout
            if response.get("id") is None and response.get("status") in {
                "NOT_AVAILABLE",
                "FAILED",
            }:
                raise RuntimeError(str(response.get("message", "SAM3 worker failed to start.")))
            if response.get("id") == request_id:
                return response

    def _build_worker_command(self) -> list[str]:
        command = [
            str(self.config["conda_executable"]), "run", "--no-capture-output", "-n",
            str(self.config["conda_env"]), "python", str(self.config["runner_script"]), "--serve",
            "--device", str(self.config.get("device") or "auto"),
        ]
        checkpoint_path = str(self.config.get("checkpoint_path") or "")
        if checkpoint_path:
            command.extend(["--checkpoint", checkpoint_path])
        model_config = str(self.config.get("model_config") or "")
        if model_config:
            command.extend(["--model-config", model_config])
        return command

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

    def _build_environment(self) -> dict[str, str] | None:
        alloc_conf = str(self.config.get("pytorch_cuda_alloc_conf") or "").strip()
        device = str(self.config.get("device") or "").strip().lower()
        force_cpu = device == "cpu"
        if not alloc_conf and not force_cpu:
            return None
        env = os.environ.copy()
        if alloc_conf:
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
        if force_cpu:
            # Some SAM3/PyTorch paths still auto-place image tensors on CUDA when
            # CUDA is visible. Hide GPUs for the subprocess so CPU mode is truly
            # all-CPU and cannot mix cuda inputs with cpu model weights.
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    @staticmethod
    def _resolve_runner_script(runner_script: str) -> Path:
        return resolve_project_path(runner_script)

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

    @classmethod
    def _normalize_multi_result(cls, value: Any) -> dict:
        if not isinstance(value, dict):
            return {
                "status": "FAILED",
                "message": "SAM3 worker JSON must be an object.",
                "query_results": [],
            }
        status = value.get("status", "FAILED")
        if status not in {"SUCCESS", "NOT_FOUND", "NOT_AVAILABLE", "FAILED"}:
            status = "FAILED"
        results = value.get("query_results", [])
        if not isinstance(results, list):
            results = []
        normalized = []
        for item in results:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "query": str(item.get("query", "")),
                **cls._normalize_result(item),
            })
        return {
            "status": status,
            "message": str(value.get("message", "")),
            "query_results": normalized,
        }

    @staticmethod
    def _summarize(value: str | None, limit: int = 500) -> str:
        text = (value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."
