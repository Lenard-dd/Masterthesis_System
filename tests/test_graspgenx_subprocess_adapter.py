import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

from llm_skill_robot.grasping.graspgenx_subprocess_adapter import (
    GraspGenXSubprocessAdapter,
)


def _make_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "graspgenx_runner.py"
    runner.write_text("# runner\n", encoding="utf-8")
    return runner


def test_default_runner_resolves_from_project_root():
    adapter = GraspGenXSubprocessAdapter()

    runner = adapter._runner_path()

    assert runner is not None
    assert runner.name == "graspgenx_zmq_runner.py"
    assert runner.exists()


def test_builds_expected_zmq_runner_command(tmp_path):
    runner = _make_runner(tmp_path)
    adapter = GraspGenXSubprocessAdapter(
        {
            "conda_env": "graspgenx",
            "conda_executable": "conda",
            "runner_script": str(runner),
            "output_dir": "data/grasping/graspgenx_outputs",
            "server_host": "127.0.0.1",
            "server_port": 5557,
            "gripper_name": "robotiq_2f_140",
            "num_grasps": 50,
            "grasp_threshold": 0.1,
            "topk_num_grasps": 10,
            "timeout_ms": 30000,
            "estimate_gripper_width": True,
            "width_axis_index": 1,
            "width_clearance_m": 0.02,
        }
    )

    command = adapter._build_command("object.ply", runner)

    assert command == [
        "conda",
        "run",
        "-n",
        "graspgenx",
        "python",
        str(runner),
        "--point-cloud",
        "object.ply",
        "--output-dir",
        "data/grasping/graspgenx_outputs",
        "--host",
        "127.0.0.1",
        "--port",
        "5557",
        "--gripper-name",
        "robotiq_2f_140",
        "--num-grasps",
        "50",
        "--grasp-threshold",
        "0.1",
        "--topk-num-grasps",
        "10",
        "--timeout-ms",
        "30000",
        "--width-axis-index",
        "1",
        "--width-clearance-m",
        "0.02",
        "--estimate-gripper-width",
    ]


def test_build_command_includes_scene_collision_filter(tmp_path):
    runner = _make_runner(tmp_path)
    adapter = GraspGenXSubprocessAdapter({
        "runner_script": str(runner),
        "scene_collision_threshold_m": 0.004,
        "scene_collision_samples": 1200,
        "scene_collision_batch_size": 2,
    })

    command = adapter._build_command(
        "object.ply", runner, scene_point_cloud_path="environment.ply"
    )

    expected_values = {
        "--scene-point-cloud": "environment.ply",
        "--collision-threshold-m": "0.004",
        "--collision-samples": "1200",
        "--collision-batch-size": "2",
        "--collision-closed-finger-samples": "3500",
        "--collision-closed-finger-threshold-m": "0.008",
        "--collision-closed-finger-backoff-m": "0.04",
        "--constraint-min-backoff-m": "0.025",
        "--constraint-max-backoff-m": "0.045",
        "--constraint-backoff-step-m": "0.0025",
        "--constraint-min-target-contact-points": "12",
        "--constraint-min-target-contact-fraction": "0.2",
        "--constraint-pad-contact-depth-m": "0.045",
        "--constraint-support-plane-margin-m": "0.012",
        "--constraint-support-plane-clearance-m": "0.01",
        "--constraint-support-plane-inlier-threshold-m": "0.003",
        "--constraint-support-plane-max-tilt-deg": "20.0",
        "--constraint-support-plane-min-inliers": "24",
        "--constraint-pregrasp-distance-m": "0.06",
        "--constraint-retreat-distance-m": "0.1",
        "--constraint-path-samples": "5",
        "--constraint-path-surface-samples": "1800",
        "--constraint-yaw-bin-deg": "15.0",
        "--constraint-max-per-yaw-bin": "2",
    }
    for flag, value in expected_values.items():
        assert command[command.index(flag) + 1] == value
    assert "--environment-constrained-generation" in command
    assert "--constraint-support-plane-enabled" in command
    assert "--collision-closed-finger-enabled" in command
    assert "--no-collision-fallback-to-object-only" in command


def test_missing_runner_script_returns_not_available(tmp_path):
    adapter = GraspGenXSubprocessAdapter({"runner_script": str(tmp_path / "missing.py")})

    result = adapter.generate_candidates("object.ply")

    assert result["status"] == "NOT_AVAILABLE"
    assert "runner_script" in result["message"]


@patch("llm_skill_robot.grasping.graspgenx_subprocess_adapter.subprocess.run")
def test_generate_candidates_parses_runner_json(mock_run, tmp_path):
    runner = _make_runner(tmp_path)
    point_cloud = tmp_path / "object.ply"
    point_cloud.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    payload = {
        "status": "SUCCESS",
        "message": "ok",
        "candidates": [
            {
                "candidate_id": "graspgenx_1",
                "valid": True,
                "score": 0.9,
                "transform": [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
            }
        ],
    }
    mock_run.return_value = Mock(returncode=0, stdout=json.dumps(payload), stderr="")
    adapter = GraspGenXSubprocessAdapter({"runner_script": str(runner)})

    result = adapter.generate_candidates(str(point_cloud))

    assert result["success"] is True
    assert result["candidates"][0]["score"] == 0.9
    mock_run.assert_called_once()


@patch("llm_skill_robot.grasping.graspgenx_subprocess_adapter.subprocess.run")
def test_generate_candidates_parses_json_after_noisy_stdout(mock_run, tmp_path):
    runner = _make_runner(tmp_path)
    point_cloud = tmp_path / "object.ply"
    point_cloud.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    payload = {
        "status": "SUCCESS",
        "message": "Received 100 candidate(s).",
        "candidates": [{"candidate_id": "graspgenx_1", "valid": True, "score": 0.96}],
    }
    mock_run.return_value = Mock(
        returncode=0,
        stdout=(
            "2026-06-25 12:57:35,435 - OpenGL - INFO - noisy log\n"
            "2026-06-25 12:57:37,026 - graspgenx.serving.zmq_client - INFO - Connected\n"
            f"{json.dumps(payload)}\n"
        ),
        stderr="warning",
    )
    adapter = GraspGenXSubprocessAdapter({"runner_script": str(runner)})

    result = adapter.generate_candidates(str(point_cloud))

    assert result["success"] is True
    assert result["message"] == "Received 100 candidate(s)."
    assert result["candidates"][0]["score"] == 0.96


@patch("llm_skill_robot.grasping.graspgenx_subprocess_adapter.subprocess.run")
def test_generate_candidates_nonzero_exit_returns_failed(mock_run, tmp_path):
    runner = _make_runner(tmp_path)
    point_cloud = tmp_path / "object.ply"
    point_cloud.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    mock_run.return_value = Mock(returncode=2, stdout="", stderr="boom")
    adapter = GraspGenXSubprocessAdapter({"runner_script": str(runner)})

    result = adapter.generate_candidates(str(point_cloud))

    assert result["status"] == "FAILED"
    assert "exited with code 2" in result["message"]
    assert "boom" in result["message"]


@patch("llm_skill_robot.grasping.graspgenx_subprocess_adapter.subprocess.run")
def test_generate_candidates_timeout_returns_failed(mock_run, tmp_path):
    runner = _make_runner(tmp_path)
    point_cloud = tmp_path / "object.ply"
    point_cloud.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    mock_run.side_effect = subprocess.TimeoutExpired(["conda"], timeout=60)
    adapter = GraspGenXSubprocessAdapter({"runner_script": str(runner), "timeout_sec": 60})

    result = adapter.generate_candidates(str(point_cloud))

    assert result["status"] == "FAILED"
    assert "timed out" in result["message"]
