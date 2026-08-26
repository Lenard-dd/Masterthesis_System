import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

from llm_skill_robot.perception.sam3_subprocess_adapter import SAM3SubprocessAdapter


def _make_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "sam3_runner.py"
    runner.write_text("# runner\n", encoding="utf-8")
    return runner


def test_builds_expected_command(tmp_path):
    runner = _make_runner(tmp_path)
    adapter = SAM3SubprocessAdapter(
        {
            "conda_env": "sam3",
            "conda_executable": "conda",
            "runner_script": str(runner),
            "output_dir": "data/perception_outputs",
            "timeout_sec": 60.0,
            "device": "cuda",
            "checkpoint_path": "checkpoint.pt",
            "model_config": "model.yaml",
        }
    )

    command = adapter._build_command("rgb.png", "red cube", 0.7)

    assert command == [
        "conda",
        "run",
        "-n",
        "sam3",
        "python",
        str(runner),
        "--image",
        "rgb.png",
        "--query",
        "red cube",
        "--min-confidence",
        "0.7",
        "--output-dir",
        "data/perception_outputs",
        "--checkpoint",
        "checkpoint.pt",
        "--model-config",
        "model.yaml",
        "--device",
        "cuda",
    ]


def test_builds_persistent_worker_command(tmp_path):
    runner = _make_runner(tmp_path)
    adapter = SAM3SubprocessAdapter(
        {
            "conda_env": "sam3",
            "conda_executable": "conda",
            "runner_script": str(runner),
            "device": "cuda",
            "checkpoint_path": "checkpoint.pt",
        }
    )

    assert adapter._build_worker_command() == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "sam3",
        "python",
        str(runner),
        "--serve",
        "--device",
        "cuda",
        "--checkpoint",
        "checkpoint.pt",
    ]


def test_normalize_multi_result_preserves_each_query_result():
    result = SAM3SubprocessAdapter._normalize_multi_result(
        {
            "status": "SUCCESS",
            "message": "processed",
            "query_results": [
                {
                    "query": "black cube",
                    "status": "SUCCESS",
                    "message": "found",
                    "detections": [{"label": "black cube"}],
                },
                {
                    "query": "red cube",
                    "status": "NOT_FOUND",
                    "message": "none",
                    "detections": [],
                },
            ],
        }
    )

    assert result["status"] == "SUCCESS"
    assert [item["query"] for item in result["query_results"]] == [
        "black cube",
        "red cube",
    ]
    assert result["query_results"][0]["detections"] == [{"label": "black cube"}]


def test_missing_runner_script_returns_not_available(tmp_path):
    adapter = SAM3SubprocessAdapter({"runner_script": str(tmp_path / "missing.py")})

    assert adapter.is_available() is False
    result = adapter.detect("rgb.png", "red cube")

    assert result["status"] == "NOT_AVAILABLE"
    assert "runner script does not exist" in result["message"]


def test_default_relative_runner_resolves_from_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    adapter = SAM3SubprocessAdapter()

    runner_script = Path(adapter.config["runner_script"])

    assert runner_script.is_absolute()
    assert runner_script.name == "sam3_runner.py"
    assert runner_script.exists()


def test_cpu_device_hides_cuda_from_subprocess(tmp_path):
    runner = _make_runner(tmp_path)
    adapter = SAM3SubprocessAdapter(
        {
            "runner_script": str(runner),
            "device": "cpu",
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
        }
    )

    env = adapter._build_environment()

    assert env is not None
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


@patch("llm_skill_robot.perception.sam3_subprocess_adapter.subprocess.run")
def test_detect_success_parses_json(mock_run, tmp_path):
    runner = _make_runner(tmp_path)
    payload = {
        "status": "SUCCESS",
        "message": "ok",
        "detections": [
            {
                "label": "red cube",
                "confidence": 0.85,
                "bbox_xyxy": [1, 2, 3, 4],
                "mask_path": "mask.png",
                "overlay_path": "overlay.png",
                "metadata": {},
            }
        ],
    }
    mock_run.return_value = Mock(returncode=0, stdout=json.dumps(payload), stderr="")
    adapter = SAM3SubprocessAdapter({"runner_script": str(runner)})

    result = adapter.detect("rgb.png", "red cube", 0.6)

    assert result == payload
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["capture_output"] is True
    assert mock_run.call_args.kwargs["text"] is True


@patch("llm_skill_robot.perception.sam3_subprocess_adapter.subprocess.run")
def test_detect_nonzero_exit_returns_failed(mock_run, tmp_path):
    runner = _make_runner(tmp_path)
    mock_run.return_value = Mock(returncode=2, stdout="", stderr="boom")
    adapter = SAM3SubprocessAdapter({"runner_script": str(runner)})

    result = adapter.detect("rgb.png", "red cube")

    assert result["status"] == "FAILED"
    assert "exited with code 2" in result["message"]
    assert "boom" in result["message"]


@patch("llm_skill_robot.perception.sam3_subprocess_adapter.subprocess.run")
def test_detect_timeout_returns_failed(mock_run, tmp_path):
    runner = _make_runner(tmp_path)
    mock_run.side_effect = subprocess.TimeoutExpired(["conda"], timeout=60)
    adapter = SAM3SubprocessAdapter({"runner_script": str(runner), "timeout_sec": 60})

    result = adapter.detect("rgb.png", "red cube")

    assert result["status"] == "FAILED"
    assert "timed out" in result["message"]


@patch("llm_skill_robot.perception.sam3_subprocess_adapter.subprocess.run")
def test_detect_invalid_json_returns_failed(mock_run, tmp_path):
    runner = _make_runner(tmp_path)
    mock_run.return_value = Mock(returncode=0, stdout="not json", stderr="debug")
    adapter = SAM3SubprocessAdapter({"runner_script": str(runner)})

    result = adapter.detect("rgb.png", "red cube")

    assert result["status"] == "FAILED"
    assert "valid JSON" in result["message"]
    assert "not json" in result["message"]
    assert "debug" in result["message"]
