from pathlib import Path

from llm_skill_robot.perception.d435i_backend import D435iBackend
from llm_skill_robot.perception.perception_factory import (
    create_perception_pipeline,
    load_perception_config,
    perception_context_from_config,
)
from llm_skill_robot.perception.sam3_subprocess_adapter import SAM3SubprocessAdapter
from llm_skill_robot.perception.unavailable_perception_backend import (
    UnavailablePerceptionBackend,
)


def _d435i_sam3_config():
    return {
        "perception": {
            "mode": "d435i_sam3",
            "detector": "sam3_subprocess",
            "target_frame": "base_link",
            "require_base_link_pose_for_pick": True,
        },
        "d435i": {
            "rgb_topic": "/rgb",
            "depth_topic": "/depth",
            "camera_info_topic": "/camera_info",
            "wait_timeout_sec": 0.01,
        },
        "sam3_subprocess": {
            "runner_script": "scripts/sam3_runner.py",
            "conda_env": "sam3",
        },
    }


def test_factory_creates_d435i_sam3_pipeline_from_config():
    pipeline = create_perception_pipeline(config=_d435i_sam3_config())

    assert isinstance(pipeline.backend, D435iBackend)
    assert isinstance(pipeline.backend.detector_adapter, SAM3SubprocessAdapter)
    assert pipeline.backend.config["rgb_topic"] == "/rgb"
    assert pipeline.backend.perception_config["target_frame"] == "base_link"


def test_factory_returns_unavailable_pipeline_for_unknown_mode():
    pipeline = create_perception_pipeline(config={"perception": {"mode": "mock"}})

    assert isinstance(pipeline.backend, UnavailablePerceptionBackend)


def test_perception_context_from_config_uses_base_link_pick_settings():
    context = perception_context_from_config(_d435i_sam3_config())

    assert context["require_base_link_pose_for_pick"] is True
    assert context["target_frame"] == "base_link"


def test_load_perception_config_returns_empty_dict_when_missing(tmp_path):
    missing_path = Path(tmp_path) / "missing_perception.yaml"

    assert load_perception_config(missing_path) == {}
