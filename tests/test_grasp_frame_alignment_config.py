import math
from pathlib import Path

import pytest
import yaml


def test_graspgenx_opening_axis_aligns_with_ros_robotiq_jaw_axis():
    config_path = Path(__file__).resolve().parents[1] / "config" / "grasping.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grasping = document["grasping"]
    transform = grasping["plan_only_preview"]["grasp_frame_to_ee_frame"]

    # GraspGenX opens along local X. The URDF mounts robotiq_140_base_link at
    # +90 degrees from tool0 and that model opens along local Y. After applying
    # the configured grasp-to-tool0 yaw, both undirected axes must be parallel.
    configured_yaw = float(transform["rotation_rpy_rad"][2])
    urdf_tool0_to_robotiq_yaw = math.pi / 2.0
    physical_jaw_yaw = configured_yaw + urdf_tool0_to_robotiq_yaw + math.pi / 2.0
    alignment = abs(math.cos(physical_jaw_yaw))

    assert grasping["graspgenx_subprocess"]["width_axis_index"] == 0
    assert alignment == pytest.approx(1.0, abs=1e-9)
