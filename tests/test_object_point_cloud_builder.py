from types import SimpleNamespace

import numpy as np
from PIL import Image

from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder


def test_numpy_camera_info_intrinsics_build_point_cloud(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    Image.fromarray(mask).save(mask_path)
    depth = np.full((4, 4), 1000, dtype=np.uint16)
    camera_info = {
        "k": np.array([100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0])
    }
    scene_object = {
        "object_id": "red_cube_1",
        "metadata": {"mask_path": str(mask_path)},
    }
    builder = ObjectPointCloudBuilder(
        {"output_dir": str(tmp_path / "point_clouds"), "depth_scale": 0.001}
    )

    result = builder.build(
        scene_object,
        {"image": depth, "encoding": "16UC1"},
        camera_info,
    )

    assert result["success"] is True
    assert result["num_points"] == 4
    assert result["point_cloud_path"].endswith("red_cube_1.ply")
    assert result["centroid_m"] == {"x": -0.005, "y": -0.005, "z": 1.0}


def test_namespace_numpy_camera_info_intrinsics_build_point_cloud(tmp_path):
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(mask_path)
    camera_info = SimpleNamespace(
        k=np.array([100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0])
    )
    builder = ObjectPointCloudBuilder({"output_dir": str(tmp_path / "clouds")})

    result = builder.build(
        {"object_id": "cube", "metadata": {"mask_path": str(mask_path)}},
        {"image": np.full((2, 2), 1.0, dtype=np.float32), "encoding": "32FC1"},
        camera_info,
    )

    assert result["success"] is True
    assert result["num_points"] == 4
