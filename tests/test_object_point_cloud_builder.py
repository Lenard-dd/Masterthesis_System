from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
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
        {
            "output_dir": str(tmp_path / "point_clouds"),
            "depth_scale": 0.001,
            "extent_percentile_low": 0.0,
            "extent_percentile_high": 100.0,
        }
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
    assert len(result["planar_principal_axes"]) == 2
    assert result["visible_aabb_dimensions_m"] == {
        "x": 0.01, "y": 0.01, "z": 0.0,
    }
    assert result["footprint_dimensions_m"] == {"x": 0.01, "y": 0.01}
    assert result["dimension_estimation"]["height_reliable"] is False
    assert result["scene_num_points"] == 16
    assert result["environment_num_points"] == 0
    assert result["scene_point_cloud_path"].endswith("red_cube_1_scene.ply")
    assert result["environment_point_cloud_path"].endswith("red_cube_1_environment.ply")


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


def test_environment_cloud_removes_dilated_target_but_keeps_table(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255
    Image.fromarray(mask).save(mask_path)
    builder = ObjectPointCloudBuilder({
        "output_dir": str(tmp_path / "clouds"),
        "scene_target_mask_dilation_px": 2,
        "scene_max_points": 1000,
    })

    result = builder.build(
        {"object_id": "small_part", "metadata": {"mask_path": str(mask_path)}},
        {"image": np.full((20, 20), 1000, dtype=np.uint16), "encoding": "16UC1"},
        {"k": np.array([100.0, 0.0, 10.0, 0.0, 100.0, 10.0, 0.0, 0.0, 1.0])},
    )

    assert result["success"] is True
    assert result["scene_num_points"] == 400
    assert result["environment_num_points"] == 336
    assert Path(result["environment_point_cloud_path"]).is_file()


def test_table_plane_ransac_estimates_object_height_from_mask_and_surrounding_table(tmp_path):
    mask_path = tmp_path / "object_mask.png"
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[12:28, 12:28] = 255
    Image.fromarray(mask).save(mask_path)
    # The table is 1.0 m away; the object top is 5 cm closer to the camera.
    depth = np.full((40, 40), 1000, dtype=np.uint16)
    depth[mask > 0] = 950
    builder = ObjectPointCloudBuilder({
        "output_dir": str(tmp_path / "clouds"),
        "table_plane_margin_px": 8,
        "table_plane_min_points": 100,
        "extent_percentile_low": 0.0,
        "extent_percentile_high": 100.0,
    })

    result = builder.build(
        {"object_id": "cube", "metadata": {"mask_path": str(mask_path)}},
        {"image": depth, "encoding": "16UC1"},
        {"k": np.array([100.0, 0.0, 20.0, 0.0, 100.0, 20.0, 0.0, 0.0, 1.0])},
    )

    estimate = result["height_estimation"]
    assert result["success"] is True
    assert estimate["usable"] is True
    assert estimate["height_m"] == pytest.approx(0.05, abs=0.001)
    assert estimate["plane_inlier_ratio"] > 0.95
    assert estimate["plane_normal"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert abs(estimate["plane_offset"]) == pytest.approx(1.0, abs=1e-6)
    assert result["scene_support_completion_num_points"] == 256
    assert result["dimension_estimation"]["height_reliable"] is True

    environment_lines = Path(result["environment_point_cloud_path"]).read_text().splitlines()
    support = np.asarray(
        [[float(value) for value in line.split()] for line in environment_lines[10:]],
        dtype=float,
    )
    # The synthesized points underneath the 0.95 m object top lie on the
    # fitted 1.0 m support plane and are therefore available to collision tests.
    assert np.count_nonzero(np.isclose(support[:, 2], 1.0, atol=1e-6)) >= 256
