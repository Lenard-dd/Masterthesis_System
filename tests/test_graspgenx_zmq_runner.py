import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "graspgenx_zmq_runner.py"
    spec = importlib.util.spec_from_file_location("graspgenx_zmq_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_ascii_ply_xyz_centers_points(tmp_path):
    runner = _load_runner_module()
    ply = tmp_path / "object.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "1.0 0.0 0.0",
                "3.0 0.0 0.0",
            ]
        ),
        encoding="utf-8",
    )

    points = runner.load_point_cloud(str(ply))

    assert points.shape == (2, 3)
    assert points[:, 0].tolist() == [-1.0, 1.0]


def test_load_point_cloud_with_metadata_records_centering(tmp_path):
    runner = _load_runner_module()
    xyz = tmp_path / "object.xyz"
    xyz.write_text("1.0 2.0 3.0\n3.0 4.0 5.0\n", encoding="utf-8")

    points, metadata = runner.load_point_cloud_with_metadata(str(xyz))

    assert points.tolist() == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    assert metadata["centered_for_graspgenx"] is True
    assert metadata["centering_method"] == "subtract_finite_xyz_mean"
    assert metadata["raw_point_count"] == 2
    assert metadata["finite_point_count"] == 2
    assert metadata["original_centroid_m"] == pytest.approx(
        {"x": 2.0, "y": 3.0, "z": 4.0}
    )
    assert metadata["centered_centroid_m"] == pytest.approx(
        {"x": 0.0, "y": 0.0, "z": 0.0}
    )


def test_make_candidates_converts_grasps_and_scores():
    runner = _load_runner_module()
    grasps = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    scores = np.array([0.87], dtype=np.float32)

    candidates = runner.make_candidates(
        grasps, scores, gripper_name="robotiq_2f_140", default_gripper_width_m=None
    )

    assert candidates[0]["candidate_id"] == "graspgenx_1"
    assert candidates[0]["valid"] is True
    assert candidates[0]["score"] == np.float32(0.87).item()
    assert candidates[0]["pose"]["position"] == pytest.approx(
        {"x": 0.1, "y": 0.2, "z": 0.3}
    )
    assert candidates[0]["approach_vector"] == pytest.approx([0.0, 0.0, 1.0])
    assert candidates[0]["closing_axis"] == pytest.approx([0.0, 1.0, 0.0])
    assert "gripper_width_m" not in candidates[0]


def test_make_candidates_estimates_width_from_point_cloud_projection():
    runner = _load_runner_module()
    grasps = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    scores = np.array([0.87], dtype=np.float32)
    point_cloud = np.array(
        [
            [0.0, -0.02, 0.0],
            [0.0, 0.025, 0.0],
            [0.0, 0.0, 0.01],
        ],
        dtype=np.float32,
    )

    candidates = runner.make_candidates(
        grasps,
        scores,
        point_cloud=point_cloud,
        gripper_name="robotiq_2f_140",
        default_gripper_width_m=None,
        estimate_gripper_width=True,
        width_axis_index=1,
        width_clearance_m=0.01,
    )

    assert candidates[0]["gripper_width_m"] == pytest.approx(0.055)
    assert (
        candidates[0]["metadata"]["gripper_width_source"]
        == "point_cloud_projection_estimate"
    )


def test_make_candidates_can_attach_explicit_default_width():
    runner = _load_runner_module()
    grasps = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    scores = np.array([0.5], dtype=np.float32)

    candidates = runner.make_candidates(
        grasps, scores, gripper_name="robotiq_2f_140", default_gripper_width_m=0.04
    )

    assert candidates[0]["gripper_width_m"] == 0.04
