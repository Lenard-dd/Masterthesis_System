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


def test_candidate_lifecycle_records_stable_yaw_and_writes_artifact(tmp_path):
    runner = _load_runner_module()
    pose = np.eye(4, dtype=np.float32)
    yaw = np.deg2rad(40.0)
    pose[:2, 0] = [np.cos(yaw), np.sin(yaw)]
    pose[:2, 1] = [-np.sin(yaw), np.cos(yaw)]

    records = runner.initialize_candidate_lifecycle(
        pose[None], [0.8], ["obb_sparse"], width_axis_index=0,
    )
    path = runner.write_candidate_lifecycle_diagnostics(
        str(tmp_path), "screw_base_link.ply", {"candidates": records},
    )

    assert records[0]["candidate_id"] == "graspgenx_raw_1"
    assert records[0]["opening_axis_yaw_deg_mod_180"] == pytest.approx(40.0)
    assert Path(path).name == "screw_base_link_candidate_lifecycle.json"
    assert Path(path).is_file()


def test_environment_constraint_diagnostics_record_per_backoff_clearances():
    from types import SimpleNamespace

    runner = _load_runner_module()
    pose = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    gripper = SimpleNamespace(
        collision_mesh=object(),
        open_pointcloud=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        close_pointcloud=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        grasp_volume=np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]], dtype=np.float32),
    )

    result = runner.constrain_grasps_to_environment(
        lambda **kwargs: np.ones(len(kwargs["grasp_poses"]), dtype=bool),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        pose,
        gripper,
        static_threshold=0.01,
        static_samples=10,
        static_batch_size=10,
        closed_threshold=0.01,
        closed_samples=10,
        closed_enabled=True,
        min_backoff_m=0.0,
        max_backoff_m=0.0,
        backoff_step_m=0.001,
        min_target_contact_points=1,
        pregrasp_distance_m=0.1,
        retreat_distance_m=0.1,
        path_samples=3,
        path_surface_samples=10,
    )

    stage = result["candidate_stages"][0]
    trial = stage["backoff_trials"][0]
    assert trial["outcome"] == "selected"
    assert trial["open_surface_min_clearance_m"] == pytest.approx(1.0)
    assert trial["closed_surface_min_clearance_m"] == pytest.approx(1.0)
    assert trial["approach_min_clearance_m"] == pytest.approx(1.0)
    assert trial["retreat_min_clearance_m"] == pytest.approx(1.0)
    assert stage["selected_trajectory"]["execution_backoff_m"] == pytest.approx(0.0)


def test_local_support_plane_tracks_raised_surface_below_target():
    runner = _load_runner_module()
    xy = np.linspace(-0.02, 0.02, 9)
    scene = np.array([[x, y, 0.04 + 0.02 * x] for x in xy for y in xy], dtype=np.float32)
    target = np.array([[x, y, 0.05] for x in (-0.005, 0.005) for y in (-0.004, 0.004)], dtype=np.float32)

    plane = runner.estimate_local_support_plane(
        scene, target, min_inliers=20, inlier_threshold_m=0.001,
    )

    assert plane is not None
    assert plane["inlier_count"] >= 20
    assert plane["normal"][2] > 0.99
    assert -plane["d"] == pytest.approx(0.04, abs=0.002)


def test_target_contact_requires_distal_finger_pad_fraction():
    runner = _load_runner_module()
    pose = np.eye(4, dtype=np.float32)
    volume = np.array([[-0.02, -0.02, 0.10], [0.02, 0.02, 0.25]], dtype=np.float32)
    target = np.array(
        [[0.0, 0.0, z] for z in np.linspace(0.10, 0.25, 20)], dtype=np.float32
    )

    broad = runner.target_contact_evidence(target, pose, volume)
    pad = runner.target_contact_evidence(
        target, pose, volume, pad_contact_depth_m=0.04,
    )

    assert broad["fraction"] == pytest.approx(1.0)
    assert 0.20 <= pad["fraction"] <= 0.30
    assert pad["contact_bounds"][0][2] == pytest.approx(0.21)


def test_environment_constraint_rejects_closed_surface_below_local_plane_margin():
    from types import SimpleNamespace

    runner = _load_runner_module()
    xy = np.linspace(-0.025, 0.025, 9)
    scene = np.array([[x, y, 0.0] for x in xy for y in xy], dtype=np.float32)
    target = np.array([[x, y, 0.02] for x in (-0.004, 0.004) for y in (-0.004, 0.004)], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 0.005
    gripper = SimpleNamespace(
        collision_mesh=object(),
        open_pointcloud=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        close_pointcloud=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        grasp_volume=np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]], dtype=np.float32),
    )

    result = runner.constrain_grasps_to_environment(
        lambda **kwargs: np.ones(len(kwargs["grasp_poses"]), dtype=bool),
        scene, target, pose[None], gripper,
        static_threshold=0.001, static_samples=10, static_batch_size=1,
        closed_threshold=0.001, closed_samples=10, closed_enabled=True,
        min_backoff_m=0.0, max_backoff_m=0.0, backoff_step_m=0.001,
        min_target_contact_points=1, pregrasp_distance_m=0.05,
        retreat_distance_m=0.05, path_samples=3, path_surface_samples=10,
        support_plane_enabled=True, support_plane_clearance_m=0.010,
        support_plane_min_inliers=20, support_plane_inlier_threshold_m=0.001,
    )

    trial = result["candidate_stages"][0]["backoff_trials"][0]
    assert result["mask"].tolist() == [False]
    assert trial["outcome"] == "rejected_support_plane"
    assert trial["support_plane_clearance_m"] == pytest.approx(0.005, abs=1e-4)


def test_local_support_plane_prefers_dominant_surface_over_sparse_higher_leakage():
    runner = _load_runner_module()
    xy = np.linspace(-0.02, 0.02, 12)
    support = np.array([[x, y, -0.003] for x in xy for y in xy], dtype=np.float32)
    leak_xy = np.linspace(-0.01, 0.01, 6)
    leakage = np.array([[x, y, 0.002] for x in leak_xy for y in leak_xy], dtype=np.float32)
    scene = np.concatenate([support, leakage], axis=0)
    target = np.array(
        [[x, y, z] for x in (-0.005, 0.005) for y in (-0.004, 0.004) for z in (-0.002, 0.003)],
        dtype=np.float32,
    )

    plane = runner.estimate_local_support_plane(
        scene, target, min_inliers=20, inlier_threshold_m=0.001,
    )

    assert plane is not None
    assert -plane["d"] == pytest.approx(-0.003, abs=0.001)
    assert plane["inlier_count"] > len(leakage)
    assert plane["inlier_ratio"] > 0.60
