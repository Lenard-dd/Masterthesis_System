from pathlib import Path
import importlib.util
import json
import sys
from types import ModuleType

import numpy as np
import pytest

from llm_skill_robot.grasping.graspgenx_grasp_planner import (
    GraspGenXGraspPlanner,
    screen_candidate_order,
)
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder


class RecordingAdapter:
    def __init__(self, output_dir):
        self.config = {"strategy": "sparse", "output_dir": str(output_dir)}
        self.calls = []

    def generate_candidates(self, path, *, strategy=None):
        self.calls.append((path, strategy))
        return {"success": True, "message": "ok", "candidates": []}


class TranslatedTF:
    target_frame = "base_link"

    def transform_points(self, points, source_frame):
        assert source_frame == "camera_color_optical_frame"
        return np.asarray(points) + [1.0, 2.0, 3.0]


def _cloud(tmp_path):
    source = tmp_path / "object.ply"
    points = np.column_stack((np.linspace(0.0, 0.09, 10), np.zeros(10), np.ones(10)))
    ObjectPointCloudBuilder._write_ascii_ply(source, points)
    metadata = {
        "point_cloud_path": str(source),
        "frame": "camera_color_optical_frame",
        "centroid_m": {"x": 0.045, "y": 0.0, "z": 1.0},
    }
    return source, metadata


def test_sparse_converts_cloud_and_localization_metadata_together(tmp_path):
    source, metadata = _cloud(tmp_path)
    adapter = RecordingAdapter(tmp_path / "outputs")
    planner = GraspGenXGraspPlanner(adapter)

    planner.generate_candidates(str(source), point_cloud=metadata, transformer=TranslatedTF())

    output = Path(adapter.calls[0][0])
    assert output != source
    assert adapter.calls[0][1] is None
    assert metadata["point_cloud_path"] == str(output)
    assert metadata["frame"] == "base_link"
    assert metadata["centroid_m"] == {"x": 1.045, "y": 2.0, "z": 4.0}
    assert metadata["original_point_cloud_path"] == str(source)
    assert np.allclose(planner._read_ascii_ply_xyz(output)[:, 2], 4.0)


def test_missing_tf_uses_original_diffusion_without_mutating_metadata(tmp_path):
    source, metadata = _cloud(tmp_path)
    before = dict(metadata)
    adapter = RecordingAdapter(tmp_path / "outputs")
    planner = GraspGenXGraspPlanner(adapter)

    result = planner.generate_candidates(str(source), point_cloud=metadata, transformer=None)

    assert adapter.calls == [(str(source), "diffusion")]
    assert metadata == before
    assert "used diffusion" in result["message"]


def test_moveit_screen_keeps_sparse_candidates_even_when_diffusion_scores_higher():
    diffusion = [{"score": 0.99 - i * 0.01, "metadata": {"generation_branch": "diffusion"}} for i in range(10)]
    sparse = [{"score": 0.30, "metadata": {"generation_branch": "obb_sparse"}}]

    selected = screen_candidate_order(diffusion + sparse, max_candidates=5, sparse_quota=2)

    assert len(selected) == 5
    assert selected[0] is sparse[0]
    assert sum(item in diffusion for item in selected) == 4


def test_moveit_screen_preserves_environment_selected_diverse_order():
    candidates = [
        {
            "candidate_id": f"safe_{index}",
            "score": score,
            "metadata": {
                "generation_branch": "obb_sparse",
                "environment_constrained_trajectory": True,
            },
        }
        for index, score in enumerate((0.40, 0.99, 0.60))
    ]

    selected = screen_candidate_order(candidates, max_candidates=3, sparse_quota=1)

    assert [item["candidate_id"] for item in selected] == ["safe_0", "safe_1", "safe_2"]


def test_sparse_adapter_requests_sparse_action_without_changing_diffusion_defaults():
    sparse = GraspGenXSubprocessAdapter({"strategy": "sparse", "sparse_quota": 8})
    legacy = GraspGenXSubprocessAdapter({"strategy": "diffusion"})

    sparse_command = sparse._build_command("object.ply")
    legacy_command = legacy._build_command("object.ply")

    assert sparse_command[sparse_command.index("--strategy") + 1] == "sparse"
    assert sparse_command[sparse_command.index("--sparse-quota") + 1] == "8"
    assert "--strategy" not in legacy_command


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_collision_filter_rejects_closed_finger_surface_collision():
    runner = _script("graspgenx_zmq_runner")
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, 0, 3] = 1.0
    scene = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    closed_surface = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]], dtype=np.float32)

    mask = runner.filter_grasps_against_closed_surface(
        scene,
        poses,
        closed_surface,
        collision_threshold=0.006,
        surface_samples=2,
        backoff_m=0.0,
    )

    assert mask.tolist() == [False, True]


def test_target_residual_cleanup_keeps_lower_support_surface():
    runner = _script("graspgenx_zmq_runner")
    target = np.array([
        [x, y, z]
        for x in (-0.01, 0.0, 0.01)
        for y in (-0.01, 0.0, 0.01)
        for z in (0.010, 0.012)
    ], dtype=np.float32)
    environment = np.array([
        [0.0, 0.0, 0.000],   # support below the target
        [0.0, 0.0, 0.011],   # leaked target edge
        [0.05, 0.05, 0.020], # unrelated scene point
    ], dtype=np.float32)

    cleaned, removed = runner.remove_target_residuals(environment, target)

    assert removed == 1
    assert np.allclose(cleaned, [
        [0.0, 0.0, 0.0],
        [0.05, 0.05, 0.02],
    ])


def test_static_collision_pose_uses_the_executed_grasp_backoff():
    runner = _script("graspgenx_zmq_runner")
    pose = np.eye(4, dtype=np.float32)[None]
    pose[0, :3, 2] = [0.0, 0.0, -1.0]

    adjusted = runner.apply_grasp_backoff(pose, 0.04)

    assert adjusted[0, :3, 3].tolist() == pytest.approx([0.0, 0.0, 0.04])
    assert pose[0, :3, 3].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("supported", [True, False])
def test_runner_round_trip_parses_negative_offsets_and_handles_legacy_server(
    tmp_path, monkeypatch, capsys, supported,
):
    source, _ = _cloud(tmp_path)
    runner = _script("graspgenx_zmq_runner")
    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.server_metadata = {"supported_actions": ["infer_sparse"] if supported else []}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def _request(self, request):
            calls.append(request)
            return {"grasps": np.eye(4)[None], "confidences": [0.8], "sources": ["obb_sparse"]}

        def infer(self, points, **kwargs):
            calls.append({"action": "infer"})
            return np.eye(4)[None], [0.8]

    module = ModuleType("graspgenx.serving.zmq_client")
    module.GraspGenXClient = Client
    monkeypatch.setitem(sys.modules, "graspgenx.serving.zmq_client", module)
    adapter = GraspGenXSubprocessAdapter({"strategy": "sparse", "output_dir": str(tmp_path)})
    command = adapter._build_command(str(source))
    monkeypatch.setattr(sys, "argv", ["runner"] + command[command.index("--point-cloud"):])

    assert runner.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["metadata"]["strategy_used"] == ("sparse" if supported else "diffusion")
    assert output["metadata"]["sparse_fallback"] is (not supported)
    if supported:
        assert calls[0]["moe_z_offsets_cm"] == [0.0]
        assert output["candidates"][0]["metadata"]["generation_branch"] == "obb_sparse"
    else:
        assert calls[0]["action"] == "infer"
        assert "fallback" in output["message"]


def test_rotated_camera_cloud_is_localized_in_base_only_once(tmp_path):
    from llm_skill_robot.ros_graspgenx_plan_only_demo import localize_graspgenx_candidate
    source, metadata = _cloud(tmp_path)

    class RotatedTF(TranslatedTF):
        def transform_points(self, points, frame):
            # Camera +Z points along base +X, rather than base +Z.
            rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            return np.asarray(points) @ rotation.T + [0.2, 0.3, 0.4]

    planner = GraspGenXGraspPlanner(RecordingAdapter(tmp_path / "outputs"))
    planner.generate_candidates(str(source), point_cloud=metadata, transformer=RotatedTF())
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1, -1, -1])
    candidate = _script("graspgenx_zmq_runner").make_candidates(
        pose[None], [0.8], "robotiq_2f_140", None,
    )[0]
    localized = localize_graspgenx_candidate(candidate, metadata)
    assert localized["pose_frame"] == "base_link"
    assert localized["pose"]["position"] == pytest.approx({"x": 1.2, "y": 0.3, "z": 0.355})
    assert localized["approach_vector"] == [0, 0, -1]
    assert localize_graspgenx_candidate(localized, metadata) == localized


def test_extended_server_returns_all_sparse_yaws_before_environment_filter(monkeypatch):
    server_script = _script("graspgenx_sparse_server")
    instances = []
    calls = []

    class Upstream:
        def __init__(self, **kwargs):
            self.default_gripper = kwargs["default_gripper"]
            instances.append(self)

        def serve_forever(self):
            pass

        def _get_sampler(self, name):
            return name

        def _dispatch(self, request):
            return {"legacy_action": request["action"]}

        def _handle_metadata(self):
            return {"model": "original"}

    def moe(pc, sampler, **kwargs):
        calls.append(kwargs)
        return {
            "grasps_obb": np.repeat(np.eye(4)[None], 3, axis=0),
            "scores_obb": np.array([0.3, 0.5, 0.4]),
            "grasps_diff": np.repeat(np.eye(4)[None], 4, axis=0),
            "scores_diff": np.array([0.99, 0.98, 0.97, 0.96]),
            "skipped_obb": False,
        }

    upstream_module = ModuleType("graspgenx.serving.zmq_server")
    upstream_module.GraspGenXZMQServer = Upstream
    moe_module = ModuleType("graspgenx.samplers.graspmoe")
    moe_module.run_graspmoe = moe
    monkeypatch.setitem(sys.modules, upstream_module.__name__, upstream_module)
    monkeypatch.setitem(sys.modules, moe_module.__name__, moe_module)
    monkeypatch.setattr(sys, "argv", ["server", "--config", "unused", "--assets_dir", "unused"])
    server_script.main()
    server = instances[0]
    assert server._dispatch({"action": "infer"}) == {"legacy_action": "infer"}
    assert "infer_sparse" in server._handle_metadata()["supported_actions"]
    result = server._dispatch({
        "action": "infer_sparse", "point_cloud": np.zeros((10, 3)),
        "topk_num_grasps": 4, "sparse_quota": 2,
    })
    assert calls[0]["obb_density"] == "sparse"
    assert result["sources"] == ["obb_sparse", "obb_sparse", "obb_sparse", "diffusion"]
    assert result["confidences"] == pytest.approx([0.5, 0.4, 0.3, 0.99])


def test_yaw_diversity_is_applied_after_scene_filtering():
    runner = _script("graspgenx_zmq_runner")
    poses = []
    scores = []
    for yaw_deg, score in ((1, .99), (2, .98), (3, .97), (45, .90), (90, .80)):
        yaw = np.deg2rad(yaw_deg)
        pose = np.eye(4, dtype=np.float32)
        pose[:2, 0] = [np.cos(yaw), np.sin(yaw)]
        pose[:2, 1] = [-np.sin(yaw), np.cos(yaw)]
        poses.append(pose)
        scores.append(score)

    selected = runner.select_yaw_diverse_candidates(
        poses, scores, maximum=3, yaw_bin_deg=15.0, max_per_bin=1,
    )

    # The third near-duplicate has a higher object-only score than the 45/90
    # degree alternatives, but diversity preserves scene-dependent directions.
    assert selected == [0, 3, 4]


def test_environment_constraint_search_returns_contact_preserving_trajectory():
    runner = _script("graspgenx_zmq_runner")
    pose = np.eye(4, dtype=np.float32)
    pose[2, 2] = -1.0
    pose[2, 3] = 0.20
    target = np.array([
        [x, y, z]
        for x in (-0.005, 0.0, 0.005)
        for y in (-0.003, 0.0, 0.003)
        for z in (-0.002, 0.0, 0.002)
    ], dtype=np.float32)

    class Gripper:
        collision_mesh = object()
        open_pointcloud = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        close_pointcloud = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        grasp_volume = np.array([[-0.02, -0.02, 0.14], [0.02, 0.02, 0.25]], dtype=np.float32)

    result = runner.constrain_grasps_to_environment(
        lambda **kwargs: np.ones(len(kwargs["grasp_poses"]), dtype=bool),
        np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
        target,
        pose[None],
        Gripper(),
        static_threshold=0.006,
        static_samples=10,
        static_batch_size=1,
        closed_threshold=0.004,
        closed_samples=10,
        closed_enabled=True,
        min_backoff_m=0.025,
        max_backoff_m=0.040,
        backoff_step_m=0.005,
        min_target_contact_points=12,
        pregrasp_distance_m=0.060,
        retreat_distance_m=0.100,
        path_samples=3,
        path_surface_samples=10,
    )

    assert result["mask"].tolist() == [True]
    assert result["selected_backoffs_m"][0] == pytest.approx(0.025)
    trajectory = result["trajectory_specs"][0]
    assert trajectory["target_contact_point_count"] == len(target)
    assert trajectory["grasp_transform"][2][3] == pytest.approx(0.225)
    assert trajectory["pregrasp_transform"][2][3] == pytest.approx(0.285)
    assert trajectory["retreat_transform"][2][3] == pytest.approx(0.325)


def test_candidate_schema_carries_dynamic_backoff_and_trajectory():
    runner = _script("graspgenx_zmq_runner")
    pose = np.eye(4, dtype=np.float32)
    spec = {"execution_backoff_m": 0.03, "target_contact_point_count": 25}

    candidate = runner.make_candidates(
        pose[None], [0.8], "robotiq_2f_140", None,
        execution_backoffs_m=[0.03], trajectory_specs=[spec],
    )[0]

    assert candidate["execution_backoff_m"] == pytest.approx(0.03)
    assert candidate["trajectory"] == spec
    assert candidate["metadata"]["environment_constrained_trajectory"] is True


def test_yaw_diversity_reserves_safe_obb_candidates_before_diffusion():
    runner = _script("graspgenx_zmq_runner")
    poses = []
    for yaw_deg in (0, 30, 60, 90, 120, 150):
        yaw = np.deg2rad(yaw_deg)
        pose = np.eye(4, dtype=np.float32)
        pose[:2, 0] = [np.cos(yaw), np.sin(yaw)]
        pose[:2, 1] = [-np.sin(yaw), np.cos(yaw)]
        poses.append(pose)
    scores = [0.99, 0.98, 0.97, 0.40, 0.39, 0.38]
    sources = ["diffusion", "diffusion", "diffusion", "obb_sparse", "obb_sparse", "obb_sparse"]

    selected = runner.select_yaw_diverse_candidates(
        poses, scores, maximum=4, yaw_bin_deg=15.0, max_per_bin=2,
        source_labels=sources, preferred_source="obb_sparse", min_preferred=2,
    )

    assert selected[:2] == [3, 4]
    assert sum(sources[index] == "obb_sparse" for index in selected) >= 2
