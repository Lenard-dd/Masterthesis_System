import copy

import pytest

from llm_skill_robot.grasping.parallel_jaw_symmetry import (
    canonicalize_sparse_parallel_jaw_candidates,
)


def _sparse(candidate_id, score, closing_axis, quaternion):
    return {
        "candidate_id": candidate_id,
        "score": score,
        "metadata": {"generation_branch": "obb_sparse"},
        "closing_axis": closing_axis,
        "approach_vector": [0.0, 0.0, 1.0],
        "transform": [
            [1.0, closing_axis[0], 0.0, 0.4],
            [0.0, closing_axis[1], 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "pose": {
            "frame": "base_link",
            "position": {"x": 0.4, "y": 0.1, "z": 0.3},
            "orientation": quaternion,
        },
    }


def test_opposite_sparse_grasps_canonicalize_and_deduplicate():
    aligned = _sparse(
        "aligned", 0.8, [1.0, 0.0, 0.0],
        {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    )
    opposite = _sparse(
        "opposite", 0.9, [-1.0, 0.0, 0.0],
        {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0},
    )

    primaries, fallbacks = canonicalize_sparse_parallel_jaw_candidates(
        [aligned, opposite], [1.0, 0.0, 0.0]
    )

    assert len(primaries) == 1
    assert len(fallbacks) == 1
    assert primaries[0]["candidate_id"] == "opposite"
    assert primaries[0]["closing_axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert primaries[0]["pose"]["orientation"] == pytest.approx(
        {"x": 0.0, "y": 0.0, "z": 0.0, "w": -1.0}
    )
    assert fallbacks[0]["closing_axis"] == pytest.approx([-1.0, 0.0, 0.0])


def test_symmetry_normalization_does_not_mutate_model_candidate():
    candidate = _sparse(
        "opposite", 0.9, [-1.0, 0.0, 0.0],
        {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0},
    )
    original = copy.deepcopy(candidate)

    canonicalize_sparse_parallel_jaw_candidates([candidate], [1.0, 0.0, 0.0])

    assert candidate == original


def test_diffusion_candidate_for_robotiq_is_also_canonicalized():
    candidate = _sparse(
        "diffusion", 0.95, [-1.0, 0.0, 0.0],
        {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0},
    )
    candidate["metadata"] = {
        "generation_branch": "diffusion",
        "gripper_name": "robotiq_2f_140",
    }

    primaries, fallbacks = canonicalize_sparse_parallel_jaw_candidates(
        [candidate], [1.0, 0.0, 0.0]
    )

    assert primaries[0]["closing_axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert fallbacks[0]["candidate_id"] == "diffusion__sym180"
    assert fallbacks[0]["metadata"]["parallel_jaw_symmetry_base_candidate_id"] == "diffusion"
