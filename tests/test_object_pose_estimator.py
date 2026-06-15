from types import SimpleNamespace

import numpy as np

from llm_skill_robot.perception.object_pose_estimator import ObjectPoseEstimator


def _camera_info():
    return SimpleNamespace(k=[100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0])


def test_bbox_center_with_constant_depth_gives_expected_pose():
    estimator = ObjectPoseEstimator()
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    detection = {"bbox_xyxy": [2, 2, 4, 4]}

    pose = estimator.estimate_pose_from_detection(detection, depth, _camera_info())

    assert pose is not None
    assert pose.frame == "camera_color_optical_frame"
    assert pose.x == 0.02
    assert pose.y == 0.02
    assert pose.z == 2.0


def test_invalid_depth_returns_none():
    estimator = ObjectPoseEstimator()
    depth = np.zeros((5, 5), dtype=np.float32)
    detection = {"bbox_xyxy": [2, 2, 4, 4]}

    pose = estimator.estimate_pose_from_detection(detection, depth, _camera_info())

    assert pose is None


def test_16uc1_depth_millimeters_are_converted_to_meters():
    estimator = ObjectPoseEstimator()
    depth = np.full((5, 5), 1500, dtype=np.uint16)
    detection = {"bbox_xyxy": [0, 0, 4, 4]}

    pose = estimator.estimate_pose_from_detection(
        detection,
        {"image": depth, "encoding": "16UC1"},
        _camera_info(),
    )

    assert pose is not None
    assert pose.x == 0.0
    assert pose.y == 0.0
    assert pose.z == 1.5


def test_missing_camera_info_returns_none():
    estimator = ObjectPoseEstimator()
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    detection = {"bbox_xyxy": [2, 2, 4, 4]}

    pose = estimator.estimate_pose_from_detection(detection, depth, None)

    assert pose is None


def test_mask_centroid_and_median_mask_depth_are_preferred():
    estimator = ObjectPoseEstimator()
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[1, 1] = 1.0
    depth[1, 2] = 3.0
    depth[2, 1] = 5.0
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    detection = {"bbox_xyxy": [4, 4, 4, 4], "mask": mask}

    pose = estimator.estimate_pose_from_detection(detection, depth, _camera_info())

    assert pose is not None
    assert pose.x == -0.015
    assert pose.y == -0.015
    assert pose.z == 3.0
