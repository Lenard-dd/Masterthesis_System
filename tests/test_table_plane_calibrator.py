import numpy as np
import pytest

from llm_skill_robot.calibration.table_plane_calibrator import TablePlaneCalibrator


def _camera_info():
    return {"k": [100.0, 0.0, 20.0, 0.0, 100.0, 15.0, 0.0, 0.0, 1.0]}


def test_calibrates_horizontal_table_plane_in_base_link():
    depth = np.full((30, 40), 0.70, dtype=np.float32)
    calibrator = TablePlaneCalibrator({
        "u_range_normalized": [0.0, 1.0],
        "v_range_normalized": [0.0, 1.0],
        "minimum_inliers": 50,
        "max_points": 2000,
    })

    result = calibrator.calibrate(
        {"image": depth, "encoding": "32FC1"}, _camera_info(), np.eye(4)
    )

    assert result["success"] is True
    assert result["table_z_m"] == pytest.approx(0.70)
    assert result["collision_table_z_m"] == pytest.approx(0.708)
    assert result["plane"]["normal"]["z"] > 0.99


def test_rejects_an_empty_or_invalid_roi():
    depth = np.zeros((30, 40), dtype=np.uint16)
    result = TablePlaneCalibrator({"minimum_inliers": 1}).calibrate(
        {"image": depth, "encoding": "16UC1"}, _camera_info(), np.eye(4)
    )

    assert result["success"] is False
