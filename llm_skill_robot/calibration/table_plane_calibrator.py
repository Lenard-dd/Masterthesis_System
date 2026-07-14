"""Fit a tabletop plane from an aligned depth image in ``base_link``."""

from __future__ import annotations

from typing import Any


class TablePlaneCalibrator:
    """Back-project a configured depth ROI and robustly fit its table plane."""

    DEFAULT_CONFIG = {
        "u_range_normalized": [0.15, 0.85],
        "v_range_normalized": [0.55, 0.95],
        "min_depth_m": 0.10,
        "max_depth_m": 2.00,
        "depth_scale": 0.001,
        "max_points": 12000,
        "ransac_iterations": 300,
        "inlier_threshold_m": 0.006,
        "max_tilt_deg": 12.0,
        "minimum_inliers": 500,
        "collision_margin_m": 0.008,
        "random_seed": 7,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        if "tabletop_calibration" in config:
            config = config.get("tabletop_calibration") or {}
        self.config = {**self.DEFAULT_CONFIG, **config}

    def calibrate(
        self,
        depth_frame: Any,
        camera_info: Any,
        camera_to_base: Any,
    ) -> dict[str, Any]:
        """Return a robust base-frame table plane and a conservative collision height."""
        try:
            import numpy as np
        except ImportError as exc:
            return self._failed(f"numpy is required for table calibration: {exc}")
        try:
            depth, encoding = self._depth_array(depth_frame, np)
            if depth.ndim != 2:
                return self._failed("Aligned depth image must be two-dimensional.")
            fx, fy, cx, cy = self._intrinsics(camera_info)
            transform = self._transform_matrix(camera_to_base, np)
            points = self._backproject_roi(depth, encoding, fx, fy, cx, cy, transform, np)
        except (TypeError, ValueError, KeyError) as exc:
            return self._failed(f"Could not prepare table points: {exc}")
        if len(points) < 3:
            return self._failed("Configured table ROI contains fewer than three valid depth points.")

        fit = self._fit_plane(points, np)
        if fit is None:
            return self._failed("Could not fit a near-horizontal table plane from the configured ROI.")
        normal, d, inliers = fit
        inlier_points = points[inliers]
        if len(inlier_points) < int(self.config["minimum_inliers"]):
            return self._failed(
                f"Only {len(inlier_points)} table inliers were found; "
                f"need at least {self.config['minimum_inliers']}."
            )
        table_z = float(-d / normal[2])
        residuals = abs(inlier_points @ normal + d)
        margin = float(self.config["collision_margin_m"])
        return {
            "success": True,
            "status": "SUCCESS",
            "message": "Table plane calibrated in base_link.",
            "frame": "base_link",
            "plane": {
                "normal": {"x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])},
                "d": float(d),
            },
            "table_z_m": table_z,
            "collision_table_z_m": table_z + margin,
            "inlier_count": int(len(inlier_points)),
            "sample_count": int(len(points)),
            "residual_median_m": float(np.median(residuals)),
            "residual_max_m": float(np.max(residuals)),
            "suggested_config": {
                "tabletop_safety": {
                    "enabled": True,
                    "frame": "base_link",
                    "table_z_m": round(table_z, 5),
                    "collision_table_z_m": round(table_z + margin, 5),
                    "min_open_finger_clearance_m": 0.02,
                    "close_sweep_compensation_m": 0.01,
                }
            },
        }

    def _backproject_roi(self, depth, encoding, fx, fy, cx, cy, transform, np):
        height, width = depth.shape
        u0, u1 = self._pixel_range(self.config["u_range_normalized"], width)
        v0, v1 = self._pixel_range(self.config["v_range_normalized"], height)
        rows, cols = np.mgrid[v0:v1, u0:u1]
        raw_depth = depth[v0:v1, u0:u1].astype(float)
        scale = 1.0 if "32FC1" in str(encoding).upper() else float(self.config["depth_scale"])
        z = raw_depth * scale
        valid = np.isfinite(z) & (z >= float(self.config["min_depth_m"])) & (z <= float(self.config["max_depth_m"]))
        z, rows, cols = z[valid], rows[valid], cols[valid]
        if z.size == 0:
            return np.empty((0, 3))
        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        points = np.column_stack((x, y, z))
        max_points = int(self.config["max_points"])
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = points[indices]
        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (transform @ homogeneous.T).T[:, :3]

    def _fit_plane(self, points, np):
        rng = np.random.default_rng(int(self.config["random_seed"]))
        threshold = float(self.config["inlier_threshold_m"])
        max_tilt_cos = float(np.cos(np.deg2rad(float(self.config["max_tilt_deg"]))))
        best_inliers = None
        best_count = 0
        for _ in range(int(self.config["ransac_iterations"])):
            sample = points[rng.choice(len(points), size=3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm <= 1e-12:
                continue
            normal /= norm
            if normal[2] < 0:
                normal = -normal
            if normal[2] < max_tilt_cos:
                continue
            d = -float(normal @ sample[0])
            inliers = abs(points @ normal + d) <= threshold
            if int(inliers.sum()) > best_count:
                best_inliers, best_count = inliers, int(inliers.sum())
        if best_inliers is None:
            return None
        inlier_points = points[best_inliers]
        centroid = inlier_points.mean(axis=0)
        _, _, vectors = np.linalg.svd(inlier_points - centroid)
        normal = vectors[-1]
        if normal[2] < 0:
            normal = -normal
        if normal[2] < max_tilt_cos:
            return None
        d = -float(normal @ centroid)
        inliers = abs(points @ normal + d) <= threshold
        return normal, d, inliers

    @staticmethod
    def _depth_array(depth_frame, np):
        if isinstance(depth_frame, dict):
            return np.asarray(depth_frame.get("image")), str(depth_frame.get("encoding", ""))
        return np.asarray(getattr(depth_frame, "image", depth_frame)), str(getattr(depth_frame, "encoding", ""))

    @staticmethod
    def _intrinsics(camera_info):
        values = camera_info.get("k") if isinstance(camera_info, dict) else getattr(camera_info, "k", None)
        if values is None:
            values = camera_info.get("K") if isinstance(camera_info, dict) else getattr(camera_info, "K", None)
        if values is None or len(values) < 6:
            raise ValueError("camera_info must provide a 3x3 K matrix")
        fx, fy, cx, cy = float(values[0]), float(values[4]), float(values[2]), float(values[5])
        if fx <= 0 or fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        return fx, fy, cx, cy

    @staticmethod
    def _transform_matrix(transform, np):
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("camera_to_base must be a 4x4 transform matrix")
        return matrix

    @staticmethod
    def _pixel_range(normalized_range, limit):
        if not isinstance(normalized_range, (list, tuple)) or len(normalized_range) != 2:
            raise ValueError("ROI ranges must contain two normalized values")
        first, second = (float(value) for value in normalized_range)
        if not 0.0 <= first < second <= 1.0:
            raise ValueError("ROI ranges must satisfy 0 <= min < max <= 1")
        return int(first * limit), max(int(second * limit), int(first * limit) + 1)

    @staticmethod
    def _failed(message):
        return {"success": False, "status": "FAILED", "message": message}
