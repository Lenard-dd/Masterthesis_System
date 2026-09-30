"""Build a masked object point cloud from aligned D435i depth data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ObjectPointCloudBuilder:
    """Project aligned depth pixels inside a SAM3 mask and save an ASCII PLY."""

    DEFAULT_CONFIG = {
        "output_dir": "data/grasping/point_clouds",
        "depth_scale": 0.001,
        "min_depth_m": 0.05,
        "max_depth_m": 2.0,
        "max_points": 50000,
        "scene_cloud_enabled": True,
        "scene_max_points": 30000,
        "scene_target_mask_dilation_px": 1,
        "scene_support_completion_enabled": True,
        "scene_support_max_points": 3000,
        # Robust AABB bounds suppress isolated depth outliers at mask edges.
        "extent_percentile_low": 2.0,
        "extent_percentile_high": 98.0,
        "table_plane_enabled": True,
        "table_plane_margin_px": 24,
        "table_plane_min_points": 250,
        "table_plane_max_points": 12000,
        "table_plane_ransac_iterations": 120,
        "table_plane_inlier_threshold_m": 0.005,
        "table_plane_min_inlier_ratio": 0.65,
        "height_percentile": 90.0,
        "min_height_m": 0.003,
        "max_height_m": 0.25,
    }

    def __init__(self, config: dict | None = None):
        config = config or {}
        if "point_cloud" in config:
            config = config.get("point_cloud") or {}
        self.config = {**self.DEFAULT_CONFIG, **config}

    def build(
        self,
        scene_object: Any,
        depth_frame: Any,
        camera_info: Any,
        frame: str | None = None,
        point_transformer: Any | None = None,
    ) -> dict:
        metadata = self._value(scene_object, "metadata", {}) or {}
        mask_path = metadata.get("mask_path")
        object_id = str(self._value(scene_object, "object_id", "object"))
        if not mask_path or not Path(str(mask_path)).is_file():
            return self._result("NOT_AVAILABLE", "Object mask is not available.")
        if depth_frame is None or camera_info is None:
            return self._result(
                "NOT_AVAILABLE", "Aligned depth frame and camera info are required."
            )

        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            return self._result("NOT_AVAILABLE", f"Point-cloud dependencies unavailable: {exc}")

        try:
            depth, encoding = self._depth_array(depth_frame, np)
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            if depth.ndim != 2 or mask.shape != depth.shape:
                return self._result(
                    "FAILED", "SAM3 mask and aligned depth image shapes do not match."
                )
            fx, fy, cx, cy = self._intrinsics(camera_info)
            scale = 1.0 if "32FC1" in encoding.upper() else float(self.config["depth_scale"])
            depth_m = depth.astype(float) * scale
            valid = (
                mask
                & np.isfinite(depth_m)
                & (depth_m >= float(self.config["min_depth_m"]))
                & (depth_m <= float(self.config["max_depth_m"]))
            )
            rows, cols = np.nonzero(valid)
            if rows.size == 0:
                return self._result("NOT_FOUND", "Object mask contains no valid depth points.")

            max_points = int(self.config["max_points"])
            if rows.size > max_points:
                indices = np.linspace(0, rows.size - 1, max_points, dtype=int)
                rows, cols = rows[indices], cols[indices]
            z = depth_m[rows, cols]
            x = (cols - cx) * z / fx
            y = (rows - cy) * z / fy
            points = np.column_stack((x, y, z))
            centroid = points.mean(axis=0)
            planar_axes = self._planar_principal_axes(points)
            dimension_points = points
            dimension_frame = frame or self._frame_id(camera_info)
            if point_transformer is not None:
                transformed = point_transformer.transform_points(points, dimension_frame)
                if transformed is not None:
                    dimension_points = np.asarray(transformed, dtype=float)
                    dimension_frame = str(point_transformer.target_frame)
            visible_extents = self._robust_extents(dimension_points, np)
            height_estimation = self._estimate_tabletop_height(
                depth_m=depth_m,
                mask=mask,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                depth_scale=1.0,
                np_module=np,
            )

            output_dir = Path(str(self.config["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self._safe_stem(object_id)}.ply"
            self._write_ascii_ply(output_path, points)
            scene_path = None
            environment_path = None
            scene_count = 0
            environment_count = 0
            if bool(self.config.get("scene_cloud_enabled", True)):
                scene_valid = (
                    np.isfinite(depth_m)
                    & (depth_m >= float(self.config["min_depth_m"]))
                    & (depth_m <= float(self.config["max_depth_m"]))
                )
                scene_rows, scene_cols = np.nonzero(scene_valid)
                scene_rows, scene_cols = self._bounded_sample_indices(
                    scene_rows, scene_cols, int(self.config["scene_max_points"]), np
                )
                scene_points = self._backproject(
                    scene_rows, scene_cols, depth_m, fx, fy, cx, cy, np
                )
                dilated_mask = self._dilate_mask(
                    mask, int(self.config.get("scene_target_mask_dilation_px", 1)), np
                )
                env_rows, env_cols = np.nonzero(scene_valid & ~dilated_mask)
                env_rows, env_cols = self._bounded_sample_indices(
                    env_rows, env_cols, int(self.config["scene_max_points"]), np
                )
                environment_points = self._backproject(
                    env_rows, env_cols, depth_m, fx, fy, cx, cy, np
                )
                support_points = self._complete_support_surface(
                    mask=mask,
                    height_estimation=height_estimation,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    np_module=np,
                )
                if support_points.shape[0]:
                    environment_points = np.vstack((environment_points, support_points))
                    if environment_points.shape[0] > int(self.config["scene_max_points"]):
                        indices = np.linspace(
                            0,
                            environment_points.shape[0] - 1,
                            int(self.config["scene_max_points"]),
                            dtype=int,
                        )
                        environment_points = environment_points[indices]
                scene_path = output_dir / f"{self._safe_stem(object_id)}_scene.ply"
                environment_path = output_dir / f"{self._safe_stem(object_id)}_environment.ply"
                self._write_ascii_ply(scene_path, scene_points)
                self._write_ascii_ply(environment_path, environment_points)
                scene_count = int(scene_points.shape[0])
                environment_count = int(environment_points.shape[0])
        except Exception as exc:
            return self._result("FAILED", f"Failed to build object point cloud: {exc}")

        return {
            "success": True,
            "status": "SUCCESS",
            "message": f"Saved object point cloud to {output_path}.",
            "point_cloud_path": str(output_path),
            "scene_point_cloud_path": str(scene_path) if scene_path else None,
            "environment_point_cloud_path": str(environment_path) if environment_path else None,
            "scene_num_points": scene_count,
            "environment_num_points": environment_count,
            "scene_target_mask_dilation_px": int(self.config.get("scene_target_mask_dilation_px", 1)),
            "scene_support_completion_num_points": int(
                locals().get("support_points", np.empty((0, 3))).shape[0]
            ),
            "num_points": int(points.shape[0]),
            "frame": frame or self._frame_id(camera_info),
            "centroid_m": {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]),
            },
            "planar_principal_axes": planar_axes,
            # This is a visible-surface AABB, not a full watertight object
            # model.  In particular, a top-down RGB-D view cannot reliably
            # observe object thickness.  Consumers use the measured footprint
            # and retain a conservative height fallback unless a later table
            # plane / multi-view estimator supplies a true height.
            "visible_aabb_dimensions_m": visible_extents,
            "footprint_dimensions_m": {
                "x": visible_extents["x"],
                "y": visible_extents["y"],
            },
            "dimensions_frame": dimension_frame,
            "dimension_estimation": {
                "method": "masked_point_cloud_robust_aabb",
                "percentiles": [
                    float(self.config["extent_percentile_low"]),
                    float(self.config["extent_percentile_high"]),
                ],
                "visible_surface_only": True,
                "height_reliable": bool(height_estimation.get("usable", False)),
            },
            "height_estimation": height_estimation,
        }

    @staticmethod
    def _depth_array(depth_frame: Any, np_module):
        if isinstance(depth_frame, dict):
            image = depth_frame.get("image")
            encoding = str(depth_frame.get("encoding", ""))
        else:
            image = getattr(depth_frame, "image", depth_frame)
            encoding = str(getattr(depth_frame, "encoding", ""))
        return np_module.asarray(image), encoding

    @staticmethod
    def _intrinsics(camera_info: Any) -> tuple[float, float, float, float]:
        values = camera_info.get("k") if isinstance(camera_info, dict) else getattr(camera_info, "k", None)
        if values is None or len(values) < 6:
            raise ValueError("CameraInfo intrinsic matrix K is unavailable.")
        fx, fy = float(values[0]), float(values[4])
        if fx <= 0 or fy <= 0:
            raise ValueError("CameraInfo focal lengths must be positive.")
        return fx, fy, float(values[2]), float(values[5])

    @staticmethod
    def _frame_id(camera_info: Any) -> str:
        header = camera_info.get("header") if isinstance(camera_info, dict) else getattr(camera_info, "header", None)
        if isinstance(header, dict):
            return str(header.get("frame_id", "camera_color_optical_frame"))
        return str(getattr(header, "frame_id", "camera_color_optical_frame"))

    @staticmethod
    def _write_ascii_ply(path: Path, points) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(points)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x, y, z in points:
                handle.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")

    @staticmethod
    def _bounded_sample_indices(rows, cols, maximum: int, np_module):
        if maximum > 0 and rows.size > maximum:
            indices = np_module.linspace(0, rows.size - 1, maximum, dtype=int)
            return rows[indices], cols[indices]
        return rows, cols

    @staticmethod
    def _dilate_mask(mask, radius_px: int, np_module):
        radius = max(0, int(radius_px))
        if radius == 0:
            return mask.astype(bool)
        padded = np_module.pad(mask.astype(bool), radius, mode="constant")
        result = np_module.zeros_like(mask, dtype=bool)
        height, width = mask.shape
        for row_offset in range(2 * radius + 1):
            for col_offset in range(2 * radius + 1):
                result |= padded[
                    row_offset:row_offset + height,
                    col_offset:col_offset + width,
                ]
        return result

    @staticmethod
    def _planar_principal_axes(points) -> list[dict[str, float]]:
        try:
            import numpy as np

            xy = np.asarray(points, dtype=float)[:, :2]
            if xy.shape[0] < 2:
                return []
            centered = xy - xy.mean(axis=0)
            covariance = np.cov(centered, rowvar=False)
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)[::-1]
            axes = []
            for index in order[:2]:
                vector = vectors[:, index]
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-9:
                    continue
                vector = vector / norm
                axes.append(
                    {
                        "x": float(vector[0]),
                        "y": float(vector[1]),
                        "z": 0.0,
                    }
                )
            return axes
        except Exception:
            return []

    def _robust_extents(self, points, np_module) -> dict[str, float]:
        """Return axis-aligned percentile extents for a masked point cloud."""
        values = np_module.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("Point cloud must have shape (N, 3) to estimate dimensions.")
        low = float(self.config["extent_percentile_low"])
        high = float(self.config["extent_percentile_high"])
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("Extent percentiles must satisfy 0 <= low < high <= 100.")
        lower, upper = np_module.percentile(values, [low, high], axis=0)
        extents = np_module.maximum(upper - lower, 0.0)
        return {
            "x": float(extents[0]),
            "y": float(extents[1]),
            "z": float(extents[2]),
        }

    def _estimate_tabletop_height(
        self,
        *,
        depth_m,
        mask,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: float,
        np_module,
    ) -> dict[str, Any]:
        """Estimate height above a locally fitted tabletop plane.

        A SAM3 mask sees only an object's visible surface, so its depth range
        alone is not its thickness.  Background points in a padded rectangle
        around the mask provide a local table plane; RANSAC rejects clutter
        and the object height is the high percentile of object-to-plane
        distances.  This is deliberately conservative and returns a quality
        record instead of fabricating a height when the table is not visible.
        """
        unavailable = {
            "usable": False,
            "height_m": None,
            "confidence": 0.0,
            "plane_inlier_ratio": 0.0,
            "height_std_m": None,
            "method": "tabletop_plane_ransac",
        }
        if not bool(self.config.get("table_plane_enabled", True)):
            return {**unavailable, "reason": "disabled"}
        rows, cols = np_module.nonzero(mask)
        if rows.size == 0:
            return {**unavailable, "reason": "empty_mask"}
        height, width = mask.shape
        margin = max(1, int(self.config["table_plane_margin_px"]))
        row_min, row_max = max(0, int(rows.min()) - margin), min(height, int(rows.max()) + margin + 1)
        col_min, col_max = max(0, int(cols.min()) - margin), min(width, int(cols.max()) + margin + 1)
        neighborhood = np_module.zeros_like(mask, dtype=bool)
        neighborhood[row_min:row_max, col_min:col_max] = True
        valid_depth = np_module.isfinite(depth_m) & (depth_m >= float(self.config["min_depth_m"])) & (
            depth_m <= float(self.config["max_depth_m"])
        )
        table_rows, table_cols = np_module.nonzero(neighborhood & ~mask & valid_depth)
        min_points = int(self.config["table_plane_min_points"])
        if table_rows.size < min_points:
            return {**unavailable, "reason": "insufficient_table_points", "table_point_count": int(table_rows.size)}
        max_points = int(self.config["table_plane_max_points"])
        if table_rows.size > max_points:
            indices = np_module.linspace(0, table_rows.size - 1, max_points, dtype=int)
            table_rows, table_cols = table_rows[indices], table_cols[indices]
        table_points = self._backproject(table_rows, table_cols, depth_m, fx, fy, cx, cy, np_module)
        plane = self._fit_plane_ransac(table_points, np_module)
        if plane is None:
            return {**unavailable, "reason": "table_plane_fit_failed", "table_point_count": int(table_points.shape[0])}
        normal, offset, inliers = plane
        inlier_ratio = float(inliers.mean())
        object_rows, object_cols = np_module.nonzero(mask & valid_depth)
        object_points = self._backproject(object_rows, object_cols, depth_m, fx, fy, cx, cy, np_module)
        distances = np_module.abs(object_points @ normal + offset)
        if distances.size == 0:
            return {**unavailable, "reason": "no_valid_object_points"}
        height_m = float(np_module.percentile(distances, float(self.config["height_percentile"])))
        upper = distances[distances >= np_module.percentile(distances, 75.0)]
        height_std_m = float(np_module.std(upper)) if upper.size else None
        min_height, max_height = float(self.config["min_height_m"]), float(self.config["max_height_m"])
        usable = (
            inlier_ratio >= float(self.config["table_plane_min_inlier_ratio"])
            and min_height <= height_m <= max_height
        )
        confidence = max(0.0, min(1.0, inlier_ratio * min(1.0, table_points.shape[0] / 1000.0)))
        return {
            "usable": bool(usable),
            "height_m": height_m,
            "confidence": confidence,
            "plane_inlier_ratio": inlier_ratio,
            "height_std_m": height_std_m,
            "method": "tabletop_plane_ransac",
            "table_point_count": int(table_points.shape[0]),
            "object_point_count": int(object_points.shape[0]),
            "reason": None if usable else "height_or_plane_quality_out_of_range",
            "plane_normal": [float(value) for value in normal],
            "plane_offset": float(offset),
        }

    def _complete_support_surface(
        self,
        *,
        mask,
        height_estimation: dict[str, Any],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        np_module,
    ):
        """Fill the support plane hidden by the target mask."""
        empty = np_module.empty((0, 3), dtype=float)
        if not bool(self.config.get("scene_support_completion_enabled", True)):
            return empty
        if not bool(height_estimation.get("usable", False)):
            return empty
        normal = np_module.asarray(height_estimation.get("plane_normal", []), dtype=float)
        if normal.shape != (3,):
            return empty
        offset = float(height_estimation.get("plane_offset", float("nan")))
        if not np_module.isfinite(offset):
            return empty
        rows, cols = np_module.nonzero(mask)
        maximum = max(0, int(self.config.get("scene_support_max_points", 3000)))
        rows, cols = self._bounded_sample_indices(rows, cols, maximum, np_module)
        if rows.size == 0:
            return empty
        rays = np_module.column_stack(
            ((cols - cx) / fx, (rows - cy) / fy, np_module.ones(rows.size))
        )
        denominators = rays @ normal
        valid = np_module.abs(denominators) > 1e-9
        depths = np_module.full(rows.size, np_module.nan, dtype=float)
        depths[valid] = -offset / denominators[valid]
        valid &= np_module.isfinite(depths)
        valid &= depths >= float(self.config["min_depth_m"])
        valid &= depths <= float(self.config["max_depth_m"])
        return rays[valid] * depths[valid, None]

    @staticmethod
    def _backproject(rows, cols, depth_m, fx, fy, cx, cy, np_module):
        z = depth_m[rows, cols]
        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        return np_module.column_stack((x, y, z))

    def _fit_plane_ransac(self, points, np_module):
        if points.shape[0] < 3:
            return None
        rng = np_module.random.default_rng(0)
        threshold = float(self.config["table_plane_inlier_threshold_m"])
        best_inliers = None
        for _ in range(int(self.config["table_plane_ransac_iterations"])):
            sample = points[rng.choice(points.shape[0], size=3, replace=False)]
            normal = np_module.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np_module.linalg.norm(normal))
            if norm <= 1e-9:
                continue
            normal = normal / norm
            offset = -float(normal @ sample[0])
            inliers = np_module.abs(points @ normal + offset) <= threshold
            if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
                best_inliers = inliers
        if best_inliers is None or int(best_inliers.sum()) < 3:
            return None
        inlier_points = points[best_inliers]
        center = inlier_points.mean(axis=0)
        _singular_values, _vectors = None, None
        try:
            _u, _singular_values, vectors = np_module.linalg.svd(inlier_points - center, full_matrices=False)
            normal = vectors[-1]
        except Exception:
            return None
        norm = float(np_module.linalg.norm(normal))
        if norm <= 1e-9:
            return None
        normal = normal / norm
        offset = -float(normal @ center)
        refined_inliers = np_module.abs(points @ normal + offset) <= threshold
        return normal, offset, refined_inliers

    @staticmethod
    def _value(value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)

    @staticmethod
    def _safe_stem(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value) or "object"

    @staticmethod
    def _result(status: str, message: str) -> dict:
        return {
            "success": False,
            "status": status,
            "message": message,
            "point_cloud_path": None,
            "num_points": 0,
            "frame": None,
            "centroid_m": None,
        }
