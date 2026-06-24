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

            output_dir = Path(str(self.config["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self._safe_stem(object_id)}.ply"
            self._write_ascii_ply(output_path, points)
        except Exception as exc:
            return self._result("FAILED", f"Failed to build object point cloud: {exc}")

        return {
            "success": True,
            "status": "SUCCESS",
            "message": f"Saved object point cloud to {output_path}.",
            "point_cloud_path": str(output_path),
            "num_points": int(points.shape[0]),
            "frame": frame or self._frame_id(camera_info),
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
        if not values or len(values) < 6:
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
        }
