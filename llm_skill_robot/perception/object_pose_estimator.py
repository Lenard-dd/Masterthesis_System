"""Approximate 3D object centroid estimation from RGB-D detections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from llm_skill_robot.perception.scene_model import ObjectPose


class ObjectPoseEstimator:
    """Estimate an approximate camera-frame centroid from aligned depth."""

    def estimate_pose_from_detection(
        self,
        detection: dict,
        depth_image,
        camera_info,
        target_frame: str = "camera_color_optical_frame",
    ) -> ObjectPose | None:
        if not isinstance(detection, dict) or depth_image is None or camera_info is None:
            return None

        intrinsics = self._camera_intrinsics(camera_info)
        if intrinsics is None:
            return None
        fx, fy, cx, cy = intrinsics

        depth_m = self._depth_to_meters(depth_image)
        if depth_m is None or depth_m.ndim != 2:
            return None

        pixel = self._select_pixel(detection, depth_m.shape)
        if pixel is None:
            return None
        u, v, mask = pixel

        z = self._median_depth(depth_m, u=u, v=v, mask=mask)
        if z is None:
            return None

        # Pinhole camera back-projection in the camera optical frame.
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return ObjectPose(frame=target_frame, x=float(x), y=float(y), z=float(z))

    @staticmethod
    def _camera_intrinsics(camera_info) -> tuple[float, float, float, float] | None:
        values = None
        if isinstance(camera_info, dict):
            values = camera_info.get("k")
            if values is None:
                values = camera_info.get("K")
        else:
            values = getattr(camera_info, "k", None)
            if values is None:
                values = getattr(camera_info, "K", None)

        if values is None or len(values) < 6:
            return None

        fx = float(values[0])
        fy = float(values[4])
        cx = float(values[2])
        cy = float(values[5])
        if fx <= 0.0 or fy <= 0.0:
            return None
        return fx, fy, cx, cy

    @staticmethod
    def _depth_to_meters(depth_image) -> np.ndarray | None:
        image = depth_image
        encoding = ""
        if hasattr(depth_image, "image"):
            image = depth_image.image
            encoding = str(getattr(depth_image, "encoding", ""))
        elif isinstance(depth_image, dict):
            image = depth_image.get("image")
            encoding = str(depth_image.get("encoding", ""))
        else:
            encoding = str(getattr(depth_image, "encoding", ""))

        if image is None:
            return None

        depth = np.asarray(image)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]

        # ROS depth encoding convention:
        # - 16UC1 is unsigned integer millimeters, so convert mm -> meters.
        # - 32FC1 is already float meters.
        if encoding.upper() == "16UC1" or depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    def _select_pixel(
        self,
        detection: dict,
        depth_shape: tuple[int, int],
    ) -> tuple[float, float, np.ndarray | None] | None:
        mask = self._load_mask(detection, depth_shape)
        if mask is not None and np.any(mask):
            rows, cols = np.nonzero(mask)
            return float(np.mean(cols)), float(np.mean(rows)), mask

        bbox = detection.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        height, width = depth_shape
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        if not (0 <= u < width and 0 <= v < height):
            return None
        return u, v, None

    @staticmethod
    def _load_mask(detection: dict, depth_shape: tuple[int, int]) -> np.ndarray | None:
        mask = detection.get("mask")
        if mask is None and detection.get("mask_path"):
            path = Path(str(detection["mask_path"]))
            if path.exists():
                try:
                    from PIL import Image

                    mask = np.asarray(Image.open(path).convert("L"))
                except Exception:
                    mask = None

        if mask is None:
            return None

        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
        if mask_array.shape != depth_shape:
            return None
        return mask_array > 0

    @staticmethod
    def _median_depth(
        depth_m: np.ndarray,
        u: float,
        v: float,
        mask: np.ndarray | None,
    ) -> float | None:
        if mask is not None and np.any(mask):
            values = depth_m[mask]
        else:
            height, width = depth_m.shape
            center_u = int(round(u))
            center_v = int(round(v))
            y1 = max(0, center_v - 2)
            y2 = min(height, center_v + 3)
            x1 = max(0, center_u - 2)
            x2 = min(width, center_u + 3)
            values = depth_m[y1:y2, x1:x2].reshape(-1)

        valid = values[np.isfinite(values) & (values > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))
