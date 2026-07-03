"""Transform approximate object poses with ROS 2 TF."""

from __future__ import annotations

import math
import time

from llm_skill_robot.perception.scene_model import ObjectPose


class TFPoseTransformer:
    """Transform ObjectPose positions into a target TF frame."""

    def __init__(self, node, target_frame: str = "base_link"):
        self.node = node
        self.target_frame = target_frame
        self.available = False
        self._buffer = None
        self._listener = None
        self._rclpy = None

        try:
            import rclpy
            from tf2_ros import Buffer, TransformListener
        except ImportError:
            return

        self._rclpy = rclpy
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node)
        self.available = True

    def transform_pose(
        self,
        object_pose: ObjectPose,
        timeout_sec: float = 1.0,
    ) -> ObjectPose | None:
        if not self.available or self._buffer is None or self._rclpy is None:
            return None
        if object_pose.frame == self.target_frame:
            return object_pose

        transform = self._lookup_transform(object_pose.frame, timeout_sec)
        if transform is None:
            return None

        return self._apply_transform(object_pose, transform)

    def transform_pose_dict(
        self,
        pose: dict,
        timeout_sec: float = 1.0,
    ) -> dict | None:
        """Transform a full pose dictionary into the target TF frame."""
        if not self.available or self._buffer is None or self._rclpy is None:
            return None
        if pose.get("frame") == self.target_frame:
            return pose

        transform = self._lookup_transform(pose["frame"], timeout_sec)
        if transform is None:
            return None

        return self._apply_transform_to_pose_dict(pose, transform)

    def _lookup_transform(self, source_frame: str, timeout_sec: float):
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while True:
            try:
                return self._buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    self._rclpy.time.Time(),
                )
            except Exception as exc:
                if time.monotonic() >= deadline:
                    break
                try:
                    self._rclpy.spin_once(self.node, timeout_sec=0.05)
                except Exception:
                    time.sleep(0.05)
        return None

    def _apply_transform(self, object_pose: ObjectPose, transform) -> ObjectPose:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        x, y, z = self._rotate_vector(
            (object_pose.x, object_pose.y, object_pose.z),
            (rotation.x, rotation.y, rotation.z, rotation.w),
        )

        # R3/R4 only estimate an approximate object centroid, not a full 6D
        # grasp pose. Keep orientation identity after transforming position.
        return ObjectPose(
            frame=self.target_frame,
            x=float(x + translation.x),
            y=float(y + translation.y),
            z=float(z + translation.z),
        )

    def _apply_transform_to_pose_dict(self, pose: dict, transform) -> dict:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        position = pose["position"]
        x, y, z = self._rotate_vector(
            (
                float(position["x"]),
                float(position["y"]),
                float(position["z"]),
            ),
            (rotation.x, rotation.y, rotation.z, rotation.w),
        )
        orientation = pose.get("orientation", {})
        qx, qy, qz, qw = self._multiply_quaternions(
            (rotation.x, rotation.y, rotation.z, rotation.w),
            (
                float(orientation.get("x", 0.0)),
                float(orientation.get("y", 0.0)),
                float(orientation.get("z", 0.0)),
                float(orientation.get("w", 1.0)),
            ),
        )
        return {
            "frame": self.target_frame,
            "position": {
                "x": float(x + translation.x),
                "y": float(y + translation.y),
                "z": float(z + translation.z),
            },
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        }

    @staticmethod
    def _rotate_vector(
        vector: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        qx, qy, qz, qw = quaternion
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            return vector

        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        vx, vy, vz = vector

        # Quaternion vector rotation: v' = v + 2*w*(q_vec x v) +
        # 2*(q_vec x (q_vec x v)).
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        return (
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        )

    @staticmethod
    def _multiply_quaternions(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        ax, ay, az, aw = left
        bx, by, bz, bw = right
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        w = aw * bw - ax * bx - ay * by - az * bz
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return 0.0, 0.0, 0.0, 1.0
        return x / norm, y / norm, z / norm, w / norm
