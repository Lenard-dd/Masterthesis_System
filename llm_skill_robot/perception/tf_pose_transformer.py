"""Transform approximate object poses with ROS 2 TF."""

from __future__ import annotations

import math

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

        try:
            transform = self._buffer.lookup_transform(
                self.target_frame,
                object_pose.frame,
                self._rclpy.time.Time(),
                timeout=self._rclpy.duration.Duration(seconds=timeout_sec),
            )
        except Exception:
            return None

        return self._apply_transform(object_pose, transform)

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
