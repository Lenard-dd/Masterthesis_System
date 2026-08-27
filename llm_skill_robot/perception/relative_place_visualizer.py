"""RViz MarkerArray preview for plan-only relative placement poses."""

from __future__ import annotations

from typing import Any


class RelativePlaceVisualizer:
    """Publish scene-relative place-pose evidence; never issue robot commands."""

    def __init__(self, node=None, topic: str = "/relative_place_preview") -> None:
        self.node = node
        self.topic = topic
        self._publisher = None

    def publish(self, place_pose_plan: dict[str, Any]) -> dict[str, Any]:
        if self.node is None:
            return {
                "success": False,
                "message": "ROS node is required to publish a relative place preview.",
                "topic": self.topic,
            }
        try:
            from geometry_msgs.msg import Point
            from visualization_msgs.msg import Marker, MarkerArray
        except Exception as exc:
            return {
                "success": False,
                "message": f"RViz Marker messages are unavailable: {exc}",
                "topic": self.topic,
            }

        target = _pose_dict(place_pose_plan.get("target_pose"))
        source = _pose_dict(place_pose_plan.get("source_pose"))
        references = []
        for raw_pose in place_pose_plan.get("reference_poses", []):
            pose = _pose_dict(raw_pose)
            if pose is not None:
                references.append(pose)
        if target is None or source is None:
            return {
                "success": False,
                "message": "Relative place plan has no source or target pose to preview.",
                "topic": self.topic,
            }
        frame = str(target["frame"])
        if source["frame"] != frame or any(reference["frame"] != frame for reference in references):
            return {
                "success": False,
                "message": "Relative place preview requires all poses in one frame.",
                "topic": self.topic,
            }
        try:
            if self._publisher is None:
                # Match RViz's explicit MarkerArray subscription QoS.  RViz is
                # already running in the GUI before this plan-only skill runs,
                # so durability is not needed and a volatile publisher avoids
                # QoS incompatibilities across RViz versions.
                from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

                self._publisher = self.node.create_publisher(
                    MarkerArray,
                    self.topic,
                    QoSProfile(
                        depth=5,
                        durability=DurabilityPolicy.VOLATILE,
                        reliability=ReliabilityPolicy.RELIABLE,
                    ),
                )
            markers = MarkerArray()
            markers.markers.append(_delete_all_marker(Marker))
            markers.markers.append(_shape_marker(
                Marker, frame, 1, source, Marker.SPHERE, (0.15, 0.45, 1.0, 0.85), 0.045,
            ))
            for index, reference in enumerate(references, start=2):
                markers.markers.append(_shape_marker(
                    Marker, frame, index, reference, Marker.CUBE, (1.0, 0.75, 0.05, 0.85), 0.045,
                ))
            markers.markers.append(_shape_marker(
                Marker, frame, 20, target, Marker.SPHERE, (0.1, 0.95, 0.25, 1.0), 0.05,
            ))
            markers.markers.append(_target_label_marker(
                Marker,
                frame,
                21,
                target,
                "TARGET",
            ))
            markers.markers.append(_relation_lines_marker(Marker, Point, frame, source, references, target))
            self._publisher.publish(markers)
        except Exception as exc:
            return {
                "success": False,
                "message": f"Failed to publish relative place preview: {exc}",
                "topic": self.topic,
            }
        subscription_count = self._publisher.get_subscription_count()
        return {
            "success": True,
            # DDS endpoint discovery is asynchronous.  A zero count here is
            # not proof that RViz is unavailable, so do not report a false
            # preview failure immediately after the publisher is created.
            "message": "Relative place preview published. No robot motion has been executed.",
            "topic": self.topic,
            "subscriber_count": subscription_count,
            "marker_count": len(markers.markers) - 1,
        }


def _pose_dict(value: Any) -> dict[str, float | str] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {
            "frame": str(value["frame"]),
            "x": float(value["x"]),
            "y": float(value["y"]),
            "z": float(value["z"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _header(marker, frame: str) -> None:
    marker.header.frame_id = frame
    marker.ns = "relative_place_preview"
    marker.action = marker.ADD


def _delete_all_marker(Marker):
    marker = Marker()
    marker.action = Marker.DELETEALL
    return marker


def _shape_marker(Marker, frame, marker_id, pose, marker_type, color, size):
    marker = Marker()
    _header(marker, frame)
    marker.id = marker_id
    marker.type = marker_type
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = pose["x"]
    marker.pose.position.y = pose["y"]
    marker.pose.position.z = pose["z"]
    marker.scale.x = marker.scale.y = marker.scale.z = size
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    return marker


def _target_label_marker(Marker, frame, marker_id, target, text):
    marker = Marker()
    _header(marker, frame)
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = target["x"]
    marker.pose.position.y = target["y"]
    marker.pose.position.z = target["z"] + 0.23
    marker.scale.z = 0.045
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.1, 1.0, 0.25, 1.0)
    marker.text = text
    return marker


def _relation_lines_marker(Marker, Point, frame, source, references, target):
    marker = Marker()
    _header(marker, frame)
    marker.id = 30
    marker.type = Marker.LINE_LIST
    marker.scale.x = 0.006
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.9, 0.9, 0.9, 0.9)
    for pose in [source, *references]:
        marker.points.extend([
            Point(x=pose["x"], y=pose["y"], z=pose["z"]),
            Point(x=target["x"], y=target["y"], z=target["z"]),
        ])
    return marker
