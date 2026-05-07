"""RViz visual preview publishing for backend-generated trajectories."""

from typing import Any, Dict


class TrajectoryVisualizer:
    """Publish cached backend-generated trajectories for RViz inspection."""

    def __init__(self, node=None, topic: str = "/display_planned_path"):
        self.node = node
        self.topic = topic
        self._publisher = None

    def publish_preview(self, plan_entry_or_summary) -> Dict[str, Any]:
        """Publish a DisplayTrajectory if a raw trajectory is available."""
        raw_trajectory = getattr(plan_entry_or_summary, "raw_trajectory", None)
        trajectory_start = getattr(plan_entry_or_summary, "trajectory_start", None)
        if raw_trajectory is None:
            return {
                "success": False,
                "message": "No raw trajectory available for RViz visual preview.",
            }

        if self.node is None:
            return {
                "success": False,
                "message": "ROS node is required to publish RViz visual preview.",
            }

        try:
            from moveit_msgs.msg import DisplayTrajectory
        except Exception as e:
            return {
                "success": False,
                "message": f"moveit_msgs DisplayTrajectory is not available: {e}",
            }

        try:
            if self._publisher is None:
                self._publisher = self.node.create_publisher(
                    DisplayTrajectory,
                    self.topic,
                    10,
                )

            display_trajectory = DisplayTrajectory()
            if trajectory_start is not None:
                display_trajectory.trajectory_start = trajectory_start
            display_trajectory.trajectory.append(raw_trajectory)
            self._publisher.publish(display_trajectory)
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to publish RViz visual preview: {e}",
            }

        return {
            "success": True,
            "message": f"Publishing trajectory preview to RViz topic: {self.topic}",
            "topic": self.topic,
        }

    def clear_preview(self) -> Dict[str, Any]:
        """Clear the RViz DisplayTrajectory preview when possible."""
        if self.node is None:
            return {
                "success": False,
                "message": "ROS node is required to clear RViz visual preview.",
            }

        try:
            from moveit_msgs.msg import DisplayTrajectory
        except Exception as e:
            return {
                "success": False,
                "message": f"moveit_msgs DisplayTrajectory is not available: {e}",
            }

        try:
            if self._publisher is None:
                self._publisher = self.node.create_publisher(
                    DisplayTrajectory,
                    self.topic,
                    10,
                )
            self._publisher.publish(DisplayTrajectory())
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to clear RViz visual preview: {e}",
            }

        return {
            "success": True,
            "message": f"Cleared trajectory preview on RViz topic: {self.topic}",
            "topic": self.topic,
        }
