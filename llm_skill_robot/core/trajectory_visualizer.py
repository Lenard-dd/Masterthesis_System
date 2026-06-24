"""RViz visual preview publishing for backend-generated trajectories."""

from typing import Any, Dict


class TrajectoryVisualizer:
    """Publish cached backend-generated trajectories for RViz inspection only."""

    def __init__(self, node=None, topic: str = "/display_planned_path"):
        self.node = node
        self.topic = topic
        self._publisher = None

    def publish_preview(self, plan_entry_or_summary) -> Dict[str, Any]:
        """Publish a DisplayTrajectory preview without executing motion."""
        display_trajectory = getattr(plan_entry_or_summary, "display_trajectory", None)
        robot_trajectory = getattr(plan_entry_or_summary, "robot_trajectory", None)
        trajectory_start = getattr(plan_entry_or_summary, "trajectory_start", None)
        if display_trajectory is None and robot_trajectory is None:
            return {
                "success": False,
                "message": "No trajectory available for RViz visual preview.",
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

            preview_message = display_trajectory
            if preview_message is None:
                preview_message = DisplayTrajectory()
                if trajectory_start is not None:
                    preview_message.trajectory_start = trajectory_start
                preview_message.trajectory.append(robot_trajectory)
            self._publisher.publish(preview_message)
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to publish RViz visual preview: {e}",
            }

        message = "Visual preview published. No motion has been executed."
        print(message)
        return {
            "success": True,
            "message": message,
            "topic": self.topic,
        }

    def publish_sequence(self, plan_entries) -> Dict[str, Any]:
        """Publish multiple cached plan-only segments as one RViz preview."""
        entries = list(plan_entries or [])
        trajectories = [
            getattr(entry, "robot_trajectory", None)
            for entry in entries
            if getattr(entry, "robot_trajectory", None) is not None
        ]
        if not trajectories:
            return {
                "success": False,
                "message": "No trajectories available for RViz sequence preview.",
            }
        if self.node is None:
            return {
                "success": False,
                "message": "ROS node is required to publish RViz visual preview.",
            }
        try:
            from moveit_msgs.msg import DisplayTrajectory

            if self._publisher is None:
                self._publisher = self.node.create_publisher(DisplayTrajectory, self.topic, 10)
            message = DisplayTrajectory()
            first_start = getattr(entries[0], "trajectory_start", None)
            if first_start is not None:
                message.trajectory_start = first_start
            message.trajectory.extend(trajectories)
            self._publisher.publish(message)
        except Exception as exc:
            return {
                "success": False,
                "message": f"Failed to publish RViz sequence preview: {exc}",
            }
        result_message = "Visual sequence preview published. No motion has been executed."
        print(result_message)
        return {"success": True, "message": result_message, "topic": self.topic}

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
