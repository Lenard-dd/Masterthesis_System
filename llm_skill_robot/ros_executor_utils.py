"""Helpers for code which may receive either owned or externally-spun ROS nodes."""

from __future__ import annotations


def node_has_executor(node) -> bool:
    """Return whether callbacks for *node* are already managed by an executor.

    Calling ``rclpy.spin_once(node)`` for such a node is unsafe: rclpy's global
    executor will reassign the node away from its existing executor.  GUI
    runtime code therefore waits for the owning executor to deliver callbacks
    instead of spinning the shared node itself.
    """

    if node is None:
        return False
    try:
        return getattr(node, "executor", None) is not None
    except Exception:
        return False
