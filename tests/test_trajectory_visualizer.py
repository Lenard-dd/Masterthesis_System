import sys
import types

from llm_skill_robot.core.schemas import MotionPlanCacheEntry, MotionPlanSummary
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer


class FakeRobotTrajectory:
    pass


class FakeDisplayTrajectory:
    def __init__(self):
        self.trajectory = []
        self.trajectory_start = None


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, message):
        self.published.append(message)


class FakeNode:
    def __init__(self):
        self.publisher = FakePublisher()
        self.publisher_args = None

    def create_publisher(self, *args):
        self.publisher_args = args
        return self.publisher


def _install_fake_moveit_msgs(monkeypatch):
    moveit_msgs = types.ModuleType("moveit_msgs")
    moveit_msgs_msg = types.ModuleType("moveit_msgs.msg")
    moveit_msgs_msg.DisplayTrajectory = FakeDisplayTrajectory
    monkeypatch.setitem(sys.modules, "moveit_msgs", moveit_msgs)
    monkeypatch.setitem(sys.modules, "moveit_msgs.msg", moveit_msgs_msg)


def test_trajectory_visualizer_imports():
    visualizer = TrajectoryVisualizer()

    assert visualizer.topic == "/display_planned_path"


def test_publish_preview_returns_failure_without_raw_trajectory():
    summary = MotionPlanSummary(
        plan_id="plan-no-raw",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Target command cached.",
    )
    entry = MotionPlanCacheEntry(
        plan_id="plan-no-raw",
        skill_id="move_to_named_target",
        summary=summary,
        created_at="2026-05-07T12:00:00Z",
        raw_plan_available=False,
    )
    visualizer = TrajectoryVisualizer()

    result = visualizer.publish_preview(entry)

    assert result["success"] is False
    assert result["message"] == "No trajectory available for RViz visual preview."


def test_publish_preview_only_publishes_display_trajectory_to_default_topic(monkeypatch, capsys):
    _install_fake_moveit_msgs(monkeypatch)
    summary = MotionPlanSummary(
        plan_id="plan-preview",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
    )
    robot_trajectory = FakeRobotTrajectory()
    entry = MotionPlanCacheEntry(
        plan_id="plan-preview",
        skill_id="move_to_named_target",
        summary=summary,
        created_at="2026-05-07T12:00:00Z",
        robot_trajectory=robot_trajectory,
    )
    node = FakeNode()
    visualizer = TrajectoryVisualizer(node=node)

    result = visualizer.publish_preview(entry)

    assert result["success"] is True
    assert result["message"] == "Visual preview published. No motion has been executed."
    assert result["topic"] == "/display_planned_path"
    assert node.publisher_args == (FakeDisplayTrajectory, "/display_planned_path", 10)
    assert len(node.publisher.published) == 1
    assert node.publisher.published[0].trajectory == [robot_trajectory]
    assert "Visual preview published. No motion has been executed." in capsys.readouterr().out
