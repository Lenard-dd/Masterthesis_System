def test_ros_llm_rviz_sim_demo_imports_without_ros_initialization():
    """Importing the LLM RViz sim demo must not initialize ROS or call APIs."""
    import llm_skill_robot.ros_llm_rviz_sim_demo as demo

    assert callable(demo.main)


def test_llm_generator_factory_returns_configured_generator(monkeypatch):
    import llm_skill_robot.ros_llm_rviz_sim_demo as demo

    fake_client = object()
    monkeypatch.setattr(demo, "load_llm_config", lambda: {"llm": {}})
    monkeypatch.setattr(demo, "OpenAILLMClient", lambda **_kwargs: fake_client)

    generator = demo._create_llm_plan_generator()

    assert generator is not None
    assert generator.llm_client is fake_client
    assert generator.require_observe_before_detection is True


def test_detection_execution_requires_successful_immediately_preceding_observe():
    from llm_skill_robot.core.plan import PlanStep, RobotPlan
    from llm_skill_robot.ros_llm_rviz_sim_demo import (
        _observe_completed_immediately_before,
    )

    plan = RobotPlan(task="detect", steps=[
        PlanStep(
            step_id="1",
            skill_id="move_to_named_target",
            description="observe",
            parameters={"target_name": "observe"},
        ),
        PlanStep(
            step_id="2",
            skill_id="detect_object",
            description="detect",
            parameters={"query": "cube"},
        ),
    ])

    assert _observe_completed_immediately_before(
        plan, 1, [{"success": True}]
    ) is True
    assert _observe_completed_immediately_before(
        plan, 1, [{"success": False}]
    ) is False


def test_grasp_motion_target_uses_pregrasp_grasp_and_retreat_poses():
    from llm_skill_robot.core.plan import PlanStep
    from llm_skill_robot.ros_llm_rviz_sim_demo import _grasp_motion_target_pose

    context = {
        "selected_grasp_candidate": {
            "candidate_id": "g1",
            "score": 0.9,
            "pose": {
                "frame": "base_link",
                "position": {"x": 0.4, "y": 0.1, "z": 0.3},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
        "point_cloud": {"frame": "base_link"},
    }
    config = {"plan_only_preview": {}}

    poses = {}
    for skill_id in ("move_to_pregrasp", "approach_grasp", "retreat_grasp"):
        poses[skill_id] = _grasp_motion_target_pose(
            PlanStep(
                step_id="1", skill_id=skill_id,
                description=skill_id, parameters={"object_id": "cube_1"},
            ),
            context,
            config,
            transformer=None,
        )

    assert poses["move_to_pregrasp"] == context["grasp_motion_poses"]["pregrasp"]
    assert poses["approach_grasp"] == context["grasp_motion_poses"]["grasp"]
    assert poses["retreat_grasp"] == context["grasp_motion_poses"]["retreat"]
    assert all(pose["frame"] == "base_link" for pose in poses.values())


def test_llm_review_uses_full_demo_candidate_selection_details():
    from types import SimpleNamespace

    from llm_skill_robot.ros_llm_rviz_sim_demo import (
        _apply_full_demo_candidate_selection,
    )

    poses = {
        "pregrasp": {"frame": "base_link"},
        "grasp": {"frame": "base_link"},
        "retreat": {"frame": "base_link"},
    }

    class FakeSelector:
        grasping_config = {
            "plan_only_preview": {
                "velocity_scale": 0.05,
                "acceleration_scale": 0.05,
            }
        }
        real_arm_safety = SimpleNamespace(
            limits=SimpleNamespace(
                max_velocity_scale=0.04,
                max_acceleration_scale=0.04,
            )
        )
        motion_backend = SimpleNamespace(
            planning_kwargs_by_skill={
                "move_to_pregrasp": {"planner_id": "PTP"},
                "approach_grasp": {"planner_id": "LIN"},
                "retreat_grasp": {"planner_id": "LIN"},
            }
        )

        def select_safe_grasp_candidate(self, candidates, point_cloud):
            assert candidates[0]["candidate_id"] == "g1"
            assert point_cloud["frame"] == "base_link"
            return {
                "success": True,
                "candidate": candidates[0],
                "candidate_id": "g1",
                "poses": poses,
                "grasp_plan": {"candidate_id": "g1"},
                "candidate_screening": {"selected_variant": "full_pose"},
            }

    context = {
        "grasp_candidates": [{"candidate_id": "g1", "score": 0.9}],
        "point_cloud": {"frame": "base_link"},
    }

    result = _apply_full_demo_candidate_selection(context, FakeSelector())

    assert result["success"] is True
    assert context["selected_grasp_candidate"]["candidate_id"] == "g1"
    assert context["grasp_motion_poses"] == poses
    assert context["grasp_motion_planning_kwargs"]["approach_grasp"]["planner_id"] == "LIN"
    assert context["grasp_motion_velocity_scale"] == 0.04
    assert context["grasp_motion_acceleration_scale"] == 0.04
