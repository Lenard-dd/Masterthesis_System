import json


def test_ros_rviz_sim_execution_demo_imports_without_ros_initialization():
    """Importing the demo module must not initialize rclpy or ask for input."""
    import llm_skill_robot.ros_rviz_sim_execution_demo as demo

    assert callable(demo.main)


def test_ros_rviz_sim_demo_hardcoded_plan_is_stable():
    import llm_skill_robot.ros_rviz_sim_execution_demo as demo

    plan = demo._create_plan()

    assert demo.RVIZ_SIM_MODE_WARNING == (
        "RVIZ_SIM MODE ONLY. This demo is not intended for real robot execution."
    )
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.skill_id == "move_to_named_target"
    assert step.parameters["target"] == "home"
    assert step.parameters["velocity_scale"] == 0.03
    assert step.parameters["acceleration_scale"] == 0.03


def test_ros_rviz_sim_demo_audit_event_has_milestone_fields(tmp_path):
    import llm_skill_robot.ros_rviz_sim_execution_demo as demo
    from llm_skill_robot.core.audit_logger import AuditLogger

    log_file = tmp_path / "audit.log"
    audit_logger = AuditLogger(log_file=str(log_file))

    demo._audit_event(
        audit_logger,
        "plan_generated",
        {
            "decision": "GENERATED",
            "message": "Plan generated.",
        },
    )

    entry = json.loads(log_file.read_text().strip())
    assert entry["event"] == "plan_generated"
    assert entry["timestamp"]
    assert entry["mode"] == "rviz_sim"
    assert entry["plan_id"] is None
    assert entry["decision"] == "GENERATED"
    assert entry["message"] == "Plan generated."
