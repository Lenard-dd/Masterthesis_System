from llm_skill_robot.core.grasp_motion_policy import GraspMotionPolicy


def test_policy_maps_pilz_stages_and_respects_safety_scales():
    policy = GraspMotionPolicy.from_grasping_config(
        {
            "plan_only_preview": {
                "velocity_scale": 0.2,
                "acceleration_scale": 0.2,
                "moveit": {
                    "pipeline_id": "pilz_industrial_motion_planner",
                    "stage_planner_ids": {
                        "pregrasp": "PTP", "grasp": "LIN", "retreat": "LIN"
                    },
                },
            }
        },
        max_velocity_scale=0.05,
        max_acceleration_scale=0.06,
    )

    assert policy.velocity_scale == 0.05
    assert policy.acceleration_scale == 0.06
    assert policy.kwargs_for("move_to_pregrasp")["planner_id"] == "PTP"
    assert policy.kwargs_for("approach_grasp")["planner_id"] == "LIN"
    assert policy.kwargs_for("retreat_grasp")["planner_id"] == "LIN"


def test_policy_pregrasp_variant_does_not_mutate_other_stages():
    policy = GraspMotionPolicy.from_grasping_config({}, max_velocity_scale=1.0, max_acceleration_scale=1.0)
    constrained = policy.with_pregrasp_variant({"orientation_tolerance": 0.05})

    assert constrained.kwargs_for("move_to_pregrasp")["orientation_tolerance"] == 0.05
    assert "orientation_tolerance" not in constrained.kwargs_for("approach_grasp")
    assert "orientation_tolerance" not in policy.kwargs_for("move_to_pregrasp")
