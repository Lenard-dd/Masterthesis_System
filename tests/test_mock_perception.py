from llm_skill_robot.agent.mock_perception import MockPerceptionAdapter
from llm_skill_robot.core.plan import PlanStep


def _step(skill_id, **parameters):
    return PlanStep(step_id="1", skill_id=skill_id, description=skill_id, parameters=parameters)


def test_mock_perception_supplies_consistent_agent_observations():
    adapter = MockPerceptionAdapter()
    context = {}
    detection = adapter.execute(_step("detect_object", query="red cube"), context)
    cloud = adapter.execute(_step("build_object_point_cloud", object_id="mock_red_cube_1"), context)
    grasp = adapter.execute(_step("generate_grasp_pose", object_id="mock_red_cube_1"), context)

    assert detection["output"]["object_id"] == "mock_red_cube_1"
    assert cloud["output"]["point_cloud_path"].startswith("mock://")
    assert grasp["output"]["candidate_id"].startswith("mock_grasp_")
    assert set(context["grasp_motion_poses"]) == {"pregrasp", "grasp", "retreat"}
    assert context["grasp_motion_poses"]["pregrasp"]["position"]["z"] == 0.30
