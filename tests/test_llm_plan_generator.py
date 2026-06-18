import pytest

from llm_skill_robot.agent.llm_plan_generator import LLMPlanGenerator
from llm_skill_robot.agent.mock_plan_generator import PlanGenerationError
from llm_skill_robot.core.schemas import VerificationDecision, VerificationResult


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def generate_text(self, messages: list[dict]) -> str:
        self.messages.append(messages)
        return self.responses.pop(0)


def test_valid_json_response_parses_into_robot_plan():
    client = FakeLLMClient(
        [
            """
            {
              "task": "move to home",
              "steps": [
                {
                  "step_id": 1,
                  "skill_id": "move_to_named_target",
                  "description": "Move to home",
                  "parameters": {
                    "target": "home",
                    "velocity_scale": 0.03,
                    "acceleration_scale": 0.03
                  }
                }
              ]
            }
            """
        ]
    )
    generator = LLMPlanGenerator(client)

    plan = generator.generate_plan("move to home", available_skills=["move_to_named_target"])

    assert plan.task == "move to home"
    assert plan.steps[0].step_id == "1"
    assert plan.steps[0].skill_id == "move_to_named_target"
    assert plan.steps[0].parameters["target"] == "home"


def test_markdown_fenced_json_is_parsed_correctly():
    client = FakeLLMClient(
        [
            """```json
{
  "task": "open gripper",
  "steps": [
    {
      "step_id": 1,
      "skill_id": "open_gripper",
      "description": "Open the gripper",
      "parameters": {}
    }
  ]
}
```"""
        ]
    )
    generator = LLMPlanGenerator(client)

    plan = generator.generate_plan("open gripper", available_skills=["open_gripper"])

    assert plan.steps[0].skill_id == "open_gripper"
    assert plan.steps[0].parameters == {}


def test_invalid_json_raises_plan_generation_error():
    client = FakeLLMClient(["not json"])
    generator = LLMPlanGenerator(client)

    with pytest.raises(PlanGenerationError, match="valid RobotPlan JSON"):
        generator.generate_plan("move somewhere", available_skills=["move_to_named_target"])


def test_repair_plan_returns_corrected_robot_plan():
    client = FakeLLMClient(
        [
            """
            {
              "task": "move to observe",
              "steps": [
                {
                  "step_id": 1,
                  "skill_id": "move_to_named_target",
                  "description": "Move to observe",
                  "parameters": {
                    "target": "observe",
                    "velocity_scale": 0.03,
                    "acceleration_scale": 0.03
                  }
                }
              ]
            }
            """
        ]
    )
    generator = LLMPlanGenerator(client)
    verification = VerificationResult(
        decision=VerificationDecision.BLOCK,
        reasons=["Invalid value 'watch' for parameter 'target_name'"],
    )

    plan = generator.repair_plan(
        user_instruction="move to observe",
        available_skills=["move_to_named_target"],
        previous_plan_or_text='{"target": "watch"}',
        verification_result=verification,
    )

    assert plan.steps[0].parameters["target"] == "observe"
    assert "repair" in client.messages[0][1]["content"]


def test_prompt_includes_default_motion_scales_and_fake_response_uses_them():
    client = FakeLLMClient(
        [
            """
            {
              "task": "move to home",
              "steps": [
                {
                  "step_id": 1,
                  "skill_id": "move_to_named_target",
                  "description": "Move to home",
                  "parameters": {
                    "target": "home",
                    "velocity_scale": 0.03,
                    "acceleration_scale": 0.03
                  }
                }
              ]
            }
            """
        ]
    )
    generator = LLMPlanGenerator(
        client,
        default_velocity_scale=0.03,
        default_acceleration_scale=0.03,
    )

    plan = generator.generate_plan("move to home", available_skills=["move_to_named_target"])
    prompt = client.messages[0][1]["content"]

    assert '"velocity_scale_default": 0.03' in prompt
    assert '"acceleration_scale_default": 0.03' in prompt
    assert plan.steps[0].parameters["velocity_scale"] == 0.03
    assert plan.steps[0].parameters["acceleration_scale"] == 0.03


def test_prompt_includes_robotiq_gripper_rules():
    client = FakeLLMClient(
        [
            """
            {
              "task": "open gripper",
              "steps": [
                {
                  "step_id": 1,
                  "skill_id": "open_gripper",
                  "description": "Open gripper",
                  "parameters": {}
                }
              ]
            }
            """
        ]
    )
    generator = LLMPlanGenerator(client)

    generator.generate_plan("open gripper", available_skills=["open_gripper"])
    prompt = client.messages[0][1]["content"]

    assert '"type": "robotiq_2f_140"' in prompt
    assert '"max_width_m": 0.14' in prompt
    assert "set_gripper_width" in prompt
    assert "Do not generate raw gripper commands" in prompt


def test_prompt_includes_perception_rules_for_object_pose():
    client = FakeLLMClient(
        [
            """
            {
              "task": "detect red cube",
              "steps": [
                {
                  "step_id": 1,
                  "skill_id": "detect_object",
                  "description": "Detect red cube",
                  "parameters": {
                    "query": "red cube",
                    "min_confidence": 0.5,
                    "require_pose": true
                  }
                }
              ]
            }
            """
        ]
    )
    generator = LLMPlanGenerator(client)

    generator.generate_plan("pick up the red cube", available_skills=["detect_object", "pick_object"])
    prompt = client.messages[0][1]["content"]

    assert "detect_object before pick_object" in prompt
    assert "natural-language query" in prompt
    assert "pick_object with the same query" in prompt
    assert "Do not use move_to_named_target for pickup or place actions" in prompt
    assert "red cube" in prompt
    assert "Do not invent object poses" in prompt
