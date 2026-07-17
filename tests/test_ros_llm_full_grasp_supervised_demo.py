import importlib

import pytest

from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.ros_llm_full_grasp_supervised_demo import (
    EXPECTED_REAL_FULL_GRASP_SEQUENCE,
    execute_approved_plan,
    validated_object_query,
)


def _plan() -> RobotPlan:
    parameters = [
        {"target_name": "observe"},
        {"query": "black cube"},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>"},
        {},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>", "during_contact": True},
        {"object_id": "<resolved>"},
        {"object_id": "<resolved>"},
    ]
    return RobotPlan(
        task="grasp black cube",
        steps=[
            PlanStep(
                step_id=str(index),
                skill_id=skill_id,
                description=skill_id,
                parameters=parameters[index - 1],
            )
            for index, skill_id in enumerate(
                EXPECTED_REAL_FULL_GRASP_SEQUENCE, start=1
            )
        ],
    )


def test_module_imports_without_ros_initialization():
    module = importlib.import_module(
        "llm_skill_robot.ros_llm_full_grasp_supervised_demo"
    )
    assert callable(module.main)


def test_exact_supervised_plan_resolves_object_query():
    assert validated_object_query(_plan()) == "black cube"


def test_plan_without_observe_is_blocked():
    plan = _plan()
    plan.steps[0].parameters["target_name"] = "safe_home"

    with pytest.raises(ValueError, match="observe"):
        validated_object_query(plan)


def test_plan_without_contact_context_is_blocked():
    plan = _plan()
    plan.steps[8].parameters["during_contact"] = False

    with pytest.raises(ValueError, match="during_contact=true"):
        validated_object_query(plan)


def test_plan_with_missing_or_reordered_step_is_blocked():
    plan = _plan()
    plan.steps.pop(7)

    with pytest.raises(ValueError, match="exact supervised sequence"):
        validated_object_query(plan)


def test_approved_plan_routes_to_existing_full_grasp_controller():
    class FakeController:
        def __init__(self):
            self.queries = []

        def run(self, query):
            self.queries.append(query)
            return {"status": "DONE", "query": query}

    controller = FakeController()

    report = execute_approved_plan(_plan(), controller)

    assert report == {"status": "DONE", "query": "black cube"}
    assert controller.queries == ["black cube"]
