"""Retained CLI modules must import without starting interactive execution."""

import importlib

import pytest


@pytest.mark.parametrize("module_name", [
    'llm_skill_robot.llm_plan_demo',
    'llm_skill_robot.nl_plan_demo',
    'llm_skill_robot.perception_demo',
    'llm_skill_robot.ros_check_real_arm',
    'llm_skill_robot.ros_d435i_sam3_demo',
    'llm_skill_robot.ros_gripper_rviz_demo',
    'llm_skill_robot.ros_nl_rviz_sim_demo',
])
def test_cli_import_has_callable_main(module_name):
    assert callable(importlib.import_module(module_name).main)
