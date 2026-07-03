import yaml

from llm_skill_robot.utils import get_config_dir


def test_ur5_config_uses_installed_moveit_planning_group():
    with (get_config_dir() / "ur5.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["group_name"] == "ur5_manipulator"
