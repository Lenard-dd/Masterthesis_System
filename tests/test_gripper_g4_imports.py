def test_ros_check_gripper_model_imports_without_ros_initialization():
    import llm_skill_robot.ros_check_gripper_model as module

    assert callable(module.main)
    assert module._is_gripper_joint("robotiq_finger_joint") is True
    assert module._is_gripper_joint("shoulder_pan_joint") is False
    assert module.EXPECTED_ROBOTIQ_2F140_JOINTS == ("robotiq_finger_joint",)


def test_robotiq_joint_state_sim_imports_without_ros_initialization():
    from llm_skill_robot.robot.robotiq_2f140_joint_state_sim import (
        Robotiq2F140JointStateSim,
    )

    assert Robotiq2F140JointStateSim.__name__ == "Robotiq2F140JointStateSim"
