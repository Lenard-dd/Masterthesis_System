from llm_skill_robot.grasping.grasp_verifier import GraspVerifier


def test_verification_is_uncertain_without_feedback_or_perception():
    result = GraspVerifier().verify_grasp("cube_1")

    assert result["status"] == "UNCERTAIN"


def test_verification_is_likely_success_with_contact_and_disappearance():
    result = GraspVerifier().verify_grasp(
        "cube_1",
        gripper_state={
            "command_sent": True,
            "feedback_available": True,
            "current_width_m": 0.03,
            "object_may_be_held": True,
        },
        perception_result={"success": True, "objects": []},
    )

    assert result["status"] == "LIKELY_SUCCESS"
    assert result["signals"]["object_visible"] is False


def test_verification_is_likely_failed_when_fully_closed_and_object_visible():
    result = GraspVerifier().verify_grasp(
        "cube_1",
        gripper_state={
            "command_sent": True,
            "feedback_available": True,
            "current_width_m": 0.0,
            "object_may_be_held": False,
        },
        perception_result={
            "success": True,
            "objects": [{"object_id": "cube_1"}],
        },
    )

    assert result["status"] == "LIKELY_FAILED"


def test_contact_feedback_alone_never_claims_definite_success():
    result = GraspVerifier().verify_grasp(
        "cube_1",
        gripper_state={
            "command_sent": True,
            "feedback_available": True,
            "current_width_m": 0.03,
            "object_may_be_held": True,
        },
    )

    assert result["status"] == "UNCERTAIN"
    assert result["status"] != "SUCCESS"
