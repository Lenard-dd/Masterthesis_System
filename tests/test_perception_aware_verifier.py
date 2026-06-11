from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import VerificationDecision
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.perception.scene_model import ObjectPose, SceneModel, SceneObject


def _verifier():
    return PlanVerifier(SkillLibrary())


def _plan(step: PlanStep) -> RobotPlan:
    return RobotPlan(task="perception aware plan", steps=[step])


def _pick_step(parameters: dict) -> PlanStep:
    return PlanStep(
        step_id="1",
        skill_id="pick_object",
        description="Pick object",
        parameters=parameters,
    )


def _scene_object(**overrides) -> SceneObject:
    data = {
        "object_id": "cup_1",
        "label": "cup",
        "confidence": 0.9,
        "pose_available": True,
        "pose": ObjectPose(frame="base_link", x=0.1, y=0.2, z=0.3),
        "graspable": True,
    }
    data.update(overrides)
    return SceneObject(**data)


def test_detect_object_allowed_with_warning_when_perception_unavailable():
    step = PlanStep(
        step_id="1",
        skill_id="detect_object",
        description="Detect cup",
        parameters={"query": "cup"},
    )

    result = _verifier().verify_plan(
        _plan(step),
        context={"perception_available": False},
    )

    assert result.decision == VerificationDecision.ALLOW
    assert (
        "Perception backend is not available; detect_object may fail at execution time."
        in result.warnings
    )


def test_pick_object_blocked_with_no_scene():
    result = _verifier().verify_plan(_plan(_pick_step({"object_id": "cup_1"})))

    assert result.decision == VerificationDecision.BLOCK
    assert "Cannot verify pick_object because no detected scene object is available." in result.reasons


def test_pick_object_blocked_when_object_missing():
    scene = SceneModel(objects=[_scene_object(object_id="box_1", label="box")])

    result = _verifier().verify_plan(
        _plan(_pick_step({"object_id": "cup_1"})),
        context={"scene": scene},
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("requested object is not in the detected scene" in reason for reason in result.reasons)


def test_pick_object_blocked_when_pose_unavailable():
    scene = SceneModel(
        objects=[
            _scene_object(
                pose_available=False,
                pose=None,
            )
        ]
    )

    result = _verifier().verify_plan(
        _plan(_pick_step({"object_id": "cup_1"})),
        context={"scene": scene},
    )

    assert result.decision == VerificationDecision.BLOCK
    assert "Object pose is not available." in result.reasons


def test_pick_object_blocked_when_object_not_graspable():
    scene = SceneModel(objects=[_scene_object(graspable=False)])

    result = _verifier().verify_plan(
        _plan(_pick_step({"object_id": "cup_1"})),
        context={"scene": scene},
    )

    assert result.decision == VerificationDecision.BLOCK
    assert "Object is not marked as graspable." in result.reasons


def test_pick_object_allowed_when_object_is_pose_available_graspable_and_confident():
    scene = SceneModel(objects=[_scene_object()])

    result = _verifier().verify_plan(
        _plan(_pick_step({"object_id": "cup_1", "min_confidence": 0.5})),
        context={"scene": scene, "perception_available": True},
    )

    assert result.decision == VerificationDecision.ALLOW
    assert result.reasons == []


def test_pick_object_blocked_when_confidence_below_minimum():
    scene = SceneModel(objects=[_scene_object(confidence=0.4)])

    result = _verifier().verify_plan(
        _plan(_pick_step({"object_id": "cup_1", "min_confidence": 0.5})),
        context={"scene": scene},
    )

    assert result.decision == VerificationDecision.BLOCK
    assert any("below required minimum" in reason for reason in result.reasons)
