from llm_skill_robot.grasping.full_grasp_state_machine import FullGraspStateMachine


def test_valid_sequence_reaches_done():
    machine = FullGraspStateMachine()
    sequence = (
        machine.OBJECT_DETECTED,
        machine.POINT_CLOUD_BUILT,
        machine.GRASP_GENERATED,
        machine.PREGRASP_PLANNED,
        machine.AT_PREGRASP,
        machine.APPROACH_PLANNED,
        machine.AT_GRASP_POSE,
        machine.GRIPPER_CLOSED,
        machine.RETREAT_PLANNED,
        machine.RETREATED,
        machine.GRASP_VERIFIED,
        machine.DONE,
    )

    for state in sequence:
        assert machine.transition(state)["allowed"] is True

    assert machine.current_state() == machine.DONE


def test_cannot_jump_from_init_to_grasp_pose():
    machine = FullGraspStateMachine()

    result = machine.transition(machine.AT_GRASP_POSE)

    assert result["allowed"] is False
    assert machine.current_state() == machine.INIT


def test_cannot_close_gripper_before_grasp_pose():
    machine = FullGraspStateMachine()

    assert machine.transition(machine.GRIPPER_CLOSED)["allowed"] is False
    assert machine.require_state(machine.AT_GRASP_POSE)["allowed"] is False
    assert machine.current_state() == machine.INIT


def test_abort_works_from_any_state():
    sequence = (
        FullGraspStateMachine.OBJECT_DETECTED,
        FullGraspStateMachine.POINT_CLOUD_BUILT,
        FullGraspStateMachine.GRASP_GENERATED,
        FullGraspStateMachine.PREGRASP_PLANNED,
        FullGraspStateMachine.AT_PREGRASP,
        FullGraspStateMachine.APPROACH_PLANNED,
        FullGraspStateMachine.AT_GRASP_POSE,
        FullGraspStateMachine.GRIPPER_CLOSED,
        FullGraspStateMachine.RETREAT_PLANNED,
        FullGraspStateMachine.RETREATED,
        FullGraspStateMachine.GRASP_VERIFIED,
        FullGraspStateMachine.DONE,
    )
    for index, state in enumerate((FullGraspStateMachine.INIT, *sequence)):
        machine = FullGraspStateMachine()
        for next_state in sequence[:index]:
            assert machine.transition(next_state)["allowed"] is True

        result = machine.abort("operator requested abort")

        assert result["allowed"] is True
        assert result["previous_state"] == state
        assert machine.current_state() == machine.ABORTED


def test_context_updates_correctly():
    machine = FullGraspStateMachine()
    machine.update_context(
        selected_object_id="cube_1",
        selected_grasp_candidate_id="grasp_4",
        selected_grasp_score=0.87,
        point_cloud_path="/tmp/cube_1.ply",
    )
    machine.transition(machine.OBJECT_DETECTED)
    machine.transition(machine.POINT_CLOUD_BUILT)
    machine.transition(machine.GRASP_GENERATED)
    machine.transition(machine.PREGRASP_PLANNED)

    context = machine.context()
    assert context["selected_object_id"] == "cube_1"
    assert context["selected_grasp_candidate_id"] == "grasp_4"
    assert context["selected_grasp_score"] == 0.87
    assert context["point_cloud_path"] == "/tmp/cube_1.ply"
    assert context["grasp_plan_available"] is True
    assert context["pregrasp_planned"] is True
