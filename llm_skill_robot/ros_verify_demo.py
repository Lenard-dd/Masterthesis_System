"""Read-only ROS 2 verification demo for a UR5 plan."""

import json
import sys

from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.utils import get_config_dir, get_skills_dir


def _to_jsonable(value):
    """Convert pydantic models and enums into JSON-friendly values."""
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def main():
    """Read UR5 state and verify a hardcoded plan without executing it."""
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    try:
        from llm_skill_robot.robot.ur5_state_backend import UR5StateBackend
    except ImportError as e:
        print(f"Error: Failed to import UR5StateBackend: {e}", file=sys.stderr)
        sys.exit(1)

    backend = None
    try:
        rclpy.init()
        backend = UR5StateBackend(joint_states_topic="/joint_states", robot_name="ur5")
        robot_state = backend.get_robot_state(timeout_sec=3.0)

        skill_library = SkillLibrary(skills_dir=str(get_skills_dir()))
        plan_verifier = PlanVerifier(skill_library, config_dir=str(get_config_dir()))
        plan = RobotPlan(
            task="Verify UR5 home target",
            steps=[
                PlanStep(
                    step_id="1",
                    skill_id="move_to_named_target",
                    description="Verify moving to home target would be allowed",
                    parameters={
                        "target_name": "home",
                        "velocity_scale": 0.1,
                    },
                )
            ],
        )

        verification_result = plan_verifier.verify_plan(
            plan,
            context={"robot_state": robot_state},
        )

        output = {
            "current_robot_state": robot_state,
            "generated_plan": _to_jsonable(plan),
            "verification_result": _to_jsonable(verification_result),
        }
        print(json.dumps(output, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except Exception as e:
                print(f"Warning: Error during shutdown: {e}", file=sys.stderr)
        if "rclpy" in locals() and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
