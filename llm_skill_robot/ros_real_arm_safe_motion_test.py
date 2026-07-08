"""Interactive restricted single-motion test for the real UR5 arm."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.safety.hardware_mode import (
    assert_arm_execution_allowed,
    load_hardware_mode,
    print_hardware_mode,
)
from llm_skill_robot.safety.real_arm_safety import (
    RealArmSafetyConfig,
    is_real_motion_skill_allowed,
    load_real_arm_safety,
    print_real_arm_safety,
    require_execute_confirmation,
    validate_motion_summary,
    validate_named_target,
)


SAFE_NAMED_TARGETS = ("safe_home", "observe")
BLOCKED_SKILLS = ("approach_grasp", "pick_object", "lift_object")


@dataclass
class LastMotionPlan:
    plan_id: str
    target_name: str
    skill_id: str
    summary: dict[str, Any]


def main() -> None:
    """Start the interactive restricted real-arm motion test."""
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    try:
        from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
        from llm_skill_robot.robot.ur5_state_backend import UR5StateBackend
    except ImportError as exc:
        print(f"Error: failed to import ROS backends: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        hardware_mode = load_hardware_mode()
        real_arm_safety = load_real_arm_safety()
        assert_arm_execution_allowed(hardware_mode)
    except Exception as exc:
        print("REAL UR5 RESTRICTED MOTION TEST")
        print(f"BLOCKED: safety configuration does not allow real arm motion: {exc}")
        return

    velocity_scale = min(0.05, real_arm_safety.limits.max_velocity_scale)
    acceleration_scale = min(0.05, real_arm_safety.limits.max_acceleration_scale)
    logger = AuditLogger("real_arm_motion_audit.log", hardware_mode=hardware_mode)

    print_startup_banner()
    print_hardware_mode(hardware_mode)
    print_real_arm_safety(real_arm_safety)
    print(
        "\nRestricted scales: "
        f"velocity_scale={velocity_scale:.3f}, "
        f"acceleration_scale={acceleration_scale:.3f}"
    )

    node = None
    backend = None
    state_backend = None
    last_plan: LastMotionPlan | None = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_real_arm_safe_motion_test")
        backend = UR5MoveItPlanBackend(
            node=node,
            dry_run=False,
            execution_mode="real_robot",
            hardware_mode=hardware_mode,
        )
        state_backend = UR5StateBackend(node=node)

        while True:
            print_menu()
            choice = input("Select: ").strip()
            if choice.lower() == "q":
                break
            if choice == "1":
                last_plan = plan_named_target(
                    backend,
                    "safe_home",
                    real_arm_safety,
                    logger,
                    velocity_scale,
                    acceleration_scale,
                )
            elif choice == "2":
                execute_last_plan_for_target(
                    backend,
                    last_plan,
                    "safe_home",
                    real_arm_safety,
                    logger,
                )
            elif choice == "3":
                last_plan = plan_named_target(
                    backend,
                    "observe",
                    real_arm_safety,
                    logger,
                    velocity_scale,
                    acceleration_scale,
                )
            elif choice == "4":
                execute_last_plan_for_target(
                    backend,
                    last_plan,
                    "observe",
                    real_arm_safety,
                    logger,
                )
            elif choice == "5":
                show_current_state(state_backend)
            else:
                print("Unknown selection.")
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as exc:
        print(f"ERROR: restricted real-arm test failed: {exc}")
        logger.log_hardware_safety_event(
            "real_arm_motion_failed",
            reason=str(exc),
            real_arm_execution_blocked=True,
        )
    finally:
        if backend is not None:
            backend.shutdown()
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


def print_startup_banner() -> None:
    print("REAL UR5 RESTRICTED MOTION TEST")
    print("This can move the REAL robot.")
    print("Allowed motions:")
    print("- safe_home")
    print("- observe")
    print("- selected pregrasp if available")
    print("Blocked:")
    print("- approach_grasp")
    print("- pick_object")
    print("- lift_object")


def print_menu() -> None:
    print("\nMenu:")
    print("1. Plan to safe_home")
    print("2. Execute safe_home")
    print("3. Plan to observe")
    print("4. Execute observe")
    print("5. Show current state")
    print("q. Quit")


def plan_named_target(
    backend,
    target_name: str,
    safety_config: RealArmSafetyConfig,
    logger: AuditLogger,
    velocity_scale: float,
    acceleration_scale: float,
) -> LastMotionPlan | None:
    logger.log_hardware_safety_event(
        "real_arm_motion_requested",
        skill_id="move_to_named_target",
        target_name=target_name,
    )
    target_check = validate_menu_target(target_name, safety_config)
    if not target_check["allowed"]:
        print(f"BLOCKED: {target_check['reason']}")
        logger.log_hardware_safety_event(
            "real_arm_motion_rejected",
            skill_id="move_to_named_target",
            target_name=target_name,
            reason=target_check["reason"],
            real_arm_execution_blocked=True,
        )
        return None

    print(f"Planning to {target_name}. No motion will be executed.")
    result = backend.plan_to_named_target(
        target_name,
        velocity_scale=velocity_scale,
        acceleration_scale=acceleration_scale,
        skill_id="move_to_named_target",
    )
    summary = build_restricted_motion_summary(
        result.get("summary", {}),
        target_name=target_name,
        velocity_scale=velocity_scale,
        acceleration_scale=acceleration_scale,
    )
    logger.log_hardware_safety_event(
        "real_arm_motion_planned",
        skill_id="move_to_named_target",
        target_name=target_name,
        plan_id=result.get("plan_id"),
        planning_success=bool(result.get("success")),
        summary=summary,
    )
    if not result.get("success"):
        print(f"Planning failed: {result.get('message')}")
        return None

    print_motion_summary(summary)
    print(f"Planned target {target_name}; plan_id={result.get('plan_id')}")
    return LastMotionPlan(
        plan_id=str(result["plan_id"]),
        target_name=target_name,
        skill_id="move_to_named_target",
        summary=summary,
    )


def execute_last_plan_for_target(
    backend,
    last_plan: LastMotionPlan | None,
    target_name: str,
    safety_config: RealArmSafetyConfig,
    logger: AuditLogger,
) -> dict[str, Any]:
    if last_plan is None:
        message = "No valid plan exists. Plan first."
        print(f"BLOCKED: {message}")
        logger.log_hardware_safety_event(
            "real_arm_motion_rejected",
            target_name=target_name,
            reason=message,
            real_arm_execution_blocked=True,
        )
        return {"success": False, "status": "BLOCKED", "message": message}
    if last_plan.target_name != target_name:
        message = f"Last plan is for {last_plan.target_name}, not {target_name}."
        print(f"BLOCKED: {message}")
        logger.log_hardware_safety_event(
            "real_arm_motion_rejected",
            plan_id=last_plan.plan_id,
            target_name=target_name,
            reason=message,
            real_arm_execution_blocked=True,
        )
        return {"success": False, "status": "BLOCKED", "message": message}

    precheck = validate_execution_precheck(last_plan, safety_config)
    if not precheck["allowed"]:
        print(f"BLOCKED: {precheck['reason']}")
        logger.log_hardware_safety_event(
            "real_arm_motion_rejected",
            plan_id=last_plan.plan_id,
            skill_id=last_plan.skill_id,
            target_name=last_plan.target_name,
            reason=precheck["reason"],
            real_arm_execution_blocked=True,
        )
        return {"success": False, "status": "BLOCKED", "message": precheck["reason"]}

    print_motion_summary(last_plan.summary)
    confirmation = input("Type EXECUTE to move the REAL UR5: ")
    confirmation_check = validate_execute_confirmation(
        confirmation,
        safety_config.confirmation_phrase,
    )
    if not confirmation_check["allowed"]:
        print("CANCELLED: exact EXECUTE confirmation was not provided.")
        logger.log_hardware_safety_event(
            "real_arm_motion_rejected",
            plan_id=last_plan.plan_id,
            skill_id=last_plan.skill_id,
            target_name=last_plan.target_name,
            reason=confirmation_check["reason"],
            real_arm_execution_blocked=True,
        )
        return {"success": False, "status": "CANCELLED", "message": confirmation_check["reason"]}

    logger.log_hardware_safety_event(
        "real_arm_motion_confirmed",
        plan_id=last_plan.plan_id,
        skill_id=last_plan.skill_id,
        target_name=last_plan.target_name,
    )
    result = backend.execute_cached_plan_real(last_plan.plan_id, final_confirmed=True)
    event = "real_arm_motion_executed" if result.get("success") else "real_arm_motion_failed"
    logger.log_hardware_safety_event(
        event,
        plan_id=last_plan.plan_id,
        skill_id=last_plan.skill_id,
        target_name=last_plan.target_name,
        execution_success=bool(result.get("success")),
        message=result.get("message"),
    )
    print(result.get("message", "Execution finished."))
    return result


def validate_menu_target(
    target_name: str,
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    if target_name not in SAFE_NAMED_TARGETS:
        return {"allowed": False, "reason": f"Unknown or unsafe target '{target_name}'."}
    return validate_named_target(target_name, config)


def validate_execution_precheck(
    plan: LastMotionPlan,
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    skill_check = is_real_motion_skill_allowed(plan.skill_id, config)
    if not skill_check["allowed"]:
        return skill_check
    target_check = validate_menu_target(plan.target_name, config)
    if not target_check["allowed"]:
        return target_check
    return validate_motion_summary(plan.summary, config)


def validate_execute_confirmation(
    text: str,
    phrase: str = "EXECUTE",
) -> dict[str, Any]:
    if require_execute_confirmation(text, phrase):
        return {"allowed": True, "reason": "Exact EXECUTE confirmation accepted."}
    return {"allowed": False, "reason": "Exact EXECUTE confirmation is required."}


def build_restricted_motion_summary(
    summary: dict[str, Any],
    target_name: str,
    velocity_scale: float,
    acceleration_scale: float,
) -> dict[str, Any]:
    trajectory_preview = summary.get("trajectory_preview", {}) or {}
    return {
        "target": target_name,
        "plan_id": summary.get("plan_id"),
        "max_joint_delta_rad": summary.get("max_joint_delta_rad"),
        "total_joint_delta_rad": (
            summary.get("total_joint_delta_rad")
            or trajectory_preview.get("total_joint_delta_rad")
        ),
        "max_cartesian_step_m": (
            summary.get("max_cartesian_step_m")
            or trajectory_preview.get("max_cartesian_step_m")
        ),
        "velocity_scale": velocity_scale,
        "acceleration_scale": acceleration_scale,
    }


def print_motion_summary(summary: dict[str, Any]) -> None:
    print("\nMotion summary:")
    print(f"target: {summary.get('target')}")
    print(f"max_joint_delta_rad: {summary.get('max_joint_delta_rad')}")
    print(f"total_joint_delta_rad: {summary.get('total_joint_delta_rad')}")
    print(f"max_cartesian_step_m: {summary.get('max_cartesian_step_m')}")
    print(f"velocity_scale: {summary.get('velocity_scale')}")
    print(f"acceleration_scale: {summary.get('acceleration_scale')}")


def show_current_state(state_backend) -> None:
    state = state_backend.get_robot_state(timeout_sec=2.0)
    print("\nCurrent state:")
    print(state.get("status", {}).get("message", "state unavailable"))
    for name, value in sorted(state.get("joint_positions", {}).items()):
        print(f"- {name}: {value:.6f} rad")


if __name__ == "__main__":
    main()
