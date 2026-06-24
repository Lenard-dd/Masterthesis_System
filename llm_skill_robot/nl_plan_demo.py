"""Natural-language to RobotPlan demo using the rule-based mock generator."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from llm_skill_robot.agent.mock_plan_generator import MockPlanGenerator, PlanGenerationError
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan import RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import ValidationDecision, ValidationResult
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _audit_event(audit_logger: AuditLogger, event: str, payload: Dict[str, Any]) -> None:
    if hasattr(audit_logger, "_write_entry"):
        audit_logger._write_entry(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **payload,
            }
        )


def _print_available_skills(skill_library: SkillLibrary) -> None:
    print("Available skills:")
    for skill_id in sorted(skill_library.list_skills()):
        print(f"- {skill_id}")


def _print_generated_plan(plan: RobotPlan) -> None:
    print("\nGenerated Skill Plan:")
    for step in plan.steps:
        if step.skill_id == "move_to_named_target":
            target = step.parameters.get("target") or step.parameters.get("target_name")
            print(f"{step.step_id}. {step.skill_id} target={target}")
        else:
            print(f"{step.step_id}. {step.skill_id}")


def _print_verification_result(result) -> None:
    print("\nVerification:")
    print(result.decision.value)
    if result.reasons:
        print(f"Reasons: {result.reasons}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


class LetterHumanValidator(HumanValidator):
    """HumanValidator variant with the prompt used by the NL plan demo."""

    def validate_plan(self, plan: RobotPlan) -> ValidationResult:
        del plan
        while True:
            choice = input(
                "\nHuman Validation:\n"
                "Approve this plan? [a]pprove / [r]eject / [m]odify: "
            ).strip().lower()
            if choice in {"a", "approve"}:
                return ValidationResult(decision=ValidationDecision.APPROVE)
            if choice in {"r", "reject"}:
                return ValidationResult(decision=ValidationDecision.REJECT)
            if choice in {"m", "modify"}:
                comments = input("Please describe the requested modifications: ").strip()
                return ValidationResult(
                    decision=ValidationDecision.MODIFY,
                    comments=comments,
                )
            print("Invalid choice. Please enter a, r, or m.")


def main() -> None:
    print_hardware_mode(load_hardware_mode())
    audit_logger = AuditLogger()
    skill_library = SkillLibrary(skills_dir=str(_project_root() / "skills"))
    plan_generator = MockPlanGenerator()
    plan_verifier = PlanVerifier(skill_library)
    human_validator = LetterHumanValidator()

    _print_available_skills(skill_library)
    user_instruction = input("\nEnter a robot task:\n> ").strip()
    _audit_event(
        audit_logger,
        "nl_instruction_received",
        {
            "instruction": user_instruction,
            "message": "Natural-language instruction received.",
        },
    )

    try:
        plan = plan_generator.generate_plan(
            user_instruction=user_instruction,
            available_skills=skill_library.list_skills(),
        )
    except PlanGenerationError as exc:
        print(f"\nPlan generation failed: {exc}")
        _audit_event(
            audit_logger,
            "plan_generation_failed",
            {
                "instruction": user_instruction,
                "decision": "FAILED",
                "message": str(exc),
            },
        )
        return

    _print_generated_plan(plan)
    _audit_event(
        audit_logger,
        "plan_generated",
        {
            "decision": "GENERATED",
            "message": "MockPlanGenerator produced a RobotPlan.",
            "plan": plan.model_dump(),
        },
    )

    verification_result = plan_verifier.verify_plan(plan)
    _print_verification_result(verification_result)
    _audit_event(
        audit_logger,
        "plan_verified",
        {
            "decision": verification_result.decision.value,
            "message": "Plan verification completed.",
            "reasons": verification_result.reasons,
            "warnings": verification_result.warnings,
        },
    )

    if verification_result.decision.value != "ALLOW":
        print("\nPlan was not presented for human validation because verification did not allow it.")
        return

    validation_result = human_validator.validate_plan(plan)
    print("\nHuman Decision:")
    print(validation_result.decision.value)
    _audit_event(
        audit_logger,
        "human_validated",
        {
            "decision": validation_result.decision.value,
            "message": "Human validation completed.",
            "comments": validation_result.comments,
        },
    )

    print(f"\nAudit log: {audit_logger.log_file.resolve()}")


if __name__ == "__main__":
    main()
