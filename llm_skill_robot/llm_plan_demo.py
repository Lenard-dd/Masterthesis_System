"""Interactive LLM RobotPlan generation demo.

This demo stops at plan generation, verification, optional repair, and human
validation. It does not call MoveIt or execute robot motion.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from llm_skill_robot.agent.llm_client import (
    LLMClientConfigurationError,
    OpenAILLMClient,
    load_llm_config,
)
from llm_skill_robot.agent.llm_plan_generator import LLMPlanGenerator
from llm_skill_robot.agent.mock_plan_generator import PlanGenerationError
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan import RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import VerificationDecision
from llm_skill_robot.core.skill_library import SkillLibrary


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


def _print_plan(plan: RobotPlan, label: str = "Generated LLM RobotPlan") -> None:
    print(f"\n{label}:")
    for step in plan.steps:
        if step.skill_id == "move_to_named_target":
            target = step.parameters.get("target") or step.parameters.get("target_name")
            print(f"{step.step_id}. {step.skill_id} target={target}")
        else:
            print(f"{step.step_id}. {step.skill_id}")


def _print_verification(result) -> None:
    print("\nVerification:")
    print(result.decision.value)
    if result.reasons:
        print(f"Reasons: {result.reasons}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def _available_skill_objects(skill_library: SkillLibrary) -> list:
    return [
        skill_library.get_skill(skill_id)
        for skill_id in skill_library.list_skills()
        if skill_library.get_skill(skill_id) is not None
    ]


def main() -> None:
    audit_logger = AuditLogger()
    llm_config = load_llm_config().get("llm", {})
    skill_library = SkillLibrary()
    plan_verifier = PlanVerifier(skill_library)
    human_validator = HumanValidator()

    _print_available_skills(skill_library)
    user_instruction = input("\nEnter a robot task:\n> ").strip()
    _audit_event(
        audit_logger,
        "user_instruction",
        {
            "instruction": user_instruction,
            "message": "User instruction received for LLM plan generation.",
        },
    )

    try:
        llm_client = OpenAILLMClient(
            model=llm_config.get("model"),
            temperature=llm_config.get("temperature"),
            max_retries=llm_config.get("max_retries"),
            timeout_sec=llm_config.get("timeout_sec"),
        )
    except LLMClientConfigurationError as e:
        print(f"\nLLM client is not configured: {e}")
        _audit_event(
            audit_logger,
            "llm_client_configuration_failed",
            {"decision": "FAILED", "message": str(e)},
        )
        print(f"\nAudit log: {audit_logger.log_file.resolve()}")
        return

    generator = LLMPlanGenerator(llm_client)
    available_skills = _available_skill_objects(skill_library)

    try:
        plan = generator.generate_plan(
            user_instruction=user_instruction,
            available_skills=available_skills,
        )
    except PlanGenerationError as e:
        print(f"\nLLM plan generation failed: {e}")
        _audit_event(
            audit_logger,
            "llm_plan_generated",
            {"decision": "FAILED", "message": str(e)},
        )
        print(f"\nAudit log: {audit_logger.log_file.resolve()}")
        return

    _print_plan(plan)
    _audit_event(
        audit_logger,
        "llm_plan_generated",
        {
            "decision": "GENERATED",
            "message": "LLMPlanGenerator produced a RobotPlan.",
            "plan": plan.model_dump(),
        },
    )

    verification_result = plan_verifier.verify_plan(plan)
    _print_verification(verification_result)
    _audit_event(
        audit_logger,
        "verification_result",
        {
            "decision": verification_result.decision.value,
            "message": "Verification completed.",
            "reasons": verification_result.reasons,
            "warnings": verification_result.warnings,
        },
    )

    repair_attempts = 0
    if verification_result.decision == VerificationDecision.BLOCK:
        choice = input("\nTry LLM repair? [y/N] ").strip().lower()
        if choice == "y":
            repair_attempts += 1
            _audit_event(
                audit_logger,
                "repair_attempts",
                {
                    "decision": "STARTED",
                    "message": "LLM repair requested.",
                    "attempt": repair_attempts,
                    "previous_plan": plan.model_dump(),
                    "verification_result": verification_result.model_dump(),
                },
            )
            try:
                plan = generator.repair_plan(
                    user_instruction=user_instruction,
                    available_skills=available_skills,
                    previous_plan_or_text=plan,
                    verification_result=verification_result,
                )
            except PlanGenerationError as e:
                print(f"\nLLM repair failed: {e}")
                _audit_event(
                    audit_logger,
                    "repair_attempts",
                    {
                        "decision": "FAILED",
                        "message": str(e),
                        "attempt": repair_attempts,
                    },
                )
                print(f"\nAudit log: {audit_logger.log_file.resolve()}")
                return

            _print_plan(plan, label="Repaired LLM RobotPlan")
            verification_result = plan_verifier.verify_plan(plan)
            _print_verification(verification_result)
            _audit_event(
                audit_logger,
                "repair_attempts",
                {
                    "decision": "COMPLETED",
                    "message": "LLM repair completed.",
                    "attempt": repair_attempts,
                    "repaired_plan": plan.model_dump(),
                    "verification_result": verification_result.model_dump(),
                },
            )
            _audit_event(
                audit_logger,
                "verification_result",
                {
                    "decision": verification_result.decision.value,
                    "message": "Verification completed after repair.",
                    "reasons": verification_result.reasons,
                    "warnings": verification_result.warnings,
                },
            )

    if verification_result.decision != VerificationDecision.ALLOW:
        print("\nPlan was not presented for human validation because verification did not allow it.")
        print(f"\nAudit log: {audit_logger.log_file.resolve()}")
        return

    validation_result = human_validator.validate_plan(plan)
    print("\nHuman Validation:")
    print(validation_result.decision.value)
    _audit_event(
        audit_logger,
        "human_validation_result",
        {
            "decision": validation_result.decision.value,
            "message": "Human validation completed.",
            "comments": validation_result.comments,
        },
    )
    print(f"\nAudit log: {audit_logger.log_file.resolve()}")


if __name__ == "__main__":
    main()
