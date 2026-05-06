from .plan import RobotPlan
from .schemas import ValidationResult, ValidationDecision


class HumanValidator:
    def validate_plan(self, plan: RobotPlan) -> ValidationResult:
        """Present the plan to the user for validation."""
        print("=== Robot Plan Validation ===")
        print(f"Task: {plan.task}")
        print("\nSteps:")
        for i, step in enumerate(plan.steps, 1):
            print(f"{i}. {step.description}")
            if step.parameters:
                print(f"   Parameters: {step.parameters}")

        print("\nOptions:")
        print("1. Approve")
        print("2. Reject")
        print("3. Request modification")

        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                if choice == "1":
                    return ValidationResult(decision=ValidationDecision.APPROVE)
                elif choice == "2":
                    return ValidationResult(decision=ValidationDecision.REJECT)
                elif choice == "3":
                    comments = input("Please describe the requested modifications: ").strip()
                    return ValidationResult(decision=ValidationDecision.MODIFY, comments=comments)
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                return ValidationResult(decision=ValidationDecision.REJECT)