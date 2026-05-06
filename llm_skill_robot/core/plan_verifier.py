import yaml
from pathlib import Path
from typing import Dict, Any
from .skill_library import SkillLibrary
from .plan import RobotPlan
from .schemas import VerificationResult, VerificationDecision


class PlanVerifier:
    def __init__(self, skill_library: SkillLibrary, config_dir: str = "config"):
        self.skill_library = skill_library
        self.config_dir = Path(config_dir)
        self.robot_capabilities = self.load_robot_capabilities()

    def load_robot_capabilities(self) -> Dict[str, Any]:
        """Load robot capabilities from config."""
        capabilities_file = self.config_dir / "robot_capabilities.yaml"
        if capabilities_file.exists():
            with open(capabilities_file, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def verify_plan(self, plan: RobotPlan) -> VerificationResult:
        """Verify a robot plan against skills and capabilities."""
        reasons = []
        warnings = []

        for step in plan.steps:
            skill = self.skill_library.get_skill(step.skill_id)
            if not skill:
                reasons.append(f"Unknown skill: {step.skill_id}")
                continue

            # Check required parameters
            for param in skill.parameters:
                if param.required:
                    if param.name not in step.parameters:
                        reasons.append(f"Missing required parameter '{param.name}' for skill '{step.skill_id}'")
                    else:
                        value = step.parameters[param.name]
                        # Check type
                        if not self.check_parameter_type(value, param.type):
                            reasons.append(f"Invalid type for parameter '{param.name}' in skill '{step.skill_id}': expected {param.type}")
                        # Check allowed values
                        if param.allowed_values and value not in param.allowed_values:
                            reasons.append(f"Invalid value '{value}' for parameter '{param.name}' in skill '{step.skill_id}': allowed {param.allowed_values}")

        # Simple robot capabilities check (placeholder)
        # For now, just check if there are any capabilities defined
        if not self.robot_capabilities:
            warnings.append("No robot capabilities loaded")

        decision = VerificationDecision.BLOCK if reasons else VerificationDecision.ALLOW
        return VerificationResult(decision=decision, reasons=reasons, warnings=warnings)

    def check_parameter_type(self, value: Any, expected_type: str) -> bool:
        """Check if a parameter value matches the expected type."""
        type_map = {
            "string": str,
            "float": float,
            "int": int,
            "bool": bool
        }
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        return False