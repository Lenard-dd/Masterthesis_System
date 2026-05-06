import json
import datetime
from pathlib import Path
from typing import Any, Dict
from .plan import RobotPlan
from .schemas import VerificationResult, ValidationResult


class AuditLogger:
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = Path(log_file)

    def log_plan_generation(self, plan: RobotPlan):
        """Log a generated plan."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "plan_generated",
            "plan": plan.model_dump()
        }
        self._write_entry(entry)

    def log_verification(self, plan: RobotPlan, result: VerificationResult):
        """Log a verification result."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "plan_verified",
            "plan_task": plan.task,
            "verification_result": result.model_dump()
        }
        self._write_entry(entry)

    def log_validation(self, plan: RobotPlan, result: ValidationResult):
        """Log a validation result."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "plan_validated",
            "plan_task": plan.task,
            "validation_result": result.model_dump()
        }
        self._write_entry(entry)

    def _write_entry(self, entry: Dict[str, Any]):
        """Write a log entry to the file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')