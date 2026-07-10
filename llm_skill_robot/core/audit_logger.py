import json
import datetime
from pathlib import Path
from typing import Any, Dict
from .plan import RobotPlan
from .schemas import VerificationResult, ValidationResult, ExecutionReport


class AuditLogger:
    def __init__(self, log_file: str = "audit.log", hardware_mode=None):
        self.log_file = Path(log_file)
        if hardware_mode is None:
            try:
                from llm_skill_robot.safety.hardware_mode import load_hardware_mode

                hardware_mode = load_hardware_mode()
            except Exception:
                hardware_mode = None
        self.hardware_mode = hardware_mode

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

    def log_execution_report(self, report: ExecutionReport):
        """Log an execution report."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "plan_executed",
            "task": report.task,
            "execution_report": report.model_dump()
        }
        self._write_entry(entry)

    def log_hardware_safety_event(self, event: str, **fields: Any):
        """Record a hardware-mode decision using stable HW5 audit fields."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event,
            **fields,
        }
        self._write_entry(entry)

    def _write_entry(self, entry: Dict[str, Any]):
        """Write a log entry to the file."""
        entry = dict(entry)
        mode_name = getattr(self.hardware_mode, "mode", None)
        entry.setdefault("hardware_mode", mode_name)
        entry.setdefault("real_camera_used", False)
        entry.setdefault("real_gripper_command_requested", False)
        entry.setdefault("real_gripper_command_sent", False)
        entry.setdefault("real_arm_execution_blocked", False)
        entry.setdefault("restricted_real_motion_mode_active", False)
        entry.setdefault("unsafe_llm_step_blocked", False)
        entry.setdefault("real_arm_step_requires_confirmation", False)
        entry.setdefault("real_gripper_step_requires_confirmation", False)
        entry.setdefault("real_arm_motion_executed", False)
        entry.setdefault("grasp_candidate_reviewed", False)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
