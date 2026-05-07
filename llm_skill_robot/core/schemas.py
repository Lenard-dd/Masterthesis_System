from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Literal, Optional
from enum import Enum


class VerificationDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class VerificationResult(BaseModel):
    decision: VerificationDecision
    reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ValidationDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MODIFY = "MODIFY"


class ValidationResult(BaseModel):
    decision: ValidationDecision
    comments: Optional[str] = None


class StepExecutionResult(BaseModel):
    step_id: int
    skill_id: str
    status: Literal["SUCCESS", "FAILED", "SKIPPED"]
    message: str
    output: Dict[str, Any] = Field(default_factory=dict)


class ExecutionReport(BaseModel):
    task: str
    status: Literal["SUCCESS", "FAILED", "NOT_EXECUTED"]
    step_results: List[StepExecutionResult] = Field(default_factory=list)
    summary: str


class MotionPlanSummary(BaseModel):
    plan_id: str
    target_type: str
    target_name: Optional[str] = None
    target_pose: Optional[Dict[str, Any]] = None
    planning_group: str
    success: bool
    message: str
    num_trajectory_points: int = 0
    duration_sec: Optional[float] = None
    joint_names: List[str] = Field(default_factory=list)
    max_joint_delta_rad: Optional[float] = None
    max_velocity_abs: Optional[float] = None
    trajectory_preview: Dict[str, Any] = Field(default_factory=dict)


class MotionPlanCacheEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_id: str
    skill_id: str
    step_id: Optional[int] = None
    summary: MotionPlanSummary
    created_at: str
    raw_plan_available: bool = False
    raw_trajectory: Optional[Any] = None
    trajectory_start: Optional[Any] = None
    target_command: Optional[Dict[str, Any]] = None
