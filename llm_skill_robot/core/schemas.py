from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    mode: str = "plan_only"
    display_trajectory: Optional[Any] = None
    robot_trajectory: Optional[Any] = None
    joint_trajectory: Optional[Any] = None
    target_command: Optional[Dict[str, Any]] = None
    raw_plan_available: bool = False
    trajectory_start: Optional[Any] = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_raw_trajectory(cls, data: Any) -> Any:
        if isinstance(data, dict) and "raw_trajectory" in data and "robot_trajectory" not in data:
            data = dict(data)
            data["robot_trajectory"] = data.pop("raw_trajectory")
        return data

    @property
    def raw_trajectory(self) -> Optional[Any]:
        """Backward-compatible alias for the RViz-preview robot trajectory."""
        return self.robot_trajectory

    @property
    def has_executable_trajectory(self) -> bool:
        return (
            self.joint_trajectory is not None
            or self.robot_trajectory is not None
            or self.target_command is not None
        )

    @property
    def has_visual_preview(self) -> bool:
        return self.display_trajectory is not None or self.robot_trajectory is not None


CachedMotionPlan = MotionPlanCacheEntry
