from pydantic import BaseModel, Field
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