from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class VerificationDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class VerificationResult(BaseModel):
    decision: VerificationDecision
    reasons: List[str] = []
    warnings: List[str] = []


class ValidationDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MODIFY = "MODIFY"


class ValidationResult(BaseModel):
    decision: ValidationDecision
    comments: Optional[str] = None