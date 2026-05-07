from pydantic import BaseModel, Field
from typing import List, Optional, Any
from enum import Enum


class ParameterType(str, Enum):
    STRING = "string"
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


class Parameter(BaseModel):
    name: str
    type: ParameterType
    required: bool = False
    allowed_values: Optional[List[Any]] = None
    default: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class Execution(BaseModel):
    backend_method: str


class Skill(BaseModel):
    skill_id: str = Field(..., description="Unique identifier for the skill")
    name: str
    description: str
    parameters: List[Parameter] = []
    preconditions: List[str] = []
    effects: List[str] = []
    execution: Execution
