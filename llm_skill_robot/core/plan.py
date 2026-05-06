from pydantic import BaseModel, Field
from typing import List, Dict, Any


class PlanStep(BaseModel):
    step_id: str
    skill_id: str
    description: str
    parameters: Dict[str, Any] = {}


class RobotPlan(BaseModel):
    task: str
    steps: List[PlanStep]