from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ObjectDetection(BaseModel):
    object_id: str
    position: Dict[str, float]  # x, y, z
    confidence: float


class SceneModel(BaseModel):
    objects: List[ObjectDetection] = Field(default_factory=list)
