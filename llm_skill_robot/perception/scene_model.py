from typing import Any, Literal

from pydantic import BaseModel, Field


class ObjectPose(BaseModel):
    frame: str
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


class SceneObject(BaseModel):
    object_id: str
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    mask_available: bool = False
    pose_available: bool = False
    pose: ObjectPose | None = None
    graspable: bool = False
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneModel(BaseModel):
    objects: list[SceneObject] = Field(default_factory=list)
    frame: str = "base_link"
    timestamp: str | None = None
    source: str = "perception_pipeline"

    def find_by_label(self, label: str) -> list[SceneObject]:
        return [scene_object for scene_object in self.objects if scene_object.label == label]

    def find_by_id(self, object_id: str) -> SceneObject | None:
        for scene_object in self.objects:
            if scene_object.object_id == object_id:
                return scene_object
        return None

    def has_object(self, label_or_id: str) -> bool:
        return self.find_by_id(label_or_id) is not None or bool(
            self.find_by_label(label_or_id)
        )


DetectionStatus = Literal["SUCCESS", "NOT_FOUND", "NOT_AVAILABLE", "FAILED"]


class DetectionResult(BaseModel):
    query: str
    status: DetectionStatus
    objects: list[SceneObject] = Field(default_factory=list)
    message: str
    scene: SceneModel | None = None
