import json

from PIL import Image

from llm_skill_robot.perception.labeled_region_recognition import (
    LabeledRegionResolver,
    LabeledRegionVisionClient,
    labeled_region_letter,
)
from llm_skill_robot.perception.perception_pipeline import PerceptionPipeline
from llm_skill_robot.perception.scene_model import ObjectPose, SceneModel, SceneObject
from llm_skill_robot.perception.spatial_reasoning import RelativePlacePosePlanner


class _FakeVisionLLM:
    def __init__(self, readings):
        self.readings = readings
        self.messages = None

    def generate_text(self, messages):
        self.messages = messages
        return json.dumps({"readings": self.readings})


def _paper(object_id, bbox):
    return SceneObject(
        object_id=object_id,
        label="white rectangular paper",
        confidence=0.91,
        mask_available=True,
        pose_available=True,
        pose=ObjectPose(frame="base_link", x=0.4, y=0.1, z=0.001),
        metadata={
            "query": "white rectangular paper",
            "bbox_xyxy": bbox,
            "mask_path": f"/tmp/{object_id}.png",
        },
    )


def test_labeled_region_query_recognizes_area_letter_only():
    assert labeled_region_letter("area A") == "A"
    assert labeled_region_letter("the area A") == "A"
    assert labeled_region_letter("Region b") == "B"
    assert labeled_region_letter("区域 C") == "C"
    assert labeled_region_letter("white rectangular paper") is None
    assert labeled_region_letter("area alpha") is None


def test_vision_client_sends_one_crop_per_sam3_candidate(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    llm = _FakeVisionLLM([
        {"candidate_id": "paper_1", "label": "A", "readable": True},
        {"candidate_id": "paper_2", "label": "UNKNOWN", "readable": False},
    ])
    client = LabeledRegionVisionClient(llm, image_detail="high", crop_padding_px=2)

    readings = client.read_labels(image_path, [
        {"candidate_id": "paper_1", "bbox_xyxy": [5, 5, 45, 35]},
        {"candidate_id": "paper_2", "bbox_xyxy": [50, 10, 95, 70]},
    ])

    assert readings == {"paper_1": "A", "paper_2": "UNKNOWN"}
    image_parts = [
        item for item in llm.messages[1]["content"] if item["type"] == "image_url"
    ]
    assert len(image_parts) == 2
    assert all(item["image_url"]["detail"] == "high" for item in image_parts)
    assert all(item["image_url"]["url"].startswith("data:image/png;base64,") for item in image_parts)


def test_resolver_relabels_only_the_unique_matching_paper(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    llm = _FakeVisionLLM([
        {"candidate_id": "paper_1", "label": "B", "readable": True},
        {"candidate_id": "paper_2", "label": "A", "readable": True},
    ])
    resolver = LabeledRegionResolver(LabeledRegionVisionClient(llm))
    scene = SceneModel(objects=[
        _paper("paper-object-1", [0, 0, 45, 40]),
        _paper("paper-object-2", [50, 0, 99, 40]),
    ])

    resolved, details = resolver.resolve(scene, image_path, ["area A"])

    assert details["status"] == "SUCCESS"
    assert resolved.objects[0].label == "white rectangular paper"
    assert resolved.objects[1].label == "area A"
    assert resolved.objects[1].object_id == "paper-object-2"
    assert resolved.objects[1].metadata["region_label"] == "A"
    assert resolved.objects[1].pose == scene.objects[1].pose


def test_resolver_rejects_duplicate_label_matches(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    llm = _FakeVisionLLM([
        {"candidate_id": "paper_1", "label": "A", "readable": True},
        {"candidate_id": "paper_2", "label": "A", "readable": True},
    ])
    resolver = LabeledRegionResolver(LabeledRegionVisionClient(llm))
    scene = SceneModel(objects=[
        _paper("paper-object-1", [0, 0, 45, 40]),
        _paper("paper-object-2", [50, 0, 99, 40]),
    ])

    resolved, details = resolver.resolve(scene, image_path, ["area A"])

    assert details["status"] == "NOT_FOUND"
    assert details["ambiguous_queries"] == ["area A"]
    assert all(item.label == "white rectangular paper" for item in resolved.objects)


def test_pipeline_routes_area_query_to_paper_then_restores_semantic_label(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class Backend:
        def __init__(self):
            self.queries = None
            self.latest_scene = SceneModel()

        def detect_objects(self, queries, min_confidence, require_pose):
            self.queries = queries
            scene = SceneModel(objects=[
                SceneObject(
                    object_id="nut_1", label="nut", confidence=0.9,
                    pose_available=True,
                    pose=ObjectPose(frame="base_link", x=0.3, y=0.0, z=0.01),
                    metadata={"query": "nut"},
                ),
                _paper("paper-object-1", [10, 10, 90, 70]),
            ])
            self.latest_scene = scene
            return {
                "status": "SUCCESS", "message": "localized", "image_path": str(image_path),
                "objects": [item.model_dump() for item in scene.objects],
                "scene": scene.model_dump(),
                "query_results": [
                    {"query": "nut", "status": "SUCCESS", "message": "found", "object_count": 1},
                    {"query": "white rectangular paper", "status": "SUCCESS", "message": "found", "object_count": 1},
                ],
            }

    backend = Backend()
    llm = _FakeVisionLLM([
        {"candidate_id": "paper_1", "label": "A", "readable": True},
    ])
    resolver = LabeledRegionResolver(LabeledRegionVisionClient(llm))
    pipeline = PerceptionPipeline(
        backend,
        labeled_region_resolver=resolver,
        labeled_region_config={
            "enabled": True, "visual_query": "white rectangular paper",
        },
    )

    result = pipeline.detect_objects(["nut", "area A"])

    assert backend.queries == ["nut", "white rectangular paper"]
    assert result["success"] is True
    assert [item["query"] for item in result["query_results"]] == ["nut", "area A"]
    assert pipeline.get_latest_scene().find_by_label("area A")[0].object_id == "paper-object-1"
    assert backend.latest_scene.find_by_label("area A")
    place = RelativePlacePosePlanner().compute(
        pipeline.get_latest_scene(),
        source_id="nut",
        relation="on_top_of",
        reference_ids=["area A"],
    )
    assert place["reference_object_ids"] == ["paper-object-1"]


def test_pipeline_fails_localization_when_label_reading_is_not_unique(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    class Backend:
        def detect_objects(self, queries, min_confidence, require_pose):
            scene = SceneModel(objects=[
                _paper("paper-object-1", [0, 0, 45, 40]),
                _paper("paper-object-2", [50, 0, 99, 40]),
            ])
            return {
                "status": "SUCCESS", "message": "localized", "image_path": str(image_path),
                "objects": [item.model_dump() for item in scene.objects],
                "scene": scene.model_dump(),
                "query_results": [{
                    "query": "white rectangular paper", "status": "SUCCESS",
                    "message": "found", "object_count": 2,
                }],
            }

    llm = _FakeVisionLLM([
        {"candidate_id": "paper_1", "label": "A", "readable": True},
        {"candidate_id": "paper_2", "label": "A", "readable": True},
    ])
    pipeline = PerceptionPipeline(
        Backend(),
        labeled_region_resolver=LabeledRegionResolver(LabeledRegionVisionClient(llm)),
        labeled_region_config={
            "enabled": True, "visual_query": "white rectangular paper",
        },
    )

    result = pipeline.detect_objects(["area A"])

    assert result["success"] is False
    assert result["status"] == "NOT_FOUND"
    assert result["query_results"][0]["object_count"] == 0
    assert "exactly one match" in result["message"]
