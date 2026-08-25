import json

from llm_skill_robot.perception.scene_understanding import (
    SceneDescriptionSkillBackend,
    SceneUnderstandingClient,
)


class _FakeVisionClient:
    def __init__(self, response: str):
        self.response = response
        self.messages = None

    def generate_text(self, messages):
        self.messages = messages
        return self.response


class _CaptureCamera:
    def __init__(self, image_path):
        self.image_path = image_path

    def capture_rgb_image(self):
        return {"status": "SUCCESS", "message": "saved", "path": str(self.image_path)}


class _Pipeline:
    def __init__(self, image_path):
        self.backend = _CaptureCamera(image_path)


def _scene_json():
    return json.dumps({
        "scene_type": "tabletop_workspace",
        "summary": "A tabletop contains a black cube.",
        "candidate_objects": [{
            "query": "black cube",
            "description": "A small black cube near the center.",
            "manipulable": True,
            "confidence": 0.93,
        }],
        "hazards": ["table edge at the bottom of the image"],
    })


def test_scene_understanding_sends_image_and_validates_json(tmp_path):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png-data")
    fake_client = _FakeVisionClient(_scene_json())
    client = SceneUnderstandingClient(fake_client, image_detail="low")

    description = client.describe_image(image_path, max_candidates=3)

    assert description.scene_type == "tabletop_workspace"
    assert description.candidate_objects[0].query == "black cube"
    image_part = fake_client.messages[1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["detail"] == "low"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_scene_description_skill_returns_unverified_semantic_result(tmp_path):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png-data")
    backend = SceneDescriptionSkillBackend(
        _Pipeline(image_path), SceneUnderstandingClient(_FakeVisionClient(_scene_json()))
    )

    result = backend.describe_scene(max_candidates=2)

    assert result["success"] is True
    assert result["status"] == "SUCCESS"
    assert result["scene_description"]["candidate_objects"][0]["manipulable"] is True
    assert "unverified" in result["message"]


def test_scene_description_requires_rgb_capture_backend():
    backend = SceneDescriptionSkillBackend(perception_pipeline=object())

    result = backend.describe_scene()

    assert result["success"] is False
    assert result["status"] == "NOT_AVAILABLE"
