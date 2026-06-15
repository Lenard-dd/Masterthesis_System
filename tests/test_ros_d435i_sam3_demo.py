from llm_skill_robot.ros_d435i_sam3_demo import _print_detection_result
from llm_skill_robot.perception.scene_model import (
    DetectionResult,
    ObjectPose,
    SceneModel,
    SceneObject,
)


def test_print_detection_result_includes_pose_when_available(capsys):
    scene_object = SceneObject(
        object_id="d435i_sam3_1",
        label="red cube",
        confidence=0.95,
        mask_available=True,
        pose_available=True,
        pose=ObjectPose(
            frame="camera_color_optical_frame",
            x=0.12,
            y=-0.04,
            z=0.63,
        ),
        source="d435i_sam3",
        metadata={
            "bbox_xyxy": [1, 2, 3, 4],
            "mask_path": "mask.png",
            "overlay_path": "overlay.png",
        },
    )
    result = DetectionResult(
        query="red cube",
        status="SUCCESS",
        objects=[scene_object],
        message="detected",
        scene=SceneModel(
            frame="camera_color_optical_frame",
            source="d435i_sam3",
            objects=[scene_object],
        ),
    )

    _print_detection_result(result)

    output = capsys.readouterr().out
    assert "pose_available: True" in output
    assert "pose:" in output
    assert "frame: camera_color_optical_frame" in output
    assert "x: 0.12" in output
    assert "y: -0.04" in output
    assert "z: 0.63" in output
