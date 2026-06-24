"""Interactive perception pipeline skeleton demo."""

from llm_skill_robot.perception.perception_pipeline import PerceptionPipeline
from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode


NOTE = "This demo uses UnavailablePerceptionBackend until D435i/SAM3 is connected."


def _print_detection_result(result: dict) -> None:
    scene = result.get("scene") or {}
    objects = result.get("objects") or []

    print("\nDetectionResult:")
    print(f"status: {result.get('status')}")
    print(f"message: {result.get('message')}")
    print(f"objects: {objects}")
    print("\nScene summary:")
    print(f"frame: {scene.get('frame')}")
    print(f"source: {scene.get('source')}")
    print(f"object_count: {len(scene.get('objects') or [])}")


def main() -> None:
    print_hardware_mode(load_hardware_mode())
    print(NOTE)
    pipeline = PerceptionPipeline()
    query = input("\nEnter object query:\n> ").strip()
    result = pipeline.detect_object(query=query)
    _print_detection_result(result)


if __name__ == "__main__":
    main()
