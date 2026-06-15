"""Interactive single-image test for the SAM3 subprocess adapter."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from llm_skill_robot.perception.sam3_subprocess_adapter import SAM3SubprocessAdapter


def _load_perception_config() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / "config" / "perception.yaml",
        repo_root / "config" / "perception.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


def main() -> None:
    config = _load_perception_config()
    image_path = input("Image path: ").strip()
    query = input("Query: ").strip()
    min_confidence = (
        config.get("perception", {}).get("min_confidence_default", 0.5)
        if isinstance(config, dict)
        else 0.5
    )

    adapter = SAM3SubprocessAdapter(config)
    result = adapter.detect(
        rgb_image_path=image_path,
        query=query,
        min_confidence=float(min_confidence),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
