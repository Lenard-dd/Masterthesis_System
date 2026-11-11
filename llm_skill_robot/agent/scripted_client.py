"""Deterministic offline Agent client for scripted operation and tests."""

from __future__ import annotations

import json
from typing import Any


class ScriptedDemoLLMClient:
    """Deterministic no-network Agent path for demonstrations and CI."""

    def __init__(self, object_query: str = "red cube"):
        self.responses = [
            {
                "kind": "composite_skill_call",
                "message": "Start the reviewed supervised pick workflow.",
                "composite_skill_call": {
                    "skill_name": "safe_pick_object",
                    "arguments": {"object_query": object_query},
                },
            },
            {"kind": "complete", "message": "The requested pick workflow completed."},
        ]

    def generate_text(self, _messages: list[dict[str, Any]]) -> str:
        if not self.responses:
            return json.dumps({"kind": "complete", "message": "Demo complete."})
        return json.dumps(self.responses.pop(0))
