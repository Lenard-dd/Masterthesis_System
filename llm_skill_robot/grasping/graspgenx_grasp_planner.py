"""GraspGenX candidate generation and deterministic best-candidate selection."""

from __future__ import annotations

import math
from typing import Any


class GraspGenXGraspPlanner:
    def __init__(self, adapter):
        self.adapter = adapter

    def generate_candidates(self, point_cloud_path: str) -> dict:
        return self.adapter.generate_candidates(point_cloud_path)

    def select_best_candidate(self, candidates: list[Any]) -> Any | None:
        valid = [candidate for candidate in candidates if self._is_valid(candidate)]
        if not valid:
            return None
        return max(valid, key=lambda candidate: self._score(candidate))

    @classmethod
    def _is_valid(cls, candidate: Any) -> bool:
        if cls._value(candidate, "valid", True) is not True:
            return False
        return math.isfinite(cls._score(candidate))

    @classmethod
    def _score(cls, candidate: Any) -> float:
        try:
            return float(cls._value(candidate, "score", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    @staticmethod
    def _value(candidate: Any, name: str, default: Any = None) -> Any:
        return candidate.get(name, default) if isinstance(candidate, dict) else getattr(candidate, name, default)
