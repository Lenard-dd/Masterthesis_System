"""Interactive mock demo for the constrained mixed-granularity Agent runtime.

The demo deliberately uses no ROS or hardware.  It exercises the same
``AgentSessionRunner -> AgentRuntime -> PlanVerifier -> ExecutionManager``
path used by later RViz and real-hardware adapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_skill_robot.agent.agent_controller import AgentController
from llm_skill_robot.agent.agent_runtime import AgentRuntime
from llm_skill_robot.agent.agent_session import AgentSessionRunner
from llm_skill_robot.agent.agent_state import AgentState
from llm_skill_robot.agent.llm_client import (
    LLMClientConfigurationError,
    OpenAILLMClient,
    load_llm_config,
)
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.execution_manager import ExecutionManager
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.skill_library import SkillLibrary


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


class AgentDemoMockBackend:
    """Mock implementations of the existing primitive skill backend methods."""

    def __init__(self) -> None:
        self.current_target: str | None = None
        self.gripper_open = False

    @staticmethod
    def _ok(message: str, **output: Any) -> dict[str, Any]:
        return {"success": True, "message": message, "output": output}

    def move_to_named_target(self, target: str, **_kwargs):
        self.current_target = target
        return self._ok(f"Mock arm moved to named target {target}.", target=target)

    def detect_object(self, query: str, **_kwargs):
        return self._ok(
            f"Mock perception detected {query}.",
            object_id="mock_cube_1",
            query=query,
        )

    def build_object_point_cloud(self, object_id: str, **_kwargs):
        return self._ok(
            "Mock point cloud built.", object_id=object_id, point_cloud_path="mock://cube_1"
        )

    def generate_grasp_pose(self, object_id: str, **_kwargs):
        candidate = {
            "candidate_id": "mock_grasp_1",
            "object_id": object_id,
            "score": 0.95,
            "pose": {"frame": "base_link"},
        }
        return self._ok(
            "Mock GraspGenX candidate generated.",
            candidate_id="mock_grasp_1",
            candidate=candidate,
        )

    def open_gripper(self, **_kwargs):
        self.gripper_open = True
        return self._ok("Mock gripper opened.", gripper_open=True)

    def close_gripper(self, **_kwargs):
        self.gripper_open = False
        return self._ok("Mock gripper closed.", gripper_open=False)

    def plan_to_pregrasp(self, object_id: str, **_kwargs):
        return self._ok("Mock pregrasp planned.", object_id=object_id)

    def plan_approach_grasp(self, object_id: str, **_kwargs):
        return self._ok("Mock grasp approach planned.", object_id=object_id)

    def plan_retreat_grasp(self, object_id: str, **_kwargs):
        return self._ok("Mock grasp retreat planned.", object_id=object_id)

    def verify_grasp(self, object_id: str, **_kwargs):
        return self._ok("Mock grasp verified.", object_id=object_id, grasp_verified=True)


def _print_human_gate(label: str, *, auto_approve: bool, state: AgentState) -> bool:
    if auto_approve:
        print(f"[auto-approved] {label}")
        return True
    answer = input(f"\n{label}\nType APPROVE to continue:\n> ").strip()
    state.human_messages.append(answer)
    return answer == "APPROVE"


def create_mock_session(
    llm_client,
    *,
    auto_approve: bool = False,
    audit_logger: AuditLogger | None = None,
) -> AgentSessionRunner:
    """Create a hardware-free session runner that uses production safety code."""
    library = SkillLibrary()
    backend = AgentDemoMockBackend()

    def primitive_human_gate(plan, _proposal, state: AgentState):
        approved = _print_human_gate(
            f"Primitive Tool approval required: {plan.steps[0].skill_id}",
            auto_approve=auto_approve,
            state=state,
        )
        if approved and plan.steps[0].skill_id == "review_grasp_candidate":
            state.context["grasp_candidate_confirmation"] = "YES"
        return approved

    runtime = AgentRuntime(
        AgentController(llm_client),
        PlanVerifier(library),
        ExecutionManager(library, backend),
        human_gate=primitive_human_gate,
    )

    def composite_human_gate(skill, state: AgentState):
        return _print_human_gate(
            f"Composite Skill approval required: {skill.skill_name}",
            auto_approve=auto_approve,
            state=state,
        )

    return AgentSessionRunner(
        runtime,
        composite_human_gate=composite_human_gate,
        audit_logger=audit_logger,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Constrained Agent Runtime mock demo")
    parser.add_argument("--scripted", action="store_true", help="Use deterministic no-API decisions.")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve demo HITL gates.")
    parser.add_argument("--goal", help="Task goal; defaults to an interactive prompt.")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--audit-log", default="agent_runtime_demo_audit.jsonl")
    args = parser.parse_args(argv)

    goal = args.goal or input("Enter a mock robot task:\n> ").strip()
    if not goal:
        print("A non-empty task goal is required.")
        return
    if args.scripted:
        llm_client = ScriptedDemoLLMClient()
        print("SCRIPTED MOCK AGENT DEMO — no LLM API and no hardware are used.")
    else:
        try:
            config = load_llm_config().get("llm", {})
            llm_client = OpenAILLMClient(
                model=config.get("model"),
                temperature=config.get("temperature"),
                max_retries=config.get("max_retries"),
                timeout_sec=config.get("timeout_sec"),
            )
        except LLMClientConfigurationError as exc:
            print(f"LLM client is not configured: {exc}")
            print("Use --scripted to run the no-network reproducible demo.")
            return
        print("LIVE LLM MOCK AGENT DEMO — no ROS or hardware are used.")

    logger = AuditLogger(args.audit_log)
    state = AgentState(user_goal=goal, safety_mode="mock")
    runner = create_mock_session(
        llm_client, auto_approve=args.auto_approve, audit_logger=logger
    )
    result = runner.run(state, max_turns=args.max_turns)

    print("\n=== Agent Session Result ===")
    print(json.dumps(result.model_dump(), indent=2))
    print("\n=== Agent State ===")
    print(json.dumps(state.model_dump(), indent=2, default=str))
    print(f"\nAudit log: {Path(args.audit_log).resolve()}")


if __name__ == "__main__":
    main()
