"""Run the constrained Agent runtime against the existing RViz/fake-hardware stack.

This entry point deliberately refuses real-robot execution.  It reuses the
perception, GraspGenX, MoveIt planning and RViz gripper adapters that existed
before the Agent runtime, but lets the Agent execute exactly one reviewed
Primitive Tool at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_skill_robot.agent.agent_controller import AgentController
from llm_skill_robot.agent.agent_runtime import AgentRuntime
from llm_skill_robot.agent.agent_session import AgentSessionRunner
from llm_skill_robot.agent.agent_state import AgentState
from llm_skill_robot.agent.llm_client import (
    LLMClientConfigurationError,
    OpenAILLMClient,
    load_llm_config,
)
from llm_skill_robot.agent.rviz_agent_factory import (
    create_rviz_agent_resources,
    validate_rviz_execution_config,
)
from llm_skill_robot.agent_runtime_demo import ScriptedDemoLLMClient
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.execution_mode import is_execution_allowed, load_execution_config
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.ros_rviz_sim_execution_demo import _load_safety_config


RVIZ_AGENT_WARNING = (
    "RVIZ AGENT SIMULATION ONLY — real-robot execution is refused by this entry point."
)


def _create_llm_client(scripted: bool):
    if scripted:
        return ScriptedDemoLLMClient()
    config = load_llm_config().get("llm", {})
    return OpenAILLMClient(
        model=config.get("model"),
        temperature=config.get("temperature"),
        max_retries=config.get("max_retries"),
        timeout_sec=config.get("timeout_sec"),
    )


def _approve(label: str, *, auto_approve: bool, state: AgentState) -> bool:
    if auto_approve:
        print(f"[auto-approved] {label}")
        return True
    answer = input(f"\n{label}\nType APPROVE to continue:\n> ").strip()
    state.human_messages.append(answer)
    return answer == "APPROVE"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RViz fake-hardware Agent runtime demo")
    parser.add_argument("--scripted", action="store_true", help="Use deterministic no-network Agent decisions.")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all HITL gates.")
    parser.add_argument("--goal", help="Robot task goal; defaults to an interactive prompt.")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--audit-log", default="ros_agent_runtime_rviz_sim_audit.jsonl")
    parser.add_argument(
        "--mock-perception",
        action="store_true",
        help="Use deterministic object, point-cloud and grasp observations; do not wait for a camera.",
    )
    args = parser.parse_args(argv)

    goal = args.goal or input("Enter an RViz robot task:\n> ").strip()
    if not goal:
        print("A non-empty task goal is required.")
        return

    audit_logger = AuditLogger(args.audit_log)
    resources = None
    node = None
    rclpy = None
    try:
        try:
            import rclpy as imported_rclpy
        except ImportError:
            print("Error: rclpy not available. Source ROS 2 before running this demo.", file=sys.stderr)
            return
        rclpy = imported_rclpy

        execution_config = load_execution_config().model_dump()
        if not is_execution_allowed("rviz_sim", execution_config):
            print("Execution config does not allow rviz_sim mode. Exiting safely.")
            return
        execution_config["mode"] = "rviz_sim"
        validate_rviz_execution_config(execution_config)

        llm_client = _create_llm_client(args.scripted)
        rclpy.init()
        node = rclpy.create_node("ros_agent_runtime_rviz_sim_demo")
        resources = create_rviz_agent_resources(
            node,
            execution_config=execution_config,
            safety_config=_load_safety_config(),
            audit_logger=audit_logger,
            mock_perception=args.mock_perception,
        )
        assert resources.skill_executor is not None

        def primitive_human_gate(plan, _proposal, state: AgentState):
            approved = _approve(
                f"Primitive Tool approval required: {plan.steps[0].skill_id}",
                auto_approve=args.auto_approve,
                state=state,
            )
            if approved and plan.steps[0].skill_id == "review_grasp_candidate":
                state.context["grasp_candidate_confirmation"] = "YES"
            return approved

        runtime = AgentRuntime(
            AgentController(llm_client),
            PlanVerifier(SkillLibrary()),
            resources.skill_executor,
            human_gate=primitive_human_gate,
        )

        def composite_human_gate(skill, state: AgentState):
            return _approve(
                f"Composite Skill approval required: {skill.skill_name}",
                auto_approve=args.auto_approve,
                state=state,
            )

        state = AgentState(user_goal=goal, safety_mode="rviz_sim")
        print(RVIZ_AGENT_WARNING)
        if args.scripted:
            print("SCRIPTED AGENT — no LLM API is used; RViz/fake-hardware resources remain active.")
        if args.mock_perception:
            print("MOCK PERCEPTION — camera, point-cloud and GraspGenX inputs are deterministic.")
        runner = AgentSessionRunner(
            runtime, composite_human_gate=composite_human_gate, audit_logger=audit_logger
        )
        result = runner.run(state, max_turns=args.max_turns)
        print("\n=== Agent Session Result ===")
        print(json.dumps(result.model_dump(), indent=2, default=str))
        print("\n=== Agent State ===")
        print(json.dumps(state.model_dump(), indent=2, default=str))
    except LLMClientConfigurationError as exc:
        print(f"LLM client is not configured: {exc}")
        print("Use --scripted for a reproducible no-network Agent decision path.")
    except Exception as exc:
        print(f"RViz Agent demo failed safely: {exc}", file=sys.stderr)
    finally:
        if resources is not None:
            resources.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy is not None:
            try:
                rclpy.shutdown()
            except Exception:
                pass
        print(f"Audit log: {Path(args.audit_log).resolve()}")


if __name__ == "__main__":
    main()
