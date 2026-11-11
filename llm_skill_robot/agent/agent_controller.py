"""Single-step, constrained LLM Agent controller.

The controller proposes one reviewed tool intent at a time.  It does not
execute it: callers must route its ``PlanStep`` through the existing verifier,
HITL gate, state machine, and execution manager.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_skill_robot.agent.agent_state import AgentState, AgentStatus
from llm_skill_robot.agent.composite_skills.base import (
    BaseCompositeSkill,
    CompositeSkillCall,
)
from llm_skill_robot.agent.composite_skills.registry import CompositeSkillRegistry
from llm_skill_robot.agent.tool_registry import (
    AgentToolCall,
    AgentToolRegistry,
    AgentToolResult,
    ToolSafetyTier,
)
from llm_skill_robot.core.plan import PlanStep


class AgentDecisionError(RuntimeError):
    """Raised when the LLM response is not a valid constrained Agent decision."""


class AgentDecisionKind(str, Enum):
    TOOL_CALL = "tool_call"
    COMPOSITE_SKILL_CALL = "composite_skill_call"
    REQUEST_HUMAN = "request_human"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class ApprovalStage(str, Enum):
    """Named external review points; the Agent can request, never bypass them."""

    TASK_INTENT = "task_intent"          # A: task and target are understood.
    GRASP_CANDIDATE = "grasp_candidate"  # B: proposed grasp is acceptable.
    TRAJECTORY = "trajectory"             # C: planned motion is acceptable.
    EXECUTION = "execution"               # D: final motion/actuation release.


class AgentDecision(BaseModel):
    """The only response format accepted from an Agent LLM turn."""

    model_config = ConfigDict(extra="forbid")

    kind: AgentDecisionKind
    message: str = Field(..., min_length=1)
    tool_call: AgentToolCall | None = None
    composite_skill_call: CompositeSkillCall | None = None

    @model_validator(mode="after")
    def _validate_shape(self):
        if self.kind == AgentDecisionKind.TOOL_CALL:
            if self.tool_call is None or self.composite_skill_call is not None:
                raise ValueError("tool_call alone is required when kind is tool_call.")
        elif self.kind == AgentDecisionKind.COMPOSITE_SKILL_CALL:
            if self.composite_skill_call is None or self.tool_call is not None:
                raise ValueError(
                    "composite_skill_call alone is required when kind is composite_skill_call."
                )
        elif self.tool_call is not None or self.composite_skill_call is not None:
            raise ValueError("Calls are allowed only for tool_call or composite_skill_call.")
        return self


class AgentProposal(BaseModel):
    """A validated next action ready for verification, not execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    decision: AgentDecision
    plan_step: PlanStep | None = None
    composite_skill: BaseCompositeSkill | None = None
    safety_tier: ToolSafetyTier | None = None
    requires_human_gate: bool = False
    approval_stages: list[ApprovalStage] = Field(default_factory=list)


class AgentController:
    """Build prompts and validate one next action against the Tool Registry."""

    # Deterministic multilingual intent aliases form a safety boundary before
    # the LLM chooses a tool.  Keep action-specific aliases separate so a
    # Chinese phrase meaning "close the gripper" cannot select ``open_gripper``.
    _PICK_TERMS = (
        "pick", "grasp", "lift", "take", "抓取", "拿起", "拾取", "夹取",
        "抓住", "抓起", "つかむ", "持ち上げる", "greifen", "aufheben",
    )
    _OBSERVE_TERMS = (
        "observe", "detect", "find", "locate", "localize", "localise",
        "localization", "look", "识别", "检测", "定位", "查找", "寻找",
        "查看", "观察", "发现", "検出", "探す", "erkennen", "finden",
    )
    _SCENE_DESCRIPTION_TERMS = (
        "describe scene", "describe the scene", "scan scene", "scene overview", "what is on the table",
        "场景描述", "扫描场景", "识别场景", "环境识别", "场景信息", "桌面上有什么",
    )
    _SCENE_SCAN_VERBS = (
        "scan", "describe", "analyse", "analyze", "summarize", "inspect",
        "扫描", "描述", "分析", "查看",
    )
    _SCENE_NOUNS = (
        "scene", "scenario", "senario", "workspace", "environment", "tabletop", "table",
        "场景", "环境", "桌面", "工作区",
    )
    _PLACE_TERMS = (
        "place", "put", "drop", "放置", "放下", "摆放", "置于", "置く",
        "置いて", "ablegen", "platzieren",
    )
    _SPATIAL_RELATION_TERMS = (
        "between", "middle", "right of", "right_of", "left of", "left_of",
        "on top of", "on_top_of", "above", "over", "stack",
        "中间", "右边", "左边", "上方", "正上方", "叠放",
    )
    _MOVE_TERMS = (
        "move", "go home", "return home", "回到", "移动", "回原点", "回家",
        "前往", "移動", "戻る", "bewegen", "zurück",
    )
    _OPEN_GRIPPER_TERMS = (
        "open gripper", "open the gripper", "open claw", "release gripper",
        "打开夹爪", "张开夹爪", "松开夹爪", "打开夹具", "张开夹具", "松开夹具",
        "打开夹子", "グリッパーを開く", "グリッパーを開いて", "greifer öffnen",
    )
    _CLOSE_GRIPPER_TERMS = (
        "close gripper", "close the gripper", "close claw", "clamp gripper",
        "关闭夹爪", "闭合夹爪", "合上夹爪", "合拢夹爪", "夹紧夹爪",
        "关闭夹具", "闭合夹具", "合上夹具", "合拢夹具", "夹紧夹具",
        "关闭夹子", "闭合夹子", "グリッパーを閉じる", "グリッパーを閉じて",
        "greifer schließen",
    )
    _GENERIC_GRIPPER_TERMS = ("gripper", "claw", "夹爪", "夹具", "夹子", "グリッパー", "greifer")

    SYSTEM_PROMPT = (
        "You are a constrained robot task agent. Decide exactly one next action. "
        "You may only use tools supplied in the tool catalog. Never output ROS, "
        "shell, driver, trajectory, or joint commands. Return only JSON matching "
        "the AgentDecision schema. Output exactly one JSON object: no Markdown, "
        "no code fence, no explanation, and no additional top-level keys. "
        "For kind='composite_skill_call', skill_name and arguments MUST be nested "
        "inside composite_skill_call; never put them at the top level. "
        "For kind='tool_call', tool_name and arguments MUST be nested inside "
        "tool_call. A tool proposal is not permission to execute; verification "
        "and human safety gates run outside the model."
    )

    def __init__(
        self,
        llm_client,
        tool_registry: AgentToolRegistry | None = None,
        composite_skill_registry: CompositeSkillRegistry | None = None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry or AgentToolRegistry()
        self.composite_skill_registry = composite_skill_registry or CompositeSkillRegistry()

    def propose_next_action(self, state: AgentState) -> AgentProposal:
        """Ask the model for one action and return a validated plan-only proposal."""
        if state.status != AgentStatus.ACTIVE:
            raise AgentDecisionError(
                f"Cannot decide next action while agent is {state.status.value}."
            )
        # Scene scanning is read-only and has a deliberately narrow argument
        # schema. Route clear near-matches deterministically so common wording
        # such as "scan the scenario" (or its frequent typo "senario") does
        # not depend on an LLM choosing the exact registered phrase.
        if self._scene_description_requested(state.user_goal):
            return self.prepare_tool_call(
                state,
                AgentToolCall(tool_name="describe_scene", arguments={}),
                "I interpret this as a request to describe the current scene. "
                "I will run a read-only scene scan now.",
            )
        relative_place_call = self._direct_relative_place_call(state.user_goal)
        if relative_place_call is not None:
            return self.prepare_tool_call(
                state,
                relative_place_call,
                "I will compute the requested plan-only relative placement pose "
                "from the current verified localization results.",
            )
        response = self.llm_client.generate_text(self._build_messages(state))
        decision = self._parse_decision(response)
        decision = self._apply_goal_policy(state, decision)

        if decision.kind == AgentDecisionKind.REQUEST_HUMAN:
            state.request_human_input(decision.message)
            return AgentProposal(
                decision=decision,
                requires_human_gate=True,
                approval_stages=[ApprovalStage.TASK_INTENT],
            )
        if decision.kind == AgentDecisionKind.COMPLETE:
            state.status = AgentStatus.COMPLETED
            return AgentProposal(decision=decision)
        if decision.kind == AgentDecisionKind.BLOCKED:
            state.status = AgentStatus.BLOCKED
            return AgentProposal(decision=decision)
        if decision.kind == AgentDecisionKind.COMPOSITE_SKILL_CALL:
            assert decision.composite_skill_call is not None
            try:
                composite_skill = self.composite_skill_registry.create(
                    decision.composite_skill_call
                )
            except (TypeError, ValueError) as exc:
                raise AgentDecisionError(
                    f"Invalid composite skill request: {exc}"
                ) from exc
            state.active_composite_skill = decision.composite_skill_call.skill_name
            state.context["active_composite_skill"] = {
                "skill_name": decision.composite_skill_call.skill_name,
                "arguments": decision.composite_skill_call.arguments,
            }
            return self._with_approval_policy(state, AgentProposal(
                decision=decision,
                composite_skill=composite_skill,
                safety_tier=ToolSafetyTier.HUMAN_REVIEW,
                requires_human_gate=True,
            ))

        assert decision.tool_call is not None
        return self.prepare_tool_call(state, decision.tool_call, decision.message)

    def prepare_tool_call(
        self,
        state: AgentState,
        call: AgentToolCall,
        message: str = "Tool call prepared by a constrained workflow.",
    ) -> AgentProposal:
        """Validate one non-LLM tool intent for a Composite Skill runner.

        Composite skills may choose only their fixed, reviewed primitive
        sequence.  They still use this method so their calls receive the same
        schema validation and risk classification as LLM-proposed calls.
        """
        if state.status != AgentStatus.ACTIVE:
            raise AgentDecisionError(
                f"Cannot prepare a tool call while agent is {state.status.value}."
            )
        decision = AgentDecision(
            kind=AgentDecisionKind.TOOL_CALL,
            message=message,
            tool_call=call,
        )
        tool = self.tool_registry.get(call.tool_name)
        if tool is None:
            raise AgentDecisionError(f"Requested unknown Agent tool: {call.tool_name}")
        try:
            plan_step = tool.to_plan_step(call, state.next_step_id)
        except (TypeError, ValueError) as exc:
            raise AgentDecisionError(f"Invalid arguments for {tool.name}: {exc}") from exc
        state.record_proposal(call)
        requires_human_gate = tool.safety_tier in {
            ToolSafetyTier.HUMAN_REVIEW,
            ToolSafetyTier.HUMAN_CONFIRM,
        }
        return self._with_approval_policy(state, AgentProposal(
            decision=decision,
            plan_step=plan_step,
            safety_tier=tool.safety_tier,
            requires_human_gate=requires_human_gate,
        ))

    def record_tool_result(self, state: AgentState, result: Any) -> AgentToolResult:
        """Normalize trusted backend feedback and make it available next turn."""
        if not state.proposed_calls:
            raise AgentDecisionError("No proposed tool call is awaiting a result.")
        tool = self.tool_registry.get(state.proposed_calls[-1].tool_name)
        assert tool is not None  # Calls are validated before being recorded.
        normalized = tool.normalize_result(result)
        state.record_tool_result(normalized)
        return normalized

    def _build_messages(self, state: AgentState) -> list[dict[str, str]]:
        allowed_actions = sorted(self._allowed_actions_for_goal(state))
        prompt = {
            "agent_state": state.model_dump(),
            "tool_catalog": [tool.model_context() for tool in self.tool_registry.list_tools()],
            "composite_skill_catalog": self.composite_skill_registry.list_skills(),
            "decision_schema": AgentDecision.model_json_schema(),
            "policy": {
                "one_tool_per_turn": True,
                "composite_skills": (
                    "Use safe_pick_object only when the user's goal explicitly asks "
                    "to pick, grasp, lift, or take an object. It is a fixed supervised "
                    "workflow; do not recreate its internal grasp stages as LLM calls."
                ),
                "goal_routing": (
                    "Choose only an action that matches the user goal and the allowed "
                    "next actions. For scene-description requests use describe_scene; "
                    "for precise localization of one object use detect_object with "
                    "a single query. When the user asks to localize, find, or detect "
                    "two or more distinct objects, use detect_objects exactly once "
                    "with arguments.queries containing one exact query per object. "
                    "Do not split a multi-object localization request into several "
                    "detect_object calls. When a request explicitly combines "
                    "localization with a relative placement computation, choose "
                    "detect_objects first; the GUI will run the dependent plan-only "
                    "compute_place_pose stage only after trusted localization succeeds. "
                    "For a relative placement request with already-localized objects, "
                    "use compute_place_pose. "
                    "Convert the requested relation to exactly one of between, "
                    "right_of, left_of, or on_top_of; use localized object IDs or "
                    "unambiguous labels. This tool only computes a pose and does not "
                    "place an object. For a "
                    "named target movement use move_to_named_target. For a task outside "
                    "the available capabilities, or for an ambiguous target, return "
                    "kind='request_human' rather than selecting a grasp skill."
                ),
                "target_grounding": (
                    "For every object query, use the object description from the user's "
                    "goal. Never invent or reuse a demonstration object such as red cube."
                ),
                "mixed_workflows": (
                    "A completed Composite Skill does not end a task automatically. On "
                    "the next turn, inspect trusted state and choose a permitted primitive "
                    "tool when it is needed to finish the remaining goal."
                ),
                "completed_workflows": (
                    "completed_composite_skills is trusted execution history. If it "
                    "already contains the requested skill_name with the same arguments, "
                    "return kind='complete' and do not propose that workflow again."
                ),
                "human_review_tools_require_external_gate": [
                    ToolSafetyTier.HUMAN_REVIEW.value,
                    ToolSafetyTier.HUMAN_CONFIRM.value,
                ],
                "on_uncertainty": "request_human or choose a safe observation tool",
            },
            "allowed_next_actions": allowed_actions,
            "approval_stages": {
                "A_task_intent": "Confirm task intent and target before a task workflow.",
                "B_grasp_candidate": "Confirm the selected grasp candidate.",
                "C_trajectory": "Confirm the planned trajectory before motion.",
                "D_execution": "Final release before actuation or execution.",
            },
        }
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
        ]

    @staticmethod
    def _normalized_goal(goal: str) -> str:
        return " ".join(goal.lower().split())

    def _allowed_actions_for_goal(self, state: AgentState) -> set[str]:
        """Return the safe next actions justified by the user's stated goal.

        This is intentionally conservative. It supplies a deterministic safety
        boundary around the LLM rather than trusting a prompt alone to avoid an
        unrelated grasp request.
        """
        goal = self._normalized_goal(state.user_goal)
        actions: set[str] = set()
        scene_description_requested = self._scene_description_requested(goal)
        pick_requested = self._contains_any(goal, self._PICK_TERMS)
        localization_requested = self._contains_any(goal, self._OBSERVE_TERMS)
        relative_place_requested = (
            self._contains_any(goal, self._PLACE_TERMS)
            and self._contains_any(goal, self._SPATIAL_RELATION_TERMS)
        )
        if pick_requested:
            # The Composite Skill owns its internal grasp sequence. Additional
            # primitives are exposed only when separately requested by the
            # user's task (for example, "observe, then pick, then go home").
            # Keep safe_pick_object available after completion as well: the
            # session runner has a dedicated duplicate-workflow safeguard.
            actions.add("safe_pick_object")
        if scene_description_requested:
            actions.add("describe_scene")
        if localization_requested and not scene_description_requested:
            # The LLM receives both localization schemas and selects the
            # one-object or multi-object form from the user's wording.
            # ``move_to_named_target`` is retained for existing reviewed
            # observe-then-move workflows; it remains subject to verification.
            actions.update({"detect_object", "detect_objects", "move_to_named_target"})
        if self._contains_any(goal, self._PLACE_TERMS):
            if relative_place_requested:
                # Relative placement is plan-only in this phase. A future
                # reviewed pick/place workflow may consume this pose, but the
                # LLM must not jump directly to physical placement.
                # If localization was explicitly requested in the same
                # instruction, it is the mandatory first stage.  The GUI adds
                # compute_place_pose as a dependency after localization; do
                # not let the LLM skip directly to metric computation using
                # stale or absent object coordinates.
                if not localization_requested:
                    actions.add("compute_place_pose")
            else:
                actions.add("place_object")
        if localization_requested and relative_place_requested:
            # A relative relation always needs a source plus one or two
            # references, so force the first Agent turn to use the single
            # shared-frame multi-object localization tool.
            actions.difference_update({"detect_object", "move_to_named_target"})
            actions.add("detect_objects")
        if self._contains_any(goal, self._MOVE_TERMS):
            actions.add("move_to_named_target")
        open_requested = self._contains_any(goal, self._OPEN_GRIPPER_TERMS)
        close_requested = self._contains_any(goal, self._CLOSE_GRIPPER_TERMS)
        if open_requested:
            actions.add("open_gripper")
        if close_requested:
            actions.add("close_gripper")
        if (
            not open_requested
            and not close_requested
            and self._contains_any(goal, self._GENERIC_GRIPPER_TERMS)
        ):
            actions.update({"open_gripper", "close_gripper", "get_gripper_state"})
        return actions

    @staticmethod
    def _contains_any(goal: str, aliases: tuple[str, ...]) -> bool:
        return any(alias in goal for alias in aliases)

    @classmethod
    def _scene_description_requested(cls, goal: str) -> bool:
        """Recognize scene scans while tolerating ordinary wording mistakes."""
        normalized = cls._normalized_goal(goal)
        if cls._contains_any(normalized, cls._SCENE_DESCRIPTION_TERMS):
            return True
        return (
            cls._contains_any(normalized, cls._SCENE_SCAN_VERBS)
            and cls._contains_any(normalized, cls._SCENE_NOUNS)
        )

    @classmethod
    def _direct_relative_place_call(cls, goal: str) -> AgentToolCall | None:
        """Parse an unambiguous plan-only relative-place request.

        This is deliberately limited to commands that *do not* also request
        localization.  Their arguments are still only labels: the downstream
        planner resolves them against the latest trusted RGB-D scene and fails
        safely if either label is absent or ambiguous.
        """
        normalized = cls._normalized_goal(goal)
        if cls._contains_any(normalized, cls._OBSERVE_TERMS):
            return None
        if not cls._contains_any(normalized, cls._PLACE_TERMS):
            return None

        def clean(value: str) -> str:
            value = value.strip(" .,!?:;")
            return re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE).strip()

        between = re.search(
            r"\b(?:place|put)\s+(?P<source>.+?)\s+"
            r"(?:between|in the middle of)\s+(?P<first>.+?)\s+(?:and|&)\s+(?P<second>[^.!?,;]+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if between:
            source, first, second = (
                clean(between.group("source")),
                clean(between.group("first")),
                clean(between.group("second")),
            )
            if source and first and second:
                return AgentToolCall(
                    tool_name="compute_place_pose",
                    arguments={
                        "source_id": source,
                        "relation": "between",
                        "reference_ids": [first, second],
                    },
                )

        for expression, relation in (
            (r"on top of", "on_top_of"),
            (r"above", "on_top_of"),
            (r"over", "on_top_of"),
            (r"right of", "right_of"),
            (r"left of", "left_of"),
        ):
            match = re.search(
                rf"\b(?:place|put)\s+(?P<source>.+?)\s+{expression}\s+(?P<reference>[^.!?,;]+)",
                normalized,
                flags=re.IGNORECASE,
            )
            if match:
                source, reference = clean(match.group("source")), clean(match.group("reference"))
                if source and reference:
                    return AgentToolCall(
                        tool_name="compute_place_pose",
                        arguments={
                            "source_id": source,
                            "relation": relation,
                            "reference_ids": [reference],
                        },
                    )
        return None

    def _apply_goal_policy(self, state: AgentState, decision: AgentDecision) -> AgentDecision:
        """Reject unsafe or unsupported routing before creating a plan proposal."""
        if decision.kind in {AgentDecisionKind.REQUEST_HUMAN, AgentDecisionKind.COMPLETE, AgentDecisionKind.BLOCKED}:
            return decision

        action_name = (
            decision.composite_skill_call.skill_name
            if decision.composite_skill_call is not None
            else decision.tool_call.tool_name if decision.tool_call is not None else None
        )
        # Preserve the existing strict unknown-tool error instead of silently
        # treating an invented tool as an intent clarification.
        if decision.tool_call is not None and self.tool_registry.get(action_name or "") is None:
            return decision

        allowed = self._allowed_actions_for_goal(state)
        if not allowed:
            return AgentDecision(
                kind=AgentDecisionKind.REQUEST_HUMAN,
                message=(
                    "This request is outside the currently supported robot tasks. "
                    "Please ask to observe an object, pick a specified object, place an "
                    "already-held object, move to a named target, or operate the gripper."
                ),
            )

        if action_name not in allowed:
            return AgentDecision(
                kind=AgentDecisionKind.REQUEST_HUMAN,
                message=(
                    f"The proposed action {action_name!r} does not match the stated task. "
                    "Please clarify the intended supported robot operation."
                ),
            )

        if decision.composite_skill_call is not None:
            query = str(decision.composite_skill_call.arguments.get("object_query", "")).strip()
            goal = self._normalized_goal(state.user_goal)
            default_target_markers = ("red cube", "red", "cube", "红色", "红", "方块")
            if query.lower() == "red cube" and not any(marker in goal for marker in default_target_markers):
                return AgentDecision(
                    kind=AgentDecisionKind.REQUEST_HUMAN,
                    message="The proposed target reused a default example rather than the user's object. Please specify the object to pick.",
                )
        return decision

    def _with_approval_policy(self, state: AgentState, proposal: AgentProposal) -> AgentProposal:
        """Attach A/B/C/D gates based on the reviewed action category."""
        tool_name = proposal.plan_step.skill_id if proposal.plan_step else None
        # Scene description and object localization are read-only perception
        # operations. They cannot command the robot; localization only
        # produces geometry for a later, separately reviewed manipulation.
        if tool_name in {"describe_scene", "detect_object", "detect_objects", "compute_place_pose"}:
            proposal.approval_stages = []
            proposal.requires_human_gate = False
            return proposal

        stages: list[ApprovalStage] = []
        if not state.context.get("task_intent_approved"):
            stages.append(ApprovalStage.TASK_INTENT)

        if tool_name == "review_grasp_candidate":
            stages.append(ApprovalStage.GRASP_CANDIDATE)
        elif tool_name in {"preview_grasp_motion", "move_to_pregrasp", "approach_grasp", "retreat_grasp"}:
            stages.append(ApprovalStage.TRAJECTORY)
        elif tool_name in {"open_gripper", "close_gripper", "pick_object", "place_object", "set_gripper_width", "test_gripper_width"}:
            stages.append(ApprovalStage.EXECUTION)

        proposal.approval_stages = stages
        proposal.requires_human_gate = proposal.requires_human_gate or bool(stages)
        return proposal

    @staticmethod
    def _parse_decision(response: str) -> AgentDecision:
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip() if len(lines) >= 2 else text
        try:
            payload = json.loads(text)
            return AgentDecision.model_validate(
                AgentController._normalize_common_flat_call_shape(payload)
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentDecisionError(f"LLM did not return valid AgentDecision JSON: {exc}") from exc

    @staticmethod
    def _normalize_common_flat_call_shape(payload: Any) -> Any:
        """Safely repair a frequent provider/model JSON nesting mistake.

        The prompt demands nested calls.  This narrow compatibility path only
        moves the two known call fields into their required object; all other
        unexpected fields remain and are rejected by ``extra='forbid'``.
        """
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        kind = normalized.get("kind")
        if (
            kind == AgentDecisionKind.COMPOSITE_SKILL_CALL.value
            and "composite_skill_call" not in normalized
            and "skill_name" in normalized
            and "arguments" in normalized
        ):
            normalized["composite_skill_call"] = {
                "skill_name": normalized.pop("skill_name"),
                "arguments": normalized.pop("arguments"),
            }
        elif (
            kind == AgentDecisionKind.TOOL_CALL.value
            and "tool_call" not in normalized
            and "tool_name" in normalized
            and "arguments" in normalized
        ):
            normalized["tool_call"] = {
                "tool_name": normalized.pop("tool_name"),
                "arguments": normalized.pop("arguments"),
            }
        return normalized
                        