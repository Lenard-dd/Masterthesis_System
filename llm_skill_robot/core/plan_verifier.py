import yaml
from pathlib import Path
from typing import Dict, Any
from .skill_library import SkillLibrary
from .plan import RobotPlan
from .schemas import VerificationResult, VerificationDecision
from ..perception.scene_model import SceneModel, SceneObject
from ..utils import get_config_dir
from ..safety.hardware_mode import (
    HardwareMode,
    clamp_or_reject_gripper_command,
)
from ..safety.real_arm_safety import load_real_arm_safety, validate_motion_summary


class PlanVerifier:
    REAL_GRIPPER_SKILLS = {
        "open_gripper", "close_gripper", "set_gripper_width", "test_gripper_width"
    }
    CANDIDATE_GRIPPER_SKILLS = {"set_gripper_width", "test_gripper_width"}
    PLAN_ONLY_ARM_SKILLS = {
        "move_to_named_target",
        "move_to_pregrasp",
        "approach_grasp",
        "retreat_grasp",
        "preview_grasp_motion",
    }
    REAL_ARM_EXECUTION_SKILLS = {
        "execute_trajectory",
        "execute_arm_motion",
        "move_real_arm",
    }
    RESTRICTED_REAL_ARM_BLOCKED_SKILLS = {
        "approach_grasp",
        "lift_object",
        "pick_object",
        "execute_pick_sequence",
    }

    def __init__(
        self,
        skill_library: SkillLibrary,
        config_dir: str = None,
        hardware_mode: HardwareMode | None = None,
    ):
        self.skill_library = skill_library
        if config_dir is None:
            self.config_dir = get_config_dir()
        else:
            self.config_dir = Path(config_dir)
        self.robot_capabilities = self.load_robot_capabilities()
        self.hardware_mode = hardware_mode

    def load_robot_capabilities(self) -> Dict[str, Any]:
        """Load robot capabilities from config."""
        capabilities_file = self.config_dir / "robot_capabilities.yaml"
        if capabilities_file.exists():
            with open(capabilities_file, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def verify_plan(self, plan: RobotPlan, context: Dict[str, Any] = None) -> VerificationResult:
        """Verify a robot plan against skills and capabilities."""
        context = context or {}
        reasons = []
        warnings = []
        hardware_mode = self._hardware_mode_from_context(context) or self.hardware_mode

        for step in plan.steps:
            skill = self.skill_library.get_skill(step.skill_id)
            if not skill:
                if (
                    hardware_mode is not None
                    and self._restricted_real_motion_active(hardware_mode)
                    and step.skill_id in self.RESTRICTED_REAL_ARM_BLOCKED_SKILLS
                ):
                    reasons.append(
                        f"{step.skill_id} is blocked in restricted_real_motion mode."
                    )
                reasons.append(f"Unknown skill: {step.skill_id}")
                continue

            # Check required parameters
            for param in skill.parameters:
                value = self._get_step_parameter(step, param.name)
                if value is None:
                    if (
                        step.skill_id == "pick_object"
                        and param.name == "object_id"
                        and step.parameters.get("query") is not None
                    ):
                        continue
                    if param.required:
                        reasons.append(
                            f"Missing required parameter '{param.name}' "
                            f"for skill '{step.skill_id}'"
                        )
                    continue

                # Check type
                if not self.check_parameter_type(value, param.type):
                    reasons.append(
                        f"Invalid type for parameter '{param.name}' in "
                        f"skill '{step.skill_id}': expected {param.type}"
                    )
                    continue
                # Check allowed values
                if param.allowed_values and value not in param.allowed_values:
                    reasons.append(
                        f"Invalid value '{value}' for parameter '{param.name}' "
                        f"in skill '{step.skill_id}': allowed {param.allowed_values}"
                    )
                if param.min_value is not None and value < param.min_value:
                    reasons.append(
                        f"Invalid value '{value}' for parameter '{param.name}' "
                        f"in skill '{step.skill_id}': minimum {param.min_value}"
                    )
                if param.max_value is not None and value > param.max_value:
                    reasons.append(
                        f"Invalid value '{value}' for parameter '{param.name}' "
                        f"in skill '{step.skill_id}': maximum {param.max_value}"
                    )

            if step.skill_id == "detect_object":
                self._verify_detect_object_context(context, warnings)
            elif step.skill_id == "pick_object":
                if hardware_mode is not None:
                    reasons.append(
                        "pick_object direct execution is blocked in arm-safe hardware mode."
                    )
                else:
                    self._verify_pick_object_context(step, context, reasons)
            elif step.skill_id == "place_object":
                self._verify_place_object_context(step, context, reasons, warnings)

            if hardware_mode is not None:
                self._verify_hardware_mode_step(
                    step, hardware_mode, context, reasons, warnings
                )

        # Simple robot capabilities check (placeholder)
        # For now, just check if there are any capabilities defined
        if not self.robot_capabilities:
            warnings.append("No robot capabilities loaded")

        robot_state = context.get("robot_state")
        if robot_state:
            status = robot_state.get("status", {})
            if not status.get("joint_state_received", False):
                warnings.append(status.get("message", "Robot state context has no joint state."))

        decision = VerificationDecision.BLOCK if reasons else VerificationDecision.ALLOW
        return VerificationResult(decision=decision, reasons=reasons, warnings=warnings)

    def _verify_hardware_mode_step(
        self,
        step,
        mode: HardwareMode,
        context: Dict[str, Any],
        reasons: list[str],
        warnings: list[str],
    ) -> None:
        restricted_real_motion = self._restricted_real_motion_active(mode)
        if restricted_real_motion:
            self._verify_restricted_real_motion_step(
                step, context, reasons
            )

        if step.skill_id in self.REAL_ARM_EXECUTION_SKILLS:
            reasons.append("Real arm execution is disabled by hardware_mode.yaml.")
        if step.parameters.get("execution_mode") == "real_robot" or step.parameters.get("execute") is True:
            reasons.append("Real arm execution request is blocked by hardware_mode.yaml.")
        if any(
            step.parameters.get(key) is True
            for key in ("real_arm_execution", "allow_arm_execution", "send_trajectory")
        ):
            reasons.append("Real arm trajectory execution is blocked by hardware_mode.yaml.")
        if step.skill_id in self.PLAN_ONLY_ARM_SKILLS:
            if not mode.arm.allow_plan_only:
                reasons.append("Arm plan-only preview is disabled by hardware_mode.yaml.")
            if step.parameters.get("plan_only") is False:
                if not (
                    restricted_real_motion
                    and step.skill_id in {"move_to_named_target", "move_to_pregrasp"}
                ):
                    reasons.append(f"{step.skill_id} must remain plan-only in arm-safe mode.")
            requested_mode = context.get("arm_execution_mode")
            allowed_requested_modes = {None, "dry_run", "plan_only"}
            if restricted_real_motion and step.skill_id in {
                "move_to_named_target", "move_to_pregrasp"
            }:
                allowed_requested_modes.add("restricted_real_motion")
            if step.skill_id != "move_to_named_target" and requested_mode not in allowed_requested_modes:
                reasons.append(f"{step.skill_id} may use only a plan-only backend.")
            if requested_mode == "real_robot" and not restricted_real_motion:
                reasons.append("Real arm execution is disabled by hardware_mode.yaml.")

        if step.skill_id in self.REAL_GRIPPER_SKILLS:
            width = step.parameters.get("width_m")
            if width is not None:
                try:
                    clamp_or_reject_gripper_command(
                        width_m=width,
                        speed_mps=step.parameters.get("speed_mps", 0.02),
                        force_N=step.parameters.get("force_N", 10.0),
                        mode=mode,
                    )
                except (ValueError, PermissionError) as exc:
                    reasons.append(f"Unsafe real gripper command: {exc}")
            elif step.skill_id == "set_gripper_width":
                reasons.append("set_gripper_width requires a validated width_m.")

            if not self._trusted_confirmation_for_step(step, context):
                reasons.append(
                    f"{step.skill_id} requires exact trusted YES confirmation at execution time."
                )

        later_hardware_stage = step.skill_id in (
            self.CANDIDATE_GRIPPER_SKILLS | self.PLAN_ONLY_ARM_SKILLS
        ) and step.skill_id != "move_to_named_target"
        if later_hardware_stage and not context.get("grasp_candidate_approved", False):
            reasons.append(
                f"{step.skill_id} requires a reviewed and human-approved grasp candidate."
            )

        raw_keys = {
            "ros_command", "command_topic", "action_name", "service_name", "driver_command"
        }.intersection(step.parameters)
        if raw_keys:
            reasons.append(
                "Raw ROS or gripper driver parameters are forbidden: "
                + ", ".join(sorted(raw_keys))
            )

    @staticmethod
    def _trusted_confirmation_for_step(step, context: Dict[str, Any]) -> bool:
        confirmations = context.get("gripper_confirmations", {})
        if isinstance(confirmations, dict) and confirmations.get(str(step.step_id)) == "YES":
            return True
        return context.get("gripper_confirmation") == "YES"

    @classmethod
    def _restricted_real_motion_active(cls, mode: HardwareMode) -> bool:
        return (
            mode.mode == "real_camera_real_gripper_restricted_arm"
            or getattr(mode.arm, "execution_mode", None) == "restricted_real_motion"
        )

    def _verify_restricted_real_motion_step(
        self,
        step,
        context: Dict[str, Any],
        reasons: list[str],
    ) -> None:
        if step.skill_id in self.RESTRICTED_REAL_ARM_BLOCKED_SKILLS:
            reasons.append(
                f"{step.skill_id} is blocked in restricted_real_motion mode."
            )

        if step.skill_id == "close_gripper" and self._is_contact_close_step(step):
            reasons.append(
                "close_gripper during contact or on an object is blocked in restricted_real_motion mode."
            )

        if step.skill_id == "move_to_pregrasp":
            self._verify_restricted_move_to_pregrasp_context(step, context, reasons)

    def _verify_restricted_move_to_pregrasp_context(
        self,
        step,
        context: Dict[str, Any],
        reasons: list[str],
    ) -> None:
        grasp_plan = context.get("grasp_plan") or context.get("selected_grasp_plan")
        pregrasp_pose = self._pregrasp_pose_from_plan(grasp_plan)
        if pregrasp_pose is None:
            reasons.append("move_to_pregrasp requires a valid GraspPlan.")
            return

        frame = self._pose_frame(pregrasp_pose)
        if frame != "base_link":
            reasons.append("move_to_pregrasp requires pregrasp_pose.frame == base_link.")

        summary = (
            context.get("pregrasp_motion_summary")
            or context.get("trajectory_summary")
            or context.get("motion_summary")
        )
        if not isinstance(summary, dict):
            reasons.append("move_to_pregrasp requires a validated trajectory summary.")
        else:
            config = context.get("real_arm_safety")
            if config is None:
                try:
                    config = load_real_arm_safety()
                except Exception as exc:
                    reasons.append(f"Cannot load real_arm_safety.yaml: {exc}")
                    config = None
            if config is not None:
                try:
                    validation = validate_motion_summary(summary, config)
                except ValueError as exc:
                    reasons.append(f"Invalid trajectory summary: {exc}")
                else:
                    if not validation.get("allowed", False):
                        reasons.append(str(validation.get("reason", "Trajectory summary rejected.")))

        if context.get("arm_confirmation_required") is False:
            reasons.append("move_to_pregrasp requires EXECUTE confirmation at execution time.")

    @staticmethod
    def _pregrasp_pose_from_plan(grasp_plan):
        if grasp_plan is None:
            return None
        if isinstance(grasp_plan, dict):
            return (
                grasp_plan.get("pregrasp_pose")
                or grasp_plan.get("pregrasp")
                or grasp_plan.get("pose")
            )
        return (
            getattr(grasp_plan, "pregrasp_pose", None)
            or getattr(grasp_plan, "pregrasp", None)
            or getattr(grasp_plan, "pose", None)
        )

    @staticmethod
    def _pose_frame(pose) -> str | None:
        if isinstance(pose, dict):
            return pose.get("frame") or pose.get("frame_id")
        return getattr(pose, "frame", None) or getattr(pose, "frame_id", None)

    @staticmethod
    def _is_contact_close_step(step) -> bool:
        if step.parameters.get("during_contact") is True:
            return True
        if step.parameters.get("on_object") is True:
            return True
        text = f"{step.description} {step.parameters}".lower()
        return "during_contact" in text or "contact" in text or "on object" in text

    @staticmethod
    def _hardware_mode_from_context(context: Dict[str, Any]) -> HardwareMode | None:
        value = context.get("hardware_mode_object")
        return value if isinstance(value, HardwareMode) else None

    def check_parameter_type(self, value: Any, expected_type: str) -> bool:
        """Check if a parameter value matches the expected type."""
        type_map = {
            "string": str,
            "float": float,
            "int": int,
            "bool": bool
        }
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        return False

    def _get_step_parameter(self, step, parameter_name: str):
        """Return a step parameter, including accepted aliases for demos."""
        if parameter_name in step.parameters:
            return step.parameters[parameter_name]
        if step.skill_id == "move_to_named_target" and parameter_name == "target_name":
            return step.parameters.get("target")
        return None

    def _verify_detect_object_context(
        self,
        context: Dict[str, Any],
        warnings: list[str],
    ) -> None:
        if not context.get("perception_available", False):
            warnings.append(
                "Perception backend is not available; detect_object may fail at execution time."
            )

    def _verify_pick_object_context(
        self,
        step,
        context: Dict[str, Any],
        reasons: list[str],
    ) -> None:
        object_id = step.parameters.get("object_id")
        query = step.parameters.get("query")
        if object_id is None and query is None:
            reasons.append(
                "Missing required parameter 'object_id' or 'query' for skill 'pick_object'"
            )
            return
        if context.get("skip_perception_scene_checks", False):
            return

        scene = self._scene_from_context(context)
        if scene is None or not scene.objects:
            reasons.append(
                "Cannot verify pick_object because no detected scene object is available."
            )
            return

        scene_object = self._find_scene_object(
            scene,
            object_id=object_id,
            query=query,
        )
        if scene_object is None:
            reasons.append(
                "Cannot verify pick_object because the requested object is not "
                "in the detected scene."
            )
            return

        if self._base_link_pose_required(context):
            required_frame = context.get("target_frame", "base_link")
            if (
                not scene_object.pose_available
                or scene_object.pose is None
                or scene_object.pose.frame != required_frame
            ):
                reasons.append(
                    "pick_object blocked because no verified base_link object pose is available."
                )
        elif not scene_object.pose_available:
            reasons.append("Object pose is not available.")
        if not scene_object.graspable:
            reasons.append("Object is not marked as graspable.")

        min_confidence = step.parameters.get("min_confidence")
        if (
            min_confidence is not None
            and scene_object.confidence < float(min_confidence)
        ):
            reasons.append(
                f"Object confidence {scene_object.confidence} is below "
                f"required minimum {min_confidence}."
            )

    def _verify_place_object_context(
        self,
        step,
        context: Dict[str, Any],
        reasons: list[str],
        warnings: list[str],
    ) -> None:
        target_id = step.parameters.get("target_id")
        if target_id is None:
            return
        if context.get("skip_perception_scene_checks", False):
            return

        scene = self._scene_from_context(context)
        if scene is None or not scene.objects:
            warnings.append(
                "Cannot fully verify place_object because no detected scene "
                "object is available."
            )
            return
        if scene.find_by_id(target_id) is None and not scene.find_by_label(target_id):
            reasons.append(
                "Cannot verify place_object because target_id is not in the "
                "detected scene."
            )

    @staticmethod
    def _scene_from_context(context: Dict[str, Any]) -> SceneModel | None:
        scene = context.get("scene")
        if scene is None:
            return None
        if isinstance(scene, SceneModel):
            return scene
        if isinstance(scene, dict):
            return SceneModel.model_validate(scene)
        return None

    @staticmethod
    def _base_link_pose_required(context: Dict[str, Any]) -> bool:
        return bool(context.get("require_base_link_pose_for_pick", False))

    @staticmethod
    def _find_scene_object(
        scene: SceneModel,
        object_id: str | None = None,
        query: str | None = None,
    ) -> SceneObject | None:
        if object_id:
            scene_object = scene.find_by_id(object_id)
            if scene_object is not None:
                return scene_object
        if query:
            matches = scene.find_by_label(query)
            if matches:
                return matches[0]
            scene_object = scene.find_by_id(query)
            if scene_object is not None:
                return scene_object
        return None
