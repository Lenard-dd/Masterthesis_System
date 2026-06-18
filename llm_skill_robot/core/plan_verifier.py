import yaml
from pathlib import Path
from typing import Dict, Any
from .skill_library import SkillLibrary
from .plan import RobotPlan
from .schemas import VerificationResult, VerificationDecision
from ..perception.scene_model import SceneModel, SceneObject
from ..utils import get_config_dir


class PlanVerifier:
    def __init__(self, skill_library: SkillLibrary, config_dir: str = None):
        self.skill_library = skill_library
        if config_dir is None:
            self.config_dir = get_config_dir()
        else:
            self.config_dir = Path(config_dir)
        self.robot_capabilities = self.load_robot_capabilities()

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

        for step in plan.steps:
            skill = self.skill_library.get_skill(step.skill_id)
            if not skill:
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
                self._verify_pick_object_context(step, context, reasons)
            elif step.skill_id == "place_object":
                self._verify_place_object_context(step, context, reasons, warnings)

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

        if not scene_object.pose_available:
            reasons.append("Object pose is not available.")
        elif self._base_link_pose_required(context):
            required_frame = context.get("target_frame", "base_link")
            if scene_object.pose is None or scene_object.pose.frame != required_frame:
                reasons.append(
                    f"Object pose is not available in required frame '{required_frame}'."
                )
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
