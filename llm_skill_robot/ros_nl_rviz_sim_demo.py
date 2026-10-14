"""Natural-language RViz/fake-hardware simulated execution demo for UR5."""

from datetime import datetime, timezone
import sys
import time
from typing import Any, Dict, List

import yaml

from llm_skill_robot.agent.mock_plan_generator import MockPlanGenerator, PlanGenerationError
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.execution_mode import is_execution_allowed, load_execution_config
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan import RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import ValidationDecision
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.core.trajectory_reviewer import TrajectoryReviewer
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer
from llm_skill_robot.perception.perception_factory import (
    create_perception_pipeline,
    load_perception_config,
    perception_context_from_config,
)
from llm_skill_robot.perception.perception_pipeline import PerceptionPipeline
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner, generate_candidates_with_context
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.grasping.grasp_verifier import GraspVerifier
from llm_skill_robot.robot.robotiq_2f140_sim_backend import Robotiq2F140SimBackend
from llm_skill_robot.robot.robotiq_2f140_joint_state_sim import Robotiq2F140JointStateSim
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode
from llm_skill_robot.ros_rviz_sim_execution_demo import _load_safety_config
from llm_skill_robot.utils import get_config_dir

NL_RVIZ_SIM_WARNING = (
    "NATURAL LANGUAGE RVIZ_SIM DEMO. Real robot execution is disabled."
)
GRIPPER_SKILL_IDS = {
    "open_gripper",
    "close_gripper",
    "set_gripper_width",
    "get_gripper_state",
}
PERCEPTION_SKILL_IDS = {
    "detect_object", "detect_objects", "build_object_point_cloud", "compute_place_pose",
}
GRASP_PLANNING_SKILL_IDS = {"generate_grasp_pose", "review_grasp_candidate"}
OBJECT_RELATED_SKILL_IDS = {"pick_object", "place_object"}


def _audit_event(audit_logger: AuditLogger, event: str, payload: Dict[str, Any]) -> None:
    if hasattr(audit_logger, "_write_entry"):
        audit_logger._write_entry(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "mode": "rviz_sim",
                "plan_id": None,
                "decision": None,
                "message": None,
                **payload,
            }
        )


def _print_generated_plan(plan: RobotPlan) -> None:
    print("\nGenerated Skill Plan:")
    for step in plan.steps:
        if step.skill_id == "move_to_named_target":
            target = step.parameters.get("target") or step.parameters.get("target_name")
            print(f"{step.step_id}. {step.skill_id} target={target}")
        else:
            print(f"{step.step_id}. {step.skill_id}")


def _print_verification_result(result) -> None:
    print("\nVerification:")
    print(result.decision.value)
    if result.reasons:
        print(f"Reasons: {result.reasons}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def _print_motion_plan_summary(plan_result: dict) -> None:
    summary = plan_result["summary"]
    print("\nMotion Plan Summary:")
    print(f"Plan ID: {plan_result['plan_id']}")
    print(f"Target: {summary.get('target_name') or summary.get('target_pose')}")
    print(f"Planning group: {summary.get('planning_group')}")
    print(f"Trajectory points: {summary.get('num_trajectory_points')}")
    print(f"Duration sec: {summary.get('duration_sec')}")
    print(f"Planning success: {str(plan_result['success']).lower()}")
    print(f"Message: {plan_result['message']}")


def _print_trajectory_validation(result: Dict[str, Any]) -> None:
    print("\nTrajectory Validation:")
    print(result["decision"])
    if result.get("reasons"):
        print(f"Reasons: {result['reasons']}")
    if result.get("warnings"):
        print(f"Warnings: {result['warnings']}")


def _load_gripper_config() -> Dict[str, Any]:
    config_file = get_config_dir() / "gripper.yaml"
    if not config_file.exists():
        return {}
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def _load_gripper_joint_mapping_config() -> Dict[str, Any]:
    config_file = get_config_dir() / "gripper_joint_mapping.yaml"
    if not config_file.exists():
        return {}
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def _load_grasping_config() -> Dict[str, Any]:
    config_file = get_config_dir() / "grasping.yaml"
    if not config_file.exists():
        return {}
    with config_file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _relative_place_support_kwargs() -> Dict[str, Any]:
    """Load the calibrated base_link support plane with a bounded flat fallback."""
    config_file = get_config_dir() / "tabletop_safety.yaml"
    if not config_file.exists():
        return {"support_plane": None, "flat_table_fallback": True}
    with config_file.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    tabletop = document.get("tabletop_safety", {}) or {}
    policy = tabletop.get("relative_place_support", {}) or {}
    plane = tabletop.get("plane") if bool(tabletop.get("enabled", False)) else None
    support_plane = None
    if bool(policy.get("use_calibrated_plane", True)) and isinstance(plane, dict):
        support_plane = {
            "frame": str(tabletop.get("frame", "base_link")),
            "normal": plane.get("normal"),
            "d": plane.get("d"),
        }
    return {
        "support_plane": support_plane,
        "flat_table_fallback": bool(policy.get("flat_table_fallback", True)),
        "max_source_support_error_m": float(
            policy.get("max_source_support_error_m", 0.03)
        ),
        "supported_surface_release_clearance_m": float(
            policy.get("supported_surface_release_clearance_m", 0.001)
        ),
    }


def _create_gripper_joint_state_sim(node) -> Robotiq2F140JointStateSim | None:
    config = _load_gripper_joint_mapping_config()
    mapping = config.get("gripper_joint_mapping", {})
    if not mapping.get("enabled", False):
        return None

    joint_names = mapping.get("joint_names", []) or []
    if joint_names:
        print("\nWarning: RViz gripper joint-state simulation is enabled.")
        print("Only the configured Robotiq gripper joints will be published:")
        for joint_name in joint_names:
            print(f"- {joint_name}")
        print(
            "If joint_state_publisher_gui is also publishing these joints, "
            "RViz may flicker or ignore updates. Close joint_state_publisher_gui "
            "or disable duplicate publishing."
        )

    joint_state_sim = Robotiq2F140JointStateSim(node=node, config=config)
    joint_state_sim.start_periodic_publishing()
    return joint_state_sim


def execute_non_motion_skill(
    step,
    gripper_backend: Robotiq2F140SimBackend | None,
    context: Dict[str, Any] | None = None,
    perception_pipeline: PerceptionPipeline | None = None,
    point_cloud_builder: ObjectPointCloudBuilder | None = None,
    grasp_planner: GraspGenXGraspPlanner | None = None,
    scene_description_backend=None,
    place_pose_visualizer=None,
) -> Dict[str, Any]:
    context = context if context is not None else {}
    if step.skill_id == "describe_scene":
        if scene_description_backend is None:
            from llm_skill_robot.perception.scene_understanding import (
                SceneDescriptionSkillBackend,
            )

            scene_description_backend = SceneDescriptionSkillBackend(
                perception_pipeline or PerceptionPipeline()
            )
        parameters = dict(step.parameters)
        result = scene_description_backend.describe_scene(**parameters)
        if result.get("success"):
            # This is semantic model output, explicitly not a geometry-verified
            # scene. It must never be used as a pick/place target.
            context["unverified_scene_description"] = result.get("scene_description")
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "parameters": parameters,
            "mode": "rviz_sim",
            "success": result.get("success", False),
            "status": result.get("status"),
            "message": result.get("message"),
            "output": result,
        }

    if step.skill_id == "detect_object":
        perception_pipeline = perception_pipeline or PerceptionPipeline()
        parameters = dict(step.parameters)
        try:
            result = perception_pipeline.detect_object(
                query=parameters.get("query", ""),
                min_confidence=parameters.get("min_confidence", 0.5),
                require_pose=parameters.get("require_pose", True),
            )
        except Exception as exc:
            result = {
                "success": False,
                "status": "FAILED",
                "message": f"detect_object failed: {exc}",
                "objects": [],
                "scene": perception_pipeline.get_latest_scene().model_dump(),
            }
        latest_scene = perception_pipeline.get_latest_scene()
        context["scene"] = latest_scene
        context["perception_available"] = result.get("status") != "NOT_AVAILABLE"
        if result.get("objects"):
            selected = max(
                result["objects"],
                key=lambda value: float(value.get("confidence", 0.0) or 0.0),
            )
            context["selected_object"] = selected
            context["resolved_object_id"] = selected.get("object_id")
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "parameters": parameters,
            "mode": "rviz_sim",
            "success": result.get("success", False),
            "status": result.get("status"),
            "message": result.get("message"),
            "objects": result.get("objects", []),
            "scene": result.get("scene", {}),
            "output": result,
        }

    if step.skill_id == "detect_objects":
        perception_pipeline = perception_pipeline or PerceptionPipeline()
        parameters = dict(step.parameters)
        try:
            result = perception_pipeline.detect_objects(
                queries=parameters.get("queries", []),
                min_confidence=parameters.get("min_confidence", 0.5),
                require_pose=parameters.get("require_pose", True),
            )
        except Exception as exc:
            result = {
                "success": False,
                "status": "FAILED",
                "message": f"detect_objects failed: {exc}",
                "objects": [],
                "scene": perception_pipeline.get_latest_scene().model_dump(),
                "query_results": [],
            }
        latest_scene = perception_pipeline.get_latest_scene()
        context["scene"] = latest_scene
        context["localized_objects"] = result.get("objects", [])
        context["perception_available"] = result.get("status") != "NOT_AVAILABLE"
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "parameters": parameters,
            "mode": "rviz_sim",
            "success": result.get("success", False),
            "status": result.get("status"),
            "message": result.get("message"),
            "objects": result.get("objects", []),
            "query_results": result.get("query_results", []),
            "scene": result.get("scene", {}),
            "output": result,
        }

    if step.skill_id == "compute_place_pose":
        from llm_skill_robot.perception.spatial_reasoning import RelativePlacePosePlanner

        parameters = dict(step.parameters)
        latest_scene = context.get("scene")
        if latest_scene is None and perception_pipeline is not None:
            latest_scene = perception_pipeline.get_latest_scene().model_dump()
        try:
            geometry_updates = _enrich_relative_place_geometry(
                latest_scene or {},
                source_id=str(parameters.get("source_id", "")),
                reference_ids=list(parameters.get("reference_ids", [])),
                perception_pipeline=perception_pipeline,
                point_cloud_builder=point_cloud_builder,
            )
            relation_constraints = parameters.get("relation_constraints")
            planner = RelativePlacePosePlanner(**_relative_place_support_kwargs())
            if isinstance(relation_constraints, list) and relation_constraints:
                result = planner.compute_composite(
                    latest_scene or {},
                    source_id=str(parameters.get("source_id", "")),
                    relation_specs=[dict(item) for item in relation_constraints if isinstance(item, dict)],
                )
            else:
                result = planner.compute(
                    latest_scene or {},
                    source_id=str(parameters.get("source_id", "")),
                    relation=str(parameters.get("relation", "")),
                    reference_ids=list(parameters.get("reference_ids", [])),
                )
            result["geometry_updates"] = geometry_updates
            result["success"] = True
        except Exception as exc:
            result = {
                "success": False,
                "status": "FAILED",
                "message": f"compute_place_pose failed: {exc}",
            }
        if result.get("success"):
            context["place_pose_plan"] = result
            if place_pose_visualizer is not None:
                result["rviz_preview"] = place_pose_visualizer.publish(result)
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "parameters": parameters,
            "mode": "rviz_sim",
            "success": bool(result.get("success", False)),
            "status": result.get("status", "FAILED"),
            "message": result.get("message", "Place-pose computation failed."),
            "output": result,
        }

    if step.skill_id == "build_object_point_cloud":
        perception_pipeline = perception_pipeline or PerceptionPipeline()
        point_cloud_builder = point_cloud_builder or ObjectPointCloudBuilder()
        selected = _resolve_scene_object(step.parameters.get("object_id"), context)
        # A supervised pick that reuses a prior multi-object localization
        # intentionally carries only object_id through its small task context.
        # The full latest scene retains the SAM3 mask path required for point
        # cloud construction. Prefer that rich scene object when the context
        # entry is merely an identity stub; do not call SAM3 again.
        scene_selected = _resolve_scene_object(
            step.parameters.get("object_id"),
            {"scene": perception_pipeline.get_latest_scene()},
        )
        if selected is None or (
            scene_selected is not None and not _scene_object_has_mask_path(selected)
        ):
            selected = scene_selected
        if selected is None:
            result = {
                "success": False,
                "status": "NOT_FOUND",
                "message": "No detected object is available for point-cloud building.",
            }
        else:
            camera = getattr(perception_pipeline, "backend", None)
            readiness = _wait_for_point_cloud_inputs(camera)
            depth_getter = getattr(camera, "_get_latest_depth_image", None)
            metadata = (
                getattr(selected, "metadata", None)
                if not isinstance(selected, dict)
                else selected.get("metadata")
            ) or {}
            capture_id = str(metadata.get("rgbd_capture_id", "")).strip()
            snapshot_getter = getattr(camera, "get_rgbd_snapshot", None)
            snapshot = (
                snapshot_getter(capture_id)
                if capture_id and callable(snapshot_getter)
                else None
            )
            snapshot_depth = snapshot.get("depth_frame") if isinstance(snapshot, dict) else None
            snapshot_camera_info = (
                snapshot.get("camera_info") if isinstance(snapshot, dict) else None
            )
            depth_frame = (
                snapshot_depth
                if snapshot_depth is not None
                else depth_getter() if callable(depth_getter) else None
            )
            camera_info = (
                snapshot_camera_info
                if snapshot_camera_info is not None
                else getattr(camera, "latest_camera_info_msg", None)
            )
            camera_config = getattr(camera, "config", {}) or {}
            source_frame = (
                camera_config.get("depth_frame", "camera_color_optical_frame")
                if isinstance(camera_config, dict)
                else "camera_color_optical_frame"
            )
            point_transformer = getattr(camera, "pose_transformer", None)
            point_cloud_kwargs = {"frame": source_frame}
            if point_transformer is not None:
                point_cloud_kwargs["point_transformer"] = point_transformer
            result = point_cloud_builder.build(
                selected,
                depth_frame,
                camera_info,
                **point_cloud_kwargs,
            )
            result.setdefault("rgbd_capture_id", capture_id or None)
            result.setdefault(
                "rgbd_snapshot_used",
                snapshot_depth is not None and snapshot_camera_info is not None,
            )
            result.setdefault("camera_readiness", readiness)
            if result.get("success"):
                _apply_point_cloud_geometry(selected, result)
                result["scene"] = perception_pipeline.get_latest_scene().model_dump()
            if not result.get("success") and not readiness.get("ready", False):
                topics = camera_config if isinstance(camera_config, dict) else {}
                result["message"] = (
                    f"{result.get('message', 'Point-cloud building failed.')} "
                    "RGB-D readiness: "
                    f"depth_received={readiness.get('depth_received', False)}, "
                    f"camera_info_received={readiness.get('camera_info_received', False)}; "
                    f"depth_topic={topics.get('depth_topic', 'unknown')}, "
                    f"camera_info_topic={topics.get('camera_info_topic', 'unknown')}."
                )
        if result.get("success"):
            context["point_cloud"] = result
            context["resolved_object_id"] = _object_value(selected, "object_id")
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "parameters": dict(step.parameters),
            "mode": "rviz_sim",
            "success": bool(result.get("success", False)),
            "status": result.get("status", "FAILED"),
            "message": result.get("message", "Point-cloud building failed."),
            "output": result,
        }

    if step.skill_id == "generate_grasp_pose":
        point_cloud = context.get("point_cloud")
        point_cloud_path = (
            point_cloud.get("point_cloud_path")
            if isinstance(point_cloud, dict)
            else None
        )
        if not point_cloud_path:
            result = {
                "success": False,
                "status": "NOT_AVAILABLE",
                "message": "Build an object point cloud before generating grasp poses.",
                "candidates": [],
            }
        else:
            if grasp_planner is None:
                document = _load_grasping_config()
                config = document.get("grasping", document)
                grasp_planner = GraspGenXGraspPlanner(
                    GraspGenXSubprocessAdapter(
                        config.get("graspgenx_subprocess", {})
                    )
                )
            result = generate_candidates_with_context(
                grasp_planner, point_cloud,
                getattr(getattr(perception_pipeline, "backend", None), "pose_transformer", None),
            )
        candidates = result.get("candidates", []) if isinstance(result, dict) else []
        if candidates:
            context["grasp_candidates"] = candidates
            context["selected_grasp_candidate"] = grasp_planner.select_best_candidate(candidates)
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "parameters": dict(step.parameters),
            "mode": "rviz_sim",
            "success": bool(candidates),
            "status": result.get("status", "SUCCESS" if candidates else "FAILED"),
            "message": result.get("message", f"Generated {len(candidates)} grasp candidate(s)."),
            "output": result,
        }

    if step.skill_id == "review_grasp_candidate":
        candidate = context.get("selected_grasp_candidate")
        result = {
            "success": candidate is not None,
            "status": "SUCCESS" if candidate is not None else "NOT_AVAILABLE",
            "message": (
                "Selected grasp candidate is available for RViz review."
                if candidate is not None
                else "No generated grasp candidate is available for review."
            ),
            "candidate": candidate,
        }
        context["grasp_candidate_approved"] = candidate is not None
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "parameters": dict(step.parameters),
            "mode": "rviz_sim",
            **result,
            "output": result,
        }

    if step.skill_id == "verify_grasp":
        object_id = step.parameters.get("object_id") or context.get("resolved_object_id")
        if object_id == "<resolved>":
            object_id = context.get("resolved_object_id")
        gripper_state = (
            gripper_backend.get_gripper_state()
            if gripper_backend is not None else None
        )
        verification = GraspVerifier().verify_grasp(
            str(object_id or "unknown"),
            gripper_state=gripper_state,
            perception_result=None,
        )
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "parameters": dict(step.parameters),
            "mode": "rviz_sim",
            "success": True,
            "status": verification["status"],
            "message": verification["message"],
            "output": verification,
        }

    if step.skill_id not in GRIPPER_SKILL_IDS:
        result = {
            "success": False,
            "message": f"Unsupported non-motion skill in this demo: {step.skill_id}",
            "output": {},
        }
    else:
        if gripper_backend is None:
            return {
                "step_id": step.step_id,
                "skill_id": step.skill_id,
                "parameters": dict(step.parameters),
                "mode": "rviz_sim",
                "real_gripper_enabled": False,
                "success": False,
                "message": "Gripper backend is not available.",
                "output": {"mode": "rviz_sim", "real_gripper_enabled": False},
            }
        backend_method = getattr(gripper_backend, step.skill_id)
        result = backend_method(**dict(step.parameters))

    return {
        "step_id": step.step_id,
        "skill_id": step.skill_id,
        "parameters": dict(step.parameters),
        "mode": result.get("output", {}).get("mode", "rviz_sim"),
        "real_gripper_enabled": result.get("output", {}).get("real_gripper_enabled", False),
        **result,
    }


def _object_value(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _scene_object_has_mask_path(value: Any) -> bool:
    """Whether an object carries the SAM3 artifact required for a cloud."""
    metadata = _object_value(value, "metadata", {}) or {}
    return isinstance(metadata, dict) and bool(str(metadata.get("mask_path", "")).strip())


def _resolve_scene_object(object_id: Any, context: Dict[str, Any]) -> Any | None:
    selected = context.get("selected_object")
    resolved_id = context.get("resolved_object_id")
    requested = str(object_id or "")
    if requested == "<resolved>":
        requested = str(resolved_id or "")
    if selected is not None and (
        not requested or str(_object_value(selected, "object_id", "")) == requested
    ):
        return selected
    scene = context.get("scene")
    objects = _object_value(scene, "objects", []) or []
    if requested:
        for value in objects:
            if str(_object_value(value, "object_id", "")) == requested:
                return value
        label_matches = [
            value for value in objects
            if str(_object_value(value, "label", "")) == requested
        ]
        if len(label_matches) == 1:
            return label_matches[0]
        # A requested but unknown/ambiguous label must not silently select the
        # highest-confidence object for a geometric placement calculation.
        return None
    return max(
        objects,
        key=lambda value: float(_object_value(value, "confidence", 0.0) or 0.0),
        default=None,
    )


def _enrich_relative_place_geometry(
    scene: Any,
    *,
    source_id: str,
    reference_ids: list[str],
    perception_pipeline: PerceptionPipeline | None,
    point_cloud_builder: ObjectPointCloudBuilder | None,
) -> list[dict[str, Any]]:
    """Attach measured masked-point-cloud footprints before pose computation.

    Geometry enrichment is best effort.  A failed or unavailable point cloud
    never invents dimensions: the placement planner explicitly falls back to
    its configured conservative default size.
    """
    if perception_pipeline is None:
        return [{"status": "SKIPPED", "message": "No live perception pipeline."}]
    builder = point_cloud_builder or ObjectPointCloudBuilder()
    camera = getattr(perception_pipeline, "backend", None)
    depth_getter = getattr(camera, "_get_latest_depth_image", None)
    camera_info = getattr(camera, "latest_camera_info_msg", None)
    camera_config = getattr(camera, "config", {}) or {}
    source_frame = (
        camera_config.get("depth_frame", "camera_color_optical_frame")
        if isinstance(camera_config, dict)
        else "camera_color_optical_frame"
    )
    transformer = getattr(camera, "pose_transformer", None)
    depth = depth_getter() if callable(depth_getter) else None
    updates: list[dict[str, Any]] = []
    for identifier in [source_id, *reference_ids]:
        selected = _resolve_scene_object(identifier, {"scene": scene})
        if selected is None:
            updates.append({"object": identifier, "status": "NOT_FOUND"})
            continue
        point_cloud_kwargs = {"frame": source_frame}
        if transformer is not None:
            point_cloud_kwargs["point_transformer"] = transformer
        result = builder.build(selected, depth, camera_info, **point_cloud_kwargs)
        if result.get("success"):
            _apply_point_cloud_geometry(selected, result)
            updates.append({
                "object": _object_value(selected, "object_id", identifier),
                "status": "UPDATED",
                "footprint_dimensions_m": result.get("footprint_dimensions_m"),
                "dimensions_frame": result.get("dimensions_frame"),
            })
        else:
            updates.append({
                "object": _object_value(selected, "object_id", identifier),
                "status": result.get("status", "FAILED"),
                "message": result.get("message", "Point-cloud geometry unavailable."),
            })
    return updates


def _apply_point_cloud_geometry(scene_object: Any, point_cloud: dict[str, Any]) -> None:
    """Write only frame-compatible, explicitly qualified geometry to a scene object."""
    metadata = dict(_object_value(scene_object, "metadata", {}) or {})
    pose = _object_value(scene_object, "pose")
    pose_frame = _object_value(pose, "frame") if pose is not None else None
    dimensions_frame = point_cloud.get("dimensions_frame")
    if pose_frame and dimensions_frame == pose_frame:
        metadata["footprint_dimensions_m"] = dict(point_cloud.get("footprint_dimensions_m") or {})
        metadata["visible_aabb_dimensions_m"] = dict(point_cloud.get("visible_aabb_dimensions_m") or {})
        metadata["dimension_estimation"] = dict(point_cloud.get("dimension_estimation") or {})
        height_estimation = dict(point_cloud.get("height_estimation") or {})
        metadata["height_estimation"] = height_estimation
        if bool(height_estimation.get("usable", False)):
            footprint = metadata["footprint_dimensions_m"]
            metadata["dimensions_m"] = {
                "x": float(footprint["x"]),
                "y": float(footprint["y"]),
                "z": float(height_estimation["height_m"]),
            }
        metadata["dimension_frame"] = dimensions_frame
    else:
        warnings = list(metadata.get("warnings", []) or [])
        warning = "Point-cloud dimensions were not used because their frame differs from the localized pose frame."
        if warning not in warnings:
            warnings.append(warning)
        metadata["warnings"] = warnings
    if isinstance(scene_object, dict):
        scene_object["metadata"] = metadata
    else:
        scene_object.metadata = metadata


def _wait_for_point_cloud_inputs(camera: Any, timeout_sec: float = 5.0) -> Dict[str, Any]:
    """Wait for aligned depth and CameraInfo after RGB detection.

    A camera attached to an existing executor must never be spun here because
    rclpy would move its node away from that executor.  Standalone demos still
    spin nodes which have no executor owner.
    """
    if camera is None:
        return {"ready": False, "depth_received": False, "camera_info_received": False}
    node = getattr(camera, "node", None)
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while time.monotonic() < deadline:
        depth_ready = getattr(camera, "latest_depth_msg", None) is not None
        info_ready = getattr(camera, "latest_camera_info_msg", None) is not None
        if depth_ready and info_ready:
            return {
                "ready": True,
                "depth_received": True,
                "camera_info_received": True,
            }
        if node is not None:
            from llm_skill_robot.ros_executor_utils import node_has_executor

            if not node_has_executor(node):
                try:
                    import rclpy

                    rclpy.spin_once(node, timeout_sec=0.05)
                    continue
                except Exception:
                    pass
        time.sleep(0.05)
    return {
        "ready": False,
        "depth_received": getattr(camera, "latest_depth_msg", None) is not None,
        "camera_info_received": getattr(camera, "latest_camera_info_msg", None) is not None,
    }


def _print_gripper_step_result(result: Dict[str, Any]) -> None:
    output = result.get("output", {})
    print("\nGripper Step:")
    print(f"Skill: {result.get('skill_id')}")
    print(f"Mode: {output.get('mode', result.get('mode', 'rviz_sim'))}")
    print(f"Result: {'SUCCESS' if result.get('success') else 'FAILED'}")
    if "current_width_m" in output:
        print(f"Current width: {output['current_width_m']:.2f} m")
    print(
        "Real gripper enabled: "
        f"{str(output.get('real_gripper_enabled', False)).lower()}"
    )
    print(f"Message: {result.get('message')}")


def _print_perception_step_result(result: Dict[str, Any]) -> None:
    parameters = result.get("parameters", {})
    print("\nPerception Step:")
    print(f"Query: {parameters.get('query')}")
    print(f"Status: {result.get('status')}")
    print(f"Message: {result.get('message')}")


def _print_staged_grasp_step_result(result: Dict[str, Any]) -> None:
    print("\nStaged Grasp Step:")
    print(f"Skill: {result.get('skill_id')}")
    print(f"Status: {result.get('status', 'UNKNOWN')}")
    print(f"Result: {'SUCCESS' if result.get('success') else 'FAILED'}")
    print(f"Message: {result.get('message')}")


def _audit_perception_result(
    audit_logger: AuditLogger,
    step,
    result: Dict[str, Any],
) -> None:
    _audit_event(
        audit_logger,
        "perception_skill_executed",
        {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "query": dict(step.parameters).get("query"),
            "status": result.get("status"),
            "message": result.get("message"),
            "objects_count": len(result.get("objects", [])),
            "scene_updated": bool(result.get("scene") is not None),
            "real_camera_used": (
                result.get("status") == "SUCCESS"
                and (result.get("scene") or {}).get("source") == "d435i_sam3"
            ),
            "result": result,
        },
    )


def _verify_object_step_for_execution(
    *,
    step,
    skill_library: SkillLibrary,
    execution_context: Dict[str, Any],
) -> Dict[str, Any] | None:
    verification_result = PlanVerifier(skill_library).verify_plan(
        RobotPlan(task=f"verify {step.skill_id}", steps=[step]),
        context=execution_context,
    )
    if verification_result.decision.value == "ALLOW":
        return None

    if step.skill_id == "pick_object":
        print("pick_object blocked because object pose is not available.")

    message = (
        "; ".join(verification_result.reasons)
        or "Object-related step blocked by verification."
    )
    return {
        "step_id": step.step_id,
        "skill_id": step.skill_id,
        "parameters": dict(step.parameters),
        "success": False,
        "message": message,
        "reasons": verification_result.reasons,
        "warnings": verification_result.warnings,
        "mode": "rviz_sim",
        "output": {
            "mode": "rviz_sim",
            "verification_decision": verification_result.decision.value,
            "reasons": verification_result.reasons,
            "warnings": verification_result.warnings,
        },
    }


def _run_motion_step(
    *,
    step,
    backend: UR5MoveItPlanBackend,
    node,
    execution_config: Dict[str, Any],
    trajectory_validator: TrajectoryValidator,
    trajectory_reviewer: TrajectoryReviewer,
    audit_logger: AuditLogger,
    target_pose: Dict[str, Any] | None = None,
    plan_kwargs: Dict[str, Any] | None = None,
    velocity_scale: float | None = None,
    acceleration_scale: float | None = None,
) -> Dict[str, Any]:
    motion_kwargs = {
        "velocity_scale": (
            step.parameters.get("velocity_scale", 0.03)
            if velocity_scale is None else velocity_scale
        ),
        "acceleration_scale": (
            step.parameters.get("acceleration_scale", 0.03)
            if acceleration_scale is None else acceleration_scale
        ),
        "skill_id": step.skill_id,
        "step_id": int(step.step_id),
        **dict(plan_kwargs or {}),
    }
    if target_pose is None:
        target = step.parameters.get("target") or step.parameters.get("target_name")
        plan_result = backend.plan_to_named_target(target=target, **motion_kwargs)
    else:
        plan_result = backend.plan_to_pose(
            frame=target_pose["frame"],
            position=target_pose["position"],
            orientation=target_pose["orientation"],
            **motion_kwargs,
        )
    _print_motion_plan_summary(plan_result)
    _audit_event(
        audit_logger,
        "motion_planned",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": "SUCCESS" if plan_result["success"] else "FAILED",
            "message": plan_result["message"],
            "summary": plan_result["summary"],
        },
    )

    trajectory_result = trajectory_validator.validate_motion_plan_summary(
        plan_result["summary"]
    )
    _print_trajectory_validation(trajectory_result)
    _audit_event(
        audit_logger,
        "trajectory_validated",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": trajectory_result["decision"],
            "message": "Trajectory validation completed.",
            "reasons": trajectory_result.get("reasons", []),
            "warnings": trajectory_result.get("warnings", []),
        },
    )
    if trajectory_result["decision"] != "ALLOW":
        return {
            "success": False,
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "message": "Trajectory validation blocked RViz/fake hardware execution.",
        }

    print("\nTrajectory Review:")
    review_approved = trajectory_reviewer.ask_for_review_approval(
        plan_result["summary"]
    )
    _audit_event(
        audit_logger,
        "trajectory_reviewed",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": "APPROVE" if review_approved else "REJECT",
            "message": (
                "Trajectory review approved."
                if review_approved
                else "Trajectory review rejected."
            ),
        },
    )
    if not review_approved:
        return {
            "success": False,
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "message": "Trajectory review was not approved.",
        }

    print("\nVisual Trajectory Preview:")
    preview_topic = execution_config.get(
        "visual_preview_topic",
        "/display_planned_path",
    )
    trajectory_visualizer = TrajectoryVisualizer(node=node, topic=preview_topic)
    cached_plan = backend.get_plan(plan_result["plan_id"])
    preview_result = trajectory_visualizer.publish_preview(cached_plan)
    _audit_event(
        audit_logger,
        "visual_preview_published",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": "PUBLISHED" if preview_result["success"] else "FAILED",
            "message": preview_result["message"],
            "preview_result": preview_result,
        },
    )
    if preview_result["success"]:
        print(preview_result["message"])
    else:
        print(
            "No raw trajectory available. Please rely on textual trajectory "
            "review and MoveIt/RViz planning display."
        )
    print("Please inspect RViz. No motion has been executed yet.")

    print("\nFinal Confirmation:")
    choice = input("\nExecute this plan in RViz/fake hardware only? [y/N] ").strip()
    final_confirmed = choice.lower() == "y"
    _audit_event(
        audit_logger,
        "final_confirmed",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": "CONFIRMED" if final_confirmed else "DECLINED",
            "message": (
                "Final confirmation provided."
                if final_confirmed
                else "Final confirmation not provided."
            ),
            "confirmed": final_confirmed,
        },
    )
    if not final_confirmed:
        return {
            "success": False,
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "message": "Final confirmation was not provided.",
        }

    execution_result = backend.execute_cached_plan_simulated(
        plan_result["plan_id"],
        final_confirmed=True,
        trajectory_validation_allowed=trajectory_result["decision"] == "ALLOW",
        trajectory_review_approved=review_approved,
    )
    _audit_event(
        audit_logger,
        "simulated_execution_result",
        {
            "plan_id": execution_result["plan_id"],
            "step_id": step.step_id,
            "decision": "SUCCESS" if execution_result["success"] else "FAILED",
            "message": execution_result["message"],
            "execution_result": execution_result,
        },
    )
    return {
        "step_id": step.step_id,
        "skill_id": step.skill_id,
        **execution_result,
    }


def _print_execution_report(step_results: List[Dict[str, Any]]) -> None:
    success = all(result.get("success", False) for result in step_results)
    print("\nExecution Report:")
    print("SUCCESS" if success else "FAILED")
    for result in step_results:
        print(
            f"Step {result.get('step_id')} {result.get('skill_id')}: "
            f"{'SUCCESS' if result.get('success') else 'FAILED'} - {result.get('message')}"
        )


def main() -> None:
    hardware_mode = load_hardware_mode()
    print_hardware_mode(hardware_mode)
    print(NL_RVIZ_SIM_WARNING)
    audit_logger = AuditLogger()

    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        print(f"\nAudit log: {audit_logger.log_file.resolve()}")
        sys.exit(1)

    backend = None
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_nl_rviz_sim_demo")

        skill_library = SkillLibrary()
        safety_config = _load_safety_config()
        execution_config = load_execution_config().model_dump()
        execution_mode = "rviz_sim"
        _audit_event(
            audit_logger,
            "nl_rviz_sim_demo_started",
            {"message": NL_RVIZ_SIM_WARNING},
        )

        if not is_execution_allowed(execution_mode, execution_config):
            print("\nExecution config does not allow rviz_sim mode. Exiting safely.")
            _audit_event(
                audit_logger,
                "rviz_sim_execution_not_allowed",
                {
                    "decision": "BLOCK",
                    "message": "Execution config does not allow rviz_sim mode.",
                },
            )
            return
        if execution_config.get("allow_real_robot_execution", False):
            print("\nreal_robot execution is enabled in config. Exiting safely.")
            _audit_event(
                audit_logger,
                "real_robot_execution_config_blocked_demo",
                {
                    "decision": "BLOCK",
                    "message": "real_robot execution is enabled in config.",
                },
            )
            return

        user_instruction = input("\nEnter a robot task:\n> ").strip()
        plan = MockPlanGenerator().generate_plan(
            user_instruction=user_instruction,
            available_skills=skill_library.list_skills(),
        )
        _print_generated_plan(plan)
        _audit_event(
            audit_logger,
            "plan_generated",
            {
                "decision": "GENERATED",
                "message": "MockPlanGenerator produced a RobotPlan.",
                "plan": plan.model_dump(),
            },
        )

        verification_result = PlanVerifier(skill_library).verify_plan(
            plan,
            context={"skip_perception_scene_checks": True},
        )
        _print_verification_result(verification_result)
        _audit_event(
            audit_logger,
            "plan_verified",
            {
                "decision": verification_result.decision.value,
                "message": "Plan verification completed.",
                "reasons": verification_result.reasons,
                "warnings": verification_result.warnings,
            },
        )
        if verification_result.decision.value != "ALLOW":
            print(
                "\nPlan blocked by verification. "
                "No planning or execution requested."
            )
            return

        validation_result = HumanValidator().validate_plan(plan)
        print("\nHuman Validation:")
        print(
            "approved"
            if validation_result.decision == ValidationDecision.APPROVE
            else validation_result.decision.value.lower()
        )
        _audit_event(
            audit_logger,
            "human_validated",
            {
                "decision": validation_result.decision.value,
                "message": "Human validation completed.",
                "comments": validation_result.comments,
            },
        )
        if validation_result.decision != ValidationDecision.APPROVE:
            print("\nPlan was not approved. No planning or execution requested.")
            return

        backend = UR5MoveItPlanBackend(
            node=node,
            dry_run=False,
            mode=execution_mode,
            execution_config=execution_config,
            hardware_mode=hardware_mode,
        )
        gripper_joint_state_sim = _create_gripper_joint_state_sim(node)
        gripper_backend = Robotiq2F140SimBackend(
            _load_gripper_config(),
            joint_state_sim=gripper_joint_state_sim,
        )
        perception_config = load_perception_config()
        perception_pipeline = create_perception_pipeline(
            node=node,
            config=perception_config,
        )
        grasping_document = _load_grasping_config()
        grasping_config = grasping_document.get("grasping", grasping_document)
        point_cloud_builder = ObjectPointCloudBuilder(
            grasping_config.get("point_cloud", {})
        )
        grasp_planner = GraspGenXGraspPlanner(
            GraspGenXSubprocessAdapter(
                grasping_config.get("graspgenx_subprocess", {})
            )
        )
        execution_context: Dict[str, Any] = {
            "scene": perception_pipeline.get_latest_scene(),
            "perception_available": False,
            **perception_context_from_config(perception_config),
        }
        trajectory_validator = TrajectoryValidator(safety_config)
        trajectory_reviewer = TrajectoryReviewer()
        step_results = []

        for step in plan.steps:
            if step.skill_id in OBJECT_RELATED_SKILL_IDS:
                result = _verify_object_step_for_execution(
                    step=step,
                    skill_library=skill_library,
                    execution_context=execution_context,
                )
                if result is None:
                    result = {
                        "success": False,
                        "step_id": step.step_id,
                        "skill_id": step.skill_id,
                        "parameters": dict(step.parameters),
                        "message": (
                            f"{step.skill_id} execution is not implemented in RViz sim "
                            "without a verified perception-backed object pose."
                        ),
                        "mode": "rviz_sim",
                        "output": {"mode": "rviz_sim"},
                    }
                _audit_event(
                    audit_logger,
                    "object_skill_verified",
                    {
                        "step_id": step.step_id,
                        "skill_id": step.skill_id,
                        "parameters": dict(step.parameters),
                        "decision": "SUCCESS" if result["success"] else "BLOCK",
                        "message": result["message"],
                        "result": result,
                    },
                )
            elif step.skill_id == "move_to_named_target":
                result = _run_motion_step(
                    step=step,
                    backend=backend,
                    node=node,
                    execution_config=execution_config,
                    trajectory_validator=trajectory_validator,
                    trajectory_reviewer=trajectory_reviewer,
                    audit_logger=audit_logger,
                )
            else:
                result = execute_non_motion_skill(
                    step,
                    gripper_backend,
                    context=execution_context,
                    perception_pipeline=perception_pipeline,
                    point_cloud_builder=point_cloud_builder,
                    grasp_planner=grasp_planner,
                )
                if step.skill_id in PERCEPTION_SKILL_IDS:
                    _print_perception_step_result(result)
                    _audit_perception_result(audit_logger, step, result)
                elif step.skill_id in GRASP_PLANNING_SKILL_IDS:
                    _print_staged_grasp_step_result(result)
                    _audit_event(
                        audit_logger,
                        "staged_grasp_skill_executed",
                        {
                            "step_id": step.step_id,
                            "skill_id": step.skill_id,
                            "decision": "SUCCESS" if result["success"] else "FAILED",
                            "message": result["message"],
                            "result": result,
                        },
                    )
                else:
                    _print_gripper_step_result(result)
                    _audit_event(
                        audit_logger,
                        "gripper_skill_executed",
                        {
                            "step_id": step.step_id,
                            "skill_id": step.skill_id,
                            "parameters": dict(step.parameters),
                            "decision": "SUCCESS" if result["success"] else "FAILED",
                            "message": result["message"],
                            "result": result,
                            "mode": "rviz_sim",
                            "real_gripper_enabled": False,
                        },
                    )
            step_results.append(result)
            if (
                not result.get("success", False)
                and step.skill_id != "detect_object"
            ):
                break

        _print_execution_report(step_results)
        _audit_event(
            audit_logger,
            "nl_rviz_sim_execution_report",
            {
                "decision": (
                    "SUCCESS"
                    if all(r.get("success", False) for r in step_results)
                    else "FAILED"
                ),
                "message": "Natural-language RViz simulation demo completed.",
                "step_results": step_results,
            },
        )

    except PlanGenerationError as e:
        print(f"\nPlan generation failed: {e}")
        _audit_event(
            audit_logger,
            "plan_generation_failed",
            {
                "decision": "FAILED",
                "message": str(e),
            },
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print(f"\nAudit log: {audit_logger.log_file.resolve()}")
        if "gripper_joint_state_sim" in locals() and gripper_joint_state_sim is not None:
            gripper_joint_state_sim.stop()
        if backend is not None:
            backend.shutdown()
        if node is not None:
            node.destroy_node()
        if "rclpy" in locals() and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
