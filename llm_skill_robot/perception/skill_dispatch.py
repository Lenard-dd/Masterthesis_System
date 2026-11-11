"""Shared non-motion skill dispatch for GUI and diagnostic workflows."""

from __future__ import annotations

import time
from typing import Any, Dict
import yaml
from llm_skill_robot.perception.perception_pipeline import PerceptionPipeline
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner, generate_candidates_with_context
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.grasping.grasp_verifier import GraspVerifier
from llm_skill_robot.robot.robotiq_2f140_sim_backend import Robotiq2F140SimBackend
from llm_skill_robot.utils import get_config_dir

GRIPPER_SKILL_IDS = {
    "open_gripper",
    "close_gripper",
    "set_gripper_width",
    "get_gripper_state",
}


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
