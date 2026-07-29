"""Safe lazy factory for resources used by an RViz Agent session."""

from dataclasses import dataclass
from typing import Any

from llm_skill_robot.agent.rviz_execution_context import RvizExecutionContext


@dataclass
class RvizAgentResources:
    context: RvizExecutionContext
    gripper_joint_state_sim: Any = None

    def shutdown(self) -> None:
        if self.gripper_joint_state_sim is not None:
            self.gripper_joint_state_sim.stop()
        shutdown = getattr(self.context.motion_backend, "shutdown", None)
        if callable(shutdown):
            shutdown()


def validate_rviz_execution_config(config: dict[str, Any]) -> None:
    if config.get("mode") != "rviz_sim":
        raise ValueError("RViz Agent execution requires mode='rviz_sim'.")
    if config.get("allow_real_robot_execution", False):
        raise ValueError("RViz Agent refuses configuration with real robot execution enabled.")


def create_rviz_agent_resources(node, *, execution_config: dict[str, Any], safety_config: dict[str, Any], audit_logger=None) -> RvizAgentResources:
    """Create persistent fake-hardware resources only when explicitly called."""
    validate_rviz_execution_config(execution_config)
    from llm_skill_robot.core.trajectory_reviewer import TrajectoryReviewer
    from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
    from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
    from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
    from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
    from llm_skill_robot.perception.perception_factory import create_perception_pipeline, load_perception_config
    from llm_skill_robot.robot.robotiq_2f140_sim_backend import Robotiq2F140SimBackend
    from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
    from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer
    from llm_skill_robot.grasping.grasp_execution_backend import GraspExecutionBackend
    from llm_skill_robot.grasping.grasp_verifier import GraspVerifier
    from llm_skill_robot.ros_full_grasp_supervised_demo import FullGraspSupervisedDemoController
    from llm_skill_robot.safety.real_arm_safety import load_real_arm_safety
    from llm_skill_robot.safety.tabletop_safety import TabletopSafety
    from llm_skill_robot.ros_full_grasp_supervised_demo import _load_yaml
    from llm_skill_robot.ros_nl_rviz_sim_demo import _load_grasping_config, _load_gripper_config

    perception_config = load_perception_config()
    grasping_document = _load_grasping_config()
    grasping_config = grasping_document.get("grasping", grasping_document)
    context = RvizExecutionContext(
        node=node,
        motion_backend=UR5MoveItPlanBackend(node=node, dry_run=False, mode="rviz_sim", execution_config=execution_config),
        perception_pipeline=create_perception_pipeline(node=node, config=perception_config),
        point_cloud_builder=ObjectPointCloudBuilder(grasping_config.get("point_cloud", {})),
        grasp_planner=GraspGenXGraspPlanner(GraspGenXSubprocessAdapter(grasping_config.get("graspgenx_subprocess", {}))),
        gripper_backend=Robotiq2F140SimBackend(_load_gripper_config()),
        trajectory_validator=TrajectoryValidator(safety_config),
        trajectory_reviewer=TrajectoryReviewer(),
        execution_config=execution_config, safety_config=safety_config,
    )
    transformer = TFPoseTransformer(node, target_frame="base_link")
    tabletop_safety = TabletopSafety.from_config(_load_yaml("tabletop_safety.yaml"))
    motion_backend = GraspExecutionBackend(
        context.motion_backend, load_real_arm_safety(), audit_logger=audit_logger,
        tabletop_safety=tabletop_safety,
    )
    context.runtime_state.update({
        "grasping_config": grasping_config,
        "transformer": transformer,
        "full_demo_selector": FullGraspSupervisedDemoController(
            hardware_mode=None, real_arm_safety=load_real_arm_safety(),
            perception_pipeline=context.perception_pipeline,
            point_cloud_builder=context.point_cloud_builder,
            grasp_planner=context.grasp_planner, motion_backend=motion_backend,
            gripper_backend=context.gripper_backend, grasp_verifier=GraspVerifier(),
            grasping_config=grasping_config, transformer=transformer,
            audit_logger=audit_logger, tabletop_safety=tabletop_safety,
        ),
    })
    return RvizAgentResources(context=context)


def create_non_motion_executor(resources: RvizAgentResources):
    """Adapt the existing RViz non-motion implementation to Agent Tool calls."""
    from llm_skill_robot.ros_nl_rviz_sim_demo import execute_non_motion_skill

    def execute(step, agent_context: dict[str, Any]) -> dict[str, Any]:
        state = resources.context.merge_agent_context(agent_context)
        result = execute_non_motion_skill(
            step,
            resources.context.gripper_backend,
            context=state,
            perception_pipeline=resources.context.perception_pipeline,
            point_cloud_builder=resources.context.point_cloud_builder,
            grasp_planner=resources.context.grasp_planner,
        )
        output = result.get("output", {}) if isinstance(result, dict) else {}
        if isinstance(output, dict):
            if step.skill_id == "detect_object":
                selected = state.get("selected_object")
                if isinstance(selected, dict):
                    output.setdefault("object_id", selected.get("object_id"))
            elif step.skill_id == "generate_grasp_pose":
                candidate = state.get("selected_grasp_candidate")
                if isinstance(candidate, dict):
                    output.setdefault("candidate", candidate)
                    output.setdefault("candidate_id", candidate.get("candidate_id"))
        return result

    return execute


def create_candidate_reviewer(resources: RvizAgentResources):
    """Apply the existing full-demo candidate safety screen once per workflow."""
    from llm_skill_robot.ros_llm_rviz_sim_demo import _apply_full_demo_candidate_selection

    def review(step, rviz_context: RvizExecutionContext) -> dict[str, Any]:
        selector = rviz_context.runtime_state.get("full_demo_selector")
        if selector is None:
            return {"success": False, "message": "Full grasp candidate selector is unavailable.", "output": {}}
        result = _apply_full_demo_candidate_selection(rviz_context.runtime_state, selector)
        return {
            "success": bool(result.get("success", False)),
            "message": result.get("message", "Candidate review completed."),
            "output": result,
        }
    return review


def create_motion_executor(resources: RvizAgentResources, audit_logger):
    """Reuse the reviewed RViz motion path for one Agent primitive motion."""
    from llm_skill_robot.ros_llm_rviz_sim_demo import _grasp_motion_target_pose
    from llm_skill_robot.ros_nl_rviz_sim_demo import _run_motion_step

    def execute(step, rviz_context: RvizExecutionContext) -> dict[str, Any]:
        target_pose = None
        if step.skill_id != "move_to_named_target":
            grasping_config = rviz_context.runtime_state.get("grasping_config", {})
            transformer = rviz_context.runtime_state.get("transformer")
            target_pose = _grasp_motion_target_pose(
                step, rviz_context.runtime_state, grasping_config, transformer
            )
        return _run_motion_step(
            step=step, backend=rviz_context.motion_backend, node=rviz_context.node,
            execution_config=rviz_context.execution_config,
            trajectory_validator=rviz_context.trajectory_validator,
            trajectory_reviewer=rviz_context.trajectory_reviewer,
            audit_logger=audit_logger, target_pose=target_pose,
        )
    return execute
