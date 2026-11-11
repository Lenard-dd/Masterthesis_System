# Diagnostics and optional simulation

The primary application is the real-robot grasp/place GUI. Start with the
[GUI operating guide](../../hitl_gui/README.md) and
[external dependency inventory](external_dependencies.md).
The commands here isolate components for troubleshooting; they are not a
second mandatory startup sequence. Run them from a sourced ROS Humble
workspace with `ROS_DOMAIN_ID=27` and the appropriate Python environment.

## Read-only and plan-only checks

| Command (`ros2 run llm_skill_robot …`) | Purpose |
|---|---|
| `ros-check-real-arm` | Inspect robot/controller readiness without motion |
| `ros-check-real-gripper` | Inspect gripper interfaces without commands |
| `ros-get-ur5-state` | Read joint state |
| `ros-check-gripper-model` | Inspect gripper model/joint mapping |
| `ros-graspgenx-plan-only-demo` | Generate and preview grasp motion without execution |
| `ros-d435i-sam3-demo` | Exercise camera/segmentation, requiring the camera and model |
| `sam3-subprocess-single-image-test` | Exercise segmentation on an image; enter a local image path when prompted |

Camera topics, model paths and TF must match configuration. On the lab GPU,
retain the SAM3/GraspGenX handoff; do not keep both models loaded manually.

## Interactive real-hardware diagnostics

These entry points can command hardware and retain their own confirmation and
safety gates. Do not run them as automated unit tests.

| Command suffix | Scope |
|---|---|
| `ros-real-arm-safe-motion-test` | Restricted named-target arm motion |
| `ros-real-gripper-test` | Gripper commands |
| `ros-graspgenx-gripper-width-test` | Candidate width inspection and confirmed gripper command |
| `ros-graspgenx-pregrasp-real-demo` | Real pregrasp motion |
| `ros-pregrasp-gripper-real-test` | Pregrasp plus candidate-width command |
| `ros-restricted-real-hw-demo` | Incremental supervised hardware operations |
| `ros-full-grasp-supervised-demo` | Staged real grasp with interactive confirmations |
| `ros-llm-full-grasp-supervised-demo` | Language-generated staged real grasp |
| `ros-hw-perception-gripper-demo` | Isolated perception/gripper workflow |
| `ros-tabletop-calibration` | Tabletop calibration utility; review the resulting configuration |

The separate CLI workflows are maintained diagnostics. Their confirmation
prompts and orchestration are not identical to the GUI's task-scoped review/Auto
behavior. Follow the GUI guide for the final application workflow.

## Optional mock and fake-hardware use

For a first look, use the explicit mock command in the GUI README. For MoveIt
simulation use the GUI's fake-hardware component controls with the lab
`ur5_launcher`; never connect simulation validation to a real controller.

Additional retained examples include `llm-skill-demo`, `nl-plan-demo`,
`llm-plan-demo`, `perception-demo`, `agent-runtime-demo`,
`ros-nl-rviz-sim-demo`, `ros-llm-rviz-sim-demo`,
`ros-rviz-sim-execution-demo`, `ros-agent-runtime-rviz-sim-demo` and
`ros-gripper-rviz-demo`. Language examples using the real provider need the
configured API key. Some examples use simulated robot/perception behavior by
design; that does not describe the primary real-robot application.

Use `ros2 pkg executables llm_skill_robot` to inspect the installed CLI list.
Rebuild/source after removing old entry points or changing package metadata.
The obsolete standalone `moveit_plan_demo`, `ros_moveit_plan_only_demo` and
`ros_verify_demo` modules were retired; use the checks above instead.
