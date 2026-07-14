# LLM Skill Robot

An LLM-based skill composition and human-in-the-loop validation system for a UR5 robot with gripper.

This project provides a pure Python core that can be unit-tested, with ROS 2 integration planned for future development.

## Setup

Install the package in development mode:

```bash
pip install -e ".[dev]"
```

## Testing

Run the test suite:

```bash
pytest
```

## Pure Python Demo

Run the CLI demo in a Python virtual environment:

```bash
source .venv/bin/activate
pytest
python -m llm_skill_robot.main_cli
```

## MoveIt Plan-Only Demo

Preview a dry-run MoveIt plan without executing robot motion:

```bash
cd ~/dev_ws/src/LLM_Ros
source .venv/bin/activate
python -m llm_skill_robot.moveit_plan_demo
```

`pymoveit2` is the preferred MoveIt Python interface for ROS 2 Humble demos in this project.

## Perception Skeleton Demo

Exercise the perception pipeline without camera hardware:

```bash
cd ~/dev_ws/src/LLM_Ros
source .venv/bin/activate
python -m llm_skill_robot.perception_demo
```

Run the same demo through the ROS 2 entry point:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot perception-demo
```

Expected current output is `NOT_AVAILABLE` because D435i/SAM3 is not connected yet.

## D435i ROS 2 Camera Backend

Launch the RealSense D435i with RGB-D sync and aligned depth enabled:

```bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true
```

The perception config reads these expected topics from `config/perception.yaml`:

- RGB image: `/camera/camera/color/image_raw`
- aligned depth image: `/camera/camera/aligned_depth_to_color/image_raw`
- camera info: `/camera/camera/color/camera_info`

The D435i backend saves the latest RGB frame to
`data/perception_runtime/latest_rgb.png` and passes that path to the SAM3
subprocess adapter. Normal unit tests use mocks and do not require a running
ROS graph or a live camera.

For robot pick/place, the camera optical frame must also be connected to
`base_link` through TF. Without a valid camera-to-base transform, perception can
still detect objects and estimate an approximate camera-frame centroid, but
`pick_object` should remain blocked when `base_link` pose is required.

The RViz/LLM plan demos also route the `detect_object` skill through the
configured perception pipeline. With `perception.mode: d435i_sam3` and
`detector: sam3_subprocess`, `detect_object` uses the live D435i frame and runs
SAM3 through the conda subprocess adapter. The main ROS 2 environment still does
not import SAM3 directly. `pick_object` remains verification-blocked unless the
detected object has a verified 3D pose in `base_link`.

Run the live camera-to-SAM3 demo after launching the D435i separately:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
ros2 run llm_skill_robot ros-d435i-sam3-demo
```

## SAM3 Conda Subprocess Inference

SAM3 is expected to live in a separate conda environment named `sam3`.
The main ROS 2 / `LLM_Ros` environment must not import SAM3 directly; it keeps
`rclpy`, `cv_bridge`, MoveIt, and robot-side logic separate from model
inference. SAM3 inference is called through JSON over a subprocess boundary.

Run the conda-side runner directly:

```bash
cd ~/dev_ws/src/LLM_Ros
conda run -n sam3 python scripts/sam3_runner.py --image data/perception_samples/sample_001/rgb.png --query "red cube"
```

Run the main-project single-image adapter test:

```bash
cd ~/dev_ws/src/LLM_Ros
source .venv/bin/activate
python -m llm_skill_robot.sam3_subprocess_single_image_test
```

## Natural Language Plan Demo

Convert a natural-language instruction into a structured `RobotPlan`, verify it, and ask for human approval. This demo does not call a real LLM API, MoveIt, or robot execution.

```bash
cd ~/dev_ws/src/LLM_Ros
source .venv/bin/activate
python -m llm_skill_robot.nl_plan_demo
```

Run the same demo through the ROS 2 entry point:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot nl-plan-demo
```

Run the natural-language RViz/fake-hardware simulation demo:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-nl-rviz-sim-demo
```

## LLM Plan Demo

Use a real LLM client to generate `RobotPlan` JSON, verify it, optionally ask the LLM to repair verification errors, and then ask for human validation. This demo does not call MoveIt or execute robot motion.

```bash
cd ~/dev_ws/src/LLM_Ros
source .venv/bin/activate
pip install -e ".[dev,llm]"
export OPENAI_API_KEY=...
python -m llm_skill_robot.llm_plan_demo
```

Run the same demo through the ROS 2 entry point:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
export OPENAI_API_KEY=...
ros2 run llm_skill_robot llm-plan-demo
```

Run the LLM-generated plan through the RViz/fake-hardware simulation pipeline:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
export OPENAI_API_KEY=...
ros2 run llm_skill_robot ros-llm-rviz-sim-demo
```

## Unified UR5 + Robotiq MoveIt/RViz Flow

Use this flow when you want the UR5 arm and Robotiq 2F-140 gripper to appear in the same MoveIt/RViz model. This flow remains RViz/fake-hardware only: it uses `Robotiq2F140SimBackend`, so no real Robotiq commands are published even when the separate hardware configuration enables the real-gripper backend.

Build the updated packages:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select ur5_robotiq_description ur5_robotiq_moveit_config llm_skill_robot --symlink-install
source install/setup.zsh
```

Terminal 1 starts UR fake hardware with the combined UR5 + Robotiq robot description:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 \
  robot_ip:=192.168.56.101 \
  use_fake_hardware:=true \
  launch_rviz:=false \
  description_package:=ur5_robotiq_description \
  description_file:=ur5_robotiq_2f140.urdf.xacro
```

Terminal 2 starts MoveIt and RViz with the matching combined SRDF:

```bash
ros2 launch ur5_robotiq_moveit_config ur5_robotiq_moveit.launch.py
```

Terminal 3 runs the demo. Arm motion continues through the existing MoveIt/RViz fake-hardware path, while gripper skills publish the simulated `robotiq_finger_joint` state for RViz visualization:

```bash
ros2 run llm_skill_robot ros-nl-rviz-sim-demo
```

For LLM-generated plans:

```bash
export OPENAI_API_KEY=...
ros2 run llm_skill_robot ros-llm-rviz-sim-demo
```

Do not run `joint_state_publisher_gui` at the same time for `robotiq_finger_joint`; duplicate `/joint_states` publishers can make RViz flicker or ignore the `open_gripper`, `close_gripper`, and `set_gripper_width` updates.

Use the provider-selectable natural language model interface directly in Python:

```python
from llm_skill_robot.agent.llm_interface import NaturalLanguageModelInterface
from llm_skill_robot.core.skill_library import SkillLibrary

skills = SkillLibrary()
llm = NaturalLanguageModelInterface(provider="openai")
llm.set_api_key("your_api_key_here")

plan = llm.generate_robot_plan(
    "move to observe and open the gripper",
    available_skills=[
        skills.get_skill(skill_id)
        for skill_id in skills.list_skills()
    ],
)
print(plan.model_dump())
```

`ChatGPTInterface` remains available as a backward-compatible shortcut for the OpenAI/ChatGPT provider. Other providers can be added behind `NaturalLanguageModelInterface` without changing `LLMPlanGenerator`.

## ROS 2 Demo

Run the package as an ament_python node from a ROS 2 Humble workspace:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.bash
ros2 run llm_skill_robot llm-skill-demo
```

Run the same plan-only demo through ROS 2 packaging:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot moveit-plan-demo
```

Run the real MoveIt plan-only demo without executing robot motion:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-moveit-plan-only-demo
```

Run the RViz/fake-hardware simulated execution demo:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-rviz-sim-execution-demo
```

After you approve the text trajectory review, the demo attempts to publish a visual trajectory preview to RViz on `/display_planned_path`. You must still inspect RViz and give the final confirmation before any RViz/fake-hardware simulated execution is requested.

## Milestone 1 RViz Simulation Demo Checklist

Use this checklist when recording or documenting the thesis Milestone 1 demo:

- Confirm the terminal prints `RVIZ_SIM MODE ONLY. This demo is not intended for real robot execution.`
- Confirm `config/execution.yaml` keeps `allow_real_robot_execution: false`.
- Build and source the ROS 2 workspace before running the demo.
- Start RViz/fake hardware in a separate terminal if visual inspection is required.
- Run `ros2 run llm_skill_robot ros-rviz-sim-execution-demo`.
- Confirm the generated plan is `move_to_named_target` with `target=home`, `velocity_scale=0.03`, and `acceleration_scale=0.03`.
- Confirm plan verification returns `ALLOW`.
- Approve the human validation prompt only after checking the printed plan.
- Confirm the Motion Plan Summary is printed and includes the cached `plan_id`.
- Confirm trajectory validation returns `ALLOW`.
- Read the trajectory review and approve it only if the plan is acceptable for RViz/fake hardware.
- Confirm the visual preview is published to `/display_planned_path`.
- Inspect RViz after the message `Please inspect RViz. No motion has been executed yet.`
- Give final confirmation only after RViz inspection.
- Confirm the Execution Report says `Mode: rviz_sim`.
- Save or reference the printed audit log path. The audit log should include `plan_generated`, `plan_verified`, `human_validated`, `motion_planned`, `trajectory_validated`, `trajectory_reviewed`, `visual_preview_published`, `final_confirmed`, and `simulated_execution_result`.

## Robotiq 2F-140 Gripper Skills

The project provides separate simulation and real-hardware Robotiq backends. The real backend is `Robotiq2F140RealBackend`; it remains fail-closed until its driver adapter is explicitly configured in `config/gripper.yaml`.

Available gripper skills are `open_gripper`, `close_gripper`, `set_gripper_width`, and `get_gripper_state`. Every real command must be explicitly human-confirmed (`confirmed=True` or `human_confirmed=True`) and must pass the width, speed, and force limits in `config/hardware_mode.yaml`. Validation or confirmation failure never publishes a command. `close_gripper()` defaults to the configured `safe_close_width_m` of `0.04 m`; it does not silently command `0.0 m` on real hardware.

Supported adapter modes are `dry_run`, `topic_float`, explicitly mapped `topic_joint_trajectory`, and `action_gripper_command`. The current hardware configuration uses the discovered `control_msgs/action/GripperCommand` action `/gripper_controller/gripper_cmd` and subscribes to `/gripper_state` as `std_msgs/msg/Float32`. In this installed driver, both action position and state are `0.0–0.7 rad` finger-joint positions (`0.0` open, `0.7` closed), so the backend explicitly converts them to and from the public `0.0–0.14 m` gap-width API. Configure only interfaces confirmed by the diagnostic. Because `GripperCommand` has no speed field, `driver.controller_speed_mps` must match the real controller/driver setting and is checked against `hardware_mode.yaml` before sending. Before operating it, inspect the ROS graph without moving the gripper:

```bash
ros2 run llm_skill_robot ros-check-real-gripper
```

This diagnostic prints the active hardware safety settings, the driver configuration, and matching topics/actions/services containing `robotiq`, `gripper`, or `finger`. It never constructs the real backend or sends a command. UR5 arm motion is controlled separately by the restricted real-arm safety configuration.

After configuring and checking the installed driver, start the interactive real-gripper test:

```bash
ros2 run llm_skill_robot ros-check-real-gripper
ros2 run llm_skill_robot ros-real-gripper-test
```

The test prints every active width, speed, and force limit before accepting input. It shows a command preview and requires the exact, case-sensitive response `YES` before every real command. Safe close uses `safe_close_width_m` (`0.04 m` by default), never an implicit fully closed `0.0 m`. The CLI contains no UR5 arm backend or arm command path.

GraspGenX is connected through a subprocess JSON runner configured in `config/grasping.yaml`. The default runner is `scripts/graspgenx_zmq_runner.py`; it connects to a GraspGenX ZMQ server at `localhost:5556` and asks for `robotiq_2f_140` candidates. Start the GraspGenX server before running the ROS demos, for example from the GraspGenX repository and its `graspgenx` conda environment:

```bash
conda activate graspgenx
cd ~/GraspGenX/GraspGenX
python client-server/graspgenx_server.py \
  --config ~/GraspGenX/GraspGenX/ext/graspgenx_checkpoints/release \
  --assets_dir ~/GraspGenX/GraspGenX/assets \
  --default_gripper robotiq_2f_140 \
  --port 5556
```

Then, to test only the width recommended by a GraspGenX candidate, run:

```bash
ros2 run llm_skill_robot ros-graspgenx-gripper-width-test
```

This flow detects an object with the real D435i and SAM3, builds its masked point cloud, selects the highest-scoring valid GraspGenX candidate, and reviews its `gripper_width_m`. The ZMQ GraspGenX API returns poses and scores; by default `scripts/graspgenx_zmq_runner.py` estimates `gripper_width_m` from the object point-cloud extent along the candidate gripper opening axis and marks the source as `point_cloud_projection_estimate` in candidate metadata. This is a geometric estimate, so the operator must review it and the value must still pass `hardware_mode.yaml` limits. It does not move to the candidate pose, execute MoveIt, or command any UR5 arm joint. The only possible hardware action is setting the real gripper width after the value passes safety limits and the operator types the exact response `YES`. Missing point clouds, candidates, or candidate widths stop the flow without a command.

For a MoveIt plan-only preview of the selected GraspGenX pose, run:

```bash
ros2 run llm_skill_robot ros-graspgenx-plan-only-demo
```

This demo derives pregrasp, grasp, and retreat poses, requests plan-only MoveIt trajectories, validates them, and publishes the resulting sequence to RViz for human review. It never calls MoveIt execution, never sends a trajectory to the UR5 controller, and never commands the real gripper. It remains plan-only even when `hardware_mode.yaml` enables supervised real-arm execution for a separate demo.

## Full supervised real grasp demo

Only after validating camera, TF, MoveIt, gripper feedback, and the plan-only workflow, start the real supervised grasp demo:

```bash
ros2 run llm_skill_robot ros-full-grasp-supervised-demo
```

This command can move the real UR5 and close the real Robotiq gripper on an object. It never offers a single “execute full pick” confirmation: every arm motion requires a new exact `EXECUTE`, and the contact close requires a separate exact `YES`. Planning, trajectory validation, trajectory review, candidate-score checks, and the full-grasp state machine remain mandatory. Any rejected confirmation, failed plan, invalid state transition, or insufficient verification evidence stops the sequence; `UNCERTAIN` verification is reported honestly rather than treated as success.

If the RViz preview looks twisted, overly complex, or ends at an unexpected gripper pose, check the frame convention before tuning MoveIt. GraspGenX reports a grasp frame, while MoveIt plans the UR5 end-effector frame from `config/ur5.yaml` (`tool0` by default). Configure the measured offset in `config/grasping.yaml`:

```yaml
plan_only_preview:
  grasp_frame_to_ee_frame:
    enabled: true
    translation_m: [0.0, 0.0, 0.0]
    rotation_rpy_rad: [0.0, 0.0, 0.0]
```

The convention is `T_base_ee = T_base_grasp * T_grasp_ee`. `translation_m` is expressed in the GraspGenX grasp frame, and `rotation_rpy_rad` is the roll/pitch/yaw from that grasp frame to MoveIt's `tool0` frame. Leave it disabled until you have measured/calibrated the Robotiq/tool0 offset.

## Unified hardware-safe perception and gripper demo

The unified demo uses the real D435i camera, SAM3 and GraspGenX subprocesses, an explicitly confirmed real-gripper width test, and MoveIt/RViz plan-only previews. It never executes real UR5 arm motion.

Terminal 1 — start the aligned real D435i streams:

```bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true
```

Terminal 2 — start UR fake hardware and MoveIt/RViz. Each launch remains running, so use two tabs within this terminal session:

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.56.101 use_fake_hardware:=true launch_rviz:=false
```

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_rviz:=true
```

Terminal 3 — inspect the real-gripper interfaces, then start the demo:

```bash
ros2 run llm_skill_robot ros-check-real-gripper
ros2 run llm_skill_robot ros-hw-perception-gripper-demo
```

The camera data is real. The real gripper may move only after the selected candidate has been reviewed and the operator enters the exact, case-sensitive response `YES`. All UR5 pregrasp, approach, and retreat paths are planning/RViz previews only; no trajectory is sent to the real UR5 controller.

## Robotiq RViz Joint Diagnostics

Before trying to animate the Robotiq 2F-140 in RViz, inspect the joint names being published on `/joint_states`:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-check-gripper-model
```

The diagnostic only reads `/joint_states`; it does not move the robot or publish gripper commands. It prints joints containing keywords such as `robotiq`, `finger`, `gripper`, `knuckle`, `inner`, or `outer`.

Use the printed joint names to fill `config/gripper_joint_mapping.yaml`:

```yaml
gripper_joint_mapping:
  enabled: true
  publish_topic: /joint_states
  publish_rate_hz: 20.0
  mode: rviz_sim
  real_gripper_enabled: false
  joint_names:
    - robotiq_finger_joint
  open_positions:
    robotiq_finger_joint: 0.0
  closed_positions:
    robotiq_finger_joint: 0.7
  max_width_m: 0.14
  min_width_m: 0.0
```

The currently detected active Robotiq 2F-140 joint is:

- `robotiq_finger_joint`

The other Robotiq finger and knuckle joints are mimic joints in the xacro model, so they are not shown as separate sliders in `joint_state_publisher_gui`. Publishing only `robotiq_finger_joint` lets `robot_state_publisher` compute the mimic joint transforms.

Only enable simulated gripper joint state publishing when `joint_state_publisher_gui` or another node is not already publishing the same gripper joints. Publishing the same joint names from two nodes can cause RViz flicker or inconsistent gripper poses.

## Robotiq 2F-140 RViz Visual Gripper Control

This is RViz/fake hardware visualization only. It does not connect to, command, or enable real Robotiq hardware.

First launch the UR5 + Robotiq visualization:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
ros2 launch ur5_robotiq_description view_ur5_robotiq.launch.py use_placeholder_gripper:=false
```

Then check the detected gripper joints:

```bash
ros2 run llm_skill_robot ros-check-gripper-model
```

Then run the interactive RViz gripper visual demo:

```bash
ros2 run llm_skill_robot ros-gripper-rviz-demo
```

Interactive commands:

- `o` = open
- `c` = close
- `w` = set width in meters
- `s` = state
- `q` = quit

The demo publishes only this active Robotiq gripper joint to `/joint_states`:

- `robotiq_finger_joint`

If RViz does not update, check `config/gripper_joint_mapping.yaml` against the output of `ros-check-gripper-model`. If RViz flickers or ignores updates, close `joint_state_publisher_gui` or avoid duplicate joint state publishers for `robotiq_finger_joint`. If open and close move in the wrong direction, adjust `closed_positions` in `config/gripper_joint_mapping.yaml`.

## Read UR5 Joint State

Read the current UR5 joint state from `/joint_states` without sending any robot commands:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-get-ur5-state
```

## Real UR5 Restricted Status Check

Before any restricted real-arm test, inspect the real UR5 driver, `/joint_states`, ros2_control controllers, MoveIt planning service, and active safety configuration:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-check-real-arm
```

The diagnostic is read-only. It subscribes to `/joint_states`, checks for the six UR5 joints, queries controller status when `/controller_manager/list_controllers` is available, and checks whether MoveIt's `/plan_kinematic_path` service is visible. It prints `REAL UR5 STATUS CHECK`, current joint positions, whether `hardware_mode.arm.allow_arm_execution` is enabled, and whether `real_arm_safety.execution_mode` is `restricted_real_motion`. It does not execute arm motion, send trajectories, or command the gripper.

## Real UR5 Restricted Single-Motion Test

After the read-only status check is clean and the robot workspace is physically clear, start the interactive restricted real-arm test:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-real-arm-safe-motion-test
```

This CLI can move the real UR5. It is limited to single named-target motions such as `safe_home` and `observe`, uses the low velocity and acceleration limits from `config/real_arm_safety.yaml`, and requires the exact phrase `EXECUTE` before every motion. It will not run a pick sequence and blocks `approach_grasp`, `pick_object`, and `lift_object`. Every requested, planned, rejected, confirmed, executed, and failed real-arm motion is written to `real_arm_motion_audit.log`.

## Real UR5 GraspGenX Pregrasp Demo

This demo uses the real D435i camera, SAM3, GraspGenX, and MoveIt to compute a candidate grasp, then allows one restricted real UR5 motion to the pregrasp pose only. It does not execute approach, contact, gripper close, lift, or `pick_object`.

Terminal 1 — start the aligned D435i streams:

```bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true
```

Terminal 2 — start the real UR driver and MoveIt for the UR5, using your lab's real robot IP and launch setup.

Terminal 3 — start the GraspGenX ZMQ server in the `graspgenx` environment, as shown in the GraspGenX section above.

Terminal 4 — build, source, run the read-only status check, then start the pregrasp demo:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-check-real-arm
ros2 run llm_skill_robot ros-graspgenx-pregrasp-real-demo
```

The demo asks for an object query, optionally opens the real gripper only if you type the exact response `YES`, plans to the selected pregrasp pose, prints the trajectory summary, and moves the real UR5 only if you type the exact phrase `EXECUTE`. It stops immediately after the pregrasp motion and prints `No approach, contact, close, or lift was executed.`

## Real UR5 Pregrasp + Candidate Width Test

After the pregrasp-only demo is stable, use this test to combine one restricted pregrasp motion with explicit real-gripper commands:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-pregrasp-gripper-real-test
```

The test detects an object, selects a GraspGenX candidate, prints its `candidate_id`, score, `gripper_width_m`, and pose, then asks separately before each real action. Type `YES` to open the real gripper, type `EXECUTE` to move the real UR5 to pregrasp, and type `YES` again to set the gripper to the selected candidate width. Missing widths, widths outside `hardware_mode.yaml` limits, or widths below `safe_close_width_m` are rejected. The test never calls `approach_grasp`, never closes on the object, never lifts, and never executes `pick_object`.

## Unified Restricted Real Hardware Demo

Use this menu-driven demo for incremental lab testing with the real D435i, SAM3, GraspGenX, real Robotiq gripper, and restricted real UR5 motion:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-restricted-real-hw-demo
```

Recommended launch sequence:

1. Start the aligned D435i streams.
2. Start the real UR driver, controllers, and MoveIt.
3. Start the GraspGenX ZMQ server.
4. Run `ros2 run llm_skill_robot ros-check-real-arm`.
5. Run `ros2 run llm_skill_robot ros-check-real-gripper`.
6. Run `ros2 run llm_skill_robot ros-restricted-real-hw-demo`.

The menu supports status checks, safe named-target moves, object detection, point-cloud building, GraspGenX candidate generation, candidate review, gripper open, UR5 move-to-pregrasp, candidate-width gripper test, and return to `safe_home`. It intentionally has no `approach_grasp`, no contact close, no lift, and no full `pick_object`. Every real arm motion requires exact `EXECUTE`; every real gripper command requires exact `YES`; each action is logged to `restricted_real_hw_demo_audit.log`.

## ROS Verification Demo

Read the current UR5 joint state and verify a hardcoded plan without moving the robot:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-verify-demo
```

## Mock Execution Pipeline

This demo uses a pure Python mock execution pipeline:

- skills are defined in YAML and loaded by `SkillLibrary`
- `PlanVerifier` checks each step for required parameters and allowed values
- `HumanValidator` prompts for approval before execution
- `MockRobotBackend` simulates robot state and skill execution
- `ExecutionManager` runs steps and produces an `ExecutionReport`
- `ExecutionEvaluator` summarizes whether the task succeeded

## Project Structure

- `config/`: Configuration files for skills, robot capabilities, and safety
- `llm_skill_robot/`: Main package
  - `core/`: Core logic for skills, plans, verification, and validation
  - `robot/`: Robot backend interfaces and mock implementations
  - `perception/`: Perception system interfaces and mock implementations
- `skills/`: YAML definitions of robot skills
- `tests/`: Unit tests
