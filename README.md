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

Use this flow when you want the UR5 arm and Robotiq 2F-140 gripper to appear in the same MoveIt/RViz model. This remains RViz/fake-hardware only: no real Robotiq driver is connected, no real gripper commands are published, and `real_gripper_enabled` stays `false`.

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

The Robotiq 2F-140 gripper skills are currently simulation-only and are intended for planning, verification, and RViz/fake-hardware flows. The gripper configuration lives in `config/gripper.yaml`, with `control_mode: rviz_sim` and `real_gripper_enabled: false`.

Available gripper skills are `open_gripper`, `close_gripper`, `set_gripper_width`, and `get_gripper_state`. Real gripper execution remains disabled: this project does not connect to the Robotiq driver or publish real gripper commands in this step.

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
