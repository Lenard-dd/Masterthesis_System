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
