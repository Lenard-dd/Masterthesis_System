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
