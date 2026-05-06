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

## ROS 2 Demo

Run the package as an ament_python node from a ROS 2 Humble workspace:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.bash
ros2 run llm_skill_robot llm-skill-demo
```

## Read UR5 Joint State

Read the current UR5 joint state from `/joint_states` without sending any robot commands:

```bash
cd ~/dev_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select llm_skill_robot --symlink-install
source install/setup.zsh
ros2 run llm_skill_robot ros-get-ur5-state
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
