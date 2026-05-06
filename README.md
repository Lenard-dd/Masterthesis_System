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

## Demo

Run the CLI demo:

```bash
python -m llm_skill_robot.main_cli
```

## Project Structure

- `config/`: Configuration files for skills, robot capabilities, and safety
- `llm_skill_robot/`: Main package
  - `core/`: Core logic for skills, plans, verification, and validation
  - `robot/`: Robot backend interfaces and mock implementations
  - `perception/`: Perception system interfaces and mock implementations
- `skills/`: YAML definitions of robot skills
- `tests/`: Unit tests
