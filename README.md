# LLM Skill Robot — robot backend

This repository provides the ROS 2 and Python backend for the **Milo Robot
Collaboration Studio** in the companion `hitl_gui` repository. Together they
connect language planning, human review, D435i/SAM3 perception, GraspGenX grasp
candidates, MoveIt trajectory planning, and UR5/Robotiq execution.
The primary laboratory workflow performs supervised real grasping and placement;
simulation and mock backends remain available for development and testing.

**Start with the [GUI setup and operating guide](../hitl_gui/README.md).** It
contains installation, the current lab startup command, external hardware
requirements, and the first task workflow. Keep both repositories as siblings
under the same workspace's `src/` directory. The folder is `LLM_Ros`, but the
ROS package and Python import are named `llm_skill_robot`.

## How the system fits together

```text
Browser → hitl_gui.GuiController → agent bridge / runtime adapters
        → language plan → skill validation → human review
        → perception → grasp candidates → MoveIt plan → trajectory review
        → configured execution backend → verification / task receipts
```

The GUI shares the backend's skill definitions, configuration and execution
logic. CLI demos are additional entry points into this backend. Real execution
requires both the GUI execution flag and the backend hardware/safety gates.
The GUI also supports task-scoped Auto approval; see its operating guide.

## Reading order and repository layout

| Path | Responsibility |
|---|---|
| `llm_skill_robot/core/schemas.py` | Shared plan, trajectory and execution data |
| `skills/`, `core/skill_library.py` | YAML skill contracts and loading |
| `agent/` (inside `llm_skill_robot/`) | Language planning, tools, sessions and orchestration |
| `perception/` | RGB-D capture, segmentation adapter, localization and placement geometry |
| `grasping/` | GraspGenX adapter, grasp execution and verification |
| `robot/` | Mock, UR5/MoveIt and Robotiq backends |
| `safety/`, `calibration/` | Execution limits and tabletop calibration |
| `config/` | Lab-specific frames, model paths, targets and safety settings |
| `perception/skill_dispatch.py` | Shared non-motion skill execution |
| `grasping/candidate_preview.py` | Shared grasp poses, preview and MoveIt screening |
| `agent/scripted_client.py` | Offline scripted Agent client |
| `scripts/` | SAM3 subprocess and GraspGenX server/client runners |
| `tests/` | Regression tests, including safety and mocked ROS contracts |
| `docs/` | Focused operational references |

Except for `config/`, `scripts/`, `skills/`, `tests/` and `docs/`, the module
directories above live under `llm_skill_robot/`. Read the GUI's `app_state.py`
and `gui_controller.py` alongside these modules to follow a task end to end.

## Configuration to review on another workstation

- `perception.yaml`: camera topics, optical frame, SAM3 environment and checkpoint.
- `grasping.yaml`: GraspGenX environment, server endpoint and grasp-to-tool transform.
- `ur5.yaml`: joint names, `base_link`, `tool0`, and `ur5_manipulator` group.
- `named_targets.yaml`: recorded joint configurations for this lab robot.
- `hardware_mode.yaml`, `real_arm_safety.yaml`, `tabletop_safety.yaml`,
  `tabletop_calibration.yaml`, `safety.yaml`, `execution.yaml`: execution gates,
  workspace limits and calibration. Revalidate them for another physical setup.
- `gripper.yaml`, `gripper_joint_mapping.yaml`: driver interfaces and width mapping.
- `llm.yaml`: provider/model settings; credentials come from `OPENAI_API_KEY`.

Robot descriptions, meshes, MoveIt configuration and hardware launch packages
were reused from the laboratory computer. They are **external dependencies**,
not implemented or distributed by these two repositories. See the
[external dependency guide](docs/external_dependencies.md) for sources, observed
versions, model environments and the exact hardware/model call relationships.

Set `LLM_SKILL_CONFIG_DIR` to a complete external copy of `config/` to use
workstation-specific configuration without editing Git files. An invalid
override raises an error. The [GUI guide](../hitl_gui/README.md) describes the
matching GUI override and read-only installation checker.

## Development and diagnostics

From the configured ROS-compatible Python environment:

```bash
cd ~/dev_ws/src/LLM_Ros
python -m pip install -e '.[dev,llm]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

Tests use mocks for hardware interfaces; a passing suite does not establish
real robot or model performance. Do not run interactive real-hardware CLI
commands as automated tests.

- [Standalone diagnostics and demos](docs/diagnostics.md)
- [Sparse grasp strategy](docs/graspgenx_sparse.md)
- [Repository cleanup review](docs/repository_review.md)

Documentation and explanatory comments are in English. Chinese and other
language literals in parsers and tests intentionally preserve multilingual
instruction support.
