# Handoff cleanup and maintenance plan

Updated 2026-11-11. The primary application is supervised **real UR5/Robotiq
grasping and placement**. Simulation and mock workflows remain supported for
inspection and regression testing. Real execution still requires explicit
session opt-in and the existing safety/review gates.

## Completed

- Reorganized both READMEs around the real laboratory workflow, installation,
  operation and code structure. Removed stale prototype descriptions from GUI
  package/state/panel/launch docstrings.
- Added [external dependencies](external_dependencies.md): laboratory source
  locations, hardware/model call relationships, observed versions and model
  acquisition requirements, including the local `ur_rtde` dependency.
- Reduced [diagnostics](diagnostics.md) to the maintained CLI commands and
  optional simulation entry points. Removed obsolete startup instructions.
- Corrected [sparse strategy](graspgenx_sparse.md) documentation to match the
  current zero-height sampling, environment/closed-finger collision checks,
  backoff search and client-side quota handling.
- Removed unused `perception/sam3_adapter.py`,
  `perception/mock_perception.py` and `robot/base_ros_backend.py` from
  `llm_skill_robot/`. Removed only the tests specific to those old placeholders;
  the unavailable-perception test now checks the actual subprocess adapter.
- Retired `moveit_plan_demo.py`, `ros_moveit_plan_only_demo.py` and
  `ros_verify_demo.py`: no current GUI imports or registered console entry
  points depended on them. Use the maintained read-only and grasp-plan preview
  diagnostics instead.
- Consolidated nine tiny import-test files into `tests/test_cli_imports.py`;
  seven checks remain after retiring two obsolete demo modules. Files containing
  additional behavior tests were retained.
- Isolated the trajectory dictionary conversion test with scoped fake ROS
  message types, removing an import-state dependency exposed by combined runs.
- Renamed GUI historical test files to `test_simulated_grasp_closed_loop.py`
  and `test_session_configuration_and_receipts.py` without removing coverage.
- Updated two component-start tests to expect on-demand GraspGenX startup.
  Updated the real-gripper routing test with an owning pick/place task and the
  current post-release observation request. Its synchronous fake backend runs
  inline to isolate this unit test from application thread-pool shutdown;
  production threading behavior is unchanged.
- Earlier cleanup removed two generated PNGs and two machine-specific editor
  files, added ignore rules and declared missing Python/ROS dependencies.
  Temporary recovery copies of those four artifacts are in
  `/tmp/robot-handoff-local-artifacts/` on the review machine.

No grasp algorithm, hardware calibration, safety limits or execution behavior
was intentionally changed. The removed modules remain recoverable from Git
history. No commit or push was performed as part of this cleanup.

## Runtime separation and portability follow-up

- GUI imports now target `perception/skill_dispatch.py`,
  `grasping/candidate_preview.py` and `agent/scripted_client.py`, not demo modules.
  Function bodies were relocated with compatibility exports for existing CLIs;
  candidate screening and motion decisions were not redesigned.
- `LLM_SKILL_CONFIG_DIR` selects a complete backend config directory;
  `HITL_GUI_CONFIG_FILE` selects a complete GUI YAML file. Defaults are preserved.
  Missing overrides fail explicitly rather than silently selecting lab defaults.
- `hitl-gui-check` / `python -m hitl_gui.diagnostics` performs read-only installation
  checks. `--live` adds lab resource checks and a bounded conda environment listing;
  it does not prove hardware readiness or invoke model inference.

## Retained intentionally

- `agent/mock_perception.py` and GUI mock backends support regression testing;
  they are distinct from the unused empty-scene placeholder removed above.
- Diffusion generation remains part of the grasp pipeline and a fallback for
  sparse requests. Legacy collision modes and schema aliases still have active
  configuration or compatibility roles.
- Plan validation, trajectory approval/cache binding, real-execution gating,
  gripper confirmation, contact feedback and tabletop/collision tests protect
  the real workflow. Simulation in a test fixture does not make its assertions
  obsolete.
- Logs, receipts and experiment metrics are current application features.
  Their generated files, model outputs, build artifacts and caches stay out of Git.
- `docs/thesis_project_summary/` is already untracked/ignored. It need not be
  shipped; archive/delete the local thesis material only when no longer needed.

## Next changes, in order

1. **Review CLI compatibility exports.** Shared GUI skill dispatch, grasp
   candidate preview/screening and the scripted client now live in runtime
   modules. Existing CLI modules re-export their former names for compatibility.
   Migrate remaining CLI/test imports gradually before removing those exports.
2. **Review remaining standalone CLI workflows by purpose.** Preserve robot
   status, calibration, camera/model isolation and plan-only preview checks.
   Consider retiring menu-driven or early language demos when their remaining
   troubleshooting value is covered by the GUI. Remove entry points, docs and
   exclusive tests together; do not delete by filename alone.
3. **Organize tests by behavior.** Group real execution contracts, shared
   planning/perception logic and optional simulation tests. Keep simulated
   fixtures where they verify real safety contracts. Add a bounded test harness
   for application executor/lifecycle behavior; the routing unit test does not
   validate thread-pool shutdown or real ROS timing.
4. **Validate destination installation.** Full configuration overrides and a
   read-only installation checker are available. Next record source snapshots
   for unversioned lab packages and validate installation on a new workstation.
   A complete environment lock and live inference checks remain separate work.
5. **Migrate historical config names only with compatibility.** `phase9` and
   related stored fields still have runtime consumers. A future rename should
   load old configuration/receipts and include migration tests.

Author/maintainer metadata now identifies Zhufeng Que
(`zhufeng.que@rwth-aachen.de`). Before public redistribution, confirm the
intended license text. Both application packages declare MIT but have no tracked
license file; laboratory hardware/model assets have separate provenance.

## Validation

The latest combined run passes with `/usr/bin/python3` (Python 3.10):

- **803 passed**, with no excluded tests (600 backend and 203 GUI tests).
- The earlier cleanup passed 795 tests; eight new configuration/diagnostics
  checks cover explicit override selection, invalid-path behavior, read-only
  execution and credential redaction.
- AST comparison confirmed all 38 relocated function/class definitions match
  their original implementations. GUI source has no direct demo imports.
- Python syntax, XML/YAML parsing, local Markdown links and `git diff --check`
  passed.
- Basic installation checks passed. `--live` also passed after sourcing ROS
  Humble and the lab workspace and setting `ROS_DOMAIN_ID=27`. An unsourced
  terminal correctly reported missing ROS package indexing/domain settings.
  The checker did not start any hardware or model process.

From the sourced workspace/environment, run the suites separately:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider src/LLM_Ros/tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider src/hitl_gui/test
```

A final upload review built both wheels offline from copies containing only the
intended Git files, using setuptools 82.0.1. It found and fixed missing backend
console scripts by declaring `project.dynamic = ["scripts"]` in pyproject.toml.
All 27 backend entry points and their modules are now present in the wheel;
the GUI wheel contains both `hitl_gui` and `hitl-gui-check`. Runtime resources
and the extracted modules were also checked in the built packages.
A filename/content-pattern scan found no generated build/cache files or matching
private-key/API-token patterns in the intended upload snapshot. This is not a
full Git-history secret audit.

Fresh dependency installation, colcon build, live model inference and physical
robot operation were not performed during cleanup. Passing mocked regression
tests does not independently validate physical grasp performance.
