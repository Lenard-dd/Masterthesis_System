#!/usr/bin/env bash
set -euo pipefail

# Launcher that ensures conda and ROS envs are active, then runs the test
# Usage: bash scripts/run_graspgenx_gripper_width_test.sh

# Activate conda env 'sam3'
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate sam3 || true
fi

# Source ROS and workspace setup
if [ -f "/opt/ros/humble/setup.bash" ]; then
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
fi
if [ -f "$HOME/dev_ws/install/setup.bash" ]; then
  # shellcheck source=/dev/null
  source "$HOME/dev_ws/install/setup.bash"
fi

echo "Using python: $(which python3)"
python3 -c "import sys, importlib; print(sys.version); import numpy; print('numpy', getattr(numpy,'__version__',None)); import cv_bridge; print('cv_bridge ok', cv_bridge.__file__)" || {
  echo "Failed to import cv_bridge or numpy in this environment." >&2
  exit 2
}

# Run the module entrypoint directly under the activated Python
python3 -m llm_skill_robot.ros_graspgenx_gripper_width_test
