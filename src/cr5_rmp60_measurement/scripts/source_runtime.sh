#!/usr/bin/env bash
# Source this file; locate the workspace independently of the user's account.
export CR5_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CR5_WORKSPACE="${CR5_WORKSPACE:-$(cd "${CR5_PROJECT_DIR}/../.." && pwd)}"
export CR5_ROS_PREFIX="${CR5_ROS_PREFIX:-/opt/ros/humble}"
if [[ ! -f "${CR5_ROS_PREFIX}/setup.bash" || ! -f "${CR5_WORKSPACE}/install/setup.bash" ]]; then
  echo "Missing ROS/workspace setup. Follow docs/INSTALLATION.md first." >&2
  return 1
fi
source "${CR5_ROS_PREFIX}/setup.bash"
source "${CR5_WORKSPACE}/install/setup.bash"
source "${CR5_PROJECT_DIR}/scripts/sanitize_moveit_overlay.sh"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cr5_sphere_workbench_mpl}"
