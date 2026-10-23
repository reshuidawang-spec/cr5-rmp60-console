#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/source_runtime.sh"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cr5_sphere_workbench_mpl}"
export IP_address="${IP_address:-192.168.5.1}"
export DOBOT_TYPE="${DOBOT_TYPE:-cr5}"
mkdir -p "${MPLCONFIGDIR}"
WORKBENCH_LOG="/tmp/cr5_five_branch_workbench.log"

if [[ "${CR5_START_DRIVER:-1}" == 1 ]] && ! timeout 3 ros2 service list 2>/dev/null | grep -q "/dobot_bringup_ros2/srv/GetPose"; then
  nohup bash "${PROJECT_DIR}/scripts/start_dobot_driver.sh" \
    >/tmp/cr5_dobot_bringup_from_workbench.log 2>&1 &
fi

exec python3 scripts/five_branch_sphere_csv_workbench.py --runtime-root "${PROJECT_DIR}" >>"${WORKBENCH_LOG}" 2>&1
