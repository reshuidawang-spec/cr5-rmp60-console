#!/usr/bin/env bash
set -eo pipefail
BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${CR5_RUNTIME_ROOT:-$(cd "${BRIDGE_DIR}/.." && pwd)}"
source "${RUNTIME_ROOT}/scripts/source_runtime.sh"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cr5_sphere_workbench_mpl}"
mkdir -p "${MPLCONFIGDIR}"
cd "${RUNTIME_ROOT}"
exec python3 "${BRIDGE_DIR}/five_branch_sphere_csv_workbench.py" --runtime-root "${RUNTIME_ROOT}" "$@"
