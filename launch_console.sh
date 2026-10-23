#!/usr/bin/env bash
set -eo pipefail
CR5_PUBLIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Open the console without starting a robot driver unless explicitly requested.
export CR5_START_DRIVER="${CR5_START_DRIVER:-0}"
exec bash "${CR5_PUBLIC_ROOT}/src/cr5_rmp60_measurement/scripts/launch_five_branch_sphere_workbench.sh" "$@"
