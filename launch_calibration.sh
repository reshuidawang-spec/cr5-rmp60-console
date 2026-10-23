#!/usr/bin/env bash
set -eo pipefail
CR5_PUBLIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CR5_PUBLIC_ROOT}/src/cr5_rmp60_measurement/scripts/source_runtime.sh"
cd "${CR5_PUBLIC_ROOT}/src/cr5_rmp60_measurement"
exec /usr/bin/python3 scripts/probe_calibration_gui.py "$@"
