#!/usr/bin/env bash
set -eo pipefail
CR5_PUBLIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${CR5_PUBLIC_ROOT}/src/cr5_rmp60_measurement/scripts/install_runtime.sh" "$@"
