#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/source_runtime.sh"
export IP_address="${IP_address:-192.168.5.1}"
export DOBOT_TYPE="${DOBOT_TYPE:-cr5}"
exec ros2 launch cr_robot_ros2 dobot_bringup_ros2.launch.py
