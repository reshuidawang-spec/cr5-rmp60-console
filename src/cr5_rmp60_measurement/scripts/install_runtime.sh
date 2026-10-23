#!/usr/bin/env bash
# Explicit actions only. Installing/building never launches a robot driver.
set -eo pipefail
CR5_INSTALL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CR5_INSTALL_WORKSPACE="${CR5_WORKSPACE:-$(cd "${CR5_INSTALL_PROJECT}/../.." && pwd)}"
case "${1:---help}" in
  --system)
    source /etc/os-release
    if [[ "${ID}" != ubuntu || "${VERSION_ID}" != 22.04 ]]; then
      echo "Supported installation: Ubuntu 22.04 and ROS 2 Humble." >&2
      exit 1
    fi
    mapfile -t CR5_APT_PACKAGES < <(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "${CR5_INSTALL_PROJECT}/dependencies/apt-runtime.txt")
    sudo apt-get update
    sudo apt-get install -y "${CR5_APT_PACKAGES[@]}"
    ;;
  --build)
    source "${CR5_ROS_PREFIX:-/opt/ros/humble}/setup.bash"
    cd "${CR5_INSTALL_WORKSPACE}"
    colcon build --packages-select dobot_msgs_v4 cr_robot_ros2 dobot_rviz cr5_moveit \
      --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    ;;
  --check)
    source "${CR5_INSTALL_PROJECT}/scripts/source_runtime.sh"
    exec /usr/bin/python3 "${CR5_INSTALL_PROJECT}/scripts/check_runtime_environment.py"
    ;;
  *)
    echo "Usage: bash scripts/install_runtime.sh --system | --build | --check"
    echo "First configure the official ROS 2 Humble apt repository; see docs/INSTALLATION.md."
    ;;
esac
