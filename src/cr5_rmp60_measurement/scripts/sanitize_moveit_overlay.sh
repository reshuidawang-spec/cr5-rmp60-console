#!/usr/bin/env bash
# The workspace contains an old locally-built MoveIt OMPL package (2.5.9),
# while /opt/ros/humble currently provides MoveIt 2.5.10.  Keep the Dobot
# overlay, but remove only that stale package from runtime lookup paths so
# move_group loads the matching system plugin.

remove_exact_path_entry() {
  local variable_name="$1"
  local rejected_entry="$2"
  local current_value="${!variable_name-}"
  local rebuilt=""
  local entry
  local old_ifs="${IFS}"
  IFS=':'
  for entry in ${current_value}; do
    if [ -n "${entry}" ] && [ "${entry}" != "${rejected_entry}" ]; then
      if [ -n "${rebuilt}" ]; then
        rebuilt="${rebuilt}:${entry}"
      else
        rebuilt="${entry}"
      fi
    fi
  done
  IFS="${old_ifs}"
  export "${variable_name}=${rebuilt}"
}

STALE_MOVEIT_PREFIX="${CR5_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/install/moveit_planners_ompl"
remove_exact_path_entry AMENT_PREFIX_PATH "${STALE_MOVEIT_PREFIX}"
remove_exact_path_entry CMAKE_PREFIX_PATH "${STALE_MOVEIT_PREFIX}"
remove_exact_path_entry COLCON_PREFIX_PATH "${STALE_MOVEIT_PREFIX}"
remove_exact_path_entry LD_LIBRARY_PATH "${STALE_MOVEIT_PREFIX}/lib"
unset STALE_MOVEIT_PREFIX
unset -f remove_exact_path_entry
