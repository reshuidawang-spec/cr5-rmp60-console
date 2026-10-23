#!/usr/bin/env python3
"""Check imports, ROS assets, xacro and native geometry without robot connections."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []

    def check(label, action):
        try:
            action()
            print(f"OK   {label}")
        except Exception as exc:
            failures.append(label)
            print(f"FAIL {label}: {exc}")

    for name in ("numpy", "scipy", "yaml", "PyQt5.QtWidgets", "matplotlib",
                 "lxml.etree", "rclpy", "xacro", "ament_index_python.packages",
                 "moveit_msgs.srv", "sensor_msgs.msg", "geometry_msgs.msg",
                 "trajectory_msgs.msg", "rcl_interfaces.srv", "rosidl_runtime_py",
                 "dobot_msgs_v4.srv", "moveit_configs_utils"):
        check(name, lambda name=name: importlib.import_module(name))
    for name in ("g++", "pkg-config", "ros2", "colcon"):
        def command(name=name):
            if not shutil.which(name):
                raise RuntimeError("command missing from PATH")
        check(name, command)

    def ros_assets():
        from ament_index_python.packages import get_package_share_directory
        for name in ("cr5_moveit", "dobot_rviz", "cr_robot_ros2", "geometric_shapes",
                     "eigen_stl_containers", "moveit_ros_move_group", "moveit_planners_ompl"):
            get_package_share_directory(name)
    check("ROS package assets", ros_assets)

    def models():
        import xacro
        for name in ("cr5_with_rmp60.urdf.xacro", "cr5_moveit_with_rmp60.urdf.xacro"):
            model = xacro.process_file(str(ROOT / "urdf" / name), mappings={
                "initial_positions_file": str(ROOT / "config/moveit/rmp60_initial_positions.yaml")})
            if len(model.getElementsByTagName("collision")) < 10:
                raise RuntimeError(f"missing collision geometry: {name}")
    check("portable robot xacro models", models)

    def native():
        from native_collision import NativeCollision
        import numpy as np
        checker = NativeCollision()
        try:
            a = checker.shape(1, [1.])
            b = checker.shape(1, [1.])
            ta = np.eye(4); tb = np.eye(4); tb[0, 3] = 3.
            if abs(checker.distance(a, ta, b, tb) - 1.) > 1e-6:
                raise RuntimeError("unexpected native distance result")
        finally:
            checker.close()
    check("FCL native compilation and distance", native)

    def defaults():
        from sphere_motion_contract import load_policy
        if load_policy()["commissioning"].get("verified") is not False:
            raise RuntimeError("shared policy must leave real collection disabled")
        json.loads((ROOT / "config/five_branch_workbench_profile.json").read_text())
    check("distribution configuration", defaults)
    print(f"\n{len(failures)} failed checks; no driver or robot motion was started.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
