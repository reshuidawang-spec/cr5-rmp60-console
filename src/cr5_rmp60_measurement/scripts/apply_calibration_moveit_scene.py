#!/usr/bin/env python3
"""Apply calibration-sphere workcell collision objects to MoveIt."""
import argparse
import fcntl
import os
import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_FIT = PROJECT_DIR / "data/2026.6.16/yneg_near_fourth_refit_20260616.json"
OBJECT_IDS = (
    "calibration_table",
    "calibration_sphere",
    "calibration_sphere_stem",
    "calibration_magnetic_base",
)


def load_sphere_center(path):
    """Load sphere center."""
    data = json.loads(Path(path).read_text())
    center = data.get("sphere_center_mm")
    if not isinstance(center, list) or len(center) != 3:
        raise ValueError(f"{path}: missing 3-value sphere_center_mm")
    return [float(value) for value in center]


def pose_xyz_mm(x, y, z):
    """Pose xyz mm."""
    pose = Pose()
    pose.position.x = float(x) * 0.001
    pose.position.y = float(y) * 0.001
    pose.position.z = float(z) * 0.001
    pose.orientation.w = 1.0
    return pose


def quaternion_from_matrix(matrix):
    """Return normalized xyzw quaternion for a 3x3 rotation matrix."""
    m = [[float(value) for value in row] for row in matrix]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * scale
        qx = (m[2][1] - m[1][2]) / scale
        qy = (m[0][2] - m[2][0]) / scale
        qz = (m[1][0] - m[0][1]) / scale
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw = (m[2][1] - m[1][2]) / scale
        qx = 0.25 * scale
        qy = (m[0][1] + m[1][0]) / scale
        qz = (m[0][2] + m[2][0]) / scale
    elif m[1][1] > m[2][2]:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw = (m[0][2] - m[2][0]) / scale
        qx = (m[0][1] + m[1][0]) / scale
        qy = 0.25 * scale
        qz = (m[1][2] + m[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw = (m[1][0] - m[0][1]) / scale
        qx = (m[0][2] + m[2][0]) / scale
        qy = (m[1][2] + m[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    return [qx / norm, qy / norm, qz / norm, qw / norm]


def pose_from_scene(center_mm, rotation=None):
    pose = pose_xyz_mm(*center_mm)
    if rotation is not None:
        qx, qy, qz, qw = quaternion_from_matrix(rotation)
        pose.orientation.x, pose.orientation.y = qx, qy
        pose.orientation.z, pose.orientation.w = qz, qw
    return pose


def primitive_box_mm(x, y, z):
    """Primitive box mm."""
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [float(x) * 0.001, float(y) * 0.001, float(z) * 0.001]
    return primitive


def primitive_sphere_mm(radius):
    """Primitive sphere mm."""
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [float(radius) * 0.001]
    return primitive


def primitive_cylinder_mm(height, radius):
    """Primitive cylinder mm."""
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.CYLINDER
    primitive.dimensions = [float(height) * 0.001, float(radius) * 0.001]
    return primitive


def collision_object(object_id, frame_id, primitive, pose, operation=CollisionObject.ADD):
    """Collision object."""
    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = object_id
    obj.primitives.append(primitive)
    obj.primitive_poses.append(pose)
    obj.operation = operation
    return obj


def remove_object(object_id, frame_id):
    """Remove object."""
    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = object_id
    obj.operation = CollisionObject.REMOVE
    return obj


def build_objects(args):
    """Build objects."""
    sphere_center = load_sphere_center(args.reference_fit_json)
    objects = []

    table_thickness = args.table_thickness_mm
    table_center_z = args.table_z_mm - table_thickness * 0.5
    objects.append(
        collision_object(
            "calibration_table",
            args.frame_id,
            primitive_box_mm(args.table_size_x_mm, args.table_size_y_mm, table_thickness),
            pose_xyz_mm(args.table_center_x_mm, args.table_center_y_mm, table_center_z),
        )
    )

    objects.append(
        collision_object(
            "calibration_sphere",
            args.frame_id,
            primitive_sphere_mm(args.sphere_radius_mm),
            pose_xyz_mm(*sphere_center),
        )
    )

    if args.stem_center_mm is not None:
        stem_center = [float(value) for value in args.stem_center_mm]
        stem_height = float(args.stem_length_mm)
        stem_radius = float(args.stem_diameter_mm) / 2.0
    else:
        stem_top_z = sphere_center[2] - args.sphere_radius_mm
        stem_height = max(0.0, stem_top_z - args.table_z_mm)
        stem_radius = float(args.stem_radius_mm)
        stem_center = [sphere_center[0], sphere_center[1], args.table_z_mm + stem_height * 0.5]
    if stem_height > 1e-6:
        objects.append(
            collision_object(
                "calibration_sphere_stem",
                args.frame_id,
                primitive_cylinder_mm(stem_height, stem_radius),
                pose_xyz_mm(*stem_center),
            )
        )

    if args.magnetic_base_size_mm:
        if len(args.magnetic_base_size_mm) != 3 or len(args.magnetic_base_center_mm) != 3:
            raise ValueError("--magnetic-base-size-mm and --magnetic-base-center-mm must each contain 3 values")
        size = args.magnetic_base_size_mm
        center = args.magnetic_base_center_mm
        objects.append(
            collision_object(
                "calibration_magnetic_base",
                args.frame_id,
                primitive_box_mm(size[0], size[1], size[2]),
                pose_xyz_mm(center[0], center[1], center[2]),
            )
        )

    return objects


def build_objects_from_scene(path, fallback_frame_id="base_link"):
    """Build collision objects from the same JSON scene shown by the GUI."""
    scene = json.loads(Path(path).read_text())
    frame_id = str(scene.get("frame_id", fallback_frame_id))
    required = ("table_center", "table_rotation", "table_size", "table_thickness",
                "sphere_center", "sphere_radius", "base_center", "base_rotation", "base_size")
    missing = [key for key in required if key not in scene]
    if missing:
        raise ValueError("scene JSON missing: " + ", ".join(missing))
    objects = [
        collision_object(
            "calibration_table", frame_id,
            primitive_box_mm(*scene["table_size"], scene["table_thickness"]),
            pose_from_scene(scene["table_center"], scene["table_rotation"]),
        ),
        collision_object(
            "calibration_sphere", frame_id, primitive_sphere_mm(scene["sphere_radius"]),
            pose_from_scene(scene["sphere_center"]),
        ),
        collision_object(
            "calibration_magnetic_base", frame_id, primitive_box_mm(*scene["base_size"]),
            pose_from_scene(scene["base_center"], scene["base_rotation"]),
        ),
    ]
    stem = scene.get("sphere_stem")
    if stem and float(stem.get("length_mm", 0)) > 0:
        objects.append(collision_object(
            "calibration_sphere_stem", frame_id,
            primitive_cylinder_mm(stem["length_mm"], float(stem["diameter_mm"]) / 2.0),
            pose_from_scene(stem["center_mm"], stem.get("rotation")),
        ))
    return objects


class SceneApplier(Node):
    def __init__(self):
        super().__init__("calibration_moveit_scene_applier")
        self.cli = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

    def apply(self, objects, timeout_sec):
        """Apply."""
        if not self.cli.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/apply_planning_scene service is not available; start MoveIt move_group first")
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = list(objects)
        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self.cli.call_async(req)
        deadline = self.get_clock().now().nanoseconds / 1e9 + timeout_sec
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.get_clock().now().nanoseconds / 1e9 > deadline:
                raise RuntimeError("timed out applying MoveIt planning scene")
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError("MoveIt rejected planning scene update")


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", help="exact GUI scene JSON; overrides legacy geometry arguments")
    parser.add_argument("--reference-fit-json", default=str(DEFAULT_REFERENCE_FIT))
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--sphere-radius-mm", type=float, default=10.0)
    parser.add_argument("--stem-radius-mm", type=float, default=4.0)
    parser.add_argument("--stem-center-mm", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--stem-diameter-mm", type=float, default=8.0)
    parser.add_argument("--stem-length-mm", type=float, default=1.0)
    parser.add_argument("--table-z-mm", type=float, default=0.0)
    parser.add_argument("--table-thickness-mm", type=float, default=20.0)
    parser.add_argument("--table-size-x-mm", type=float, default=1200.0)
    parser.add_argument("--table-size-y-mm", type=float, default=900.0)
    parser.add_argument("--table-center-x-mm", type=float, default=-400.0)
    parser.add_argument("--table-center-y-mm", type=float, default=120.0)
    parser.add_argument("--magnetic-base-size-mm", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--magnetic-base-center-mm", nargs=3, type=float, default=[-401.9, 126.5, 35.0])
    parser.add_argument("--remove", action="store_true", help="remove calibration objects instead of adding them")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be positive")
    for label, value in (
        ("--sphere-radius-mm", args.sphere_radius_mm),
        ("--stem-radius-mm", args.stem_radius_mm),
        ("--stem-diameter-mm", args.stem_diameter_mm),
        ("--stem-length-mm", args.stem_length_mm),
        ("--table-thickness-mm", args.table_thickness_mm),
        ("--table-size-x-mm", args.table_size_x_mm),
        ("--table-size-y-mm", args.table_size_y_mm),
    ):
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"{label} must be positive")

    try:
        if args.remove:
            objects = [remove_object(object_id, args.frame_id) for object_id in OBJECT_IDS]
        elif args.scene_json:
            objects = build_objects_from_scene(args.scene_json, args.frame_id)
        else:
            objects = build_objects(args)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc

    scene_lock = open(f"/tmp/cr5_sphere_acm_{os.environ.get('ROS_DOMAIN_ID', '0')}.lock", "a")
    try:
        fcntl.flock(scene_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        scene_lock.close()
        raise SystemExit("碰撞预检或采集正在使用当前场景，禁止中途修改球位和障碍物")
    rclpy.init()
    node = SceneApplier()
    try:
        node.apply(objects, args.timeout_sec)
        action = "removed" if args.remove else "applied"
        print(f"{action} MoveIt calibration scene objects: {', '.join(obj.id for obj in objects)}")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        scene_lock.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
