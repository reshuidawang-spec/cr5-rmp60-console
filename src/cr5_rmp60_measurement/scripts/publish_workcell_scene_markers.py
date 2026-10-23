#!/usr/bin/env python3
"""Publish the table-relative workcell as persistent RViz visual markers."""

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def quaternion_from_matrix(matrix):
    m = matrix
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        values = ((m[2][1] - m[1][2]) / scale, (m[0][2] - m[2][0]) / scale,
                  (m[1][0] - m[0][1]) / scale, 0.25 * scale)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        values = (0.25 * scale, (m[0][1] + m[1][0]) / scale,
                  (m[0][2] + m[2][0]) / scale, (m[2][1] - m[1][2]) / scale)
    elif m[1][1] > m[2][2]:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        values = ((m[0][1] + m[1][0]) / scale, 0.25 * scale,
                  (m[1][2] + m[2][1]) / scale, (m[0][2] - m[2][0]) / scale)
    else:
        scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        values = ((m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale,
                  0.25 * scale, (m[1][0] - m[0][1]) / scale)
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def base_marker(node, marker_id, marker_type, namespace, frame_id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.color.a = 1.0
    return marker


def set_pose(marker, center_mm, rotation=None):
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = [float(v) / 1000 for v in center_mm]
    if rotation is not None:
        qx, qy, qz, qw = quaternion_from_matrix(rotation)
        marker.pose.orientation.x, marker.pose.orientation.y = qx, qy
        marker.pose.orientation.z, marker.pose.orientation.w = qz, qw


def solid(node, marker_id, marker_type, namespace, frame_id, center, size, rotation, color):
    marker = base_marker(node, marker_id, marker_type, namespace, frame_id)
    set_pose(marker, center, rotation)
    marker.scale.x, marker.scale.y, marker.scale.z = [float(v) / 1000 for v in size]
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    return marker


def transform_point(origin, rotation, local):
    return [origin[i] + sum(rotation[i][j] * local[j] for j in range(3)) for i in range(3)]


def make_markers(node, scene):
    frame = scene.get("frame_id", "base_link")
    markers = MarkerArray()
    markers.markers.append(solid(
        node, 1, Marker.CUBE, "workcell_table", frame, scene["table_center"],
        [*scene["table_size"], scene["table_thickness"]], scene["table_rotation"],
        (0.45, 0.48, 0.52, 0.55),
    ))
    markers.markers.append(solid(
        node, 2, Marker.CUBE, "workcell_magnetic_base", frame, scene["base_center"],
        scene["base_size"], scene["base_rotation"], (0.95, 0.55, 0.08, 0.9),
    ))
    diameter = float(scene["sphere_radius"]) * 2
    markers.markers.append(solid(
        node, 3, Marker.SPHERE, "workcell_standard_sphere", frame, scene["sphere_center"],
        [diameter, diameter, diameter], None, (0.12, 0.45, 0.95, 1.0),
    ))
    stem = scene.get("sphere_stem")
    if stem:
        markers.markers.append(solid(
            node, 4, Marker.CYLINDER, "workcell_sphere_stem", frame, stem["center_mm"],
            [stem["diameter_mm"], stem["diameter_mm"], stem["length_mm"]],
            stem.get("rotation"), (0.72, 0.74, 0.78, 1.0),
        ))
    origin = scene.get("table_top_center", [0, 0, 0])
    rotation = scene["table_rotation"]
    axes = base_marker(node, 5, Marker.LINE_LIST, "workcell_table_axes", frame)
    axes.scale.x = 0.006
    axis_length = 150.0
    for index, color in enumerate(((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.35, 1.0))):
        start = Point(x=origin[0] / 1000, y=origin[1] / 1000, z=origin[2] / 1000)
        local = [0.0, 0.0, 0.0]
        local[index] = axis_length
        end_mm = transform_point(origin, rotation, local)
        end = Point(x=end_mm[0] / 1000, y=end_mm[1] / 1000, z=end_mm[2] / 1000)
        axes.points.extend((start, end))
        for _ in range(2):
            axes.colors.append(type(axes.color)(r=color[0], g=color[1], b=color[2], a=1.0))
    markers.markers.append(axes)
    label = base_marker(node, 6, Marker.TEXT_VIEW_FACING, "workcell_labels", frame)
    label.pose.position.x = scene["sphere_center"][0] / 1000
    label.pose.position.y = scene["sphere_center"][1] / 1000
    label.pose.position.z = (scene["sphere_center"][2] + 35.0) / 1000
    label.scale.z = 0.025
    label.color.r = label.color.g = label.color.b = 0.95
    label.text = "standard sphere"
    markers.markers.append(label)
    return markers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", required=True)
    parser.add_argument("--subscriber-timeout-sec", type=float, default=8.0)
    args = parser.parse_args()
    scene = json.loads(Path(args.scene_json).read_text())
    rclpy.init()
    node = Node("workcell_scene_marker_publisher")
    qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(MarkerArray, "/rmp60_measurement_markers", qos)
    markers = make_markers(node, scene)
    wait_deadline = time.monotonic() + max(0.0, args.subscriber_timeout_sec)
    while publisher.get_subscription_count() == 0 and time.monotonic() < wait_deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    publish_deadline = time.monotonic() + 1.5
    while time.monotonic() < publish_deadline:
        publisher.publish(markers)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.1)
    print(f"published {len(markers.markers)} workcell markers; subscribers={publisher.get_subscription_count()}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
