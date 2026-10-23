#!/usr/bin/env python3
"""Publish one non-executing five-branch path as RViz markers."""

import argparse
import csv
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def point_mm(row, prefix):
    point = Point()
    point.x = float(row[f"{prefix}_x"]) / 1000.0
    point.y = float(row[f"{prefix}_y"]) / 1000.0
    point.z = float(row[f"{prefix}_z"]) / 1000.0
    return point


def marker_base(node, marker_id, marker_type, namespace):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.color.a = 1.0
    return marker


def make_markers(node, row):
    prefixes = ("safe_flange", "transition_flange", "start_flange", "target_flange")
    labels = ("SAFE", "TRANSITION", "START", "TARGET")
    points = [point_mm(row, prefix) for prefix in prefixes]
    array = MarkerArray()
    line = marker_base(node, 1, Marker.LINE_STRIP, "planned_flange_path")
    line.scale.x = 0.004
    line.color.r, line.color.g, line.color.b = 0.9, 0.1, 0.1
    line.points = points
    array.markers.append(line)
    colors = ((0.15, 0.55, 0.95), (0.95, 0.65, 0.1), (0.2, 0.8, 0.3), (0.9, 0.1, 0.1))
    for index, (point, label, color) in enumerate(zip(points, labels, colors), start=10):
        ball = marker_base(node, index, Marker.SPHERE, "planned_flange_points")
        ball.pose.position = point
        ball.scale.x = ball.scale.y = ball.scale.z = 0.018
        ball.color.r, ball.color.g, ball.color.b = color
        array.markers.append(ball)
        text = marker_base(node, index + 20, Marker.TEXT_VIEW_FACING, "planned_flange_labels")
        text.pose.position = Point(x=point.x, y=point.y, z=point.z + 0.025)
        text.scale.z = 0.018
        text.color.r = text.color.g = text.color.b = 0.95
        text.text = label
        array.markers.append(text)
    return array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    with Path(args.plan).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit("--plan must contain exactly one row")
    rclpy.init()
    node = Node("five_branch_plan_marker_publisher")
    qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(MarkerArray, "/rmp60_measurement_markers", qos)
    markers = make_markers(node, rows[0])
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        publisher.publish(markers)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.1)
    node.destroy_node()
    rclpy.shutdown()
    print(f"published RViz path markers for {rows[0].get('plan_id', '')}")


if __name__ == "__main__":
    main()
