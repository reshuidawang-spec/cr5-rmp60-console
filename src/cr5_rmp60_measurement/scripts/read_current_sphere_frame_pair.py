#!/usr/bin/env python3
"""Read-only comparison of existing ROS feedback with ROS GetPose(0,0)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path


def read_pair(runtime, output):
    sys.path.insert(0, str(Path(runtime).resolve() / "scripts"))
    import rclpy
    from dobot_msgs_v4.srv import GetPose
    from dobot_msgs_v4.msg import ToolVectorActual
    from sensor_msgs.msg import JointState
    node = None
    rclpy.init()
    try:
        node = rclpy.create_node("sphere_read_only_frame_pair")
        state = {}

        def on_pose(message):
            state["pose"] = [float(getattr(message, n)) for n in ("x", "y", "z", "rx", "ry", "rz")]
            state["pose_time"] = time.time()

        def on_joints(message):
            values = dict(zip(message.name, message.position))
            if all(f"joint{i}" in values for i in range(1, 7)):
                state["joints"] = [math.degrees(values[f"joint{i}"]) for i in range(1, 7)]
                state["joint_time"] = time.time()

        node.create_subscription(ToolVectorActual, "/dobot_msgs_v4/msg/ToolVectorActual", on_pose, 10)
        node.create_subscription(JointState, "/joint_states_robot", on_joints, 10)

        def fresh_snapshot(after_wall_time=0):
            deadline = time.monotonic()+5.0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                now = time.time()
                if (state.get("pose_time", 0) > after_wall_time and
                        state.get("joint_time", 0) > after_wall_time and
                        now-state["pose_time"] < 0.3 and now-state["joint_time"] < 0.3):
                    return {"pose": list(state["pose"]), "joints": list(state["joints"]),
                            "pose_wall_time": state["pose_time"], "joint_wall_time": state["joint_time"],
                            "source": "existing ROS ToolVectorActual + joint_states_robot; no new robot TCP client"}
            raise RuntimeError("既有ROS反馈话题未在5秒内提供新鲜位姿和关节状态")
        client = node.create_client(GetPose, "/dobot_bringup_ros2/srv/GetPose")
        if not client.wait_for_service(timeout_sec=8.0):
            raise RuntimeError("GetPose服务在8秒内不可用")
        before = fresh_snapshot()
        request = GetPose.Request()
        request.user = request.tool = 0
        started = time.time()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=8.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("GetPose查询未在8秒内返回")
        response = future.result()
        received = time.time()
        after = fresh_snapshot(after_wall_time=received)
        if response.res != 0:
            raise RuntimeError(f"GetPose错误：{response.res} {response.robot_return}")
        pose = [float(v) for v in response.robot_return.strip().strip("{}").split(",")]
        if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
            raise ValueError("GetPose没有返回6个有限位姿值")
        joint_change = max(abs(a-b) for a, b in zip(before["joints"], after["joints"]))
        feedback_change = math.dist(before["pose"][:3], after["pose"][:3])
        report = {"purpose": "read_only_GetPose00_vs_feedback", "motion_commands_sent": False,
                  "request_started_wall_time": started, "response_received_wall_time": received,
                  "query": "GetPose(user=0,tool=0) via existing ROS driver",
                  "getpose00": pose, "feedback_before": before, "feedback_after": after,
                  "feedback_bracket_joint_change_deg": joint_change,
                  "feedback_bracket_position_change_mm": feedback_change,
                  "stationary_bracket": joint_change < 0.05 and feedback_change < 0.1,
                  "getpose00_minus_feedback_position_mm": [pose[i]-after["pose"][i] for i in range(3)],
                  "getpose00_minus_feedback_euler_deg": [(pose[i]-after["pose"][i]+180)%360-180 for i in range(3,6)],
                  "scope": "One current stationary comparison does not establish the historical User/Tool transform for all orientations."}
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        return report
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(read_pair(args.runtime_root, args.output), ensure_ascii=False, indent=2))
