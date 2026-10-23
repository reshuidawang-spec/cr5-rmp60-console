#!/usr/bin/env python3
"""
MoveIt 预飞检查器 —— 五支球杆标定计划的运动学可行性验证。

本脚本是 auto_calibrate_five_branch_sphere.py 生成的标定计划的 "预飞门"（preflight gate），
在真实执行之前，利用 MoveIt 的完整机器人运动学模型对每一个规划的探测位姿进行可到达性
和碰撞检测验证，确保计划可以被安全执行。

检查流水线（pipeline）包含三个层次：
  1. IK 求解（Inverse Kinematics）
     对 safe / transition / start / target 四个关键法兰位姿分别调用 /compute_ik 服务，
     判断机器人能否以无碰撞的方式到达该位姿。每个后续位姿以之前成功求解的关节状态作为种子
     （seed），以保证运动连续性。
  2. 笛卡尔路径规划（Cartesian Path）
     对每一段运动（safe→transition、transition→start、start→target）调用
     /compute_cartesian_path 服务，检查该段是否可以在指定精度下完整执行。
  3. 关节连续性检查
     比较相邻位姿的关节角度差，确保每一步的关节运动不超过 --max-ik-joint-step-deg 阈值，
     避免规划中出现过大的关节跳跃。

检查结果写入输出 CSV，包含以下关键字段：
  - moveit_status: "OK" 表示所有检查通过，"FAIL" 表示至少一项检查未通过
  - moveit_*_ik_code: 各阶段的 IK 返回码（MoveItErrorCodes）
  - moveit_*_fraction: 各段的笛卡尔路径完成比例（1.0 表示完全可达）
  - moveit_joint_continuity_ok: 关节连续性检查是否通过
  - moveit_*_joint_delta_deg: 各段的关节角度差（度）
"""
import argparse
import csv
csv.field_size_limit(16 * 1024 * 1024)
import math
import time
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState

from cross_probe_model import euler_to_matrix
from ros_wait_utils import wait_for_future


POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")


def parse_float(row, field):
    """
    安全地将 CSV 行中的字符串字段解析为浮点数。

    如果字段缺失、为空或值为非有限数（NaN/inf），则抛出 ValueError，
    避免静默传播无效数值。

    参数:
        row: CSV 行字典（由 csv.DictReader 产生）
        field: 字段名
    返回:
        浮点数值
    抛出:
        ValueError: 字段缺失、为空或数值非有限
    """
    value = row.get(field, "")
    if value in (None, ""):
        raise ValueError(f"missing {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def pose_from_row(row, prefix):
    """
    从 CSV 行字典中提取一个 6 关节法兰位姿列表（x, y, z, rx, ry, rz）。

    查找格式为 "{prefix}_{name}" 的字段（name in POSE_NAMES），
    逐个解析为浮点数。注意：位置单位为毫米（mm），进入 ROS 后在
    pose_stamped_from_flange_pose 中转换为米（m）。

    参数:
        row: CSV 行字典
        prefix: 字段前缀，如 "start_flange"、"target_flange"
    返回:
        [x, y, z, rx, ry, rz] 六个浮点数
    抛出:
        ValueError: 任意字段缺失或数值无效
    """
    return [parse_float(row, f"{prefix}_{name}") for name in POSE_NAMES]


def quat_from_matrix(m):
    """
    从 3x3 旋转矩阵计算单位四元数。

    采用稳定算法：根据矩阵迹（trace）和最大对角元素选择不同分支，
    避免接近 180 度旋转时的奇异性。返回字典包含 w, x, y, z 四分量。

    参数:
        m: 3x3 旋转矩阵（列表的列表）
    返回:
        {"w": float, "x": float, "y": float, "z": float} 单位四元数
    """
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return {
            "w": 0.25 * s,
            "x": (m[2][1] - m[1][2]) / s,
            "y": (m[0][2] - m[2][0]) / s,
            "z": (m[1][0] - m[0][1]) / s,
        }
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return {
            "w": (m[2][1] - m[1][2]) / s,
            "x": 0.25 * s,
            "y": (m[0][1] + m[1][0]) / s,
            "z": (m[0][2] + m[2][0]) / s,
        }
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return {
            "w": (m[0][2] - m[2][0]) / s,
            "x": (m[0][1] + m[1][0]) / s,
            "y": 0.25 * s,
            "z": (m[1][2] + m[2][1]) / s,
        }
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return {
        "w": (m[1][0] - m[0][1]) / s,
        "x": (m[0][2] + m[2][0]) / s,
        "y": (m[1][2] + m[2][1]) / s,
        "z": 0.25 * s,
    }


def pose_stamped_from_flange_pose(pose, frame_id, euler_sequence):
    """
    将 6 分量法兰位姿转换为 ROS geometry_msgs/PoseStamped 消息。

    主要转换步骤：
      1. 位置从毫米（mm）转换为米（m）
      2. 欧拉角（rx, ry, rz）通过 euler_to_matrix 转为旋转矩阵
      3. 旋转矩阵再转为单位四元数
      4. 四元数归一化后填充到 PoseStamped 消息

    参数:
        pose: 6 元素列表 [x_mm, y_mm, z_mm, rx, ry, rz]
        frame_id: 消息的参考坐标系（如 "base_link"）
        euler_sequence: 欧拉角旋转顺序（如 "xyz"）
    返回:
        geometry_msgs.msg.PoseStamped
    """
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = pose[0] * 0.001
    msg.pose.position.y = pose[1] * 0.001
    msg.pose.position.z = pose[2] * 0.001
    quat = quat_from_matrix(euler_to_matrix(euler_sequence, pose[3:6]))
    norm = math.sqrt(sum(value * value for value in quat.values()))
    msg.pose.orientation.x = quat["x"] / norm
    msg.pose.orientation.y = quat["y"] / norm
    msg.pose.orientation.z = quat["z"] / norm
    msg.pose.orientation.w = quat["w"] / norm
    return msg


class PlanMoveItChecker(Node):
    """
    ROS2 Node 封装，用于标定计划的 MoveIt 可执行性检查。

    主要职责：
      - 订阅机器人关节状态话题，维护最新的关节状态信息
      - 调用 /compute_ik 服务对每个目标位姿进行逆运动学求解
      - 调用 /compute_cartesian_path 服务对每一段运动进行笛卡尔路径规划
      - 检查相邻位姿之间的关节连续性（角度变化不超限）
    """

    def __init__(self, joint_state_topic, seed_joints_deg=None):
        super().__init__("five_branch_plan_moveit_checker")
        self.joint_state_topic = joint_state_topic
        self.uses_explicit_seed = seed_joints_deg is not None
        self.joint_state = None
        self.joint_state_wall_time = None
        if seed_joints_deg is None:
            self.create_subscription(JointState, joint_state_topic, self._joint_state_cb, 10)
        else:
            self.joint_state = JointState()
            self.joint_state.name = [f"joint{index}" for index in range(1, 7)]
            self.joint_state.position = [math.radians(value) for value in seed_joints_deg]
            self.joint_state_wall_time = time.monotonic()
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        self.cartesian_cli = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.display_publisher = self.create_publisher(DisplayTrajectory, "/display_planned_path", 10)

    def publish_display_trajectory(self, start_state, trajectories):
        """Animate already-computed Cartesian segments in RViz; never execute them."""
        trajectories = [trajectory for trajectory in trajectories
                        if trajectory is not None and trajectory.joint_trajectory.points]
        if not trajectories:
            return
        message = DisplayTrajectory()
        message.model_id = "cr5_robot"
        message.trajectory_start = start_state
        message.trajectory = trajectories
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            self.display_publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.05)

    def _joint_state_cb(self, msg):
        """
        关节状态话题（/joint_states_robot）的回调函数。

        缓存最新的 JointState 消息和接收时间，供 IK 求解时作为种子状态使用。
        如果消息超时超过 1 秒，wait_ready 会报错以避免使用过时数据。

        参数:
            msg: sensor_msgs.msg.JointState 消息，包含关节名称和位置
        """
        self.joint_state = msg
        self.joint_state_wall_time = time.monotonic()

    def wait_ready(self, timeout_sec):
        """
        等待必要服务可用、关节状态首次到达且数据完整新鲜。

        检查内容：
          1. /compute_ik 和 /compute_cartesian_path 服务是否可用
          2. joint_state_topic 是否在超时时间内收到首个消息
          3. 关节状态是否新鲜（距离上次接收不超过 1 秒）
          4. 关节名称列表是否包含全部 6 个标准关节 joint1 ~ joint6

        参数:
            timeout_sec: 最长等待秒数
        抛出:
            RuntimeError: 任一条件不满足时抛出异常，附带原因说明
        """
        if not self.ik_cli.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/compute_ik service is not available")
        if not self.cartesian_cli.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/compute_cartesian_path service is not available")
        deadline = self.get_clock().now().nanoseconds / 1e9 + timeout_sec
        while rclpy.ok() and self.joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.get_clock().now().nanoseconds / 1e9 > deadline:
                raise RuntimeError(f"{self.joint_state_topic} is not available")
        if (
            not self.uses_explicit_seed
            and (self.joint_state_wall_time is None or time.monotonic() - self.joint_state_wall_time > 1.0)
        ):
            raise RuntimeError("robot joint state is stale")
        required = {f"joint{index}" for index in range(1, 7)}
        missing = sorted(required - set(self.joint_state.name))
        if missing:
            raise RuntimeError("robot joint state is missing: " + ", ".join(missing))

    def solve_ik(self, pose, seed_state, args, avoid_collisions=True):
        """
        调用 MoveIt 的 /compute_ik 服务对单个法兰位姿进行逆运动学求解。

        参数:
            pose: 6 分量法兰位姿 [x_mm, y_mm, z_mm, rx, ry, rz]
            seed_state: 种子关节状态（RobotState 消息），用于 IK 初始猜测
            args: 命令行参数对象，包含 group、ik_link、frame_id、euler_sequence 等配置
            avoid_collisions: 是否启用碰撞避免（默认 True）
        返回:
            (成功标志, 错误码, RobotState 解答)
            成功标志为 True 当且仅当 error_code.val == MoveItErrorCodes.SUCCESS
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name = args.group
        req.ik_request.ik_link_name = args.ik_link
        req.ik_request.avoid_collisions = bool(avoid_collisions)
        req.ik_request.robot_state = seed_state
        req.ik_request.pose_stamped = pose_stamped_from_flange_pose(pose, args.frame_id, args.euler_sequence)
        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.timeout = Duration(sec=int(args.ik_timeout), nanosec=int((args.ik_timeout % 1.0) * 1e9))
        future = self.ik_cli.call_async(req)
        result = wait_for_future(self, future, args.ik_timeout + 2.0, "/compute_ik")
        return result.error_code.val == MoveItErrorCodes.SUCCESS, result.error_code.val, result.solution

    def compute_path(self, start_state, waypoints, args, avoid_collisions=True):
        """
        调用 MoveIt 的 /compute_cartesian_path 服务进行笛卡尔路径规划。

        检查机器人能否以笛卡尔直线运动方式从当前状态依次经过所有路点。
        结果中的 fraction 字段表示路径完成比例（0~1），只有达到
        --min-fraction 阈值且 error_code 为 SUCCESS 才算通过。

        参数:
            start_state: 起始关节状态（RobotState 消息）
            waypoints: 法兰位姿列表，每个元素为 6 分量 [x, y, z, rx, ry, rz]
            args: 命令行参数对象，包含 max_step_m、jump_threshold 等规划参数
            avoid_collisions: 是否启用碰撞避免（默认 True）
        返回:
            (成功标志, 错误码, 完成比例, RobotState 最终状态)
        """
        req = GetCartesianPath.Request()
        req.header.frame_id = args.frame_id
        req.header.stamp = self.get_clock().now().to_msg()
        req.start_state = start_state
        req.group_name = args.group
        req.link_name = args.ik_link
        req.max_step = args.max_step_m
        req.jump_threshold = args.jump_threshold
        req.avoid_collisions = bool(avoid_collisions)
        req.waypoints = [
            pose_stamped_from_flange_pose(pose, args.frame_id, args.euler_sequence).pose
            for pose in waypoints
        ]
        future = self.cartesian_cli.call_async(req)
        result = wait_for_future(self, future, max(5.0, args.ik_timeout + 2.0), "/compute_cartesian_path")
        ok = result.error_code.val == MoveItErrorCodes.SUCCESS and result.fraction >= args.min_fraction
        path_joint_step = trajectory_max_joint_step_deg(start_state, result.solution)
        return ok, result.error_code.val, result.fraction, result.solution, path_joint_step


def make_seed_state(joint_state):
    """
    用当前关节状态构建 RobotState 消息作为 IK 求解的种子（seed）。

    参数:
        joint_state: sensor_msgs.msg.JointState 消息
    返回:
        moveit_msgs.msg.RobotState
    """
    state = RobotState()
    state.joint_state = joint_state
    return state


def max_joint_delta_deg(start_state, end_state):
    """
    计算两个 RobotState 之间关节角度的最大变化量（度）。

    用于检查运动连续性：如果某个关节在相邻位姿之间的变化超过
    --max-ik-joint-step-deg 阈值，说明规划中存在不连续的跳跃，
    可能导致实际执行时的奇异或干涉。

    角度差计算时通过模 2*pi 处理换向，确保结果在 (-pi, pi] 范围内。

    参数:
        start_state: 起始 RobotState
        end_state: 终止 RobotState
    返回:
        最大关节角度差的绝对值（度）。如果缺少标准关节名称则返回无穷大。
    """
    start = dict(zip(start_state.joint_state.name, start_state.joint_state.position))
    end = dict(zip(end_state.joint_state.name, end_state.joint_state.position))
    names = [f"joint{index}" for index in range(1, 7)]
    if any(name not in start or name not in end for name in names):
        return math.inf
    deltas = []
    for name in names:
        delta = (float(end[name]) - float(start[name]) + math.pi) % (2.0 * math.pi) - math.pi
        deltas.append(abs(math.degrees(delta)))
    return max(deltas)


def trajectory_max_joint_step_deg(start_state, trajectory):
    """Return the largest adjacent joint step in a Cartesian path result."""
    joint_trajectory = trajectory.joint_trajectory
    names = list(joint_trajectory.joint_names)
    required = [f"joint{index}" for index in range(1, 7)]
    if any(name not in names for name in required):
        return math.inf
    previous = dict(zip(start_state.joint_state.name, start_state.joint_state.position))
    if any(name not in previous for name in required):
        return math.inf
    maximum = 0.0
    for point in joint_trajectory.points:
        current = dict(zip(names, point.positions))
        if any(name not in current for name in required):
            return math.inf
        for name in required:
            delta = (float(current[name]) - float(previous[name]) + math.pi) % (2.0 * math.pi) - math.pi
            maximum = max(maximum, abs(math.degrees(delta)))
        previous = current
    return maximum if joint_trajectory.points else math.inf


def write_csv(path, rows):
    """
    将检查结果行列表写入 CSV 文件。

    自动创建父目录，汇总所有行的键作为 CSV 列头。
    如果 path 为空或 rows 为空则不执行写入。

    参数:
        path: 输出 CSV 文件路径（字符串或 Path 对象）。如果为 None 或空串则跳过。
        rows: 字典列表，每个字典对应一行 CSV 输出
    """
    if not path or not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """
    解析命令行参数。

    关键参数说明：
      --plan: 来自 auto_calibrate_five_branch_sphere.py --plan-output 的 CSV 计划文件
      --check-probe-with-moveit-collision: 对 probe 段也启用碰撞检测。
        注意：如果标定球被加载为碰撞物体，此检查通常会失败，
        因为红宝石球（ruby ball）预期与球面接触。
      --max-ik-joint-step-deg: 关节连续性阈值（度），相邻位姿之间任一关节
        角度变化超过此值则检查不通过。
      --min-fraction: 笛卡尔路径最小完成比例（默认 1.0，即必须 100% 完成）。

    返回:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="CSV from auto_calibrate_five_branch_sphere.py --plan-output")
    parser.add_argument("--checked-output")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--group", default="cr5_group")
    parser.add_argument("--ik-link", default="Link6", help="use Link6 for Dobot flange-pose plans")
    parser.add_argument("--joint-state-topic", default="/joint_states_robot")
    parser.add_argument(
        "--seed-joints-deg", nargs=6, type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="explicit current robot joint angles in degrees; bypasses joint-state topic waiting",
    )
    parser.add_argument("--euler-sequence", default="xyz")
    parser.add_argument("--ik-timeout", type=float, default=0.2)
    parser.add_argument("--max-step-m", type=float, default=0.001)
    parser.add_argument("--jump-threshold", type=float, default=0.0)
    parser.add_argument("--min-fraction", type=float, default=1.0)
    parser.add_argument("--service-timeout-sec", type=float, default=5.0)
    parser.add_argument("--max-ik-joint-step-deg", type=float, default=15.0)
    parser.add_argument(
        "--joint-continuity-source",
        choices=("endpoint-ik", "cartesian-trajectory"),
        default="endpoint-ik",
        help=(
            "gate continuity using independently solved endpoint IK values or the actual "
            "joint samples returned by each Cartesian path"
        ),
    )
    parser.add_argument(
        "--trust-safe-pose",
        action="store_true",
        help=(
            "treat the plan's safe pose as an already validated staging pose. "
            "The current robot state -> safe pose joint delta is still reported, "
            "but it is not included in the per-row joint-continuity gate."
        ),
    )
    parser.add_argument(
        "--safe-transition-mode",
        choices=("cartesian", "joint"),
        default="cartesian",
        help=(
            "how to preflight the safe->transition segment. Use joint when the "
            "executor will move this staging segment with MovJ instead of a "
            "Cartesian line; transition->start and start->target are still "
            "checked as Cartesian paths."
        ),
    )
    parser.add_argument(
        "--check-probe-with-moveit-collision",
        action="store_true",
        help=(
            "also require the short probe segment to be MoveIt collision-free. "
            "This usually fails if the calibration sphere is loaded as a collision object, "
            "because the active ruby ball is expected to contact it."
        ),
    )
    parser.add_argument(
        "--publish-display-trajectory", action="store_true",
        help="publish successful Cartesian segments to RViz /display_planned_path; never execute",
    )
    parser.add_argument("--sphere-contact-safety", action="store_true",
                        help="require full-tool sweeps and ruby-only sphere contact policy")
    parser.add_argument("--motion-policy", help="joint execution and measured uncertainty policy JSON")
    return parser.parse_args()


def main():
    """
    主入口：读取标定计划 CSV，逐行进行 MoveIt 预飞检查。

    对计划 CSV 中的每一行执行以下检查步骤：
      1. 加载该行对应的 safe、transition、start、target 法兰位姿
      2. 依次对 safe（如有）→ transition → start → target 进行 IK 求解
      3. 对 safe→transition（如有）、transition→start、start→target
         三段分别进行笛卡尔路径规划
      4. 检查相邻位姿之间的关节连续性
      5. 综合以上全部检查结果，写入 moveit_status 字段

    所有检查完成后，输出带 moveit_* 字段的 CSV，并根据失败数量决定
    退出码（如有失败则返回非零退出码）。

    注意：如果 plan 行的 preflight_mode 为 "transition-only"，则只检查
    transition 一个位姿，跳过 start/target 及其路径规划。
    """
    args = parse_args()
    if (
        args.ik_timeout <= 0
        or args.max_step_m <= 0
        or args.service_timeout_sec <= 0
        or args.max_ik_joint_step_deg <= 0
    ):
        raise SystemExit("timeouts and --max-step-m must be positive")
    if not 0.0 < args.min_fraction <= 1.0:
        raise SystemExit("--min-fraction must be in (0, 1]")

    try:
        with Path(args.plan).open(newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise SystemExit(str(exc)) from exc

    rclpy.init()
    node = PlanMoveItChecker(args.joint_state_topic, args.seed_joints_deg)
    checked_rows = []
    failures = 0
    strict = None
    try:
        node.wait_ready(args.service_timeout_sec)
        from sphere_contact_safety import requires_sphere_safety
        if args.sphere_contact_safety or any(requires_sphere_safety(row) for row in rows):
            from moveit_sphere_contact_safety import SphereContactPreflight
            strict = SphereContactPreflight(node, args)
        sequential_seed = make_seed_state(node.joint_state)
        for row in rows:
            checked = dict(row)
            plan_id = row.get("plan_id", "")
            try:
                if strict is not None:
                    proof, sequential_seed = strict.check_row(row, sequential_seed)
                    checked.update(proof)
                    checked_rows.append(checked)
                    print(f"{plan_id}: OK (full sphere contact safety)")
                    continue
                transition_only = row.get("preflight_mode", "").strip() == "transition-only"
                transition = pose_from_row(row, "transition_flange")
                start = None if transition_only else pose_from_row(row, "start_flange")
                target = None if transition_only else pose_from_row(row, "target_flange")
                safe = None
                if all(row.get(f"safe_flange_{name}") not in (None, "") for name in POSE_NAMES):
                    safe = pose_from_row(row, "safe_flange")
                seed = make_seed_state(node.joint_state)
                safe_ok = True
                safe_code = ""
                safe_state = seed
                if safe is not None:
                    safe_ok, safe_code, safe_state = node.solve_ik(safe, seed, args, True)
                transition_ok, transition_code, transition_state = node.solve_ik(
                    transition,
                    safe_state if safe_ok else seed,
                    args,
                    True,
                )
                start_ok = transition_only
                start_code = ""
                start_state = transition_state
                target_ok = transition_only
                target_code = ""
                if not transition_only:
                    start_ok, start_code, start_state = node.solve_ik(
                        start,
                        transition_state if transition_ok else seed,
                        args,
                        True,
                    )
                    target_ok, target_code, target_state = node.solve_ik(
                        target,
                        start_state if start_ok else seed,
                        args,
                        args.check_probe_with_moveit_collision,
                    )
                else:
                    target_state = start_state
                safe_joint_delta = 0.0 if safe is None else max_joint_delta_deg(seed, safe_state)
                transition_joint_delta = max_joint_delta_deg(
                    safe_state if safe is not None and safe_ok else seed,
                    transition_state,
                )
                start_joint_delta = 0.0 if transition_only else max_joint_delta_deg(transition_state, start_state)
                target_joint_delta = 0.0 if transition_only else max_joint_delta_deg(start_state, target_state)
                gated_safe_joint_delta = 0.0 if args.trust_safe_pose else safe_joint_delta
                endpoint_ik_continuity_ok = max(
                    gated_safe_joint_delta,
                    transition_joint_delta,
                    start_joint_delta,
                    target_joint_delta,
                ) <= args.max_ik_joint_step_deg
                approach_ok = transition_only
                probe_ok = transition_only
                safe_transition_ok = safe is None
                safe_transition_fraction = "" if safe is None else "0.000000"
                approach_fraction = 0.0
                probe_fraction = 0.0
                safe_transition_code = "" if safe is None else ""
                approach_code = ""
                probe_code = ""
                safe_transition_path_joint_step = 0.0
                approach_path_joint_step = 0.0
                probe_path_joint_step = 0.0
                safe_transition_trajectory = None
                approach_trajectory = None
                probe_trajectory = None
                if safe is not None and safe_ok:
                    if args.safe_transition_mode == "joint":
                        safe_transition_ok = transition_ok
                        safe_transition_code = "JOINT"
                        safe_transition_fraction = "1.000000"
                    else:
                        safe_transition_ok, safe_transition_code, safe_transition_fraction_raw, safe_transition_trajectory, safe_transition_path_joint_step = node.compute_path(
                            safe_state,
                            [transition],
                            args,
                            True,
                        )
                        safe_transition_fraction = f"{safe_transition_fraction_raw:.6f}"
                if transition_ok and not transition_only:
                    approach_ok, approach_code, approach_fraction, approach_trajectory, approach_path_joint_step = node.compute_path(
                        transition_state,
                        [start],
                        args,
                        True,
                    )
                if start_ok and not transition_only:
                    probe_ok, probe_code, probe_fraction, probe_trajectory, probe_path_joint_step = node.compute_path(
                        start_state,
                        [target],
                        args,
                        args.check_probe_with_moveit_collision,
                    )
                if args.joint_continuity_source == "cartesian-trajectory":
                    continuity_values = [gated_safe_joint_delta]
                    if safe is not None:
                        continuity_values.append(
                            transition_joint_delta
                            if args.safe_transition_mode == "joint"
                            else safe_transition_path_joint_step
                        )
                    if not transition_only:
                        continuity_values.extend((approach_path_joint_step, probe_path_joint_step))
                    joint_continuity_ok = max(continuity_values) <= args.max_ik_joint_step_deg
                else:
                    joint_continuity_ok = endpoint_ik_continuity_ok
                ok = (
                    safe_ok
                    and transition_ok
                    and start_ok
                    and target_ok
                    and safe_transition_ok
                    and approach_ok
                    and probe_ok
                    and joint_continuity_ok
                )
                if ok and args.publish_display_trajectory:
                    node.publish_display_trajectory(
                        safe_state if safe is not None else seed,
                        [safe_transition_trajectory, approach_trajectory, probe_trajectory],
                    )
                checked.update(
                    {
                        "moveit_status": "OK" if ok else "FAIL",
                        "moveit_safe_ik_code": str(safe_code),
                        "moveit_transition_ik_code": str(transition_code),
                        "moveit_start_ik_code": str(start_code),
                        "moveit_target_ik_code": str(target_code),
                        "moveit_safe_transition_path_code": str(safe_transition_code),
                        "moveit_safe_transition_fraction": str(safe_transition_fraction),
                        "moveit_approach_path_code": str(approach_code),
                        "moveit_approach_fraction": f"{approach_fraction:.6f}",
                        "moveit_probe_path_code": str(probe_code),
                        "moveit_probe_fraction": f"{probe_fraction:.6f}",
                        "moveit_probe_collision_checked": str(int(args.check_probe_with_moveit_collision)),
                        "moveit_trust_safe_pose": str(int(args.trust_safe_pose)),
                        "moveit_safe_transition_mode": args.safe_transition_mode,
                        "moveit_joint_continuity_source": args.joint_continuity_source,
                        "moveit_joint_continuity_ok": str(int(joint_continuity_ok)),
                        "moveit_endpoint_ik_continuity_ok": str(int(endpoint_ik_continuity_ok)),
                        "moveit_safe_joint_delta_deg": f"{safe_joint_delta:.6f}",
                        "moveit_transition_joint_delta_deg": f"{transition_joint_delta:.6f}",
                        "moveit_start_joint_delta_deg": f"{start_joint_delta:.6f}",
                        "moveit_target_joint_delta_deg": f"{target_joint_delta:.6f}",
                        "moveit_safe_transition_trajectory_joint_step_deg": f"{safe_transition_path_joint_step:.6f}",
                        "moveit_approach_trajectory_joint_step_deg": f"{approach_path_joint_step:.6f}",
                        "moveit_probe_trajectory_joint_step_deg": f"{probe_path_joint_step:.6f}",
                    }
                )
                if not ok:
                    failures += 1
                print(f"{plan_id}: {checked['moveit_status']}")
            except (ValueError, RuntimeError, KeyError, TimeoutError) as exc:
                failures += 1
                checked.update({"moveit_status": "FAIL", "moveit_error": str(exc)})
                print(f"{plan_id}: FAIL ({exc})")
            checked_rows.append(checked)
    finally:
        if strict is not None:
            strict.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    write_csv(args.checked_output, checked_rows)
    if args.checked_output:
        print(f"saved checked plan: {args.checked_output}")
    if failures:
        raise SystemExit(f"MoveIt preflight failed for {failures}/{len(rows)} plan rows")
    print(f"MoveIt preflight OK for {len(rows)} plan rows")


if __name__ == "__main__":
    main()
