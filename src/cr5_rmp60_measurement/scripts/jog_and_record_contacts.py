#!/usr/bin/env python3
"""
交互式笛卡尔 JOG 测头接触记录工具
====================================

本模块提供交互式命令行界面，用于手动/半自动方式控制机器人进行笛卡尔
增量运动（jog），并在运动过程中实时监控 RMP60 测头的 DI1 触发信号，
记录触发时刻的位姿到 CSV 文件。

核心功能：
  1. 交互式终端面板（替代示教器）：通过键盘命令控制机器人沿
     X/Y/Z/Rx/Ry/Rz 轴做增量运动。
  2. DI1 接触检测：每次增量移动期间同时监控 FeedInfo 中的 DI1 状态，
     一旦检测到测头触发（DI1 上升沿），立即停止机器人并记录触发位姿。
  3. 自动反向回退：触发后可沿运动反方向自动回退指定距离。
  4. 丰富的元数据记录：每次接触记录包含工件 ID、测球 ID、分支标签、
     操作员备注等，便于后续校准分析。

与 probe_touch.py 的关系：
  - 继承 ProbeTouch 基类的所有服务客户端和 FeedInfo 订阅
  - 基于 ProbeTouch 的异步调用原语 (call_async_checked) 构建移动循环
  - 使用 ProbeTouch 的 stop_fast_then_confirm 安全停止策略

适用场景：
  - 手动标定测头各分支（branch）的接触点
  - 对标准球进行多点接触采样
  - 现场调试和故障排查

坐标系约定：
  - 所有位姿均为机器人基坐标系下的法兰末端位姿 [x, y, z, rx, ry, rz]
  - z 轴向上为正
  - 角度单位：度（deg），长度单位：毫米（mm）
"""
import argparse
import csv
import math
import shlex
import time
from pathlib import Path

import rclpy
from dobot_msgs_v4.srv import MovL

from probe_touch import PROBE_SPIN_TIMEOUT_SEC, ProbeTouch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data/jog_probe_contacts.csv"
POST_REACH_TRIGGER_GRACE_SEC = 0.25

AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "rx": 3,
    "ry": 4,
    "rz": 5,
}


def angle_delta_deg(actual, expected):
    """
    计算两个角度值之间的带符号最小差值（度），考虑 ±180° 环绕。

    例如：actual=350, expected=10 时，差值为 -20° 而非 +340°。
    这对于姿态到达判断中的角度误差计算至关重要。

    Args:
        actual (float): 实际角度值（度）。
        expected (float): 期望角度值（度）。

    Returns:
        float: 带符号的最小角度差值，范围 (-180, 180]。
    """
    return (float(actual) - float(expected) + 180.0) % 360.0 - 180.0


def format_pose(pose):
    """
    将位姿列表格式化为可读字符串，保留 4 位小数。

    例如：[123.4567, 45.6789, 300.1234, 0.0000, 0.0000, 90.0000]

    Args:
        pose (list[float]): 6 元素位姿 [x, y, z, rx, ry, rz]。

    Returns:
        str: 格式化后的位姿字符串。
    """
    return "[" + ", ".join(f"{float(value):.4f}" for value in pose) + "]"


def pose_fields(prefix, pose):
    """
    将位姿列表转换为 CSV 字段字典，键名为 "{prefix}_{轴名}"。

    例如 prefix="start", pose=[1,2,3,4,5,6] 时返回：
      {"start_x": "1.0000", "start_y": "2.0000", ..., "start_rz": "6.0000"}

    Args:
        prefix (str): 字段前缀，如 "start", "target", "trigger", "stop"。
        pose (list[float] | None): 6 元素位姿，或 None（此时所有字段值为空字符串）。

    Returns:
        dict[str, str]: CSV 字段字典。
    """
    names = ("x", "y", "z", "rx", "ry", "rz")
    return {
        f"{prefix}_{name}": "" if pose is None else f"{float(pose[index]):.4f}"
        for index, name in enumerate(names)
    }


def vector_fields(prefix, values, count, digits=6):
    """
    将向量值列表转换为 CSV 字段字典，键名为 "{prefix}_{序号}"。

    用于记录关节角等数值数组。支持 values 为 None 或长度不足的情况，
    此时对应字段填为空字符串。

    Args:
        prefix (str): 字段前缀，如 "trigger_joint"。
        values (list[float] | None): 向量值列表。
        count (int): 期望的字段数量。
        digits (int): 小数位数。

    Returns:
        dict[str, str]: CSV 字段字典。
    """
    return {
        f"{prefix}_{index + 1}": "" if values is None or index >= len(values) else f"{float(values[index]):.{digits}f}"
        for index in range(count)
    }


def write_row(path, row):
    """
    将一行数据追加写入 CSV 文件。

    如果文件不存在，自动创建目录和文件，并写入表头。
    每次写入都是追加模式（"a"），不会覆盖已有数据。

    Args:
        path (str | Path): CSV 输出路径。
        row (dict): 包含所有字段的字典，键的顺序由 dict 保持。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def fresh_feed_di1(node, max_age_sec=0.2):
    """
    读取 FeedInfo 缓存中的 DI1 状态，并确保缓存足够新鲜。

    与 ProbeTouch.read_di1() 不同，本函数只依赖 FeedInfo 缓存，
    不调用 DI 服务。这在高频循环（如 wait_jog）中避免了重复的
    服务调用开销。

    判断标准：
      - last_feed_time 不为 None
      - 缓存年龄（当前时间 - last_feed_time）<= max_age_sec

    Args:
        node (ProbeTouch): ProbeTouch 节点实例。
        max_age_sec (float): 最大可接受缓存年龄（秒）。

    Returns:
        bool: True 表示 DI1 为高电平（测头触发）。

    Raises:
        RuntimeError: FeedInfo 不存在或缓存已过期。
    """
    if node.last_feed_time is None:
        raise RuntimeError("FeedInfo is not available; cannot read DI1")
    age = time.monotonic() - node.last_feed_time
    if age > max_age_sec:
        raise RuntimeError(f"FeedInfo is stale ({age:.3f}s); cannot read DI1")
    return bool(node.di1)


def current_pose(node, max_age_sec=0.2):
    """
    获取当前位姿，优先使用 FeedInfo 缓存，必要时回退到 GetPose 服务。

    这是一种"轻量优先"策略：
      - FeedInfo 缓存的读取是本地操作，无服务调用开销。
      - 仅在缓存过期或不可用时才调用 GetPose 服务。
      - 适用于频繁获取位姿的场景（如交互式 jog）。

    Args:
        node (ProbeTouch): ProbeTouch 节点实例。
        max_age_sec (float): FeedInfo 缓存的最大可接受年龄。

    Returns:
        list[float]: 当前法兰位姿 [x, y, z, rx, ry, rz]。
    """
    if node.last_feed_time is not None and time.monotonic() - node.last_feed_time <= max_age_sec:
        snapshot = node.feed_snapshot()
        pose = snapshot.get("pose")
        if pose is not None:
            return list(pose)
    return node.get_pose()


def make_contact_row(args, state, command, start_pose, target_pose, trigger_snapshot, stop_pose):
    """
    生成一条完整的测头接触记录行（字典），用于写入 CSV。

    CSV 包含以下主要字段：
      - timestamp: 记录生成时间（Unix 时间戳，秒.毫秒）
      - sample_index: 当次会话中的样本序号（从 0 递增）
      - session_id / workpiece_id / artifact_id / artifact_type: 工件和会话元信息
      - physical_ball_id / branch / operator_note: 操作员标注（可从交互式命令设置）
      - command: 触发接触的 JOG 命令（如 "z-"）
      - linear_step_mm / angular_step_deg / speed / auto_retract_mm: 当前运动参数
      - start_flange_* (x/y/z/rx/ry/rz): 运动起始位姿
      - target_flange_*: 运动目标位姿（可能未到达）
      - trigger_flange_*: DI1 触发瞬间的 FeedInfo 位姿
      - stop_flange_*: 停止后的 GetPose 位姿
      - trigger_feed_sequence: 触发时的 FeedInfo 序列号
      - trigger_digital_input_bits: 触发时的完整数字输入位图
      - trigger_di1: DI1 的布尔值
      - trigger_joint_1..6: 触发时的关节角

    注意：当触发位姿不可用时（snapshot 中 pose 为 None），
    trigger_flange_* 字段留空，由后续分析代码决定如何处理。

    Args:
        args: 命令行参数对象。
        state (dict): 当前会话状态。
        command (str): 触发的 JOG 命令字符串。
        start_pose (list[float]): 运动起始位姿。
        target_pose (list[float]): 运动目标位姿。
        trigger_snapshot (dict): DI1 触发时的 FeedInfo 快照。
        stop_pose (list[float]): 停止后的位姿。

    Returns:
        dict: 完整的 CSV 行字典。
    """
    trigger_pose = trigger_snapshot.get("pose")
    row = {
        "timestamp": f"{time.time():.3f}",
        "sample_index": str(state["sample_index"]),
        "session_id": args.session_id,
        "workpiece_id": args.workpiece_id,
        "artifact_id": args.artifact_id,
        "artifact_type": args.artifact_type,
        "physical_ball_id": state["physical_ball_id"],
        "branch": state["branch"],
        "operator_note": state["operator_note"],
        "command": command,
        "linear_step_mm": f"{state['linear_step_mm']:.4f}",
        "angular_step_deg": f"{state['angular_step_deg']:.4f}",
        "speed": str(state["speed"]),
        "auto_retract_mm": f"{state['auto_retract_mm']:.4f}",
        "trigger_feed_sequence": str(trigger_snapshot.get("sequence", "")),
        "trigger_feed_wall_time": (
            "" if trigger_snapshot.get("wall_time") is None else f"{float(trigger_snapshot['wall_time']):.6f}"
        ),
        "trigger_digital_input_bits": (
            "" if trigger_snapshot.get("digital_input_bits") is None else str(trigger_snapshot["digital_input_bits"])
        ),
        "trigger_di1": "" if trigger_snapshot.get("di1") is None else str(int(bool(trigger_snapshot["di1"]))),
    }
    row.update(pose_fields("start_flange", start_pose))
    row.update(pose_fields("target_flange", target_pose))
    row.update(pose_fields("trigger_flange", trigger_pose))
    row.update(pose_fields("stop_flange", stop_pose))
    row.update(vector_fields("trigger_joint", trigger_snapshot.get("joints"), 6))
    return row


def jog_target(current_pose, axis, sign, state):
    """
    基于当前位姿和 JOG 指令计算目标位姿。

    沿指定轴（x/y/z/rx/ry/rz）施加一个步长的增量。
    步长由 state 中的 linear_step_mm（平移轴）或 angular_step_deg
    （旋转轴）决定。

    旋转轴的判断：轴名以 'r' 开头（rx/ry/rz）视为旋转轴，使用角度步长；
    否则视为平移轴，使用线性步长。

    Args:
        current_pose (list[float]): 当前法兰位姿。
        axis (str): 轴名，'x', 'y', 'z', 'rx', 'ry', 'rz' 之一。
        sign (float): 运动方向，+1.0 为正方向，-1.0 为负方向。
        state (dict): 会话状态，包含 linear_step_mm 和 angular_step_deg。

    Returns:
        list[float]: 目标法兰位姿。
    """
    target = list(current_pose)
    index = AXIS_INDEX[axis]
    step = state["angular_step_deg"] if axis.startswith("r") else state["linear_step_mm"]
    target[index] += sign * step
    return target


def vector_jog_target(current_pose, direction, distance_mm):
    """
    沿基坐标系下的任意空间向量方向 JOG 指定距离。

    将 direction 向量归一化后乘以 distance_mm，加到当前位置的前 3 维
    （X/Y/Z）上。旋转部分保持不变。

    这实现了沿任意方向的线性运动，不限于坐标轴方向。

    Args:
        current_pose (list[float]): 当前法兰位姿 [x, y, z, rx, ry, rz]。
        direction (list[float]): 运动方向向量 [dx, dy, dz]（无需归一化）。
        distance_mm (float): 沿方向向量的运动距离（毫米）。

    Returns:
        list[float]: 目标法兰位姿。

    Raises:
        ValueError: direction 为零向量或 distance_mm <= 0。
    """
    length = math.sqrt(sum(value * value for value in direction))
    if length <= 1e-9:
        raise ValueError("vector direction cannot be zero")
    if distance_mm <= 0:
        raise ValueError("vector distance must be positive")
    target = list(current_pose)
    for index in range(3):
        target[index] += direction[index] / length * distance_mm
    return target


def pose_reached(pose, target, position_tolerance_mm, orientation_tolerance_deg):
    """
    检查当前位姿是否已到达目标位姿的容差范围内。

    位置误差：三维欧几里得距离（X/Y/Z 的平方和开根号）。
    姿态误差：各角度差值的最大绝对值（使用 angle_delta_deg 处理 ±180° 环绕）。

    Args:
        pose (list[float]): 当前位姿。
        target (list[float]): 目标位姿。
        position_tolerance_mm (float): 位置容差（毫米）。
        orientation_tolerance_deg (float): 姿态容差（度）。

    Returns:
        bool: 是否在容差范围内。
    """
    position_error = math.sqrt(sum((pose[index] - target[index]) ** 2 for index in range(3)))
    orientation_error = max(abs(angle_delta_deg(pose[index], target[index])) for index in range(3, 6))
    return position_error <= position_tolerance_mm and orientation_error <= orientation_tolerance_deg


def issue_movl(node, target_pose):
    """
    发起一次异步 MovL 运动，不等待完成立即返回 future。

    使用 call_async_checked 确保服务端已接受请求。返回的 future
    可在后续循环中通过 check_ready_future 轮询最终结果。
    这种"先发后查"模式让主循环能够同时监控运动状态和 DI1 信号。

    Args:
        node (ProbeTouch): ProbeTouch 节点实例。
        target_pose (list[float]): 目标位姿 [x, y, z, rx, ry, rz]。

    Returns:
        Future: 异步调用 future，用于后续检查结果。
    """
    req = MovL.Request()
    req.mode = False
    req.a, req.b, req.c, req.d, req.e, req.f = target_pose
    future, _ = node.call_async_checked(node.movl_cli, req)
    return future


def wait_jog(
    node,
    target_pose,
    timeout_sec,
    position_tolerance_mm,
    orientation_tolerance_deg,
    command_future=None,
    command_client=None,
    stop_requested=None,
    progress_callback=None,
):
    """
    等待一次 JOG 运动完成，同时监控 DI1 触发——双重监控逻辑。

    本函数在整个 JOG 运动期间同时关注两件事：

    1. DI1 触发检测（优先级最高）
       在每次 spin_once 中检查 FeedInfo 快照的 di1 字段。
       一旦发现 DI1 从 0 变为 1（测头接触工件），立即：
         a. 保存触发时刻的快照（trigger_snapshot）
         b. 调用 stop_fast_then_confirm 停止机器人
         c. 跳出循环

    2. 位姿到达检测
       持续比较 feed_pose 与 target_pose 之间的位置误差和姿态误差。
       当两者都在容差范围内时，标记为 reached_target。
       此时进入"到达后宽限期"（POST_REACH_TRIGGER_GRACE_SEC = 0.25 秒）：
         - 在宽限期内继续监控 DI1，捕获"恰好到达时触发"的边界情况
         - 这是为了防止在极慢运动中触发信号刚好在到达时刻到达

    超时处理：
       如果 timeout_sec 内既未触发也未到达，执行 stop_fast_then_confirm
       并抛出 RuntimeError。

    安全停止策略：
       - DI1 触发时：立即 stop_fast_then_confirm
       - FeedInfo 过期时：立即 stop_fast_then_confirm（通信异常保护）
       - 超时时：立即 stop_fast_then_confirm

    Args:
        node (ProbeTouch): ProbeTouch 节点实例。
        target_pose (list[float]): 本次 JOG 的目标位姿。
        timeout_sec (float): 等待超时秒数。
        position_tolerance_mm (float): 位置容差（毫米）。
        orientation_tolerance_deg (float): 姿态容差（度）。
        command_future: 可选的异步运动服务 future。
        command_client: future 对应的服务客户端；默认使用 MovL 客户端。
        stop_requested: 可选的线程安全取消检查；取消时主动 Stop，返回
            (trigger_snapshot, False)。Stop 期间收到的 DI1 仍会被记录。
        progress_callback: 可选的反馈快照回调，用于运动期间刷新显示。

    Returns:
        tuple: (trigger_snapshot, reached_target)
            - trigger_snapshot (dict | None): DI1 触发时的 FeedInfo 快照；
              未触发则为 None。
            - reached_target (bool): 是否到达了目标位姿。

    Raises:
        RuntimeError: JOG 超时（既未触发也未到达目标位姿）。
    """
    deadline = time.monotonic() + timeout_sec
    trigger_snapshot = None
    reached_target = False

    def stop_on_request():
        node.stop_fast_then_confirm()
        # Stop 的服务等待会继续接收 FeedInfo；保留与松开同时发生的触发。
        snapshot = node.feed_snapshot()
        return (dict(snapshot) if snapshot.get("di1") else None), False

    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=PROBE_SPIN_TIMEOUT_SEC)
        if command_future is not None and command_future.done():
            try:
                node.check_ready_future(command_client or node.movl_cli, command_future)
            except Exception:
                node.stop_fast_then_confirm()
                raise
        if node.last_feed_time is None or time.monotonic() - node.last_feed_time > 0.2:
            node.stop_fast_then_confirm()
            raise RuntimeError("FeedInfo became stale during jog; motion stopped")
        snapshot = node.feed_snapshot()
        if progress_callback is not None:
            progress_callback(snapshot)
        if snapshot.get("di1"):
            trigger_snapshot = dict(snapshot)
            node.stop_fast_then_confirm()
            break
        if stop_requested is not None and stop_requested():
            return stop_on_request()
        pose = snapshot.get("pose")
        if pose is not None and pose_reached(pose, target_pose, position_tolerance_mm, orientation_tolerance_deg):
            reached_target = True
            grace_deadline = time.monotonic() + POST_REACH_TRIGGER_GRACE_SEC
            while rclpy.ok() and time.monotonic() < grace_deadline:
                rclpy.spin_once(node, timeout_sec=PROBE_SPIN_TIMEOUT_SEC)
                late_snapshot = node.feed_snapshot()
                if late_snapshot.get("di1"):
                    trigger_snapshot = dict(late_snapshot)
                    node.stop_fast_then_confirm()
                    break
                if stop_requested is not None and stop_requested():
                    return stop_on_request()
            break
    if trigger_snapshot is None and not reached_target:
        node.stop_fast_then_confirm()
        raise RuntimeError("jog command timed out; motion stopped")
    return trigger_snapshot, reached_target


def retract_opposite(node, start_pose, target_pose, retract_mm, timeout_sec, stop_requested=None):
    """
    沿 JOG 运动的反方向回退指定距离。

    在 DI1 触发并停止后调用，计算运动方向向量（target - start）的
    单位方向，然后从停止位姿沿反方向移动 retract_mm 距离。

    这种"原路返回"的回退策略能有效使测头脱离接触表面，避免
    测头长时间受压。同时也不移动其他轴，仅回退 X/Y/Z 位置。

    计算步骤：
      1. 计算方向向量：direction = target_pose - start_pose（前 3 轴）
      2. 归一化为单位向量
      3. 计算回退位姿：retract_pose = current_pose - direction/|direction| * retract_mm
      4. 保持位姿的旋转部分不变

    Args:
        node (ProbeTouch): ProbeTouch 节点实例。
        start_pose (list[float]): JOG 运动起始位姿。
        target_pose (list[float]): JOG 运动目标位姿（用于计算方向）。
        retract_mm (float): 回退距离（毫米）。<= 0 则不回退。
        timeout_sec (float): 回退运动超时秒数。
        stop_requested: 可选取消检查，供按住点动的急停/关闭打断回退。

    Returns:
        list[float] | None: 回退后的位姿，或 None（未执行回退）。
    """
    if retract_mm <= 0:
        return None
    direction = [target_pose[index] - start_pose[index] for index in range(3)]
    length = math.sqrt(sum(value * value for value in direction))
    if length <= 1e-9:
        return None
    pose = current_pose(node, max_age_sec=0.5)
    retract_pose = list(pose)
    for index in range(3):
        retract_pose[index] -= direction[index] / length * retract_mm
    if stop_requested is not None:
        if stop_requested():
            return None
        future = issue_movl(node, retract_pose)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=PROBE_SPIN_TIMEOUT_SEC)
            if stop_requested():
                node.stop_fast_then_confirm()
                return None
            if node.last_feed_time is None or time.monotonic() - node.last_feed_time > 0.2:
                node.stop_fast_then_confirm()
                raise RuntimeError("FeedInfo became stale during retract; motion stopped")
            node.check_ready_future(node.movl_cli, future)
            actual = node.feed_snapshot().get("pose")
            # 回退时 DI1 初始为高，允许沿反方向解除触发。
            if actual is not None and pose_reached(actual, retract_pose, 0.03, 0.08):
                return retract_pose
        node.stop_fast_then_confirm()
        raise RuntimeError("retract command timed out; motion stopped")
    node.move_l(retract_pose, timeout_sec=timeout_sec)
    node.wait_until_pose(retract_pose, timeout_sec=timeout_sec)
    return retract_pose


def print_status(node, state):
    """
    打印当前状态摘要：位姿、DI1 状态和会话参数。

    在交互式提示符中用于快速查看当前状态，包括：
      - 当前法兰位姿（6 个值，4 位小数）
      - DI1 电平（0 或 1）
      - 线性步长、角度步长、速度倍率
      - 当前测球 ID、分支标签、自动回退距离

    Args:
        node (ProbeTouch): ProbeTouch 节点实例。
        state (dict): 当前会话状态字典。
    """
    pose = current_pose(node, max_age_sec=0.5)
    di1 = fresh_feed_di1(node, max_age_sec=0.5)
    print(f"pose: {format_pose(pose)}")
    print(f"DI1: {int(di1)}")
    print(
        "state: "
        f"linear_step={state['linear_step_mm']:.4f}mm "
        f"angular_step={state['angular_step_deg']:.4f}deg "
        f"speed={state['speed']} "
        f"ball={state['physical_ball_id']} "
        f"branch={state['branch']} "
        f"auto_retract={state['auto_retract_mm']:.4f}mm"
    )


def print_help():
    """打印交互式命令帮助信息，列出所有支持的指令和参数。"""
    print(
        """
commands:
  x+ x- y+ y- z+ z-       jog linear axes by current step
  vec <dx> <dy> <dz> [mm]  jog along a normalized base-frame vector
  rx+ rx- ry+ ry- rz+ rz- jog orientation axes by current angular step
  step <mm>               set linear jog step, e.g. step 0.5
  rstep <deg>             set angular jog step, e.g. rstep 1
  speed <1..5>            set SpeedFactor used before jogs
  retract <mm>            set automatic opposite linear retract after trigger
  ball <id>               set physical ruby ball id
  branch <name>           set branch label written to CSV
  note <text>             set operator note written to CSV
  pose                    print current pose and DI1
  di                      print DI1
  stop                    send Stop()
  help                    show this help
  quit                    exit
""".strip()
    )


def parse_jog_command(command):
    """
    解析 JOG 命令字符串，提取轴名称和运动方向。

    支持的格式：
      - ax+：沿 axis 正方向运动（如 "x+", "rz+"）
      - ax-：沿 axis 负方向运动（如 "y-", "rx-"）

    支持的轴：x, y, z, rx, ry, rz

    Args:
        command (str): 命令字符串（小写化后匹配）。

    Returns:
        tuple[str | None, float | None]:
            (axis, sign)，axis 为轴名，sign 为 +1.0 或 -1.0。
            无法解析时返回 (None, None)。
    """
    cmd = command.lower()
    for axis in ("rx", "ry", "rz", "x", "y", "z"):
        if cmd == axis + "+":
            return axis, 1.0
        if cmd == axis + "-":
            return axis, -1.0
    return None, None


def execute_jog(node, args, state, command, target_from_start):
    """
    执行一次完整的 JOG 运动并记录接触。

    完整流程：
      1. 等待 FeedInfo 新鲜
      2. 检查 DI1 初始状态（必须为低电平）
      3. 获取起始位姿
      4. 通过 target_from_start 回调计算目标位姿
      5. 设置速度倍率
      6. 发起异步 MovL（issue_movl）
      7. 等待运动完成或 DI1 触发（wait_jog，双重监控）
      8. 检查 MovL 的最终结果
      9. 如果触发了：
         a. 增加 sample_index
         b. 生成 CSV 记录行（make_contact_row）
         c. 写入 CSV 文件
         d. 可选的反向回退（retract_opposite）
         e. 打印回退后的 DI1 状态

    异常场景处理：
      - 初始 DI1 为高电平：打印警告，不运动（防止碰撞）
      - wait_jog 超时：抛出 RuntimeError
      - FeedInfo 过期：自动 stop，抛出 RuntimeError

    Args:
        node (ProbeTouch): ProbeTouch 节点实例。
        args: 命令行参数对象。
        state (dict): 会话状态（包含速度、步长等参数）。
        command (str): JOG 命令字符串（用于 CSV 记录）。
        target_from_start (Callable): 接收起始位姿、返回目标位姿的回调函数。
    """
    node.wait_fresh_feed()
    if fresh_feed_di1(node):
        print("DI1 is already ON; release probe before jogging")
        return

    start_pose = current_pose(node)
    target_pose = target_from_start(start_pose)
    print(f"target: {format_pose(target_pose)}")
    node.set_speed(state["speed"])
    future = issue_movl(node, target_pose)
    trigger_snapshot, reached_target = wait_jog(
        node,
        target_pose,
        args.timeout_sec,
        args.position_tolerance_mm,
        args.orientation_tolerance_deg,
    )
    if future.done():
        node.check_ready_future(node.movl_cli, future)
    if trigger_snapshot is None:
        print("reached target; no DI1 trigger")
        return

    stop_pose = current_pose(node, max_age_sec=0.5)
    state["sample_index"] += 1
    row = make_contact_row(args, state, command, start_pose, target_pose, trigger_snapshot, stop_pose)
    write_row(args.output, row)
    print(f"DI1 triggered; recorded sample {state['sample_index']}")
    print(f"trigger pose: {format_pose(trigger_snapshot.get('pose') or stop_pose)}")
    print(f"stop pose:    {format_pose(stop_pose)}")
    retract_pose = retract_opposite(node, start_pose, target_pose, state["auto_retract_mm"], args.timeout_sec)
    if retract_pose is not None:
        print(f"retracted:    {format_pose(retract_pose)}")
    print(f"DI1 after retract: {int(fresh_feed_di1(node, max_age_sec=0.5))}")


def main():
    """
    命令行入口：启动交互式 JOG 测头接触记录终端。

    命令行参数：
      --output              CSV 输出路径（默认 data/jog_probe_contacts.csv）
      --session-id          会话标识符
      --workpiece-id        工件标识符（默认 calibration_sphere_20mm）
      --artifact-id         工件类型标识（默认 standard_sphere_20mm）
      --artifact-type       工件类型（默认 sphere）
      --physical-ball-id    物理测球 ID（默认 1）
      --branch              分支标签（默认 y_neg）
      --operator-note       操作员备注
      --linear-step-mm      线性步长（毫米，默认 0.5）
      --angular-step-deg    角度步长（度，默认 1.0）
      --auto-retract-mm     自动反向回退距离（默认 1.0 mm）
      --speed               速度倍率 1~5（默认 1）
      --timeout-sec         运动超时秒数（默认 5.0）
      --position-tolerance-mm   位置容差（默认 0.08 mm）
      --orientation-tolerance-deg  姿态容差（默认 0.08 deg）

    交互式命令：
      x+ x- y+ y- z+ z-       笛卡尔轴增量运动
      vec <dx> <dy> <dz> [mm]  沿任意向量方向运动
      rx+ rx- ry+ ry- rz+ rz-  旋转轴增量运动
      step <mm>                设置线性步长
      rstep <deg>              设置角度步长
      speed <1..5>             设置速度倍率
      retract <mm>             设置自动回退距离
      ball <id>                设置测球 ID
      branch <name>            设置分支标签
      note <text>              设置备注
      pose                     打印当前位姿和状态
      di                       打印 DI1 状态
      stop                     发送停止命令
      help                     显示帮助
      quit                     退出

    流程：
      1. 解析参数并初始化状态
      2. 初始化 ROS2 并创建 ProbeTouch 节点
      3. 等待服务就绪和 FeedInfo 更新
      4. 进入交互式循环，等待用户输入 JOG 命令
      5. 每次 JOG 执行 execute_jog 流程
      6. 退出时清理并关闭 ROS2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--session-id", default="")
    parser.add_argument("--workpiece-id", default="calibration_sphere_20mm")
    parser.add_argument("--artifact-id", default="standard_sphere_20mm")
    parser.add_argument("--artifact-type", default="sphere")
    parser.add_argument("--physical-ball-id", default="1")
    parser.add_argument("--branch", default="y_neg")
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--linear-step-mm", type=float, default=0.5)
    parser.add_argument("--angular-step-deg", type=float, default=1.0)
    parser.add_argument("--auto-retract-mm", type=float, default=1.0)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.08)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.08)
    args = parser.parse_args()

    if args.linear_step_mm <= 0 or args.angular_step_deg <= 0:
        raise SystemExit("step sizes must be positive")
    if not 1 <= args.speed <= 5:
        raise SystemExit("--speed must be between 1 and 5")
    if args.auto_retract_mm < 0:
        raise SystemExit("--auto-retract-mm cannot be negative")

    state = {
        "linear_step_mm": float(args.linear_step_mm),
        "angular_step_deg": float(args.angular_step_deg),
        "auto_retract_mm": float(args.auto_retract_mm),
        "speed": int(args.speed),
        "physical_ball_id": args.physical_ball_id,
        "branch": args.branch,
        "operator_note": args.operator_note,
        "sample_index": 0,
    }

    rclpy.init()
    node = ProbeTouch()
    try:
        node.wait_services(10.0)
        node.wait_fresh_feed()
        print("interactive jog recorder ready")
        print(f"output: {args.output}")
        print_help()
        print_status(node, state)

        while True:
            try:
                line = input("jog> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break
            if not line:
                continue
            parts = shlex.split(line)
            command = parts[0].lower()

            if command in ("quit", "exit", "q"):
                break
            if command == "help":
                print_help()
                continue
            if command == "pose":
                print_status(node, state)
                continue
            if command == "di":
                print(f"DI1: {int(fresh_feed_di1(node, max_age_sec=0.5))}")
                continue
            if command == "stop":
                node.stop_fast_then_confirm()
                print("Stop sent")
                continue
            if command == "step" and len(parts) == 2:
                value = float(parts[1])
                if value <= 0:
                    print("step must be positive")
                else:
                    state["linear_step_mm"] = value
                continue
            if command == "rstep" and len(parts) == 2:
                value = float(parts[1])
                if value <= 0:
                    print("rstep must be positive")
                else:
                    state["angular_step_deg"] = value
                continue
            if command == "speed" and len(parts) == 2:
                value = int(parts[1])
                if not 1 <= value <= 5:
                    print("speed must be 1..5")
                else:
                    state["speed"] = value
                    node.set_speed(value)
                continue
            if command == "retract" and len(parts) == 2:
                value = float(parts[1])
                if value < 0:
                    print("retract cannot be negative")
                else:
                    state["auto_retract_mm"] = value
                continue
            if command == "ball" and len(parts) == 2:
                state["physical_ball_id"] = parts[1]
                continue
            if command == "branch" and len(parts) == 2:
                state["branch"] = parts[1]
                continue
            if command == "note":
                state["operator_note"] = line.partition(" ")[2]
                continue
            if command in ("vec", "vector", "line") and len(parts) in (4, 5):
                direction = [float(parts[index]) for index in range(1, 4)]
                distance_mm = float(parts[4]) if len(parts) == 5 else state["linear_step_mm"]
                label = f"vec {direction[0]:g} {direction[1]:g} {direction[2]:g} {distance_mm:g}"
                execute_jog(
                    node,
                    args,
                    state,
                    label,
                    lambda start_pose, direction=direction, distance_mm=distance_mm: vector_jog_target(
                        start_pose, direction, distance_mm
                    ),
                )
                continue

            axis, sign = parse_jog_command(command)
            if axis is None:
                print("unknown command; type help")
                continue
            execute_jog(
                node,
                args,
                state,
                command,
                lambda start_pose, axis=axis, sign=sign: jog_target(start_pose, axis, sign, state),
            )
    except (RuntimeError, ValueError, TimeoutError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
