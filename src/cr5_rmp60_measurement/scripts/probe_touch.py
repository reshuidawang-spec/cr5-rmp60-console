#!/usr/bin/env python3
"""
RMP60 触发式测头垂直探测基类与工具函数
========================================

本模块是 RMP60 触发式测头操作的基础底座。核心是 ProbeTouch ROS2 Node,
它封装了与 Dobot CR5 机器人控制器的以下通信通道：

  - GetPose 服务 ：读取当前法兰末端位姿
  - MovL 服务    ：以直线运动 (关节运动) 方式移动到目标位姿
  - SpeedFactor 服务：设置运动速度倍率
  - Stop 服务     ：紧急停止机器人
  - DI 服务      ：读取机器人数字输入口状态
  - FeedInfo 话题：高频订阅机器人状态反馈（位姿、关节角、数字输入位等）

DI1（数字输入 1）是连接 RMP60 测头的触发信号通道。
当测头探针接触工件表面时，RMP60 会发出触发信号，DI1 从 0 跳变为 1。
模块检测到 DI1 上升沿后立即发送 Stop 命令，并记录触发时刻的机器人位姿，
再根据法兰-测头长度（flange_mm + stylus_mm）估算探针球心接触点坐标。

坐标系约定：
  - 所有位姿均为机器人基坐标系下的法兰末端位姿 [x, y, z, rx, ry, rz]
  - z 轴向上为正；垂直探测方向为机器人 Z 负方向（向下）
  - 角度单位：度（deg）
  - 长度单位：毫米（mm）

历史说明：
  第一版仅支持沿机器人 Z 负方向的垂直探测，只使用 DI1 信号。
  未配置机器人工具坐标系（Tool frame），法兰到探针尖端的长度仅用于
  记录估算的接触点坐标时使用。
"""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dobot_msgs_v4.srv import DI, GetPose, MovL, MoveJog, SpeedFactor, Stop, StopMoveJog

VERTICAL_PROBE_ACK_ARG = "--ack-vertical-stylus-ok"
VERTICAL_PROBE_ACK_DEST = "ack_vertical_stylus_ok"
PROBE_SPIN_TIMEOUT_SEC = 0.001
PROBE_MOVL_ACCEPT_TIMEOUT_SEC = 0.02
PROBE_SEGMENT_SETTLE_SEC = 0.03
PROBE_SEGMENT_POSITION_TOLERANCE_MM = 0.03


def parse_pose(text):
    """
    从机器人服务返回的字符串中解析 6 自由度位姿。

    服务返回格式通常为："{x, y, z, rx, ry, rz}"
    使用正则提取花括号内的内容，按逗号分割并转换为浮点数。

    Args:
        text (str): 包含位姿的字符串，如 "{123.4, 56.7, 89.0, 0.0, 0.0, 90.0}"。

    Returns:
        list[float]: 6 元素位姿列表 [x, y, z, rx, ry, rz]。

    Raises:
        ValueError: 字符串格式不符合预期，或位数不为 6。
    """
    match = re.search(r"\{([^}]*)\}", text)
    if not match:
        raise ValueError(f"cannot parse pose from: {text}")
    values = [float(item.strip()) for item in match.group(1).split(",")]
    if len(values) != 6:
        raise ValueError(f"expected 6 pose values, got {len(values)}: {text}")
    return values


def check_service_result(client, result):
    """
    检查 ROS2 服务调用结果，遇到错误码时抛出 RuntimeError。

    具体检查：
      1. result 为 None 时报错（服务无响应）
      2. 如果 result 有 res 属性且 res != 0，说明服务端返回了错误码，
         附带 robot_return 字段的详细错误信息

    Args:
        client: ROS2 服务客户端（用于获取服务名）。
        result: 服务调用返回的结果对象。

    Returns:
        校验通过的结果对象。

    Raises:
        RuntimeError: 服务返回错误码或 result 为 None。
    """
    if result is None:
        raise RuntimeError(f"{client.srv_name} returned no result")
    if hasattr(result, "res") and int(result.res) != 0:
        detail = getattr(result, "robot_return", "")
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{client.srv_name} returned res={result.res}{suffix}")
    return result


class ProbeTouch(Node):
    """
    RMP60 触发式测头 ROS2 Node 基类。

    职责：
      1. 维护与机器人控制器的服务客户端（GetPose, MovL, SpeedFactor, Stop, DI）
      2. 订阅 FeedInfo 话题，缓存最新机器人状态（位姿、关节角、DI 状态）
      3. 提供探测循环的核心原语：wait_fresh_feed, wait_until_pose, call_async_checked
      4. 提供 stop_fast_then_confirm 等安全停止策略

    典型用法：
        node = ProbeTouch()
        node.wait_services(10.0)
        run_probe_cycle(node, args)

    注：本类不假定任何特定的工具坐标系；工具长度（flange_mm + stylus_mm）
    在 run_probe_cycle 中由调用方传入，用于最终计算接触点。
    """

    def __init__(self):
        """初始化 ProbeTouch 节点。

        创建以下 ROS2 客户端和订阅：
          - FeedInfo 订阅：/dobot_bringup_ros2/msg/FeedInfo，高频回调更新缓存
          - DI 服务客户端：/dobot_bringup_ros2/srv/DI
          - GetPose 服务客户端：/dobot_bringup_ros2/srv/GetPose
          - MovL 服务客户端：/dobot_bringup_ros2/srv/MovL
          - SpeedFactor 服务客户端：/dobot_bringup_ros2/srv/SpeedFactor
          - Stop 服务客户端：/dobot_bringup_ros2/srv/Stop

        内部状态变量：
          - self.di1 (bool | None)：最新 FeedInfo 中的 DI1 值
          - self.last_feed_time (float | None)：上次收到 FeedInfo 的 monotonic 时间
          - self.feed_pose (list[float] | None)：最新 FeedInfo 中的工具位姿
          - self.feed_joints (list[float] | None)：最新 FeedInfo 中的关节角
          - self.feed_sequence (int)：收到 FeedInfo 的累计计数
        """
        super().__init__("probe_touch")
        self.di1 = None
        self.last_feed_time = None
        self.last_feed_wall_time = None
        self.feed_sequence = 0
        self.feed_pose = None
        self.feed_joints = None
        self.feed_digital_input_bits = None
        self.feed_jog_status = None
        self.feed_robot_mode = None
        self.feed_error_status = None
        self.create_subscription(String, "/dobot_bringup_ros2/msg/FeedInfo", self._feed_cb, 10)
        self.di_cli = self.create_client(DI, "/dobot_bringup_ros2/srv/DI")
        self.pose_cli = self.create_client(GetPose, "/dobot_bringup_ros2/srv/GetPose")
        self.movl_cli = self.create_client(MovL, "/dobot_bringup_ros2/srv/MovL")
        self.move_jog_cli = self.create_client(MoveJog, "/dobot_bringup_ros2/srv/MoveJog")
        self.stop_move_jog_cli = self.create_client(StopMoveJog, "/dobot_bringup_ros2/srv/StopMoveJog")
        self.speed_cli = self.create_client(SpeedFactor, "/dobot_bringup_ros2/srv/SpeedFactor")
        self.stop_cli = self.create_client(Stop, "/dobot_bringup_ros2/srv/Stop")

    def _feed_cb(self, msg):
        """
        FeedInfo 话题的回调函数。

        从 JSON 格式的 FeedInfo 消息中解析出以下信息并缓存到实例变量：
          - digital_input_bits -> self.di1（最低位对应 DI1）
          - tool_vector_actual -> self.feed_pose（6 自由度法兰位姿）
          - q_actual -> self.feed_joints（6 个关节角）
          - 同时更新 self.last_feed_time（monotonic）和 self.last_feed_wall_time（wall）

        DI1 状态提取逻辑：digital_input_bits 的最低比特位（bit 0）对应 DI1，
        因此 di1 = bool(bits & 0x01)。
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.feed_sequence += 1
        digital_input_bits = int(data.get("digital_input_bits", 0))
        self.feed_digital_input_bits = digital_input_bits
        self.di1 = bool(digital_input_bits & 0x01)
        self.feed_jog_status = data.get("JogStatus")
        self.feed_robot_mode = data.get("robot_mode")
        self.feed_error_status = data.get("ErrorStatus")
        pose = data.get("tool_vector_actual")
        if isinstance(pose, list) and len(pose) >= 6:
            self.feed_pose = [float(v) for v in pose[:6]]
        joints = data.get("q_actual")
        if isinstance(joints, list) and len(joints) >= 6:
            self.feed_joints = [float(v) for v in joints[:6]]
        self.last_feed_time = time.monotonic()
        self.last_feed_wall_time = time.time()

    def feed_snapshot(self):
        """
        返回当前 FeedInfo 缓存的快照字典。

        快照包含：
          - sequence (int)：FeedInfo 累计接收计数
          - monotonic_time (float | None)：上次接收的 monotonic 时间
          - wall_time (float | None)：上次接收的 wall 时间
          - pose (list[float] | None)：最新法兰位姿 [x, y, z, rx, ry, rz]
          - joints (list[float] | None)：最新关节角 [j1..j6]
          - digital_input_bits (int | None)：完整数字输入位图
          - di1 (bool | None)：DI1 触发状态

        返回的 pose 和 joints 均为拷贝，外部修改不影响内部缓存。
        """
        return {
            "sequence": self.feed_sequence,
            "monotonic_time": self.last_feed_time,
            "wall_time": self.last_feed_wall_time,
            "pose": list(self.feed_pose) if self.feed_pose is not None else None,
            "joints": list(self.feed_joints) if self.feed_joints is not None else None,
            "digital_input_bits": self.feed_digital_input_bits,
            "di1": self.di1,
            "jog_status": self.feed_jog_status,
            "robot_mode": self.feed_robot_mode,
            "error_status": self.feed_error_status,
        }

    def wait_services(self, timeout_sec):
        """
        等待所有必要的 ROS2 服务可用。

        依次等待 DI、GetPose、MovL、SpeedFactor、Stop 五个服务，
        任一服务在 timeout_sec 内不可用则抛出 RuntimeError。

        Args:
            timeout_sec (float): 等待超时秒数（对所有服务累计使用同一 deadline）。
        """
        deadline = time.monotonic() + timeout_sec
        for client in (self.di_cli, self.pose_cli, self.movl_cli, self.speed_cli, self.stop_cli):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not client.wait_for_service(timeout_sec=remaining):
                raise RuntimeError(f"service not available: {client.srv_name}")

    def call(self, client, request, timeout_sec=10.0):
        """
        同步调用 ROS2 服务（内部通过 spin 驱动异步 future）。

        在 timeout_sec 内循环 spin_once 等待 future 完成，
        超时则抛出 TimeoutError。

        Args:
            client: ROS2 服务客户端（已等待就绪）。
            request: 服务请求对象。
            timeout_sec (float): 调用超时秒数。

        Returns:
            服务响应对象，已通过 check_service_result 校验。

        Raises:
            TimeoutError: 超时后 future 仍未完成。
            RuntimeError: 服务返回错误码。
        """
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if future.done():
                return check_service_result(client, future.result())
        raise TimeoutError(f"service timeout: {client.srv_name}")

    def call_async_checked(self, client, request, accept_timeout_sec=PROBE_MOVL_ACCEPT_TIMEOUT_SEC):
        """
        发起异步服务调用，并在短时间内检查服务端是否已接受请求。

        与 call() 不同，本方法不等待请求完成，仅等待服务端确认已接收请求
        （accept_timeout_sec 内 future 变为 done）。如果超时则返回
        (future, None)，调用方后续可通过 check_ready_future() 轮询结果。

        本设计用于探测循环中的分段 MovL 调用：先发出运动指令，
        然后一边监控 FeedInfo 一边等待运动完成或被 DI1 中断。
        这种"先发后查"的模式避免了同步等待阻塞探测监控循环。

        Args:
            client: ROS2 服务客户端。
            request: 服务请求对象。
            accept_timeout_sec (float): 等待服务端接受请求的超时秒数。

        Returns:
            tuple[Future, result | None]: (future, 立即完成时的结果, 否则 None)
        """
        future = client.call_async(request)
        deadline = time.monotonic() + accept_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=PROBE_SPIN_TIMEOUT_SEC)
            if future.done():
                result = check_service_result(client, future.result())
                return future, result
        return future, None

    def check_ready_future(self, client, future):
        """
        检查异步调用 future 是否已完成，若完成则返回校验后的结果。

        这是 call_async_checked 的配套方法，用于在探测循环中安全地
        轮询之前发起的异步调用的最终结果，而不会阻塞循环。

        Args:
            client: 发起调用的服务客户端。
            future (Future | None): call_async 返回的 future。

        Returns:
            校验后的服务结果（如果 future 已完成），否则返回 None。
        """
        if future is not None and future.done():
            return check_service_result(client, future.result())
        return None

    def get_pose(self):
        """
        调用 GetPose 服务获取当前法兰末端位姿。

        使用 user=0, tool=0，表示返回的是机器人基坐标系下的默认工具位姿。
        返回值通过 parse_pose 从服务返回的字符串中解析出 [x, y, z, rx, ry, rz]。

        Returns:
            list[float]: 6 元素位姿列表。
        """
        req = GetPose.Request()
        req.user = 0
        req.tool = 0
        return parse_pose(self.call(self.pose_cli, req).robot_return)

    def read_di1(self):
        """
        通过 DI 服务读取数字输入 DI1 的当前状态。

        优先调用 DI 服务。若 DI 服务超时，则回退到最近 FeedInfo 缓存中的
        DI1 值（要求缓存时间不超过 0.2 秒）。这种"服务优先、FeedInfo 回退"
        的策略兼顾了实时可靠性（DI 服务直接读取硬件）和鲁棒性（服务故障时
        仍有高频 FeedInfo 作为后备）。

        返回值通过解析服务返回字符串中的 "{value}" 格式获得。

        Returns:
            bool: True 表示 DI1 为高电平（测头触发），False 为低电平。

        Raises:
            TimeoutError: DI 服务超时且 FeedInfo 缓存不可用或过期。
        """
        req = DI.Request()
        req.index = 1
        try:
            result = self.call(self.di_cli, req)
        except TimeoutError:
            if self.di1 is not None and self.last_feed_time is not None and time.monotonic() - self.last_feed_time <= 0.2:
                self.get_logger().warning("DI service timed out; using fresh FeedInfo DI1 state")
                return bool(self.di1)
            raise
        match = re.search(r"\{([^}]*)\}", result.robot_return)
        if match:
            return bool(int(match.group(1).split(",", 1)[0]))
        self.get_logger().warning("DI service returned no robot_return; using FeedInfo DI1 state")
        return bool(self.di1)

    def wait_jog_idle(self, timeout_sec=2.0):
        """Wait until FeedInfo reports controller jog status idle."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.03)
            if self.feed_jog_status in (None, 0):
                return
        raise TimeoutError("timed out waiting for JogStatus=0")

    def start_move_jog(self, axis_id, param_value=None, timeout_sec=2.0, retry_on_busy=True):
        """Start a controller-level continuous jog using Dobot MoveJog."""
        if not self.move_jog_cli.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"service not available: {self.move_jog_cli.srv_name}")
        req = MoveJog.Request()
        req.axis_id = str(axis_id)
        req.param_value = list(param_value or [])
        try:
            self.wait_jog_idle(timeout_sec=timeout_sec)
            return self.call(self.move_jog_cli, req, timeout_sec=timeout_sec)
        except RuntimeError as exc:
            if retry_on_busy and "res=-6" in str(exc):
                self.stop_move_jog(timeout_sec=timeout_sec)
                self.wait_jog_idle(timeout_sec=timeout_sec)
                time.sleep(0.2)
                return self.start_move_jog(
                    axis_id,
                    param_value=param_value,
                    timeout_sec=timeout_sec,
                    retry_on_busy=False,
                )
            raise

    def stop_move_jog(self, timeout_sec=2.0):
        """Stop a controller-level continuous jog using Dobot StopMoveJog."""
        if not self.stop_move_jog_cli.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"service not available: {self.stop_move_jog_cli.srv_name}")
        result = self.call(self.stop_move_jog_cli, StopMoveJog.Request(), timeout_sec=timeout_sec)
        self.wait_jog_idle(timeout_sec=timeout_sec)
        return result

    def wait_fresh_feed(self, timeout_sec=3.0, max_age_sec=0.2):
        """
        等待 FeedInfo 话题的缓存更新到足够"新鲜"的状态。

        在探测或运动开始前调用此方法，确保后续操作基于最新的机器人状态。
        判断标准：self.last_feed_time 不为 None 且其距今时间 <= max_age_sec。

        Args:
            timeout_sec (float): 最长等待时间（秒），超时抛出 RuntimeError。
            max_age_sec (float): 判定为"新鲜"的最大缓存时长（秒）。

        Raises:
            RuntimeError: 超时后 FeedInfo 仍未更新到新鲜状态。
        """
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.last_feed_time is not None and time.monotonic() - self.last_feed_time <= max_age_sec:
                return
        raise RuntimeError("FeedInfo is not fresh; refusing to move")

    def wait_until_pose(
        self,
        target_pose,
        position_tolerance_mm=0.2,
        orientation_tolerance_deg=0.5,
        timeout_sec=10.0,
        command_future=None,
        command_client=None,
    ):
        """
        通过监控 FeedInfo 缓存等待机器人到达目标位姿。

        在 timeout_sec 内持续检查 self.feed_pose 与目标位姿之间的误差：
          - 位置误差 = 欧几里得距离（前 3 维）
          - 姿态误差 = 各角度差值的最大绝对值（后 3 维，考虑 ±180° 环绕）
        两者分别在 position_tolerance_mm 和 orientation_tolerance_deg 内
        即视为到达。

        若超时后仍未到达，则发送 stop_repeated 并抛出 RuntimeError。
        这种"软等待+超时停止"的策略保证了：
          1. 正常到达时及时返回，无额外延迟
          2. 异常情况（如碰撞、通信中断）下有安全停止兜底

        Args:
            target_pose (list[float]): 目标位姿 [x, y, z, rx, ry, rz]。
            position_tolerance_mm (float): 位置容差（毫米）。
            orientation_tolerance_deg (float): 姿态容差（度）。
            timeout_sec (float): 等待超时秒数。

        Returns:
            list[float]: 到达时的实际位姿（来自 feed_pose 或 get_pose）。

        Raises:
            RuntimeError: 超时且位姿未收敛。
        """
        deadline = time.monotonic() + timeout_sec
        last_pose = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if command_future is not None and command_future.done():
                try:
                    self.check_ready_future(command_client or self.movl_cli, command_future)
                except Exception:
                    self.stop_fast_then_confirm()
                    raise
            if self.feed_pose is None:
                continue
            last_pose = list(self.feed_pose)
            position_error = sum((last_pose[index] - target_pose[index]) ** 2 for index in range(3)) ** 0.5
            orientation_error = max(
                abs((last_pose[index] - target_pose[index] + 180.0) % 360.0 - 180.0)
                for index in range(3, 6)
            )
            if position_error <= position_tolerance_mm and orientation_error <= orientation_tolerance_deg:
                return last_pose
        pose = self.get_pose()
        position_error = sum((pose[index] - target_pose[index]) ** 2 for index in range(3)) ** 0.5
        orientation_error = max(
            abs((pose[index] - target_pose[index] + 180.0) % 360.0 - 180.0)
            for index in range(3, 6)
        )
        if position_error <= position_tolerance_mm and orientation_error <= orientation_tolerance_deg:
            return pose
        self.stop_repeated()
        observed = last_pose if last_pose is not None else pose
        raise RuntimeError(f"positioning move did not reach requested pose; last observed pose was {observed}")

    def move_l(self, pose, timeout_sec):
        """
        调用 MovL 服务让机器人直线运动到目标位姿。

        这是同步调用，会阻塞直到运动完成或超时。

        Args:
            pose (list[float]): 目标位姿 [x, y, z, rx, ry, rz]。
            timeout_sec (float): 运动超时秒数。

        Returns:
            服务响应对象。
        """
        req = MovL.Request()
        req.mode = False
        req.a, req.b, req.c, req.d, req.e, req.f = pose
        return self.call(self.movl_cli, req, timeout_sec=timeout_sec)

    def set_speed(self, ratio):
        """
        调用 SpeedFactor 服务设置机器人运动速度倍率。

        探测时通常使用低速（1~5 之间），以确保触发精度和安全性。

        Args:
            ratio (int): 速度倍率值（接口要求整数，Dobot CR5 通常 1~100）。

        Returns:
            服务响应对象。
        """
        req = SpeedFactor.Request()
        req.ratio = int(ratio)
        return self.call(self.speed_cli, req)

    def stop(self, timeout_sec=0.5):
        """
        调用 Stop 服务紧急停止机器人。

        Args:
            timeout_sec (float): 停止服务调用超时秒数。

        Returns:
            服务响应对象。
        """
        return self.call(self.stop_cli, Stop.Request(), timeout_sec=timeout_sec)

    def stop_repeated(self, count=3, interval_sec=0.05):
        """
        多次重复发送 Stop 命令，确保机器人确实停止。

        在某些情况下（如高速运动中触发 DI1），单次 Stop 可能因通信延迟
        或控制器忙而丢失。重复发送提高了停止的可靠性。

        Args:
            count (int): 重复发送 Stop 的次数。
            interval_sec (float): 每次发送之间的间隔秒数。

        Returns:
            最后一次 Stop 调用的结果。
        """
        last = None
        for _ in range(count):
            try:
                last = self.stop()
            except Exception as exc:
                self.get_logger().warning(f"Stop() failed once: {exc}")
            time.sleep(interval_sec)
        return last

    def stop_fast_then_confirm(self, confirm_count=1, confirm_interval_sec=0.0):
        """
        快速停止策略：先立刻异步发出 Stop，再同步重复确认。

        两步策略的好处：
          1. 异步 Stop 立即发出，不等待结果，最大化响应速度。
          2. 随后的 stop_repeated 确保停止已被可靠执行。
          3. 最后检查异步 future 的结果，仅在明确失败时记录警告，
             因为此时机器人已经通过重复 Stop 停止了。

        这是探测循环检测到 DI1 触发后的首选停止方式。

        Args:
            confirm_count (int): 确认停止的重复次数。
            confirm_interval_sec (float): 确认停止的间隔秒数。

        Returns:
            最后一次 stop_repeated 的结果。
        """
        future = self.stop_cli.call_async(Stop.Request())
        last = self.stop_repeated(count=confirm_count, interval_sec=confirm_interval_sec)
        if future.done():
            try:
                check_service_result(self.stop_cli, future.result())
            except RuntimeError as exc:
                self.get_logger().warning(
                    f"first async Stop returned an error but robot is already stopped via backup: {exc}"
                )
        return last


def write_contact(path, row):
    """
    将一次测头接触记录追加写入 CSV 文件。

    如果文件的表头（header）与 row 的键不一致，则会将原文件重命名备份
    （加上时间戳后缀），然后创建新文件写入。这保证了 CSV 格式变更时
    不会静默破坏已有数据。

    Args:
        path (Path): CSV 输出路径。
        row (dict): 包含所有字段的字典，键作为 CSV 表头。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = path.exists()
    if exists:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header != fieldnames:
            backup = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
            os.replace(path, backup)
            print(f"CSV header changed; moved old file to {backup}")
            exists = False
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_probe_motion(approach_mm, allow_long_approach, speed=None, prefix=""):
    """
    验证探测运动参数的有效性。

    - approach_mm 必须为正数
    - approach_mm > 5.0 时必须显式允许（--allow-long-approach）
    - speed（如果提供）必须通过 validate_probe_speed 校验

    Args:
        approach_mm (float): 探测进给距离（毫米）。
        allow_long_approach (bool): 是否允许超过 5 mm 的进给。
        speed (int | None): 速度倍率（可选）。
        prefix (str): 错误消息前缀，用于区分不同探测阶段。

    Raises:
        ValueError: 任一参数校验失败。
    """
    if approach_mm <= 0:
        raise ValueError(f"{prefix}--approach-mm must be positive")
    if approach_mm > 5.0 and not allow_long_approach:
        raise ValueError(f"{prefix}--approach-mm > 5 requires --allow-long-approach")
    if speed is not None:
        validate_probe_speed(speed, prefix=prefix)


def validate_probe_speed(speed, prefix=""):
    """
    验证探测速度倍率是否在安全范围内（1~5）。

    探测速度过高会导致：
      1. 触发后过冲（overshoot）增大，降低接触点定位精度。
      2. 碰撞风险增加，可能损坏测头或工件。

    Args:
        speed (int): 速度倍率值。
        prefix (str): 错误消息前缀。

    Raises:
        ValueError: speed 不在 1~5 范围内。
    """
    if not 1 <= speed <= 5:
        raise ValueError(f"{prefix}--speed must be between 1 and 5 for probing")


def validate_probe_args(args, prefix=""):
    """
    综合验证所有探测命令行参数。

    检查项包括：
      - 实际执行（--execute）时必须确认垂直测头已安装（--ack-vertical-stylus-ok）
      - 探测运动参数（approach_mm, allow_long_approach, speed）
      - 回退距离（retract_mm）必须为正数
      - 超时（timeout）必须为正数
      - 分段步长（probe_step_mm）不能为负数

    Args:
        args: argparse.Namespace 解析结果。
        prefix (str): 错误消息前缀。

    Raises:
        ValueError: 任一参数校验失败。
    """
    if getattr(args, "execute", False) and not getattr(args, VERTICAL_PROBE_ACK_DEST, False):
        raise ValueError(
            f"{prefix}vertical probing real motion requires explicit on-site confirmation; "
            f"add {VERTICAL_PROBE_ACK_ARG} only after confirming the vertical stylus and this path are safe"
        )
    validate_probe_motion(
        args.approach_mm,
        getattr(args, "allow_long_approach", False),
        getattr(args, "speed", None),
        prefix=prefix,
    )
    retract_mm = getattr(args, "retract_mm", None)
    if retract_mm is not None and retract_mm <= 0:
        raise ValueError(f"{prefix}--retract-mm must be positive")
    timeout = getattr(args, "timeout", None)
    if timeout is not None and timeout <= 0:
        raise ValueError(f"{prefix}--timeout must be positive")
    probe_step_mm = getattr(args, "probe_step_mm", 0.0)
    if probe_step_mm is not None and probe_step_mm < 0:
        raise ValueError(f"{prefix}--probe-step-mm cannot be negative")


def add_vertical_probe_ack_argument(parser):
    """
    向 ArgumentParser 添加垂直测头安全确认参数 --ack-vertical-stylus-ok。

    这是一个安全开关，要求操作员在运行垂直探测之前，必须亲自到现场
    确认垂直探针已正确安装且路径安全。不使用此标志时，即使指定了
    --execute 也不会真正执行运动（dry-run 模式）。
    """
    parser.add_argument(
        VERTICAL_PROBE_ACK_ARG,
        dest=VERTICAL_PROBE_ACK_DEST,
        action="store_true",
        help="permit real vertical probing after confirming the vertical stylus is installed and safe",
    )


def effective_probe_step(total_distance_mm, probe_step_mm):
    """
    计算有效的探测分段步长。

    如果 probe_step_mm 为 0 或 None，则不分段，一次走完全程。
    否则取 probe_step_mm 与 total_distance_mm 中的较小值，
    保证分段步长不会超过总距离。

    Args:
        total_distance_mm (float): 探测总距离（毫米）。
        probe_step_mm (float | None): 预设分段步长（毫米）；0 或 None 表示不分段。

    Returns:
        float: 实际使用的分段步长。
    """
    step = float(probe_step_mm or 0.0)
    if step <= 0.0:
        return float(total_distance_mm)
    return min(float(total_distance_mm), step)


def wait_segment_boundary_trigger(node, deadline, stale_message):
    """
    在分段探测的段间边界等待 DI1 触发。

    当机器人完成一段 MovL 并停留在段间边界时，此函数执行短暂的
    等待窗口（PROBE_SEGMENT_SETTLE_SEC），在这段时间内监控 FeedInfo
    中的 DI1 状态。如果在边界处恰好发生了触发，可以在此捕获。

    这种机制解决了以下问题：在极慢的探测速度下，DI1 触发可能发生在
    段间边界处（即前一段刚刚停止时），此时主循环正在准备下一段运动，
    如果不在边界处额外等待，触发信号可能被遗漏。

    参数:
        node (ProbeTouch): ProbeTouch 节点实例。
        deadline (float): 绝对超时时间戳（time.monotonic）。
        stale_message (str): FeedInfo 过期时的错误消息。

    返回:
        dict | None: 触发时的 FeedInfo 快照，若未触发则返回 None。

    异常:
        RuntimeError: FeedInfo 在等待期间过期。
    """
    settle_until = min(time.monotonic() + PROBE_SEGMENT_SETTLE_SEC, deadline)
    while rclpy.ok() and time.monotonic() < settle_until:
        rclpy.spin_once(node, timeout_sec=PROBE_SPIN_TIMEOUT_SEC)
        if node.last_feed_time is None or time.monotonic() - node.last_feed_time > 0.2:
            node.stop_repeated()
            raise RuntimeError(stale_message)
        snapshot = node.feed_snapshot()
        if snapshot.get("di1"):
            return snapshot
    return None


def run_probe_cycle(node, args, cycle_index=None):
    """
    执行一次完整的垂直探测循环。

    流程：
      1. 校验参数，计算法兰-测头总长度（flange_mm + stylus_mm）
      2. 获取当前位姿，计算目标位姿（Z 减去 approach_mm）
      3. 检查 DI1 初始状态（必须为低电平）
      4. 如果 --execute 未指定，则 dry-run 退出
      5. 分段（segmented）探测：
         a. 每次发出一个长度为 probe_step 的 MovL
         b. 在循环中监控 FeedInfo，检查 DI1 是否变为 1
         c. 每段到达后执行 wait_segment_boundary_trigger（捕获段间触发）
         d. 若 DI1 触发，执行 stop_fast_then_confirm 停止
         e. 若所有段执行完毕且未触发，标记为 reached_target
      6. 停止后获取最终位姿，计算估计接触点
      7. 回退 retract_mm 距离
      8. 检查回退后 DI1 是否已恢复低电平（未恢复则报错）
      9. 将记录写入 CSV（通过 write_contact）

    记录字段：
      - timestamp: 触发时的时间戳
      - cycle: 循环索引
      - flange_x/y/z/rx/ry/rz: 触发时的法兰位姿
      - tip_x/y/z_est: 估计的探针球心接触坐标
      - tool_length_mm: 法兰-测头总长度
      - probe_step_mm: 分段步长
      - stop_flange_*: 停止后的法兰位姿

    参数:
        node (ProbeTouch): ProbeTouch 节点实例。
        args: 命令行参数。
        cycle_index (int | None): 可选的循环编号，用于日志前缀。

    返回:
        dict | None: 接触记录的字典（触发时），或 None（未触发）。
    """
    validate_probe_args(args)
    tool_length = args.flange_mm + args.stylus_mm
    start_pose = node.get_pose()
    target_pose = start_pose.copy()
    target_pose[2] -= args.approach_mm
    prefix = f"[cycle {cycle_index}] " if cycle_index is not None else ""
    print(f"{prefix}current flange pose: {start_pose}")
    print(f"{prefix}probe target pose:   {target_pose}")
    print(f"{prefix}assumed flange-to-tip length: {tool_length:.1f} mm")
    print(f"{prefix}probe step: {effective_probe_step(args.approach_mm, getattr(args, 'probe_step_mm', 0.0)):.4f} mm")

    node.wait_fresh_feed()
    if node.read_di1() or node.di1:
        raise RuntimeError("DI1 is already triggered before probing; retract or clear the probe first")

    if not args.execute:
        print(f"{prefix}dry-run only; add --execute to start the probing motion")
        return None

    node.set_speed(args.speed)
    probe_step = effective_probe_step(args.approach_mm, getattr(args, "probe_step_mm", 0.0))
    issued_distance = 0.0
    segment_target = start_pose.copy()
    move_future = None
    immediate_move_result = None

    def issue_next_segment():
        """Issue next segment."""
        nonlocal issued_distance, segment_target, move_future, immediate_move_result
        remaining = args.approach_mm - issued_distance
        step = min(probe_step, remaining)
        issued_distance += step
        segment_target = start_pose.copy()
        segment_target[2] -= issued_distance
        move_req = MovL.Request()
        move_req.mode = False
        move_req.a, move_req.b, move_req.c, move_req.d, move_req.e, move_req.f = segment_target
        move_future, immediate_move_result = node.call_async_checked(
            node.movl_cli,
            move_req,
            accept_timeout_sec=PROBE_MOVL_ACCEPT_TIMEOUT_SEC,
        )
        if immediate_move_result is not None:
            print(f"{prefix}MovL accepted immediately")

    issue_next_segment()

    triggered = False
    trigger_pose = None
    reached_target = False
    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=PROBE_SPIN_TIMEOUT_SEC)
        if immediate_move_result is None:
            immediate_move_result = node.check_ready_future(node.movl_cli, move_future)
        if node.last_feed_time is None or time.monotonic() - node.last_feed_time > 0.2:
            node.stop_repeated()
            raise RuntimeError("FeedInfo became stale during probing; motion stopped")
        snapshot = node.feed_snapshot()
        if snapshot.get("di1"):
            triggered = True
            trigger_pose = snapshot["pose"] if snapshot.get("pose") is not None else None
            node.stop_fast_then_confirm()
            break
        if snapshot.get("pose") is not None:
            segment_tolerance = min(PROBE_SEGMENT_POSITION_TOLERANCE_MM, max(0.01, probe_step * 0.1))
            if abs(snapshot["pose"][2] - segment_target[2]) <= segment_tolerance:
                if issued_distance >= args.approach_mm - 1e-9:
                    reached_target = True
                    break
                boundary_trigger = wait_segment_boundary_trigger(
                    node,
                    deadline,
                    "FeedInfo became stale during probing; motion stopped",
                )
                if boundary_trigger is not None:
                    triggered = True
                    trigger_pose = boundary_trigger["pose"] if boundary_trigger.get("pose") is not None else None
                    node.stop_fast_then_confirm()
                    break
                issue_next_segment()

    row = None
    if triggered:
        contact_pose = node.get_pose()
        measurement_pose = trigger_pose if trigger_pose is not None else contact_pose
        tip_z = measurement_pose[2] - tool_length
        row = {
            "timestamp": f"{time.time():.3f}",
            "cycle": "" if cycle_index is None else str(cycle_index),
            "flange_x": f"{measurement_pose[0]:.4f}",
            "flange_y": f"{measurement_pose[1]:.4f}",
            "flange_z": f"{measurement_pose[2]:.4f}",
            "rx": f"{measurement_pose[3]:.4f}",
            "ry": f"{measurement_pose[4]:.4f}",
            "rz": f"{measurement_pose[5]:.4f}",
            "tip_x_est": f"{measurement_pose[0]:.4f}",
            "tip_y_est": f"{measurement_pose[1]:.4f}",
            "tip_z_est": f"{tip_z:.4f}",
            "tool_length_mm": f"{tool_length:.1f}",
            "probe_step_mm": f"{float(getattr(args, 'probe_step_mm', 0.0) or 0.0):.4f}",
            "stop_flange_x": f"{contact_pose[0]:.4f}",
            "stop_flange_y": f"{contact_pose[1]:.4f}",
            "stop_flange_z": f"{contact_pose[2]:.4f}",
        }
        print(f"{prefix}triggered at flange pose: {measurement_pose}")
        print(f"{prefix}stopped at flange pose: {contact_pose}")
        print(f"{prefix}estimated tip contact xyz: [{measurement_pose[0]:.4f}, {measurement_pose[1]:.4f}, {tip_z:.4f}]")
    else:
        node.stop_repeated()
        contact_pose = node.get_pose()
        reason = "target reached" if reached_target else "timeout"
        print(f"{prefix}no trigger before {reason}; final flange pose: {contact_pose}")

    retract_pose = contact_pose.copy()
    retract_pose[2] += args.retract_mm
    node.move_l(retract_pose, timeout_sec=10.0)
    node.wait_until_pose(retract_pose, timeout_sec=10.0)
    print(f"{prefix}retracted to z + {args.retract_mm:.1f} mm")
    node.wait_fresh_feed()
    probe_still_triggered = bool(node.read_di1() or node.di1)
    if row is not None:
        write_contact(Path(args.output), row)
        print(f"{prefix}saved: {args.output}")
    if probe_still_triggered:
        raise RuntimeError("DI1 remains triggered after retract; leave motion stopped and inspect the probe")
    return row


def main():
    """
    命令行入口：解析参数、初始化 ROS2、执行一次垂直探测循环。

    命令行参数说明：
      --execute         实际执行机器人运动（缺省为 dry-run）
      --approach-mm     垂直探测进给距离（默认 3.0 mm）
      --retract-mm      触发后回退距离（默认 3.0 mm）
      --speed           探测速度倍率 1~5（默认 1）
      --flange-mm       法兰到测头分支原点距离（默认 125.4 mm）
      --stylus-mm       测杆球心距（默认 75.0 mm）
      --probe-step-mm   分段探测步长，0 为不分段（默认 0.5 mm）
      --timeout         探测超时秒数（默认 20.0 s）
      --output          CSV 输出路径
      --ack-vertical-stylus-ok  安全确认：垂直探针已安装且路径安全

    流程：
      1. 校验参数
      2. 初始化 ROS2 并创建 ProbeTouch 节点
      3. 等待所有服务就绪
      4. 执行探测循环
      5. 清理并关闭 ROS2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="actually move the robot")
    parser.add_argument("--approach-mm", type=float, default=3.0, help="vertical probing distance")
    parser.add_argument("--retract-mm", type=float, default=3.0, help="retract distance after trigger")
    parser.add_argument("--speed", type=int, default=1, help="SpeedFactor ratio during probing")
    parser.add_argument("--allow-long-approach", action="store_true", help="allow approach distance > 5 mm")
    parser.add_argument("--flange-mm", type=float, default=125.4, help="robot flange to cross-stylus branch origin")
    parser.add_argument("--stylus-mm", type=float, default=75.0, help="vertical stylus ball-centre distance")
    parser.add_argument("--timeout", type=float, default=20.0, help="probing timeout seconds")
    parser.add_argument(
        "--probe-step-mm",
        type=float,
        default=0.5,
        help="segment probing MovL into this step size; 0 disables segmented probing",
    )
    parser.add_argument("--output", default="data/probe_contacts.csv", help="CSV output path")
    add_vertical_probe_ack_argument(parser)
    args = parser.parse_args()
    try:
        validate_probe_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rclpy.init()
    node = ProbeTouch()
    try:
        node.wait_services(10.0)
        run_probe_cycle(node, args)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
