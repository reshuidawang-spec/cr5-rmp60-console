#!/usr/bin/env python3
"""
Qt5 GUI for CR5 + RMP60 touch probe calibration: 手动点动、位姿记录、半自动法向探测、DI1 触发采集.

本模块提供了一个完整的 PyQt5 图形界面，用于 Dobot CR5 机器人搭载 RMP60 测头时的
标定数据采集工作流。主要功能包括：

  - ROS2 连接管理：通过 ProbeTouch 节点连接 Dobot 驱动，实时接收 FeedInfo 反馈
  - 实时状态显示：法兰位姿 (x/y/z/rx/ry/rz)、关节角、DI1 数字输入、Feed 新鲜度
  - 坐标点动控制：沿选定轴 (X/Y/Z/RX/RY/RZ) 正负方向步进点动，支持步长和速度设置
  - 复合方向点动：在任意三维矢量方向上点动指定距离
  - 半自动法向短探测：基于球面拟合 JSON，计算法向接近向量并执行带 DI1 捕获的短探测
  - 位姿记录：手动记录当前位姿到 CSV，或通过 DI1 触发的接触事件自动记录
  - 自动回退：DI1 触发后沿接近路径反向回退，确保测头安全
  - CSv 输出：统一的字段格式，与 jog_and_record_contacts 模块兼容

依赖关系：
  - jog_and_record_contacts: 底层点动、位姿获取、接触记录函数
  - probe_touch: ROS2 服务/话题封装，DI1 监控与运动指令
  - semi_auto_normal_probe: 球面拟合加载与法向接近规划

CSV 记录格式：
  capture_pose 命令行记录：
    timestamp, sample_index, session_id, workpiece_id, artifact_id, artifact_type,
    physical_ball_id, branch, operator_note, command, linear_step_mm, angular_step_deg,
    speed, auto_retract_mm, trigger_feed_sequence, trigger_feed_wall_time,
    trigger_digital_input_bits, trigger_di1,
    start_flange_x/y/z/rx/ry/rz, target_flange_*, trigger_flange_*, stop_flange_*,
    trigger_joint_1..6

  接触触发记录（由 jog_and_record_contacts.make_contact_row 生成）：
    包含上述字段外加 retract_flange_x/y/z 等回退路径信息
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import rclpy
from PyQt5.QtCore import QEvent, QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from jog_and_record_contacts import (
    AXIS_INDEX,
    format_pose,
    issue_movl,
    jog_target,
    make_contact_row,
    current_pose,
    pose_reached,
    retract_opposite,
    vector_jog_target,
    wait_jog,
)
from probe_touch import PROBE_SPIN_TIMEOUT_SEC, ProbeTouch
from held_jog import HeldJogRequest
from contact_record_store import (
    delete_contact_records, read_contact_records, resolve_contact_output, save_contact_records,
)
from semi_auto_normal_probe import compute_plan, load_fit


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data/gui_probe_contacts.csv"
DEFAULT_NORMAL_FIT = PROJECT_DIR / "data/2026.6.10/yneg_distance_only_coarse_fit_20260610.json"
POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")


def pose_field_map(prefix, pose):
    """
    将位姿向量 (6-element list) 转换为带前缀的字典字段。

    Args:
        prefix: 字段名前缀，如 "start_flange", "trigger_flange"
        pose: 长度为 6 的位姿列表 [x, y, z, rx, ry, rz]，或 None

    Returns:
        dict: { "prefix_x": "...", "prefix_y": "...", ... }，
               值为格式化为 4 位小数的字符串，位姿为 None 时为空字符串
    """
    return {
        f"{prefix}_{name}": "" if pose is None else f"{float(pose[index]):.4f}"
        for index, name in enumerate(POSE_NAMES)
    }


def vector_field_map(prefix, values, count, digits=6):
    """
    将向量转换为带前缀的字典字段，用于记录关节角等数组数据。

    Args:
        prefix: 字段名前缀，如 "trigger_joint"
        values: 向量值列表或 None
        count: 期望的元素个数
        digits: 小数位数，默认 6

    Returns:
        dict: { "prefix_1": "...", "prefix_2": "...", ... }
    """
    return {
        f"{prefix}_{index + 1}": (
            "" if values is None or index >= len(values) else f"{float(values[index]):.{digits}f}"
        )
        for index in range(count)
    }


def write_stable_row(path, row):
    """
    稳定地写入一行 CSV 记录。如果现有 CSV 的列名与新记录不匹配，
    则自动将原文件重命名为带时间戳的备份，再重新创建新文件。

    这样设计是为了防止因字段变更导致数据错列。

    Args:
        path: CSV 文件路径
        row: dict，列名到值的映射
    """
    path = resolve_contact_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open(encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), None)
        if header != list(row.keys()):
            backup = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
            path.replace(backup)
            exists = False
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def existing_max_sample_index(path):
    """
    读取已有 CSV 文件，返回最大的 sample_index 值，用于新记录接续编号。

    支持 "sample_index" 和 "gui_sample_index" 两种字段名。

    Args:
        path: CSV 文件路径

    Returns:
        int: 当前最大样本编号，文件不存在时返回 0
    """
    path = resolve_contact_output(path)
    if not path.exists():
        return 0
    max_index = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            value = row.get("sample_index") or row.get("gui_sample_index") or ""
            try:
                max_index = max(max_index, int(float(value)))
            except (TypeError, ValueError):
                continue
    return max_index


def make_capture_row(settings, sample_index, snapshot, command="capture_pose"):
    """
    构建「位姿记录」命令行的 CSV 数据行。

    将当前快照中的位姿、DI1 状态、关节角等数据与 UI 设置合并为一条记录。
    本函数处理的是手动"记录当前位姿"场景，与 make_contact_row（接触触发）不同。

    Args:
        settings: dict，包含所有 UI 参数（session_id, workpiece_id 等）
        sample_index: int，当前样本编号
        snapshot: dict，从 ProbeTouch.feed_snapshot() 获得的快照，
                  包含 pose, sequence, wall_time, digital_input_bits, di1, joints 等字段
        command: 命令名称字符串，默认为 "capture_pose"

    Returns:
        dict: CSV 行数据
    """
    pose = snapshot.get("pose")
    row = {
        "timestamp": f"{time.time():.3f}",
        "sample_index": str(sample_index),
        "session_id": settings["session_id"],
        "workpiece_id": settings["workpiece_id"],
        "artifact_id": settings["artifact_id"],
        "artifact_type": settings["artifact_type"],
        "physical_ball_id": settings["physical_ball_id"],
        "branch": settings["branch"],
        "operator_note": settings["operator_note"],
        "command": command,
        "linear_step_mm": f"{settings['linear_step_mm']:.4f}",
        "angular_step_deg": f"{settings['angular_step_deg']:.4f}",
        "speed": str(settings["speed"]),
        "auto_retract_mm": f"{settings['auto_retract_mm']:.4f}",
        "trigger_feed_sequence": str(snapshot.get("sequence", "")),
        "trigger_feed_wall_time": (
            "" if snapshot.get("wall_time") is None else f"{float(snapshot['wall_time']):.6f}"
        ),
        "trigger_digital_input_bits": (
            "" if snapshot.get("digital_input_bits") is None else str(snapshot["digital_input_bits"])
        ),
        "trigger_di1": "" if snapshot.get("di1") is None else str(int(bool(snapshot["di1"]))),
    }
    # 为了与接触记录保持相同的列结构，将当前位姿同时填入
    # start/target/trigger/stop 四组字段（本场景下这四个位置相同）
    row.update(pose_field_map("start_flange", pose))
    row.update(pose_field_map("target_flange", pose))
    row.update(pose_field_map("trigger_flange", pose))
    row.update(pose_field_map("stop_flange", pose))
    row.update(vector_field_map("trigger_joint", snapshot.get("joints"), 6))
    return row


class RosProbeWorker(QObject):
    """
    工作线程对象：封装所有 ROS2 操作，通过 Qt 信号/槽与 GUI 线程通信。

    本对象的生命周期在单独的 QThread 中运行，所有耗时操作（点动、等待 DI1、
    文件写入等）均不阻塞 GUI。通过以下信号向 GUI 报告结果：

    Signals:
        connected(bool):     连接成功/失败
        status(str, str):    日志消息 (level, message)
        poseUpdated(list, bool, object, float): 位姿更新
                         (pose, di1, joints, feed_age)
        sampleRecorded(dict): 新记录已写入 CSV
        busyChanged(bool):    繁忙状态变化，GUI 据此启用/禁用按钮

    工作流程：
        1. GUI 通过信号 connectRequested 触发 connect_robot()
        2. 成功后 refresh_timer 每 800ms 自动触发 refresh()
        3. 用户通过按钮触发点动/采集/探测等操作
        4. 每个操作完成后自动 refresh() 更新显示
    """

    # ---------- 信号定义 ----------
    # 这些信号由 GUI 线程发射，由本对象（在 worker_thread 中）的槽函数处理
    connected = pyqtSignal(bool)
    status = pyqtSignal(str, str)
    poseUpdated = pyqtSignal(list, bool, object, float)
    sampleRecorded = pyqtSignal(dict)
    recordsLoaded = pyqtSignal(str, list)
    recordsSaved = pyqtSignal(str)
    busyChanged = pyqtSignal(bool)
    continuousJogFinished = pyqtSignal(object)
    shutdownFinished = pyqtSignal()

    def __init__(self):
        """
        初始化工作线程，设置初始状态。

        - node: ROS2 ProbeTouch 节点实例，连接后创建
        - sample_index: 从已有 CSV 恢复的样本编号
        - busy: 操作进行标志，防止并发操作
        """
        super().__init__()
        self.node = None
        self.sample_index = 0
        self.busy = False
        # 记录上一次点动的起点和目标点，用于自动回退路径计算
        self.last_start_pose = None
        self.last_target_pose = None
        self.last_command = ""

    def set_busy(self, value):
        """
        设置繁忙状态并发射 busyChanged 信号。

        Args:
            value: True 表示操作开始，False 表示操作结束
        """
        if self.busy != value:
            self.busy = value
            self.busyChanged.emit(value)

    def ensure_node(self):
        """
        确保 ROS2 节点已创建，否则抛出 RuntimeError。

        Returns:
            ProbeTouch 节点实例
        """
        if self.node is None:
            raise RuntimeError("尚未连接 ROS2 Dobot 驱动")
        return self.node

    def notify_sample_recorded(self, output, row):
        self.sampleRecorded.emit(row)
        self.recordsLoaded.emit(str(Path(output).resolve()), read_contact_records(output))

    @pyqtSlot(str)
    def load_records(self, output):
        if self.busy:
            return
        self.set_busy(True)
        try:
            self.recordsLoaded.emit(str(Path(output).resolve()), read_contact_records(output))
        except Exception as exc:
            self.status.emit("ERROR", f"读取采集记录失败: {exc}")
        finally:
            self.set_busy(False)

    @pyqtSlot(str, list)
    def delete_records(self, output, selected):
        if self.busy:
            return
        self.set_busy(True)
        try:
            remaining, backup = delete_contact_records(output, selected)
            self.recordsLoaded.emit(str(Path(output).resolve()), remaining)
            self.status.emit("SUCCESS", f"已删除 {len(selected)} 条采集记录，原文件备份: {backup}")
        except Exception as exc:
            self.status.emit("ERROR", f"删除采集记录失败: {exc}")
        finally:
            self.set_busy(False)

    @pyqtSlot(str, str)
    def save_records(self, source, destination):
        if self.busy:
            return
        self.set_busy(True)
        try:
            path, rows, backup = save_contact_records(source, destination)
            self.sample_index = max(self.sample_index, existing_max_sample_index(path))
            self.recordsSaved.emit(str(path))
            self.recordsLoaded.emit(str(path), rows)
            self.status.emit("SUCCESS", f"已手动保存 {len(rows)} 条采集记录: {path}")
            if backup is not None:
                self.status.emit("INFO", f"目标文件原有内容已备份: {backup}")
        except Exception as exc:
            self.status.emit("ERROR", f"手动保存失败: {exc}")
        finally:
            self.set_busy(False)

    @staticmethod
    def fresh_feed_di1(node, max_age_sec=0.2):
        """
        获取最新的 DI1 状态，并进行新鲜度检查。

        FeedInfo 是机器人周期性发布的状态消息，如果上一次消息时间超过
        max_age_sec，说明机器人可能已停止通信（如急停或断线），此时不应
        做出运动决策。

        Args:
            node: ProbeTouch 节点实例
            max_age_sec: 允许的最大 FeedInfo 滞后时间（秒）

        Returns:
            bool: DI1 当前状态（True = 触发）
        """
        if node.last_feed_time is None:
            raise RuntimeError("尚未收到 FeedInfo，无法判断 DI1")
        age = time.monotonic() - node.last_feed_time
        if age > max_age_sec:
            raise RuntimeError(f"FeedInfo 已过期 {age:.3f}s，拒绝判断 DI1")
        return bool(node.di1)

    # ---------- 槽函数：由 GUI 信号的连接触发 ----------

    @pyqtSlot()
    def connect_robot(self):
        """
        连接到 ROS2 Dobot 驱动。

        初始化 rclpy（如未初始化），创建 ProbeTouch 节点，等待服务就绪
        和首次 FeedInfo。连接成功后在状态栏显示"已连接"。
        """
        if self.busy:
            return
        self.set_busy(True)
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            if self.node is None:
                self.node = ProbeTouch()
            self.node.wait_services(10.0)
            self.node.wait_fresh_feed()
            self.status.emit("SUCCESS", "已连接 ROS2 服务并收到 FeedInfo")
            self.connected.emit(True)
            self.refresh()
        except Exception as exc:
            self.status.emit("ERROR", str(exc))
            self.connected.emit(False)
        finally:
            self.set_busy(False)

    @pyqtSlot()
    def shutdown(self):
        """
        关机：销毁 ROS2 节点并关闭 rclpy。

        由窗口 closeEvent 触发，确保程序退出时 ROS2 资源被正确释放。
        """
        try:
            if self.node is not None:
                self.node.destroy_node()
                self.node = None
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:
            self.status.emit("WARN", f"关闭 ROS 节点时出错: {exc}")
        finally:
            self.shutdownFinished.emit()

    @pyqtSlot()
    def refresh(self):
        """
        刷新最新位姿和 DI1 状态。

        通过 rclpy.spin_once 处理一次 ROS2 回调，然后从节点中读取
        最新的 FeedInfo 快照，获取当前法兰位姿、DI1 信号、关节角和
        Feed 新鲜度。结果通过 poseUpdated 信号发送到 GUI。
        """
        if self.busy or self.node is None:
            return
        try:
            node = self.ensure_node()
            rclpy.spin_once(node, timeout_sec=PROBE_SPIN_TIMEOUT_SEC)
            snapshot = node.feed_snapshot()
            pose = snapshot.get("pose") or current_pose(node, max_age_sec=0.5)
            di1 = bool(snapshot.get("di1"))
            feed_age = 999.0
            if node.last_feed_time is not None:
                feed_age = time.monotonic() - node.last_feed_time
            self.poseUpdated.emit(list(pose), di1, snapshot.get("joints"), feed_age)
        except Exception as exc:
            self.status.emit("WARN", f"刷新失败: {exc}")

    @pyqtSlot()
    def stop(self):
        """
        紧急停止：发送 stop_fast_then_confirm 指令。

        急停按钮对应的操作，通过 DT 指令让机器人立即制动。
        """
        if self.node is None:
            self.status.emit("WARN", "尚未连接，无法 Stop")
            return
        self.set_busy(True)
        try:
            self.node.stop_fast_then_confirm()
            self.status.emit("WARN", "已发送 Stop")
        except Exception as exc:
            self.status.emit("ERROR", f"Stop 失败: {exc}")
        finally:
            self.set_busy(False)

    @pyqtSlot(str, float, dict, object)
    def run_held_jog(self, axis, sign, settings, request):
        """一次按住只发送一条有界 MovL，取消由运动循环直接处理。"""
        try:
            if self.busy or request.stop_requested():
                return
            settings = dict(settings)
            for name, maximum in (("continuous_linear_limit_mm", 10.0), ("continuous_angular_limit_deg", 20.0)):
                value = float(settings[name])
                if not math.isfinite(value) or not 0.1 <= value <= maximum:
                    raise ValueError(f"持续点动行程超出范围: {name}")
                settings[name] = value
            settings["linear_step_mm"] = settings["continuous_linear_limit_mm"]
            settings["angular_step_deg"] = settings["continuous_angular_limit_deg"]
            command = f"held {axis}{'+' if sign > 0 else '-'}"
            self._execute_jog(
                settings, command,
                lambda pose: jog_target(pose, axis, sign, settings),
                hold_request=request,
            )
        except Exception as exc:
            self.status.emit("ERROR", f"持续点动启动失败: {exc}")
        finally:
            request.stop()
            self.continuousJogFinished.emit(request)

    @pyqtSlot(dict)
    def retract_last(self, settings):
        """
        沿上一次点动路径反向回退。

        当 DI1 触发后，机器人停在接触点附近。此函数计算从 target 回到
        start 的相反方向，沿该方向移动 auto_retract_mm 距离，确保测头
        脱离接触。

        仅当上一次运动存在明确的平移方向时有效（纯旋转无平移时无法计算回退方向）。

        Args:
            settings: dict，包含 auto_retract_mm, timeout_sec 等参数
        """
        if self.busy:
            return
        self.set_busy(True)
        try:
            node = self.ensure_node()
            if self.last_start_pose is None or self.last_target_pose is None:
                self.status.emit("WARN", "没有可用的上一次点动路径，无法自动回退")
                return
            node.stop_fast_then_confirm()
            retract_pose = retract_opposite(
                node,
                self.last_start_pose,
                self.last_target_pose,
                settings["auto_retract_mm"],
                settings["timeout_sec"],
            )
            if retract_pose is None:
                self.status.emit("WARN", "上一次点动没有平移方向，无法按路径回退")
                return
            self.status.emit(
                "SUCCESS",
                f"已按上次路径反向回退 {settings['auto_retract_mm']:.3f}mm: {self.last_command}",
            )
            self.refresh()
        except Exception as exc:
            self.status.emit("ERROR", f"按上次路径回退失败: {exc}")
        finally:
            self.set_busy(False)

    @pyqtSlot(dict)
    def capture_pose(self, settings):
        """
        记录当前位姿到 CSV（手动非接触记录）。

        从 FeedInfo 快照获取当前位姿和 DI1 状态，构建 CSV 行并写入文件。
        与接触触发记录不同，此操作为用户主动点击"记录当前位姿"按钮触发，
        用于在安全位置记录参考位姿。

        Args:
            settings: dict，所有 UI 参数
        """
        if self.busy:
            return
        self.set_busy(True)
        try:
            node = self.ensure_node()
            node.wait_fresh_feed()
            rclpy.spin_once(node, timeout_sec=PROBE_SPIN_TIMEOUT_SEC)
            snapshot = node.feed_snapshot()
            if snapshot.get("pose") is None:
                snapshot["pose"] = current_pose(node, max_age_sec=0.5)
            if snapshot.get("di1") is None:
                snapshot["di1"] = self.fresh_feed_di1(node, max_age_sec=0.5)
            self.sample_index += 1
            row = make_capture_row(settings, self.sample_index, snapshot)
            write_stable_row(settings["output"], row)
            self.notify_sample_recorded(settings["output"], row)
            self.status.emit("SUCCESS", f"已记录当前位姿: sample {self.sample_index}")
        except Exception as exc:
            self.status.emit("ERROR", f"记录当前位姿失败: {exc}")
        finally:
            self.set_busy(False)

    @pyqtSlot(str, float, dict)
    def jog_axis(self, axis, sign, settings):
        """
        沿指定坐标轴方向点动。

        调用 jog_target 生成目标位姿（原位上增加步长），然后执行移动
        并监控 DI1 触发。

        Args:
            axis: 轴名称，"x" / "y" / "z" / "rx" / "ry" / "rz"
            sign: 方向，1.0 为正方向，-1.0 为负方向
            settings: dict，包含步长、速度等参数
        """
        self._execute_jog(settings, f"{axis}{'+' if sign > 0 else '-'}", lambda pose: jog_target(pose, axis, sign, settings))

    @pyqtSlot(float, float, float, float, dict)
    def jog_vector(self, dx, dy, dz, distance_mm, settings):
        """
        沿指定矢量方向点动指定距离。

        用于自定义接近方向，如沿 X-/Y+ 复合方向接近球体。

        Args:
            dx: 矢量 x 分量
            dy: 矢量 y 分量
            dz: 矢量 z 分量
            distance_mm: 移动距离（毫米）
            settings: dict，包含速度等参数
        """
        command = f"vec {dx:g} {dy:g} {dz:g} {distance_mm:g}"
        self._execute_jog(settings, command, lambda pose: vector_jog_target(pose, [dx, dy, dz], distance_mm))

    def _normal_plan_from_settings(self, node, settings):
        """
        从设置加载球面拟合，计算法向接近规划。

        加载预拟合的 JSON 文件，获取球心坐标、局部偏移量和分支信息，
        然后结合当前机器人位姿计算法向接近向量和目标位姿。

        Args:
            node: ProbeTouch 节点
            settings: dict，包含 normal_fit_json, normal_search_mm 等

        Returns:
            dict: compute_plan 的返回结果，包含 approach 向量、target_pose 等

        Raises:
            RuntimeError: 分支不匹配或搜索距离超出范围
        """
        search_mm = float(settings["normal_search_mm"])
        if search_mm <= 0.0 or search_mm > 1.0:
            raise RuntimeError("法向搜索距离必须在 0..1.0mm 内")
        fit, sphere_center, local_offset = load_fit(settings["normal_fit_json"])
        fit_branch = fit.get("branch")
        if fit_branch and fit_branch != settings["branch"]:
            raise RuntimeError(f"拟合文件分支 {fit_branch!r} 与当前分支 {settings['branch']!r} 不一致")
        node.wait_fresh_feed()
        pose = current_pose(node, max_age_sec=0.5)
        plan = compute_plan(
            pose,
            sphere_center,
            local_offset,
            settings["normal_euler_sequence"],
            search_mm,
        )
        plan["initial_fit"] = fit
        return plan

    @pyqtSlot(dict)
    def dry_run_normal_probe(self, settings):
        """
        法向探测的干运行：计算法向方向和目标位姿但实际不移动。

        用于在 GUI 上显示计算结果，让操作员确认接近方向和距离是否安全，
        然后再勾选"已确认路径安全"执行实际运动。

        Args:
            settings: dict，包含法向探测相关参数
        """
        if self.busy:
            return
        self.set_busy(True)
        try:
            node = self.ensure_node()
            plan = self._normal_plan_from_settings(node, settings)
            approach = plan["approach"]
            target = plan["target_pose"]
            self.status.emit(
                "INFO",
                "法向 dry-run: "
                f"approach=[{approach[0]:.6f}, {approach[1]:.6f}, {approach[2]:.6f}], "
                f"距离={plan['center_distance_mm']:.4f}mm, "
                f"target={format_pose(target)}",
            )
        except Exception as exc:
            self.status.emit("ERROR", f"法向 dry-run 失败: {exc}")
        finally:
            self.set_busy(False)

    @pyqtSlot(dict)
    def jog_normal_vector(self, settings):
        """
        沿法向接近方向点动指定距离（不等待 DI1）。

        与 execute_normal_probe 不同，此操作只沿法向方向移动但不进入
        探测模式，用于微调位置或确认路径。

        Args:
            settings: dict，包含法向探测参数
        """
        if self.busy:
            return
        try:
            if not settings["normal_ack_safe"]:
                self.status.emit("WARN", "请先勾选已确认路径安全")
                return
            node = self.ensure_node()
            plan = self._normal_plan_from_settings(node, settings)
            approach = plan["approach"]
            distance_mm = float(settings["normal_search_mm"])
            command = f"normal_jog {approach[0]:.6f} {approach[1]:.6f} {approach[2]:.6f} {distance_mm:.4f}"
            self._execute_jog(
                settings,
                command,
                lambda pose, approach=approach, distance_mm=distance_mm: vector_jog_target(
                    pose,
                    approach,
                    distance_mm,
                ),
            )
        except Exception as exc:
            self.status.emit("ERROR", f"法向复合点动失败: {exc}")

    @pyqtSlot(dict)
    def execute_normal_probe(self, settings):
        """
        执行完整的法向短探测：沿法向接近 -> 等待 DI1 触发 -> 记录接触 -> 回退。

        这是半自动法向探测的核心流程：
        1. 从球面拟合计算法向接近方向
        2. 检查 DI1 未触发（确保安全起始状态）
        3. 沿法向方向移动指定距离
        4. 监控移动过程中 DI1 是否触发（接触检测）
        5. 记录触发时刻的位姿快照到 CSV
        6. 沿接近路径反向回退

        Args:
            settings: dict，包含所有法向探测参数
        """
        if self.busy:
            return
        self.set_busy(True)
        try:
            if not settings["normal_ack_safe"]:
                self.status.emit("WARN", "请先勾选已确认法向短探测路径安全")
                return
            node = self.ensure_node()
            plan = self._normal_plan_from_settings(node, settings)
            if self.fresh_feed_di1(node):
                self.status.emit("WARN", "DI1 已经触发，请先退开测针")
                return
            start_pose = plan["flange_pose"]
            target_pose = plan["target_pose"]
            self.last_start_pose = list(start_pose)
            self.last_target_pose = list(target_pose)
            approach = plan["approach"]
            command = (
                f"normal {approach[0]:.6f} {approach[1]:.6f} {approach[2]:.6f} "
                f"{settings['normal_search_mm']:.4f}"
            )
            self.last_command = command
            node.set_speed(settings["speed"])
            future = issue_movl(node, target_pose)
            # wait_jog 在 moveL 执行期间持续监控 DI1，返回触发时刻的快照
            trigger_snapshot, reached_target = wait_jog(
                node,
                target_pose,
                settings["timeout_sec"],
                settings["position_tolerance_mm"],
                settings["orientation_tolerance_deg"],
            )
            if future.done():
                node.check_ready_future(node.movl_cli, future)
            if trigger_snapshot is None and reached_target:
                self.status.emit("WARN", f"{command}: 到达目标，无 DI1 触发")
                retract_opposite(node, start_pose, target_pose, settings["auto_retract_mm"], settings["timeout_sec"])
                self.refresh()
                return
            # DI1 触发：记录触发位姿并回退
            stop_pose = current_pose(node, max_age_sec=0.5)
            self.sample_index += 1
            settings["sample_index"] = self.sample_index
            args = SimpleNamespace(
                session_id=settings["session_id"],
                workpiece_id=settings["workpiece_id"],
                artifact_id=settings["artifact_id"],
                artifact_type=settings["artifact_type"],
            )
            row = make_contact_row(args, settings, command, start_pose, target_pose, trigger_snapshot, stop_pose)
            write_stable_row(settings["output"], row)
            retract_opposite(node, start_pose, target_pose, settings["auto_retract_mm"], settings["timeout_sec"])
            self.notify_sample_recorded(settings["output"], row)
            self.status.emit("SUCCESS", f"{command}: DI1 触发，已记录 sample {self.sample_index}")
            self.refresh()
        except Exception as exc:
            self.status.emit("ERROR", f"法向短探测执行失败: {exc}")
        finally:
            self.set_busy(False)

    def _execute_jog(self, settings, command, target_from_start, hold_request=None):
        """
        执行点动操作并监控 DI1 触发的通用方法。

        核心流程：
        1. 检查 DI1 是否已触发（已触发则拒绝点动，避免碰撞）
        2. 通过 target_from_start 回调生成目标位姿
        3. 设置速度，发送 moveL 指令
        4. 在 wait_jog 中循环监控：位置逼近 + DI1 状态
        5. 如 DI1 触发：记录接触行（含触发快照）到 CSV，自动回退
        6. 如到达目标但 DI1 未触发：仅刷新显示（无接触记录）

        Args:
            settings: dict，所有 UI 参数
            command: str，命令描述字符串
            target_from_start: callable，接收当前位姿返回目标位姿的函数
        """
        if self.busy:
            return
        self.set_busy(True)
        motion_attempted = False
        try:
            node = self.ensure_node()
            node.wait_fresh_feed()
            if hold_request is not None and hold_request.stop_requested():
                return
            if self.fresh_feed_di1(node):
                self.status.emit("WARN", 'DI1 已经触发，请先退开测针；如确认安全，可点"按上次路径回退"')
                return
            start_pose = current_pose(node)
            target_pose = target_from_start(start_pose)
            self.last_start_pose = list(start_pose)
            self.last_target_pose = list(target_pose)
            self.last_command = command
            node.set_speed(settings["speed"])
            # 松开可能发生在等待反馈/设置速度期间，发运动指令前必须再次检查。
            if hold_request is not None and hold_request.stop_requested():
                return
            motion_attempted = True
            future = issue_movl(node, target_pose)
            last_display_time = 0.0

            def update_display(snapshot):
                nonlocal last_display_time
                now = time.monotonic()
                pose = snapshot.get("pose")
                if pose is not None and now - last_display_time >= 0.1:
                    last_display_time = now
                    self.poseUpdated.emit(
                        list(pose), bool(snapshot.get("di1")), snapshot.get("joints"),
                        now - node.last_feed_time,
                    )

            trigger_snapshot, reached_target = wait_jog(
                node,
                target_pose,
                settings["timeout_sec"],
                settings["position_tolerance_mm"],
                settings["orientation_tolerance_deg"],
                command_future=future,
                stop_requested=hold_request.stop_requested if hold_request is not None else None,
                progress_callback=update_display if hold_request is not None else None,
            )
            if future.done() and (hold_request is None or reached_target):
                node.check_ready_future(node.movl_cli, future)
            if trigger_snapshot is None:
                if hold_request is not None:
                    reason = "到达单次行程上限，请松开后重新按住" if reached_target else "已响应松开或保护停止"
                else:
                    reason = "到达目标，无 DI1 触发"
                self.status.emit("INFO", f"{command}: {reason}")
                self.refresh()
                return
            # DI1 触发：记录接触 CSV 行，含触发快照中的位姿、DI1 状态、关节角等
            stop_pose = current_pose(node, max_age_sec=0.5)
            self.sample_index += 1
            settings["sample_index"] = self.sample_index
            args = SimpleNamespace(
                session_id=settings["session_id"],
                workpiece_id=settings["workpiece_id"],
                artifact_id=settings["artifact_id"],
                artifact_type=settings["artifact_type"],
            )
            row = make_contact_row(args, settings, command, start_pose, target_pose, trigger_snapshot, stop_pose)
            write_stable_row(settings["output"], row)
            retract_pose = None
            if hold_request is None or not hold_request.emergency:
                retract_pose = retract_opposite(
                    node, start_pose, target_pose,
                    settings["auto_retract_mm"], settings["timeout_sec"],
                    stop_requested=(lambda: hold_request.emergency) if hold_request is not None else None,
                )
            if retract_pose is not None:
                row["retract_flange_x"] = f"{retract_pose[0]:.4f}"
                row["retract_flange_y"] = f"{retract_pose[1]:.4f}"
                row["retract_flange_z"] = f"{retract_pose[2]:.4f}"
            self.notify_sample_recorded(settings["output"], row)
            self.status.emit("SUCCESS", f"{command}: DI1 触发，已记录 sample {self.sample_index}")
            self.refresh()
        except Exception as exc:
            if hold_request is not None and motion_attempted:
                try:
                    node.stop_fast_then_confirm()
                except Exception as stop_exc:
                    self.status.emit("ERROR", f"持续点动 Stop 失败: {stop_exc}")
            self.status.emit("ERROR", f"{command} 执行失败: {exc}")
        finally:
            self.set_busy(False)


class ProbeCalibrationWindow(QMainWindow):
    """
    主窗口类：构建 CR5 RMP60 标定采集控制台的完整 GUI。

    界面布局为三栏式：
    - 左栏（权重 2）：连接控制、实时位姿显示、数据元信息编辑
    - 中栏（权重 2）：坐标点动、复合方向点动、半自动法向探测、执行约束
    - 右栏（权重 3）：采集记录表格、状态日志

    线程模型：
    - 所有 ROS2 操作在 RosProbeWorker 中执行，运行于独立的 QThread
    - GUI 线程通过信号/槽与工作线程通信
    - 800ms 定时器自动刷新位姿显示

    DI1 监控集成：
    - 每次 refresh() 和点动前都检查 DI1 状态
    - 使用 BadgeWarn 样式（橙色）高亮触发状态
    - DI1 已触发时拒绝点动请求，防止意外碰撞

    信号定义（发射到工作线程）：
    - connectRequested / refreshRequested / stopRequested
    - retractLastRequested / captureRequested
    - jogAxisRequested / jogVectorRequested
    - normalDryRunRequested / normalJogRequested / normalProbeRequested
    - shutdownRequested
    """

    # ---------- 信号：到工作线程的请求 ----------
    connectRequested = pyqtSignal()
    refreshRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    retractLastRequested = pyqtSignal(dict)
    captureRequested = pyqtSignal(dict)
    loadRecordsRequested = pyqtSignal(str)
    deleteRecordsRequested = pyqtSignal(str, list)
    saveRecordsRequested = pyqtSignal(str, str)
    jogAxisRequested = pyqtSignal(str, float, dict)
    jogVectorRequested = pyqtSignal(float, float, float, float, dict)
    normalDryRunRequested = pyqtSignal(dict)
    normalJogRequested = pyqtSignal(dict)
    normalProbeRequested = pyqtSignal(dict)
    continuousJogStartRequested = pyqtSignal(str, float, dict, object)
    shutdownRequested = pyqtSignal()

    def __init__(self, args):
        """
        初始化主窗口。

        设置工作线程和所有信号/槽连接，构建 GUI，启动刷新定时器。

        Args:
            args: argparse 命名空间，包含所有命令行参数默认值
        """
        super().__init__()
        self.args = args
        self.args.output = str(resolve_contact_output(self.args.output))

        # ---------- 创建工作线程 ----------
        # worker 对象运行在独立的 worker_thread 中，
        # 通过 moveToThread 将对象归属到该线程
        self.worker_thread = QThread(self)
        self.worker = RosProbeWorker()

        # 从已有 CSV 恢复样本编号，实现断点续采
        self.worker.sample_index = existing_max_sample_index(self.args.output)
        self.sample_index_output = str(self.args.output)

        self.worker.moveToThread(self.worker_thread)

        # ---------- 信号/槽连接 ----------
        # GUI -> Worker 的请求连接（使用信号而非直接调用是为了线程安全）
        self.shutdownRequested.connect(self.worker.shutdown)
        self.worker.shutdownFinished.connect(self.worker_thread.quit, Qt.DirectConnection)
        self.connectRequested.connect(self.worker.connect_robot)
        self.refreshRequested.connect(self.worker.refresh)
        self.stopRequested.connect(self.worker.stop)
        self.retractLastRequested.connect(self.worker.retract_last)
        self.captureRequested.connect(self.worker.capture_pose)
        self.loadRecordsRequested.connect(self.worker.load_records)
        self.deleteRecordsRequested.connect(self.worker.delete_records)
        self.saveRecordsRequested.connect(self.worker.save_records)
        self.jogAxisRequested.connect(self.worker.jog_axis)
        self.jogVectorRequested.connect(self.worker.jog_vector)
        self.normalDryRunRequested.connect(self.worker.dry_run_normal_probe)
        self.normalJogRequested.connect(self.worker.jog_normal_vector)
        self.normalProbeRequested.connect(self.worker.execute_normal_probe)
        self.continuousJogStartRequested.connect(self.worker.run_held_jog)

        # Worker -> GUI 的回调连接
        self.worker.connected.connect(self.on_connected)
        self.worker.status.connect(self.append_log)
        self.worker.poseUpdated.connect(self.on_pose_updated)
        self.worker.recordsLoaded.connect(self.on_records_loaded)
        self.worker.recordsSaved.connect(self.on_records_saved)
        self.worker.busyChanged.connect(self.on_busy_changed)
        self.worker.continuousJogFinished.connect(self.on_continuous_jog_finished)

        self.worker_thread.start()

        self.pose_labels = {}
        self.joint_labels = []
        self.motion_buttons = []
        self.normal_buttons = []
        self.continuous_buttons = []
        self.gui_busy = False
        self.continuous_hold = None
        self.continuous_request = None
        self.continuous_button = None
        self.records_path = None
        self.connected = False
        self.setup_ui()
        self.apply_style()

        self.continuous_heartbeat_timer = QTimer(self)
        self.continuous_heartbeat_timer.setInterval(50)
        self.continuous_heartbeat_timer.timeout.connect(self.keep_continuous_jog_alive)

        # ---------- 自动刷新定时器 ----------
        # 每 800ms 请求工作线程更新一次位姿和 DI1 状态
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(800)
        self.refresh_timer.timeout.connect(self.refreshRequested.emit)
        self.refresh_timer.start()
        QTimer.singleShot(0, self.reload_records)
        if self.args.auto_connect:
            QTimer.singleShot(300, self.connectRequested.emit)

    def closeEvent(self, event):
        """
        窗口关闭事件：停止定时器、发送关闭信号、等待工作线程安全退出。

        先取消运动，再等 ROS 节点清理完成，避免销毁尚在运行的工作线程。

        Args:
            event: QCloseEvent
        """
        self.refresh_timer.stop()
        self.stop_continuous_step_jog(emergency=True)
        if self.worker_thread.isRunning():
            self.shutdownRequested.emit()
            if not self.worker_thread.wait(2000):
                event.ignore()
                QTimer.singleShot(250, self.close)
                return
        super().closeEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.WindowDeactivate and hasattr(self, "continuous_request"):
            self.stop_continuous_step_jog()
        super().changeEvent(event)

    def hideEvent(self, event):
        if hasattr(self, "continuous_request"):
            self.stop_continuous_step_jog()
        super().hideEvent(event)

    # ---------- GUI 构建方法 ----------

    def setup_ui(self):
        """
        构建完整的 UI 布局。

        使用 QScrollArea 包裹主内容以支持小屏幕滚动显示。
        整体结构：
        ┌──────────────────────────────────────────────────────┐
        │  标题栏 (CR5 RMP60 标定采集控制台)    Feed | DI1 | 连接 │
        ├──────────┬──────────────────┬────────────────────────┤
        │ 左栏(2)  │    中栏(2)       │    右栏(3)             │
        │ 连接采集  │  坐标点动        │  采集记录表            │
        │ 实时位姿  │  复合方向点动    │  状态日志              │
        │ 数据元信息│  半自动法向      │                        │
        │          │  执行约束        │                        │
        └──────────┴──────────────────┴────────────────────────┘
        """
        self.setWindowTitle("CR5 RMP60 标定采集控制台")
        self.resize(1280, 760)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self.setCentralWidget(scroll)

        root = QWidget()
        root.setMinimumSize(1020, 680)
        scroll.setWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(16, 14, 16, 14)
        main.setSpacing(12)

        # --- 顶部标题栏与状态徽章 ---
        header = QHBoxLayout()
        title = QLabel("CR5 RMP60 标定采集控制台")
        title.setObjectName("Title")
        subtitle = QLabel("位姿记录 / 低速点动 / DI1 触发采集")
        subtitle.setObjectName("Subtitle")
        title_box = QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self.connection_badge = QLabel("未连接")
        self.connection_badge.setObjectName("BadgeOff")
        self.di_badge = QLabel("DI1 -")
        self.di_badge.setObjectName("BadgeNeutral")
        self.feed_badge = QLabel("Feed -")
        self.feed_badge.setObjectName("BadgeNeutral")
        header.addWidget(self.feed_badge)
        header.addWidget(self.di_badge)
        header.addWidget(self.connection_badge)
        main.addLayout(header)

        # --- 三栏主内容区 ---
        content = QHBoxLayout()
        content.setSpacing(12)
        main.addLayout(content, 1)

        # 左栏：连接 + 位姿 + 元信息
        left = QVBoxLayout()
        left.addWidget(self.create_connection_group())
        left.addWidget(self.create_pose_group())
        left.addWidget(self.create_metadata_group())
        content.addLayout(left, 2)

        # 中栏：点动 + 法向 + 安全
        center = QVBoxLayout()
        center.addWidget(self.create_jog_group())
        center.addWidget(self.create_continuous_jog_group())
        center.addWidget(self.create_vector_group())
        center.addWidget(self.create_normal_probe_group())
        center.addWidget(self.create_safety_group())
        content.addLayout(center, 2)

        # 右栏：记录表 + 日志
        right = QVBoxLayout()
        right.addWidget(self.create_record_table_group(), 3)
        right.addWidget(self.create_log_group(), 2)
        content.addLayout(right, 3)

    def create_connection_group(self):
        """
        创建「连接与采集」分组框。

        包含按钮：
        - 连接 ROS2 驱动
        - 刷新位姿
        - 记录当前位姿（手动非接触记录）
        - 按上次路径回退（DI1 触发后安全退开）
        - STOP（紧急停止）
        """
        group = QGroupBox("连接与采集")
        layout = QGridLayout(group)
        self.connect_btn = QPushButton("连接 ROS2 驱动")
        self.refresh_btn = QPushButton("刷新位姿")
        self.capture_btn = QPushButton("记录当前位姿")
        self.retract_last_btn = QPushButton("按上次路径回退")
        self.stop_btn = QPushButton("急停 / STOP")
        self.stop_btn.setObjectName("StopButton")
        self.connect_btn.clicked.connect(self.connectRequested.emit)
        self.refresh_btn.clicked.connect(self.refreshRequested.emit)
        self.capture_btn.clicked.connect(lambda: self.captureRequested.emit(self.settings()))
        self.retract_last_btn.clicked.connect(lambda: self.retractLastRequested.emit(self.settings()))
        self.stop_btn.clicked.connect(self.emergency_stop)
        layout.addWidget(self.connect_btn, 0, 0)
        layout.addWidget(self.refresh_btn, 0, 1)
        layout.addWidget(self.capture_btn, 1, 0)
        layout.addWidget(self.retract_last_btn, 1, 1)
        layout.addWidget(self.stop_btn, 2, 0, 1, 2)
        return group

    def create_pose_group(self):
        """
        创建「实时法兰位姿」分组框。

        实时显示当前机器人法兰盘的六维位姿 (x, y, z, rx, ry, rz)
        和六个关节角 (J1-J6)。数值由 refresh 定时器自动更新。
        位姿值采用绿色等宽字体高亮显示。
        """
        group = QGroupBox("实时法兰位姿")
        layout = QGridLayout(group)
        for index, name in enumerate(POSE_NAMES):
            label = QLabel(name.upper())
            value = QLabel("--")
            value.setObjectName("PoseValue")
            self.pose_labels[name] = value
            layout.addWidget(label, index // 3 * 2, index % 3)
            layout.addWidget(value, index // 3 * 2 + 1, index % 3)
        joint_title = QLabel("关节角")
        joint_title.setObjectName("SectionLabel")
        layout.addWidget(joint_title, 4, 0, 1, 3)
        for index in range(6):
            value = QLabel(f"J{index + 1}: --")
            value.setObjectName("JointValue")
            self.joint_labels.append(value)
            layout.addWidget(value, 5 + index // 3, index % 3)
        return group

    def create_metadata_group(self):
        """
        创建「数据元信息」分组框。

        编辑当前采集会话的元数据，这些字段会随每行记录写入 CSV：
        - 输出 CSv 路径（可浏览选择或切换文件）
        - Session ID / 工件 ID / 标准件 ID
        - 测球编号 (physical_ball_id)
        - 分支标签 (y_neg, y_pos, x_neg, x_pos, z_down, unknown)
        - 操作备注
        """
        group = QGroupBox("数据元信息")
        layout = QFormLayout(group)
        self.output_edit = QLineEdit(str(self.args.output))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        self.output_browse_btn = QPushButton("选择/追加")
        self.output_browse_btn.clicked.connect(self.choose_output)
        output_row.addWidget(self.output_browse_btn)
        self.output_edit.editingFinished.connect(self.reload_records)
        save_row = QHBoxLayout()
        self.manual_save_btn = QPushButton("手动保存 CSV")
        self.manual_save_btn.clicked.connect(self.manual_save_records)
        self.save_folder_btn = QPushButton("保存到文件夹")
        self.save_folder_btn.clicked.connect(self.save_records_to_folder)
        save_row.addWidget(self.manual_save_btn)
        save_row.addWidget(self.save_folder_btn)
        self.session_edit = QLineEdit(self.args.session_id)
        self.workpiece_edit = QLineEdit(self.args.workpiece_id)
        self.artifact_edit = QLineEdit(self.args.artifact_id)
        self.ball_edit = QLineEdit(self.args.physical_ball_id)
        self.branch_combo = QComboBox()
        self.branch_combo.addItems(["y_neg", "y_pos", "x_neg", "x_pos", "z_down", "unknown"])
        self.branch_combo.setCurrentText(self.args.branch)
        self.note_edit = QLineEdit(self.args.operator_note)
        layout.addRow("输出 CSV", output_row)
        layout.addRow("保存数据", save_row)
        save_hint = QLabel("采集后自动写入；手动保存可另存当前全部记录，后续采集继续写入新位置。")
        save_hint.setWordWrap(True)
        save_hint.setObjectName("Hint")
        layout.addRow(save_hint)
        layout.addRow("Session", self.session_edit)
        layout.addRow("工件 ID", self.workpiece_edit)
        layout.addRow("标准件 ID", self.artifact_edit)
        layout.addRow("测球编号", self.ball_edit)
        layout.addRow("分支标签", self.branch_combo)
        layout.addRow("备注", self.note_edit)
        return group

    def create_jog_group(self):
        """
        创建「坐标点动」分组框。

        包含：
        - 平移步长 (mm) 和 旋转步长 (deg) 参数
        - 速度倍率 (1-5)
        - 触发后自动回退距离 (mm)
        - 六轴正负方向点动按钮 (12个)

        按钮使用 4 列 x 3 行布局：
        X- X+  Y- Y+
        Z- Z+  RX- RX+
        RY- RY+ RZ- RZ+
        """
        group = QGroupBox("坐标点动")
        layout = QGridLayout(group)
        self.linear_step_spin = self.double_spin(0.01, 10.0, self.args.linear_step_mm, 3, 0.1)
        self.angular_step_spin = self.double_spin(0.1, 20.0, self.args.angular_step_deg, 2, 0.5)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 5)
        self.speed_spin.setValue(self.args.speed)
        self.retract_spin = self.double_spin(0.0, 10.0, self.args.auto_retract_mm, 3, 0.1)
        layout.addWidget(QLabel("平移步长 mm"), 0, 0)
        layout.addWidget(self.linear_step_spin, 0, 1)
        layout.addWidget(QLabel("转动步长 deg"), 0, 2)
        layout.addWidget(self.angular_step_spin, 0, 3)
        layout.addWidget(QLabel("速度倍率"), 1, 0)
        layout.addWidget(self.speed_spin, 1, 1)
        layout.addWidget(QLabel("触发回退 mm"), 1, 2)
        layout.addWidget(self.retract_spin, 1, 3)

        buttons = [
            ("X-", "x", -1.0, 2, 0), ("X+", "x", 1.0, 2, 1),
            ("Y-", "y", -1.0, 2, 2), ("Y+", "y", 1.0, 2, 3),
            ("Z-", "z", -1.0, 3, 0), ("Z+", "z", 1.0, 3, 1),
            ("RX-", "rx", -1.0, 3, 2), ("RX+", "rx", 1.0, 3, 3),
            ("RY-", "ry", -1.0, 4, 0), ("RY+", "ry", 1.0, 4, 1),
            ("RZ-", "rz", -1.0, 4, 2), ("RZ+", "rz", 1.0, 4, 3),
        ]
        for text, axis, sign, row, col in buttons:
            btn = QPushButton(text)

            # 每个按钮的 clicked 信号需要绑定当前 axis/sign 的副本，
            # 使用 lambda 默认参数避免闭包中的变量共享问题
            btn.clicked.connect(lambda _, axis=axis, sign=sign: self.jogAxisRequested.emit(axis, sign, self.settings()))
            self.motion_buttons.append(btn)
            layout.addWidget(btn, row, col)
        return group

    def create_continuous_jog_group(self):
        """
        创建「持续点动」分组框。

        一次按住发送一条有行程上限的 MovL，松开时主动 Stop。
        行程内持续监控 DI1，触发后复用单步点动的记录和回退逻辑。
        """
        group = QGroupBox("持续点动（按住）")
        layout = QGridLayout(group)
        hint = QLabel("按住连续移动，松开发送 Stop；到达单次上限后需重新按住。DI1 触发记录并回退。")
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        layout.addWidget(hint, 0, 0, 1, 4)
        self.continuous_linear_spin = self.double_spin(0.1, 10.0, 10.0, 2, 0.5)
        self.continuous_angular_spin = self.double_spin(0.1, 20.0, 10.0, 2, 0.5)
        layout.addWidget(QLabel("单次平移上限 mm"), 1, 0)
        layout.addWidget(self.continuous_linear_spin, 1, 1)
        layout.addWidget(QLabel("单次转动上限 deg"), 1, 2)
        layout.addWidget(self.continuous_angular_spin, 1, 3)
        buttons = [
            ("X-", "x", -1.0, 1, 0), ("X+", "x", 1.0, 1, 1),
            ("Y-", "y", -1.0, 1, 2), ("Y+", "y", 1.0, 1, 3),
            ("Z-", "z", -1.0, 2, 0), ("Z+", "z", 1.0, 2, 1),
            ("RX-", "rx", -1.0, 2, 2), ("RX+", "rx", 1.0, 2, 3),
            ("RY-", "ry", -1.0, 3, 0), ("RY+", "ry", 1.0, 3, 1),
            ("RZ-", "rz", -1.0, 3, 2), ("RZ+", "rz", 1.0, 3, 3),
        ]
        for text, axis, sign, row, col in buttons:
            btn = QPushButton(text)
            btn.pressed.connect(lambda axis=axis, sign=sign, button=btn: self.start_continuous_step_jog(axis, sign, button))
            btn.released.connect(self.stop_continuous_step_jog)
            self.continuous_buttons.append(btn)
            layout.addWidget(btn, row + 1, col)
        stop_jog_btn = QPushButton("停止持续点动")
        stop_jog_btn.clicked.connect(lambda: self.stop_continuous_step_jog())
        layout.addWidget(stop_jog_btn, 5, 0, 1, 4)
        return group

    def create_vector_group(self):
        """
        创建「复合方向点动」分组框。

        允许用户输入任意三维矢量 (dx, dy, dz) 和移动距离，
        机器人沿该方向移动指定距离。方向向量会自动归一化。

        默认的 dx=-1, dy=1, dz=0 组合用于 X-/Y+ 复合方向接近球体。
        """
        group = QGroupBox("复合方向点动")
        layout = QGridLayout(group)
        self.vec_x_spin = self.double_spin(-10.0, 10.0, -1.0, 3, 0.1)
        self.vec_y_spin = self.double_spin(-10.0, 10.0, 1.0, 3, 0.1)
        self.vec_z_spin = self.double_spin(-10.0, 10.0, 0.0, 3, 0.1)
        self.vec_dist_spin = self.double_spin(0.01, 20.0, 0.2, 3, 0.1)
        self.vec_btn = QPushButton("执行复合点动")
        self.vec_btn.clicked.connect(self.emit_vector_jog)
        layout.addWidget(QLabel("dx"), 0, 0)
        layout.addWidget(self.vec_x_spin, 0, 1)
        layout.addWidget(QLabel("dy"), 0, 2)
        layout.addWidget(self.vec_y_spin, 0, 3)
        layout.addWidget(QLabel("dz"), 1, 0)
        layout.addWidget(self.vec_z_spin, 1, 1)
        layout.addWidget(QLabel("距离 mm"), 1, 2)
        layout.addWidget(self.vec_dist_spin, 1, 3)
        layout.addWidget(self.vec_btn, 2, 0, 1, 4)
        hint = QLabel("当前常用：dx=-1, dy=1, dz=0，用于 X-/Y+ 复合接近。")
        hint.setObjectName("Hint")
        layout.addWidget(hint, 3, 0, 1, 4)
        return group

    def create_normal_probe_group(self):
        """
        创建「半自动法向」分组框。

        实现基于球面拟合的法向短探测工作流：
        1. 选择拟合 JSON 文件（通过 browse 按钮或直接输入路径）
        2. 设置法向搜索距离 (0.01-1.0mm)
        3. 勾选「已确认路径安全」才能执行实际运动
        4. 「计算法向」- 干运行，只计算不运动，显示计算结果
        5. 「沿法向点动」- 沿法向方向移动但不进入探测模式
        6. 「执行短探测」- 完整流程：接近 -> 等待 DI1 -> 记录 -> 回退
        """
        group = QGroupBox("半自动法向")
        layout = QGridLayout(group)
        self.normal_fit_edit = QLineEdit(self.args.normal_fit_json)
        browse = QPushButton("选择")
        browse.clicked.connect(self.choose_normal_fit)
        layout.addWidget(QLabel("拟合 JSON"), 0, 0)
        layout.addWidget(self.normal_fit_edit, 0, 1, 1, 2)
        layout.addWidget(browse, 0, 3)

        self.normal_search_spin = self.double_spin(0.01, 1.0, self.args.normal_search_mm, 3, 0.05)
        self.normal_ack_check = QCheckBox("已确认路径安全")
        layout.addWidget(QLabel("搜索 mm"), 1, 0)
        layout.addWidget(self.normal_search_spin, 1, 1)
        layout.addWidget(self.normal_ack_check, 1, 2, 1, 2)

        self.normal_dry_btn = QPushButton("计算法向")
        self.normal_jog_btn = QPushButton("沿法向点动")
        self.normal_execute_btn = QPushButton("执行短探测")
        self.normal_dry_btn.clicked.connect(lambda: self.normalDryRunRequested.emit(self.settings()))
        self.normal_jog_btn.clicked.connect(lambda: self.normalJogRequested.emit(self.settings()))
        self.normal_execute_btn.clicked.connect(lambda: self.normalProbeRequested.emit(self.settings()))
        self.normal_buttons.extend([self.normal_dry_btn, self.normal_jog_btn, self.normal_execute_btn])
        layout.addWidget(self.normal_dry_btn, 2, 0)
        layout.addWidget(self.normal_jog_btn, 2, 1)
        layout.addWidget(self.normal_execute_btn, 2, 2, 1, 2)
        return group

    def create_safety_group(self):
        """
        创建「执行约束」分组框。

        设置运动执行的安全参数：
        - 单步超时时间 (1-30s)：超时未到位则停止
        - 到位位置容差 (mm)：判定是否到达目标位置
        - 到位姿态容差 (deg)：判定是否到达目标姿态

        底部显示安全警告文字。
        """
        group = QGroupBox("执行约束")
        layout = QFormLayout(group)
        self.timeout_spin = self.double_spin(1.0, 30.0, self.args.timeout_sec, 1, 1.0)
        self.pos_tol_spin = self.double_spin(0.01, 1.0, self.args.position_tolerance_mm, 3, 0.01)
        self.rot_tol_spin = self.double_spin(0.01, 2.0, self.args.orientation_tolerance_deg, 3, 0.01)
        layout.addRow("单步超时 s", self.timeout_spin)
        layout.addRow("到位位置容差 mm", self.pos_tol_spin)
        layout.addRow("到位姿态容差 deg", self.rot_tol_spin)
        warning = QLabel("运动前必须确认现场路径安全；DI1 已触发时拒绝点动。")
        warning.setWordWrap(True)
        warning.setObjectName("WarningText")
        layout.addRow(warning)
        return group

    def create_record_table_group(self):
        """
        创建「采集记录」分组框。

        使用 QTableWidget 显示已经记录的样本的摘要信息：
        列：序号 | 命令 | X | Y | Z | DI1 | 备注
        启动、切换输出文件以及写入或删除后，显示当前 CSV 中的记录。
        """
        group = QGroupBox("采集记录")
        layout = QVBoxLayout(group)
        self.record_table = QTableWidget(0, 7)
        self.record_table.setHorizontalHeaderLabels(["#", "命令", "X", "Y", "Z", "DI1", "备注"])
        self.record_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.record_table.verticalHeader().setVisible(False)
        self.record_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.record_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.record_table.itemSelectionChanged.connect(self.update_delete_button)
        layout.addWidget(self.record_table)
        actions = QHBoxLayout()
        self.reload_records_btn = QPushButton("刷新记录")
        self.reload_records_btn.clicked.connect(self.reload_records)
        self.delete_records_btn = QPushButton("删除选中记录")
        self.delete_records_btn.clicked.connect(self.delete_selected_records)
        self.delete_records_btn.setEnabled(False)
        actions.addWidget(self.reload_records_btn)
        actions.addWidget(self.delete_records_btn)
        layout.addLayout(actions)
        hint = QLabel("按 Ctrl/Shift 可多选；删除会同步修改 CSV，并自动备份原文件。")
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        layout.addWidget(hint)
        return group

    def create_log_group(self):
        """
        创建「状态日志」分组框。

        使用 QTextEdit（只读）以彩色 HTML 格式显示操作日志。
        不同级别使用不同颜色：SUCCESS(绿)、WARN(橙)、ERROR(红)、INFO(蓝)。
        """
        group = QGroupBox("状态日志")
        layout = QVBoxLayout(group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return group

    # ---------- UI 辅助方法 ----------

    def settings(self):
        """
        从所有 UI 控件收集当前参数设置，用于传递给工作线程的各个操作。

        如果用户切换了输出文件路径，自动恢复该文件的最大样本编号，
        实现断点续采。

        Returns:
            dict: 包含所有操作参数的字典
        """
        output = self.current_output_path()
        self.output_edit.setText(output)
        # 检测输出路径是否切换，切换时从新文件恢复样本编号
        if output != self.sample_index_output:
            self.worker.sample_index = existing_max_sample_index(output)
            self.sample_index_output = output
        return {
            "output": output,
            "session_id": self.session_edit.text().strip(),
            "workpiece_id": self.workpiece_edit.text().strip(),
            "artifact_id": self.artifact_edit.text().strip(),
            "artifact_type": "sphere",
            "physical_ball_id": self.ball_edit.text().strip(),
            "branch": self.branch_combo.currentText(),
            "operator_note": self.note_edit.text().strip(),
            "linear_step_mm": self.linear_step_spin.value(),
            "angular_step_deg": self.angular_step_spin.value(),
            "continuous_linear_limit_mm": self.continuous_linear_spin.value(),
            "continuous_angular_limit_deg": self.continuous_angular_spin.value(),
            "speed": self.speed_spin.value(),
            "auto_retract_mm": self.retract_spin.value(),
            "timeout_sec": self.timeout_spin.value(),
            "position_tolerance_mm": self.pos_tol_spin.value(),
            "orientation_tolerance_deg": self.rot_tol_spin.value(),
            "normal_fit_json": self.normal_fit_edit.text().strip(),
            "normal_search_mm": self.normal_search_spin.value(),
            "normal_ack_safe": self.normal_ack_check.isChecked(),
            "normal_euler_sequence": self.args.normal_euler_sequence,
            "sample_index": 0,
        }

    def emit_vector_jog(self):
        """
        发射复合方向点动信号前进行校验。

        如果矢量 (dx, dy, dz) 接近零向量则弹出警告并阻止操作。
        方向向量后续在 vector_jog_target 中会被归一化。
        """
        dx = self.vec_x_spin.value()
        dy = self.vec_y_spin.value()
        dz = self.vec_z_spin.value()
        if math.sqrt(dx * dx + dy * dy + dz * dz) <= 1e-9:
            QMessageBox.warning(self, "方向错误", "复合方向不能为零向量。")
            return
        self.jogVectorRequested.emit(dx, dy, dz, self.vec_dist_spin.value(), self.settings())

    def choose_output(self):
        """选择已有 CSV 继续追加，或将当前记录保存到新 CSV。"""
        if self.gui_busy or self.continuous_request is not None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self, "选择/新建输出 CSV（已有文件将追加）", self.current_output_path(),
            "CSV (*.csv)", options=QFileDialog.DontConfirmOverwrite,
        )
        if not selected:
            return
        path = str(resolve_contact_output(selected, self.output_filename()))
        source = self.records_path or self.sample_index_output
        if not Path(path).exists() and Path(source).is_file():
            self.save_records_to(path)
            return
        self.output_edit.setText(path)
        self.worker.sample_index = existing_max_sample_index(path)
        self.sample_index_output = path
        self.append_log("INFO", f"输出 CSV 已选择为追加模式；下一条 sample 从 {self.worker.sample_index + 1} 开始")
        self.reload_records()

    def output_filename(self):
        return Path(self.records_path or self.sample_index_output).name

    def manual_save_records(self):
        """手动保存当前完整数据，可选择新文件名。"""
        if self.gui_busy or self.continuous_request is not None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self, "手动保存采集记录为 CSV", self.current_output_path(), "CSV (*.csv)",
        )
        if selected:
            self.save_records_to(selected)

    def save_records_to_folder(self):
        """选目录后自动使用当前 CSV 文件名，保存已有记录。"""
        if self.gui_busy or self.continuous_request is not None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "选择采集数据保存文件夹", str(Path(self.current_output_path()).parent),
        )
        if folder:
            self.save_records_to(Path(folder) / self.output_filename())

    def save_records_to(self, destination):
        if self.gui_busy or self.continuous_request is not None:
            return
        source = self.records_path or self.sample_index_output
        path = resolve_contact_output(destination, self.output_filename())
        if not Path(source).is_file():
            self.output_edit.setText(str(path))
            self.worker.sample_index = existing_max_sample_index(path)
            self.sample_index_output = str(path)
            self.append_log("INFO", f"已设置保存位置: {path}；采集记录后自动生成 CSV")
            self.reload_records()
            return
        self.on_busy_changed(True)
        self.saveRecordsRequested.emit(source, str(path))

    def on_records_saved(self, output):
        self.output_edit.setText(output)
        self.sample_index_output = output

    def choose_normal_fit(self):
        """
        打开文件选择对话框，选择法向拟合 JSON 文件。
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择粗拟合 JSON",
            self.normal_fit_edit.text(),
            "JSON (*.json)",
        )
        if path:
            self.normal_fit_edit.setText(path)

    # ---------- 信号槽回调 ----------

    def on_connected(self, connected):
        """
        处理连接状态变化：更新连接徽章。

        Args:
            connected: bool，是否连接成功
        """
        self.connected = connected
        if not connected:
            self.stop_continuous_step_jog(emergency=True)
        self.connection_badge.setText("已连接" if connected else "未连接")
        self.connection_badge.setObjectName("BadgeOn" if connected else "BadgeOff")
        self.connection_badge.style().unpolish(self.connection_badge)
        self.connection_badge.style().polish(self.connection_badge)

    def on_pose_updated(self, pose, di1, joints, feed_age):
        """
        处理位姿更新事件：刷新所有显示标签和状态徽章。

        Args:
            pose: [x, y, z, rx, ry, rz] 列表
            di1: bool，DI1 状态
            joints: 关节角列表或 None
            feed_age: FeedInfo 滞后时间（秒）
        """
        for index, name in enumerate(POSE_NAMES):
            self.pose_labels[name].setText(f"{pose[index]:.4f}")
        if joints:
            for index, value in enumerate(joints[:6]):
                self.joint_labels[index].setText(f"J{index + 1}: {float(value):.4f}")
        self.di_badge.setText(f"DI1 {int(di1)}")
        # DI1 触发时显示橙色警告，正常时显示绿色
        self.di_badge.setObjectName("BadgeWarn" if di1 else "BadgeOn")
        self.feed_badge.setText(f"Feed {feed_age:.2f}s")
        # Feed 滞后超过 0.2s 显示橙色警告
        self.feed_badge.setObjectName("BadgeWarn" if feed_age > 0.2 else "BadgeOn")
        for label in (self.di_badge, self.feed_badge):
            label.style().unpolish(label)
            label.style().polish(label)

    def on_sample_recorded(self, row):
        """
        处理新记录事件：在记录表中追加一行摘要。

        表中显示：样本编号、命令、触发位姿的 X/Y/Z、DI1 状态、备注。

        Args:
            row: dict，CSV 行数据
        """
        r = self.record_table.rowCount()
        self.record_table.insertRow(r)
        values = [
            row.get("sample_index") or row.get("gui_sample_index", ""),
            row.get("command", ""),
            row.get("trigger_flange_x", ""),
            row.get("trigger_flange_y", ""),
            row.get("trigger_flange_z", ""),
            row.get("trigger_di1", ""),
            row.get("operator_note", ""),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if col == 0:
                item.setData(Qt.UserRole, dict(row))
            self.record_table.setItem(r, col, item)
        self.record_table.scrollToBottom()

    def current_output_path(self):
        return str(resolve_contact_output(
            self.output_edit.text().strip() or DEFAULT_OUTPUT, self.output_filename(),
        ))

    def reload_records(self):
        if self.gui_busy or self.continuous_request is not None:
            return
        raw_path = Path(self.output_edit.text().strip() or DEFAULT_OUTPUT).expanduser()
        if raw_path.is_dir() and Path(self.records_path or self.sample_index_output).is_file():
            self.save_records_to(raw_path)
            return
        output = self.current_output_path()
        if self.records_path and Path(self.records_path).is_file() and output != self.records_path and not Path(output).exists():
            self.save_records_to(output)
            return
        self.output_edit.setText(output)
        self.on_busy_changed(True)
        self.loadRecordsRequested.emit(output)

    def on_records_loaded(self, output, rows):
        if output != self.current_output_path():
            return
        self.records_path = output
        self.record_table.setRowCount(0)
        for row in rows:
            self.on_sample_recorded(row)
        self.update_delete_button()

    def update_delete_button(self):
        self.delete_records_btn.setEnabled(
            not self.gui_busy and self.continuous_request is None
            and self.records_path == self.current_output_path()
            and bool(self.record_table.selectionModel().selectedRows())
        )

    def delete_selected_records(self):
        if self.gui_busy or self.continuous_request is not None:
            return
        if self.records_path != self.current_output_path():
            self.reload_records()
            self.append_log("WARN", "输出文件已切换，请选择该文件中的记录后再删除")
            return
        selected = [
            self.record_table.item(index.row(), 0).data(Qt.UserRole)
            for index in self.record_table.selectionModel().selectedRows()
        ]
        if not selected:
            self.append_log("WARN", "请先选择要删除的采集记录，可按 Ctrl/Shift 多选")
            return
        self.on_busy_changed(True)
        self.deleteRecordsRequested.emit(self.records_path, selected)

    def on_busy_changed(self, busy):
        """
        处理工作线程繁忙状态变化：禁用/启用所有操作按钮。

        当有运动操作正在执行时，防止用户同时发起多个操作。

        Args:
            busy: bool，True 表示工作中，False 表示空闲
        """
        self.gui_busy = busy
        self.connect_btn.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy)
        self.capture_btn.setEnabled(not busy)
        self.retract_last_btn.setEnabled(not busy)
        self.vec_btn.setEnabled(not busy)
        self.output_edit.setEnabled(not busy)
        self.output_browse_btn.setEnabled(not busy)
        self.manual_save_btn.setEnabled(not busy)
        self.save_folder_btn.setEnabled(not busy)
        self.reload_records_btn.setEnabled(not busy)
        self.update_delete_button()
        for btn in self.motion_buttons:
            btn.setEnabled(not busy)
        for btn in self.normal_buttons:
            btn.setEnabled(not busy)
        for btn in self.continuous_buttons:
            # 按住的按钮保持启用，保证 released 信号正常到达。
            btn.setEnabled(not busy or btn is self.continuous_button)
        for spin in (self.continuous_linear_spin, self.continuous_angular_spin):
            spin.setEnabled(not busy)

    def start_continuous_step_jog(self, axis, sign, button=None):
        """
        一次按住只发送一条有界 MovL；50ms 定时器只续心跳，不发送运动。
        """
        if not self.connected:
            self.append_log("WARN", "尚未连接，无法持续点动")
            return
        if self.gui_busy or self.continuous_request is not None:
            self.append_log("WARN", "当前运动未完成，无法启动持续点动")
            return
        settings = self.settings()
        self.continuous_hold = (axis, sign)
        self.continuous_request = HeldJogRequest()
        self.continuous_button = button
        # 同步锁住 GUI，防止工作线程的 busyChanged 到达前再次排队。
        self.on_busy_changed(True)
        self.append_log("INFO", f"持续点动开始: {axis}{'+' if sign > 0 else '-'}，松开发送 Stop")
        self.continuous_heartbeat_timer.start()
        self.continuousJogStartRequested.emit(axis, sign, settings, self.continuous_request)

    def stop_continuous_step_jog(self, emergency=False):
        """
        直接设置线程安全取消令牌，避免 Stop 槽排在运行中的运动槽之后。
        """
        if self.continuous_request is None:
            return
        self.continuous_request.stop(emergency=emergency)
        self.continuous_hold = None
        self.continuous_heartbeat_timer.stop()

    def keep_continuous_jog_alive(self):
        """GUI 无响应、按钮不再按下或窗口失焦时，持续运动停止。"""
        if self.continuous_request is None or self.continuous_hold is None:
            return
        if self.continuous_button is not None and not self.continuous_button.isDown():
            self.stop_continuous_step_jog()
            return
        self.continuous_request.keep_alive()

    def on_continuous_jog_finished(self, request):
        if request is not self.continuous_request:
            return
        self.continuous_heartbeat_timer.stop()
        self.continuous_hold = None
        self.continuous_request = None
        self.continuous_button = None
        self.on_busy_changed(False)
        self.refreshRequested.emit()

    def emergency_stop(self):
        self.stop_continuous_step_jog(emergency=True)
        self.stopRequested.emit()

    def append_log(self, level, message):
        """
        在状态日志框中追加一条带颜色和时间戳的日志消息。

        Args:
            level: 日志级别，如 "SUCCESS", "WARN", "ERROR", "INFO"
            message: 日志内容文本
        """
        color = {
            "SUCCESS": "#20C997",
            "WARN": "#F59F00",
            "ERROR": "#FF6B6B",
            "INFO": "#74C0FC",
        }.get(level, "#DEE2E6")
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f'<span style="color:#868E96">[{ts}]</span> '
                             f'<span style="color:{color};font-weight:600">{level}</span> '
                             f'<span style="color:#DEE2E6">{message}</span>')

    @staticmethod
    def double_spin(minimum, maximum, value, decimals, step):
        """
        创建 QDoubleSpinBox 的工厂方法，统一设置范围和步长。

        Args:
            minimum: 最小值
            maximum: 最大值
            value: 默认值
            decimals: 小数位数
            step: 单步步长

        Returns:
            QDoubleSpinBox 实例
        """
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setKeyboardTracking(False)
        return spin

    def apply_style(self):
        """
        应用深色主题样式表。

        使用 QPalette 和 QSS 实现统一的暗色界面风格：
        - 背景色：深灰 (#151A1F)
        - 文字色：浅灰 (#E9ECEF)
        - 输入框：深色底 (#0F1317)
        - 绿色用于位姿数值 (#69DB7C)
        - 红色用于 STOP 按钮 (#C92A2A)
        - 各类徽章的 On/Off/Warn/Neutral 颜色编码
        """
        app = QApplication.instance()
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#151A1F"))
        palette.setColor(QPalette.WindowText, QColor("#E9ECEF"))
        palette.setColor(QPalette.Base, QColor("#0F1317"))
        palette.setColor(QPalette.AlternateBase, QColor("#20262D"))
        palette.setColor(QPalette.Text, QColor("#F1F3F5"))
        palette.setColor(QPalette.Button, QColor("#2B333B"))
        palette.setColor(QPalette.ButtonText, QColor("#F8F9FA"))
        palette.setColor(QPalette.Highlight, QColor("#1971C2"))
        palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        app.setPalette(palette)
        app.setFont(QFont("Noto Sans CJK SC", 10))
        self.setStyleSheet("""
            QMainWindow { background: #151A1F; }
            QGroupBox {
                border: 1px solid #343A40;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px;
                background: #1B2026;
                color: #F1F3F5;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel { color: #DDE2E6; }
            QLabel#Title { font-size: 24px; font-weight: 700; color: #F8F9FA; }
            QLabel#Subtitle { color: #ADB5BD; }
            QLabel#PoseValue {
                color: #69DB7C;
                background: #0F1317;
                border: 1px solid #2B333B;
                border-radius: 4px;
                padding: 8px;
                font-family: "DejaVu Sans Mono";
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#JointValue { color: #CED4DA; font-family: "DejaVu Sans Mono"; }
            QLabel#SectionLabel { color: #74C0FC; font-weight: 700; margin-top: 8px; }
            QLabel#Hint { color: #ADB5BD; }
            QLabel#WarningText { color: #FFD43B; }
            QLabel#BadgeOn, QLabel#BadgeOff, QLabel#BadgeWarn, QLabel#BadgeNeutral {
                border-radius: 4px;
                padding: 7px 12px;
                font-weight: 700;
                min-width: 72px;
                qproperty-alignment: AlignCenter;
            }
            QLabel#BadgeOn { background: #087F5B; color: #E6FCF5; }
            QLabel#BadgeOff { background: #862E2E; color: #FFF5F5; }
            QLabel#BadgeWarn { background: #E67700; color: #FFF9DB; }
            QLabel#BadgeNeutral { background: #343A40; color: #DEE2E6; }
            QPushButton {
                background: #2B333B;
                border: 1px solid #495057;
                border-radius: 5px;
                padding: 9px 12px;
                color: #F8F9FA;
                font-weight: 600;
            }
            QPushButton:hover { background: #36424C; border-color: #74C0FC; }
            QPushButton:pressed { background: #1971C2; }
            QPushButton:disabled { background: #212529; color: #6C757D; border-color: #343A40; }
            QPushButton#StopButton { background: #C92A2A; border-color: #FF8787; color: white; }
            QPushButton#StopButton:hover { background: #E03131; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #0F1317;
                color: #F1F3F5;
                border: 1px solid #495057;
                border-radius: 4px;
                padding: 6px;
            }
            QTextEdit, QTableWidget {
                background: #0F1317;
                color: #DEE2E6;
                border: 1px solid #343A40;
                border-radius: 4px;
            }
            QHeaderView::section {
                background: #2B333B;
                color: #F8F9FA;
                border: 0;
                padding: 7px;
                font-weight: 700;
            }
        """)


def parse_args():
    """
    解析命令行参数。

    所有参数都有合理的默认值，大多数参数可在 GUI 界面中实时修改。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--session-id", default="session_gui_probe_calibration")
    parser.add_argument("--workpiece-id", default="calibration_sphere_20mm")
    parser.add_argument("--artifact-id", default="standard_sphere_20mm")
    parser.add_argument("--physical-ball-id", default="1")
    parser.add_argument("--branch", default="y_neg")
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--linear-step-mm", type=float, default=0.2)
    parser.add_argument("--angular-step-deg", type=float, default=1.0)
    parser.add_argument("--auto-retract-mm", type=float, default=1.0)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.08)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.08)
    parser.add_argument("--normal-fit-json", default=str(DEFAULT_NORMAL_FIT))
    parser.add_argument("--normal-search-mm", type=float, default=0.2)
    parser.add_argument("--normal-euler-sequence", default="xyz")
    parser.add_argument("--auto-connect", action="store_true", help="GUI 启动后自动连接 ROS2/Dobot")
    return parser.parse_args()


def main():
    """程序入口：解析参数、创建 GUI 应用、显示主窗口。"""
    args = parse_args()
    app = QApplication(sys.argv)
    window = ProbeCalibrationWindow(args)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
