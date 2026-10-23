#!/usr/bin/env python3
"""Operator workbench: manual five-branch localization -> guarded automatic batches."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSpinBox, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from cross_probe_model import DEFAULT_GEOMETRY, branch_local_offset_mm, load_geometry
from direct_tcp_execute_five_branch_plan import Dashboard, FeedbackReader
from probe_calibration_gui import RosProbeWorker, existing_max_sample_index
from sphere_experiment_core import (
    BRANCHES, append_jsonl, export_manual_contacts_csv, fit_anchor_center, fit_coarse_center, fit_manual_relative_seed,
    fit_relocated_center_z,
    fit_new_installation_geometry, fit_usable_for_motion, dataset_split_for_placement,
    validate_fit_dimensions, validate_fixture_fit_height,
    read_csv, read_jsonl, sha256,
)
from sphere_experiment_runner import StageRunner, require_complete_stage
from sphere_simulation import apply_scene_to_moveit, freeze_preview_plans, plan_preview, scene_checks, scene_from_profile
from workbench_profile import (
    DEFAULT_PROFILE, fixture_sphere_height_base_mm, load_profile, profile_sha256, register_fixture_to_sphere,
    require_supported_contact_geometry,
    save_profile, validate_profile,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_LABELS = {
    "smoke": "1 冒烟：五分支各1点",
    "geometry": "2 P0几何：48点",
    "anchor": "3 球心锚点：Z分支8点",
    "evaluation": "4 五分支评价：90点",
}
PLACEMENT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def sanitize_moveit_runtime_environment():
    """Ignore the stale workspace OMPL build while retaining the Dobot overlay."""
    workspace = Path(os.environ.get("CR5_WORKSPACE", str(ROOT.parents[1])))
    stale_prefix = str(workspace / "install/moveit_planners_ompl")
    rejected = {
        "AMENT_PREFIX_PATH": stale_prefix,
        "CMAKE_PREFIX_PATH": stale_prefix,
        "COLCON_PREFIX_PATH": stale_prefix,
        "LD_LIBRARY_PATH": stale_prefix + "/lib",
    }
    for variable, entry_to_remove in rejected.items():
        entries = [entry for entry in os.environ.get(variable, "").split(":")
                   if entry and entry != entry_to_remove]
        os.environ[variable] = ":".join(entries)


def validate_placement_id(value: str) -> str:
    placement = str(value).strip()
    if not placement:
        raise ValueError("球位编号不能为空")
    if not PLACEMENT_ID_PATTERN.fullmatch(placement):
        raise ValueError("球位编号只能包含字母、数字、下划线和短横线，且最长64字符")
    return placement


def parse_xyz(text: str, label: str) -> list[float]:
    values = [part.strip() for part in text.replace("，", ",").split(",")]
    if len(values) != 3:
        raise ValueError(f"{label}需要三个逗号分隔的毫米值")
    return [float(value) for value in values]


def parse_xy(text: str, label: str) -> list[float]:
    values = [part.strip() for part in text.replace("，", ",").split(",")]
    if len(values) != 2:
        raise ValueError(f"{label}需要两个逗号分隔的毫米值")
    return [float(value) for value in values]


class AcquireThread(QThread):
    message = pyqtSignal(str)
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, runner: StageRunner):
        super().__init__()
        self.runner = runner
        self.runner.log = self.message.emit

    def run(self):
        try:
            output = self.runner.run()
            self.succeeded.emit(str(output))
        except BaseException as exc:
            self.failed.emit(str(exc))

    def request_stop(self):
        self.runner.request_stop()


class SceneApplyThread(QThread):
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, scene):
        super().__init__()
        self.scene = scene

    def run(self):
        try:
            self.succeeded.emit(apply_scene_to_moveit(self.scene, 35.0))
        except BaseException as exc:
            self.failed.emit(str(exc))


class Workbench(QMainWindow):
    jogConnectRequested = pyqtSignal()
    jogRefreshRequested = pyqtSignal()
    jogStopRequested = pyqtSignal()
    jogAxisRequested = pyqtSignal(str, float, dict)
    jogShutdownRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        sanitize_moveit_runtime_environment()
        self.setWindowTitle("CR5 五向测针标准球实验台")
        self.resize(980, 780)
        self.feedback = None
        self.last_di = None
        self.off_since = None
        self.armed = False
        self.latest_snapshot = None
        self.acquire_worker = None
        self.moveit_process = None
        self.moveit_log_handle = None
        self.moveit_profile_hash = None
        self.scene_worker = None
        self.profile_path = DEFAULT_PROFILE
        self.profile = load_profile(self.profile_path)
        self.sim_dialog = None
        self._build_ui()
        self._setup_jog_worker()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_feedback)
        self.timer.start(50)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(800)
        self.refresh_timer.timeout.connect(self.jogRefreshRequested.emit)
        self.refresh_timer.start()
        self.moveit_timer = QTimer(self)
        self.moveit_timer.setInterval(1000)
        self.moveit_timer.timeout.connect(self.poll_moveit)
        self.moveit_timer.start()
        self.workflow_timer = QTimer(self)
        self.workflow_timer.setInterval(1000)
        self.workflow_timer.timeout.connect(self.update_workflow_status)
        self.workflow_timer.start()

    def _build_ui(self):
        content = QWidget()
        self.setCentralWidget(content)
        layout = QVBoxLayout(content)

        top = QHBoxLayout()
        self.connect_button = QPushButton("连接机器人")
        self.connect_button.clicked.connect(self.connect_feedback)
        self.refresh_button = QPushButton("刷新状态")
        self.refresh_button.clicked.connect(self.refresh_robot)
        self.emergency_button = QPushButton("软件停止 / STOP")
        self.emergency_button.setStyleSheet("background:#c92a2a;color:white;font-weight:bold;padding:8px")
        self.emergency_button.clicked.connect(self.emergency_stop)
        self.status = QLabel("未连接")
        self.live_pose = QLabel("XYZ/RXYZ: --")
        top.addWidget(self.connect_button)
        top.addWidget(self.refresh_button)
        top.addWidget(self.emergency_button)
        top.addWidget(self.status, 1)
        top.addWidget(self.live_pose, 2)
        layout.addLayout(top)

        workflow_box = QGroupBox("推荐执行流程（复用现有五向标准球测量逻辑）")
        workflow_layout = QVBoxLayout(workflow_box)
        workflow_buttons = QHBoxLayout()
        self.coarse_flow_button = QPushButton("① 启动仿真并连接真机：粗定位")
        self.coarse_flow_button.clicked.connect(self.start_coarse_workflow)
        self.precision_flow_button = QPushButton("② 粗球心后：精标定与正式采集")
        self.precision_flow_button.clicked.connect(self.start_precision_workflow)
        workflow_buttons.addWidget(self.coarse_flow_button)
        workflow_buttons.addWidget(self.precision_flow_button)
        workflow_layout.addLayout(workflow_buttons)
        self.workflow_status = QLabel("正在检查当前球位数据…")
        self.workflow_status.setWordWrap(True)
        workflow_layout.addWidget(self.workflow_status)
        layout.addWidget(workflow_box)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        model_page = QWidget()
        model_layout = QVBoxLayout(model_page)
        model_scroll = QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_content = QWidget()
        form = QFormLayout(model_content)
        self.session_edit = QLineEdit(str(ROOT / "data/runs" / date.today().isoformat() / "five_branch_workbench"))
        self.install_edit = QLineEdit()
        self.install_edit.setPlaceholderText("例：CR5_A_base2_probe3；重装后必须新编号")
        self.sphere_id_edit = QLineEdit("standard_sphere_S1")
        self.placement_edit = QLineEdit("P0")
        self.placement_edit.setPlaceholderText("自由输入，例如 P0、P12、train_A01、locked_test_02")
        self.dataset_split_combo = QComboBox()
        self.dataset_split_combo.addItem("先选择球位用途", "")
        for label, value in (("训练 train", "train"), ("验证 validation", "validation"),
                             ("锁定测试 locked_test", "locked_test"), ("诊断 diagnostic", "diagnostic")):
            self.dataset_split_combo.addItem(label, value)
        self.placement_edit.textChanged.connect(self._placement_changed)
        self.ip_edit = QLineEdit()
        self.profile_edit = QLineEdit(str(self.profile_path))
        profile_buttons = QHBoxLayout()
        for label, callback in (("重新载入", self.load_profile_from_path), ("保存", self.save_profile_current),
                                ("另存为", self.save_profile_as), ("恢复历史预填", self.restore_default_profile)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            profile_buttons.addWidget(button)
        profile_widget = QWidget(); profile_widget.setLayout(profile_buttons)
        self.table_size_edit = QLineEdit()
        self.robot_base_xyz_edit = QLineEdit()
        self.robot_base_rxyz_edit = QLineEdit()
        self.base_center_edit = QLineEdit()
        self.base_size_edit = QLineEdit()
        self.stem_diameter_edit = QLineEdit()
        self.stem_length_edit = QLineEdit()
        self.sphere_diameter_edit = QLineEdit()
        self.connector_size_edit = QLineEdit()
        self.probe_body_size_edit = QLineEdit()
        self.center_stem_size_edit = QLineEdit()
        self.probe_ball_diameter_edit = QLineEdit()
        self.tool_mount_xyz_edit = QLineEdit()
        self.tool_mount_rxyz_edit = QLineEdit()
        self.radius_edit = QLineEdit()
        self.probe_radius_edit = QLineEdit()
        self.radius_edit.hide(); self.probe_radius_edit.hide()
        self.frame_check = QCheckBox("已核对当前User0/Tool0、实物安装和现场障碍")
        session_row = QWidget()
        session_layout = QHBoxLayout(session_row)
        session_layout.setContentsMargins(0, 0, 0, 0)
        session_layout.addWidget(self.session_edit, 1)
        choose_session = QPushButton("选择…")
        choose_session.clicked.connect(self.choose_session_dir)
        session_layout.addWidget(choose_session)
        form.addRow("运行目录", session_row)
        form.addRow("安装编号", self.install_edit)
        form.addRow("标准球编号", self.sphere_id_edit)
        form.addRow("球位编号", self.placement_edit)
        form.addRow("球位预注册用途", self.dataset_split_combo)
        form.addRow("机器人IP", self.ip_edit)
        form.addRow("配置文件", self.profile_edit)
        form.addRow(profile_widget)
        hint = QLabel("台面原点=台面顶面中心；机械臂位姿相对台面，磁座中心直接使用base_link坐标。历史预填值必须按当前实物确认。")
        hint.setStyleSheet("color:#b26a00")
        form.addRow(hint)
        form.addRow("台面长,宽 (mm)", self.table_size_edit)
        form.addRow("机械臂基座相对台面 XYZ (mm)", self.robot_base_xyz_edit)
        form.addRow("机械臂基座相对台面 RXYZ (deg)", self.robot_base_rxyz_edit)
        form.addRow("磁座中心（base_link XYZ, mm）", self.base_center_edit)
        form.addRow("磁座尺寸XYZ", self.base_size_edit)
        form.addRow("球杆直径", self.stem_diameter_edit)
        form.addRow("球杆长度", self.stem_length_edit)
        form.addRow("标准球直径", self.sphere_diameter_edit)
        form.addRow("法兰连接件 直径,长度", self.connector_size_edit)
        form.addRow("测头 直径,长度", self.probe_body_size_edit)
        form.addRow("中心支撑杆 直径,长度", self.center_stem_size_edit)
        form.addRow("红宝石测球直径", self.probe_ball_diameter_edit)
        form.addRow("工具安装XYZ", self.tool_mount_xyz_edit)
        form.addRow("工具安装RXYZ", self.tool_mount_rxyz_edit)
        form.addRow(self.frame_check)
        model_scroll.setWidget(model_content)
        model_layout.addWidget(model_scroll)
        self.model_preview_button = QPushButton("生成/刷新 RViz 仿真模型")
        self.model_preview_button.clicked.connect(self.open_simulation)
        moveit_row = QHBoxLayout()
        moveit_row.addWidget(self.model_preview_button)
        self.moveit_stop_button = QPushButton("关闭 RViz / MoveIt 仿真")
        self.moveit_stop_button.clicked.connect(self.stop_moveit)
        self.moveit_stop_button.setEnabled(False)
        moveit_row.addWidget(self.moveit_stop_button)
        model_layout.addLayout(moveit_row)
        self.tabs.addTab(model_page, "1 模型配置")

        manual_page = QWidget()
        manual_layout = QVBoxLayout(manual_page)
        manual_box = QGroupBox("五分支触发与粗球心")
        grid = QGridLayout(manual_box)
        self.branch_combo = QComboBox()
        self.branch_combo.addItems(BRANCHES)
        self.branch_combo.currentIndexChanged.connect(self.disarm_contact)
        self.arm_button = QPushButton("等待当前分支下一次触发")
        self.arm_button.clicked.connect(self.arm_contact)
        self.safe_button = QPushButton("记录当前释放安全位")
        self.safe_button.clicked.connect(self.record_safe_pose)
        self.fit_button = QPushButton("拟合粗球心")
        self.fit_button.clicked.connect(self.fit_center)
        self.save_button = QPushButton("保存当前球位数据")
        self.save_button.clicked.connect(self.save_current_data)
        self.counts = QLabel("手动点：0")
        grid.addWidget(self.branch_combo, 0, 0)
        grid.addWidget(self.arm_button, 0, 1)
        grid.addWidget(self.safe_button, 0, 2)
        grid.addWidget(self.fit_button, 0, 3)
        grid.addWidget(self.save_button, 0, 4)
        grid.addWidget(self.counts, 1, 0, 1, 5)
        manual_layout.addWidget(manual_box)
        jog_box = QGroupBox("XYZ / RXYZ 单步移动（DI1触发即停并回退）")
        jog = QGridLayout(jog_box)
        self.linear_step_spin = QDoubleSpinBox(); self.linear_step_spin.setRange(0.01, 10); self.linear_step_spin.setDecimals(3)
        self.angular_step_spin = QDoubleSpinBox(); self.angular_step_spin.setRange(0.01, 5); self.angular_step_spin.setDecimals(3)
        self.speed_spin = QSpinBox(); self.speed_spin.setRange(1, 5)
        self.retract_spin = QDoubleSpinBox(); self.retract_spin.setRange(0.05, 10); self.retract_spin.setDecimals(3)
        jog.addWidget(QLabel("平移步长mm"), 0, 0); jog.addWidget(self.linear_step_spin, 0, 1)
        jog.addWidget(QLabel("旋转步长deg"), 0, 2); jog.addWidget(self.angular_step_spin, 0, 3)
        jog.addWidget(QLabel("速度级别"), 0, 4); jog.addWidget(self.speed_spin, 0, 5)
        jog.addWidget(QLabel("触发回退mm"), 0, 6); jog.addWidget(self.retract_spin, 0, 7)
        self.jog_buttons = []
        for index, (label, axis, sign) in enumerate([
            ("X-", "x", -1), ("X+", "x", 1), ("Y-", "y", -1), ("Y+", "y", 1),
            ("Z-", "z", -1), ("Z+", "z", 1), ("RX-", "rx", -1), ("RX+", "rx", 1),
            ("RY-", "ry", -1), ("RY+", "ry", 1), ("RZ-", "rz", -1), ("RZ+", "rz", 1),
        ]):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, a=axis, s=sign: self.jog_axis(a, s))
            self.jog_buttons.append(button)
            jog.addWidget(button, 1 + index // 4, index % 4 * 2, 1, 2)
        manual_layout.addWidget(jog_box)
        manual_layout.addStretch(1)
        self.tabs.addTab(manual_page, "2 手动定位")

        auto_page = QWidget()
        auto_layout = QVBoxLayout(auto_page)
        auto_box = QGroupBox("仿真路径与全自动五分支采集")
        row = QHBoxLayout(auto_box)
        self.sim_button = QPushButton("仿真并冻结当前阶段路径")
        self.sim_button.clicked.connect(self.open_simulation)
        self.input_position_button = QPushButton("按输入球位自动规划（沿用当前安装几何）")
        self.input_position_button.clicked.connect(self.plan_from_input_position)
        self.plan_button = QPushButton("高级：直接生成正式计划")
        self.plan_button.clicked.connect(self.generate_plans)
        self.plan_button.hide()
        self.geometry_button = QPushButton("拟合P0新安装几何")
        self.geometry_button.clicked.connect(self.fit_geometry)
        self.anchor_button = QPushButton("拟合独立锚点球心")
        self.anchor_button.clicked.connect(self.fit_anchor)
        self.stage_combo = QComboBox()
        for key, label in STAGE_LABELS.items():
            self.stage_combo.addItem(label, key)
        self.run_button = QPushButton("点击自动采集所选批次")
        self.run_button.clicked.connect(self.start_acquisition)
        self.stop_button = QPushButton("停止采集")
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)
        for widget in (self.sim_button, self.input_position_button, self.plan_button, self.geometry_button, self.anchor_button,
                       self.stage_combo, self.run_button, self.stop_button):
            row.addWidget(widget)
        auto_layout.addWidget(auto_box)
        self.workflow_label = QLabel("流程：手动粗定位 → 仿真/冻结 → 冒烟 → P0几何 → 锚点 → 五分支评价")
        self.workflow_label.setWordWrap(True)
        auto_layout.addWidget(self.workflow_label)
        auto_layout.addStretch(1)
        self.tabs.addTab(auto_page, "3 仿真与自动采集")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(170)
        layout.addWidget(self.log_box)
        self.apply_profile_to_ui(self.profile)
        self._placement_changed()
        self.update_workflow_status()
        self.log("先在示教器手动触测：每分支至少2次分散触点；释放后记录该分支的安全位。")
        self.log("当前批次不会自动沿用2026-07球心、测针偏置或历史绝对安全位姿。")

    @staticmethod
    def _fmt(values):
        return ",".join(f"{float(value):g}" for value in values)

    def apply_profile_to_ui(self, profile):
        profile = validate_profile(profile)
        self.profile = profile
        robot, workcell, tool, jog = profile["robot"], profile["workcell"], profile["tool"], profile["jog"]
        table, base = workcell["table"], workcell["magnetic_base"]
        stem, sphere = workcell["sphere_stem"], workcell["standard_sphere"]
        self.ip_edit.setText(robot["ip"])
        self.table_size_edit.setText(self._fmt(table["size_xy_mm"]))
        self.robot_base_xyz_edit.setText(self._fmt(robot["base_in_table_pose_mm_deg"][:3]))
        self.robot_base_rxyz_edit.setText(self._fmt(robot["base_in_table_pose_mm_deg"][3:]))
        self.base_center_edit.setText(self._fmt(base["center_base_mm"]))
        self.base_size_edit.setText(self._fmt(base["size_xyz_mm"]))
        self.stem_diameter_edit.setText(f"{stem['diameter_mm']:g}")
        self.stem_length_edit.setText(f"{stem['length_mm']:g}")
        self.sphere_diameter_edit.setText(f"{sphere['diameter_mm']:g}")
        self.connector_size_edit.setText(self._fmt([tool["connector"]["diameter_mm"], tool["connector"]["length_mm"]]))
        self.probe_body_size_edit.setText(self._fmt([tool["probe_body"]["diameter_mm"], tool["probe_body"]["length_mm"]]))
        self.center_stem_size_edit.setText(self._fmt([tool["center_stem"]["diameter_mm"], tool["center_stem"]["length_mm"]]))
        self.probe_ball_diameter_edit.setText(f"{tool['ruby_ball_diameter_mm']:g}")
        self.tool_mount_xyz_edit.setText(self._fmt(tool["mount_xyz_mm"]))
        self.tool_mount_rxyz_edit.setText(self._fmt(tool["mount_rxyz_deg"]))
        self.linear_step_spin.setValue(jog["linear_step_mm"])
        self.angular_step_spin.setValue(jog["angular_step_deg"])
        self.speed_spin.setValue(jog["speed"])
        self.retract_spin.setValue(jog["auto_retract_mm"])
        self.radius_edit.setText(f"{sphere['diameter_mm'] / 2:g}")
        self.probe_radius_edit.setText(f"{tool['ruby_ball_diameter_mm'] / 2:g}")

    def profile_from_ui(self):
        profile = json.loads(json.dumps(self.profile))
        profile["robot"]["ip"] = self.ip_edit.text().strip()
        profile["robot"]["base_in_table_pose_mm_deg"] = [
            *parse_xyz(self.robot_base_xyz_edit.text(), "机械臂基座相对台面XYZ"),
            *parse_xyz(self.robot_base_rxyz_edit.text(), "机械臂基座相对台面RXYZ"),
        ]
        table = profile["workcell"]["table"]
        table["size_xy_mm"] = parse_xy(self.table_size_edit.text(), "台面长宽")
        base = profile["workcell"]["magnetic_base"]
        base["center_base_mm"] = parse_xyz(self.base_center_edit.text(), "磁座base_link坐标")
        base["size_xyz_mm"] = parse_xyz(self.base_size_edit.text(), "磁座尺寸")
        stem = profile["workcell"]["sphere_stem"]
        stem["diameter_mm"] = float(self.stem_diameter_edit.text())
        stem["length_mm"] = float(self.stem_length_edit.text())
        sphere = profile["workcell"]["standard_sphere"]
        sphere["diameter_mm"] = float(self.sphere_diameter_edit.text())
        connector = parse_xy(self.connector_size_edit.text(), "法兰连接件直径和长度")
        body = parse_xy(self.probe_body_size_edit.text(), "测头直径和长度")
        support = parse_xy(self.center_stem_size_edit.text(), "中心支撑杆直径和长度")
        tool = profile["tool"]
        tool["connector"] = {"diameter_mm": connector[0], "length_mm": connector[1]}
        tool["probe_body"] = {"diameter_mm": body[0], "length_mm": body[1]}
        tool["center_stem"] = {"diameter_mm": support[0], "length_mm": support[1]}
        tool["ruby_ball_diameter_mm"] = float(self.probe_ball_diameter_edit.text())
        tool["mount_xyz_mm"] = parse_xyz(self.tool_mount_xyz_edit.text(), "工具安装XYZ")
        tool["mount_rxyz_deg"] = parse_xyz(self.tool_mount_rxyz_edit.text(), "工具安装RXYZ")
        jog = profile["jog"]
        jog.update({
            "linear_step_mm": self.linear_step_spin.value(), "angular_step_deg": self.angular_step_spin.value(),
            "speed": self.speed_spin.value(), "auto_retract_mm": self.retract_spin.value(),
        })
        checked = validate_profile(profile)
        self.radius_edit.setText(f"{checked['workcell']['standard_sphere']['diameter_mm'] / 2:g}")
        self.probe_radius_edit.setText(f"{checked['tool']['ruby_ball_diameter_mm'] / 2:g}")
        return checked

    def register_profile_to_fit(self, profile, fit, *, update_ui=True):
        """Align the fixture to the fitted sphere using the current dimensions."""
        usable, reason = fit_usable_for_motion(fit)
        if not usable:
            raise ValueError("拟合未通过，禁止改写刚性夹具位置：" + reason)
        validate_fixture_fit_height(fit, profile)
        aligned = register_fixture_to_sphere(profile, fit["sphere_center_mm"])
        findings = scene_checks(scene_from_profile(aligned, fit))
        if len(findings) != 1 or not findings[0].startswith("几何自检通过"):
            raise ValueError("拟合夹具与台面几何不一致：" + "；".join(findings))
        if update_ui:
            center = aligned["workcell"]["magnetic_base"]["center_base_mm"]
            self.base_center_edit.setText(self._fmt(center))
            self.profile = aligned
        return aligned

    def load_profile_from_path(self):
        try:
            path = Path(self.profile_edit.text().strip()).expanduser().resolve()
            self.profile_path = path
            self.apply_profile_to_ui(load_profile(path))
            self.log(f"配置已重新载入：{path}")
        except Exception as exc:
            self.show_error(exc)

    def save_profile_current(self):
        try:
            path = Path(self.profile_edit.text().strip()).expanduser().resolve()
            self.profile = self.profile_from_ui()
            save_profile(path, self.profile)
            self.profile_path = path
            self.log(f"配置已保存：{path}；SHA256={profile_sha256(self.profile)[:12]}")
        except Exception as exc:
            self.show_error(exc)

    def save_profile_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "另存工作台配置", str(self.profile_path), "JSON (*.json)")
        if path:
            self.profile_edit.setText(path)
            self.save_profile_current()

    def restore_default_profile(self):
        try:
            self.profile_path = DEFAULT_PROFILE
            self.profile_edit.setText(str(DEFAULT_PROFILE))
            self.apply_profile_to_ui(load_profile(DEFAULT_PROFILE))
            self.log("已恢复工程历史预填；这些数值仍需按当前实物确认")
        except Exception as exc:
            self.show_error(exc)

    @staticmethod
    def _m(value):
        return f"{float(value) / 1000.0:.9f}"

    def _moveit_command(self, profile, use_rviz=True):
        tool = profile["tool"]
        geometry = load_geometry(DEFAULT_GEOMETRY)
        offsets = self._planning_offsets() or {branch: branch_local_offset_mm(geometry, branch) for branch in BRANCHES}
        # The thin axial stylus beyond the centre support remains fixed by the
        # five-branch geometry. The editable support length shifts that tip.
        nominal_support = float(geometry["stylus"]["cross_hub_offset_from_branch_origin_mm"])
        nominal_axial = float(geometry["branches"][0]["nominal_offset_mm"][2])
        thin_axial = nominal_axial - nominal_support
        stylus_length = float(tool["center_stem"]["length_mm"]) + thin_axial
        mount_xyz = " ".join(self._m(value) for value in tool["mount_xyz_mm"])
        mount_rpy = " ".join(f"{math.radians(float(value)):.9f}" for value in tool["mount_rxyz_deg"])
        command = [
            "ros2", "launch", str(ROOT / "launch/cr5_rmp60_moveit_demo.launch.py"),
            f"use_rviz:={'true' if use_rviz else 'false'}",
            f"adapter_length:={self._m(tool['connector']['length_mm'])}",
            f"adapter_radius:={self._m(tool['connector']['diameter_mm'] / 2)}",
            f"probe_body_length:={self._m(tool['probe_body']['length_mm'])}",
            f"probe_body_radius:={self._m(tool['probe_body']['diameter_mm'] / 2)}",
            f"center_stem_length:={self._m(tool['center_stem']['length_mm'])}",
            f"center_stem_radius:={self._m(tool['center_stem']['diameter_mm'] / 2)}",
            f"stylus_length:={self._m(stylus_length)}",
            f"stylus_ball_radius:={self._m(tool['ruby_ball_diameter_mm'] / 2)}",
            f"mount_xyz:={mount_xyz}", f"mount_rpy:={mount_rpy}",
        ]
        for branch in ("x_pos", "x_neg", "y_pos", "y_neg"):
            for axis, value in zip(("x", "y", "z"), offsets[branch]):
                command.append(f"{branch}_{axis}:={self._m(value)}")
        return command

    def _ensure_moveit(self, profile):
        profile_hash = profile_sha256(profile)
        current_offsets = self._planning_offsets()
        if self.moveit_process is not None and self.moveit_process.poll() is None:
            if self.moveit_profile_hash != profile_hash or getattr(self, "moveit_tool_offsets", None) != current_offsets:
                raise ValueError("模型参数已变化。请先关闭当前RViz/MoveIt，再重新生成模型")
            return
        existing = subprocess.run(
            ["ros2", "service", "list"], text=True, capture_output=True, timeout=4, check=False,
        )
        if "/compute_ik" in existing.stdout:
            raise ValueError("检测到其他MoveIt实例，无法保证RViz使用当前测头模型；请先关闭旧move_group")
        log_path = Path("/tmp/cr5_rmp60_workbench_moveit.log")
        self.moveit_log_handle = log_path.open("a")
        self.moveit_process = subprocess.Popen(
            self._moveit_command(profile, use_rviz=True), cwd=ROOT, stdout=self.moveit_log_handle,
            stderr=subprocess.STDOUT, start_new_session=True, text=True,
        )
        self.moveit_profile_hash = profile_hash
        self.moveit_tool_offsets = current_offsets
        self.moveit_stop_button.setEnabled(True)
        self.log(f"正在启动RViz/MoveIt（PID {self.moveit_process.pid}）；日志：{log_path}")

    def stop_moveit(self):
        process = self.moveit_process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self.moveit_log_handle is not None:
            self.moveit_log_handle.close()
            self.moveit_log_handle = None
        self.moveit_process = None
        self.moveit_profile_hash = None
        self.moveit_stop_button.setEnabled(False)
        self.log("本软件启动的 MoveIt 已停止")

    def poll_moveit(self):
        process = self.moveit_process
        if process is None or process.poll() is None:
            return
        code = process.returncode
        if self.moveit_log_handle is not None:
            self.moveit_log_handle.close()
            self.moveit_log_handle = None
        self.moveit_process = None
        self.moveit_profile_hash = None
        self.moveit_stop_button.setEnabled(False)
        self.log(f"MoveIt进程已退出，返回码={code}；请查看 /tmp/cr5_rmp60_workbench_moveit.log")

    def choose_session_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择实验数据目录", self.session_edit.text().strip())
        if path:
            self.session_edit.setText(path)
            self.log(f"数据目录已选择：{path}")

    def _setup_jog_worker(self):
        self.jog_thread = QThread(self)
        self.jog_worker = RosProbeWorker()
        self.jog_worker.moveToThread(self.jog_thread)
        self.jogConnectRequested.connect(self.jog_worker.connect_robot)
        self.jogRefreshRequested.connect(self.jog_worker.refresh)
        self.jogStopRequested.connect(self.jog_worker.stop)
        self.jogAxisRequested.connect(self.jog_worker.jog_axis)
        self.jogShutdownRequested.connect(self.jog_worker.shutdown)
        self.jog_worker.status.connect(lambda level, message: self.log(f"{level}: {message}"))
        self.jog_worker.poseUpdated.connect(self.on_jog_pose)
        self.jog_worker.sampleRecorded.connect(self.on_jog_sample)
        self.jog_worker.busyChanged.connect(self.on_jog_busy)
        self.jog_thread.start()

    def jog_settings(self):
        profile = self.profile_from_ui()
        jog = profile["jog"]
        output = self.session() / f"{self.placement()}_jog_contacts.csv"
        self.jog_worker.sample_index = existing_max_sample_index(output)
        return {
            "output": str(output), "session_id": self.install_edit.text().strip() + "_" + self.placement(),
            "workpiece_id": self.sphere_id_edit.text().strip(), "artifact_id": self.sphere_id_edit.text().strip(),
            "artifact_type": "sphere", "physical_ball_id": self.install_edit.text().strip(),
            "branch": self.branch_combo.currentText(), "operator_note": "unified workbench manual jog",
            "linear_step_mm": jog["linear_step_mm"], "angular_step_deg": jog["angular_step_deg"],
            "speed": jog["speed"], "auto_retract_mm": jog["auto_retract_mm"],
            "timeout_sec": jog["timeout_sec"], "position_tolerance_mm": jog["position_tolerance_mm"],
            "orientation_tolerance_deg": jog["orientation_tolerance_deg"], "normal_fit_json": "",
            "normal_search_mm": 0.2, "normal_ack_safe": False, "normal_euler_sequence": "xyz", "sample_index": 0,
        }

    def jog_axis(self, axis, sign):
        try:
            if self.acquire_worker is not None and self.acquire_worker.isRunning():
                raise ValueError("自动采集进行中，禁止手动点动")
            if not self.install_edit.text().strip():
                raise ValueError("先填写安装编号")
            if axis.startswith("r") and self.angular_step_spin.value() > 0.2:
                raise ValueError("靠近标准球时旋转步长不得超过0.2°")
            self.disarm_contact()
            self.jogAxisRequested.emit(axis, float(sign), self.jog_settings())
        except Exception as exc:
            self.show_error(exc)

    def on_jog_pose(self, pose, di1, joints, feed_age):
        self.live_pose.setText("XYZ/RXYZ: " + ", ".join(f"{float(value):.3f}" for value in pose))
        self.status.setText(f"ROS反馈 DI1={int(di1)}，年龄={feed_age:.3f}s")
        self.latest_snapshot = {"pose": list(pose), "joints": list(joints or [0] * 6), "di1": bool(di1)}
        if self.feedback is None:
            now = time.monotonic()
            current_di = bool(di1)
            if not current_di and self.last_di is not False:
                self.off_since = now
            self.last_di = current_di

    def on_jog_sample(self, row):
        try:
            record = {
                "kind": "manual_trigger", "installation_id": self.install_edit.text().strip(),
                "standard_sphere_id": self.sphere_id_edit.text().strip(), "placement_id": self.placement(),
                "branch": row.get("branch", self.branch_combo.currentText()),
                "pose": [float(row[f"trigger_flange_{name}"]) for name in ("x", "y", "z", "rx", "ry", "rz")],
                "joints": [float(row.get(f"trigger_joint_{index}", 0) or 0) for index in range(1, 7)],
                "feed_sequence": row.get("trigger_feed_sequence", ""),
                "digital_input_bits": row.get("trigger_digital_input_bits", ""), "di1": 1,
                "wall_time": row.get("trigger_feed_wall_time", time.time()),
                "pose_source": "ROS FeedInfo guarded jog", "robot_ip": self.ip_edit.text().strip(),
            }
            append_jsonl(self.manual_path(), record)
            self.update_counts()
            self.log(f"点动触发已纳入粗球心数据：{record['branch']}")
        except Exception as exc:
            self.show_error(exc)

    def on_jog_busy(self, busy):
        for button in self.jog_buttons:
            button.setEnabled(not busy)

    def refresh_robot(self):
        self.jogRefreshRequested.emit()
        self.log("已请求刷新机器人状态")

    def emergency_stop(self):
        if self.acquire_worker is not None and self.acquire_worker.isRunning():
            self.acquire_worker.request_stop()
        self.jogStopRequested.emit()
        ip = self.ip_edit.text().strip()
        def direct_stop():
            dashboard = None
            try:
                dashboard = Dashboard(ip)
                dashboard.stop()
            except Exception:
                pass
            finally:
                if dashboard is not None:
                    dashboard.close()
        threading.Thread(target=direct_stop, daemon=True).start()
        self.log("已请求软件Stop；控制柜物理急停仍是最终安全装置")

    def session(self) -> Path:
        raw = self.session_edit.text().strip()
        if not raw:
            raise ValueError("先选择实验数据目录")
        path = Path(raw).expanduser().resolve()
        if path == Path(path.anchor):
            raise ValueError("不能把系统根目录作为实验数据目录")
        if path.exists() and not path.is_dir():
            raise ValueError("实验数据路径指向文件，请选择目录")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def placement(self) -> str:
        return validate_placement_id(self.placement_edit.text())

    def _placement_changed(self):
        self.disarm_contact()
        try:
            split = dataset_split_for_placement(self.placement())
        except ValueError:
            split = ""
        index = self.dataset_split_combo.findData(split)
        self.dataset_split_combo.setCurrentIndex(max(index, 0))

    def dataset_split(self) -> str:
        return dataset_split_for_placement(self.placement(), self.dataset_split_combo.currentData())

    def manual_path(self) -> Path:
        return self.session() / f"{self.placement()}_manual.jsonl"

    def safe_path(self) -> Path:
        return self.session() / f"{self.placement()}_safe_poses.json"

    def log(self, message: str):
        self.log_box.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def show_error(self, exc):
        self.log(f"错误：{exc}")
        QMessageBox.critical(self, "操作未完成", str(exc))

    def connect_feedback(self):
        try:
            if self.feedback is not None:
                self.feedback.close()
            self.feedback = FeedbackReader(self.ip_edit.text().strip())
            self.feedback.start()
            self.jogConnectRequested.emit()
            self.log("只读反馈连接已启动；等待新鲜30004帧")
        except Exception as exc:
            self.show_error(exc)

    def disarm_contact(self):
        self.armed = False
        self.arm_button.setText("等待当前分支下一次触发")

    def poll_feedback(self):
        if self.feedback is None:
            return
        try:
            snap = self.feedback.snapshot(0.5)
        except RuntimeError as exc:
            self.status.setText(f"反馈失效：{exc}")
            self.last_di = None
            return
        now = time.monotonic()
        di = bool(snap["di1"])
        if not di and self.last_di is not False:
            self.off_since = now
        rising = di and self.last_di is False
        self.last_di = di
        self.latest_snapshot = snap
        self.status.setText(f"DI1={int(di)}  Pose={', '.join(f'{x:.2f}' for x in snap['pose'][:3])}  序号={snap['sequence']}")
        self.live_pose.setText("XYZ/RXYZ: " + ", ".join(f"{float(value):.3f}" for value in snap["pose"]))
        if rising and self.armed and self.off_since is not None and now - self.off_since >= 0.2:
            self.armed = False
            self.arm_button.setText("等待当前分支下一次触发")
            try:
                installation = self.install_edit.text().strip()
                if not installation:
                    raise ValueError("先填写当前安装编号")
                record = {
                    "kind": "manual_trigger", "installation_id": installation,
                    "standard_sphere_id": self.sphere_id_edit.text().strip(),
                    "placement_id": self.placement(), "branch": self.branch_combo.currentText(),
                    "pose": snap["pose"], "joints": snap["joints"],
                    "feed_sequence": snap["sequence"], "controller_timer": snap["controller_timer"],
                    "digital_input_bits": snap["digital_input_bits"], "di1": int(di),
                    "wall_time": snap["wall_time"], "pose_source": "30004 ToolVectorActual",
                    "robot_ip": self.feedback.ip,
                }
                append_jsonl(self.manual_path(), record)
                self.log(f"手动触发已记录：{record['branch']} / feed {record['feed_sequence']}")
                self.update_counts()
            except Exception as exc:
                self.show_error(exc)

    def update_counts(self):
        rows = read_jsonl(self.manual_path())
        summary = ", ".join(f"{branch}:{sum(row.get('branch') == branch for row in rows)}" for branch in BRANCHES)
        self.counts.setText(summary)
        self.update_workflow_status()

    def update_workflow_status(self):
        """Show progress without changing files or authorizing motion."""
        try:
            raw_session = self.session_edit.text().strip()
            placement = validate_placement_id(self.placement_edit.text())
            if not raw_session:
                self.workflow_status.setText("下一步：先选择运行目录并填写安装编号。")
                return
            session = Path(raw_session).expanduser()
            manual = read_jsonl(session / f"{placement}_manual.jsonl")
            counts = {branch: sum(row.get("branch") == branch for row in manual) for branch in BRANCHES}
            safe_path = session / f"{placement}_safe_poses.json"
            safe = json.loads(safe_path.read_text()).get("poses", {}) if safe_path.exists() else {}
            coarse = session / f"{placement}_manual_coarse_preview.json"
            coarse_fit = json.loads(coarse.read_text()) if coarse.exists() else None
            coarse_ok = fit_usable_for_motion(coarse_fit)[0] if coarse_fit else False
            if coarse_ok:
                try:
                    validate_fixture_fit_height(coarse_fit, self.profile_from_ui())
                except ValueError:
                    coarse_ok = False
            coarse_stale = bool(coarse_ok and coarse_fit.get("method") != "configured_fixture_center" and coarse_fit.get("source_manual_sha256") !=
                                sha256(session / f"{placement}_manual.jsonl"))
            if coarse_stale:
                coarse_ok = False
            geometry = session / "P0_geometry_fit.json"
            if placement != "P0" and coarse_ok:
                coarse_ok = (
                    geometry.exists() and coarse_fit.get("geometry_fit_sha256") == sha256(geometry)
                    and abs(float(coarse_fit.get("fixed_sphere_center_z_mm", math.inf))
                            - fixture_sphere_height_base_mm(self.profile_from_ui())) <= 1e-6
                )
            anchor = session / f"{placement}_anchor_fit.json"
            manifest_path = session / f"{placement}_manifest.json"
            plans = []
            if manifest_path.exists():
                plans = list(json.loads(manifest_path.read_text()).get("plans", {}))
            count_text = "、".join(f"{branch}={counts[branch]}" for branch in BRANCHES)
            if placement != "P0" and not geometry.exists():
                recommendation = "下一步：先完成本次安装的P0五分支几何标定。"
            elif placement != "P0" and counts["z"] < 3 and not coarse_ok:
                recommendation = "下一步：用Z针触发至少3处不同球面位置。"
            elif placement == "P0" and any(counts[branch] < 2 for branch in BRANCHES):
                recommendation = "下一步：每分支取得至少2个分散触点，并在释放状态记录安全位。"
            elif not coarse.exists():
                recommendation = "下一步：点击“拟合粗球心”。"
            elif coarse_stale:
                recommendation = "下一步：手动触点已增加，重新点击“拟合粗球心”以更新来源哈希。"
            elif not coarse_ok:
                recommendation = "下一步：粗拟合质量不合格，核对TCP/法兰语义、测针偏置和触点；当前结果不能刷新RViz或规划。"
            elif placement != "P0" and any(counts[branch] < 1 for branch in BRANCHES):
                recommendation = "下一步：打开仿真，自动搜索五针姿态和安全位，再做整批连续碰撞检查。"
            elif placement == "P0" and not geometry.exists():
                recommendation = "下一步：打开精测规划，冻结并执行冒烟与P0几何批次。"
            elif not anchor.exists():
                recommendation = "下一步：生成锚点批次；采集后拟合独立锚点球心。"
            elif "evaluation" not in plans:
                recommendation = "下一步：用独立锚点球心生成五分支评价路径。"
            else:
                recommendation = "下一步：在RViz复核冻结路径，然后执行所选正式批次。"
            self.workflow_status.setText(
                f"球位 {placement}｜粗触点：{count_text}｜粗球心：{'完成' if coarse_ok else '未通过' if coarse.exists() else '未完成'}｜"
                f"安全位：{len(safe)}/5｜P0几何：{'完成' if geometry.exists() else '未完成'}｜"
                f"锚点球心：{'完成' if anchor.exists() else '未完成'}｜冻结批次：{','.join(plans) or '无'}\n{recommendation}"
            )
        except Exception as exc:
            self.workflow_status.setText("流程状态暂不可用：" + str(exc))

    def start_coarse_workflow(self):
        """Open the entered model and connect feedback; never command motion."""
        try:
            if not self.install_edit.text().strip():
                raise ValueError("先填写当前安装编号；底座或测针重装后必须使用新编号")
            self.profile = self.profile_from_ui()
            if not self.open_simulation():
                return
            self.connect_feedback()
            self.tabs.setCurrentIndex(1)
            self.log("粗定位流程已启动：RViz仅显示输入模型；请逐分支触发并记录释放安全位。")
            self.update_workflow_status()
        except Exception as exc:
            self.show_error(exc)

    def start_precision_workflow(self):
        """Open current-fit planning; path generation remains non-executing."""
        try:
            _, fit, _, enabled, _ = self._planning_context(require_fit=True)
            usable, reason = fit_usable_for_motion(fit)
            if not enabled or not usable:
                raise ValueError("粗球心尚未达到精测路径门槛：" + reason)
            if not self.open_simulation():
                return
            self.tabs.setCurrentIndex(2)
            self.log("精标定流程已打开：生成路径→RViz显示→MoveIt检查→冻结；冻结前不会真机运动。")
            self.update_workflow_status()
        except Exception as exc:
            self.show_error(exc)

    def save_current_data(self):
        try:
            output, count = export_manual_contacts_csv(
                self.session(), self.placement(), self.install_edit.text().strip(),
                self.sphere_id_edit.text().strip(), self.ip_edit.text().strip(),
            )
            self.log(f"已保存 {count} 条手动触发记录：{output}；原始JSONL保留在同一目录")
            QMessageBox.information(self, "保存完成", f"已保存 {count} 条手动触发记录到：\n{output}")
        except Exception as exc:
            self.show_error(exc)

    def plan_from_input_position(self):
        """Explicit coordinate input keeps the calibrated installation, never fabricates hits."""
        try:
            profile = self.profile_from_ui()
            require_supported_contact_geometry(profile)
            self._geometry_offsets()  # Validates installation, sphere, model and fixed height.
            geometry_path = self.session() / "P0_geometry_fit.json"
            if not geometry_path.exists():
                raise ValueError("先完成本次安装的P0几何标定，再按输入球位自动规划")
            if self.placement() == "P0" or (self.session() / f"{self.placement()}_manifest.json").exists():
                raise ValueError("按输入坐标移动标准球时请使用新的球位编号，保留P0安装标定和已有冻结计划")
            from workbench_profile import derived_fixture_base
            center = derived_fixture_base(profile)["sphere_center"]
            sphere_radius = profile["workcell"]["standard_sphere"]["diameter_mm"] / 2
            probe_radius = profile["tool"]["ruby_ball_diameter_mm"] / 2
            fit = {"schema_version": 1, "method": "configured_fixture_center",
                "sphere_center_mm": list(center), "fixed_sphere_center_z_mm": center[2],
                "sphere_radius_mm": sphere_radius, "probe_radius_mm": probe_radius,
                "contact_center_distance_mm": sphere_radius + probe_radius,
                "installation_id": self.install_edit.text().strip(),
                "standard_sphere_id": self.sphere_id_edit.text().strip(), "placement_id": self.placement(),
                "geometry_fit_sha256": sha256(geometry_path),
                "position_source": "operator_configured_fixture_coordinates",
                "warning": "输入坐标用于模型规划；未生成任何触发记录，误差必须处于实测运动验证范围内"}
            path = self.session() / f"{self.placement()}_manual_coarse_preview.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fit, ensure_ascii=False, indent=2) + "\n")
            self.open_simulation()
            self.log("按输入球位搜索五针姿态；正式冻结前仍须整批连续碰撞检查")
        except Exception as exc:
            self.show_error(exc)

    def _planning_context(self, require_fit=True):
        """Return the exact fit and stages that may be previewed/frozen now."""
        placement = self.placement()
        session = self.session()
        preview_path = session / f"{placement}_manual_coarse_preview.json"
        profile = self.profile_from_ui()
        def input_model_only():
            sphere = profile["workcell"]["standard_sphere"]
            input_center = scene_from_profile(profile)["sphere_center"]
            fit = {
                "sphere_center_mm": list(input_center),
                "sphere_radius_mm": sphere["diameter_mm"] / 2.0,
                "probe_radius_mm": profile["tool"]["ruby_ball_diameter_mm"] / 2.0,
                "contact_center_distance_mm": (
                    sphere["diameter_mm"] + profile["tool"]["ruby_ball_diameter_mm"]
                ) / 2.0,
                "installation_id": self.install_edit.text().strip(),
                "standard_sphere_id": self.sphere_id_edit.text().strip(),
                "placement_id": placement,
                "method": "operator_input_model_only",
            }
            return profile, fit, (), False, []

        if not preview_path.exists():
            if require_fit:
                raise ValueError("先采集手动触点并点击“拟合粗球心”")
            return input_model_only()

        fit = json.loads(preview_path.read_text())
        try:
            require_supported_contact_geometry(profile)
            validate_fit_dimensions(fit, profile)
            validate_fixture_fit_height(fit, profile)
        except ValueError:
            if require_fit:
                raise
            return input_model_only()
        manual_path = self.manual_path()
        manual = read_jsonl(manual_path)
        if fit.get("method") != "configured_fixture_center" and fit.get("source_manual_sha256") != sha256(manual_path):
            raise ValueError("手动点在粗球心拟合后改变，请先重新拟合")
        if (fit.get("installation_id") != self.install_edit.text().strip() or
                fit.get("standard_sphere_id") != self.sphere_id_edit.text().strip() or
                fit.get("placement_id") != placement):
            raise ValueError("粗球心与当前安装、标准球或球位不匹配")
        usable, reason = fit_usable_for_motion(fit)
        if not usable:
            if require_fit:
                raise ValueError("粗球心不可用于仿真或路径：" + reason)
            return input_model_only()
        if any(row.get("robot_ip") != self.ip_edit.text().strip() for row in manual):
            raise ValueError("手动触点的机器人IP与当前界面不一致")

        geometry_fit = session / "P0_geometry_fit.json"
        anchor_fit_path = session / f"{placement}_anchor_fit.json"
        if placement != "P0" and not geometry_fit.exists():
            raise ValueError("先在P0完成本次安装的五分支相对几何标定")
        if placement != "P0":
            if fit.get("geometry_fit_sha256") != sha256(geometry_fit):
                raise ValueError("P0几何标定在三面Z针拟合后改变，请重新拟合当前球位")
            expected_z = fixture_sphere_height_base_mm(profile)
            if abs(float(fit.get("fixed_sphere_center_z_mm", math.inf)) - expected_z) > 1e-6:
                raise ValueError("磁座/球杆高度在三面Z针拟合后改变，请重新拟合当前球位")
        if placement == "P0" and not geometry_fit.exists():
            stages = ("smoke", "geometry")
        elif not anchor_fit_path.exists():
            stages = ("anchor",) if placement == "P0" else ("smoke", "anchor")
            if placement == "P0":
                geometry_result = json.loads(geometry_fit.read_text())
                validate_fixture_fit_height(geometry_result, profile)
                fit["sphere_center_mm"] = geometry_result["sphere_center_mm"]
                fit["planning_center_source"] = "P0_fixed_reference_branch_geometry_seed"
        else:
            stages = ("evaluation",)
            anchor_fit = json.loads(anchor_fit_path.read_text())
            contacts = session / f"{placement}_anchor_contacts.csv"
            if (anchor_fit.get("installation_id") != self.install_edit.text().strip() or
                    anchor_fit.get("standard_sphere_id") != self.sphere_id_edit.text().strip() or
                    anchor_fit.get("placement_id") != placement or
                    anchor_fit.get("source_sha256") != sha256(contacts)):
                raise ValueError("锚点球心与当前安装、球位或锚点触点不匹配")
            validate_fixture_fit_height(anchor_fit, profile)
            fit["sphere_center_mm"] = anchor_fit["sphere_center_mm"]
            fit["planning_center_source"] = "independent_A_anchor_seed_for_E"
        # Dimensions can be corrected after fitting. Re-registering here keeps
        # the fitted sphere, magnetic-base height and stem length consistent.
        profile = self.register_profile_to_fit(profile, fit)
        return profile, fit, stages, True, manual

    def _safe_record(self):
        path = self.safe_path()
        record = json.loads(path.read_text()) if path.exists() else {}
        if record and (record.get("installation_id") != self.install_edit.text().strip() or
                record.get("standard_sphere_id") != self.sphere_id_edit.text().strip() or
                record.get("placement_id") != self.placement() or
                record.get("robot_ip") != self.ip_edit.text().strip()):
            raise ValueError("安全位文件与当前安装、球位或机器人不匹配")
        return record

    def open_simulation(self):
        try:
            from sphere_simulation_dialog import SphereSimulationDialog

            profile, fit, stages, planning_enabled, manual = self._planning_context(require_fit=False)
            scene = scene_from_profile(
                profile, fit if planning_enabled else None,
                self.latest_snapshot.get("pose") if self.latest_snapshot else None,
            )
            self._ensure_moveit(profile)
            if self.scene_worker is not None and self.scene_worker.isRunning():
                raise ValueError("RViz场景正在刷新，请稍候")
            self.scene_worker = SceneApplyThread(scene)
            self.scene_worker.succeeded.connect(
                lambda message: self.log("RViz场景已刷新：" + (message or "完成"))
            )
            self.scene_worker.failed.connect(
                self.scene_load_failed
            )
            self.scene_worker.start()
            safe_record = self._safe_record()
            self.sim_dialog = SphereSimulationDialog(
                scene=scene, fit=fit, placement=self.placement(), manual=manual,
                safe_poses=safe_record.get("poses", {}), geometry_path=DEFAULT_GEOMETRY,
                branch_offsets=self._planning_offsets() if planning_enabled else None, stages=stages,
                planning_enabled=planning_enabled, dataset_split=self.dataset_split() if planning_enabled else None,
                parent=self,
            )
            self.sim_dialog.freezeRequested.connect(
                lambda approved, chosen_fit=fit, chosen_profile=profile:
                self.freeze_simulation_plans(approved[0], chosen_fit, chosen_profile, approved[1])
            )
            self.sim_dialog.show()
            if not planning_enabled:
                self.log("已按台面相对坐标生成模型；粗拟合前仅核对模型，真实路径须在球心拟合后生成。")
            return True
        except Exception as exc:
            self.show_error(exc)
            return False

    def scene_load_failed(self, message):
        self.log("RViz场景加载失败：" + message)
        QMessageBox.warning(
            self, "RViz场景未加载",
            "机械臂仿真服务没有成功接收台面和标准球。\n\n" + message +
            "\n\n请点击“关闭 RViz / MoveIt 仿真”，然后重新生成。",
        )

    def freeze_simulation_plans(self, plans, fit, profile, checked_statuses):
        try:
            ids = [row.get("plan_id") for rows in plans.values() for row in rows]
            if (not ids or len(set(ids)) != len(ids) or set(checked_statuses) != set(ids)
                    or any(checked_statuses[plan_id] != "OK" for plan_id in ids)):
                raise ValueError("整批候选路径必须全部通过MoveIt检查才能冻结")
            current_profile, current_fit, _, enabled, _ = self._planning_context(require_fit=True)
            if any(row.get("dataset_split") != self.dataset_split()
                   for rows in plans.values() for row in rows):
                raise ValueError("球位预注册用途在预览后发生变化，请重新生成仿真")
            if not enabled or profile_sha256(profile) != profile_sha256(current_profile):
                raise ValueError("模型配置在预览后发生变化，请重新生成仿真")
            if current_fit.get("sphere_center_mm") != fit.get("sphere_center_mm"):
                raise ValueError("球心或实验阶段在预览后发生变化，请重新生成仿真")
            save_profile(Path(self.profile_edit.text()).expanduser().resolve(), current_profile)
            paths = freeze_preview_plans(
                self.session(), self.placement(), plans, fit, DEFAULT_GEOMETRY,
                scene_from_profile(current_profile, current_fit)["table_z"], current_profile,
                checked_statuses,
            )
            self.log("仿真中查看的路径已原样冻结：" + ", ".join(path.name for path in paths.values()))
            QMessageBox.information(self, "路径已冻结", "正式计划已保存。真机执行时仍会逐点进行实时安全检查。")
        except Exception as exc:
            self.show_error(exc)

    def arm_contact(self):
        try:
            if (self.session() / f"{self.placement()}_manifest.json").exists():
                raise ValueError("当前球位计划已冻结；新增手动点请新建运行目录或球位")
            if not self.install_edit.text().strip():
                raise ValueError("先填写当前安装编号")
            if self.latest_snapshot is None or self.last_di is not False:
                raise ValueError("需有新鲜反馈，且测头必须先释放为DI1=0")
            self.armed = True
            self.arm_button.setText("已准备：等待DI1上升沿")
            self.log(f"准备记录 {self.branch_combo.currentText()} 下一次触发；请用示教器手动移动")
        except Exception as exc:
            self.show_error(exc)

    def record_safe_pose(self):
        try:
            if (self.session() / f"{self.placement()}_manifest.json").exists():
                raise ValueError("当前球位计划已冻结；安全位不可修改")
            if self.latest_snapshot is None or self.last_di is not False:
                raise ValueError("安全位必须在DI1=0的释放状态下记录")
            branch = self.branch_combo.currentText()
            path = self.safe_path()
            installation = self.install_edit.text().strip()
            if not installation:
                raise ValueError("先填写当前安装编号")
            record = json.loads(path.read_text()) if path.exists() else {
                "installation_id": installation, "placement_id": self.placement(),
                "standard_sphere_id": self.sphere_id_edit.text().strip(),
                "robot_ip": self.ip_edit.text().strip(), "poses": {},
            }
            if (record.get("installation_id") != installation or
                    record.get("placement_id") != self.placement() or
                    record.get("standard_sphere_id") != self.sphere_id_edit.text().strip() or
                    record.get("robot_ip") != self.ip_edit.text().strip()):
                raise ValueError("安全位文件属于另一安装、球位或机器人")
            poses = record["poses"]
            if branch in poses:
                raise ValueError(f"{branch}安全位已存在；为避免静默改计划，请新建运行目录")
            poses[branch] = self.latest_snapshot["pose"]
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            self.log(f"已记录 {branch} 安全位；生成计划后仍需MoveIt检查")
        except Exception as exc:
            self.show_error(exc)

    def _geometry_offsets(self):
        path = self.session() / "P0_geometry_fit.json"
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        if (result.get("installation_id") != self.install_edit.text().strip() or
                result.get("standard_sphere_id") != self.sphere_id_edit.text().strip() or
                result.get("geometry_sha256") != sha256(DEFAULT_GEOMETRY)):
            raise ValueError("P0几何文件的安装编号或名义几何版本不匹配")
        validate_fixture_fit_height(result, self.profile_from_ui())
        return result["local_offsets_mm"]

    def _planning_offsets(self):
        frozen = self._geometry_offsets()
        if frozen is not None:
            return frozen
        path = self.session() / "P0_manual_seed_geometry.json"
        if not path.exists():
            return None
        seed = json.loads(path.read_text())
        if (seed.get("installation_id") != self.install_edit.text().strip() or
                seed.get("standard_sphere_id") != self.sphere_id_edit.text().strip() or
                seed.get("geometry_sha256") != sha256(DEFAULT_GEOMETRY)):
            raise ValueError("临时几何种子安装编号或名义几何文件不匹配")
        return seed["local_offsets_mm"]

    def fit_center(self):
        try:
            self.profile = self.profile_from_ui()
            require_supported_contact_geometry(self.profile)
            if (self.session() / f"{self.placement()}_manifest.json").exists():
                raise ValueError("当前球位已有冻结计划；粗球心不可重新拟合")
            rows = read_jsonl(self.manual_path())
            installation = self.install_edit.text().strip()
            if any(row.get("installation_id") != installation or row.get("placement_id") != self.placement()
                   or row.get("standard_sphere_id") != self.sphere_id_edit.text().strip()
                   or row.get("robot_ip") != self.ip_edit.text().strip() for row in rows):
                raise ValueError("手动数据安装ID、球位或机器人IP与当前界面不一致")
            geometry = load_geometry(DEFAULT_GEOMETRY)
            fit = None
            nominal_error = None
            if self.placement() != "P0":
                offsets = self._geometry_offsets()
                if offsets is None:
                    raise ValueError("后续球位三面Z针定位前，必须完成当前安装的P0五分支几何标定")
                fit = fit_relocated_center_z(
                    rows, offsets["z"], fixture_sphere_height_base_mm(self.profile),
                    float(self.radius_edit.text()), float(self.probe_radius_edit.text()),
                )
                fit["geometry_fit_sha256"] = sha256(self.session() / "P0_geometry_fit.json")
            else:
                try:
                    fit = fit_coarse_center(
                        rows, geometry, float(self.radius_edit.text()),
                        float(self.probe_radius_edit.text()), self._geometry_offsets(),
                        fixed_sphere_center_z_mm=fixture_sphere_height_base_mm(self.profile),
                    )
                except ValueError as exc:
                    nominal_error = exc
            usable = fit_usable_for_motion(fit)[0] if fit is not None else False
            if not usable and self.placement() == "P0" and len(rows) >= 20:
                try:
                    seed = fit_manual_relative_seed(
                        rows, geometry, float(self.radius_edit.text()) + float(self.probe_radius_edit.text())
                    )
                    seed["installation_id"] = installation
                    seed["standard_sphere_id"] = self.sphere_id_edit.text().strip()
                    seed["geometry_sha256"] = sha256(DEFAULT_GEOMETRY)
                    (self.session() / "P0_manual_seed_geometry.json").write_text(
                        json.dumps(seed, ensure_ascii=False, indent=2) + "\n"
                    )
                    fit = fit_coarse_center(
                        rows, geometry, float(self.radius_edit.text()), float(self.probe_radius_edit.text()),
                        seed["local_offsets_mm"],
                        fixed_sphere_center_z_mm=fixture_sphere_height_base_mm(self.profile),
                    )
                    self.log("名义偏置不够一致；已从手动多姿态数据求得临时相对几何种子，正式G0仍需另测。")
                except ValueError as exc:
                    self.log(f"手动联合几何种子未通过：{exc}")
            if fit is None:
                raise ValueError(f"粗球心尚不可用：{nominal_error}")
            fit["installation_id"] = installation
            fit["standard_sphere_id"] = self.sphere_id_edit.text().strip()
            fit["placement_id"] = self.placement()
            fit["source_manual_jsonl"] = str(self.manual_path())
            fit["source_manual_sha256"] = sha256(self.manual_path())
            usable, reason = fit_usable_for_motion(fit)
            fit_path = self.session() / f"{self.placement()}_manual_coarse_preview.json"
            if not usable:
                fit_path.write_text(json.dumps(fit, ensure_ascii=False, indent=2) + "\n")
                self.log(f"拟合诊断已保存：RMS={fit['rms_radial_residual_mm']:.3f} mm；"
                         f"最大残差={fit['max_abs_radial_residual_mm']:.3f} mm；{reason}。"
                         "磁座坐标和RViz保持输入模型，请核对TCP/法兰坐标语义与测针几何。")
                return
            self.profile = self.register_profile_to_fit(self.profile, fit)
            magnetic = self.profile["workcell"]["magnetic_base"]
            fit["fixture_registration"] = {
                "method": "fitted_sphere_plus_operator_measured_dimensions",
                "magnetic_base_center_base_mm": list(magnetic["center_base_mm"]),
                "magnetic_base_size_xyz_mm": list(magnetic["size_xyz_mm"]),
                "dimensions_inferred_from_sphere_contacts": False,
            }
            fit_path.write_text(json.dumps(fit, ensure_ascii=False, indent=2) + "\n")
            self.log(f"粗球心：{[round(x, 3) for x in fit['sphere_center_mm']]} mm；"
                     f"RMS={fit['rms_radial_residual_mm']:.3f}，max={fit['max_abs_radial_residual_mm']:.3f} mm；"
                     f"自动入场={'可继续预检' if usable else reason}")
            self.log(
                "已按拟合球心和当前磁座/球杆/标准球尺寸回写磁座中心："
                f"{[round(x, 3) for x in magnetic['center_base_mm']]} mm。"
                "球心Z固定为台面Z+磁座高度+球杆长度+标准球半径；高度尺寸更改后须重新拟合。"
            )
            if self.moveit_process is not None and self.moveit_process.poll() is None:
                scene = scene_from_profile(self.profile, fit, self.latest_snapshot.get("pose") if self.latest_snapshot else None)
                if self.scene_worker is None or not self.scene_worker.isRunning():
                    self.scene_worker = SceneApplyThread(scene)
                    self.scene_worker.succeeded.connect(lambda message: self.log("真实球心已刷新到RViz场景"))
                    self.scene_worker.failed.connect(lambda message: self.log("RViz真实球心刷新失败：" + message))
                    self.scene_worker.start()
        except Exception as exc:
            self.show_error(exc)

    def generate_plans(self):
        self.open_simulation()

    def fit_geometry(self):
        try:
            self.profile = self.profile_from_ui()
            require_supported_contact_geometry(self.profile)
            coarse_fit = json.loads((self.session() / "P0_manual_coarse_preview.json").read_text())
            validate_fit_dimensions(coarse_fit, self.profile)
            validate_fixture_fit_height(coarse_fit, self.profile)
            if self.placement() != "P0":
                raise ValueError("只能在P0拟合本次安装几何")
            path = self.session() / "P0_geometry_fit.json"
            if path.exists():
                raise ValueError("P0几何已冻结；重标定请新建运行目录")
            contacts = self.session() / "P0_geometry_contacts.csv"
            require_complete_stage(self.session() / "P0_geometry_plan.csv", contacts, "P0 geometry")
            if any(row.get("session_id") != self.install_edit.text().strip() + "_P0"
                   or row.get("standard_sphere_id") != self.sphere_id_edit.text().strip()
                   for row in read_csv(contacts)):
                raise ValueError("几何触点不属于当前安装编号和P0")
            result = fit_new_installation_geometry(
                contacts, DEFAULT_GEOMETRY,
                float(self.radius_edit.text()) + float(self.probe_radius_edit.text()), path,
                fixed_sphere_center_z_mm=fixture_sphere_height_base_mm(self.profile),
            )
            result["installation_id"] = self.install_edit.text().strip()
            result["standard_sphere_id"] = self.sphere_id_edit.text().strip()
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            self.log(f"五分支相对几何已冻结：{path}；球心={result['sphere_center_mm']}")
        except Exception as exc:
            self.show_error(exc)

    def fit_anchor(self):
        try:
            self.profile = self.profile_from_ui()
            require_supported_contact_geometry(self.profile)
            coarse_fit = json.loads((self.session() / f"{self.placement()}_manual_coarse_preview.json").read_text())
            validate_fit_dimensions(coarse_fit, self.profile)
            validate_fixture_fit_height(coarse_fit, self.profile)
            placement = self.placement()
            output = self.session() / f"{placement}_anchor_fit.json"
            if output.exists():
                raise ValueError("锚点球心已冻结；不能覆盖")
            geometry_path = self.session() / "P0_geometry_fit.json"
            if not geometry_path.exists():
                raise ValueError("先完成P0新安装几何")
            geometry = json.loads(geometry_path.read_text())
            if (geometry.get("installation_id") != self.install_edit.text().strip() or
                    geometry.get("standard_sphere_id") != self.sphere_id_edit.text().strip()):
                raise ValueError("几何文件属于另一安装编号")
            validate_fixture_fit_height(geometry, self.profile)
            contacts = self.session() / f"{placement}_anchor_contacts.csv"
            require_complete_stage(self.session() / f"{placement}_anchor_plan.csv", contacts, "anchor")
            if any(row.get("session_id") != self.install_edit.text().strip() + "_" + placement
                   or row.get("standard_sphere_id") != self.sphere_id_edit.text().strip()
                   for row in read_csv(contacts)):
                raise ValueError("锚点触点不属于当前安装和球位")
            result = fit_anchor_center(
                contacts, geometry["local_offsets_mm"]["z"],
                float(self.radius_edit.text()) + float(self.probe_radius_edit.text()), output,
                fixed_sphere_center_z_mm=fixture_sphere_height_base_mm(self.profile),
            )
            result["installation_id"] = self.install_edit.text().strip()
            result["placement_id"] = placement
            result["standard_sphere_id"] = self.sphere_id_edit.text().strip()
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            self.log(f"独立A组球心已冻结：{[round(x, 3) for x in result['sphere_center_mm']]} mm；现在可生成E计划")
        except Exception as exc:
            self.show_error(exc)

    def start_acquisition(self):
        try:
            if self.acquire_worker is not None and self.acquire_worker.isRunning():
                raise ValueError("已有采集批次正在运行")
            stage = self.stage_combo.currentData()
            if self.feedback is not None and self.feedback.ip != self.ip_edit.text().strip():
                raise ValueError("界面只读反馈与自动采集的机器人IP不一致，请重新连接")
            profile, active_fit, _, _, _ = self._planning_context(require_fit=True)
            manifest = json.loads((self.session() / f"{self.placement()}_manifest.json").read_text())
            if manifest.get("dataset_split") != self.dataset_split():
                raise ValueError("界面球位用途与已冻结计划不一致")
            if (self.moveit_process is not None and self.moveit_process.poll() is None and
                    self.moveit_profile_hash != profile_sha256(profile)):
                raise ValueError("工具参数在MoveIt启动后发生变化；请停止并按当前参数重新启动MoveIt")
            scene = scene_from_profile(profile, active_fit)
            table = profile["workcell"]["table"]
            base = profile["workcell"]["magnetic_base"]
            runner = StageRunner(
                session_dir=self.session(), placement=self.placement(), stage=stage,
                robot_ip=self.ip_edit.text().strip(), installation_id=self.install_edit.text().strip(),
                standard_sphere_id=self.sphere_id_edit.text().strip(),
                geometry_path=DEFAULT_GEOMETRY, table_z_mm=scene["table_z"],
                table_center_xy=scene["table_center_xy"], table_size_xy=scene["table_size"],
                table_thickness_mm=scene["table_thickness"],
                magnetic_base_center=scene["base_center"], magnetic_base_size=base["size_xyz_mm"],
                frame_confirmed=self.frame_check.isChecked(), workbench_profile=profile,
            )
            self.acquire_worker = AcquireThread(runner)
            self.acquire_worker.message.connect(self.log)
            self.acquire_worker.succeeded.connect(lambda output: self.log(f"采集完成：{output}"))
            self.acquire_worker.failed.connect(self.show_error)
            self.acquire_worker.finished.connect(self._acquisition_finished)
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            for widget in (*self.jog_buttons, self.arm_button, self.safe_button, self.fit_button,
                           self.sim_button, self.input_position_button, self.plan_button, self.geometry_button, self.anchor_button):
                widget.setEnabled(False)
            if self.feedback is not None:
                self.feedback.close()
                self.feedback = None
                self.last_di = None
                self.latest_snapshot = None
                self.status.setText("自动采集中：实时反馈由执行线程独占")
            self.acquire_worker.start()
        except Exception as exc:
            self.show_error(exc)

    def stop_acquisition(self):
        if self.acquire_worker is not None and self.acquire_worker.isRunning():
            self.log("操作员请求停止；发送Stop并终止当前执行进程")
            self.acquire_worker.request_stop()

    def _acquisition_finished(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for widget in (*self.jog_buttons, self.arm_button, self.safe_button, self.fit_button,
                       self.sim_button, self.input_position_button, self.plan_button, self.geometry_button, self.anchor_button):
            widget.setEnabled(True)
        self.connect_feedback()

    def closeEvent(self, event):
        if self.acquire_worker is not None and self.acquire_worker.isRunning():
            self.stop_acquisition()
            event.ignore()
            self.log("正在停止机器人和工作线程；完成后再关闭窗口")
            return
        if self.feedback is not None:
            self.feedback.close()
        if self.moveit_process is not None:
            self.stop_moveit()
        self.refresh_timer.stop()
        self.moveit_timer.stop()
        self.workflow_timer.stop()
        self.jogShutdownRequested.emit()
        self.jog_thread.quit()
        self.jog_thread.wait(2000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = Workbench()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
