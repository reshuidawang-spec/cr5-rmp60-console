#!/usr/bin/env python3
"""Interactive, non-executing workcell and five-branch path preview."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import mpl_toolkits
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)
# Ubuntu's system mpl_toolkits can shadow the user-installed Matplotlib 3D
# package. Register the toolkit shipped beside the active Matplotlib first.
_active_toolkits = Path(matplotlib.__file__).resolve().parents[1] / "mpl_toolkits"
if _active_toolkits.is_dir() and str(_active_toolkits) not in mpl_toolkits.__path__:
    mpl_toolkits.__path__.insert(0, str(_active_toolkits))
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.projections import register_projection
register_projection(Axes3D)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import font_manager

_cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _cjk_font.exists():
    font_manager.fontManager.addfont(str(_cjk_font))
    matplotlib.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(_cjk_font)).get_name(), "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from cross_probe_model import branch_local_offset_mm, euler_to_matrix, load_geometry
from sphere_experiment_core import BRANCHES, ball_center_from_feedback
from sphere_simulation import automatic_plan_preview, moveit_preflight_batch, moveit_preflight_one, plan_preview, publish_plan_to_rviz, scene_checks


class PreviewWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, *args):
        super().__init__()
        self.function = function
        self.args = args

    def run(self):
        try:
            self.succeeded.emit(self.function(*self.args))
        except BaseException as exc:
            self.failed.emit(str(exc))


def _box_edges(center, size, rotation=None):
    half = np.asarray(size, dtype=float) / 2
    center = np.asarray(center, dtype=float)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    for axis in range(3):
        for a in (-1, 1):
            for b in (-1, 1):
                first = np.zeros(3)
                second = np.zeros(3)
                others = [item for item in range(3) if item != axis]
                first[axis], second[axis] = -half[axis], half[axis]
                first[others], second[others] = [a * half[others[0]], b * half[others[1]]], [
                    a * half[others[0]], b * half[others[1]]
                ]
                yield rotation @ first + center, rotation @ second + center


def _pose(row, prefix):
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")])


def _pose6(row, prefix):
    return [float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z", "rx", "ry", "rz")]


def _plot_cylinder(ax, start, end, radius, color, alpha=0.65):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    length = np.linalg.norm(axis)
    if length <= 1e-9 or radius <= 0:
        return
    unit = axis / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    side_a = np.cross(unit, helper)
    side_a /= np.linalg.norm(side_a)
    side_b = np.cross(unit, side_a)
    theta, along = np.mgrid[0:2 * np.pi:18j, 0:length:2j]
    surface = start[:, None, None] + unit[:, None, None] * along[None, :, :]
    surface += radius * (side_a[:, None, None] * np.cos(theta)[None, :, :] +
                         side_b[:, None, None] * np.sin(theta)[None, :, :])
    ax.plot_surface(surface[0], surface[1], surface[2], color=color, alpha=alpha, linewidth=0)


class SphereSimulationDialog(QDialog):
    freezeRequested = pyqtSignal(object)

    def __init__(self, *, scene: dict, fit: dict, placement: str, manual: list[dict],
                 safe_poses: dict, geometry_path: Path, branch_offsets: dict | None,
                 stages: tuple[str, ...] = (), planning_enabled: bool = True,
                 dataset_split: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{placement} 标准球与路径仿真预览（不驱动机器人）")
        self.resize(1200, 820)
        self.scene = scene
        self.fit = fit
        self.placement = placement
        self.manual = manual
        self.safe_poses = safe_poses
        self.geometry_path = geometry_path
        self.branch_offsets = branch_offsets
        self.stages = tuple(stages)
        self.dataset_split = dataset_split
        self.planning_enabled = bool(planning_enabled)
        self.plans = {}
        self.worker = None
        self.checked = {}
        self.geometry_ok = not any(item.startswith("错误") for item in scene_checks(scene))

        layout = QVBoxLayout(self)
        findings = scene_checks(scene)
        self.finding_label = QLabel("；".join(findings))
        self.finding_label.setWordWrap(True)
        layout.addWidget(self.finding_label)
        self.figure = Figure(figsize=(11, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=3)

        controls = QHBoxLayout()
        self.stage_combo = QComboBox()
        self.stage_combo.currentIndexChanged.connect(self.refresh_table)
        self.branch_combo = QComboBox()
        self.branch_combo.addItem("全部分支", "")
        for branch in BRANCHES:
            self.branch_combo.addItem(branch, branch)
        self.branch_combo.currentIndexChanged.connect(self.refresh_table)
        self.plan_button = QPushButton("自动搜索五针姿态与候选路径")
        self.plan_button.clicked.connect(self.generate_preview)
        self.plan_button.setEnabled(self.planning_enabled)
        self.check_button = QPushButton("MoveIt检查选中路径（不运动）")
        self.check_button.clicked.connect(self.check_selected)
        self.check_button.setEnabled(False)
        self.check_all_button = QPushButton("MoveIt检查整批路径（冻结前必做）")
        self.check_all_button.clicked.connect(self.check_all)
        self.check_all_button.setEnabled(False)
        self.rviz_button = QPushButton("在RViz显示选中路径")
        self.rviz_button.clicked.connect(self.show_selected_in_rviz)
        self.rviz_button.setEnabled(False)
        self.freeze_button = QPushButton("冻结当前预览为正式计划")
        self.freeze_button.clicked.connect(lambda: self.freezeRequested.emit((self.plans, dict(self.checked))))
        self.freeze_button.setEnabled(False)
        controls.addWidget(QLabel("批次"))
        controls.addWidget(self.stage_combo)
        controls.addWidget(QLabel("分支"))
        controls.addWidget(self.branch_combo)
        controls.addWidget(self.plan_button)
        controls.addWidget(self.rviz_button)
        controls.addWidget(self.check_button)
        controls.addWidget(self.check_all_button)
        controls.addWidget(self.freeze_button)
        layout.addLayout(controls)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["计划ID", "分支", "几何检查", "MoveIt检查"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.render)
        layout.addWidget(self.table, stretch=2)
        self.message = QLabel("预览仅用于检查输入和候选路径；真机前还会重新做实时状态、场景及入口检查。")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.render()

    def _start(self, function, args, callback):
        if self.worker is not None and self.worker.isRunning():
            return
        self.plan_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.check_all_button.setEnabled(False)
        self.rviz_button.setEnabled(False)
        self.freeze_button.setEnabled(False)
        self.worker = PreviewWorker(function, *args)
        self.worker.succeeded.connect(callback)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _finished(self):
        self.plan_button.setEnabled(self.planning_enabled)
        self.check_button.setEnabled(self.geometry_ok and bool(self._selected_row()))
        self.check_all_button.setEnabled(self.geometry_ok and bool(self.plans))
        self.rviz_button.setEnabled(bool(self._selected_row()))
        self._update_freeze_state()

    def _update_freeze_state(self):
        ids = [row.get("plan_id") for rows in self.plans.values() for row in rows]
        self.freeze_button.setEnabled(
            self.geometry_ok and bool(ids) and all(self.checked.get(plan_id) == "OK" for plan_id in ids)
            and not (self.worker is not None and self.worker.isRunning())
        )

    def _failed(self, message):
        self.message.setText("预览失败：" + message)
        QMessageBox.warning(self, "仿真预检未通过", message)

    def generate_preview(self):
        self.plans = {}
        self.checked.clear()
        self.freeze_button.setEnabled(False)
        self.message.setText("正在搜索五针姿态及安全位，需要当前关节状态和MoveIt；全程不执行运动…")
        self._start(automatic_plan_preview, (self.scene, self.fit, self.placement,
            self.stages, self.manual, self.dataset_split), self._plans_ready)

    def _plans_ready(self, plans):
        self.plans = plans
        self.checked.clear()
        self.stage_combo.blockSignals(True)
        self.stage_combo.clear()
        for stage, rows in plans.items():
            self.stage_combo.addItem(f"{stage}（{len(rows)}条）", stage)
        self.stage_combo.blockSignals(False)
        self.message.setText("候选路径已生成。选一行查看四个法兰位姿；MoveIt检查需要已启动的move_group服务。")
        self.check_all_button.setEnabled(self.geometry_ok and bool(plans))
        self.freeze_button.setEnabled(False)
        self.refresh_table()

    def _visible_rows(self):
        rows = self.plans.get(self.stage_combo.currentData(), [])
        branch = self.branch_combo.currentData()
        return [row for row in rows if not branch or row.get("branch") == branch]

    def refresh_table(self):
        rows = self._visible_rows()
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = [row.get("plan_id", ""), row.get("branch", ""),
                      row.get("probe_collision_status", ""),
                      self.checked.get(row.get("plan_id", ""), "未检查")]
            for col, value in enumerate(values):
                self.table.setItem(index, col, QTableWidgetItem(str(value)))
        self.table.blockSignals(False)
        if rows:
            self.table.selectRow(0)
        else:
            self.render()
        self.check_button.setEnabled(self.geometry_ok and bool(self._selected_row()))
        self.rviz_button.setEnabled(bool(self._selected_row()))

    def _selected_row(self):
        index = self.table.currentRow()
        rows = self._visible_rows()
        return rows[index] if 0 <= index < len(rows) else None

    def check_selected(self):
        row = self._selected_row()
        if row is None:
            return
        joints = None  # Require a fresh current joint state, not a historical trigger.
        self.message.setText("正在运行MoveIt IK、笛卡尔路径及碰撞预检；不会执行轨迹…")
        self._start(moveit_preflight_one, (self.scene, self.fit, row, joints), self._checked_ready)

    def check_all(self):
        joints = None  # Require a fresh current joint state, not a historical trigger.
        self.checked.clear()
        self.freeze_button.setEnabled(False)
        self.message.setText("正在检查当前预览中的全部路径；完成后才能冻结。")
        self._start(moveit_preflight_batch, (self.scene, self.plans, joints), self._all_checked_ready)

    def _all_checked_ready(self, statuses):
        self.checked = statuses
        self.refresh_table()
        self.message.setText(f"整批 {len(statuses)} 条路径均通过MoveIt预检，可冻结当前预览。")
        self._update_freeze_state()

    def show_selected_in_rviz(self):
        row = self._selected_row()
        if row is None:
            return
        self.message.setText("正在把选中路径发布到RViz；只显示，不会运动机器人…")
        self._start(publish_plan_to_rviz, (row,), self._rviz_ready)

    def _rviz_ready(self, message):
        self.message.setText(message or "选中路径已显示在RViz。红线是法兰路径，彩色点依次为安全、过渡、起始和目标位。")

    def _checked_ready(self, result):
        plan_id = result.get("plan_id", "")
        status = result.get("moveit_status", "FAIL")
        self.checked[plan_id] = status
        self.message.setText(f"{plan_id}: MoveIt {status}。" + (result.get("moveit_error") or result.get("preflight_log", ""))[-500:])
        selected_id = self._selected_row().get("plan_id") if self._selected_row() else None
        self.refresh_table()
        self._update_freeze_state()
        if selected_id:
            for index, row in enumerate(self._visible_rows()):
                if row.get("plan_id") == selected_id:
                    self.table.selectRow(index)
                    break

    def render(self):
        self.figure.clear()
        ax3d = self.figure.add_subplot(121, projection="3d")
        ax2d = self.figure.add_subplot(122)
        scene = self.scene
        center = np.asarray(scene["sphere_center"])
        radius = scene["sphere_radius"]
        u, v = np.mgrid[0:2 * np.pi:18j, 0:np.pi:10j]
        ax3d.plot_wireframe(center[0] + radius * np.cos(u) * np.sin(v),
                            center[1] + radius * np.sin(u) * np.sin(v),
                            center[2] + radius * np.cos(v), color="royalblue", alpha=0.7)
        for point_a, point_b in _box_edges(scene["base_center"], scene["base_size"], scene.get("base_rotation")):
            ax3d.plot(*zip(point_a, point_b), color="darkorange", alpha=0.8)
        stem = scene.get("sphere_stem")
        if stem:
            stem_center = np.asarray(stem["center_mm"], dtype=float)
            half = float(stem["length_mm"]) / 2
            _plot_cylinder(ax3d, stem_center - [0, 0, half], stem_center + [0, 0, half],
                           float(stem["diameter_mm"]) / 2, "silver")
        table_center = scene.get("table_center", [*scene["table_center_xy"], scene["table_z"] - scene["table_thickness"] / 2])
        for point_a, point_b in _box_edges(table_center, [*scene["table_size"], scene["table_thickness"]], scene.get("table_rotation")):
            ax3d.plot(*zip(point_a, point_b), color="gray", alpha=0.4)
        path = self._selected_row()
        if path:
            prefixes = ("safe_flange", "transition_flange", "start_flange", "target_flange")
            points = np.asarray([_pose(path, prefix) for prefix in prefixes])
            ax3d.plot(points[:, 0], points[:, 1], points[:, 2], "-o", color="crimson")
            ax2d.plot(points[:, 0], points[:, 1], "-o", color="crimson")
            for label, point in zip(("safe", "transition", "start", "target"), points):
                ax3d.text(*point, label, fontsize=8)
            if all(f"initial_offset_{axis}" in path for axis in ("x", "y", "z")):
                offset = [float(path[f"initial_offset_{axis}"]) for axis in ("x", "y", "z")]
                tip = np.asarray([ball_center_from_feedback(_pose6(path, prefix), offset) for prefix in prefixes])
                ax3d.plot(tip[:, 0], tip[:, 1], tip[:, 2], "-o", color="forestgreen")
                ax2d.plot(tip[:, 0], tip[:, 1], "-o", color="forestgreen")
        flange_pose = scene.get("flange_pose")
        tool = scene.get("tool")
        if flange_pose is not None and tool:
            flange = np.asarray(flange_pose[:3], dtype=float)
            rotation = np.asarray(euler_to_matrix("xyz", flange_pose[3:6]), dtype=float)
            mount = np.asarray(tool.get("mount_xyz_mm", [0, 0, 0]), dtype=float)
            origin = flange + rotation @ mount
            axis = rotation @ np.array([0.0, 0.0, 1.0])
            connector_end = origin + axis * float(tool["connector"]["length_mm"])
            body_end = connector_end + axis * float(tool["probe_body"]["length_mm"])
            _plot_cylinder(ax3d, origin, connector_end, float(tool["connector"]["diameter_mm"]) / 2, "dimgray")
            _plot_cylinder(ax3d, connector_end, body_end, float(tool["probe_body"]["diameter_mm"]) / 2, "steelblue")
            geometry = load_geometry(self.geometry_path)
            hub = flange + rotation @ np.asarray([0, 0, float(geometry["derived_nominal"]["flange_to_cross_hub_mm"])])
            offsets = self.branch_offsets or {branch: branch_local_offset_mm(geometry, branch) for branch in BRANCHES}
            for branch in BRANCHES:
                tip_world = flange + rotation @ np.asarray(offsets[branch], dtype=float)
                _plot_cylinder(ax3d, hub, tip_world, float(geometry["stylus"].get("rod_radius_mm", 0.8)), "silver", 0.9)
                ax3d.scatter(*tip_world, color="darkred", s=14)
        ax3d.scatter(*center, color="royalblue", s=30)
        ax3d.set(xlabel="X / mm", ylabel="Y / mm", zlabel="Z / mm",
                 title="球/磁座与路径（红=法兰，绿=触测球心）")
        span = 180.0
        ax3d.set_xlim(center[0] - span, center[0] + span)
        ax3d.set_ylim(center[1] - span, center[1] + span)
        ax3d.set_zlim(max(float(table_center[2]) - 20, center[2] - span), center[2] + span)
        tx, ty = scene["table_center_xy"]
        sx, sy = scene["table_size"]
        bx, by, _ = scene["base_center"]
        bsx, bsy, _ = scene["base_size"]
        from matplotlib.patches import Polygon, Circle
        table_rot = np.asarray(scene.get("table_rotation", np.eye(3)), dtype=float)
        base_rot = np.asarray(scene.get("base_rotation", np.eye(3)), dtype=float)
        table_xy = [((table_rot @ np.array([dx, dy, 0.0])) + np.asarray(scene.get("table_top_center", [tx, ty, scene["table_z"]])))[:2]
                    for dx, dy in ((-sx/2, -sy/2), (sx/2, -sy/2), (sx/2, sy/2), (-sx/2, sy/2))]
        base_xy = [((base_rot @ np.array([dx, dy, 0.0])) + np.asarray(scene["base_center"]))[:2]
                   for dx, dy in ((-bsx/2, -bsy/2), (bsx/2, -bsy/2), (bsx/2, bsy/2), (-bsx/2, bsy/2))]
        ax2d.add_patch(Polygon(table_xy, closed=True, fill=False, edgecolor="gray", lw=2))
        ax2d.add_patch(Polygon(base_xy, closed=True, fill=False, edgecolor="darkorange", lw=2))
        ax2d.add_patch(Circle(center[:2], radius, fill=False, edgecolor="royalblue", lw=2))
        ax2d.plot(center[0], center[1], "+", color="royalblue")
        ax2d.set(xlabel="X / mm", ylabel="Y / mm", title="台面、磁座、球 XY 俯视")
        ax2d.set_aspect("equal", adjustable="box")
        ax2d.set_xlim(tx - sx * 0.6, tx + sx * 0.6)
        ax2d.set_ylim(ty - sy * 0.6, ty + sy * 0.6)
        ax2d.grid(alpha=0.2)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.message.setText("请等待当前离线规划或MoveIt检查结束后再关闭预览")
            event.ignore()
        else:
            super().closeEvent(event)
