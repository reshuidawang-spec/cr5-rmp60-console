#!/usr/bin/env python3
"""
五分支十字探针球心标定（自动规划 + 可选执行 + 偏置拟合）。

================================================================================
工作流总览（Workflow Overview）
================================================================================

本脚本实现了"五分支十字探针"的球心标定流水线。十字探针有五个测针分支（z、x_pos、
x_neg、y_pos、y_neg），每个分支的球头（ruby ball）相对于法兰坐标系有一个局部偏置
（local_offset）。标定的目标是在固定球心（已知参考标定结果）的约束下，精确估计每个
分支的实际偏置。

完整工作流包含四个阶段：

1. 规划生成（Plan Generation）
   - 读入参考分支的标定结果（reference fit），其中包含标定球球心坐标
     sphere_center_mm
   - 对每个分支、每个采样点，根据名义或种子偏置计算逼近方向、起止位姿
   - 进行碰撞检测（含球体碰撞 + 桌面碰撞），输出每项的碰撞状态
   - 输出 CSV 格式规划文件

2. 预检（Preflight Checks）
   - 检查 safe_pose 到 transition 的朝向变化是否超限
   - 检查所有规划项的碰撞状态是否为 OK，拒绝风险项

3. 带保护的运动执行（Guarded Execution）
   - 按分支分组执行，每个分支依次处理所有采样点
   - 三阶段运动：safe_pose → transition (安全远距离) → start (逼近起点) →
     probe (沿逼近方向直线探测)
   - MovL 沿 target 方向匀速推进，同时监测 DI1 触发信号
   - 提前触发处理：可跳过当前分支剩余点（--skip-branch-on-early-trigger），
     或直接中止（传统行为）

4. 偏置拟合（Offset Fitting）
   - 基于固定球心标量距离约束，对每个分支做鲁棒非线性最小二乘
   - 球心约束：ball_center = flange_position + R * local_offset
   - 球面方程：||flange + R*offset - sphere_center|| = contact_distance
   - approach 只用于初值；真实径向法向由当前球心和测球中心计算
   - 使用 Huber IRLS、几何残差离群点剔除和 Jacobian 可观测性门控

================================================================================
数学背景（Mathematical Background）
================================================================================

- 法兰位姿 P = [x, y, z, rx, ry, rz]
- 旋转矩阵 R = euler_to_matrix(rx, ry, rz)，按给定欧拉角序列
- 局部偏置 o = local_ball_offset_mm（待估计量）
- 接触距离 d = contact_center_distance_mm = sphere_radius + ball_radius
- 球心 S = sphere_center_mm（固定已知）

接触时刻满足：
    ||P[:3] + R * o - S|| = d

对多个方向/姿态的接触点做鲁棒 Gauss-Newton 求解 o。

================================================================================
安全机制（Safety Mechanisms）
================================================================================

- safe_pose: 用户示教的机器人安全位姿
- safe_pose_policy: 控制何时回到 safe_pose（每次采样/每分支/仅首尾）
- guarded motion: 运动过程中持续监测 DI1，提前触发则立即停止并反转退回
- 碰撞检测: 使用胶囊体 + 球体代理模型，检查与标定球和桌面的最小距离
- per-branch-standoff: 可为不同分支单独设置逼近起点距离
- max_local_step: 限制当前位姿到 transition 的最大移动距离

================================================================================
执行策略（Execution Strategy）
================================================================================

默认模式为 dry-run（仅生成规划），不执行机器人运动。
需要同时指定 --execute 和 --ack-five-branch-calibration-path 才能执行真实运动。
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from dobot_msgs_v4.srv import MovJ

from cross_probe_model import (
    DEFAULT_GEOMETRY,
    branch_local_offset_mm,
    branch_map,
    euler_to_matrix,
    load_geometry,
    mat_vec,
    nominal_branch_origin_mm,
)
from jog_and_record_contacts import (
    current_pose,
    format_pose,
    fresh_feed_di1,
    issue_movl,
    pose_fields,
    vector_fields,
    wait_jog,
)
from probe_touch import ProbeTouch
from sphere_offset_fit import solve_fixed_center_sphere_offset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_FIT = PROJECT_DIR / "data/yneg_gui_normal_absolute_fit_20260610.json"
DEFAULT_PLAN = PROJECT_DIR / "data/five_branch_sphere_calibration_plan_20260612.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data/five_branch_sphere_calibration_contacts_20260612.csv"
DEFAULT_CALIBRATION_OUTPUT = PROJECT_DIR / "data/five_branch_sphere_calibration_20260612.json"
DEFAULT_RESIDUAL_OUTPUT = PROJECT_DIR / "data/five_branch_sphere_calibration_residuals_20260612.csv"
# 默认的五个分支名称，对应十字探针的五个方向
DEFAULT_BRANCHES = ("z", "x_pos", "x_neg", "y_pos", "y_neg")
POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")


class PositioningTriggerError(RuntimeError):
    """
    定位过程中 DI1 提前触发的异常。

    当机器人在从 safe_pose 向 transition 或 start 运动的过程中，
    DI1 信号提前跳变（说明路径上有意外障碍），运动立即停止。
    该异常携带触发时刻的快照、停止位姿和反向退回位姿，供上层
    恢复处理。

    Attributes:
        label: 触发时正在执行的运动阶段标签（如 "transition", "start"）
        trigger_snapshot: 触发时刻的 DI1 快照数据
        stop_pose: 机器人停止时的位姿
        recovery_pose: 反向退回后的位姿
    """

    def __init__(self, label, trigger_snapshot, stop_pose, recovery_pose):
        super().__init__(f"DI1 triggered while moving to {label}; motion stopped and retracted")
        self.label = label
        self.trigger_snapshot = trigger_snapshot
        self.stop_pose = stop_pose
        self.recovery_pose = recovery_pose


def parse_float(value, label):
    """
    将字符串解析为浮点数，对空值或非法值抛出带标签的异常。

    Args:
        value: 待解析的字符串或 None
        label: 字段名称，用于错误消息

    Returns:
        解析后的 float

    Raises:
        ValueError: 值为空、非数值或非有限数时抛出
    """
    if value in (None, ""):
        raise ValueError(f"missing {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def norm(vector):
    """
    计算三维向量的 L2 范数（欧几里得长度）。

    Args:
        vector: 类数组三维向量

    Returns:
        标量范数值
    """
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def normalize(vector, label):
    """
    将三维向量归一化为单位向量。零向量会抛出异常。

    Args:
        vector: 三维向量
        label: 向量名称，用于错误消息

    Returns:
        归一化后的单位向量（长度为1）

    Raises:
        ValueError: 向量长度接近零时抛出
    """
    length = norm(vector)
    if length <= 1e-12:
        raise ValueError(f"{label} cannot be zero")
    return [float(value) / length for value in vector]


def add(a, b):
    """
    逐元素向量加法。

    Args:
        a: 第一个三维向量
        b: 第二个三维向量

    Returns:
        [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    """
    return [float(a[index]) + float(b[index]) for index in range(3)]


def sub(a, b):
    """
    逐元素向量减法。

    Args:
        a: 被减数向量
        b: 减数向量

    Returns:
        [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
    """
    return [float(a[index]) - float(b[index]) for index in range(3)]


def scale(vector, value):
    """
    向量标量乘法（逐元素乘标量）。

    Args:
        vector: 三维向量
        value: 标量因子

    Returns:
        缩放后的向量 [v[i] * value]
    """
    return [float(component) * float(value) for component in vector]


def transpose_mat_vec(matrix, vector):
    """
    计算矩阵转置与向量的乘积 (M^T * v)。

    这是求解偏置时的关键操作：将接触方程从世界坐标系转换回
    局部法兰坐标系，需要用到旋转矩阵的转置。
    等价于 mat_vec(transpose(matrix), vector)。

    Args:
        matrix: 3x3 矩阵
        vector: 三维向量

    Returns:
        M^T * v 的结果向量
    """
    return [sum(float(matrix[row][col]) * float(vector[row]) for row in range(3)) for col in range(3)]


def pose_with_position(position, orientation):
    """
    将位置向量和朝向向量合并为一个 6-DOF 位姿。

    Args:
        position: 三维位置 [x, y, z]
        orientation: 三维朝向 [rx, ry, rz]（欧拉角）

    Returns:
        六维位姿 [x, y, z, rx, ry, rz]
    """
    return [position[0], position[1], position[2], orientation[0], orientation[1], orientation[2]]


def pose_dict(prefix, pose):
    """
    将六维位姿转换为带前缀的字典，用于 CSV 列输出。

    Args:
        prefix: 列名前缀，如 "transition_flange"
        pose: 六维位姿 [x, y, z, rx, ry, rz]

    Returns:
        {"{prefix}_x": "x", "{prefix}_y": "y", ...} 格式的字典
    """
    return {f"{prefix}_{POSE_NAMES[index]}": f"{float(pose[index]):.4f}" for index in range(6)}


def angle_delta_deg(actual, expected):
    """
    计算两个角度的有符号差值，结果归一化到 [-180, 180) 度。

    用于比较机器人朝向的差异，考虑角度周期性（如 359° 和 1° 只差 2°）。

    Args:
        actual: 实际角度（度）
        expected: 期望角度（度）

    Returns:
        归一化后的角度差值
    """
    return (float(actual) - float(expected) + 180.0) % 360.0 - 180.0


def max_orientation_delta_deg(a, b):
    """
    计算两个六维位姿之间的最大朝向分量差值。

    取 rx、ry、rz 三个分量各自差值绝对值的最大值，
    用于判断 safe_pose 到 transition 之间的朝向变化是否可接受。

    Args:
        a: 第一个六维位姿
        b: 第二个六维位姿

    Returns:
        三个朝向分量中最大绝对值差值（度）
    """
    return max(abs(angle_delta_deg(a[index], b[index])) for index in range(3, 6))


def point_segment_distance(point, start, end):
    """
    计算三维点到线段的最近距离。

    如果垂足在线段范围内，返回点到直线的垂直距离；
    如果垂足在线段端点外，返回点到最近端点的距离。

    用于碰撞检测：评估球心到探针胶囊体轴线的距离。

    Args:
        point: 三维空间点
        start: 线段起点
        end: 线段终点

    Returns:
        点到线段的最短距离
    """
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / denom)
    t = max(0.0, min(1.0, t))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def transform_point(pose, rotation, local_point):
    """
    将局部坐标系下的点变换到世界坐标系。

    公式：world_point = pose[:3] + R * local_point
    用于将探针代理模型上定义的点（如胶囊端点、球心）
    从法兰局部坐标系变换到世界坐标系，再进行碰撞检测。

    Args:
        pose: 法兰六维位姿
        rotation: 3x3 旋转矩阵（由欧拉角计算得到）
        local_point: 局部坐标系下的三维点

    Returns:
        世界坐标系下的三维点
    """
    return add(pose[:3], mat_vec(rotation, local_point))


def load_reference_fit(path):
    """
    加载参考分支的标定结果 JSON 文件。

    参考标定结果提供了标定球的球心坐标 sphere_center_mm 和
    参考分支本身的局部偏置 local_ball_offset_mm。

    Args:
        path: JSON 文件路径

    Returns:
        字典，包含 sphere_center_mm 和 local_ball_offset_mm
            等字段

    Raises:
        ValueError: 缺少必要字段或字段格式不正确时抛出
    """
    data = json.loads(Path(path).read_text())
    for field in ("sphere_center_mm", "local_ball_offset_mm"):
        if field not in data:
            raise ValueError(f"reference fit missing {field}")
        if not isinstance(data[field], list) or len(data[field]) != 3:
            raise ValueError(f"reference fit {field} must contain 3 values")
    return data


def residual_orientations(fit):
    """
    从参考标定结果的残差行中提取各采样点的朝向（欧拉角）。

    用于生成规划时循环使用参考标定中各采样点的朝向，保证
    新规划的采样朝向与参考标定一致。

    Args:
        fit: 参考标定结果字典

    Returns:
        朝向列表，每个元素为 [rx, ry, rz]；若无残差行则返回空列表
    """
    orientations = []
    for row in fit.get("residual_rows", []):
        try:
            orientations.append(
                [
                    parse_float(row.get("rx"), "reference residual rx"),
                    parse_float(row.get("ry"), "reference residual ry"),
                    parse_float(row.get("rz"), "reference residual rz"),
                ]
            )
        except ValueError:
            continue
    return orientations


def branch_direction(geometry, branch_name):
    """
    获取指定分支在法兰局部坐标系下的名义方向单位向量。

    从几何配置中读取分支方向向量并归一化。

    Args:
        geometry: 探针几何配置字典
        branch_name: 分支名称（如 "z", "x_pos"）

    Returns:
        单位方向向量 [dx, dy, dz]

    Raises:
        ValueError: 分支名称不存在时抛出
    """
    branches = branch_map(geometry)
    if branch_name not in branches:
        raise ValueError(f"unknown branch {branch_name!r}; choices: {', '.join(sorted(branches))}")
    return normalize(branches[branch_name]["direction"], f"{branch_name} direction")


def probe_dimensions(geometry, args):
    """
    从几何配置和命令行参数计算探针的所有关键尺寸。

    包括法兰适配器、探针本体、测针和球头的尺寸。
    撞针杆碰撞半径（rod_collision_radius）有特殊逻辑：
    如果显式指定则用指定值；如果配置的杆半径大于或等于球半径，
    则取杆半径的一半（考虑实际安装后杆比球粗的情况）；否则直接用配置值。

    Args:
        geometry: 探针几何配置字典
        args: 命令行参数（可能包含 rod_collision_radius_mm 覆盖）

    Returns:
        字典，包含 adapter_length_mm、ball_radius_mm 等所有关键尺寸
    """
    adapter = geometry["flange_adapter"]
    body = geometry["probe_body"]
    stylus = geometry["stylus"]
    configured_rod_radius = float(stylus["rod_radius_mm"])
    ball_radius = float(stylus.get("ball_radius_mm", 0.5 * float(stylus["ball_diameter_mm"])))
    if args.rod_collision_radius_mm is not None:
        rod_collision_radius = float(args.rod_collision_radius_mm)
    elif configured_rod_radius >= ball_radius:
        rod_collision_radius = 0.5 * configured_rod_radius
    else:
        rod_collision_radius = configured_rod_radius
    branch_origin_z = float(
        geometry.get("derived_nominal", {}).get(
            "flange_to_branch_origin_mm",
            float(adapter["effective_axial_length_mm"]) + float(body["length_mm"]),
        )
    )
    cross_hub_z = float(
        geometry.get("derived_nominal", {}).get(
            "flange_to_cross_hub_mm",
            branch_origin_z + float(stylus.get("cross_hub_offset_from_branch_origin_mm", 0.0)),
        )
    )
    side_rod_radius = float(stylus.get("side_rod_radius_mm", configured_rod_radius))
    axial_rod_radius = float(stylus.get("axial_rod_radius_mm", configured_rod_radius))
    center_stem_radius = float(stylus.get("center_stem_radius_mm", configured_rod_radius))
    return {
        "adapter_length_mm": float(adapter["effective_axial_length_mm"]),
        "adapter_radius_mm": float(adapter.get("radius_mm", 30.0)),
        "probe_body_length_mm": float(body["length_mm"]),
        "probe_body_radius_mm": 0.5 * float(body["diameter_mm"]),
        "branch_center_distance_mm": float(stylus["branch_center_distance_mm"]),
        "flange_to_branch_origin_mm": branch_origin_z,
        "flange_to_cross_hub_mm": cross_hub_z,
        "ball_radius_mm": ball_radius,
        "configured_rod_radius_mm": configured_rod_radius,
        "rod_collision_radius_mm": rod_collision_radius,
        "side_rod_radius_mm": side_rod_radius,
        "side_rod_collision_radius_mm": float(stylus.get("side_rod_collision_radius_mm", rod_collision_radius)),
        "axial_rod_radius_mm": axial_rod_radius,
        "axial_rod_collision_radius_mm": float(stylus.get("axial_rod_collision_radius_mm", rod_collision_radius)),
        "center_stem_radius_mm": center_stem_radius,
        "center_stem_collision_radius_mm": float(
            stylus.get("center_stem_collision_radius_mm", center_stem_radius)
        ),
    }


def branch_collision_hub_mm(geometry, branch_name, ball_center, dims):
    """Return the local-frame stylus hub used for collision modelling.

    The real five-way stylus has a lower cross hub.  Side branches leave from
    that hub and are approximately horizontal; they should not be modelled as
    long diagonal rods starting at the RMP60 body bottom.
    """
    if branch_name == "z":
        return [0.0, 0.0, dims["flange_to_cross_hub_mm"]]
    direction = branch_direction(geometry, branch_name)
    horizontal = [direction[0], direction[1], 0.0]
    if norm(horizontal) <= 1e-9:
        return [0.0, 0.0, dims["flange_to_cross_hub_mm"]]
    horizontal = normalize(horizontal, f"{branch_name} horizontal hub direction")
    return [
        float(ball_center[0]) - horizontal[0] * dims["branch_center_distance_mm"],
        float(ball_center[1]) - horizontal[1] * dims["branch_center_distance_mm"],
        float(ball_center[2]),
    ]


def branch_rod_collision_radius(branch_name, dims):
    """Return the collision radius for one stylus branch."""
    if branch_name == "z":
        return dims["axial_rod_collision_radius_mm"]
    return dims["side_rod_collision_radius_mm"]


def offset_source_for_branch(branch, reference_fit, geometry, seed_offsets=None):
    """
    确定指定分支使用哪个偏置值作为规划初始值，以及其来源。

    优先级：
    1. 如果命令行指定了该分支的种子偏置（seed_offsets），优先使用
    2. 如果该分支就是参考分支本身，使用参考标定结果中的偏置
    3. 否则使用几何名义值（nominal geometry）

    Args:
        branch: 分支名称
        reference_fit: 参考标定结果
        geometry: 探针几何配置
        seed_offsets: 可选的种子偏置字典 {branch: offset}

    Returns:
        (offset_vector, source_label) 元组，
        source_label 为 "branch_seed_fit" / "reference_fit" / "nominal_geometry"
    """
    if seed_offsets and branch in seed_offsets:
        return seed_offsets[branch], "branch_seed_fit"
    if reference_fit.get("branch") == branch:
        return [float(v) for v in reference_fit["local_ball_offset_mm"]], "reference_fit"
    return branch_local_offset_mm(geometry, branch), "nominal_geometry"


def load_branch_seed_offsets(items):
    """
    解析 --branch-seed-fit 参数，加载每个分支的种子偏置。

    格式为 "BRANCH=PATH"，支持两种格式的 JSON 文件：
    1. 直接包含 local_ball_offset_mm 字段
    2. 包含 branches 列表，其中对应分支有 estimated_offset_mm 字段

    Args:
        items: 字符串列表，每项为 "BRANCH=PATH" 格式

    Returns:
        字典 {branch_name: [offset_x, offset_y, offset_z]}

    Raises:
        ValueError: 格式错误或找不到偏置值时抛出
    """
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--branch-seed-fit entries must be BRANCH=PATH")
        branch, path = item.split("=", 1)
        branch = branch.strip()
        path = path.strip()
        if not branch or not path:
            raise ValueError("--branch-seed-fit entries must be BRANCH=PATH")
        data = json.loads(Path(path).read_text())
        offset = data.get("frozen_offset_mm")
        if not isinstance(offset, list) or len(offset) != 3:
            offset = data.get("local_ball_offset_mm")
        if not isinstance(offset, list) or len(offset) != 3:
            fit_result = next(
                (
                    branch_result
                    for branch_result in data.get("branches", [])
                    if branch_result.get("branch") == branch and branch_result.get("ok", True)
                ),
                None,
            )
            if fit_result is not None:
                offset = fit_result.get("estimated_offset_mm")
        if not isinstance(offset, list) or len(offset) != 3:
            raise ValueError(
                f"{path}: missing 3-value frozen_offset_mm, local_ball_offset_mm "
                f"or branches[].estimated_offset_mm for {branch}"
            )
        result[branch] = [float(value) for value in offset]
    return result


def local_collision_primitives(geometry, active_branch, active_offset, args, collision_offsets=None):
    """
    构建用于碰撞检测的代理几何体（primitive）列表。

    包括三个部分：
    1. 法兰适配器和探针本体的胶囊体（固定，与分支无关）
    2. 非活动分支的测针杆 + 球头（使用校准偏置或名义偏置）
    3. 活动分支自身的测针杆（减去 stem_exclusion 长度避免自碰撞）

    每个几何体定义在法兰局部坐标系下，碰撞检测时通过变换矩阵
    将其转换到世界坐标系。

    Args:
        geometry: 探针几何配置
        active_branch: 当前正在探测的分支名称
        active_offset: 活动分支的偏置向量
        args: 命令行参数
        collision_offsets: 其他分支的校准偏置字典（可选）

    Returns:
        (primitives, dims, active_origin) 三元组：
        - primitives: 代理几何体列表
        - dims: 探针尺寸字典
        - active_origin: 分支原点（名义值）
    """
    dims = probe_dimensions(geometry, args)
    nominal_origin = nominal_branch_origin_mm(geometry)
    cross_hub = [0.0, 0.0, dims["flange_to_cross_hub_mm"]]
    active_origin = branch_collision_hub_mm(geometry, active_branch, active_offset, dims)
    # 计算活动分支校准后的测针方向向量
    calibrated_active_vector = sub(active_offset, active_origin)
    calibrated_active_length = norm(calibrated_active_vector)
    if calibrated_active_length <= 1e-9:
        raise ValueError(f"{active_branch} calibrated offset coincides with the stylus hub")
    calibrated_active_direction = normalize(
        calibrated_active_vector,
        f"{active_branch} calibrated stem direction",
    )
    branch_dirs = {
        name: normalize(branch["direction"], f"{name} direction")
        for name, branch in branch_map(geometry).items()
    }

    # 法兰适配器和探针本体（固定部分）的胶囊体
    primitives = [
        {
            "name": "adapter_body_nominal",
            "type": "capsule",
            "a": [0.0, 0.0, 0.0],
            "b": [0.0, 0.0, dims["adapter_length_mm"]],
            "radius": dims["adapter_radius_mm"],
            "model": "nominal",
        },
        {
            "name": "rmp60_body_nominal",
            "type": "capsule",
            "a": [0.0, 0.0, dims["adapter_length_mm"]],
            "b": [0.0, 0.0, dims["adapter_length_mm"] + dims["probe_body_length_mm"]],
            "radius": dims["probe_body_radius_mm"],
            "model": "nominal",
        },
        {
            "name": "center_support_stem",
            "type": "capsule",
            "a": nominal_origin,
            "b": cross_hub,
            "radius": dims["center_stem_collision_radius_mm"],
            "model": "real_five_way_stylus",
        },
    ]

    # 非活动分支：使用校准偏置（若有）或名义偏置构建测针杆 + 球头
    for branch_name, direction in branch_dirs.items():
        if branch_name == active_branch:
            continue
        calibrated_offset = (collision_offsets or {}).get(branch_name)
        if calibrated_offset is not None:
            center = [float(value) for value in calibrated_offset]
            hub = branch_collision_hub_mm(geometry, branch_name, center, dims)
            vector = sub(center, hub)
            if norm(vector) <= 1e-9:
                raise ValueError(f"{branch_name} collision offset coincides with the stylus hub")
            primitives.append(
                {
                    "name": f"calibrated_{branch_name}_stem",
                    "type": "capsule",
                    "a": hub,
                    "b": center,
                    "radius": branch_rod_collision_radius(branch_name, dims),
                    "model": "calibrated_other_branch",
                }
            )
            primitives.append(
                {
                    "name": f"calibrated_{branch_name}_ball",
                    "type": "sphere",
                    "center": center,
                    "radius": dims["ball_radius_mm"],
                    "model": "calibrated_other_branch",
                }
            )
            continue
        center = branch_local_offset_mm(geometry, branch_name)
        hub = branch_collision_hub_mm(geometry, branch_name, center, dims)
        primitives.append(
            {
                "name": f"nominal_{branch_name}_stem",
                "type": "capsule",
                "a": hub,
                "b": center,
                "radius": branch_rod_collision_radius(branch_name, dims),
                "model": "nominal_other_branch",
            }
        )
        primitives.append(
            {
                "name": f"nominal_{branch_name}_ball",
                "type": "sphere",
                "center": center,
                "radius": dims["ball_radius_mm"],
                "model": "nominal_other_branch",
            }
        )

    # 活动分支的测针杆（减去 stem_exclusion 避免自碰撞）
    active_rod_length = max(0.0, calibrated_active_length - args.target_stem_exclusion_mm)
    primitives.append(
        {
            "name": f"calibrated_{active_branch}_stem",
            "type": "capsule",
            "a": active_origin,
            "b": add(active_origin, scale(calibrated_active_direction, active_rod_length)),
            "radius": branch_rod_collision_radius(active_branch, dims),
            "model": "calibrated_active_branch",
        }
    )
    return primitives, dims, active_origin


def primitive_clearance_to_sphere(primitive, pose, sphere_center, sphere_radius, euler_sequence):
    """
    计算单个代理几何体到标定球面的最小距离（间隙）。

    对于球体代理：距离 = ||球心 - 标定球心|| - 两球半径和
    对于胶囊体代理：距离 = 点到线段距离 - 胶囊半径 - 标定球半径

    正值表示有间隙，负值表示发生碰撞。

    Args:
        primitive: 代理几何体字典（含 type, center/a/b, radius 等）
        pose: 法兰六维位姿（用于将几何体从局部坐标系变换到世界坐标系）
        sphere_center: 标定球球心
        sphere_radius: 标定球半径
        euler_sequence: 欧拉角旋转序列（如 "xyz"）

    Returns:
        间隙距离（mm），负值表示碰撞
    """
    rotation = euler_to_matrix(euler_sequence, pose[3:6])
    if primitive["type"] == "sphere":
        center = transform_point(pose, rotation, primitive["center"])
        distance = norm(sub(center, sphere_center))
    elif primitive["type"] == "capsule":
        start = transform_point(pose, rotation, primitive["a"])
        end = transform_point(pose, rotation, primitive["b"])
        distance = point_segment_distance(sphere_center, start, end)
    else:
        raise ValueError(f"unknown primitive type {primitive['type']!r}")
    return distance - sphere_radius - float(primitive["radius"])


def collision_summary(pose, primitives, sphere_center, sphere_radius, args):
    """
    评估指定位姿下所有代理几何体与标定球之间的碰撞状态。

    取所有几何体中间隙最小的那个作为最终评估结果。

    Args:
        pose: 法兰六维位姿
        primitives: 代理几何体列表
        sphere_center: 标定球球心
        sphere_radius: 标定球半径
        args: 命令行参数（含 min_probe_clearance_mm 等阈值）

    Returns:
        {
            "status": "COLLISION"|"LOW_CLEARANCE"|"OK",
            "clearance": 最小间隙值,
            "primitive": 最近的几何体名称,
            "primitive_model": 几何体所属模型类型,
            "obstacle": 障碍物名称（固定为 "calibration_sphere"）
        }
    """
    clearances = [
        (primitive_clearance_to_sphere(primitive, pose, sphere_center, sphere_radius, args.euler_sequence), primitive)
        for primitive in primitives
    ]
    clearance, primitive = min(clearances, key=lambda item: item[0])
    if clearance < 0.0:
        status = "COLLISION"
    elif clearance < args.min_probe_clearance_mm:
        status = "LOW_CLEARANCE"
    else:
        status = "OK"
    return {
        "status": status,
        "clearance": clearance,
        "primitive": primitive["name"],
        "primitive_model": primitive.get("model", ""),
        "obstacle": "calibration_sphere",
    }


def collision_severity(summary):
    """
    将碰撞状态映射为数值严重程度等级。

    Args:
        summary: collision_summary 返回的字典

    Returns:
        0=OK, 1=LOW_CLEARANCE, 2=COLLISION
    """
    return {"OK": 0, "LOW_CLEARANCE": 1, "COLLISION": 2}[summary["status"]]


def worst_collision_summary(summaries):
    """
    从多个碰撞评估结果中找出最严重的一个。

    先按严重程度排序，同等程度下间隙更小的更严重。

    Args:
        summaries: collision_summary 结果列表

    Returns:
        最严重的那个摘要
    """
    return max(summaries, key=lambda item: (collision_severity(item), -item["clearance"]))


def primitive_clearance_to_table_plane(primitive, pose, table_z_mm, euler_sequence):
    """
    计算单个代理几何体到桌面平面的最小 Z 方向间隙。

    用于防止探针或适配器在运动过程中碰撞桌面。
    桌面假设为水平面（Z = table_z_mm）。

    Args:
        primitive: 代理几何体字典
        pose: 法兰六维位姿
        table_z_mm: 桌面 Z 坐标
        euler_sequence: 欧拉角旋转序列

    Returns:
        间隙距离（mm），负值表示已穿透桌面
    """
    rotation = euler_to_matrix(euler_sequence, pose[3:6])
    if primitive["type"] == "sphere":
        center = transform_point(pose, rotation, primitive["center"])
        min_z = center[2] - float(primitive["radius"])
    elif primitive["type"] == "capsule":
        start = transform_point(pose, rotation, primitive["a"])
        end = transform_point(pose, rotation, primitive["b"])
        min_z = min(start[2], end[2]) - float(primitive["radius"])
    else:
        raise ValueError(f"unknown primitive type {primitive['type']!r}")
    return min_z - float(table_z_mm)


def table_plane_collision_summary(pose, primitives, args):
    """
    评估指定位姿下所有几何体与桌面之间的碰撞状态。

    Args:
        pose: 法兰六维位姿
        primitives: 代理几何体列表
        args: 命令行参数（含 table_plane_z_mm、min_table_clearance_mm）

    Returns:
        {
            "status": "COLLISION"|"LOW_CLEARANCE"|"OK",
            "clearance": 最小 Z 间隙,
            "primitive": 最近几何体名称（加 "table_plane:" 前缀）,
            "primitive_model": 几何体所属模型类型,
            "obstacle": "table_plane"
        }
    """
    clearances = [
        (primitive_clearance_to_table_plane(primitive, pose, args.table_plane_z_mm, args.euler_sequence), primitive)
        for primitive in primitives
    ]
    clearance, primitive = min(clearances, key=lambda item: item[0])
    if clearance < 0.0:
        status = "COLLISION"
    elif clearance < args.min_table_clearance_mm:
        status = "LOW_CLEARANCE"
    else:
        status = "OK"
    return {
        "status": status,
        "clearance": clearance,
        "primitive": f"table_plane:{primitive['name']}",
        "primitive_model": primitive.get("model", ""),
        "obstacle": "table_plane",
    }


def combined_collision_summary(pose, primitives, sphere_center, sphere_radius, args):
    """
    综合球体碰撞和桌面碰撞的评估结果，返回最严重的一项。

    Args:
        pose: 法兰六维位姿
        primitives: 代理几何体列表
        sphere_center: 标定球球心
        sphere_radius: 标定球半径
        args: 命令行参数

    Returns:
        最严重的碰撞摘要
    """
    summaries = [collision_summary(pose, primitives, sphere_center, sphere_radius, args)]
    if not args.disable_table_plane_check:
        summaries.append(table_plane_collision_summary(pose, primitives, args))
    return worst_collision_summary(summaries)


def sample_pose_segment_clearance(start_pose, end_pose, primitives, sphere_center, sphere_radius, args):
    """
    在线段路径上等间距采样，评估整段路径的碰撞状态。

    用于检查 transition→start 和 start→target 等运动路径上
    所有中间位姿的碰撞风险。采用等间距采样近似连续检查。

    Args:
        start_pose: 起始六维位姿
        end_pose: 终止六维位姿
        primitives: 代理几何体列表
        sphere_center: 标定球球心
        sphere_radius: 标定球半径
        args: 命令行参数（含 collision_segment_samples 采样数）

    Returns:
        整段路径中最严重的碰撞摘要
    """
    samples = max(2, int(args.collision_segment_samples))
    worst = None
    for index in range(samples):
        t = index / float(samples - 1)
        pose = [
            float(start_pose[axis]) + (float(end_pose[axis]) - float(start_pose[axis])) * t
            for axis in range(6)
        ]
        summary = combined_collision_summary(pose, primitives, sphere_center, sphere_radius, args)
        if worst is None or collision_severity(summary) > collision_severity(worst) or (
            summary["status"] == worst["status"] and summary["clearance"] < worst["clearance"]
        ):
            worst = summary
    return worst


def choose_orientations(args, reference_fit, current_orientation=None):
    """
    确定规划中使用的朝向列表。

    有三种来源（优先级从高到低）：
    1. 命令行显式指定 --orientation（只取 rx, ry, rz）
    2. --orientation-mode 为 "current" 时使用机器人当前朝向
    3. 默认从参考标定的残差行中提取各采样点的朝向并循环使用

    Args:
        args: 命令行参数
        reference_fit: 参考标定结果
        current_orientation: 机器人当前朝向（仅 mode=current 时需要）

    Returns:
        朝向列表，每个元素为 [rx, ry, rz]

    Raises:
        ValueError: mode=current 但未提供朝向时抛出；
                    参考标定无残差行且未提供朝向时抛出
    """
    if args.orientation:
        return [list(args.orientation[3:6])]
    if args.orientation_mode == "current":
        if current_orientation is None:
            raise ValueError("--orientation-mode current requires live robot pose")
        return [list(current_orientation)]
    orientations = residual_orientations(reference_fit)
    if not orientations:
        raise ValueError("reference fit has no residual row orientations; provide --orientation")
    return orientations


def make_plan(args, reference_fit, geometry, current_orientation=None, seed_offsets=None):
    """
    生成完整的五分支球心标定规划。

    这是规划生成阶段的核心函数。对每个分支的每个采样点：
    1. 确定使用的偏置值（种子偏置 / 参考偏置 / 名义偏置）
    2. 计算逼近方向（approach），为分支局部方向旋转到世界坐标系
    3. 计算关键位姿：
       - contact_pose: 球头刚好接触标定球时的法兰位姿
       - transition_pose: 安全远距离位姿（standoff + transition_standoff）
       - start_pose: 逼近起点位姿（standoff 距离外）
       - target_pose: 探测终点位姿（overtravel 穿透量）
    4. 关键几何关系：
       ball_center = flange_position + R * local_offset        （球头中心）
       sphere_center = ball_center + approach * contact_distance  （球面方程）
       => flange_position = sphere_center - approach * d - R * offset
    5. 对 transition、start、target 位姿及运动路径做碰撞检测
    6. 输出 CSV 行，包含所有位姿、碰撞状态、探针尺寸等信息

    Args:
        args: 命令行参数
        reference_fit: 参考标定结果
        geometry: 探针几何配置
        current_orientation: 机器人当前朝向（可选）
        seed_offsets: 种子偏置字典（可选）

    Returns:
        规划项列表，每项为包含 row（CSV行）、branch、approach、各关键位姿等的字典
    """
    # 从参考标定中获取固定的球心坐标
    sphere_center = [float(v) for v in reference_fit["sphere_center_mm"]]
    # 接触距离 = 球半径 + 探针球头半径（球心到球心）
    contact_distance = float(
        reference_fit.get("contact_center_distance_mm", args.sphere_radius_mm + args.probe_radius_mm)
    )
    orientations = choose_orientations(args, reference_fit, current_orientation=current_orientation)
    rows = []

    for branch in args.branches:
        active_offset, offset_source = offset_source_for_branch(branch, reference_fit, geometry, seed_offsets)
        # 分支名义方向（在法兰局部坐标系下）
        local_direction = branch_direction(geometry, branch)
        primitives, dims, active_origin = local_collision_primitives(geometry, branch, active_offset, args)
        for sample_index in range(1, args.samples_per_branch + 1):
            orientation = orientations[(sample_index - 1) % len(orientations)]
            # 旋转矩阵将局部坐标变换到世界坐标
            rotation = euler_to_matrix(args.euler_sequence, orientation)
            # 逼近方向 = R * local_direction（分支方向旋转到世界坐标系）
            approach = normalize(mat_vec(rotation, local_direction), f"{branch} sample {sample_index} approach")
            # 偏置旋转到世界坐标系：rotated_offset = R * active_offset
            rotated_offset = mat_vec(rotation, active_offset)

            # 球头中心刚好接触标定球表面时的位置
            contact_ball_center = sub(sphere_center, scale(approach, contact_distance))
            # transition 位姿的球头中心（距离更远，安全距离）
            transition_ball_center = sub(contact_ball_center, scale(approach, args.transition_standoff_mm))
            # start 位姿的球头中心（standoff 距离外）
            start_ball_center = sub(contact_ball_center, scale(approach, args.standoff_mm))
            # target 位姿的球头中心（过穿量，确保触发 DI1）
            target_ball_center = add(contact_ball_center, scale(approach, args.overtravel_mm))

            # 法兰位姿 = 球头中心 - 旋转后的偏置
            # 即：flange_position = ball_center - R * offset
            transition_pose = pose_with_position(sub(transition_ball_center, rotated_offset), orientation)
            start_pose = pose_with_position(sub(start_ball_center, rotated_offset), orientation)
            contact_pose = pose_with_position(sub(contact_ball_center, rotated_offset), orientation)
            target_pose = pose_with_position(sub(target_ball_center, rotated_offset), orientation)

            # 对各个关键位姿和运动路径进行碰撞检测
            collision_checks = [
                combined_collision_summary(transition_pose, primitives, sphere_center, args.sphere_radius_mm, args),
                combined_collision_summary(start_pose, primitives, sphere_center, args.sphere_radius_mm, args),
                combined_collision_summary(contact_pose, primitives, sphere_center, args.sphere_radius_mm, args),
                combined_collision_summary(target_pose, primitives, sphere_center, args.sphere_radius_mm, args),
                sample_pose_segment_clearance(
                    transition_pose,
                    start_pose,
                    primitives,
                    sphere_center,
                    args.sphere_radius_mm,
                    args,
                ),
                sample_pose_segment_clearance(
                    start_pose,
                    target_pose,
                    primitives,
                    sphere_center,
                    args.sphere_radius_mm,
                    args,
                ),
            ]
            if args.safe_pose is not None:
                collision_checks.append(
                    sample_pose_segment_clearance(
                        args.safe_pose,
                        transition_pose,
                        primitives,
                        sphere_center,
                        args.sphere_radius_mm,
                        args,
                    )
                )
            worst_collision = worst_collision_summary(collision_checks)
            plan_id = f"{branch}_{sample_index}"
            safe_transition_orientation_delta = None
            if args.safe_pose is not None:
                safe_transition_orientation_delta = max_orientation_delta_deg(transition_pose, args.safe_pose)

            # 构建 CSV 输出行
            row = {
                "plan_id": plan_id,
                "branch": branch,
                "sample_in_branch": str(sample_index),
                "physical_ball_id": str(args.physical_ball_id),
                "approach_x": f"{approach[0]:.6f}",
                "approach_y": f"{approach[1]:.6f}",
                "approach_z": f"{approach[2]:.6f}",
                "sphere_center_x": f"{sphere_center[0]:.6f}",
                "sphere_center_y": f"{sphere_center[1]:.6f}",
                "sphere_center_z": f"{sphere_center[2]:.6f}",
                "contact_center_distance_mm": f"{contact_distance:.6f}",
                "offset_source": offset_source,
                "initial_offset_x": f"{active_offset[0]:.6f}",
                "initial_offset_y": f"{active_offset[1]:.6f}",
                "initial_offset_z": f"{active_offset[2]:.6f}",
                "estimated_branch_origin_x": f"{active_origin[0]:.4f}",
                "estimated_branch_origin_y": f"{active_origin[1]:.4f}",
                "estimated_branch_origin_z": f"{active_origin[2]:.4f}",
                "standoff_mm": f"{args.standoff_mm:.4f}",
                "transition_standoff_mm": f"{args.transition_standoff_mm:.4f}",
                "overtravel_mm": f"{args.overtravel_mm:.4f}",
                "probe_travel_mm": f"{args.standoff_mm + args.overtravel_mm:.4f}",
                "safe_transition_orientation_delta_deg": (
                    "" if safe_transition_orientation_delta is None else f"{safe_transition_orientation_delta:.4f}"
                ),
                "max_safe_transition_orientation_delta_deg": f"{args.max_safe_transition_orientation_delta_deg:.4f}",
                "probe_collision_status": worst_collision["status"],
                "min_probe_clearance_mm": f"{worst_collision['clearance']:.4f}",
                "closest_probe_part": worst_collision["primitive"],
                "closest_probe_model": worst_collision["primitive_model"],
                "closest_obstacle": worst_collision["obstacle"],
                "probe_adapter_length_mm": f"{dims['adapter_length_mm']:.4f}",
                "probe_adapter_radius_mm": f"{dims['adapter_radius_mm']:.4f}",
                "probe_body_length_mm": f"{dims['probe_body_length_mm']:.4f}",
                "probe_body_radius_mm": f"{dims['probe_body_radius_mm']:.4f}",
                "probe_branch_center_distance_mm": f"{dims['branch_center_distance_mm']:.4f}",
                "probe_cross_hub_z_mm": f"{dims['flange_to_cross_hub_mm']:.4f}",
                "probe_configured_rod_radius_mm": f"{dims['configured_rod_radius_mm']:.4f}",
                "probe_rod_collision_radius_mm": f"{dims['rod_collision_radius_mm']:.4f}",
                "probe_side_rod_collision_radius_mm": f"{dims['side_rod_collision_radius_mm']:.4f}",
                "probe_axial_rod_collision_radius_mm": f"{dims['axial_rod_collision_radius_mm']:.4f}",
                "probe_center_stem_collision_radius_mm": f"{dims['center_stem_collision_radius_mm']:.4f}",
                "probe_ball_radius_mm": f"{dims['ball_radius_mm']:.4f}",
                "target_stem_exclusion_mm": f"{args.target_stem_exclusion_mm:.4f}",
                "collision_segment_samples": str(args.collision_segment_samples),
                "table_plane_check": str(int(not args.disable_table_plane_check)),
                "table_plane_z_mm": f"{args.table_plane_z_mm:.4f}",
                "min_table_clearance_mm": f"{args.min_table_clearance_mm:.4f}",
                "reference_fit_json": str(args.reference_fit_json),
                "euler_sequence": args.euler_sequence,
            }
            if args.safe_pose is not None:
                row.update(pose_dict("safe_flange", args.safe_pose))
            row.update(pose_dict("transition_flange", transition_pose))
            row.update(pose_dict("start_flange", start_pose))
            row.update(pose_dict("contact_flange", contact_pose))
            row.update(pose_dict("target_flange", target_pose))
            rows.append(
                {
                    "row": row,
                    "branch": branch,
                    "approach": approach,
                    "sphere_center": sphere_center,
                    "contact_distance": contact_distance,
                    "initial_offset": active_offset,
                    "transition_pose": transition_pose,
                    "start_pose": start_pose,
                    "contact_pose": contact_pose,
                    "target_pose": target_pose,
                }
            )
    return rows


def write_csv(path, dict_rows):
    """
    将字典列表写入 CSV 文件。

    自动创建父目录，字段名取自第一个字典的键。

    Args:
        path: 输出文件路径
        dict_rows: 字典列表，每项对应 CSV 中的一行

    Raises:
        ValueError: dict_rows 为空时抛出
    """
    if not dict_rows:
        raise ValueError("no rows to write")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dict_rows)


def pose_from_plan_row(row, prefix):
    """
    从 CSV 行的带前缀字段中解析六维位姿。

    Args:
        row: CSV 行字典
        prefix: 列名前缀（如 "transition_flange"）

    Returns:
        六维位姿 [x, y, z, rx, ry, rz]
    """
    return [parse_float(row.get(f"{prefix}_{name}"), f"{prefix}_{name}") for name in POSE_NAMES]


def plan_from_csv(path):
    """
    从 CSV 文件加载规划。

    从 CSV 行中提取分支名、逼近方向、球心、偏置和各关键位姿。

    Args:
        path: CSV 文件路径

    Returns:
        规划项列表，结构与 make_plan 返回值相同

    Raises:
        ValueError: CSV 为空时抛出
    """
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            branch = row.get("branch", "")
            approach = normalize(
                [
                    parse_float(row.get("approach_x"), "approach_x"),
                    parse_float(row.get("approach_y"), "approach_y"),
                    parse_float(row.get("approach_z"), "approach_z"),
                ],
                "approach",
            )
            sphere_center = [
                parse_float(row.get("sphere_center_x"), "sphere_center_x"),
                parse_float(row.get("sphere_center_y"), "sphere_center_y"),
                parse_float(row.get("sphere_center_z"), "sphere_center_z"),
            ]
            initial_offset = [
                parse_float(row.get("initial_offset_x"), "initial_offset_x"),
                parse_float(row.get("initial_offset_y"), "initial_offset_y"),
                parse_float(row.get("initial_offset_z"), "initial_offset_z"),
            ]
            item = {
                "row": row,
                "branch": branch,
                "approach": approach,
                "sphere_center": sphere_center,
                "contact_distance": parse_float(row.get("contact_center_distance_mm"), "contact_center_distance_mm"),
                "initial_offset": initial_offset,
                "transition_pose": pose_from_plan_row(row, "transition_flange"),
                "start_pose": pose_from_plan_row(row, "start_flange"),
                "contact_pose": pose_from_plan_row(row, "contact_flange"),
                "target_pose": pose_from_plan_row(row, "target_flange"),
            }
            if all(row.get(f"safe_flange_{name}") not in ("", None) for name in POSE_NAMES):
                item["safe_pose"] = pose_from_plan_row(row, "safe_flange")
            rows.append(item)
    if not rows:
        raise ValueError(f"{path}: no plan rows")
    return rows


def safe_pose_for_item(args, item):
    """Return per-row safe pose when a loaded plan provides one."""
    return item.get("safe_pose") or args.safe_pose


def append_stable_row(path, row):
    """
    以追加模式将一行数据写入 CSV 文件。

    如果文件不存在则先写表头；如果已存在则校验表头一致性。

    Args:
        path: CSV 文件路径
        row: 要写入的字典行

    Raises:
        ValueError: 已存在文件的表头与当前行键不一致时抛出
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    if exists:
        with output.open(newline="") as f:
            header = next(csv.reader(f), None)
        if header != list(row.keys()):
            raise ValueError(f"CSV header mismatch for {output}; choose a new --output")
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def print_plan_summary(plan):
    """
    打印规划摘要到控制台。

    显示每个规划项的 ID、分支、逼近方向、起点/终点位姿和碰撞状态。

    Args:
        plan: 规划项列表
    """
    print(f"planned branch samples: {len(plan)}")
    for item in plan:
        row = item["row"]
        print(
            f"  {row['plan_id']} branch={row['branch']} "
            f"approach=[{row['approach_x']}, {row['approach_y']}, {row['approach_z']}] "
            f"start={format_pose(item['start_pose'])} "
            f"target={format_pose(item['target_pose'])} "
            f"collision={row['probe_collision_status']} "
            f"clearance={row['min_probe_clearance_mm']}mm "
            f"part={row['closest_probe_part']}"
        )


def current_pose_from_robot(args):
    """
    连接到机器人并获取当前法兰位姿。

    初始化 ROS2 节点、创建 MovJ 服务客户端、等待服务就绪后读取位姿。

    Args:
        args: 命令行参数（含 service_timeout_sec）

    Returns:
        (node, pose) 元组，node 为 ProbeTouch 实例，pose 为六维位姿
    """
    rclpy.init()
    node = ProbeTouch()
    node.movj_cli = node.create_client(MovJ, "/dobot_bringup_ros2/srv/MovJ")
    try:
        node.wait_services(args.service_timeout_sec)
        if not node.movj_cli.wait_for_service(timeout_sec=args.service_timeout_sec):
            raise RuntimeError(f"service not available: {node.movj_cli.srv_name}")
        node.wait_fresh_feed()
        pose = current_pose(node, max_age_sec=0.5)
        return node, pose
    except Exception:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        raise


def move_to_pose_guarded(node, pose, args, label):
    """
    带保护的运动：移动到指定位姿，全程监测 DI1 触发信号。

    这是运动安全的核心实现。运动逻辑：
    1. 记录运动前位姿作为反向退出基准
    2. 根据 label 和 safe_transition_move 选择 MovJ 或 MovL
    3. 发送运动指令后调用 wait_jog 等待到达，同时监测 DI1
    4. 如果 DI1 提前触发：
       a. 获取触发时刻的停止位姿
       b. 沿来路方向反向退回 positioning_trigger_retract_mm 距离
       c. 确认退回后 DI1 恢复正常
       d. 抛出 PositioningTriggerError
    5. 如果超时未到达目标，抛出 RuntimeError

    Args:
        node: ProbeTouch 节点实例
        pose: 目标六维位姿
        args: 命令行参数
        label: 运动阶段标签（用于日志和异常消息）

    Raises:
        PositioningTriggerError: DI1 在运动过程中提前触发
        RuntimeError: 超时未到达目标位姿
    """
    previous_pose = current_pose(node, max_age_sec=0.5)
    # transition 和 safe_pose 阶段可以选择 MovJ 快速移动
    use_movj = args.safe_transition_move == "movj" and (
        label.startswith("transition ") or label.startswith("safe pose ")
    )
    future = issue_movj(node, pose) if use_movj else issue_movl(node, pose)
    trigger_snapshot, reached_target = wait_jog(
        node,
        pose,
        args.positioning_timeout_sec,
        args.position_tolerance_mm,
        args.orientation_tolerance_deg,
        command_future=future,
        command_client=node.movj_cli if use_movj else node.movl_cli,
    )
    if trigger_snapshot is not None:
        # DI1 在定位过程中触发了！立即停止并反向退回
        stop_pose = trigger_snapshot.get("pose") or current_pose(node, max_age_sec=0.5)
        # 沿来路方向计算退回避让向量
        reverse = sub(previous_pose[:3], stop_pose[:3])
        reverse_length = norm(reverse)
        if reverse_length <= 1e-9:
            raise RuntimeError(f"DI1 triggered while moving to {label}; no reverse path is available")
        retract_distance = min(float(args.positioning_trigger_retract_mm), reverse_length)
        recovery_pose = list(stop_pose)
        for index in range(3):
            recovery_pose[index] += reverse[index] / reverse_length * retract_distance
        print(
            f"DI1 triggered during positioning; retracting {retract_distance:.4f}mm "
            f"along the verified reverse path to {format_pose(recovery_pose)}"
        )
        # 执行反向退回运动
        recovery_future = issue_movl(node, recovery_pose)
        node.wait_until_pose(
            recovery_pose,
            position_tolerance_mm=args.position_tolerance_mm,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
            timeout_sec=args.positioning_timeout_sec,
            command_future=recovery_future,
        )
        node.wait_fresh_feed()
        # 确认退回后 DI1 恢复正常，否则可能卡住需要人工干预
        if fresh_feed_di1(node, max_age_sec=0.5):
            raise RuntimeError(
                f"DI1 remains triggered after reverse-path recovery from {label}; manual clearance required"
            )
        raise PositioningTriggerError(label, trigger_snapshot, stop_pose, recovery_pose)
    if not reached_target:
        raise RuntimeError(f"did not reach {label}")


def issue_movj(node, pose):
    """
    发送 MovJ（关节空间运动）请求到机器人。

    MovJ 用于快速移动，不保证末端直线轨迹。
    主要用于 safe_pose 到 transition 的远距离移动。

    Args:
        node: ProbeTouch 节点实例
        pose: 目标六维位姿

    Returns:
        Future 对象，用于检查异步结果
    """
    request = MovJ.Request()
    request.mode = False
    request.a, request.b, request.c, request.d, request.e, request.f = [float(value) for value in pose]
    future, _ = node.call_async_checked(node.movj_cli, request)
    return future


def move_to_pose_guarded_and_confirm_clear(node, pose, args, label):
    """
    带保护的运动并确认 DI1 最终状态为未触发。

    在 move_to_pose_guarded 基础上增加到达后的 DI1 确认检查。
    用于 transition 和 start 阶段，确保运动路径全程无触碰。

    Args:
        node: ProbeTouch 节点实例
        pose: 目标六维位姿
        args: 命令行参数
        label: 运动阶段标签

    Raises:
        RuntimeError: 到达目标位姿后 DI1 仍处于触发状态
    """
    move_to_pose_guarded(node, pose, args, label)
    node.wait_fresh_feed()
    if fresh_feed_di1(node, max_age_sec=0.5):
        raise RuntimeError(f"DI1 is triggered after moving to {label}; path is not clear")


def retract_to_pose_after_probe(node, pose, args, label):
    """
    探测完成（HIT 或 MISS）后退回安全位姿。

    探测结束后 DI1 可能仍处于触发状态（HIT 时），
    此函数在退回过程中仍会检测 DI1，确保退回后完全脱离接触。

    Args:
        node: ProbeTouch 节点实例
        pose: 目标退回位姿（通常为 transition_pose）
        args: 命令行参数
        label: 标签（用于错误消息）

    Raises:
        RuntimeError: 退回后 DI1 仍然触发，需要人工干预
    """
    future = issue_movl(node, pose)
    node.wait_until_pose(
        pose,
        position_tolerance_mm=args.position_tolerance_mm,
        orientation_tolerance_deg=args.orientation_tolerance_deg,
        timeout_sec=args.positioning_timeout_sec,
        command_future=future,
    )
    node.wait_fresh_feed()
    if fresh_feed_di1(node, max_age_sec=0.5):
        raise RuntimeError(f"DI1 remains triggered after {label}; manual clearance required")


def contact_row(args, item, trigger_snapshot, stop_pose, status):
    """
    构建一次接触探测的结果行，用于写入 CSV。

    包含时间戳、规划 ID、状态、分支信息、触发时刻快照等。
    trigger_snapshot 为 None 表示未触发（MISS）。

    Args:
        args: 命令行参数
        item: 规划项字典
        trigger_snapshot: 触发时刻的快照（含位姿、序列号、IO 状态等）
        stop_pose: 停止位姿
        status: 状态字符串（"HIT" / "MISS" / "EARLY_TRIGGER_*" / "NOT_RUN"）

    Returns:
        包含所有字段的字典，可直接写入 CSV
    """
    trigger_pose = trigger_snapshot.get("pose") if trigger_snapshot else None
    row = {
        "timestamp": f"{time.time():.3f}",
        "plan_id": item["row"]["plan_id"],
        "status": status,
        "session_id": args.session_id,
        "workpiece_id": args.workpiece_id,
        "artifact_id": args.artifact_id,
        "artifact_type": args.artifact_type,
        "standard_sphere_id": str(item["row"].get("standard_sphere_id", "")),
        "physical_ball_id": str(
            item["row"].get("physical_ball_id") or args.physical_ball_id
        ),
        "branch": item["branch"],
        "operator_note": args.operator_note,
        "reference_fit_json": str(item["row"].get("reference_fit_json") or args.reference_fit_json),
        "euler_sequence": args.euler_sequence,
        "approach_x": f"{item['approach'][0]:.6f}",
        "approach_y": f"{item['approach'][1]:.6f}",
        "approach_z": f"{item['approach'][2]:.6f}",
        "sphere_center_x": f"{item['sphere_center'][0]:.6f}",
        "sphere_center_y": f"{item['sphere_center'][1]:.6f}",
        "sphere_center_z": f"{item['sphere_center'][2]:.6f}",
        "contact_center_distance_mm": f"{item['contact_distance']:.6f}",
        "trigger_feed_sequence": "" if trigger_snapshot is None else str(trigger_snapshot.get("sequence", "")),
        "trigger_feed_wall_time": (
            "" if trigger_snapshot is None or trigger_snapshot.get("wall_time") is None
            else f"{float(trigger_snapshot['wall_time']):.6f}"
        ),
        "trigger_digital_input_bits": (
            "" if trigger_snapshot is None or trigger_snapshot.get("digital_input_bits") is None
            else str(trigger_snapshot["digital_input_bits"])
        ),
        "trigger_di1": (
            "" if trigger_snapshot is None or trigger_snapshot.get("di1") is None
            else str(int(bool(trigger_snapshot["di1"])))
        ),
    }
    row.update(pose_fields("start_flange", item["start_pose"]))
    row.update(pose_fields("target_flange", item["target_pose"]))
    row.update(pose_fields("trigger_flange", trigger_pose))
    row.update(pose_fields("stop_flange", stop_pose))
    row.update(vector_fields("trigger_joint", None if trigger_snapshot is None else trigger_snapshot.get("joints"), 6))
    for field in (
        "dataset_id",
        "dataset_mode",
        "dataset_group",
        "dataset_split",
        "dataset_pose_family",
        "dataset_branch_point_index",
        "dataset_repeat_index",
        "dataset_source_plan_id",
    ):
        row[field] = str(item["row"].get(field, ""))
    return row


def _recompute_item_poses_for_standoff(item, new_standoff):
    """
    根据新的 standoff 值重新计算规划项的 start/target 法兰位姿。

    当使用 --per-branch-standoff 为特定分支单独设置 standoff 时，
    需要重新计算该分支的起点（start）和探测终点（target）位姿。

    关键逻辑：
    - transition_pose 保持不变（始终从同一安全距离开始逼近）
    - 朝向（orientation）从 transition_pose 复制
    - start_ball = sphere_center - approach * (contact_distance + new_standoff)
    - target_ball = sphere_center - approach * (contact_distance - overtravel)
    - flange_position = ball_center - rotated_offset

    Args:
        item: 原始规划项字典
        new_standoff: 新的 standoff 值（mm）

    Returns:
        新的规划项字典（浅拷贝，仅更新了与位姿相关的字段）
    """
    approach = item["approach"]
    sphere_center = item["sphere_center"]
    contact_distance = item["contact_distance"]
    overtravel = float(item["row"].get("overtravel_mm", 0.2))
    orientation = list(item["contact_pose"][3:6])

    # 从原计划的 contact_pose 反推旋转后的偏置（rotated_offset）。
    # 这样不依赖 transition_standoff_mm 的默认值，避免不同规划脚本默认值变化时
    # 把球心路径重算到错误位置。
    contact_ball = sub(sphere_center, scale(approach, contact_distance))
    rotated_offset = sub(contact_ball, item["contact_pose"][:3])

    # 使用新的 standoff 重新计算 start/target 位姿
    start_ball = sub(sphere_center, scale(approach, contact_distance + new_standoff))
    target_ball = sub(sphere_center, scale(approach, contact_distance - overtravel))

    new_item = dict(item)
    # 法兰位姿 = 球头中心 - 偏置（已旋转到世界坐标系）
    new_item["start_pose"] = pose_with_position(sub(start_ball, rotated_offset), orientation)
    new_item["contact_pose"] = pose_with_position(sub(contact_ball, rotated_offset), orientation)
    new_item["target_pose"] = pose_with_position(sub(target_ball, rotated_offset), orientation)

    new_row = dict(item["row"])
    new_row["standoff_mm"] = f"{new_standoff:.4f}"
    new_row["probe_travel_mm"] = f"{new_standoff + overtravel:.4f}"
    new_row.update(pose_fields("start_flange", new_item["start_pose"]))
    new_row.update(pose_fields("contact_flange", new_item["contact_pose"]))
    new_row.update(pose_fields("target_flange", new_item["target_pose"]))
    new_item["row"] = new_row
    return new_item


def _group_plan_by_branch(plan):
    """
    将规划项按分支分组，保持各分支首次出现的顺序。

    用于执行阶段逐分支处理，每个分支内部保持采样顺序不变。

    Args:
        plan: 规划项列表

    Returns:
        [(branch_name, [item, ...]), ...] 格式的列表
    """
    groups = {}
    order = []
    for item in plan:
        branch = item["branch"]
        if branch not in groups:
            groups[branch] = []
            order.append(branch)
        groups[branch].append(item)
    return [(branch, groups[branch]) for branch in order]


def _verify_moveit_checked_plan(args, plan):
    """
    确认真实执行使用的是已通过 MoveIt 预检的计划。

    真实机器人执行时，CSV 必须包含 moveit_status 字段且每一行均为 OK。
    这可以防止误把未检查计划、MoveIt 失败计划或中间诊断候选表交给真实机器人。
    如确实需要执行未检查计划，必须显式传入 --allow-unchecked-plan。
    """
    missing = [item["row"].get("plan_id", "") for item in plan if "moveit_status" not in item["row"]]
    failed = [
        item["row"]
        for item in plan
        if "moveit_status" in item["row"]
        and str(item["row"].get("moveit_status", "")).strip().upper() != "OK"
    ]
    if (missing or failed) and not args.allow_unchecked_plan:
        details = []
        if missing:
            details.append("missing moveit_status: " + ", ".join(missing[:8]))
        if failed:
            details.append(
                "failed moveit_status: "
                + ", ".join(
                    f"{row.get('plan_id', '')}={row.get('moveit_status', '')}"
                    for row in failed[:8]
                )
            )
        raise RuntimeError(
            "real execution requires a MoveIt-checked plan with moveit_status=OK for every row; "
            + "; ".join(details)
            + ". Run scripts/check_five_branch_plan_moveit.py first, or pass "
            "--allow-unchecked-plan only after manual review."
        )
    blocked_entries = [
        item["row"] for item in plan
        if str(item["row"].get("batch_entry_preflight", "")).strip().upper() == "REQUIRED"
    ]
    if blocked_entries:
        raise RuntimeError(
            "batch entry is not authorized from the real current robot state: "
            + ", ".join(row.get("plan_id", "") for row in blocked_entries[:4])
            + ". Run and record a current-state -> first-safe MoveIt preflight for this batch first."
        )


def _verify_fixed_branch_orientation(args, plan):
    """
    默认要求同一分支内姿态基本固定。

    五向测针标定的现场安全策略是：每根针使用已确认的水平姿态，只在标准球附近
    做小范围平移和短距离触球。若同一分支内姿态变化较大，通常意味着计划使用了
    align-branch 这类重新求姿态策略，容易导致关节跳变或远距离运动。
    """
    if args.allow_varying_branch_orientation:
        return
    violations = []
    for branch, items in _group_plan_by_branch(plan):
        reference = items[0]["contact_pose"]
        spread = max(max_orientation_delta_deg(item["contact_pose"], reference) for item in items)
        if spread > args.max_branch_orientation_spread_deg:
            violations.append(f"{branch} spread={spread:.4f}deg")
    if violations:
        raise RuntimeError(
            "branch orientation is not fixed enough for near-field five-branch calibration: "
            + ", ".join(violations)
            + f"; limit={args.max_branch_orientation_spread_deg:.4f}deg. "
            "Use the fixed horizontal branch plan, or pass --allow-varying-branch-orientation "
            "only after manual MoveIt/RViz review."
        )


def _verify_plan_preflight(args, plan):
    """
    执行前的预检，对所有规划项进行安全检查。

    检查项：
    1. 朝向变化检查：safe_pose 到 transition 的朝向变化是否超限。
       过大的朝向变化可能导致机器人关节限位。
    2. 碰撞风险检查：所有规划项的碰撞状态必须为 OK。
       如果有 LOW_CLEARANCE 或 COLLISION，需要用户确认。

    每个检查在超标时默认抛出异常，除非用户使用了对应的
    --allow-* 参数手动确认风险。

    Args:
        args: 命令行参数
        plan: 规划项列表

    Raises:
        RuntimeError: 朝向变化超限或存在碰撞风险且未用户确认
    """
    _verify_moveit_checked_plan(args, plan)
    _verify_fixed_branch_orientation(args, plan)

    # 检查 safe_pose -> transition 的朝向变化是否过大
    large_orientation = [
        item["row"]
        for item in plan
        if item["row"].get("safe_transition_orientation_delta_deg") not in ("", None)
        and float(item["row"]["safe_transition_orientation_delta_deg"]) > args.max_safe_transition_orientation_delta_deg
    ]
    if large_orientation and not args.allow_large_safe_transition_orientation_change:
        detail = ", ".join(
            f"{row['plan_id']} {row['safe_transition_orientation_delta_deg']}deg"
            for row in large_orientation
        )
        raise RuntimeError(
            "safe->transition orientation change is too large; this can hit joint limits. "
            f"Use a closer taught safe pose, choose another orientation, or pass "
            f"--allow-large-safe-transition-orientation-change after manual review: {detail}"
        )
    # 检查碰撞状态
    risky = [item["row"] for item in plan if item["row"]["probe_collision_status"] != "OK"]
    if risky and not args.allow_probe_model_collision_risk:
        detail = ", ".join(
            f"{row['plan_id']} {row['probe_collision_status']} "
            f"{row['min_probe_clearance_mm']}mm {row['closest_probe_part']}"
            for row in risky
        )
        raise RuntimeError(
            "probe model collision check is not OK; inspect plan or use "
            f"--allow-probe-model-collision-risk after manual review: {detail}"
        )


def _resolve_standoff(args, item):
    """
    解析规划项实际使用的 standoff 值。

    如果 --per-branch-standoff 为该分支指定了覆盖值，则使用覆盖值；
    否则使用规划中原始 standoff 值。

    Args:
        args: 命令行参数（含 per_branch_standoff 字典）
        item: 规划项字典

    Returns:
        实际 standoff 值（mm）
    """
    branch = item["branch"]
    override = getattr(args, "per_branch_standoff", None)
    if override and branch in override:
        return float(override[branch])
    return float(item["row"].get("standoff_mm", args.standoff_mm))


def _execute_one_probe(node, args, item):
    """
    执行单次探测循环。

    探测循环包含四个阶段：
    1. Transition（过渡阶段）
       - 如果 safe_pose_policy == "each-sample"，先从当前位置回到 safe_pose
       - 计算从当前位置到 transition_pose 的距离和朝向差，检查是否超限（max_local_step）
       - 带保护运动到 transition_pose（安全远距离位姿）
       - 如果在过渡阶段发生 DI1 提前触发，记录为 EARLY_TRIGGER_TRANSITION

    2. Start（逼近起点阶段）
       - 从 transition_pose 带保护运动到 start_pose
       - 确认到达后 DI1 未被触发
       - 如果在逼近阶段发生 DI1 提前触发，记录为 EARLY_TRIGGER_START

    3. Probe（探测阶段）
       - 从 start_pose 沿直线（MovL）向 target_pose 运动
       - 使用 wait_jog 监测 DI1 信号
       - 如果 DI1 触发 -> HIT（接触成功）
       - 如果到达 target_pose 仍无触发 -> MISS（未接触）

    4. Retract（退回阶段）
       - HIT 或 MISS 后都退回 transition_pose
       - 使用 retract_to_pose_after_probe 保证退回安全

    Args:
        node: ProbeTouch 节点实例
        args: 命令行参数
        item: 规划项字典

    Returns:
        (status, trigger_snapshot, stop_pose) 三元组，
        status 为 "HIT" / "MISS" / "EARLY_TRIGGER_TRANSITION" / "EARLY_TRIGGER_START"
    """
    plan_id = item["row"]["plan_id"]

    # Phase 1: 回到 safe_pose（如果需要）
    item_safe_pose = safe_pose_for_item(args, item)
    if item_safe_pose is not None and args.safe_pose_policy == "each-sample":
        print(f"move to safe pose: {format_pose(item_safe_pose)}")
        move_to_pose_guarded_and_confirm_clear(node, item_safe_pose, args, f"safe pose before {plan_id}")

    # 检查到 transition 的移动距离和朝向变化是否在允许范围内
    current = current_pose(node, max_age_sec=0.5)
    transition_distance = norm(sub(item["transition_pose"][:3], current[:3]))
    transition_orientation_delta = max_orientation_delta_deg(item["transition_pose"], current)
    if args.max_local_step_mm > 0 and transition_distance > args.max_local_step_mm:
        raise RuntimeError(
            f"local move to transition {plan_id} is {transition_distance:.4f}mm; "
            f"limit is {args.max_local_step_mm:.4f}mm"
        )
    if (
        args.max_local_orientation_step_deg > 0
        and transition_orientation_delta > args.max_local_orientation_step_deg
    ):
        raise RuntimeError(
            f"local orientation move to transition {plan_id} is {transition_orientation_delta:.4f}deg; "
            f"limit is {args.max_local_orientation_step_deg:.4f}deg"
        )

    # Phase 2: 移动到 transition_pose（安全远距离位姿）
    print(f"move to transition: {format_pose(item['transition_pose'])}")
    try:
        move_to_pose_guarded_and_confirm_clear(node, item["transition_pose"], args, f"transition {plan_id}")
    except PositioningTriggerError as exc:
        append_stable_row(
            args.output,
            contact_row(args, item, exc.trigger_snapshot, exc.stop_pose, "EARLY_TRIGGER_TRANSITION"),
        )
        return ("EARLY_TRIGGER_TRANSITION", exc.trigger_snapshot, exc.stop_pose)

    # Phase 3: 移动到 start_pose（逼近起点）
    print(f"move to start: {format_pose(item['start_pose'])}")
    try:
        move_to_pose_guarded_and_confirm_clear(node, item["start_pose"], args, f"start {plan_id}")
    except PositioningTriggerError as exc:
        append_stable_row(
            args.output,
            contact_row(args, item, exc.trigger_snapshot, exc.stop_pose, "EARLY_TRIGGER_START"),
        )
        return ("EARLY_TRIGGER_START", exc.trigger_snapshot, exc.stop_pose)

    # 确认 start 位姿处 DI1 未触发
    if fresh_feed_di1(node, max_age_sec=0.5):
        raise RuntimeError(f"DI1 triggered at calibration start {plan_id}; path is not clear")

    # Phase 4: 探测阶段——沿直线向 target 移动并监测 DI1
    print(f"short probe target: {format_pose(item['target_pose'])}")
    future = issue_movl(node, item["target_pose"])
    trigger_snapshot, _ = wait_jog(
        node,
        item["target_pose"],
        args.timeout_sec,
        args.position_tolerance_mm,
        args.orientation_tolerance_deg,
        command_future=future,
    )
    stop_pose = current_pose(node, max_age_sec=0.5)
    if trigger_snapshot is None:
        # 到达 target 仍未触发 -> MISS
        print("  MISS")
        row = contact_row(args, item, None, stop_pose, "MISS")
        append_stable_row(args.output, row)
        retract_to_pose_after_probe(node, item["transition_pose"], args, f"retract transition {plan_id}")
        return ("MISS", None, stop_pose)

    # DI1 触发 -> HIT
    print("  HIT")
    row = contact_row(args, item, trigger_snapshot, stop_pose, "HIT")
    append_stable_row(args.output, row)
    if args.refit_after_each_hit:
        write_calibration(args, args._reference_fit, args._geometry)
    retract_to_pose_after_probe(node, item["transition_pose"], args, f"retract transition {plan_id}")
    return ("HIT", trigger_snapshot, stop_pose)


def _recover_to_safe_after_early_trigger(node, args, branch):
    """
    在分支发生提前触发后，将机器人移回 safe_pose。

    两阶段恢复策略：
    1. 尝试带保护的运动（guarded move）回到 safe_pose
    2. 如果也失败（可能 DI1 仍然触发），直接使用 MovJ 作为后备方案
       MovJ 是关节空间运动，可以绕过笛卡尔空间的障碍

    Args:
        node: ProbeTouch 节点实例
        args: 命令行参数
        branch: 触发提前触发的分支名称
    """
    if args.safe_pose is None:
        return
    try:
        move_to_pose_guarded_and_confirm_clear(node, args.safe_pose, args, f"safe pose after early trigger on {branch}")
    except (PositioningTriggerError, RuntimeError) as exc:
        print(f"  WARNING: could not reach safe pose after {branch} early trigger: {exc}")
        print(f"  attempting direct MovJ to safe pose as fallback …")
        # 后备方案：使用关节空间运动（MovJ）直接回到安全位姿
        future = issue_movj(node, args.safe_pose)
        node.wait_until_pose(
            args.safe_pose,
            position_tolerance_mm=args.position_tolerance_mm,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
            timeout_sec=args.positioning_timeout_sec,
            command_future=future,
            command_client=node.movj_cli,
        )


def execute_plan(node, args, plan):
    """
    执行五分支探测规划，带每分支提前触发恢复能力。

    执行流程：
    1. 预检（_verify_plan_preflight）
    2. 确认 DI1 初始状态为未触发
    3. 设定机器人速度
    4. 如果 safe_pose_policy 为 "plan" 或 "initial-only"，先移动到 safe_pose
    5. 按分支分组处理（_group_plan_by_branch）：
       对每个分支：
         a. 对分支内每个采样点：
            - 解析实际 standoff（支持 per-branch-standoff 覆盖）
            - 如果 standoff 与规划不同，重新计算位姿
            - 执行单次探测（_execute_one_probe）
            - 根据结果处理：
              * EARLY_TRIGGER：如果 --skip-branch-on-early-trigger 则跳过
                该分支剩余所有点（记录为 NOT_RUN），否则抛出异常中止
              * HIT：计数
              * MISS：计数
            - 如果 safe_pose_policy == "each-sample"，每次探测后回 safe_pose
         b. 如果 safe_pose_policy == "plan" 且不是最后一个分支，
            分支之间回到 safe_pose
    6. 如果 safe_pose_policy 为 "plan" 或 "final-only"，最后回到 safe_pose
    7. 打印执行摘要

    提前触发处理（Early Trigger Handling）：
    - 默认行为（skip_branch_on_early_trigger=True）：跳过当前分支剩余点，
      继续执行下一个分支
    - 传统行为（--no-skip-branch-on-early-trigger）：在任何分支上遇到
      提前触发就立即中止整个运行
    - 提前触发后自动调用 _recover_to_safe_after_early_trigger 恢复

    Args:
        node: ProbeTouch 节点实例
        args: 命令行参数
        plan: 规划项列表
    """
    skip_on_early = getattr(args, "skip_branch_on_early_trigger", True)

    _verify_plan_preflight(args, plan)

    # 确认 DI1 初始状态为未触发（探针没有意外接触）
    node.wait_fresh_feed()
    if fresh_feed_di1(node, max_age_sec=0.5):
        raise RuntimeError("DI1 is already triggered before calibration; retract or clear the probe first")
    node.set_speed(args.speed)

    # 如果策略要求，在规划开始前先移动到 safe_pose
    if args.safe_pose is not None and args.safe_pose_policy in ("plan", "initial-only"):
        print(f"move to initial safe pose: {format_pose(args.safe_pose)}")
        move_to_pose_guarded_and_confirm_clear(node, args.safe_pose, args, "safe pose before plan")

    # 按分支分组，每组内保持采样顺序
    branch_groups = _group_plan_by_branch(plan)
    skipped_branches = []
    total_hit = 0
    total_miss = 0

    for branch_index, (branch, items) in enumerate(branch_groups):
        print(f"\n{'='*50}\n branch {branch}  ({len(items)} samples)\n{'='*50}")
        branch_hit = 0
        branch_early = False

        for item in items:
            plan_id = item["row"]["plan_id"]

            # 如果该分支已发生提前触发，剩余采样点标记为 NOT_RUN
            if branch_early:
                append_stable_row(
                    args.output,
                    contact_row(args, item, None, None, "NOT_RUN"),
                )
                continue

            # 解析实际使用的 standoff（可能被 per-branch-standoff 覆盖）
            standoff = _resolve_standoff(args, item)
            original_standoff = float(item["row"].get("standoff_mm", args.standoff_mm))
            if abs(standoff - original_standoff) > 1e-6:
                # standoff 被覆盖，需要重新计算位姿
                item = _recompute_item_poses_for_standoff(item, standoff)
            print(f"\ncalibration sample {plan_id}  standoff={standoff:.3f}mm")

            # 执行单次探测
            status, trigger_snapshot, stop_pose = _execute_one_probe(node, args, item)

            if status in ("EARLY_TRIGGER_TRANSITION", "EARLY_TRIGGER_START"):
                branch_early = True
                if skip_on_early:
                    # 跳过当前分支剩余所有点，继续下一个分支
                    print(f"  *** early trigger on {plan_id}; skipping remaining {branch} points ***")
                    skipped_branches.append(branch)
                    _recover_to_safe_after_early_trigger(node, args, branch)
                else:
                    # 传统行为：直接中止
                    raise RuntimeError(
                        f"Early trigger ({status}) on {plan_id}; "
                        "use --skip-branch-on-early-trigger to continue with next branch"
                    )
            elif status == "HIT":
                branch_hit += 1
            elif status == "MISS":
                total_miss += 1

            # 如果策略为每次采样后回 safe_pose
            item_safe_pose = safe_pose_for_item(args, item)
            if item_safe_pose is not None and args.safe_pose_policy == "each-sample" and not branch_early:
                move_to_pose_guarded_and_confirm_clear(node, item_safe_pose, args, f"safe pose after {plan_id}")

        total_hit += branch_hit
        print(f"  branch {branch}: {branch_hit} HIT" + (" (early trigger, rest skipped)" if branch_early else ""))

        # 如果策略为每分支后回 safe_pose（且不是最后一个分支）
        if (not branch_early
                and args.safe_pose is not None
                and args.safe_pose_policy == "plan"
                and branch_index < len(branch_groups) - 1):
            print(f"move to safe pose between branches: {format_pose(args.safe_pose)}")
            move_to_pose_guarded_and_confirm_clear(node, args.safe_pose, args, f"safe pose after {branch}")

    # 最终回到 safe_pose
    if args.safe_pose is not None and args.safe_pose_policy in ("plan", "final-only"):
        print(f"move to final safe pose: {format_pose(args.safe_pose)}")
        move_to_pose_guarded_and_confirm_clear(node, args.safe_pose, args, "safe pose after plan")

    summary = f"execution complete: {total_hit} HIT, {total_miss} MISS"
    if skipped_branches:
        summary += f" | skipped branches: {', '.join(skipped_branches)}"
    print(f"\n{summary}")


def supported_pose_fields(row):
    """
    从 CSV 行中提取接触时刻的法兰位姿。

    支持两种格式：
    1. trigger_flange_x/y/z/rx/ry/rz（新格式）
    2. flange_x/y/z/rx/ry/rz（旧格式向后兼容）

    Args:
        row: CSV 行字典

    Returns:
        六维位姿 [x, y, z, rx, ry, rz]

    Raises:
        ValueError: 无法识别有效的位姿字段
    """
    if all(row.get(f"trigger_flange_{name}") not in (None, "") for name in POSE_NAMES):
        return [parse_float(row[f"trigger_flange_{name}"], f"trigger_flange_{name}") for name in POSE_NAMES]
    if all(row.get(name) not in (None, "") for name in ("flange_x", "flange_y", "flange_z", "rx", "ry", "rz")):
        return [parse_float(row[name], name) for name in ("flange_x", "flange_y", "flange_z", "rx", "ry", "rz")]
    raise ValueError("row missing supported pose fields")


def is_hit_row(row):
    """
    判断 CSV 行是否为有效的 HIT（接触成功）记录。

    判断逻辑：
    1. 如果存在 status 字段，直接检查是否为 "HIT"
    2. 否则检查 trigger_di1 字段是否为 "1" / "True"
    3. 兼容旧格式：source_command == "semi_auto_normal"

    Args:
        row: CSV 行字典

    Returns:
        True 表示接触成功，False 表示未接触
    """
    status = row.get("status")
    if status not in (None, ""):
        return status == "HIT"
    trigger_di1 = str(row.get("trigger_di1", "")).strip()
    if trigger_di1 in ("1", "true", "True"):
        return True
    if trigger_di1 == "0":
        return False
    return row.get("source_command", "") == "semi_auto_normal"


def fit_offsets_from_contacts(paths, branches, sphere_center, contact_distance, euler_sequence, args):
    """
    从接触数据中，基于固定球心的标量距离约束拟合每个分支偏置。

    数学原理：
    对每个接触点，已知触发法兰位置 P、旋转 R、接触中心距离 d 和固定球心 S：
        ||P + R * o - S|| = d
    其中 o 是待求的测球中心在法兰坐标系下的偏置。命令 approach 只用于生成
    鲁棒初值，不再被强制当作真实球面法向。求解采用 Huber IRLS Gauss-Newton，
    并检查 Jacobian 的秩和条件数。

    Args:
        paths: CSV 文件路径列表（主输出 + 可选 --fit-input 文件）
        branches: 需要拟合的分支名称列表
        sphere_center: 固定球心坐标 [x, y, z]
        contact_distance: 接触中心距离（球半径 + 探针球头半径）
        euler_sequence: 欧拉角旋转序列
        args: 命令行参数

    Returns:
        (results, residual_rows) 元组：
        - results: 每个分支的拟合结果列表，每项包含 ok、estimated_offset_mm、
          rms_residual_mm、max_residual_mm 等
        - residual_rows: 每个接触点的详细残差行列表
    """
    # approach-derived p_sample is retained only as a robust initial estimate and
    # a backwards-compatible diagnostic. The final fit uses the scalar sphere constraint.
    branch_samples = {branch: [] for branch in branches}
    branch_physical_ids = {branch: set() for branch in branches}
    residual_rows = []
    source_rows = {branch: [] for branch in branches}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row_index, row in enumerate(csv.DictReader(f), start=1):
                if not is_hit_row(row):
                    continue
                branch = row.get("branch", "")
                if branch not in branch_samples:
                    continue
                physical_ball_id = str(row.get("physical_ball_id", "")).strip()
                if physical_ball_id:
                    branch_physical_ids[branch].add(physical_ball_id)
                pose = supported_pose_fields(row)
                approach = normalize(
                    [parse_float(row.get("approach_x"), "approach_x"),
                     parse_float(row.get("approach_y"), "approach_y"),
                     parse_float(row.get("approach_z"), "approach_z")],
                    "approach",
                )
                rotation = euler_to_matrix(euler_sequence, pose[3:6])
                rhs = sub(sub(sphere_center, scale(approach, contact_distance)), pose[:3])
                p_sample = transpose_mat_vec(rotation, rhs)
                branch_samples[branch].append(p_sample)
                source_rows[branch].append((str(path), row_index, row, pose, approach, p_sample, rotation))

    results = []
    for branch in branches:
        samples = branch_samples[branch]
        if not samples:
            results.append({"branch": branch, "ok": False, "reason": "no HIT samples"})
            continue
        physical_ids = branch_physical_ids[branch]
        if len(physical_ids) > 1:
            raise ValueError(
                f"{branch}: contact inputs mix physical_ball_id values: "
                + ", ".join(sorted(physical_ids))
            )
        physical_ball_id = next(iter(physical_ids), str(args.physical_ball_id))

        sample_array = np.asarray(samples, dtype=float)
        sample_median = np.median(sample_array, axis=0)
        sample_distances = np.linalg.norm(sample_array - sample_median, axis=1)
        observations = source_rows[branch]
        positions = np.asarray([item[3][:3] for item in observations], dtype=float)
        rotations = np.asarray([item[6] for item in observations], dtype=float)
        fit = solve_fixed_center_sphere_offset(
            positions,
            rotations,
            sphere_center,
            contact_distance,
            sample_median,
            huber_delta_mm=args.sphere_fit_huber_delta_mm,
            max_iterations=args.sphere_fit_max_iterations,
            step_tolerance_mm=args.sphere_fit_step_tolerance_mm,
            max_condition=args.sphere_fit_max_condition,
            outlier_rejection=not args.disable_outlier_rejection,
            outlier_threshold_mm=args.outlier_threshold_mm,
            outlier_min_samples=args.outlier_min_samples,
            outlier_min_inliers=args.outlier_min_inliers,
        )
        if not fit.get("ok"):
            results.append(
                {
                    "branch": branch,
                    "physical_ball_id": physical_ball_id,
                    "ok": False,
                    "reason": str(fit.get("reason", "sphere offset fit failed")),
                    "raw_rows": len(samples),
                    "solver_method": "fixed_center_scalar_sphere_huber_irls",
                    "jacobian_rank": fit.get("jacobian_rank"),
                    "jacobian_condition": fit.get("jacobian_condition"),
                }
            )
            continue

        estimate = np.asarray(fit["estimate"], dtype=float)
        inlier_mask = np.asarray(fit["inlier_mask"], dtype=bool)
        signed_residuals = np.asarray(fit["signed_residuals"], dtype=float)
        outward_normals = np.asarray(fit["outward_normals"], dtype=float)
        ball_centers = np.asarray(fit["ball_centers"], dtype=float)
        residual_norms = []
        inlier_residual_norms = []
        inlier_signed_residuals = []
        for sample_index, (path, row_index, row, pose, approach, p_sample, _) in enumerate(observations):
            outward_normal = outward_normals[sample_index]
            inward_normal = -outward_normal
            signed_residual = float(signed_residuals[sample_index])
            residual = outward_normal * signed_residual
            residual_norm = abs(signed_residual)
            center_from_row = ball_centers[sample_index] + inward_normal * contact_distance
            approach_dot = max(-1.0, min(1.0, float(np.dot(np.asarray(approach), inward_normal))))
            approach_normal_angle = math.degrees(math.acos(approach_dot))
            residual_norms.append(residual_norm)
            if inlier_mask[sample_index]:
                inlier_residual_norms.append(residual_norm)
                inlier_signed_residuals.append(signed_residual)
            residual_rows.append(
                {
                    "source_csv": path,
                    "source_row": str(row_index),
                    "plan_id": row.get("plan_id", ""),
                    "branch": branch,
                    "physical_ball_id": str(row.get("physical_ball_id", "")),
                    "outlier_status": "inlier" if inlier_mask[sample_index] else "outlier",
                    "p_sample_distance_to_median_mm": f"{sample_distances[sample_index]:.6f}",
                    "approach_x": f"{approach[0]:.6f}",
                    "approach_y": f"{approach[1]:.6f}",
                    "approach_z": f"{approach[2]:.6f}",
                    "actual_inward_normal_x": f"{inward_normal[0]:.9f}",
                    "actual_inward_normal_y": f"{inward_normal[1]:.9f}",
                    "actual_inward_normal_z": f"{inward_normal[2]:.9f}",
                    "approach_normal_angle_deg": f"{approach_normal_angle:.6f}",
                    "p_sample_x": f"{p_sample[0]:.6f}",
                    "p_sample_y": f"{p_sample[1]:.6f}",
                    "p_sample_z": f"{p_sample[2]:.6f}",
                    "estimated_offset_x": f"{estimate[0]:.6f}",
                    "estimated_offset_y": f"{estimate[1]:.6f}",
                    "estimated_offset_z": f"{estimate[2]:.6f}",
                    "sphere_center_from_row_x": f"{center_from_row[0]:.4f}",
                    "sphere_center_from_row_y": f"{center_from_row[1]:.4f}",
                    "sphere_center_from_row_z": f"{center_from_row[2]:.4f}",
                    "residual_x": f"{residual[0]:.6f}",
                    "residual_y": f"{residual[1]:.6f}",
                    "residual_z": f"{residual[2]:.6f}",
                    "signed_radial_residual_mm": f"{signed_residual:.6f}",
                    "residual_norm_mm": f"{residual_norm:.6f}",
                    "dataset_split": row.get("dataset_split", ""),
                    "dataset_pose_family": row.get("dataset_pose_family", ""),
                    "dataset_group": row.get("dataset_group", ""),
                }
            )
        arr = np.asarray(inlier_residual_norms or residual_norms, dtype=float)
        signed_arr = np.asarray(inlier_signed_residuals or signed_residuals, dtype=float)
        results.append(
            {
                "branch": branch,
                "physical_ball_id": physical_ball_id,
                "ok": True,
                "rows": int(np.count_nonzero(inlier_mask)),
                "raw_rows": len(samples),
                "rejected_rows": int(len(samples) - np.count_nonzero(inlier_mask)),
                "estimated_offset_mm": estimate.tolist(),
                "solver_method": "fixed_center_scalar_sphere_huber_irls",
                "solver_iterations": int(fit["iterations"]),
                "solver_converged": bool(fit["converged"]),
                "jacobian_rank": int(fit["jacobian_rank"]),
                "jacobian_condition": float(fit["jacobian_condition"]),
                "rms_residual_mm": float(math.sqrt(float(np.mean(arr * arr)))),
                "max_residual_mm": float(np.max(arr)),
                "mean_residual_mm": float(np.mean(arr)),
                "mean_signed_residual_mm": float(np.mean(signed_arr)),
            }
        )
    return results, residual_rows


def write_calibration(args, reference_fit, geometry):
    """
    执行拟合并将结果写入 JSON 标定文件。

    从参考标定获取固定球心和接触距离，调用 fit_offsets_from_contacts
    进行拟合，然后将结果整理为包含所有分支估计的 JSON 格式保存。

    输出两个文件：
    1. calibration_output: JSON 格式的标定结果
    2. residual_output: CSV 格式的每个接触点残差明细

    Args:
        args: 命令行参数
        reference_fit: 参考标定结果
        geometry: 探针几何配置
    """
    sphere_center = [float(v) for v in reference_fit["sphere_center_mm"]]
    contact_distance = float(
        reference_fit.get("contact_center_distance_mm", args.sphere_radius_mm + args.probe_radius_mm)
    )
    branches, residual_rows = fit_offsets_from_contacts(
        [args.output] + list(args.fit_input or []),
        args.branches,
        sphere_center,
        contact_distance,
        args.euler_sequence,
        args,
    )
    result = {
        "timestamp": time.time(),
        "method": "fixed_center_scalar_sphere_huber_irls_offset_fit",
        "method_warning": (
            "Sphere centre is held fixed from the reference branch fit. "
            "Commanded approach is used only for initialization; the fitted normal is geometric. "
            "Branches without an observable rank-3 HIT set remain uncalibrated."
        ),
        "reference_fit_json": str(args.reference_fit_json),
        "sphere_center_mm": sphere_center,
        "contact_center_distance_mm": contact_distance,
        "geometry": str(args.geometry),
        "euler_sequence": args.euler_sequence,
        "outlier_rejection": {
            "enabled": not args.disable_outlier_rejection,
            "threshold_mm": args.outlier_threshold_mm,
            "min_samples": args.outlier_min_samples,
            "min_inliers": args.outlier_min_inliers,
        },
        "sphere_offset_solver": {
            "constraint": "norm(flange_position + R * local_offset - sphere_center) = contact_center_distance",
            "huber_delta_mm": args.sphere_fit_huber_delta_mm,
            "max_iterations": args.sphere_fit_max_iterations,
            "step_tolerance_mm": args.sphere_fit_step_tolerance_mm,
            "max_jacobian_condition": args.sphere_fit_max_condition,
        },
        "branches": branches,
    }
    output = Path(args.calibration_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved calibration JSON: {output}")
    if residual_rows:
        write_csv(args.residual_output, residual_rows)
        print(f"saved residual CSV: {args.residual_output}")
    for branch in branches:
        if not branch.get("ok"):
            print(f"  {branch['branch']}: {branch['reason']}")
            continue
        offset = branch["estimated_offset_mm"]
        print(
            f"  {branch['branch']}: rows={branch['rows']}/{branch.get('raw_rows', branch['rows'])} "
            f"rejected={branch.get('rejected_rows', 0)} "
            f"offset=[{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}] "
            f"rms={branch['rms_residual_mm']:.6f} max={branch['max_residual_mm']:.6f}"
        )


def validate_args(args):
    """
    验证所有命令行参数的有效性。

    检查规则包括：
    - 数值参数的正负性和取值范围（samples、radius、standoff、overtravel 等）
    - 安全保障检查（max_probe_travel 超过 1mm 需确认、speed 范围 1-5 等）
    - 执行模式检查（--execute 需要 --ack-five-branch-calibration-path）
    - safe_pose 相关检查（执行模式必须提供 safe_pose）
    - per_branch_standoff 格式解析
    - orientation 与 orientation_mode 互斥性
    - 各碰撞检测参数一致性

    Args:
        args: 命令行参数（会被修改，per_branch_standoff 从字符串列表解析为字典）

    Raises:
        ValueError: 存在无效参数值时抛出
    """
    if args.samples_per_branch <= 0:
        raise ValueError("--samples-per-branch must be positive")
    if args.sphere_radius_mm <= 0 or args.probe_radius_mm <= 0:
        raise ValueError("sphere/probe radius must be positive")
    if args.standoff_mm <= 0:
        raise ValueError("--standoff-mm must be positive")
    if args.transition_standoff_mm <= args.standoff_mm:
        raise ValueError("--transition-standoff-mm must be greater than --standoff-mm")
    if args.overtravel_mm <= 0:
        raise ValueError("--overtravel-mm must be positive")
    if args.standoff_mm + args.overtravel_mm > args.max_probe_travel_mm:
        raise ValueError("--standoff-mm + --overtravel-mm exceeds --max-probe-travel-mm")
    if args.max_probe_travel_mm > 1.0 and not args.allow_probe_travel_over_1mm:
        raise ValueError("--max-probe-travel-mm > 1 requires --allow-probe-travel-over-1mm")
    if args.min_probe_clearance_mm < 0:
        raise ValueError("--min-probe-clearance-mm cannot be negative")
    if args.max_safe_transition_orientation_delta_deg < 0:
        raise ValueError("--max-safe-transition-orientation-delta-deg cannot be negative")
    if args.min_table_clearance_mm < 0:
        raise ValueError("--min-table-clearance-mm cannot be negative")
    if not math.isfinite(args.table_plane_z_mm):
        raise ValueError("--table-plane-z-mm must be finite")
    if args.target_stem_exclusion_mm < 0:
        raise ValueError("--target-stem-exclusion-mm cannot be negative")
    if args.collision_segment_samples < 2:
        raise ValueError("--collision-segment-samples must be at least 2")
    if args.rod_collision_radius_mm is not None and args.rod_collision_radius_mm <= 0:
        raise ValueError("--rod-collision-radius-mm must be positive")
    if args.outlier_threshold_mm <= 0:
        raise ValueError("--outlier-threshold-mm must be positive")
    if args.outlier_min_samples < 1:
        raise ValueError("--outlier-min-samples must be positive")
    if args.outlier_min_inliers < 1:
        raise ValueError("--outlier-min-inliers must be positive")
    if args.sphere_fit_huber_delta_mm <= 0:
        raise ValueError("--sphere-fit-huber-delta-mm must be positive")
    if args.sphere_fit_max_iterations < 1:
        raise ValueError("--sphere-fit-max-iterations must be positive")
    if args.sphere_fit_step_tolerance_mm <= 0:
        raise ValueError("--sphere-fit-step-tolerance-mm must be positive")
    if args.sphere_fit_max_condition <= 1:
        raise ValueError("--sphere-fit-max-condition must be greater than 1")
    if args.max_local_step_mm < 0:
        raise ValueError("--max-local-step-mm cannot be negative")
    if args.max_local_orientation_step_deg < 0:
        raise ValueError("--max-local-orientation-step-deg cannot be negative")
    if args.max_branch_orientation_spread_deg < 0:
        raise ValueError("--max-branch-orientation-spread-deg cannot be negative")
    if not 1 <= args.speed <= 5:
        raise ValueError("--speed must be between 1 and 5")
    if args.timeout_sec <= 0 or args.positioning_timeout_sec <= 0:
        raise ValueError("timeouts must be positive")
    if args.positioning_trigger_retract_mm <= 0:
        raise ValueError("--positioning-trigger-retract-mm must be positive")
    if args.execute and not args.ack_five_branch_calibration_path:
        raise ValueError("real robot motion requires --ack-five-branch-calibration-path")
    if args.execute and args.safe_pose is None and not args.allow_no_safe_pose:
        raise ValueError("real robot motion requires --safe-pose, or explicitly pass --allow-no-safe-pose")
    # 解析每分支 standoff 覆盖值
    per_branch = {}
    for entry in args.per_branch_standoff:
        if "=" not in entry:
            raise ValueError(f"--per-branch-standoff entry must be BRANCH=MM, got: {entry}")
        branch, value = entry.split("=", 1)
        branch = branch.strip()
        try:
            mm = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"--per-branch-standoff {branch}: invalid standoff value {value!r}") from exc
        if mm <= 0:
            raise ValueError(f"--per-branch-standoff {branch}: standoff must be positive, got {mm}")
        per_branch[branch] = mm
    args.per_branch_standoff = per_branch
    if args.execute and args.orientation is not None and args.orientation_mode == "current":
        raise ValueError("--orientation and --orientation-mode current are mutually exclusive")
    if args.safe_pose is not None and len(args.safe_pose) != 6:
        raise ValueError("--safe-pose must contain X Y Z RX RY RZ")


def parse_args():
    """
    解析命令行参数。

    定义所有可用的命令行选项，包括：
    - 文件路径参数（参考标定、规划、输出等）
    - 标定参数（分支、采样数、球/探针半径等）
    - 运动参数（standoff、overtravel、速度等）
    - 安全参数（safe_pose、碰撞检测阈值、触发退回等）
    - 执行模式（dry-run / execute / fit-only 等）

    Returns:
        解析后的 argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-fit-json", default=str(DEFAULT_REFERENCE_FIT))
    parser.add_argument("--plan-input", help="execute an existing compatible plan CSV instead of generating a new plan")
    parser.add_argument("--plan-output", default=str(DEFAULT_PLAN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--calibration-output", default=str(DEFAULT_CALIBRATION_OUTPUT))
    parser.add_argument("--residual-output", default=str(DEFAULT_RESIDUAL_OUTPUT))
    parser.add_argument("--fit-input", nargs="*", default=[])
    parser.add_argument(
        "--branch-seed-fit",
        nargs="*",
        default=[],
        metavar="BRANCH=JSON",
        help="per-branch coarse fit seed, for example x_neg=data/seed.json",
    )
    parser.add_argument("--branches", nargs="+", default=list(DEFAULT_BRANCHES))
    parser.add_argument("--samples-per-branch", type=int, default=3)
    parser.add_argument("--physical-ball-id", default="five_branch_probe")
    parser.add_argument("--session-id", default="session_20260612_five_branch_sphere_calibration")
    parser.add_argument("--workpiece-id", default="calibration_sphere_20mm")
    parser.add_argument("--artifact-id", default="standard_sphere_20mm")
    parser.add_argument("--artifact-type", default="sphere")
    parser.add_argument("--operator-note", default="five branch sphere calibration")
    parser.add_argument("--geometry", default=str(DEFAULT_GEOMETRY))
    parser.add_argument("--euler-sequence", default="xyz")
    parser.add_argument("--sphere-radius-mm", type=float, default=10.0)
    parser.add_argument("--probe-radius-mm", type=float, default=1.0)
    parser.add_argument(
        "--orientation-mode",
        choices=("reference-fit-cycled", "current"),
        default="reference-fit-cycled",
    )
    parser.add_argument(
        "--orientation",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="offline orientation source; only RX RY RZ are used",
    )
    parser.add_argument("--safe-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument(
        "--safe-pose-policy",
        choices=("each-sample", "plan", "initial-only", "final-only", "none"),
        default="each-sample",
        help="when to visit --safe-pose during plan execution",
    )
    parser.add_argument(
        "--move-safe-only",
        action="store_true",
        help="move to --safe-pose and exit without loading or executing a probing plan",
    )
    parser.add_argument(
        "--report-current-pose-json",
        help="read the live flange pose, write it as JSON, and exit without motion",
    )
    parser.add_argument("--standoff-mm", type=float, default=0.5)
    parser.add_argument("--transition-standoff-mm", type=float, default=6.0)
    parser.add_argument("--overtravel-mm", type=float, default=0.3)
    parser.add_argument("--max-probe-travel-mm", type=float, default=1.0)
    parser.add_argument("--allow-probe-travel-over-1mm", action="store_true")
    parser.add_argument("--min-probe-clearance-mm", type=float, default=2.0)
    parser.add_argument("--max-safe-transition-orientation-delta-deg", type=float, default=90.0)
    parser.add_argument("--allow-large-safe-transition-orientation-change", action="store_true")
    parser.add_argument("--disable-table-plane-check", action="store_true")
    parser.add_argument("--table-plane-z-mm", type=float, default=0.0)
    parser.add_argument("--min-table-clearance-mm", type=float, default=5.0)
    parser.add_argument("--rod-collision-radius-mm", type=float)
    parser.add_argument("--target-stem-exclusion-mm", type=float, default=3.0)
    parser.add_argument("--collision-segment-samples", type=int, default=9)
    parser.add_argument("--allow-probe-model-collision-risk", action="store_true")
    parser.add_argument(
        "--allow-unchecked-plan",
        action="store_true",
        help=(
            "allow real execution of a plan that is missing moveit_status=OK rows. "
            "This is intended only for manually reviewed emergency/debug runs."
        ),
    )
    parser.add_argument(
        "--allow-varying-branch-orientation",
        action="store_true",
        help=(
            "allow a branch plan whose contact orientations vary beyond "
            "--max-branch-orientation-spread-deg. Do not use for normal fixed "
            "horizontal five-branch sphere calibration."
        ),
    )
    parser.add_argument(
        "--max-branch-orientation-spread-deg",
        type=float,
        default=2.0,
        help=(
            "maximum allowed orientation spread within each branch during real execution. "
            "Default keeps each stylus in a fixed taught posture."
        ),
    )
    parser.add_argument("--allow-no-safe-pose", action="store_true")
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--safe-transition-move", choices=("movl", "movj"), default="movl")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--positioning-timeout-sec", type=float, default=15.0)
    parser.add_argument("--positioning-trigger-retract-mm", type=float, default=5.0)
    parser.add_argument("--service-timeout-sec", type=float, default=10.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.08)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.08)
    parser.add_argument(
        "--max-local-step-mm",
        type=float,
        default=0.0,
        help="hard limit from the current pose to each transition pose; 0 disables",
    )
    parser.add_argument(
        "--max-local-orientation-step-deg",
        type=float,
        default=0.0,
        help="hard orientation limit from the current pose to each transition pose; 0 disables",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--no-fit-after-execute", action="store_true")
    parser.add_argument("--no-plan-summary", action="store_true")
    parser.add_argument(
        "--refit-after-each-hit",
        action="store_true",
        help="write updated calibration/residual files after every successful contact",
    )
    parser.add_argument("--disable-outlier-rejection", action="store_true")
    parser.add_argument("--outlier-threshold-mm", type=float, default=0.6)
    parser.add_argument("--outlier-min-samples", type=int, default=4)
    parser.add_argument("--outlier-min-inliers", type=int, default=3)
    parser.add_argument("--sphere-fit-huber-delta-mm", type=float, default=0.2)
    parser.add_argument("--sphere-fit-max-iterations", type=int, default=50)
    parser.add_argument("--sphere-fit-step-tolerance-mm", type=float, default=1e-9)
    parser.add_argument("--sphere-fit-max-condition", type=float, default=1e6)
    parser.add_argument(
        "--skip-branch-on-early-trigger",
        action="store_true",
        default=True,
        help="when an early DI1 trigger occurs on a branch, skip its remaining points and continue "
        "with the next branch instead of aborting the entire run (default: True)",
    )
    parser.add_argument(
        "--no-skip-branch-on-early-trigger",
        action="store_false",
        dest="skip_branch_on_early_trigger",
        help="legacy behaviour: abort the entire run on the first early trigger",
    )
    parser.add_argument(
        "--per-branch-standoff",
        nargs="*",
        default=[],
        metavar="BRANCH=MM",
        help="override standoff for specific branches, e.g. --per-branch-standoff y_pos=2.5 y_neg=1.0",
    )
    parser.add_argument("--ack-five-branch-calibration-path", action="store_true")
    return parser.parse_args()


def main():
    """
    主入口函数。

    根据执行模式（dry-run / execute / fit-only / move-safe-only / report-current-pose），
    执行相应的操作流程。整体流程为：

    1. 解析并验证参数
    2. 加载参考标定结果和几何配置
    3. 根据模式执行：
       - report-current-pose: 读取当前机器人位姿并保存为 JSON
       - move-safe-only: 仅移动到 safe_pose 后退出
       - fit-only: 仅执行拟合，不生成规划或运动
       - 默认：生成规划 → 可选执行 → 可选拟合
    4. 异常处理：将各类异常转换为 SystemExit

    异常转换规则：
    - OSError: 文件操作失败
    - ValueError: 参数或数据验证失败
    - RuntimeError: 运行时错误（运动失败、服务不可用等）
    - TimeoutError: 超时

    Note:
        never creates a ROS2 node - for offline planning only when
        --execute is not given.
    """
    args = parse_args()
    node = None
    try:
        validate_args(args)
        reference_fit = load_reference_fit(args.reference_fit_json)
        geometry = load_geometry(args.geometry)
        seed_offsets = load_branch_seed_offsets(args.branch_seed_fit)
        args._reference_fit = reference_fit
        args._geometry = geometry

        # 模式1：仅报告当前机器人位姿
        if args.report_current_pose_json:
            node, pose = current_pose_from_robot(args)
            output = Path(args.report_current_pose_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "flange_pose_mm_deg": [float(value) for value in pose],
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"current flange pose: {format_pose(pose)}")
            print(f"saved current pose JSON: {output}")
            return

        # 模式2：仅移动到 safe_pose
        if args.move_safe_only:
            if not args.execute:
                raise ValueError("--move-safe-only requires --execute")
            if args.safe_pose is None:
                raise ValueError("--move-safe-only requires --safe-pose")
            node, current = current_pose_from_robot(args)
            if fresh_feed_di1(node, max_age_sec=0.5):
                raise RuntimeError("DI1 is already triggered; refusing safe-pose motion")
            safe_distance = norm(sub(args.safe_pose[:3], current[:3]))
            safe_orientation_delta = max_orientation_delta_deg(args.safe_pose, current)
            if args.max_local_step_mm > 0 and safe_distance > args.max_local_step_mm:
                raise ValueError(
                    f"safe-only move is {safe_distance:.4f}mm; limit is {args.max_local_step_mm:.4f}mm"
                )
            if (
                args.max_local_orientation_step_deg > 0
                and safe_orientation_delta > args.max_local_orientation_step_deg
            ):
                raise ValueError(
                    f"safe-only orientation move is {safe_orientation_delta:.4f}deg; "
                    f"limit is {args.max_local_orientation_step_deg:.4f}deg"
                )
            node.set_speed(args.speed)
            move_to_pose_guarded_and_confirm_clear(node, args.safe_pose, args, "safe pose move only")
            print(f"reached safe pose: {format_pose(args.safe_pose)}")
            return

        # 模式3：仅拟合（不生成规划、不运动）
        if args.fit_only:
            write_calibration(args, reference_fit, geometry)
            return

        # 模式4：完整流程（生成规划 → 可选执行 → 可选拟合）
        current_orientation = None
        if (args.execute or args.orientation_mode == "current") and not args.plan_input:
            node, pose = current_pose_from_robot(args)
            current_orientation = pose[3:6]

        if args.plan_input:
            plan = plan_from_csv(args.plan_input)
            if args.execute:
                node, _ = current_pose_from_robot(args)
            if args.plan_output and str(args.plan_output) != str(args.plan_input):
                write_csv(args.plan_output, [item["row"] for item in plan])
        else:
            plan = make_plan(args, reference_fit, geometry, current_orientation=current_orientation, seed_offsets=seed_offsets)
            write_csv(args.plan_output, [item["row"] for item in plan])

        if not args.no_plan_summary:
            print_plan_summary(plan)

        if args.plan_input:
            print(f"loaded plan: {args.plan_input}")
            if args.plan_output and str(args.plan_output) != str(args.plan_input):
                print(f"saved plan copy: {args.plan_output}")
        else:
            print(f"saved plan: {args.plan_output}")

        if not args.execute:
            print("dry-run only; inspect the plan before real motion")
            return

        # 执行运动规划
        execute_plan(node, args, plan)
        print(f"saved contacts: {args.output}")

        # 执行后拟合
        if not args.no_fit_after_execute:
            write_calibration(args, reference_fit, geometry)

    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
