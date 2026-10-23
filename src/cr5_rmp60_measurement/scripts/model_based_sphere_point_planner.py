#!/usr/bin/env python3
"""
基于模型的标定球探测点位规划器。

本规划器不依赖实测接触位姿。它根据标准球体模型与交叉探针模型生成球面法线候选方向，
将每个候选方向转换为法兰盘位姿（flange pose），并输出与 check_five_branch_plan_moveit.py
及现有执行脚本兼容的 CSV 格式规划文件。

坐标系变换链路（从接触点到法兰盘位姿）:
  1. sphere_center + approach * contact_distance  →  contact_ball_center
     （标定球球心沿接近方向偏移接触距离，得到接触时探针球心位置）
  2. contact_ball_center - rotated_offset          →  contact_flange_position
     （探针球心位置减去经旋转后的探针偏置，得到法兰盘位置）
  3. orientation（由接近方向与偏置共同决定的旋转矩阵）→  flange_orientation
     （法兰盘姿态保证探针轴线沿接近方向，且偏置矢量在法兰盘坐标系中固定）

其中 rotated_offset = rotation_matrix @ offset，即探针球心相对于法兰盘坐标原点的
局部偏置经法兰盘旋转后变换到世界坐标系。
"""
import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from auto_calibrate_five_branch_sphere import (
    DEFAULT_REFERENCE_FIT,
    POSE_NAMES,
    add,
    branch_direction,
    combined_collision_summary,
    local_collision_primitives,
    load_branch_seed_offsets,
    max_orientation_delta_deg,
    norm,
    normalize,
    pose_dict,
    probe_dimensions,
    sample_pose_segment_clearance,
    scale,
    sub,
    worst_collision_summary,
)
from cross_probe_model import (
    DEFAULT_GEOMETRY,
    branch_local_offset_mm,
    euler_to_matrix,
    load_geometry,
    mat_vec,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data/model_based_sphere_plan.csv"


def dot(a, b):
    """Dot."""
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def cross(a, b):
    """Cross."""
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def mat_mult(a, b):
    """Multiply two 3x3 matrices."""
    return [[sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)] for row in range(3)]


def transpose(m):
    """Transpose."""
    return [[m[col][row] for col in range(3)] for row in range(3)]


def columns_to_matrix(columns):
    """Columns to matrix."""
    return [[columns[col][row] for col in range(3)] for row in range(3)]


def axis_angle_matrix(axis, angle_deg):
    """Axis angle matrix."""
    axis = normalize(axis, "axis")
    x, y, z = axis
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return [
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ]


def matrix_to_euler_xyz(matrix):
    """Inverse of cross_probe_model.euler_to_matrix('xyz', angles)."""
    r20 = max(-1.0, min(1.0, float(matrix[2][0])))
    ry = math.asin(-r20)
    cy = math.cos(ry)
    if abs(cy) > 1e-9:
        rx = math.atan2(matrix[2][1], matrix[2][2])
        rz = math.atan2(matrix[1][0], matrix[0][0])
    else:
        rx = 0.0
        rz = math.atan2(-matrix[0][1], matrix[1][1])
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def local_basis(direction):
    """
    以给定方向为第一轴，构造一个右手正交基 [e1, e2, e3]。

    这是姿态构造的基础工具函数，用于在局部坐标系中建立正交参考系。

    实现细节:
        1. e1 = normalize(direction)  — 第一轴取给定方向（单位化）。
        2. 选取候选矢量 q: 默认取 [0,0,1] (Z 轴)；
           若 e1 与 Z 轴夹角小于约 18 度 (dot > 0.95)，改用 [0,1,0] (Y 轴)，
           以避免数值不稳定性（当 e1 接近 Z 轴时，叉积结果接近零矢量）。
        3. e2 = normalize(q - e1 * dot(q, e1))  — Gram-Schmidt 正交化得到第二轴。
        4. e3 = normalize(cross(e1, e2))        — 叉积得到第三轴，确保右手系。

    返回:
        [e1, e2, e3]，每个元素为 3 元素列表。
    """
    e1 = normalize(direction, "branch direction")
    q = [0.0, 0.0, 1.0]
    if abs(dot(e1, q)) > 0.95:
        q = [0.0, 1.0, 0.0]
    e2 = normalize(sub(q, scale(e1, dot(e1, q))), "local secondary")
    e3 = normalize(cross(e1, e2), "local tertiary")
    return [e1, e2, e3]


def arbitrary_world_basis(normal):
    """
    以给定法线为第一轴，在世界坐标系中构造一个右手正交基 [e1, e2, e3]。

    与 local_basis 的区别: local_basis 构造的是"探针局部坐标系"的正交基，
    而本函数构造的是"世界坐标系中法线方向的正交基"。二者的结构完全相同，
    但语义不同（一个在局部，一个在世界），因此独立实现以避免混淆。

    在 rotation_for_normal 中，这两个正交基被联合使用:
        world_basis @ local_basis^T
    即: 将局部基映射到世界基的旋转矩阵，保证探针轴线与目标法线对齐。

    返回:
        [e1, e2, e3]，每个元素为 3 元素列表。
    """
    e1 = normalize(normal, "normal")
    q = [0.0, 0.0, 1.0]
    if abs(dot(e1, q)) > 0.95:
        q = [1.0, 0.0, 0.0]
    e2 = normalize(sub(q, scale(e1, dot(e1, q))), "world secondary")
    e3 = normalize(cross(e1, e2), "world tertiary")
    return [e1, e2, e3]


def rotation_for_normal(branch_local_direction, normal, anchor_rotation=None, roll_deg=0.0):
    """
    计算法兰盘旋转矩阵，使其满足：探针局部轴线方向经旋转后与世界坐标系下的目标法线方向对齐。

    该函数是实现"探针沿法线接近球面"的核心几何构造步骤：

    参数说明:
        branch_local_direction: 探针在法兰盘局部坐标系中的轴线方向（通常为分支方向）。
        normal: 世界坐标系下的目标球面法线（接近方向）。
        anchor_rotation: 可选的锚点旋转矩阵，用于稳定副法线方向（使其尽量接近锚点姿态的副轴）。
        roll_deg: 绕法线方向的额外旋转角（度），产生不同滚转角候选。

    实现步骤:
        1. 由 branch_local_direction 构造局部正交基 [e1, e2, e3]，其中 e1 = 分支方向。
        2. 若提供了 anchor_rotation，则将锚点副轴投影到法线正交平面，
           以保持候选姿态间的副轴连续性；否则直接构造任意正交基。
        3. 构建世界坐标系下的正交基：法线为第一轴，投影副轴（或任意轴）为第二轴。
        4. 通过 base = world_basis @ local_basis^T 计算从局部到世界的旋转矩阵。
        5. 叠加绕法线的滚转旋转 axis_angle_matrix(normal, roll_deg)，得到最终旋转矩阵。
    """
    local = local_basis(branch_local_direction)
    normal = normalize(normal, "candidate normal")
    if anchor_rotation is not None:
        reference_secondary = mat_vec(anchor_rotation, local[1])
        projected = sub(reference_secondary, scale(normal, dot(reference_secondary, normal)))
        if norm(projected) <= 1e-9:
            world = arbitrary_world_basis(normal)
        else:
            world = [normal, normalize(projected, "projected secondary"), None]
            world[2] = normalize(cross(world[0], world[1]), "projected tertiary")
    else:
        world = arbitrary_world_basis(normal)
    base = mat_mult(columns_to_matrix(world), transpose(columns_to_matrix(local)))
    return mat_mult(axis_angle_matrix(normal, roll_deg), base)


def fibonacci_normals(count):
    """
    使用斐波那契球面点生成算法产生均匀分布在单位球面上的法线方向。

    斐波那契球面点云（Fibonacci sphere / Fibonacci lattice）是一种将 N 个点
    近似均匀地分布在球面上的方法。它利用黄金角 (pi * (3 - sqrt(5))) 在经度
    方向上逐次旋转，同时在纬度方向上均匀采样。

    实现细节:
        golden_angle = pi * (3 - sqrt(5))   — 黄金角 (约 137.5 度)
        对第 i 个点:
            z = 1 - 2 * (i + 0.5) / count    — 纬度均匀采样 [-1, 1]
            r = sqrt(1 - z^2)                 — 当前纬度处的水平圆半径
            theta = golden_angle * i          — 经度角累加黄金角
            normal = [cos(theta)*r, sin(theta)*r, z]

    优点: 相比经纬网格法，没有极点聚集现象，分布更加均匀。
    缺点: 对特定点数的最佳分布不如优化的球面泰森多边形法。

    返回:
        列表，每个元素为 [nx, ny, nz] 单位矢量。
    """
    normals = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(max(1, count)):
        z = 1.0 - 2.0 * (index + 0.5) / float(count)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * index
        normals.append([math.cos(theta) * radius, math.sin(theta) * radius, z])
    return normals


def cone_normals(anchor_normal, cone_angle_deg, rings, samples_per_ring):
    """
    在锚点法线周围生成锥形采样法线方向集合。这是实现"anchor-cone"策略的核心函数。

    几何含义:
        以 anchor_normal 为锥轴，cone_angle_deg 为半锥角，生成多层锥面上的采样方向。
        锥面被划分为多个环（rings），每个环上的采样点数随环半径线性增加。

    采样策略参数:
        anchor_normal: 锚点法线方向，即锥轴方向（世界坐标系下的单位矢量）。
        cone_angle_deg: 锥的半角（度）。第 ring 层的锥角为 cone_angle_deg * ring / rings。
        rings: 环的总层数（不含顶点）。第 1 层为最内环，第 rings 层为最外环。
        samples_per_ring: 最内环的采样点数。第 ring 层的采样点数为 samples_per_ring * ring。

    实现步骤:
        basis = arbitrary_world_basis(anchor_normal)  — 以锚点法线为 z 轴构造正交基
        第一元素为锚点法线本身（即锥顶方向，ring=0）。
        对第 ring 层（1 到 rings）:
            - 当前锥角 angle = rad(cone_angle_deg * ring / rings)
            - 采样数 = samples_per_ring * ring
            - 对每个采样点: 方位角 azimuth = 2*pi * index / samples
            - 在 basis 的 XY 平面内构造 tangent 方向，与锚点法线合成锥面方向:
              normal = anchor * cos(angle) + tangent * sin(angle)

    返回:
        列表，包含 (1 + sum_{r=1}^{rings} samples_per_ring * r) 个法线方向。
    """
    anchor = normalize(anchor_normal, "anchor normal")
    basis = arbitrary_world_basis(anchor)
    normals = [anchor]
    for ring in range(1, max(1, rings) + 1):
        angle = math.radians(float(cone_angle_deg) * ring / float(max(1, rings)))
        samples = max(1, samples_per_ring * ring)
        for index in range(samples):
            azimuth = 2.0 * math.pi * index / float(samples)
            tangent = add(
                scale(basis[1], math.cos(azimuth)),
                scale(basis[2], math.sin(azimuth)),
            )
            normals.append(normalize(add(scale(anchor, math.cos(angle)), scale(tangent, math.sin(angle))), "cone normal"))
    return normals


def load_reference_sphere(path):
    """Load reference sphere."""
    data = json.loads(Path(path).read_text())
    center = data.get("sphere_center_mm")
    if not isinstance(center, list) or len(center) != 3:
        raise ValueError(f"{path}: missing 3-value sphere_center_mm")
    return data, [float(value) for value in center]


def load_branch_offset(args, geometry):
    """Load branch offset."""
    if args.branch_offset_mm:
        return [float(value) for value in args.branch_offset_mm], "branch_offset_mm"
    if args.branch_fit_json:
        data = json.loads(Path(args.branch_fit_json).read_text())
        offset = data.get("frozen_offset_mm")
        if not isinstance(offset, list) or len(offset) != 3:
            offset = data.get("local_ball_offset_mm")
        if not isinstance(offset, list) or len(offset) != 3:
            result = next(
                (
                    item
                    for item in data.get("branches", [])
                    if item.get("branch") == args.branch and item.get("ok", True)
                ),
                None,
            )
            if result is not None:
                offset = result.get("estimated_offset_mm")
        if not isinstance(offset, list) or len(offset) != 3:
            raise ValueError(
                f"{args.branch_fit_json}: missing 3-value frozen_offset_mm, "
                f"local_ball_offset_mm or branches[].estimated_offset_mm for {args.branch}"
            )
        return [float(value) for value in offset], "branch_fit_json"
    return branch_local_offset_mm(geometry, args.branch), "nominal_geometry"


def pose_with_position(position, orientation):
    """Pose with position."""
    return [position[0], position[1], position[2], orientation[0], orientation[1], orientation[2]]


def safe_pose_for_candidate(args, transition_pose, approach):
    """Return a candidate-local safe pose and the exit mode used."""
    radial_exit_mm = float(getattr(args, "radial_exit_mm", 0.0))
    x_positive_exit_mm = float(getattr(args, "x_positive_exit_mm", 0.0))
    if radial_exit_mm > 0.0 and x_positive_exit_mm > 0.0:
        raise ValueError("radial and base-X exits are mutually exclusive")
    if radial_exit_mm > 0.0:
        safe_pose = list(transition_pose)
        safe_pose[:3] = sub(transition_pose[:3], scale(approach, radial_exit_mm))
        return safe_pose, "radial_outward"
    if x_positive_exit_mm > 0.0:
        safe_pose = list(transition_pose)
        safe_pose[0] += x_positive_exit_mm
        return safe_pose, "base_x_positive"
    return args.safe_pose, "configured_safe_pose"


def plan_candidate(args, geometry, dims, primitives, active_origin, sphere_center, contact_distance, offset, normal, roll_deg, index):
    """
    根据一个候选法线方向（normal）及其滚转角（roll_deg），生成完整的探针接触任务规划行。

    这是规划器的核心函数，其执行流程如下:

    ===== 第一步：计算法兰盘旋转矩阵与姿态 =====
        根据 orientation_mode 选择不同的姿态计算方式:
            "align-branch":
                调用 rotation_for_normal(branch_dir, normal, anchor_rotation, roll_deg)
                计算旋转矩阵。该矩阵将探针局部轴线（分支方向）旋转到目标法线方向，
                同时可选的 anchor_rotation 保持副轴连续性，roll_deg 提供滚转变化。
            "anchor-fixed":
                直接使用 --anchor-pose 指定的固定姿态，接近方向为法线方向。

        approach = rotation @ branch_dir  — 世界坐标系下的实际接近方向（单位矢量）。
        rotated_offset = rotation @ offset — 探针偏置由局部坐标系旋转到世界坐标系。

    ===== 第二步：计算关键点处的探针球心位置 =====
        沿接近方向（approach）在不同偏移距离处计算探针球心的世界坐标:
            contact_ball_center  = sphere_center - approach * contact_distance
                （接触点：探针球心距标定球心恰好为接触距离）
            transition_ball_center = sphere_center - approach * (contact_distance + transition_standoff)
                （过渡起点：在接触点基础上额外后退 transition_standoff，用于安全规划）
            start_ball_center  = sphere_center - approach * (contact_distance + standoff)
                （接近起点：在接触点基础上后退 standoff，为接触运动预留行程）
            target_ball_center = sphere_center - approach * (contact_distance - overtravel)
                （过冲终点：在接触点基础上前冲 overtravel，确保可靠触发）

        【关键坐标变换】探针球心 → 法兰盘位置:
            flange_position = ball_center - rotated_offset
            即: 法兰盘位置 = 探针球心世界坐标 - 旋转后的探针偏置矢量。

    ===== 第三步：碰撞检测 =====
        对以下 4 个关键位姿 + 2 段运动轨迹执行碰撞检测:
            - transition_pose:   过渡起点位姿
            - start_pose:       接近起点位姿
            - contact_pose:     接触位姿
            - target_pose:      过冲终点位姿
            - transition→start: 第一段运动路径
            - start→target:     接近 + 接触 + 过冲运动路径（含回退）
        若提供了安全位姿（safe_pose），额外检测 safe→transition 段的间隙。
        碰撞检测使用 combined_collision_summary 和 sample_pose_segment_clearance。

    ===== 第四步：评分 =====
        clearance_score = 所有碰撞检测结果中的最小间隙值（负值表示碰撞侵入深度）。
        score = clearance_score
                - 0.02 * safe_transition_orientation_delta  （惩罚姿态突变）
                - 0.01 * anchor_orientation_delta            （惩罚偏离锚点姿态）

    ===== 第五步：输出规划行 =====
        将上述计算结果和所有元信息写入一个字典（CSV 行），包括:
            - 标识字段: plan_id, branch, physical_ball_id
            - 候选参数: normal, roll_deg, offset 等
            - 关键位姿: transition/start/contact/target 四组法兰盘位姿
            - 碰撞结果: status, clearance, closest_primitive/obstacle
            - 评分字段: candidate_score, anchor_delta 等

    返回:
        一个字典，键为 CSV 列名，值为对应数据。
    """
    branch_dir = branch_direction(geometry, args.branch)
    if args.orientation_mode == "anchor-fixed":
        if not args.anchor_pose:
            raise ValueError("--orientation-mode anchor-fixed requires --anchor-pose")
        rotation = euler_to_matrix(args.euler_sequence, args.anchor_pose[3:6])
        orientation = [float(value) for value in args.anchor_pose[3:6]]
        approach = normalize(normal, "planned approach")
    else:
        anchor_rotation = None
        if args.anchor_pose:
            anchor_rotation = euler_to_matrix(args.euler_sequence, args.anchor_pose[3:6])
        rotation = rotation_for_normal(branch_dir, normal, anchor_rotation=anchor_rotation, roll_deg=roll_deg)
        orientation = matrix_to_euler_xyz(rotation)
        approach = normalize(mat_vec(rotation, branch_dir), "planned approach")
    rotated_offset = mat_vec(rotation, offset)
    contact_ball_center = sub(sphere_center, scale(approach, contact_distance))
    transition_ball_center = sub(contact_ball_center, scale(approach, args.transition_standoff_mm))
    start_ball_center = sub(contact_ball_center, scale(approach, args.standoff_mm))
    target_ball_center = add(contact_ball_center, scale(approach, args.overtravel_mm))
    transition_pose = pose_with_position(sub(transition_ball_center, rotated_offset), orientation)
    start_pose = pose_with_position(sub(start_ball_center, rotated_offset), orientation)
    contact_pose = pose_with_position(sub(contact_ball_center, rotated_offset), orientation)
    target_pose = pose_with_position(sub(target_ball_center, rotated_offset), orientation)
    radial_exit_mm = float(getattr(args, "radial_exit_mm", 0.0))
    safe_pose, safe_exit_mode = safe_pose_for_candidate(args, transition_pose, approach)
    collision_checks = [
        combined_collision_summary(transition_pose, primitives, sphere_center, args.sphere_radius_mm, args),
        combined_collision_summary(start_pose, primitives, sphere_center, args.sphere_radius_mm, args),
        combined_collision_summary(contact_pose, primitives, sphere_center, args.sphere_radius_mm, args),
        combined_collision_summary(target_pose, primitives, sphere_center, args.sphere_radius_mm, args),
        sample_pose_segment_clearance(transition_pose, start_pose, primitives, sphere_center, args.sphere_radius_mm, args),
        sample_pose_segment_clearance(start_pose, target_pose, primitives, sphere_center, args.sphere_radius_mm, args),
    ]
    safe_delta = ""
    safe_distance = ""
    if safe_pose:
        collision_checks.append(
            sample_pose_segment_clearance(safe_pose, transition_pose, primitives, sphere_center, args.sphere_radius_mm, args)
        )
        safe_delta_value = max_orientation_delta_deg(safe_pose, transition_pose)
        safe_delta = f"{safe_delta_value:.4f}"
        safe_distance = f"{norm(sub(safe_pose[:3], transition_pose[:3])):.4f}"
    else:
        safe_delta_value = 0.0
    worst = worst_collision_summary(collision_checks)
    anchor_delta = ""
    if args.anchor_pose:
        anchor_delta = f"{max_orientation_delta_deg(args.anchor_pose, transition_pose):.4f}"
    clearance_score = max(-1000.0, float(worst["clearance"]))
    score = clearance_score - 0.02 * safe_delta_value
    if args.anchor_pose:
        score -= 0.01 * float(anchor_delta)
    row = {
        "plan_id": f"{args.branch}_{index:03d}",
        "branch": args.branch,
        "sample_in_branch": str(index),
        "physical_ball_id": str(args.physical_ball_id),
        "candidate_source": args.normal_mode,
        "candidate_roll_deg": f"{float(roll_deg):.4f}",
        "candidate_score": f"{score:.6f}",
        "candidate_anchor_orientation_delta_deg": anchor_delta,
        "candidate_safe_transition_distance_mm": safe_distance,
        "x_positive_exit_mm": f"{float(getattr(args, 'x_positive_exit_mm', 0.0)):.4f}",
        "radial_exit_mm": f"{radial_exit_mm:.4f}",
        "safe_exit_mode": safe_exit_mode,
        "approach_x": f"{approach[0]:.6f}",
        "approach_y": f"{approach[1]:.6f}",
        "approach_z": f"{approach[2]:.6f}",
        "sphere_center_x": f"{sphere_center[0]:.6f}",
        "sphere_center_y": f"{sphere_center[1]:.6f}",
        "sphere_center_z": f"{sphere_center[2]:.6f}",
        "contact_center_distance_mm": f"{contact_distance:.6f}",
        "offset_source": args._offset_source,
        "initial_offset_x": f"{offset[0]:.6f}",
        "initial_offset_y": f"{offset[1]:.6f}",
        "initial_offset_z": f"{offset[2]:.6f}",
        "estimated_branch_origin_x": f"{active_origin[0]:.4f}",
        "estimated_branch_origin_y": f"{active_origin[1]:.4f}",
        "estimated_branch_origin_z": f"{active_origin[2]:.4f}",
        "standoff_mm": f"{args.standoff_mm:.4f}",
        "transition_standoff_mm": f"{args.transition_standoff_mm:.4f}",
        "overtravel_mm": f"{args.overtravel_mm:.4f}",
        "probe_travel_mm": f"{args.standoff_mm + args.overtravel_mm:.4f}",
        "safe_transition_orientation_delta_deg": safe_delta,
        "max_safe_transition_orientation_delta_deg": f"{args.max_safe_transition_orientation_delta_deg:.4f}",
        "probe_collision_status": worst["status"],
        "min_probe_clearance_mm": f"{worst['clearance']:.4f}",
        "closest_probe_part": worst["primitive"],
        "closest_probe_model": worst["primitive_model"],
        "closest_obstacle": worst["obstacle"],
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
    if safe_pose:
        row.update(pose_dict("safe_flange", safe_pose))
    row.update(pose_dict("transition_flange", transition_pose))
    row.update(pose_dict("start_flange", start_pose))
    row.update(pose_dict("contact_flange", contact_pose))
    row.update(pose_dict("target_flange", target_pose))
    return row


def write_csv(path, rows):
    """Write a list of dicts to a CSV file with given fieldnames."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_normals(args, geometry):
    """
    根据命令行参数选择的模式，生成候选球面法线方向列表。这是规划流程的第一阶段。

    支持两种法线生成策略:

    1. "anchor-cone"（锚点锥采样）:
       以锚点姿态对应的接近方向（或直接指定的 --anchor-normal）为锥轴，
       在锥面上采样多层法线方向。适合在已知近似有效接近方向附近搜索最优方向。
       锚点法线的确定优先级:
           a. 直接指定的 --anchor-normal
           b. 由 --anchor-pose 计算得到的 _anchor_approach（默认）
           c. 由锚点姿态旋转矩阵乘以分支方向计算得到

    2. "fibonacci"（斐波那契球面均匀采样）:
       在单位球面上均匀生成 --candidate-count 个法线方向，无偏好方向。
       适合在缺少先验信息时进行全局搜索。

    返回:
        经过 filter_normal 筛选后的法线列表（符合 Z 分量范围约束）。
    """
    if args.normal_mode == "anchor-cone":
        if args.anchor_normal:
            anchor = args.anchor_normal
        elif getattr(args, "_anchor_approach", None) is not None:
            anchor = args._anchor_approach
        elif args.anchor_pose:
            anchor = mat_vec(euler_to_matrix(args.euler_sequence, args.anchor_pose[3:6]), branch_direction(geometry, args.branch))
        else:
            raise ValueError("--normal-mode anchor-cone requires --anchor-pose or --anchor-normal")
        return cone_normals(anchor, args.cone_angle_deg, args.cone_rings, args.cone_samples_per_ring)
    return fibonacci_normals(args.candidate_count)


def filter_normal(args, normal):
    """
    根据法线方向的 Z 分量筛选候选接近方向的有效性。

    在实际场景中，探针通常从标定球的上半球接近（Z 分量 > 0，表示从上往下接近），
    而非从下方或水平方向。此过滤函数用于剔除在运动学上不可行或可能与环境碰撞的法线方向。

    过滤条件:
        1. normal[2] < args.min_approach_z — 法线 Z 分量太小（方向太平或朝下），拒绝。
        2. normal[2] > args.max_approach_z — 法线 Z 分量太大（过于垂直），拒绝。

    默认值:
        min_approach_z = -1.0  （几乎不限制下限）
        max_approach_z =  0.35 （过滤掉 Z > 0.35 的方向，即排除过于竖直向下的接近）

    返回:
        True  — 法线方向有效，保留该候选。
        False — 法线方向无效，跳过该候选。
    """
    if normal[2] < args.min_approach_z:
        return False
    if normal[2] > args.max_approach_z:
        return False
    return True


def sort_rows(rows):
    """
    对候选规划行进行优先级排序。排序优先级由高到低依次为：

    第一优先级 — 碰撞状态 (probe_collision_status):
        "OK" (等级 0) > "LOW_CLEARANCE" (等级 1) > "COLLISION" (等级 2) > 未知状态 (等级 9)
        确保安全的接触方案排在前面。

    第二优先级 — 候选评分 (candidate_score，降序):
        评分 = 最小间隙 - 0.02 * 安全过渡姿态转角差 - 0.01 * 锚点姿态转角差
        评分越高（正数越大）表示该候选的间隙更充裕、姿态变化更平缓。

    第三优先级 — 锚点姿态转角差 (candidate_anchor_orientation_delta_deg，升序):
        在评分相同时，优先选择与锚点姿态差异较小的方案，以减少机械臂运动范围需求。

    第四优先级 — 安全过渡距离 (candidate_safe_transition_distance_mm，升序):
        在以上条件均相同时，优先选择安全位姿离过渡起点更近的方案。

    返回:
        按上述优先级排序后的列表（升序，即最优候选在首）。
    """
    status_rank = {"OK": 0, "LOW_CLEARANCE": 1, "COLLISION": 2}
    return sorted(
        rows,
        key=lambda row: (
            status_rank.get(row["probe_collision_status"], 9),
            -float(row["candidate_score"]),
            float(row.get("candidate_anchor_orientation_delta_deg") or 0.0),
            float(row.get("candidate_safe_transition_distance_mm") or 0.0),
        ),
    )


def parse_roll_angles(value):
    """Parse roll angles."""
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(float(item))
    if not result:
        raise argparse.ArgumentTypeError("roll angle list cannot be empty")
    return result


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-fit-json", default=str(DEFAULT_REFERENCE_FIT))
    parser.add_argument("--geometry", default=str(DEFAULT_GEOMETRY))
    parser.add_argument("--branch", required=True)
    parser.add_argument("--branch-fit-json", help="JSON containing local_ball_offset_mm for this branch")
    parser.add_argument(
        "--collision-branch-fit",
        nargs="*",
        default=[],
        metavar="BRANCH=JSON",
        help="calibrated offsets used to model all non-active stylus branches",
    )
    parser.add_argument("--branch-offset-mm", nargs=3, type=float, metavar=("PX", "PY", "PZ"))
    parser.add_argument(
        "--orientation-mode",
        choices=("align-branch", "anchor-fixed"),
        default="align-branch",
        help="anchor-fixed keeps flange orientation fixed and varies only sphere contact normals",
    )
    parser.add_argument("--physical-ball-id", default="model_based_probe")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--all-output", help="optional CSV with every generated candidate before top-N filtering")
    parser.add_argument("--plan-count", type=int, default=6)
    parser.add_argument("--normal-mode", choices=("anchor-cone", "fibonacci"), default="anchor-cone")
    parser.add_argument("--anchor-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--anchor-normal", nargs=3, type=float, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--cone-angle-deg", type=float, default=12.0)
    parser.add_argument("--cone-rings", type=int, default=3)
    parser.add_argument("--cone-samples-per-ring", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=240)
    parser.add_argument("--roll-angles-deg", type=parse_roll_angles, default=parse_roll_angles("0,90,180,270"))
    parser.add_argument("--safe-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--x-positive-exit-mm", type=float, default=0.0)
    parser.add_argument(
        "--radial-exit-mm", type=float, default=0.0,
        help="set each safe pose by retreating from transition opposite the current approach normal",
    )
    parser.add_argument("--euler-sequence", default="xyz")
    parser.add_argument("--sphere-radius-mm", type=float, default=10.0)
    parser.add_argument("--probe-radius-mm", type=float, default=1.0)
    parser.add_argument("--standoff-mm", type=float, default=0.5)
    parser.add_argument("--transition-standoff-mm", type=float, default=6.0)
    parser.add_argument("--overtravel-mm", type=float, default=0.3)
    parser.add_argument("--max-safe-transition-orientation-delta-deg", type=float, default=90.0)
    parser.add_argument("--min-approach-z", type=float, default=-1.0)
    parser.add_argument("--max-approach-z", type=float, default=0.35)
    parser.add_argument("--min-probe-clearance-mm", type=float, default=2.0)
    parser.add_argument("--disable-table-plane-check", action="store_true")
    parser.add_argument("--table-plane-z-mm", type=float, default=0.0)
    parser.add_argument("--min-table-clearance-mm", type=float, default=5.0)
    parser.add_argument("--rod-collision-radius-mm", type=float)
    parser.add_argument("--target-stem-exclusion-mm", type=float, default=3.0)
    parser.add_argument("--collision-segment-samples", type=int, default=9)
    parser.add_argument("--allow-low-clearance", action="store_true")
    parser.add_argument("--allow-collision", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """Validate args."""
    if args.plan_count <= 0:
        raise ValueError("--plan-count must be positive")
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    if args.cone_rings < 1 or args.cone_samples_per_ring < 1:
        raise ValueError("--cone-rings and --cone-samples-per-ring must be positive")
    if args.sphere_radius_mm <= 0 or args.probe_radius_mm <= 0:
        raise ValueError("sphere/probe radii must be positive")
    if args.standoff_mm <= 0 or args.transition_standoff_mm <= args.standoff_mm:
        raise ValueError("--transition-standoff-mm must be greater than --standoff-mm")
    if args.overtravel_mm <= 0:
        raise ValueError("--overtravel-mm must be positive")
    if args.x_positive_exit_mm < 0:
        raise ValueError("--x-positive-exit-mm cannot be negative")
    if args.radial_exit_mm < 0:
        raise ValueError("--radial-exit-mm cannot be negative")
    if args.x_positive_exit_mm > 0 and args.radial_exit_mm > 0:
        raise ValueError("--x-positive-exit-mm and --radial-exit-mm are mutually exclusive")
    if args.min_approach_z > args.max_approach_z:
        raise ValueError("--min-approach-z cannot exceed --max-approach-z")
    if args.orientation_mode == "anchor-fixed" and not args.anchor_pose:
        raise ValueError("--orientation-mode anchor-fixed requires --anchor-pose")
    if args.orientation_mode == "anchor-fixed" and any(abs(value) > 1e-9 for value in args.roll_angles_deg):
        raise ValueError("--orientation-mode anchor-fixed requires --roll-angles-deg 0")


def main():
    """Main."""
    args = parse_args()
    try:
        validate_args(args)
        reference_fit, sphere_center = load_reference_sphere(args.reference_fit_json)
        geometry = load_geometry(args.geometry)
        offset, offset_source = load_branch_offset(args, geometry)
        collision_offsets = load_branch_seed_offsets(args.collision_branch_fit)
        collision_offsets[args.branch] = list(offset)
        args._offset_source = offset_source
        args._anchor_approach = None
        if args.orientation_mode == "anchor-fixed" and args.anchor_pose:
            anchor_rotation = euler_to_matrix(args.euler_sequence, args.anchor_pose[3:6])
            anchor_ball_center = add(args.anchor_pose[:3], mat_vec(anchor_rotation, offset))
            args._anchor_approach = normalize(
                sub(sphere_center, anchor_ball_center),
                "anchor ball centre to sphere centre",
            )
        contact_distance = float(reference_fit.get("contact_center_distance_mm", args.sphere_radius_mm + args.probe_radius_mm))
        primitives, dims, active_origin = local_collision_primitives(
            geometry,
            args.branch,
            offset,
            args,
            collision_offsets=collision_offsets,
        )
        normals = [normal for normal in candidate_normals(args, geometry) if filter_normal(args, normal)]
        rows = []
        index = 1
        for normal in normals:
            for roll in args.roll_angles_deg:
                rows.append(
                    plan_candidate(
                        args,
                        geometry,
                        dims,
                        primitives,
                        active_origin,
                        sphere_center,
                        contact_distance,
                        offset,
                        normal,
                        roll,
                        index,
                    )
                )
                index += 1
        if not rows:
            raise ValueError("no candidates generated after normal filtering")
        rows = sort_rows(rows)
        if not args.allow_collision:
            rows = [row for row in rows if row["probe_collision_status"] != "COLLISION"]
        if not args.allow_low_clearance:
            rows = [row for row in rows if row["probe_collision_status"] == "OK"]
        if not rows:
            raise ValueError("no candidates left after collision filtering")
        selected = rows[: args.plan_count]
        write_csv(args.output, selected)
        if args.all_output:
            write_csv(args.all_output, rows)
        print(f"generated candidates: {index - 1}")
        print(f"selected plan rows: {len(selected)}")
        print(f"saved plan: {args.output}")
        for row in selected:
            print(
                f"  {row['plan_id']} status={row['probe_collision_status']} "
                f"clearance={row['min_probe_clearance_mm']}mm "
                f"approach=[{row['approach_x']}, {row['approach_y']}, {row['approach_z']}] "
                f"start=[{row['start_flange_x']}, {row['start_flange_y']}, {row['start_flange_z']}, "
                f"{row['start_flange_rx']}, {row['start_flange_ry']}, {row['start_flange_rz']}]"
            )
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
