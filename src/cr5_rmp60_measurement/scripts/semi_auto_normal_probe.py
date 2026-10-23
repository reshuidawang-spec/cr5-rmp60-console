#!/usr/bin/env python3
"""
半自动法向探测 —— 沿球面法线方向的短距离逼近测量。

本脚本用于在五支球杆标定过程中，对某一选定的红宝石球（ruby ball）执行
半自动的法向探测。操作员手动将机器人移动到标定球附近的合适位姿，脚本
自动计算沿两球心连线方向（即球面法线方向）的逼近位姿，执行短距离探测
运动，记录 DI1 触发时刻的位姿数据，并可选择性地重新拟合分支模型。

球面法向逼近几何原理：
  1. 从标定 JSON 中读取当前临时球心 C_s（sphere_center_mm）和
     该分支红宝石球相对于法兰的局部偏移量 p（local_ball_offset_mm）。
  2. 根据当前法兰位姿 F 和旋转矩阵 R，计算红宝石球球心:
        C_p = F_{xyz} + R * p
  3. 逼近方向即为 C_s - C_p 的单位向量，即从红宝石球心指向
     标定球心的方向，近似为球面法线方向。
  4. 沿此方向前进 --search-mm 距离得到目标法兰位姿，执行探测运动。

使用方式：
  - 手动模式（dry-run）：指定 --pose，只计算并打印计划，不执行运动
  - 自动模式：使用 --execute --ack-safe-normal-probe 执行实际机器人运动
  - 探测触发后，可选择使用 --refit-input 和 --refit-json 重新拟合分支模型
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy

from calibrate_branch_sphere_absolute import read_rows, solve
from cross_probe_model import DEFAULT_GEOMETRY, ball_radius_mm, euler_to_matrix, load_geometry, mat_vec
from jog_and_record_contacts import (
    format_pose,
    current_pose,
    fresh_feed_di1,
    issue_movl,
    pose_fields,
    retract_opposite,
    vector_fields,
    wait_jog,
)
from probe_touch import ProbeTouch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FIT = PROJECT_DIR / "data/260603.1/yneg_new_5pts_absolute_fit_20260603.json"
DEFAULT_OUTPUT = PROJECT_DIR / "data/semi_auto_yneg_normal_contacts.csv"
DEFAULT_REFIT_JSON = PROJECT_DIR / "data/semi_auto_yneg_normal_fit.json"
DEFAULT_REFIT_RESIDUALS = PROJECT_DIR / "data/semi_auto_yneg_normal_fit_residuals.csv"


def load_fit(path):
    """
    加载单分支临时拟合 JSON 文件，提取标定球心和局部偏移量。

    JSON 文件通常由 calibrate_branch_sphere_absolute.py 或 refit 流程生成，
    包含以下关键字段：
      - sphere_center_mm [3]: 标定球球心在机器人基坐标系下的坐标（mm）
      - local_ball_offset_mm [3]: 红宝石球相对于法兰的局部偏移（mm），
        在法兰坐标系下定义，用于从法兰位姿计算当前红宝石球球心。

    从这两个量和当前法兰位姿 F 即可计算红宝石球球心：
        C_p = F_{xyz} + R * p
    其中 R 为法兰旋转矩阵，p 为 local_ball_offset_mm。

    参数:
        path: JSON 文件路径（字符串或 Path 对象）
    返回:
        (data, sphere_center, local_offset)
        data: 原始 JSON 字典（包含 branch、rms_residual_mm 等元信息）
        sphere_center: 3 元素列表 [x, y, z]，标定球心坐标（mm）
        local_offset: 3 元素列表 [x, y, z]，红宝石球局部偏移（mm）
    抛出:
        ValueError: 缺少必需字段或数值维度不正确
    """
    with Path(path).open() as f:
        data = json.load(f)
    required = ("sphere_center_mm", "local_ball_offset_mm")
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"fit JSON missing fields: {missing}")
    sphere_center = [float(v) for v in data["sphere_center_mm"]]
    local_offset = [float(v) for v in data["local_ball_offset_mm"]]
    if len(sphere_center) != 3 or len(local_offset) != 3:
        raise ValueError("fit JSON sphere_center_mm and local_ball_offset_mm must contain 3 values")
    return data, sphere_center, local_offset


def vector_norm(values):
    """
    计算 3 维向量的欧几里得范数（长度）。

    参数:
        values: 可迭代的数值序列（预期长度为 3）
    返回:
        向量长度的浮点值
    """
    return math.sqrt(sum(float(v) * float(v) for v in values))


def normalize(values, label):
    """
    将 3 维向量归一化为单位向量。

    归一化方向向量用于表示逼近方向。如果输入向量长度接近零，
    则抛出 ValueError 以避免除零错误，label 参数用于错误信息
    中标识向量名称，方便调试。

    参数:
        values: 3 元素向量 [x, y, z]
        label: 向量名称/标识，用于错误提示
    返回:
        单位向量（3 元素列表）
    抛出:
        ValueError: 输入向量长度为零或接近零
    """
    length = vector_norm(values)
    if length <= 1e-12:
        raise ValueError(f"{label} cannot be zero")
    return [float(v) / length for v in values]


def compute_plan(pose, sphere_center, local_offset, euler_sequence, search_mm, approach_override=None):
    """
    计算半自动法向探测的运动计划。

    球面法向逼近计算步骤：
      1. 根据法兰位姿 F 的旋转部分和局部偏移 p，计算当前红宝石球球心:
           C_p = F_{xyz} + R(p)          （R 为欧拉角对应的旋转矩阵）
      2. 计算从红宝石球心指向标定球心的方向向量:
           to_sphere = C_s - C_p
      3. 将 to_sphere 归一化得到逼近方向 approach (单位向量)，
         这一方向近似为球面在接触点处的法线方向。
      4. 沿 approach 方向前进 search_mm 得到目标法兰位置:
           target = F_{xyz} + approach * search_mm
         法兰姿态保持不变（只平移，不旋转）。
      5. 如果提供了 approach_override，则使用该方向替代自动计算的
         to_sphere 方向（用于人为指定的特殊逼近方向）。

    参数:
        pose: 6 分量当前法兰位姿 [x, y, z, rx, ry, rz]（mm 和弧度）
        sphere_center: 3 元素标定球心坐标 [x, y, z]（mm）
        local_offset: 3 元素红宝石球局部偏移 [dx, dy, dz]（mm，法兰系）
        euler_sequence: 欧拉角旋转顺序（如 "xyz"）
        search_mm: 沿法线方向的逼近搜索距离（mm），通常为 0.2 ~ 1.0
        approach_override: 可选，手动指定的逼近方向向量的 3 元素列表，
            用于替代自动计算的法线方向
    返回:
        字典 plan，包含以下键：
            flange_pose: 当前法兰位姿
            rotation: 3x3 旋转矩阵
            sphere_center_mm: 标定球心
            local_ball_offset_mm: 红宝石球局部偏移
            ball_center_mm: 计算得到的红宝石球球心坐标
            approach: 单位逼近方向向量
            center_distance_mm: 红宝石球心到标定球心的距离估计值
            target_pose: 目标法兰位姿（逼近后的终点）
    """
    rotation = euler_to_matrix(euler_sequence, pose[3:6])
    offset_base = mat_vec(rotation, local_offset)
    ball_center = [float(pose[index]) + offset_base[index] for index in range(3)]
    to_sphere = [sphere_center[index] - ball_center[index] for index in range(3)]
    approach = (
        normalize(approach_override, "override approach")
        if approach_override is not None
        else normalize(to_sphere, "computed approach")
    )
    target_pose = list(pose)
    for index in range(3):
        target_pose[index] += approach[index] * search_mm
    return {
        "flange_pose": [float(v) for v in pose],
        "rotation": rotation,
        "sphere_center_mm": sphere_center,
        "local_ball_offset_mm": local_offset,
        "ball_center_mm": ball_center,
        "approach": approach,
        "center_distance_mm": vector_norm(to_sphere),
        "target_pose": target_pose,
    }


def print_plan(plan, search_mm, retract_mm):
    """
    在控制台打印完整的探测计划信息，供操作员确认。

    打印内容包括当前法兰位姿、临时球心、红宝石球心、逼近方向向量、
    估计的球心距、搜索距离和目标位姿等关键信息。

    参数:
        plan: compute_plan 返回的计划字典
        search_mm: 搜索距离（mm）
        retract_mm: 触发后的回退距离（mm）
    """
    print(f"current flange pose: {format_pose(plan['flange_pose'])}")
    print(
        "temporary sphere center C_s: "
        f"[{plan['sphere_center_mm'][0]:.4f}, {plan['sphere_center_mm'][1]:.4f}, {plan['sphere_center_mm'][2]:.4f}]"
    )
    print(
        "current ruby ball center C_p: "
        f"[{plan['ball_center_mm'][0]:.4f}, {plan['ball_center_mm'][1]:.4f}, {plan['ball_center_mm'][2]:.4f}]"
    )
    print(
        "computed approach: "
        f"[{plan['approach'][0]:.6f}, {plan['approach'][1]:.6f}, {plan['approach'][2]:.6f}]"
    )
    print(f"estimated C_p -> C_s distance: {plan['center_distance_mm']:.4f} mm")
    print(f"search distance: {search_mm:.4f} mm")
    print(f"target flange pose: {format_pose(plan['target_pose'])}")
    print(f"opposite retract after trigger: {retract_mm:.4f} mm")


def write_stable_row(path, row):
    """
    以追加模式将一行探测数据稳定写入 CSV 文件。

    如果目标文件已存在，则验证现有 CSV 列头与待写入行的字段名是否一致，
    防止不同格式或不同版本的数据混入同一个文件。

    参数:
        path: CSV 文件路径（字符串或 Path 对象）
        row: 字典类型的行数据，键作为 CSV 列头
    抛出:
        ValueError: 现有 CSV 列头与新数据的字段名不匹配
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open(newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != fieldnames:
            raise ValueError(
                f"CSV header mismatch for {path}; use a new --output or archive the existing file first"
            )
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_contact_row(args, plan, trigger_snapshot, stop_pose):
    """
    构造一行包含完整探测信息的接触记录。

    记录内容包括：
      - 标识字段：session_id、branch、physical_ball_id、workpiece_id 等
      - 位姿数据：启动位姿（start_flange）、目标位姿（target_flange）、
        触发位姿（trigger_flange）、停止位姿（stop_flange）
      - 逼近向量：approach_x/y/z 单位方向分量
      - 触发信息：DI1 状态、触发序列号、时间戳、触发时刻的关节角度
      - 初始拟合信息：fit_initial_rms_mm、fit_initial_max_residual_mm
      - 超程量：stop_overtravel_along_approach_mm，即停止位置相对于
        触发位置沿逼近方向的多余移动距离

    参数:
        args: 命令行参数对象（包含 session_id、branch、physical_ball_id 等元信息）
        plan: compute_plan 返回的计划字典（含 approach、ball_center_mm、initial_fit 等）
        trigger_snapshot: DI1 触发时刻的快照字典，包含 pose、joints、di1、wall_time 等
        stop_pose: 机器人停止时刻的 6 分量法兰位姿 [x, y, z, rx, ry, rz]
    返回:
        字典，可直接作为一行写入 CSV 输出文件
    """
    trigger_pose = trigger_snapshot.get("pose") or stop_pose
    row = {
        "timestamp": f"{time.time():.3f}",
        "sample_index": "",
        "session_id": args.session_id,
        "setup_id": args.setup_id,
        "workpiece_id": args.workpiece_id,
        "artifact_id": args.artifact_id,
        "artifact_type": args.artifact_type,
        "physical_ball_id": args.physical_ball_id,
        "branch": args.branch,
        "standard_pose": "semi_auto_normal",
        "face_id": "standard_sphere_semi_auto_normal",
        "operator_note": args.operator_note,
        "source_command": "semi_auto_normal",
        "fit_initial_json": str(args.fit_json),
        "fit_initial_rms_mm": "",
        "fit_initial_max_residual_mm": "",
        "search_mm": f"{args.search_mm:.4f}",
        "speed": str(args.speed),
        "auto_retract_mm": f"{args.retract_mm:.4f}",
        "approach_x": f"{plan['approach'][0]:.6f}",
        "approach_y": f"{plan['approach'][1]:.6f}",
        "approach_z": f"{plan['approach'][2]:.6f}",
        "flange_x": f"{trigger_pose[0]:.4f}",
        "flange_y": f"{trigger_pose[1]:.4f}",
        "flange_z": f"{trigger_pose[2]:.4f}",
        "rx": f"{trigger_pose[3]:.4f}",
        "ry": f"{trigger_pose[4]:.4f}",
        "rz": f"{trigger_pose[5]:.4f}",
        "planned_ball_center_x": f"{plan['ball_center_mm'][0]:.4f}",
        "planned_ball_center_y": f"{plan['ball_center_mm'][1]:.4f}",
        "planned_ball_center_z": f"{plan['ball_center_mm'][2]:.4f}",
        "planned_center_distance_mm": f"{plan['center_distance_mm']:.4f}",
        "trigger_feed_sequence": str(trigger_snapshot.get("sequence", "")),
        "trigger_feed_wall_time": (
            "" if trigger_snapshot.get("wall_time") is None else f"{float(trigger_snapshot['wall_time']):.6f}"
        ),
        "trigger_digital_input_bits": (
            "" if trigger_snapshot.get("digital_input_bits") is None else str(trigger_snapshot["digital_input_bits"])
        ),
        "trigger_di1": "" if trigger_snapshot.get("di1") is None else str(int(bool(trigger_snapshot["di1"]))),
    }
    initial_fit = plan.get("initial_fit", {})
    if initial_fit.get("rms_residual_mm") is not None:
        row["fit_initial_rms_mm"] = f"{float(initial_fit['rms_residual_mm']):.6f}"
    if initial_fit.get("max_residual_mm") is not None:
        row["fit_initial_max_residual_mm"] = f"{float(initial_fit['max_residual_mm']):.6f}"
    row.update(pose_fields("start_flange", plan["flange_pose"]))
    row.update(pose_fields("target_flange", plan["target_pose"]))
    row.update(pose_fields("trigger_flange", trigger_pose))
    row.update(pose_fields("stop_flange", stop_pose))
    row.update(vector_fields("trigger_joint", trigger_snapshot.get("joints"), 6))
    stop_delta = [stop_pose[index] - trigger_pose[index] for index in range(3)]
    row["stop_overtravel_along_approach_mm"] = f"{float(np.dot(stop_delta, plan['approach'])):.6f}"
    return row


def write_residual_csv(path, rows):
    """
    将 refit 产生的残差行列表写入 CSV 文件。

    残差数据包含每个接触点相对于拟合球面的残差距离，
    用于评估拟合质量和识别离群点。

    参数:
        path: 输出 CSV 文件路径（字符串或 Path 对象）
        rows: 字典列表，每个字典对应一个接触点的残差记录
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_refit(result, branch):
    """
    在控制台打印 refit 更新后的单分支拟合结果。

    输出内容包括：分支名称、参与拟合的数据行数、矩阵秩和条件数、
    更新的球心坐标、局部偏移量、RMS 残差和最大残差。

    参数:
        result: solve() 返回的拟合结果字典，包含 sphere_center_mm、
            local_ball_offset_mm、rms_residual_mm、max_residual_mm 等字段
        branch: 分支标识字符串（如 "y_neg"、"x_pos" 等）
    """
    center = result["sphere_center_mm"]
    offset = result["local_ball_offset_mm"]
    print("updated single-branch fit")
    print(f"  branch: {branch}")
    print(f"  rows: {result['rows']}")
    print(f"  rank: {result['rank']}")
    print(f"  condition: {result['condition']:.6f}" if result["condition"] is not None else "  condition: n/a")
    print(f"  sphere_center_mm: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  local_ball_offset_mm: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
    print(f"  rms_residual_mm: {result['rms_residual_mm']:.6f}")
    print(f"  max_residual_mm: {result['max_residual_mm']:.6f}")


def refit_after_contact(args):
    """
    在成功触发探测后，结合新数据和已有接触点进行重新拟合。

    流程：
      1. 收集 refit_input 中指定的已有 CSV 文件和当前 --output 文件
      2. 筛选属于指定分支（--branch）的所有接触行
      3. 调用 calibrate_branch_sphere_absolute.solve 进行球面拟合
      4. 打印更新后的拟合参数（球心、偏移量、残差等）
      5. 可选地将拟合结果保存为 JSON 文件和残差 CSV 文件

    参数:
        args: 命令行参数对象，包含 refit_input、output、branch、
            refit_json、refit_residual_output 等配置
    """
    inputs = list(args.refit_input or [])
    if str(args.output) not in inputs:
        inputs.append(str(args.output))
    geometry = load_geometry(args.geometry)
    probe_radius = args.probe_radius_mm if args.probe_radius_mm is not None else ball_radius_mm(geometry)
    rows = read_rows(inputs, args.branch)
    result = solve(rows, args.sphere_radius_mm, probe_radius, args.euler_sequence)
    result["branch"] = args.branch
    result["source_csv"] = inputs
    result["geometry"] = str(args.geometry)
    result["euler_sequence"] = args.euler_sequence
    print_refit(result, args.branch)
    if args.refit_json:
        output = Path(args.refit_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved refit JSON: {output}")
    if args.refit_residual_output:
        write_residual_csv(args.refit_residual_output, result["residual_rows"])
        print(f"saved refit residuals: {args.refit_residual_output}")


def current_pose_from_robot(args):
    """
    从机器人实时状态获取当前法兰位姿。

    初始化 ROS 2 节点和 ProbeTouch 客户端，等待服务和 /feed 数据就绪后，
    读取并返回当前法兰 6D 位姿。如果读取过程中发生异常，负责清理节点资源。

    参数:
        args: 命令行参数对象（含 service_timeout_sec）
    返回:
        (node, pose) 元组
        node: ProbeTouch 节点实例（调用方负责在 finally 块中销毁）
        pose: 6 分量法兰位姿 [x, y, z, rx, ry, rz]（mm 和弧度）
    抛出:
        Exception: 服务超时或位姿读取失败
    """
    rclpy.init()
    node = ProbeTouch()
    try:
        node.wait_services(args.service_timeout_sec)
        node.wait_fresh_feed()
        return node, current_pose(node, max_age_sec=0.5)
    except Exception:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        raise


def execute_probe(node, args, plan):
    """
    执行实际的探测运动并处理触发结果。

    执行流程：
      1. 检查 DI1 信号是否已触发（已触发则报错，避免在接触状态下再逼近）
      2. 设置机器人运动速度
      3. 向目标位姿（plan["target_pose"]）发起 MOVL 直线运动指令
      4. 实时监听 DI1 触发信号，同时监控是否到达目标位姿
      5. 结果处理：
         - 触发成功：记录触发时刻快照、停止位姿，构建 CSV 行并写入文件，
           然后沿相反方向回退 --retract-mm 距离
         - 未触发到达目标（miss）：按 --no-retract-on-miss 配置决定是否回退
         - 回退后再次检查 DI1 状态用于日志记录

    参数:
        node: ProbeTouch 节点实例
        args: 命令行参数（含 speed、timeout_sec、retract_mm、position_tolerance_mm 等）
        plan: compute_plan 返回的计划字典
    返回:
        成功触发时返回接触行字典（用于后续 refit），未触发返回 None
    """
    node.wait_fresh_feed()
    if fresh_feed_di1(node):
        raise RuntimeError("DI1 is already triggered before probing; retract or clear the probe first")
    node.set_speed(args.speed)
    future = issue_movl(node, plan["target_pose"])
    trigger_snapshot, reached_target = wait_jog(
        node,
        plan["target_pose"],
        args.timeout_sec,
        args.position_tolerance_mm,
        args.orientation_tolerance_deg,
    )
    if future.done():
        node.check_ready_future(node.movl_cli, future)
    if trigger_snapshot is None:
        print("reached target; no DI1 trigger")
        if args.no_retract_on_miss:
            final_pose = current_pose(node, max_age_sec=0.5)
            print(f"held at miss target: {format_pose(final_pose)}")
            print(f"DI1 at miss target: {int(fresh_feed_di1(node, max_age_sec=0.5))}")
            return None
        retract_pose = retract_opposite(node, plan["flange_pose"], plan["target_pose"], args.retract_mm, args.timeout_sec)
        if retract_pose is not None:
            print(f"retracted:    {format_pose(retract_pose)}")
        print(f"DI1 after retract: {int(fresh_feed_di1(node, max_age_sec=0.5))}")
        return None
    stop_pose = current_pose(node, max_age_sec=0.5)
    row = make_contact_row(args, plan, trigger_snapshot, stop_pose)
    write_stable_row(args.output, row)
    print(f"DI1 triggered; saved: {args.output}")
    print(f"trigger pose: {format_pose(trigger_snapshot.get('pose') or stop_pose)}")
    print(f"stop pose:    {format_pose(stop_pose)}")
    retract_pose = retract_opposite(node, plan["flange_pose"], plan["target_pose"], args.retract_mm, args.timeout_sec)
    if retract_pose is not None:
        print(f"retracted:    {format_pose(retract_pose)}")
    print(f"DI1 after retract: {int(fresh_feed_di1(node, max_age_sec=0.5))}")
    return row


def parse_args():
    """
    解析命令行参数配置。

    关键参数说明：
      --fit-json:       临时单分支拟合 JSON，提供标定球心 C_s 和局部偏移 p
      --search-mm:      沿球面法线的短距离搜索长度（mm），默认 0.2，最大 1.0
                         超过 1.0mm 需 --allow-search-over-1mm 确认
      --approach:       手动逼近方向（3 分量），替代自动计算的球面法线方向
      --execute:        实际执行机器人运动（dry-run 模式只计算不运动）
      --ack-safe-normal-probe: 操作员确认路径安全后方可执行实际运动
      --refit-input:    refit 时合并的已有 CSV 文件列表
      --no-refit:       触发后不执行重新拟合
      --no-retract-on-miss: 未触发时保持在目标位姿而非回退

    返回:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-json", default=str(DEFAULT_FIT), help="temporary absolute fit JSON used as C_s / p")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV output for new semi-auto contacts")
    parser.add_argument("--branch", default="y_neg")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--setup-id", default="semi_auto_normal_probe")
    parser.add_argument("--workpiece-id", default="calibration_sphere_20mm")
    parser.add_argument("--artifact-id", default="standard_sphere_20mm")
    parser.add_argument("--artifact-type", default="sphere")
    parser.add_argument("--physical-ball-id", default="1")
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--geometry", default=str(DEFAULT_GEOMETRY))
    parser.add_argument("--euler-sequence", default="xyz")
    parser.add_argument("--sphere-radius-mm", type=float, default=10.0)
    parser.add_argument("--probe-radius-mm", type=float)
    parser.add_argument("--search-mm", type=float, default=0.2, help="short normal search distance, max 1.0mm")
    parser.add_argument("--approach", nargs=3, type=float, metavar=("DX", "DY", "DZ"), help="override search direction in base frame")
    parser.add_argument(
        "--allow-search-over-1mm",
        action="store_true",
        help="allow 1-3mm search only after the same line has been operator/empty-path checked",
    )
    parser.add_argument("--retract-mm", type=float, default=1.0)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.08)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.08)
    parser.add_argument("--service-timeout-sec", type=float, default=10.0)
    parser.add_argument("--pose", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--execute", action="store_true", help="actually move the robot")
    parser.add_argument(
        "--ack-safe-normal-probe",
        action="store_true",
        help="confirm the selected ruby ball is near the sphere and the short normal path/retract are clear",
    )
    parser.add_argument(
        "--refit-input",
        nargs="*",
        default=[str(PROJECT_DIR / "data/260603.1/yneg_new_5pts_for_fit.csv")],
        help="existing fit CSV files to combine with --output after a successful trigger",
    )
    parser.add_argument("--refit-json", default=str(DEFAULT_REFIT_JSON))
    parser.add_argument("--refit-residual-output", default=str(DEFAULT_REFIT_RESIDUALS))
    parser.add_argument("--no-refit", action="store_true", help="do not refit after a successful trigger")
    parser.add_argument(
        "--no-retract-on-miss",
        action="store_true",
        help="if no trigger occurs, hold at the reached target instead of retracting; triggers still retract",
    )
    return parser.parse_args()


def validate_args(args):
    """
    验证命令行参数的合法性和逻辑一致性。

    检查项包括：
      - --search-mm 必须在 (0, 1.0] 范围内（超过 1.0 需 --allow-search-over-1mm）
      - --search-mm 不得超过 3.0mm
      - --retract-mm、--timeout-sec、--sphere-radius-mm 必须为正数
      - --speed 必须在 1~5 之间
      - --execute 与 --pose 互斥（--pose 仅用于离线 dry-run 模式）
      - --execute 要求 --ack-safe-normal-probe 已显式确认
      - --probe-radius-mm 可选，提供时必须为正数

    参数:
        args: argparse.Namespace 解析后的命令行参数
    抛出:
        ValueError: 任意验证项目不通过时抛出，附带详细原因说明
    """
    if args.search_mm <= 0:
        raise ValueError("--search-mm must be positive")
    if args.search_mm > 1.0 and not args.allow_search_over_1mm:
        raise ValueError("--search-mm > 1.0 requires --allow-search-over-1mm")
    if args.search_mm > 3.0:
        raise ValueError("--search-mm must be <= 3.0 for semi-auto normal probing")
    if args.retract_mm <= 0:
        raise ValueError("--retract-mm must be positive")
    if not 1 <= args.speed <= 5:
        raise ValueError("--speed must be between 1 and 5")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be positive")
    if args.execute and args.pose is not None:
        raise ValueError("--pose is for offline dry-run only; do not combine it with --execute")
    if args.execute and not args.ack_safe_normal_probe:
        raise ValueError("real motion requires --ack-safe-normal-probe after on-site path confirmation")
    if args.sphere_radius_mm <= 0:
        raise ValueError("--sphere-radius-mm must be positive")
    if args.probe_radius_mm is not None and args.probe_radius_mm <= 0:
        raise ValueError("--probe-radius-mm must be positive")


def main():
    """
    主入口：半自动法向探测的执行流程编排。

    完整流程：
      1. 解析并验证命令行参数
      2. 从 --fit-json 加载临时拟合数据（球心 C_s 和局部偏移 p）
      3. 验证 fit JSON 的分支与 --branch 一致
      4. 获取当前机器人位姿（从实时状态或 --pose 参数）
      5. 调用 compute_plan 计算探测计划（沿球面法线方向逼近）
      6. 打印计划供操作员确认
      7. dry-run 模式（无 --execute）到此结束
      8. 执行实际探测运动，监听 DI1 触发信号
      9. 成功触发且未禁用 refit 时，重新拟合分支模型

    安全机制：
      - --execute 必须配合 --ack-safe-normal-probe 使用
      - fit JSON 的 branch 必须与 --branch 一致
      - 提供 dry-run 模式供离线验证

    退出码：
      0: 全部成功完成
      非 0: 参数错误、服务不可用或文件读写异常
    """
    args = parse_args()
    node = None
    try:
        validate_args(args)
        fit, sphere_center, local_offset = load_fit(args.fit_json)
        fit_branch = fit.get("branch")
        if fit_branch and fit_branch != args.branch:
            raise ValueError(
                f"fit JSON branch {fit_branch!r} does not match --branch {args.branch!r}; "
                "use the matching initial fit for the selected physical ruby ball"
            )
        pose = args.pose
        if pose is None:
            node, pose = current_pose_from_robot(args)
        plan = compute_plan(pose, sphere_center, local_offset, args.euler_sequence, args.search_mm, args.approach)
        plan["initial_fit"] = fit
        print_plan(plan, args.search_mm, args.retract_mm)
        if not args.execute:
            print("dry-run only; add --execute and --ack-safe-normal-probe for real short probing motion")
            return
        row = execute_probe(node, args, plan)
        if row is not None and not args.no_refit:
            refit_after_contact(args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
