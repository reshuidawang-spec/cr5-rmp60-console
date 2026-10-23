#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五向测针（cross-probe / RMP60）几何模型：根据 CR5 法兰位姿计算测球球心坐标。

=============================================================================
模块功能概述
=============================================================================
本模块实现了从 CR5 机器人法兰（flange）位姿到五向测针各分支测球球心坐标的
完整正向运动学链。核心逻辑如下：

    法兰坐标系 (flange_frame)
        → 法兰面沿 Z 轴向外平移（flange_adapter + probe_body）
            → 分支原点 (branch_origin_frame)
                → 各分支在 XY 平面或 Z 轴方向上的偏移 (nominal_offset_mm)
                    → 测球球心 (ball_center_frame)

最终球心在世界坐标系（即法兰位姿所在坐标系）中的坐标由下式给出：

    ball_center = flange_position + R * (branch_origin + branch_offset)

其中：
    - flange_position       : 法兰原点在世界系中的位置 [x, y, z]
    - R                     : 由欧拉角 (rx, ry, rz) 构成的旋转矩阵
    - branch_origin         : 从法兰原点沿 Z 轴到分支原点的偏移向量
                              (adapter + body = 49.4 + 76.0 = 125.4 mm)
    - branch_offset         : 各分支在分支原点坐标系中的杆长偏移
                              （Z 分支沿 Z 轴，X/Y 分支在 XY 平面内）

=============================================================================
坐标系约定
=============================================================================
- **法兰坐标系 (flange frame)**：
  原点在 CR5 法兰盘中心，Z 轴指向工具侧（竖直向下为负 X / 沿工具方向为正 Z，
  具体取决于机器人约定）；rx/ry/rz 为绕 X/Y/Z 轴的欧拉角。

- **世界坐标系 (world frame)**：
  由法兰位姿 (x, y, z, rx, ry, rz) 定义，三个平移量 + 三个旋转角共同确定
  法兰坐标系在世界系中的位置与姿态。

- **分支原点 (branch origin frame)**：
  位于法兰 Z 轴方向、距法兰面 "adapter + probe_body" 距离处。此原点与法兰
  坐标系姿态相同（无额外旋转），仅作平移。

- **测球球心 (ball center frame)**：
  每个分支在分支原点坐标系中的偏移由 `nominal_offset_mm` 定义，该偏移经
  旋转矩阵变换到世界系后，加上法兰平移即得世界系中的球心坐标。

=============================================================================
五向测针结构（cross-probe / five-branch stylus）
=============================================================================
五向测针包含五个分支：
    1. z       — 沿 Z 轴向下（轴向），常用于垂直触碰
    2. x_pos   — X 正方向（水平偏转）
    3. x_neg   — X 负方向
    4. y_pos   — Y 正方向
    5. y_neg   — Y 负方向

每个分支在配置文件 `cross_probe_geometry.yaml` 中以如下结构描述：
    branches:
      - name: z
        direction: [0.0, 0.0, 1.0]
        nominal_offset_mm: [0.0, 0.0, 83.8709...]   # 相对分支原点的偏移

其中 `direction` 为标称触碰方向（单位向量），`nominal_offset_mm` 为测球
球心相对于分支原点的 3D 偏移（Z 分支沿 Z 轴，X/Y 分支在 XY 平面内）。

=============================================================================
依赖关系
=============================================================================
- geometry_utils.py  : 提供了 add(), normalize(), scale() 等基础向量操作
- cross_probe_geometry.yaml : YAML 格式的几何参数配置文件，定义了法兰适配器、
  探针本体、测杆和各分支的尺寸与偏移参数。
=============================================================================
"""

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from geometry_utils import add, normalize, scale


# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
# 项目根目录（本脚本所在目录的父目录）
PROJECT_DIR = Path(__file__).resolve().parents[1]
# 默认几何参数配置文件的路径
DEFAULT_GEOMETRY = PROJECT_DIR / "config/cross_probe_geometry.yaml"


# ---------------------------------------------------------------------------
# 几何参数加载
# ---------------------------------------------------------------------------
def load_geometry(path):
    """从 YAML 文件加载五向测针的几何参数。

    YAML 文件应包含以下顶层字段：
        - flange_adapter     : 法兰适配器尺寸（板厚 + 主体长度 + 安装台阶）
        - probe_body         : 探针本体尺寸（长度、直径）
        - stylus             : 测杆参数（分支间距、测球直径/半径、测杆半径）
        - branches           : 五个分支各自的名称、方向向量和偏移量
        - derived_nominal    : 由上述数据推导出的名义值（可选，程序可自动计算）

    Parameters
    ----------
    path : str or Path
        YAML 配置文件的路径。

    Returns
    -------
    dict
        解析后的几何参数字典，结构与 YAML 文件一致。

    Raises
    ------
    ValueError
        如果文件内容不是字典类型。
    OSError
        如果文件无法读取。
    """
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid geometry file: {path}")
    return data


# ---------------------------------------------------------------------------
# 基础旋转矩阵构造
# ---------------------------------------------------------------------------
# 三个函数分别构造绕 X / Y / Z 轴的旋转矩阵（弧度制），均为 3x3 矩阵，
# 以列表的列表（list of list）表示，即 matrix[row][col]。
#
#   R_x(theta) = [1    0       0   ]
#                [0  cos(t) -sin(t)]
#                [0  sin(t)  cos(t)]
#
#   R_y(theta) = [ cos(t)  0  sin(t)]
#                [   0     1    0   ]
#                [-sin(t)  0  cos(t)]
#
#   R_z(theta) = [cos(t) -sin(t)  0]
#                [sin(t)  cos(t)  0]
#                [  0       0     1]

def _rot_x(rad):
    """构造绕 X 轴的旋转矩阵（右手定则）。

    Parameters
    ----------
    rad : float
        旋转角度（弧度）。

    Returns
    -------
    list of list
        3x3 旋转矩阵。
    """
    c, s = math.cos(rad), math.sin(rad)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(rad):
    """构造绕 Y 轴的旋转矩阵（右手定则）。

    Parameters
    ----------
    rad : float
        旋转角度（弧度）。

    Returns
    -------
    list of list
        3x3 旋转矩阵。
    """
    c, s = math.cos(rad), math.sin(rad)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(rad):
    """构造绕 Z 轴的旋转矩阵（右手定则）。

    Parameters
    ----------
    rad : float
        旋转角度（弧度）。

    Returns
    -------
    list of list
        3x3 旋转矩阵。
    """
    c, s = math.cos(rad), math.sin(rad)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


# 轴名称到旋转矩阵构造函数的映射字典，供 euler_to_matrix 使用
_AXIS_ROT = {"x": _rot_x, "y": _rot_y, "z": _rot_z}


# ---------------------------------------------------------------------------
# 矩阵 / 向量运算
# ---------------------------------------------------------------------------
def mat_mult(a, b):
    """将两个 3x3 矩阵相乘：result = a * b。

    Parameters
    ----------
    a : list of list
        第一个 3x3 矩阵。
    b : list of list
        第二个 3x3 矩阵。

    Returns
    -------
    list of list
        乘积 (a @ b) 的 3x3 矩阵。
    """
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec(matrix, vector):
    """将 3x3 矩阵与 3 维向量相乘：result = matrix @ vector。

    该操作用于将旋转矩阵作用于一个偏移向量，将其从局部坐标系变换到
    世界坐标系（或法兰坐标系）。

    Parameters
    ----------
    matrix : list of list
        3x3 变换矩阵（通常为旋转矩阵）。
    vector : list of float
        3 维列向量。

    Returns
    -------
    list of float
        变换后的 3 维向量。
    """
    return [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]


# ---------------------------------------------------------------------------
# 欧拉角 → 旋转矩阵
# ---------------------------------------------------------------------------
def euler_to_matrix(sequence, angles_deg):
    """将欧拉角（角度制）转换为 3x3 旋转矩阵。

    支持两种欧拉角约定：
        - **小写序列 (如 "xyz")** ：外旋（extrinsic），每次旋转绕固定世界轴进行。
          当前项目默认使用 "xyz" 外旋，其复合旋转矩阵为：
              R = Rz(yaw) * Ry(pitch) * Rx(roll)
          即先绕 X 转 roll，再绕 Y 转 pitch，最后绕 Z 转 yaw。
        - **大写序列 (如 "XYZ")** ：内旋（intrinsic），每次旋转绕当前局部轴进行。

    Parameters
    ----------
    sequence : str
        三字符旋转序列，如 "xyz"（外旋）或 "XYZ"（内旋）。大小写决定外旋/内旋。
    angles_deg : list of float
        三个欧拉角（角度制），顺序与 sequence 中的轴对应。

    Returns
    -------
    list of list
        3x3 旋转矩阵。

    示例
    ----
    euler_to_matrix("xyz", [0, 0, 90])  → 绕 Z 轴旋转 90 度：
        [[0, -1, 0],
         [1,  0, 0],
         [0,  0, 1]]
    """
    # 将角度转换为弧度
    angles_rad = [math.radians(angle) for angle in angles_deg]
    # 依次生成各轴对应的旋转矩阵
    matrices = [_AXIS_ROT[axis.lower()](angle) for axis, angle in zip(sequence, angles_rad)]

    if sequence.isupper():
        # 内旋 (intrinsic)：按给定顺序连乘，每次绕变换后的局部轴旋转
        # 数学上等价于：R = R_seq[0] @ R_seq[1] @ R_seq[2]
        result = matrices[0]
        for matrix in matrices[1:]:
            result = mat_mult(result, matrix)
        return result

    # 外旋 (extrinsic)：每次绕固定的世界轴旋转
    # 数学上等价于：R = R_seq[2] @ R_seq[1] @ R_seq[0]
    # 即先应用的旋转矩阵放在最右边
    result = matrices[-1]
    for matrix in reversed(matrices[:-1]):
        result = mat_mult(result, matrix)
    return result


# ---------------------------------------------------------------------------
# 分支配置映射
# ---------------------------------------------------------------------------
def branch_map(geometry):
    """构建从分支名称到其 YAML 配置条目的映射字典。

    遍历配置文件中 `branches` 列表，以每个分支的 `name` 字段为键、
    整个分支配置字典为值，生成一个便于快速查找的映射。

    配置文件示例结构：
        branches:
          - name: z
            frame: rmp60_tip_z
            direction: [0.0, 0.0, 1.0]
            nominal_offset_mm: [0.0, 0.0, 83.871]
          - name: x_pos
            frame: rmp60_tip_x_pos
            direction: [-0.548, 0.836, 0.0]
            nominal_offset_mm: [-21.954, 29.898, 0.0]
          ...

    Parameters
    ----------
    geometry : dict
        由 load_geometry() 加载的完整几何参数字典。

    Returns
    -------
    dict
        键为分支名称（如 "z"、"x_pos"），值为分支配置条目的字典。

    Raises
    ------
    ValueError
        如果 `branches` 字段不是列表，或某个分支缺少 `name` 字段。
    """
    branches = geometry.get("branches", [])
    if not isinstance(branches, list):
        raise ValueError("geometry branches must be a list")
    result = {}
    for branch in branches:
        name = branch.get("name")
        if not name:
            raise ValueError("branch without name")
        result[name] = branch
    return result


# ---------------------------------------------------------------------------
# 法兰 → 分支原点的偏移
# ---------------------------------------------------------------------------
def nominal_branch_origin_mm(geometry):
    """计算法兰坐标系到分支原点的名义偏移（仅沿 Z 轴方向）。

    分支原点位于法兰 Z 轴正方向上，偏移量由法兰适配器（adapter）与探针本体
    （probe_body）的长度之和组成：

        flange_to_branch_origin = adapter_effective_length + body_length

    若配置文件中已提供 `derived_nominal.flange_to_branch_origin_mm`，则直接
    使用该值；否则根据 `flange_adapter` 和 `probe_body` 的参数自动累加计算。

    当前名义值：49.4 mm (adapter) + 76.0 mm (body) = 125.4 mm。

    由于偏移沿法兰 Z 轴方向，因此结果为 [0.0, 0.0, total_mm]。

    Parameters
    ----------
    geometry : dict
        几何参数字典。

    Returns
    -------
    list of float
        三维向量 [0, 0, Z]，表示法兰坐标系到分支原点的平移。
    """
    derived = geometry.get("derived_nominal", {})
    if "flange_to_branch_origin_mm" in derived:
        # 使用配置文件中预先计算好的名义值
        return [0.0, 0.0, float(derived["flange_to_branch_origin_mm"])]
    # 自动计算：adapter 有效轴向长度 + probe_body 长度
    adapter = float(geometry["flange_adapter"]["effective_axial_length_mm"])
    body = float(geometry["probe_body"]["length_mm"])
    return [0.0, 0.0, adapter + body]


# ---------------------------------------------------------------------------
# 分支局部偏移（法兰坐标系）
# ---------------------------------------------------------------------------
def branch_local_offset_mm(geometry, branch_name):
    """计算指定分支的测球球心在法兰坐标系中的 3D 偏移。

    该偏移量是两部分之和：
        1. **法兰到分支原点 (branch_origin)**：由 nominal_branch_origin_mm() 给出，
           始终沿法兰 Z 轴方向：[0, 0, adapter + body]。
        2. **分支偏移 (nominal_offset_mm)**：各分支在分支原点坐标系中的偏移量。
           - Z 分支沿 Z 轴：[0, 0, ~83.87]（即标称杆长）
           - X/Y 分支在 XY 平面内：[dx, dy, 0]（水平偏转）

    几何含义：将测针视为从法兰面出发，先沿 Z 轴延伸到分支原点，再从分支原点
    沿分支方向延伸到测球球心。

    以 Z 分支为例：
        local_offset  = [0, 0, 125.4] + [0, 0, 83.87]
                       = [0, 0, 209.27]
    该值即为配置文件中 `derived_nominal.flange_to_axial_ball_center_mm`。

    Parameters
    ----------
    geometry : dict
        几何参数字典。
    branch_name : str
        分支名称（如 "z", "x_pos", "x_neg", "y_pos", "y_neg"）。

    Returns
    -------
    list of float
        三维偏移向量 [dx, dy, dz]，表示法兰系中从原点到测球球心的偏移。

    Raises
    ------
    ValueError
        如果分支名称不存在，或分支的 nominal_offset_mm 格式不正确。
    """
    branches = branch_map(geometry)
    if branch_name not in branches:
        raise ValueError(f"unknown branch {branch_name!r}; choices: {', '.join(sorted(branches))}")
    branch = branches[branch_name]
    offset = branch.get("nominal_offset_mm")
    if not isinstance(offset, list) or len(offset) != 3:
        raise ValueError(f"branch {branch_name!r} has invalid nominal_offset_mm")
    # 法兰到分支原点的偏移 + 分支自身的偏移 = 法兰系中的球心偏移
    return add(nominal_branch_origin_mm(geometry), [float(v) for v in offset])


# ---------------------------------------------------------------------------
# 测球半径
# ---------------------------------------------------------------------------
def ball_radius_mm(geometry):
    """从配置文件中提取测球半径（单位：毫米）。

    优先读取 `ball_radius_mm` 字段；若不存在则尝试 `ball_diameter_mm`
    字段（取一半）。两种都未定义则报错。

    Parameters
    ----------
    geometry : dict
        几何参数字典。

    Returns
    -------
    float
        测球半径（mm）。

    Raises
    ------
    ValueError
        如果配置中既无 ball_radius_mm 也无 ball_diameter_mm。
    """
    stylus = geometry.get("stylus", {})
    if "ball_radius_mm" in stylus:
        return float(stylus["ball_radius_mm"])
    if "ball_diameter_mm" in stylus:
        return 0.5 * float(stylus["ball_diameter_mm"])
    raise ValueError("geometry stylus must define ball_radius_mm or ball_diameter_mm")


# ---------------------------------------------------------------------------
# 核心正向运动学计算
# ---------------------------------------------------------------------------
def compute_branch_point(geometry, pose, branch_name, euler_sequence="xyz", approach=None):
    """根据法兰位姿计算指定分支的测球球心位置。

    这是本模块的核心函数，实现了从机器人的法兰位姿到测针球心坐标的
    完整正向运动学变换。计算步骤：

    **第一步：解析法兰位姿**
        pose = [x, y, z, rx, ry, rz]
        前三个分量是法兰原点在世界系中的位置，后三个是欧拉角。

    **第二步：构造旋转矩阵 R**
        将 (rx, ry, rz) 按指定欧拉角序列转换为 3x3 旋转矩阵。
        默认外旋 "xyz" 对应 R = Rz(ry) * Ry(rz) * Rx(rx)，即
        先绕世界 X 轴转 rx，再绕世界 Y 轴转 ry，最后绕世界 Z 轴转 rz。

    **第三步：计算法兰系中的球心偏移**
        local_offset = branch_origin_offset + branch_nominal_offset
        这是一个在法兰坐标系中表达的向量。

    **第四步：将偏移变换到世界系**
        offset_base = R @ local_offset
        旋转矩阵 R 将法兰系中的向量旋转到世界系方向。

    **第五步：计算球心世界坐标**
        ball_center = flange_position + offset_base

    **第六步（可选）：估算表面接触点**
        若提供了 approach 向量（标称触碰方向），则沿该方向从球心外推
        一个球半径的距离，得到与工件表面的接触点估计。

    Parameters
    ----------
    geometry : dict
        从 YAML 加载的几何参数字典。
    pose : list of float
        长度为 6 的法兰位姿 [x, y, z, rx, ry, rz]，
        平移单位为 mm，旋转单位为 deg（角度制）。
    branch_name : str
        目标分支名称，如 "z"、"x_pos" 等。
    euler_sequence : str, optional
        欧拉角序列（大小写决定外旋/内旋），默认 "xyz"（外旋）。
    approach : list of float, optional
        标称触碰方向向量 [dx, dy, dz]，用于估算球心处的表面接触点。
        如不提供则不会估算接触点。

    Returns
    -------
    dict
        结果字典，包含以下字段：
        - branch                  : 分支名称
        - flange_pose_mm_deg      : 输入的法兰位姿（原样输出）
        - euler_sequence          : 使用的欧拉角序列
        - local_ball_center_offset_mm : 法兰坐标系中的球心偏移
        - ball_center_mm          : 世界坐标系中的球心坐标 [x, y, z]
        - ball_radius_mm          : 测球半径
        - convention_warning      : 关于欧拉角约定的提醒（仅用于调试）
        - approach_vector         : （可选）标称触碰方向单位向量
        - surface_contact_estimate_mm : （可选）估算的表面接触点坐标

    Raises
    ------
    ValueError
        如果 pose 长度不为 6。
    """
    if len(pose) != 6:
        raise ValueError("pose must contain x y z rx ry rz")

    # 1) 法兰位置：pose 的前三个元素
    flange_position = [float(v) for v in pose[:3]]

    # 2) 构造旋转矩阵：将欧拉角 (rx, ry, rz) 从角度制转换为旋转矩阵
    rotation = euler_to_matrix(euler_sequence, [float(v) for v in pose[3:6]])

    # 3) 法兰系中的球心偏移
    #    local_offset = flange_to_branch_origin + branch_nominal_offset
    #    该向量在法兰坐标系中表达
    local_offset = branch_local_offset_mm(geometry, branch_name)

    # 4) 将偏移旋转到世界坐标系：offset_base = R @ local_offset
    #    这一步将法兰系中的向量通过旋转矩阵变换到世界系
    offset_base = mat_vec(rotation, local_offset)

    # 5) 球心世界坐标 = 法兰原点 + 旋转后的偏移
    ball_center = add(flange_position, offset_base)

    # 组装结果字典
    result = {
        "branch": branch_name,
        "flange_pose_mm_deg": [float(v) for v in pose],
        "euler_sequence": euler_sequence,
        "local_ball_center_offset_mm": local_offset,
        "ball_center_mm": ball_center,
        "ball_radius_mm": ball_radius_mm(geometry),
        "convention_warning": (
            "default euler sequence is current ROS RPY candidate xyz; "
            "verify against CR5 GetPose before using for real calibration"
        ),
    }

    # 6) 可选：若有 approach 向量，估算表面接触点
    if approach is not None:
        # 将 approach 归一化为单位向量
        approach_unit = normalize([float(v) for v in approach], "approach vector")
        result["approach_vector"] = approach_unit
        # 接触点估计 = 球心 + approach单位向量 * 球半径
        # 即从球心沿触碰反方向外推一个半径的距离
        result["surface_contact_estimate_mm"] = add(
            ball_center,
            scale(approach_unit, result["ball_radius_mm"]),
        )
    return result


# ---------------------------------------------------------------------------
# 结果打印与输出
# ---------------------------------------------------------------------------
def print_result(result):
    """以人类可读的格式输出分支点计算结果到终端。

    Parameters
    ----------
    result : dict
        compute_branch_point() 返回的结果字典。
    """
    center = result["ball_center_mm"]
    print(f"branch: {result['branch']}")
    print(f"euler_sequence: {result['euler_sequence']}")
    print(
        "local_ball_center_offset_mm: "
        f"[{result['local_ball_center_offset_mm'][0]:.4f}, "
        f"{result['local_ball_center_offset_mm'][1]:.4f}, "
        f"{result['local_ball_center_offset_mm'][2]:.4f}]"
    )
    print(f"ball_center_mm: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"ball_radius_mm: {result['ball_radius_mm']:.4f}")
    if "surface_contact_estimate_mm" in result:
        contact = result["surface_contact_estimate_mm"]
        print(f"surface_contact_estimate_mm: [{contact[0]:.4f}, {contact[1]:.4f}, {contact[2]:.4f}]")
    print("warning: verify CR5 rx/ry/rz convention before using these points for final calibration")


# ---------------------------------------------------------------------------
# CSV 工具函数
# ---------------------------------------------------------------------------
def read_csv_rows(path):
    """读取 CSV 文件并返回每行数据的字典列表。

    使用 csv.DictReader，以 CSV 第一行为字段名，后续每行转换为字典。

    Parameters
    ----------
    path : str or Path
        CSV 文件路径。

    Returns
    -------
    list of dict
        每行为一个字典，键为 CSV 第一行的字段名。
    """
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def pose_from_row(row):
    """从 CSV 行字典中提取长度为 6 的法兰位姿。

    期望 CSV 包含以下列：
        flange_x, flange_y, flange_z, rx, ry, rz

    Parameters
    ----------
    row : dict
        由 read_csv_rows() 返回的 CSV 行字典。

    Returns
    -------
    list of float
        长度为 6 的位姿列表 [x, y, z, rx, ry, rz]。

    Raises
    ------
    ValueError
        如果缺少任何必需的字段。
    """
    fields = ["flange_x", "flange_y", "flange_z", "rx", "ry", "rz"]
    missing = [field for field in fields if row.get(field) in (None, "")]
    if missing:
        raise ValueError(f"CSV row missing pose fields: {missing}")
    return [float(row[field]) for field in fields]


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main():
    """命令行主入口：解析参数，计算测球球心，输出结果。

    支持两种输入方式：
        1. 直接命令行指定 --pose (6 个浮点数)
        2. 通过 --input-csv 指定 CSV 文件并使用 --row-index 选择行

    支持输出格式：
        - 纯文本人类可读（默认）
        - --json：标准输出打印 JSON
        - --json-output PATH：将 JSON 写入文件

    可选参数：
        --branch         : 分支名称（默认 "z"）
        --euler-sequence : 欧拉角序列（默认 "xyz"）
        --approach       : 标称触碰方向，用于估算表面接触点
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default=str(DEFAULT_GEOMETRY))
    parser.add_argument("--branch", default="z", help="branch name from geometry config")
    parser.add_argument("--pose", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--input-csv", help="CSV with flange_x/flange_y/flange_z/rx/ry/rz columns")
    parser.add_argument("--row-index", type=int, default=0, help="zero-based CSV data row index")
    parser.add_argument("--approach", nargs=3, type=float, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--euler-sequence", default="xyz")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-output", help="write result JSON")
    args = parser.parse_args()

    if args.pose is None and args.input_csv is None:
        raise SystemExit("provide --pose or --input-csv")
    if args.pose is not None and args.input_csv is not None:
        raise SystemExit("provide only one of --pose or --input-csv")

    try:
        geometry = load_geometry(args.geometry)
        pose = args.pose
        row_context = None
        if args.input_csv:
            rows = read_csv_rows(args.input_csv)
            if not 0 <= args.row_index < len(rows):
                raise ValueError(f"--row-index out of range; CSV rows: {len(rows)}")
            row_context = rows[args.row_index]
            pose = pose_from_row(row_context)
        result = compute_branch_point(
            geometry,
            pose,
            args.branch,
            euler_sequence=args.euler_sequence,
            approach=args.approach,
        )
        if row_context is not None:
            result["source_csv"] = args.input_csv
            result["source_row_index"] = args.row_index
            result["source_cycle"] = row_context.get("cycle", "")
            result["source_timestamp"] = row_context.get("timestamp", "")
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.json or args.json_output:
        text = json.dumps(result, indent=2)
        if args.json:
            print(text)
        if args.json_output:
            output = Path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n")
            print(f"saved: {output}")
        return

    print_result(result)


if __name__ == "__main__":
    main()
