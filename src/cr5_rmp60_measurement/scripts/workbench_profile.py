#!/usr/bin/env python3
"""Workcell profile: table-relative robot pose and base-relative fixture."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "config/five_branch_workbench_profile.json"


def _vector(value, size: int, label: str, *, positive=False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label}需要{size}个数值")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label}必须是有限数值")
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{label}必须全部大于0")
    return result


def _migrate_v1(profile: dict) -> dict:
    """Convert the former base_link-absolute workcell to the current model."""
    old = copy.deepcopy(profile)
    old_workcell = old["workcell"]
    table = old_workcell["table"]
    origin_base = [float(table["center_xy_mm"][0]), float(table["center_xy_mm"][1]),
                   float(table["top_z_mm"])]
    base = old_workcell["magnetic_base"]
    old["schema_version"] = 3
    old["robot"]["base_in_table_pose_mm_deg"] = [-value for value in origin_base] + [0.0, 0.0, 0.0]
    old["workcell"] = {
        "table": {
            "size_xy_mm": list(table["size_xyz_mm"][:2]),
            "collision_thickness_mm": float(table["size_xyz_mm"][2]),
        },
        "magnetic_base": {
            "center_base_mm": [float(value) for value in base["center_mm"]],
            "size_xyz_mm": list(base["size_xyz_mm"]),
        },
        "sphere_stem": {
            "diameter_mm": float(old_workcell["sphere_stem"]["diameter_mm"]),
            "length_mm": float(old_workcell["sphere_stem"]["length_mm"]),
        },
        "standard_sphere": {"diameter_mm": float(old_workcell["standard_sphere"]["diameter_mm"])},
    }
    return old


def _migrate_v2(profile: dict) -> dict:
    """Convert a table-relative magnetic-base centre to base_link coordinates."""
    old = copy.deepcopy(profile)
    pose = [float(value) for value in old["robot"]["base_in_table_pose_mm_deg"]]
    r_base_table = _transpose(rotation_xyz_deg(pose[3:]))
    t_base_table = _matvec(r_base_table, [-pose[0], -pose[1], -pose[2]])
    center_table = [float(value) for value in old["workcell"]["magnetic_base"]["center_table_mm"]]
    rotated = _matvec(r_base_table, center_table)
    old["workcell"]["magnetic_base"]["center_base_mm"] = [
        rotated[index] + t_base_table[index] for index in range(3)
    ]
    old["workcell"]["magnetic_base"].pop("center_table_mm", None)
    old["schema_version"] = 3
    return old


def validate_profile(profile: dict) -> dict:
    profile = copy.deepcopy(profile)
    version = int(profile.get("schema_version", 0))
    if version == 1:
        profile = _migrate_v1(profile)
    elif version == 2:
        profile = _migrate_v2(profile)
    elif version != 3:
        raise ValueError("不支持的工作台配置版本")
    profile["schema_version"] = 3
    robot = profile["robot"]
    if not str(robot.get("ip", "")).strip():
        raise ValueError("机器人IP不能为空")
    robot["frame_id"] = str(robot.get("frame_id", "base_link")).strip() or "base_link"
    robot["euler_sequence"] = "xyz"
    robot["base_in_table_pose_mm_deg"] = _vector(
        robot.get("base_in_table_pose_mm_deg"), 6, "机械臂基座相对台面位姿"
    )
    workcell = profile["workcell"]
    table = workcell["table"]
    table["size_xy_mm"] = _vector(table["size_xy_mm"], 2, "台面长宽", positive=True)
    table["collision_thickness_mm"] = float(table.get("collision_thickness_mm", 20.0))
    if not math.isfinite(table["collision_thickness_mm"]) or table["collision_thickness_mm"] <= 0:
        raise ValueError("台面碰撞厚度必须大于0")
    base = workcell["magnetic_base"]
    base["center_base_mm"] = _vector(base["center_base_mm"], 3, "磁座base_link坐标")
    base["size_xyz_mm"] = _vector(base["size_xyz_mm"], 3, "磁座尺寸", positive=True)
    stem = workcell["sphere_stem"]
    stem["diameter_mm"] = float(stem["diameter_mm"])
    stem["length_mm"] = float(stem["length_mm"])
    sphere = workcell["standard_sphere"]
    sphere["diameter_mm"] = float(sphere["diameter_mm"])
    tool = profile["tool"]
    tool["mount_xyz_mm"] = _vector(tool["mount_xyz_mm"], 3, "工具安装平移")
    tool["mount_rxyz_deg"] = _vector(tool["mount_rxyz_deg"], 3, "工具安装转角")
    for key, label in (("connector", "法兰连接件"), ("probe_body", "测头"), ("center_stem", "中心支撑杆")):
        item = tool[key]
        item["diameter_mm"] = float(item["diameter_mm"])
        item["length_mm"] = float(item["length_mm"])
        if (not math.isfinite(item["diameter_mm"]) or not math.isfinite(item["length_mm"]) or
                item["diameter_mm"] <= 0 or item["length_mm"] <= 0):
            raise ValueError(f"{label}直径和长度必须大于0")
    tool["ruby_ball_diameter_mm"] = float(tool["ruby_ball_diameter_mm"])
    positive = [stem["diameter_mm"], stem["length_mm"], sphere["diameter_mm"],
                tool["ruby_ball_diameter_mm"]]
    if not all(math.isfinite(item) and item > 0 for item in positive):
        raise ValueError("球杆、标准球和测球尺寸必须大于0")
    jog = profile["jog"]
    for key in ("linear_step_mm", "angular_step_deg", "auto_retract_mm", "timeout_sec",
                "position_tolerance_mm", "orientation_tolerance_deg"):
        jog[key] = float(jog[key])
        if not math.isfinite(jog[key]) or jog[key] <= 0:
            raise ValueError(f"点动参数{key}必须大于0")
    jog["speed"] = int(jog["speed"])
    if not 1 <= jog["speed"] <= 5:
        raise ValueError("速度级别必须在1到5之间")
    return profile


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(matrix):
    return [[matrix[j][i] for j in range(3)] for i in range(3)]


def _matvec(matrix, vector):
    return [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]


def rotation_xyz_deg(rxyz: list[float]) -> list[list[float]]:
    """Extrinsic xyz rotation: Rz * Ry * Rx."""
    rx, ry, rz = [math.radians(float(value)) for value in rxyz]
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    rxm = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    rym = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    rzm = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    return _matmul(rzm, _matmul(rym, rxm))


def table_to_base_transform(profile: dict) -> tuple[list[list[float]], list[float]]:
    """Return p_base = R_base_table * p_table + t_base_table."""
    checked = validate_profile(profile)
    pose = checked["robot"]["base_in_table_pose_mm_deg"]
    r_base_table = _transpose(rotation_xyz_deg(pose[3:]))
    t_base_table = _matvec(r_base_table, [-pose[0], -pose[1], -pose[2]])
    return r_base_table, t_base_table


def table_point_to_base(profile: dict, point_table: list[float]) -> list[float]:
    point = _vector(point_table, 3, "台面坐标")
    rotation, translation = table_to_base_transform(profile)
    rotated = _matvec(rotation, point)
    return [rotated[i] + translation[i] for i in range(3)]


def base_point_to_table(profile: dict, point_base: list[float]) -> list[float]:
    point = _vector(point_base, 3, "base_link坐标")
    checked = validate_profile(profile)
    pose = checked["robot"]["base_in_table_pose_mm_deg"]
    rotated = _matvec(rotation_xyz_deg(pose[3:]), point)
    return [rotated[i] + pose[i] for i in range(3)]


def derived_fixture_base(profile: dict) -> dict:
    """Derive coaxial fixture centres directly in base_link coordinates."""
    checked = validate_profile(profile)
    workcell = checked["workcell"]
    base, stem, sphere = workcell["magnetic_base"], workcell["sphere_stem"], workcell["standard_sphere"]
    bx, by, bz = base["center_base_mm"]
    base_top = bz + base["size_xyz_mm"][2] / 2.0
    return {
        "base_center": [bx, by, bz],
        "stem_center": [bx, by, base_top + stem["length_mm"] / 2.0],
        "sphere_center": [bx, by, base_top + stem["length_mm"] + sphere["diameter_mm"] / 2.0],
    }


def fixture_sphere_height_base_mm(profile: dict) -> float:
    """Known sphere Z for the upright rigid fixture resting on the table."""
    checked = validate_profile(profile)
    pose = checked["robot"]["base_in_table_pose_mm_deg"]
    if abs(pose[3]) > 0.1 or abs(pose[4]) > 0.1:
        raise ValueError("固定球高要求台面与base_link的Z轴平行，请核对基座RX/RY")
    workcell = checked["workcell"]
    table_z = table_point_to_base(checked, [0.0, 0.0, 0.0])[2]
    return (table_z + workcell["magnetic_base"]["size_xyz_mm"][2]
            + workcell["sphere_stem"]["length_mm"]
            + workcell["standard_sphere"]["diameter_mm"] / 2.0)


def magnetic_base_center_from_sphere(profile: dict, sphere_center_base_mm: list[float]) -> list[float]:
    """Register the fixture to a fitted sphere while preserving measured dimensions.

    The supported workbench setup keeps the fixture coaxial with base_link Z.
    Sphere contacts identify the sphere centre, not the magnetic-base size.  The
    base centre therefore follows from the fitted centre and the operator's
    current base height, stem length and sphere diameter.
    """
    checked = validate_profile(profile)
    sphere_center = _vector(sphere_center_base_mm, 3, "拟合球心base_link坐标")
    workcell = checked["workcell"]
    base_height = workcell["magnetic_base"]["size_xyz_mm"][2]
    stem_length = workcell["sphere_stem"]["length_mm"]
    sphere_radius = workcell["standard_sphere"]["diameter_mm"] / 2.0
    return [
        sphere_center[0],
        sphere_center[1],
        sphere_center[2] - sphere_radius - stem_length - base_height / 2.0,
    ]


def register_fixture_to_sphere(profile: dict, sphere_center_base_mm: list[float]) -> dict:
    """Return a checked profile whose fixture is registered to a fitted sphere."""
    checked = validate_profile(profile)
    checked["workcell"]["magnetic_base"]["center_base_mm"] = magnetic_base_center_from_sphere(
        checked, sphere_center_base_mm
    )
    return validate_profile(checked)


def require_supported_contact_geometry(profile: dict) -> None:
    """Refuse tool edits that the flange-frame contact solver cannot follow."""
    tool = validate_profile(profile)["tool"]
    values = [*tool["mount_xyz_mm"], *tool["mount_rxyz_deg"]]
    if any(abs(value) > 1e-9 for value in values):
        raise ValueError("工具安装XYZ/RXYZ非零时，当前接触几何未完成坐标变换；禁止粗拟合和自动路径")
    from cross_probe_model import load_geometry

    geometry = load_geometry(ROOT / "config/cross_probe_geometry.yaml")
    lengths = (
        ("法兰连接件长度", tool["connector"]["length_mm"],
         geometry["flange_adapter"]["effective_axial_length_mm"]),
        ("测头长度", tool["probe_body"]["length_mm"], geometry["probe_body"]["length_mm"]),
        ("中心支撑杆长度", tool["center_stem"]["length_mm"],
         geometry["stylus"]["cross_hub_offset_from_branch_origin_mm"]),
    )
    for label, actual, nominal in lengths:
        if abs(actual - float(nominal)) > 1e-9:
            raise ValueError(f"{label}与当前接触几何配置不一致；禁止粗拟合和自动路径，须先同步五分支偏置与碰撞模型")


def derived_fixture_table(profile: dict) -> dict:
    """Compatibility view of the base-defined fixture expressed in the table frame."""
    checked = validate_profile(profile)
    fixture = derived_fixture_base(checked)
    return {key: base_point_to_table(checked, point) for key, point in fixture.items()}


def load_profile(path: Path = DEFAULT_PROFILE) -> dict:
    return validate_profile(json.loads(Path(path).read_text()))


def canonical_text(profile: dict) -> str:
    return json.dumps(validate_profile(profile), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def profile_sha256(profile: dict) -> str:
    return hashlib.sha256(canonical_text(profile).encode("utf-8")).hexdigest()


def save_profile(path: Path, profile: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checked = validate_profile(profile)
    path.write_text(json.dumps(checked, ensure_ascii=False, indent=2) + "\n")
    return path


def frozen_profile_path(session_dir: Path, placement: str) -> Path:
    return Path(session_dir) / f"{placement}_workbench_profile.json"
