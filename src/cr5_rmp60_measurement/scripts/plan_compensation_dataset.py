#!/usr/bin/env python3
"""规划五分支补偿数据集（离线数据集规划器）。

本脚本是一个离线数据集规划器，用于生成五分支探针规划 CSV 文件，以支持球面标定。
它基于已标定的标准球心、已注册的分支偏移量、已有的成功锚点姿态以及局部探针碰撞模型，
生成一组广泛的球面法向探测点。

工作流程:
  1. 标定注册表 (calibration_registry) — 读取各分支的探针几何、已标定偏移量等。
  2. 锚点 CSV (anchor_csv) — 从已有的接触记录中提取每个分支的锚点位姿和法向。
  3. 分支候选生成 (generate_branch_candidates) — 基于球面点规划器，在指定锥角范围内
     生成大量候选探测点，并通过局部碰撞模型进行过滤。
  4. 覆盖度选择 (select_coverage_rows) — 用贪心算法迭代挑选能使法向分布最广的候选点，
     同时保持最小角度间隔。
  5. 控制点选择 (select_control_rows) — 在锚点法向附近选择额外的控制点，用于质量监控。
  6. 数据集计划 CSV (plan CSV) — 输出包含训练/验证/测试/控制分组的完整计划 CSV 文件，
     兼容 auto_calibrate_five_branch_sphere.py 的 --plan-input 参数。
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from calibration_registry import (
    DEFAULT_REGISTRY,
    load_registry,
    project_path,
    validate_registry,
)
from model_based_sphere_point_planner import (
    candidate_normals,
    filter_normal,
    load_branch_offset,
    load_reference_sphere,
    plan_candidate,
    sort_rows,
)
from auto_calibrate_five_branch_sphere import (
    DEFAULT_BRANCHES,
    POSE_NAMES,
    load_branch_seed_offsets,
    local_collision_primitives,
)
from cross_probe_model import DEFAULT_GEOMETRY, load_geometry


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR_CSV = (
    PROJECT_DIR
    / "data/2026.6.23/full_auto_five_branch_stable_20260623_1025/contacts.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "data/2026.6.24/compensation_dataset_plan.csv"
DEFAULT_ALL_OUTPUT = PROJECT_DIR / "data/2026.6.24/compensation_dataset_candidates.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_DIR / "data/2026.6.24/compensation_dataset_summary.json"
DEFAULT_SAFE_POSE = [-411.9550, 123.5480, 349.4650, -177.8070, -1.4130, -36.3840]


def read_csv(path):
    """读取 CSV 文件并返回字典列表。"""
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    """将字典列表写入 CSV 文件，自动推断字段顺序。"""
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def row_vector(row, prefix, names):
    """从行数据中提取指定前缀的向量值。"""
    return [float(row[f"{prefix}_{name}"]) for name in names]


def row_pose(row, prefix):
    """从行数据中提取带指定前缀的 6 维位姿向量。"""
    return row_vector(row, prefix, POSE_NAMES)


def row_approach(row):
    """从行数据中提取归一化的接近方向向量（approach vector）。"""
    vector = row_vector(row, "approach", ("x", "y", "z"))
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-12:
        """Parse branch pose."""
        raise ValueError("approach vector cannot be zero")
    return [value / length for value in vector]


def parse_branch_pose(items):
    """解析命令行参数中的分支锚点位姿，格式为 BRANCH=X,Y,Z,RX,RY,RZ。"""
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--branch-anchor-pose entries must be BRANCH=X,Y,Z,RX,RY,RZ")
        branch, values = item.split("=", 1)
        pose = [float(value.strip()) for value in values.split(",") if value.strip()]
        if not branch.strip() or len(pose) != 6:
            """Parse branch normal."""
            raise ValueError("--branch-anchor-pose entries must be BRANCH=X,Y,Z,RX,RY,RZ")
        result[branch.strip()] = pose
    return result


def parse_branch_normal(items):
    """解析命令行参数中的分支锚点法向，格式为 BRANCH=NX,NY,NZ，并归一化。"""
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--branch-anchor-normal entries must be BRANCH=NX,NY,NZ")
        branch, values = item.split("=", 1)
        normal = [float(value.strip()) for value in values.split(",") if value.strip()]
        if not branch.strip() or len(normal) != 3:
            """Return a unit-length copy of a 3D vector, raising ValueError on zero input."""
            raise ValueError("--branch-anchor-normal entries must be BRANCH=NX,NY,NZ")
        result[branch.strip()] = normalize(normal)
    return result


def parse_branch_fit_paths(items):
    """Parse optional per-branch fit overrides in BRANCH=PATH form."""
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--branch-fit-json entries must be BRANCH=PATH")
        branch, path = (part.strip() for part in item.split("=", 1))
        if not branch or not path:
            raise ValueError("--branch-fit-json entries must be BRANCH=PATH")
        fit_path = Path(path)
        if not fit_path.exists():
            raise ValueError(f"--branch-fit-json path does not exist: {fit_path}")
        result[branch] = fit_path.resolve()
    return result


def load_pose_family_config(path, branches):
    """Load explicitly taught pose families for pose-held-out experiments."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("pose family config schema_version must be 1")
    configured = payload.get("branches")
    if not isinstance(configured, dict):
        raise ValueError("pose family config must contain a branches object")
    result = {}
    for branch in branches:
        families = configured.get(branch)
        if not isinstance(families, list) or not families:
            raise ValueError(f"pose family config missing non-empty branch {branch!r}")
        seen_ids = set()
        normalized = []
        for family in families:
            family_id = str(family.get("id", "")).strip()
            split = str(family.get("dataset_split", "")).strip()
            pose = family.get("anchor_pose")
            normal = family.get("anchor_normal")
            safe_pose = family.get("safe_pose")
            approach_normals = family.get("approach_normals")
            local_angles = family.get("local_approach_angles_deg")
            approach_groups = family.get("approach_groups")
            repeats = family.get("repeats")
            if not family_id or family_id in seen_ids:
                raise ValueError(f"{branch}: pose family IDs must be non-empty and unique")
            if split not in ("train", "validation", "test"):
                raise ValueError(f"{branch}/{family_id}: dataset_split must be train, validation or test")
            if not isinstance(pose, list) or len(pose) != 6:
                raise ValueError(f"{branch}/{family_id}: anchor_pose must contain 6 values")
            if not isinstance(normal, list) or len(normal) != 3:
                raise ValueError(f"{branch}/{family_id}: anchor_normal must contain 3 values")
            if safe_pose is not None and (not isinstance(safe_pose, list) or len(safe_pose) != 6):
                raise ValueError(f"{branch}/{family_id}: safe_pose must contain 6 values")
            if approach_normals is not None:
                if not isinstance(approach_normals, list) or not approach_normals:
                    raise ValueError(f"{branch}/{family_id}: approach_normals must be a non-empty list")
                approach_normals = [normalize(value) for value in approach_normals]
                if local_angles is not None:
                    if not isinstance(local_angles, list) or len(local_angles) != len(approach_normals):
                        raise ValueError(
                            f"{branch}/{family_id}: local_approach_angles_deg must match approach_normals"
                        )
                    local_angles = [float(value) for value in local_angles]
                if approach_groups is not None:
                    if not isinstance(approach_groups, list) or len(approach_groups) != len(approach_normals):
                        raise ValueError(
                            f"{branch}/{family_id}: approach_groups must match approach_normals"
                        )
                    approach_groups = [str(value).strip() for value in approach_groups]
                    if any(value not in ("comp", "drift_ctrl") for value in approach_groups):
                        raise ValueError(
                            f"{branch}/{family_id}: approach_groups only supports comp/drift_ctrl"
                        )
            elif local_angles is not None:
                raise ValueError(f"{branch}/{family_id}: local angles require approach_normals")
            elif approach_groups is not None:
                raise ValueError(f"{branch}/{family_id}: approach_groups require approach_normals")
            if repeats is not None:
                try:
                    repeat_count = int(repeats)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{branch}/{family_id}: repeats must be a positive integer"
                    ) from exc
                if isinstance(repeats, bool) or repeat_count <= 0 or float(repeats) != repeat_count:
                    raise ValueError(f"{branch}/{family_id}: repeats must be a positive integer")
                repeats = repeat_count
            seen_ids.add(family_id)
            normalized.append(
                {
                    "id": family_id,
                    "dataset_split": split,
                    "anchor_pose": [float(value) for value in pose],
                    "anchor_normal": normalize(normal),
                    "safe_pose": None if safe_pose is None else [float(value) for value in safe_pose],
                    "approach_normals": approach_normals,
                    "local_approach_angles_deg": local_angles,
                    "approach_groups": approach_groups,
                    "repeats": repeats,
                    "source": str(family.get("source", "")),
                }
            )
        present_splits = {family["dataset_split"] for family in normalized}
        missing_splits = {"train", "validation", "test"} - present_splits
        if missing_splits:
            raise ValueError(
                f"{branch}: pose-held-out config requires train/validation/test families; "
                f"missing {', '.join(sorted(missing_splits))}"
            )
        result[branch] = normalized
    return result


def normalize(vector):
    """归一化三维向量，零向量将抛出 ValueError。"""
    length = math.sqrt(sum(float(value) * float(value) for value in vector))
    if length <= 1e-12:
        raise ValueError("cannot normalize zero vector")
    return [float(value) / length for value in vector]


def normal_angle_deg(a, b):
    """计算两个法向量之间的夹角（度）。"""
    dot = max(-1.0, min(1.0, sum(float(x) * float(y) for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def anchor_from_csv(path, branches):
    """从历史接触 CSV 中读取各分支的锚点位姿和接近法向。"""
    anchors = {}
    normals = {}
    if not path or not Path(path).exists():
        return anchors, normals
    for row in read_csv(path):
        branch = row.get("branch")
        if branch not in branches or branch in anchors:
            continue
        if row.get("status") not in ("", None, "HIT"):
            continue
        for prefix in ("trigger_flange", "start_flange", "transition_flange", "contact_flange"):
            """Registry branch inputs."""
            if all(row.get(f"{prefix}_{name}") not in (None, "") for name in POSE_NAMES):
                anchors[branch] = row_pose(row, prefix)
                break
        if all(row.get(f"approach_{axis}") not in (None, "") for axis in ("x", "y", "z")):
            normals[branch] = row_approach(row)
    return anchors, normals


def registry_branch_inputs(registry, branches):
    """从标定注册表中为每个分支提取可用的探针拟合 JSON 路径和探针 ID。"""
    styli = registry.get("styli", {})
    fits = {}
    stylus_ids = {}
    failures = []
    for branch in branches:
        matches = [
            (stylus_id, item)
            for stylus_id, item in styli.items()
            if item.get("branch") == branch and item.get("allowed_for_auto", False)
        ]
        if len(matches) != 1:
            """Pairwise angles."""
            failures.append(f"{branch}: expected exactly one auto-ready stylus, got {len(matches)}")
            continue
        stylus_id, item = matches[0]
        fits[branch] = project_path(item["canonical_fit_json"])
        stylus_ids[branch] = stylus_id
    if failures:
        raise ValueError("registry branch input failure: " + "; ".join(failures))
    return fits, stylus_ids


def pairwise_angles(normals):
    """计算一组法向量中所有不重复配对之间的夹角（度）。"""
    if len(normals) < 2:
        return []
    return [
        normal_angle_deg(left, right)
        for index, left in enumerate(normals)
        for right in normals[index + 1 :]
    ]


def coverage_metrics(rows):
    """计算选定探测点的覆盖度指标：法向跨度、最小配对夹角、平均最近邻夹角。"""
    normals = [row_approach(row) for row in rows]
    angles = pairwise_angles(normals)
    nearest = []
    for index, normal in enumerate(normals):
        """Candidate score."""
        others = normals[:index] + normals[index + 1 :]
        if others:
            nearest.append(min(normal_angle_deg(normal, other) for other in others))
    return {
        "normal_span_deg": max(angles) if angles else 0.0,
        "min_pairwise_normal_deg": min(angles) if angles else 0.0,
        "mean_nearest_normal_deg": sum(nearest) / len(nearest) if nearest else 0.0,
    }


def candidate_score(row, selected_normals, args):
    """计算候选点的综合评分，综合考虑法向分离度、碰撞净空、局部评分、过渡距离和方位角偏差。

    评分公式（由权重参数调节）:
      score = coverage_weight * separation
            + clearance_weight * clearance
            + local_score_weight * local_score
            - transition_distance_weight * safe_distance
            - orientation_delta_weight * anchor_delta

    其中:
      - separation: 当前候选点法向与已选点法向之间的最小夹角（角度分离度）
      - clearance: 探针最小净空距离（越大越安全）
      - local_score: 候选点本身的局部质量评分
      - safe_distance: 安全过渡距离（越小越好，故取负项）
      - anchor_delta: 与锚点姿态的方位偏差（越小越好，故取负项）
    """
    normal = row_approach(row)
    if not selected_normals:
        separation = 0.0
    else:
        separation = min(normal_angle_deg(normal, previous) for previous in selected_normals)
    """Select coverage rows."""
    clearance = float(row.get("min_probe_clearance_mm") or 0.0)
    safe_distance = float(row.get("candidate_safe_transition_distance_mm") or 0.0)
    anchor_delta = float(row.get("candidate_anchor_orientation_delta_deg") or 0.0)
    local_score = float(row.get("candidate_score") or 0.0)
    return (
        args.coverage_weight * separation
        + args.clearance_weight * clearance
        + args.local_score_weight * local_score
        - args.transition_distance_weight * safe_distance
        - args.orientation_delta_weight * anchor_delta
    )


def select_coverage_rows(rows, count, args):
    """用贪心算法从候选点中选择覆盖度最优的探测点集合。

    该算法迭代执行以下步骤:
      1. 从剩余候选点中筛选出与已选点法向夹角 >= min_separation 的合格点。
      2. 若合格点不足，则逐步降低最小间隔阈值（每次降至当前的 75%，
         但不低于 min_fallback_normal_separation_deg），以尽可能满足数量要求。
      3. 在合格点中，调用 candidate_score 函数进行综合评分，选择总分最高的点。
      4. 将选中点加入已选集合，并从剩余列表中移除。

    通过这种方式，最终选出的点集在保持足够法向分离度的同时最大化法向覆盖范围，
    并兼顾碰撞净空、局部质量和过渡代价等多因素优化。
    """
    selected = []
    selected_normals = []
    remaining = list(sort_rows(rows))
    min_separation = args.min_normal_separation_deg
    while remaining and len(selected) < count:
        eligible = []
        for row in remaining:
            normal = row_approach(row)
            if selected_normals:
                nearest = min(normal_angle_deg(normal, previous) for previous in selected_normals)
                if nearest < min_separation:
                    continue
            eligible.append(row)
        if not eligible and min_separation > args.min_fallback_normal_separation_deg:
            min_separation = max(
                args.min_fallback_normal_separation_deg,
                min_separation * 0.75,
            )
            continue
        if not eligible:
            break
        best = max(
            eligible,
            key=lambda row: (
                candidate_score(row, selected_normals, args),
                float(row.get("candidate_score") or 0.0),
                float(row.get("min_probe_clearance_mm") or 0.0),
            ),
        )
        selected.append(best)
        selected_normals.append(row_approach(best))
        remaining.remove(best)
    if len(selected) < count:
        """Split for point."""
        raise ValueError(
            f"only selected {len(selected)}/{count} rows; "
            "try increasing --cone-angle-deg, --candidate-count, or lowering --min-normal-separation-deg"
        )
    return selected


def select_control_rows(rows, count, args, anchor_normal):
    """在锚点法向附近选择可重复的控制点，用于过程质量监控。

    控制点的选择逻辑:
      1. 优先从与锚点法向夹角 <= control_max_anchor_angle_deg 的候选点中选取。
      2. 按与锚点法向的夹角升序排序，夹角相同时参考 candidate_score 和净空距离。
      3. 若满足角度约束的点不足 count 个，则放宽约束，从所有候选点中
         按与锚点法向的夹角排序选取。

    控制点不参与覆盖度优化，而是集中在锚点附近，用于在测量过程中
    快速检查分支姿态的重复性和稳定性。
    """
    if count <= 0:
        return []
    if anchor_normal is None:
        raise ValueError("--control-points-per-branch requires anchor normals")
    scored = []
    for row in rows:
        normal = row_approach(row)
        angle = normal_angle_deg(normal, anchor_normal)
        if angle > args.control_max_anchor_angle_deg:
            continue
        scored.append(
            (
                angle,
                -float(row.get("candidate_score") or 0.0),
                -float(row.get("min_probe_clearance_mm") or 0.0),
                row,
            )
        )
    if len(scored) < count:
        scored = [
            (
                normal_angle_deg(row_approach(row), anchor_normal),
                -float(row.get("candidate_score") or 0.0),
                -float(row.get("min_probe_clearance_mm") or 0.0),
                row,
            )
            for row in rows
        ]
    selected = [item[-1] for item in sorted(scored)[:count]]
    if len(selected) < count:
        raise ValueError(
            f"only selected {len(selected)}/{count} control rows; "
            "try increasing --candidate-pool-per-branch"
        )
    return selected


def split_assignments(points_per_branch, validation_fraction, test_fraction):
    """Pre-lock direction-level train/validation/test assignments.

    A deterministic low-discrepancy ordering spreads held-out point indices
    through the selected direction sequence. Repeats are assigned later from
    this direction-level map, so one physical direction cannot leak across
    dataset splits.
    """
    validation_count = int(round(points_per_branch * validation_fraction))
    test_count = int(round(points_per_branch * test_fraction))
    if validation_count + test_count >= points_per_branch:
        raise ValueError("validation/test fractions leave no training directions")

    golden_ratio_conjugate = (math.sqrt(5.0) - 1.0) / 2.0
    ordered_indices = sorted(
        range(1, points_per_branch + 1),
        key=lambda index: ((index - 0.5) * golden_ratio_conjugate) % 1.0,
    )
    test_indices = set(ordered_indices[:test_count])
    validation_indices = set(ordered_indices[test_count : test_count + validation_count])
    return {
        index: (
            "test"
            if index in test_indices
            else "validation"
            if index in validation_indices
            else "train"
        )
        for index in range(1, points_per_branch + 1)
    }


def add_dataset_fields(
    row,
    args,
    branch,
    stylus_id,
    point_index,
    repeat_index,
    split,
    source_plan_id,
    group="comp",
    sample_in_branch=None,
    pose_family="",
    points_per_branch=None,
    repeats=None,
):
    """为探测点添加完整的数据集元信息字段。

    包括:
      - plan_id: 格式为 {branch}_{group}_p{point_index:02d}_r{repeat_index:02d}
      - dataset_id / dataset_mode / dataset_group: 数据集标识信息
      - dataset_split: train/validation/test/control 数据划分标记
      - dataset_source_plan_id: 原始计划 ID，用于数据追溯
      - 所有规划参数（点数、重复次数等）持久化到每行中
    """
    result = dict(row)
    family_token = f"_{pose_family}" if pose_family else ""
    result["plan_id"] = f"{branch}{family_token}_{group}_p{point_index:02d}_r{repeat_index:02d}"
    if sample_in_branch is None:
        sample_in_branch = (point_index - 1) * args.repeats + repeat_index
    result["sample_in_branch"] = str(sample_in_branch)
    result["physical_ball_id"] = stylus_id
    result["dataset_id"] = args.dataset_id
    result["dataset_mode"] = "compensation"
    result["dataset_group"] = group
    result["dataset_branch_point_index"] = str(point_index)
    result["dataset_repeat_index"] = str(repeat_index)
    result["dataset_split"] = split
    result["dataset_pose_family"] = pose_family
    result["dataset_source_plan_id"] = source_plan_id
    result["dataset_points_per_branch"] = str(
        args.points_per_branch if points_per_branch is None else points_per_branch
    )
    result["dataset_repeats"] = str(args.repeats if repeats is None else repeats)
    result["dataset_control_points_per_branch"] = str(args.control_points_per_branch)
    result["dataset_control_repeats"] = str(args.control_repeats)
    result["dataset_cone_angle_deg"] = f"{args.cone_angle_deg:.4f}"
    return result


def planner_args_for_branch(
    args, branch, branch_fit, stylus_id, anchor_pose, anchor_normal, collision_fits, safe_pose=None
):
    """组装传递给球面点规划器的分支特定参数。

    该方法收集全局参数并与分支特有的信息（拟合 JSON 路径、探针 ID、
    锚点位姿/法向、碰撞模型）合并，构造一个 SimpleNamespace 对象。
    该对象将传递给 model_based_sphere_point_planner 中的各个函数，
    用于生成和评估候选探测点。
    """
    return SimpleNamespace(
        reference_fit_json=str(args.reference_fit_json),
        geometry=str(args.geometry),
        branch=branch,
        branch_fit_json=str(branch_fit),
        collision_branch_fit=[f"{name}={path}" for name, path in sorted(collision_fits.items())],
        branch_offset_mm=None,
        orientation_mode=args.orientation_mode,
        physical_ball_id=stylus_id,
        output=str(args.output),
        all_output=str(args.all_output) if args.all_output else None,
        plan_count=args.candidate_pool_per_branch,
        normal_mode=args.normal_mode,
        anchor_pose=list(anchor_pose) if anchor_pose else None,
        anchor_normal=list(anchor_normal) if anchor_normal else None,
        cone_angle_deg=args.cone_angle_deg,
        cone_rings=args.cone_rings,
        cone_samples_per_ring=args.cone_samples_per_ring,
        candidate_count=args.candidate_count,
        roll_angles_deg=list(args.roll_angles_deg),
        safe_pose=list(safe_pose if safe_pose is not None else args.safe_pose),
        x_positive_exit_mm=args.x_positive_exit_mm,
        radial_exit_mm=args.radial_exit_mm,
        euler_sequence=args.euler_sequence,
        sphere_radius_mm=args.sphere_radius_mm,
        probe_radius_mm=args.probe_radius_mm,
        standoff_mm=args.standoff_mm,
        transition_standoff_mm=args.transition_standoff_mm,
        overtravel_mm=args.overtravel_mm,
        max_safe_transition_orientation_delta_deg=args.max_safe_transition_orientation_delta_deg,
        min_approach_z=args.min_approach_z,
        max_approach_z=args.max_approach_z,
        min_probe_clearance_mm=args.min_probe_clearance_mm,
        disable_table_plane_check=args.disable_table_plane_check,
        table_plane_z_mm=args.table_plane_z_mm,
        min_table_clearance_mm=args.min_table_clearance_mm,
        rod_collision_radius_mm=args.rod_collision_radius_mm,
        target_stem_exclusion_mm=args.target_stem_exclusion_mm,
        collision_segment_samples=args.collision_segment_samples,
        allow_low_clearance=args.allow_low_clearance,
        allow_collision=args.allow_collision,
        _anchor_approach=None,
        _offset_source="",
    )


def generate_branch_candidates(
    args,
    branch,
    branch_fit,
    stylus_id,
    anchor_pose,
    anchor_normal,
    collision_fits,
    safe_pose=None,
    approach_normals=None,
    local_approach_angles_deg=None,
    approach_groups=None,
):
    """生成并过滤特定分支的候选探测点。

    流程:
      1. 通过 planner_args_for_branch 组装分支参数。
      2. 加载探针几何和参考球面信息（标准球心和接触距离）。
      3. 加载分支偏移量及所有分支的碰撞模型。
      4. 加载局部碰撞基元（primitive），用于快速碰撞检测。
      5. 调用 candidate_normals 生成候选法向方向，并用 filter_normal 过滤不合规方向。
      6. 对每个法向和每个滚动角，调用 plan_candidate 生成候选探测姿态。
      7. 根据 allow_collision / allow_low_clearance 开关过滤碰撞/低净空候选点。
      8. 截取前 candidate_pool_per_branch 个候选点返回。

    返回的候选点列表已按 score 排序（由 sort_rows 完成）。
    """
    local = planner_args_for_branch(
        args,
        branch,
        branch_fit,
        stylus_id,
        anchor_pose,
        anchor_normal,
        collision_fits,
        safe_pose,
    )
    geometry = load_geometry(local.geometry)
    reference_fit, sphere_center = load_reference_sphere(local.reference_fit_json)
    offset, offset_source = load_branch_offset(local, geometry)
    collision_offsets = load_branch_seed_offsets(local.collision_branch_fit)
    collision_offsets[branch] = list(offset)
    local._offset_source = offset_source
    contact_distance = float(
        reference_fit.get(
            "contact_center_distance_mm",
            args.sphere_radius_mm + args.probe_radius_mm,
        )
    )
    primitives, dims, active_origin = local_collision_primitives(
        geometry,
        branch,
        offset,
        local,
        collision_offsets=collision_offsets,
    )
    if approach_normals is None:
        normal_specs = [
            (normal, None, "comp")
            for normal in candidate_normals(local, geometry)
            if filter_normal(local, normal)
        ]
    else:
        angles = local_approach_angles_deg or [None] * len(approach_normals)
        groups = approach_groups or ["comp"] * len(approach_normals)
        normal_specs = [
            (normalize(normal), angle, group)
            for normal, angle, group in zip(approach_normals, angles, groups)
            if filter_normal(local, normalize(normal))
        ]
    rows = []
    index = 1
    for normal, local_angle, dataset_group in normal_specs:
        for roll in local.roll_angles_deg:
            row = plan_candidate(
                local,
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
            if approach_normals is not None:
                row["candidate_source"] = "explicit-local-angle-grid"
                row["candidate_local_angle_deg"] = (
                    "" if local_angle is None else f"{float(local_angle):.4f}"
                )
                row["candidate_dataset_group"] = dataset_group
            rows.append(row)
            index += 1
    if not rows:
        """Plan dataset."""
        raise ValueError(f"{branch}: no candidates generated")
    rows = sort_rows(rows)
    if not args.allow_collision:
        rows = [row for row in rows if row["probe_collision_status"] != "COLLISION"]
    if not args.allow_low_clearance:
        rows = [row for row in rows if row["probe_collision_status"] == "OK"]
    if not rows:
        raise ValueError(f"{branch}: no candidates left after local collision filtering")
    return rows[: args.candidate_pool_per_branch]


def plan_dataset(args):
    """执行完整的数据集规划流程，协调所有分支。

    完整流程:
      1. 加载并验证标定注册表（registry），获取参考球面信息。
      2. 从注册表中提取各分支的探针拟合 JSON 和探针 ID。
      3. 合并锚点位姿和法向来源：优先使用 --branch-anchor-pose / --branch-anchor-normal
         命令行参数，回退到 --anchor-csv 中的历史数据。
      4. 对每个分支:
         a. 调用 generate_branch_candidates 生成候选点。
         b. 调用 select_coverage_rows 选择覆盖度最优的探测点。
         c. 调用 select_control_rows 选择锚点附近的质量控制点。
         d. 计算覆盖度指标（法向跨度、最小夹角等）。
         e. 为选中点添加数据集元信息字段，分配 train/validation/test/control 分组。
      5. 将所有选定行写入计划 CSV 文件。
      6. 将所有候选点（含选中标记）写入候选池 CSV 文件。
      7. 汇总各分支的统计信息并写入 JSON 摘要文件。
    """
    registry = validate_registry(load_registry(args.calibration_registry))
    active_sphere = registry["standard_spheres"][registry["active_standard_sphere_id"]]
    if args.reference_fit_json is None:
        args.reference_fit_json = project_path(active_sphere["canonical_fit_json"])
    fits, stylus_ids = registry_branch_inputs(registry, args.branches)
    fit_overrides = parse_branch_fit_paths(args.branch_fit_json)
    unknown_fit_branches = sorted(set(fit_overrides) - set(args.branches))
    if unknown_fit_branches:
        raise ValueError(
            "--branch-fit-json contains branches not selected by --branches: "
            + ", ".join(unknown_fit_branches)
        )
    fits.update(fit_overrides)
    pose_families = load_pose_family_config(args.pose_family_config, args.branches)
    if pose_families is None:
        explicit_anchors = parse_branch_pose(args.branch_anchor_pose)
        explicit_normals = parse_branch_normal(args.branch_anchor_normal)
        csv_anchors, csv_normals = anchor_from_csv(args.anchor_csv, set(args.branches))
        anchors = {branch: csv_anchors.get(branch) for branch in args.branches}
        anchors.update(explicit_anchors)
        anchor_normals = {branch: csv_normals.get(branch) for branch in args.branches}
        anchor_normals.update(explicit_normals)
        missing = [branch for branch in args.branches if not anchors.get(branch)]
        if missing:
            raise ValueError("missing anchor pose for branches: " + ", ".join(missing))
        if args.normal_mode == "anchor-cone":
            missing = [branch for branch in args.branches if not anchor_normals.get(branch)]
            if missing:
                raise ValueError("missing anchor normal for branches: " + ", ".join(missing))
        pose_families = {
            branch: [{
                "id": "default", "dataset_split": "direction-held-out",
                "anchor_pose": anchors[branch], "anchor_normal": anchor_normals.get(branch),
                "safe_pose": None, "source": str(args.anchor_csv) if args.anchor_csv else "command-line",
            }]
            for branch in args.branches
        }

    selected_rows = []
    all_rows = []
    summary = {
        "schema_version": 3,
        "created_at_unix": time.time(),
        "dataset_id": args.dataset_id,
        "mode": "compensation",
        "reference_fit_json": str(args.reference_fit_json),
        "calibration_registry": str(args.calibration_registry),
        "anchor_csv": str(args.anchor_csv) if args.anchor_csv else None,
        "pose_family_config": str(args.pose_family_config) if args.pose_family_config else None,
        "split_protocol": "pose_family_held_out" if args.pose_family_config else "direction_held_out",
        "points_per_branch": args.points_per_branch,
        "repeats": args.repeats,
        "control_points_per_branch": args.control_points_per_branch,
        "control_repeats": args.control_repeats,
        "control_max_anchor_angle_deg": args.control_max_anchor_angle_deg,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "branches": {},
    }
    for branch in args.branches:
        print(f"\nplanning compensation dataset branch: {branch}", flush=True)
        family_summaries = {}
        sample_cursor = candidate_count = selected_count = selected_row_count = control_count = 0
        for family in pose_families[branch]:
            family_id = family["id"]
            family_points = (
                len(family["approach_normals"])
                if family.get("approach_normals") is not None
                else args.points_per_branch
            )
            family_repeats = family.get("repeats") or args.repeats
            candidates = generate_branch_candidates(
                args, branch, fits[branch], stylus_ids[branch], family["anchor_pose"],
                family.get("anchor_normal"), fits, family.get("safe_pose"),
                family.get("approach_normals"), family.get("local_approach_angles_deg"),
                family.get("approach_groups"),
            )
            selected = select_coverage_rows(candidates, family_points, args)
            control = select_control_rows(
                candidates, args.control_points_per_branch, args, family.get("anchor_normal")
            )
            metrics = coverage_metrics(selected)
            point_splits = (
                {i: family["dataset_split"] for i in range(1, len(selected) + 1)}
                if args.pose_family_config else
                split_assignments(args.points_per_branch, args.validation_fraction, args.test_fraction)
            )
            selected_ids = {row["plan_id"] for row in selected}
            control_ids = {row["plan_id"] for row in control}
            for row in candidates:
                candidate = dict(row)
                candidate.update({
                    "dataset_id": args.dataset_id, "dataset_mode": "compensation_candidate",
                    "dataset_pose_family": family_id,
                    "dataset_split": family["dataset_split"] if args.pose_family_config else "",
                    "candidate_selected": "1" if row["plan_id"] in selected_ids else "0",
                    "candidate_control_selected": "1" if row["plan_id"] in control_ids else "0",
                })
                all_rows.append(candidate)
            for point_index, row in enumerate(selected, start=1):
                for repeat_index in range(1, family_repeats + 1):
                    dataset_group = row.get("candidate_dataset_group", "comp")
                    selected_rows.append(add_dataset_fields(
                        row, args, branch, stylus_ids[branch], point_index, repeat_index,
                        point_splits[point_index], row["plan_id"], group=dataset_group,
                        sample_in_branch=sample_cursor + (point_index - 1) * family_repeats + repeat_index,
                        pose_family=family_id,
                        points_per_branch=family_points,
                        repeats=family_repeats,
                    ))
            sample_cursor += len(selected) * family_repeats
            selected_row_count += len(selected) * family_repeats
            for point_index, row in enumerate(control, start=1):
                for repeat_index in range(1, args.control_repeats + 1):
                    selected_rows.append(add_dataset_fields(
                        row, args, branch, stylus_ids[branch], point_index, repeat_index,
                        "control", row["plan_id"], group="ctrl",
                        sample_in_branch=sample_cursor + (point_index - 1) * args.control_repeats + repeat_index,
                        pose_family=family_id,
                    ))
            sample_cursor += len(control) * args.control_repeats
            candidate_count += len(candidates)
            selected_count += len(selected)
            control_count += len(control)
            family_summaries[family_id] = {
                **family, "candidate_rows": len(candidates), "selected_points": len(selected),
                "selected_rows_with_repeats": len(selected) * family_repeats,
                "family_repeats": family_repeats,
                "control_points": len(control), **metrics,
            }
            print(
                f"  family={family_id} split={family['dataset_split']} selected={len(selected)} "
                f"span={metrics['normal_span_deg']:.3f}deg control={len(control)}", flush=True,
            )
        summary["branches"][branch] = {
            "stylus_id": stylus_ids[branch],
            "branch_fit_json": str(fits[branch]),
            "candidate_rows": candidate_count,
            "selected_points": selected_count,
            "selected_rows_with_repeats": selected_row_count,
            "control_points": control_count,
            "control_rows_with_repeats": control_count * args.control_repeats,
            "pose_families": family_summaries,
        }
    write_csv(args.output, selected_rows)
    if args.all_output:
        write_csv(args.all_output, all_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsaved compensation plan: {args.output}", flush=True)
    """Parse args."""
    print(f"saved candidate pool: {args.all_output}", flush=True)
    print(f"saved summary: {args.summary_output}", flush=True)
    print(f"selected executable rows: {len(selected_rows)}", flush=True)
    return summary


def parse_roll_angles(value):
    """解析逗号分隔的滚动角列表。"""
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(float(item))
    if not result:
        raise argparse.ArgumentTypeError("roll angle list cannot be empty")
    return result


def parse_args():
    """解析命令行参数，配置规划器、数据集和碰撞检测参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--reference-fit-json", type=Path)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--branches", nargs="+", default=list(DEFAULT_BRANCHES))
    parser.add_argument("--dataset-id", default="five_branch_compensation_dataset")
    parser.add_argument("--anchor-csv", type=Path, default=DEFAULT_ANCHOR_CSV)
    parser.add_argument("--pose-family-config", type=Path)
    parser.add_argument(
        "--branch-fit-json", nargs="*", default=[], metavar="BRANCH=PATH",
        help="override the registry fit for selected branches without modifying the registry",
    )
    parser.add_argument("--branch-anchor-pose", nargs="*", default=[], metavar="BRANCH=X,Y,Z,RX,RY,RZ")
    parser.add_argument("--branch-anchor-normal", nargs="*", default=[], metavar="BRANCH=NX,NY,NZ")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-output", type=Path, default=DEFAULT_ALL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--points-per-branch", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--control-points-per-branch",
        type=int,
        default=0,
        help="extra near-anchor quality-control points per branch",
    )
    parser.add_argument("--control-repeats", type=int, default=1)
    parser.add_argument(
        "--control-max-anchor-angle-deg",
        type=float,
        default=6.0,
        help="prefer control points within this angle of the branch anchor normal",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--candidate-pool-per-branch", type=int, default=160)
    parser.add_argument("--normal-mode", choices=("anchor-cone", "fibonacci"), default="anchor-cone")
    parser.add_argument("--orientation-mode", choices=("anchor-fixed", "align-branch"), default="anchor-fixed")
    parser.add_argument("--cone-angle-deg", type=float, default=12.0)
    parser.add_argument("--cone-rings", type=int, default=5)
    parser.add_argument("--cone-samples-per-ring", type=int, default=12)
    parser.add_argument("--candidate-count", type=int, default=720)
    parser.add_argument("--roll-angles-deg", type=parse_roll_angles, default=parse_roll_angles("0"))
    parser.add_argument(
        "--safe-pose",
        nargs=6,
        type=float,
        default=DEFAULT_SAFE_POSE,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    parser.add_argument(
        "--x-positive-exit-mm", type=float, default=0.0,
        help="override each candidate safe pose as transition + X distance with unchanged Y/Z/orientation",
    )
    parser.add_argument(
        "--radial-exit-mm", type=float, default=0.0,
        help="override each safe pose by retreating opposite the candidate approach normal",
    )
    parser.add_argument("--euler-sequence", default="xyz")
    parser.add_argument("--sphere-radius-mm", type=float, default=10.0)
    parser.add_argument("--probe-radius-mm", type=float, default=1.0)
    parser.add_argument("--standoff-mm", type=float, default=0.8)
    parser.add_argument("--transition-standoff-mm", type=float, default=3.0)
    parser.add_argument("--overtravel-mm", type=float, default=0.2)
    parser.add_argument("--max-safe-transition-orientation-delta-deg", type=float, default=90.0)
    """Validate args."""
    parser.add_argument("--min-approach-z", type=float, default=-1.0)
    parser.add_argument("--max-approach-z", type=float, default=0.35)
    parser.add_argument("--min-probe-clearance-mm", type=float, default=0.5)
    parser.add_argument("--disable-table-plane-check", action="store_true")
    parser.add_argument("--table-plane-z-mm", type=float, default=0.0)
    parser.add_argument("--min-table-clearance-mm", type=float, default=5.0)
    parser.add_argument("--rod-collision-radius-mm", type=float)
    parser.add_argument("--target-stem-exclusion-mm", type=float, default=3.0)
    parser.add_argument("--collision-segment-samples", type=int, default=9)
    parser.add_argument("--allow-low-clearance", action="store_true")
    parser.add_argument("--allow-collision", action="store_true")
    parser.add_argument("--min-normal-separation-deg", type=float, default=1.5)
    parser.add_argument("--min-fallback-normal-separation-deg", type=float, default=0.4)
    parser.add_argument("--coverage-weight", type=float, default=8.0)
    parser.add_argument("--clearance-weight", type=float, default=0.5)
    parser.add_argument("--local-score-weight", type=float, default=0.1)
    parser.add_argument("--transition-distance-weight", type=float, default=0.01)
    parser.add_argument("--orientation-delta-weight", type=float, default=0.05)
    return parser.parse_args()


def validate_args(args):
    """验证命令行参数的内部一致性。"""
    if args.points_per_branch <= 0:
        raise ValueError("--points-per-branch must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.control_points_per_branch < 0:
        raise ValueError("--control-points-per-branch cannot be negative")
    if args.control_repeats <= 0:
        raise ValueError("--control-repeats must be positive")
    if args.control_max_anchor_angle_deg <= 0:
        raise ValueError("--control-max-anchor-angle-deg must be positive")
    if not (0.0 <= args.validation_fraction < 1.0):
        raise ValueError("--validation-fraction must be in [0, 1)")
    if not (0.0 <= args.test_fraction < 1.0):
        raise ValueError("--test-fraction must be in [0, 1)")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("--validation-fraction + --test-fraction must be < 1")
    if not args.pose_family_config and int(round(args.points_per_branch * args.validation_fraction)) < 1:
        raise ValueError("--validation-fraction must allocate at least one direction per branch")
    if not args.pose_family_config and int(round(args.points_per_branch * args.test_fraction)) < 1:
        raise ValueError("--test-fraction must allocate at least one direction per branch")
    minimum_candidates = max(args.points_per_branch, args.control_points_per_branch)
    if args.candidate_pool_per_branch < minimum_candidates:
        raise ValueError("--candidate-pool-per-branch must cover requested branch points")
    if args.cone_rings <= 0 or args.cone_samples_per_ring <= 0:
        raise ValueError("--cone-rings and --cone-samples-per-ring must be positive")
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
    if args.min_normal_separation_deg < 0 or args.min_fallback_normal_separation_deg < 0:
        raise ValueError("normal separation limits cannot be negative")
    if args.min_fallback_normal_separation_deg > args.min_normal_separation_deg:
        raise ValueError("--min-fallback-normal-separation-deg cannot exceed --min-normal-separation-deg")
    if args.orientation_mode == "anchor-fixed" and any(abs(value) > 1e-9 for value in args.roll_angles_deg):
        raise ValueError("--orientation-mode anchor-fixed requires --roll-angles-deg 0")


def main():
    """入口函数，解析参数、验证并执行数据集规划。"""
    args = parse_args()
    try:
        validate_args(args)
        plan_dataset(args)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
