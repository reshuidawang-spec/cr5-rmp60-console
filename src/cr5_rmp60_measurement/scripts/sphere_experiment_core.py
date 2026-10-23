#!/usr/bin/env python3
"""Offline data model and planning for a relocated-sphere CR5 experiment."""

from __future__ import annotations

import csv
csv.field_size_limit(16 * 1024 * 1024)
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from cross_probe_model import branch_local_offset_mm, euler_to_matrix, load_geometry, mat_vec
from sphere_offset_fit import solve_fixed_center_sphere_offset


ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("z", "x_pos", "x_neg", "y_pos", "y_neg")
POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")
DATASET_SPLITS = ("train", "validation", "locked_test", "diagnostic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _finite_vector(values, size: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return array


def ball_center_from_feedback(pose, offset) -> np.ndarray:
    pose = _finite_vector(pose, 6, "feedback pose")
    offset = _finite_vector(offset, 3, "branch offset")
    rotation = euler_to_matrix("xyz", pose[3:].tolist())
    return pose[:3] + np.asarray(mat_vec(rotation, offset.tolist()), dtype=float)


def _fit_sphere_points(points, distance, *, fixed_z=None, f_scale=0.2):
    """Fit XY when the fixture height is known, otherwise fit all three axes."""
    if fixed_z is not None and not math.isfinite(float(fixed_z)):
        raise ValueError("固定球高必须为有限数值")
    a = 2.0 * (points[1:] - points[0])
    b = np.sum(points[1:] ** 2, axis=1) - np.sum(points[0] ** 2)
    if fixed_z is not None:
        b = b - a[:, 2] * fixed_z
        a = a[:, :2]
    dimensions = 3 if fixed_z is None else 2
    initial = (np.linalg.lstsq(a, b, rcond=None)[0] if np.linalg.matrix_rank(a) == dimensions
               else points.mean(axis=0)[:dimensions])

    def center(parameters):
        return parameters if fixed_z is None else np.r_[parameters, float(fixed_z)]

    result = least_squares(
        lambda parameters: np.linalg.norm(points - center(parameters), axis=1) - distance,
        initial, loss="soft_l1", f_scale=f_scale, max_nfev=2000,
    )
    return center(result.x), result


def fit_coarse_center(
    contacts: list[dict], geometry: dict, sphere_radius_mm: float = 10.0, probe_radius_mm: float = 1.0,
    branch_offsets: dict[str, list[float]] | None = None,
    *, fixed_sphere_center_z_mm: float | None = None,
) -> dict:
    """Fixed-radius sphere centre from manually triggered nominal ball centres.

    This is a path seed, not a calibrated sphere centre or a stylus-offset fit.
    """
    if sphere_radius_mm <= 0 or probe_radius_mm <= 0:
        raise ValueError("sphere and probe radii must be positive")
    rows = [row for row in contacts if row.get("kind") == "manual_trigger"]
    counts = Counter(row.get("branch") for row in rows)
    if any(counts[branch] < 1 for branch in BRANCHES) or len(rows) < 5:
        raise ValueError("each of the five branches needs at least one manual DI rising-edge contact")
    points = np.vstack(
        [ball_center_from_feedback(
            row["pose"], (branch_offsets or {}).get(row["branch"], branch_local_offset_mm(geometry, row["branch"]))
        ) for row in rows]
    )
    distance = float(sphere_radius_mm + probe_radius_mm)
    center, result = _fit_sphere_points(points, distance, fixed_z=fixed_sphere_center_z_mm, f_scale=0.3)
    vectors = points - center
    norms = np.linalg.norm(vectors, axis=1)
    if (norms <= 1e-8).any():
        raise ValueError("a measured ball centre coincides with the fitted sphere centre")
    jacobian = -vectors / norms[:, None]
    if fixed_sphere_center_z_mm is not None:
        jacobian = jacobian[:, :2]
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else math.inf
    residual = norms - distance
    if not result.success or not np.isfinite(center).all() or condition > 100.0:
        raise ValueError(f"manual contacts do not constrain a usable 3-D centre (condition={condition:.1f})")
    return {
        "schema_version": 1,
        "method": "coarse_fixed_radius_nominal_five_branch",
        "warning": "Only a planning seed; nominal offsets and robot feedback are not independent metrology.",
        "sphere_center_mm": center.tolist(),
        **({"fixed_sphere_center_z_mm": float(fixed_sphere_center_z_mm)}
           if fixed_sphere_center_z_mm is not None else {}),
        "local_ball_offset_mm": (branch_offsets or {}).get("z", branch_local_offset_mm(geometry, "z")),
        "sphere_radius_mm": float(sphere_radius_mm),
        "probe_radius_mm": float(probe_radius_mm),
        "contact_center_distance_mm": distance,
        "manual_contact_count": len(rows),
        "branch_counts": {branch: counts[branch] for branch in BRANCHES},
        "rms_radial_residual_mm": float(np.sqrt(np.mean(residual**2))),
        "max_abs_radial_residual_mm": float(np.max(np.abs(residual))),
        "normal_condition": condition,
        "normal_singular_values": singular.tolist(),
        "pose_source": "30004 ToolVectorActual; flange interpretation requires current User/Tool verification",
    }


def fit_usable_for_motion(fit: dict) -> tuple[bool, str]:
    center = fit.get("sphere_center_mm")
    if (not isinstance(center, list) or len(center) != 3 or
            not all(math.isfinite(float(value)) for value in center)):
        return False, "粗球心坐标无效"
    if fit.get("method") == "configured_fixture_center":
        if len(fit.get("geometry_fit_sha256", "")) != 64:
            return False, "按输入球位规划需要当前安装的P0几何标定"
        return True, "输入球位仅用于规划，实际坐标误差须纳入执行验证余量"
    if fit.get("method") == "relocated_fixed_height_z_three_face":
        if fit.get("manual_contact_count", 0) < 3 or fit.get("branch_counts", {}).get("z", 0) < 3:
            return False, "后续球位至少需要3次Z针触发"
        if (not math.isfinite(float(fit.get("normal_condition", math.inf))) or
                fit.get("normal_condition", math.inf) > 15.0):
            return False, "三面Z针触点覆盖不足"
        if (not math.isfinite(float(fit.get("max_abs_radial_residual_mm", math.inf))) or
                fit.get("max_abs_radial_residual_mm", math.inf) > 0.25):
            return False, "三面Z针触点与固定球高/球径不一致"
        return True, "OK"
    if fit.get("manual_contact_count", 0) < 10:
        return False, "自动运动前至少要有10次手动触发，并覆盖五个分支"
    if any(fit.get("branch_counts", {}).get(branch, 0) < 2 for branch in BRANCHES):
        return False, "自动运动前每个分支至少要有2次手动触发"
    if (not math.isfinite(float(fit.get("normal_condition", math.inf))) or
            fit.get("normal_condition", math.inf) > 15.0):
        return False, "手动点法向覆盖不足，球心拟合病态"
    if (not math.isfinite(float(fit.get("max_abs_radial_residual_mm", math.inf))) or
            fit.get("max_abs_radial_residual_mm", math.inf) > 0.25):
        return False, "名义五测球几何与手动点不一致，最大径向残差超过0.25 mm"
    return True, "OK"


def fit_relocated_center_z(
    contacts: list[dict], local_z_offset: list[float], sphere_center_z_mm: float,
    sphere_radius_mm: float = 10.0, probe_radius_mm: float = 1.0,
) -> dict:
    """Locate a moved, upright rigid fixture with three dispersed Z-stylus hits.

    The unchanged base, stem and sphere dimensions fix the sphere height.  Two
    lateral centre coordinates remain unknown, leaving a consistency check even
    when only three contacts are supplied.
    """
    rows = [row for row in contacts if row.get("kind") == "manual_trigger" and row.get("branch") == "z"]
    if len(rows) < 3:
        raise ValueError("后续球位定位需要至少3次Z针触发；请在不同球面位置采点")
    distance = float(sphere_radius_mm + probe_radius_mm)
    if distance <= 0 or not math.isfinite(distance) or not math.isfinite(float(sphere_center_z_mm)):
        raise ValueError("球径、测球半径或固定球高无效")
    points = np.vstack([ball_center_from_feedback(row["pose"], local_z_offset) for row in rows])
    if min(float(np.linalg.norm(a - b)) for i, a in enumerate(points) for b in points[i + 1:]) < 2.0:
        raise ValueError("三面Z针触点过于接近；至少相隔2 mm")
    z = float(sphere_center_z_mm)
    result = least_squares(
        lambda xy: np.linalg.norm(points - np.r_[xy, z], axis=1) - distance,
        points[:, :2].mean(axis=0), loss="soft_l1", f_scale=0.2, max_nfev=2000,
    )
    center = np.r_[result.x, z]
    delta = points - center
    norms = np.linalg.norm(delta, axis=1)
    if not result.success or (norms <= 1e-8).any() or not np.isfinite(center).all():
        raise ValueError("三面Z针拟合未收敛")
    normals = delta / norms[:, None]
    singular = np.linalg.svd(normals[:, :2], compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else math.inf
    residual = norms - distance
    fit = {
        "schema_version": 1, "method": "relocated_fixed_height_z_three_face",
        "warning": "Fixed fixture height from the operator profile; not an independent Z measurement.",
        "sphere_center_mm": center.tolist(), "fixed_sphere_center_z_mm": z,
        "local_ball_offset_mm": list(local_z_offset),
        "sphere_radius_mm": float(sphere_radius_mm), "probe_radius_mm": float(probe_radius_mm),
        "contact_center_distance_mm": distance,
        "manual_contact_count": len(rows), "branch_counts": {"z": len(rows)},
        "rms_radial_residual_mm": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_radial_residual_mm": float(np.max(np.abs(residual))),
        "normal_condition": condition,
        "normal_singular_values": singular.tolist(),
        "pose_source": "30004 ToolVectorActual; use only with verified P0 local Z offset",
    }
    usable, reason = fit_usable_for_motion(fit)
    if not usable:
        raise ValueError(reason)
    return fit


def validate_fit_dimensions(fit: dict, profile: dict) -> None:
    """A fixed-radius fit cannot be reused after either contact radius changes."""
    sphere_radius = profile["workcell"]["standard_sphere"]["diameter_mm"] / 2.0
    probe_radius = profile["tool"]["ruby_ball_diameter_mm"] / 2.0
    fitted = [float(fit.get(key, math.nan)) for key in
              ("sphere_radius_mm", "probe_radius_mm", "contact_center_distance_mm")]
    expected = [sphere_radius, probe_radius, sphere_radius + probe_radius]
    if any(not math.isfinite(value) or abs(value - target) > 1e-9
           for value, target in zip(fitted, expected)):
        raise ValueError("标准球或红宝石测球直径已改变；必须按当前尺寸重新拟合粗球心并生成新计划")


def validate_fixture_fit_height(fit: dict, profile: dict) -> None:
    """Do not reuse a free-Z fit or a fit made before the fixture datum changed."""
    from workbench_profile import fixture_sphere_height_base_mm

    expected = fixture_sphere_height_base_mm(profile)
    constrained = float(fit.get("fixed_sphere_center_z_mm", math.nan))
    center_z = float(fit.get("sphere_center_mm", [math.nan] * 3)[2])
    if (not math.isfinite(constrained) or not math.isfinite(center_z)
            or abs(constrained - expected) > 1e-6 or abs(center_z - expected) > 1e-6):
        raise ValueError("球心必须固定为台面Z+磁座高度+球杆长度+标准球半径；旧拟合或结构高度改变后须重新拟合")


def dataset_split_for_placement(placement_id: str, requested: str | None = None) -> str:
    defaults = {"P0": "train", "P1": "train", "P2": "train", "P3": "train",
                "P4": "validation", "P5": "locked_test"}
    split = requested or defaults.get(placement_id)
    if split not in DATASET_SPLITS:
        raise ValueError("新增球位必须预先指定 train、validation、locked_test 或 diagnostic 用途")
    if placement_id in defaults and split != defaults[placement_id]:
        raise ValueError(f"{placement_id} 的预注册用途必须是 {defaults[placement_id]}")
    return split


def fit_manual_relative_seed(contacts: list[dict], geometry: dict, contact_distance_mm: float) -> dict:
    """Optional provisional joint seed; the z offset is fixed to set the gauge.

    This is useful after probe reinstallation when nominal side offsets are no
    longer close enough to plan safely. It is not the frozen G0 calibration.
    """
    rows = [row for row in contacts if row.get("kind") == "manual_trigger"]
    counts = Counter(row.get("branch") for row in rows)
    if len(rows) < 20 or any(counts[branch] < 4 for branch in BRANCHES):
        raise ValueError("provisional relative geometry needs at least four dispersed manual contacts per branch")
    if contact_distance_mm <= 0:
        raise ValueError("contact distance must be positive")
    branches = [row["branch"] for row in rows]
    poses = np.vstack([_finite_vector(row["pose"], 6, "manual pose") for row in rows])
    rotations = np.stack([euler_to_matrix("xyz", pose[3:].tolist()) for pose in poses])
    z_offset = np.asarray(branch_local_offset_mm(geometry, "z"), dtype=float)
    nominal = {branch: np.asarray(branch_local_offset_mm(geometry, branch), dtype=float) for branch in BRANCHES}
    z_points = np.vstack([
        poses[index, :3] + rotations[index] @ z_offset
        for index, branch in enumerate(branches) if branch == "z"
    ])
    try:
        center_guess = np.asarray(
            fit_coarse_center(rows, geometry, contact_distance_mm / 2.0, contact_distance_mm / 2.0)["sphere_center_mm"],
            dtype=float,
        )
    except ValueError:
        center_guess = z_points.mean(axis=0)
    parameters = np.concatenate([center_guess] + [nominal[branch] for branch in BRANCHES[1:]])

    def residual(x):
        center = x[:3]
        offset_map = {"z": z_offset}
        offset_map.update({branch: x[3 + 3 * i:6 + 3 * i] for i, branch in enumerate(BRANCHES[1:])})
        balls = np.vstack([
            poses[index, :3] + rotations[index] @ offset_map[branch]
            for index, branch in enumerate(branches)
        ])
        return np.linalg.norm(balls - center, axis=1) - contact_distance_mm

    result = least_squares(residual, parameters, loss="soft_l1", f_scale=0.2, max_nfev=5000)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-8))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else math.inf
    signed = residual(result.x)
    if not result.success or rank < 15 or condition > 1e4 or float(np.max(np.abs(signed))) > 0.5:
        raise ValueError(f"manual joint seed is not identifiable/stable: rank={rank}, condition={condition:.1f}, max={np.max(np.abs(signed)):.3f} mm")
    offsets = {"z": z_offset.tolist()}
    offsets.update({branch: result.x[3 + 3 * i:6 + 3 * i].tolist() for i, branch in enumerate(BRANCHES[1:])})
    largest_offset_change = max(
        float(np.linalg.norm(np.asarray(offsets[branch]) - nominal[branch])) for branch in BRANCHES[1:]
    )
    if largest_offset_change > 2.0:
        raise ValueError(f"manual provisional offset differs from nominal by {largest_offset_change:.2f} mm; remeasure physical geometry before auto motion")
    return {
        "schema_version": 1,
        "method": "provisional_manual_relative_seed_fixed_z_gauge",
        "warning": "Planning seed only; formal G0 calibration and independent validation are still required.",
        "sphere_center_mm": result.x[:3].tolist(),
        "local_offsets_mm": offsets,
        "rank": rank,
        "condition": condition,
        "rms_radial_residual_mm": float(np.sqrt(np.mean(signed**2))),
        "max_abs_radial_residual_mm": float(np.max(np.abs(signed))),
        "contact_count": len(rows),
        "largest_offset_change_mm": largest_offset_change,
    }


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty plan")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_manual_contacts_csv(
    session_dir: Path, placement: str, installation_id: str,
    standard_sphere_id: str, robot_ip: str,
) -> tuple[Path, int]:
    """Export the already persisted manual triggers without changing their raw JSONL."""
    session_dir = Path(session_dir)
    source = session_dir / f"{placement}_manual.jsonl"
    contacts = read_jsonl(source)
    if not contacts:
        raise ValueError("当前球位尚无手动触发记录，不能保存空数据")
    rows = []
    for index, contact in enumerate(contacts, start=1):
        if (contact.get("kind") != "manual_trigger" or
                contact.get("installation_id") != installation_id or
                contact.get("standard_sphere_id") != standard_sphere_id or
                contact.get("placement_id") != placement or
                contact.get("robot_ip") != robot_ip or
                contact.get("branch") not in BRANCHES):
            raise ValueError(f"第{index}条手动记录与当前安装、球位、标准球或机器人不一致")
        pose = _finite_vector(contact.get("pose"), 6, f"manual pose {index}")
        joints = _finite_vector(contact.get("joints"), 6, f"manual joints {index}")
        rows.append({
            "record_index": index,
            "installation_id": installation_id,
            "standard_sphere_id": standard_sphere_id,
            "placement_id": placement,
            "robot_ip": robot_ip,
            "branch": contact["branch"],
            "wall_time": contact.get("wall_time", ""),
            "feed_sequence": contact.get("feed_sequence", ""),
            "controller_timer": contact.get("controller_timer", ""),
            "digital_input_bits": contact.get("digital_input_bits", ""),
            "di1": contact.get("di1", ""),
            "pose_source": contact.get("pose_source", ""),
            **{f"pose_{name}": float(value) for name, value in zip(POSE_NAMES, pose)},
            **{f"joint_{axis}_deg": float(value) for axis, value in enumerate(joints, start=1)},
        })
    output = session_dir / f"{placement}_manual_contacts.csv"
    write_csv(output, rows)
    return output, len(rows)


def _unit(values) -> np.ndarray:
    vector = _finite_vector(values, 3, "approach normal")
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("zero approach normal")
    return vector / length


def _normal(row: dict) -> np.ndarray:
    return _unit([row[f"approach_{axis}"] for axis in "xyz"])


def select_spread(rows: list[dict], anchor_normal, count: int) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"only {len(rows)} collision-clear candidates; need {count}")
    anchor = _unit(anchor_normal)
    remaining = list(rows)
    first = min(remaining, key=lambda row: math.acos(float(np.clip(_normal(row) @ anchor, -1, 1))))
    selected = [first]
    remaining.remove(first)
    while len(selected) < count:
        chosen = max(
            remaining,
            key=lambda row: min(math.acos(float(np.clip(_normal(row) @ _normal(other), -1, 1))) for other in selected),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def generate_placement_plans(
    session_dir: Path,
    placement_id: str,
    contacts: list[dict],
    safe_poses: dict[str, list[float]],
    fit: dict,
    geometry_path: Path,
    table_z_mm: float,
    branch_offsets: dict[str, list[float]] | None = None,
    stages_to_write: tuple[str, ...] = ("smoke", "geometry", "anchor", "evaluation"),
    dataset_split: str | None = None,
) -> dict[str, Path]:
    """Call the repository's collision-aware candidate planner for each branch.

    The resulting plans remain unapproved for motion until live MoveIt and
    current-state entry checks have passed.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    if not placement_id or any(char in placement_id for char in "/\\ "):
        raise ValueError("placement ID must be non-empty and contain no spaces or slashes")
    dataset_split = dataset_split_for_placement(placement_id, dataset_split)
    manifest_path = session_dir / f"{placement_id}_manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest.get("dataset_split", dataset_split) != dataset_split:
            raise ValueError("球位预注册用途与已有计划不一致")
    for stage in stages_to_write:
        if stage not in ("smoke", "geometry", "anchor", "evaluation"):
            raise ValueError(f"unknown plan stage: {stage}")
        if (session_dir / f"{placement_id}_{stage}_plan.csv").exists():
            raise ValueError(f"{placement_id}/{stage} plan already exists; plans are immutable")
    geometry = load_geometry(geometry_path)
    seed_label = (
        "coarse" if "geometry" in stages_to_write else
        "evaluation" if stages_to_write == ("evaluation",) else "refined"
    )
    fit_path = session_dir / f"{placement_id}_{seed_label}_fit.json"
    fit_text = json.dumps(fit, indent=2, ensure_ascii=False) + "\n"
    if fit_path.exists() and fit_path.read_text() != fit_text:
        raise ValueError("coarse fit has changed; start a new placement/run rather than replacing a planned fit")
    fit_path.write_text(fit_text)
    stages: dict[str, list[dict]] = {"smoke": [], "geometry": [], "anchor": [], "evaluation": []}
    collision_seed_args = []
    if branch_offsets:
        for candidate_branch in BRANCHES:
            seed_path = session_dir / f"{placement_id}_{seed_label}_{candidate_branch}_collision_seed.json"
            seed_text = json.dumps({"local_ball_offset_mm": branch_offsets[candidate_branch]}, indent=2) + "\n"
            if seed_path.exists() and seed_path.read_text() != seed_text:
                raise ValueError(f"collision seed changed: {seed_path}")
            seed_path.write_text(seed_text)
            collision_seed_args.append(f"{candidate_branch}={seed_path}")
    for branch in BRANCHES:
        manual = [row for row in contacts if row.get("kind") == "manual_trigger" and row.get("branch") == branch]
        if not manual or branch not in safe_poses:
            raise ValueError(f"{branch}: manual contact and released taught safe pose are both required")
        anchor = manual[-1]["pose"]
        safe = _finite_vector(safe_poses[branch], 6, f"{branch} safe pose").tolist()
        offset = (branch_offsets or {}).get(branch, branch_local_offset_mm(geometry, branch))
        ball = ball_center_from_feedback(anchor, offset)
        normal = _unit(np.asarray(fit["sphere_center_mm"], dtype=float) - ball).tolist()
        raw_path = session_dir / f"{placement_id}_{seed_label}_{branch}_candidates.csv"
        command = [
            sys.executable, str(ROOT / "scripts/model_based_sphere_point_planner.py"),
            "--reference-fit-json", str(fit_path), "--geometry", str(geometry_path),
            "--branch", branch, "--branch-offset-mm", *map(str, offset),
            *(["--collision-branch-fit", *collision_seed_args] if collision_seed_args else []),
            "--physical-ball-id", f"{fit.get('installation_id', 'unidentified_installation')}:{branch}",
            "--anchor-pose", *map(str, anchor), "--anchor-normal", *map(str, normal),
            "--orientation-mode", "anchor-fixed", "--normal-mode", "anchor-cone",
            "--cone-angle-deg", "12", "--cone-rings", "4", "--cone-samples-per-ring", "12",
            "--roll-angles-deg", "0", "--plan-count", "48",
            "--safe-pose", *map(str, safe),
            "--standoff-mm", "0.5", "--overtravel-mm", "0.3",
            "--sphere-radius-mm", str(fit["sphere_radius_mm"]),
            "--probe-radius-mm", str(fit["probe_radius_mm"]),
            "--transition-standoff-mm", "6", "--table-plane-z-mm", str(table_z_mm),
            "--min-approach-z", "-1", "--max-approach-z", "1",
            "--output", str(raw_path),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"{branch} candidate planning failed:\n{result.stdout}\n{result.stderr}")
        candidates = [row for row in read_csv(raw_path) if row.get("probe_collision_status") == "OK"]
        chosen = select_spread(candidates, normal, 27 if branch == "z" else 16)
        for stage, source_rows, repeats in (
            ("smoke", chosen[:1], 1),
            ("geometry", chosen[1:13] if branch == "z" else chosen[1:10], 1),
            ("anchor", chosen[13:21] if branch == "z" else [], 1),
            ("evaluation", chosen[21:27] if branch == "z" else chosen[10:16], 3),
        ):
            for point_index, source in enumerate(source_rows, start=1):
                for repeat in range(1, repeats + 1):
                    row = dict(source)
                    row.update({
                        "plan_id": f"{placement_id}_{stage}_{branch}_{point_index:02d}_r{repeat}",
                        "standard_placement_id": placement_id,
                        "standard_sphere_id": fit.get("standard_sphere_id", ""),
                        "dataset_group": stage,
                        "sphere_safety_required": "1",
                        "dataset_branch_point_index": str(point_index),
                        "dataset_repeat_index": str(repeat),
                        "dataset_split": dataset_split,
                        "source_plan_id": source["plan_id"],
                        "coarse_fit_sha256": sha256(fit_path),
                    })
                    stages[stage].append(row)
    paths = {}
    for stage, rows in stages.items():
        if stage not in stages_to_write:
            continue
        path = session_dir / f"{placement_id}_{stage}_plan.csv"
        if path.exists():
            raise ValueError(f"{path} already exists; plans are immutable after generation")
        write_csv(path, rows)
        paths[stage] = path
    manifest = {
        "schema_version": 1,
        "placement_id": placement_id,
        "dataset_split": dataset_split,
        "coarse_fit": str(fit_path),
        "geometry": str(geometry_path),
        "geometry_sha256": sha256(geometry_path),
        "table_z_mm": float(table_z_mm),
        "plans": {stage: {"path": str(path), "sha256": sha256(path), "rows": len(stages[stage])} for stage, path in paths.items()},
        "motion_authorized": False,
        "warning": "Plans are nominal seeds. Live MoveIt, scene, tool/frame, release and entry checks are still required.",
    }
    if manifest_path.exists():
        previous_manifest["plans"].update(manifest["plans"])
        manifest = previous_manifest
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return paths


def validate_short_plan(path: Path, *, max_travel_mm: float = 1.0) -> list[dict]:
    rows = read_csv(path)
    if not rows:
        raise ValueError("empty plan")
    ids = set()
    for row in rows:
        plan_id = row.get("plan_id")
        if not plan_id or plan_id in ids:
            raise ValueError("empty or duplicate plan ID")
        ids.add(plan_id)
        if row.get("probe_collision_status") != "OK":
            raise ValueError(f"{plan_id}: local collision check failed")
        travel = float(row["standoff_mm"]) + float(row["overtravel_mm"])
        if not 0 < travel <= max_travel_mm:
            raise ValueError(f"{plan_id}: probe travel {travel:.3f} mm exceeds limit")
        for prefix in ("safe_flange", "transition_flange", "start_flange", "target_flange"):
            _finite_vector([row[f"{prefix}_{name}"] for name in POSE_NAMES], 6, prefix)
    return rows


def fit_new_installation_geometry(
    contacts_path: Path, geometry_path: Path, contact_distance_mm: float, output_path: Path,
    *, fixed_sphere_center_z_mm: float | None = None,
) -> dict:
    """Fit relative branch offsets, with the reference z offset fixed as gauge."""
    all_rows = read_csv(contacts_path)
    if len(all_rows) != 48 or any(row.get("status") != "HIT" for row in all_rows):
        raise ValueError("G0 geometry batch must contain exactly 48 successful, failure-free contacts")
    rows = all_rows
    counts = Counter(row.get("branch") for row in rows)
    if counts["z"] < 12 or any(counts[branch] < 9 for branch in BRANCHES[1:]):
        raise ValueError("geometry fit needs 12 z and 9 contacts for each side branch")
    geometry = load_geometry(geometry_path)
    z_offset = branch_local_offset_mm(geometry, "z")

    def pose(row):
        return _finite_vector([row[f"trigger_flange_{name}"] for name in POSE_NAMES], 6, "trigger pose")

    z_points = np.vstack([ball_center_from_feedback(pose(row), z_offset) for row in rows if row["branch"] == "z"])
    center, z_result = _fit_sphere_points(z_points, contact_distance_mm, fixed_z=fixed_sphere_center_z_mm)
    normal = (z_points - center) / np.linalg.norm(z_points - center, axis=1)[:, None]
    if fixed_sphere_center_z_mm is not None:
        normal = normal[:, :2]
    singular = np.linalg.svd(normal, compute_uv=False)
    z_condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else math.inf
    z_residual = np.linalg.norm(z_points - center, axis=1) - contact_distance_mm
    if not z_result.success or z_condition > 100 or np.max(np.abs(z_residual)) > 0.5:
        raise ValueError("reference branch geometry is not identifiable or has large residuals")
    offsets = {"z": z_offset}
    diagnostics = {"z": {"count": counts["z"], "normal_condition": z_condition,
                         "rms_mm": float(np.sqrt(np.mean(z_residual**2)))}}
    for branch in BRANCHES[1:]:
        branch_poses = np.vstack([pose(row) for row in rows if row["branch"] == branch])
        rotations = np.stack([euler_to_matrix("xyz", item[3:].tolist()) for item in branch_poses])
        fit = solve_fixed_center_sphere_offset(
            branch_poses[:, :3], rotations, center, contact_distance_mm,
            branch_local_offset_mm(geometry, branch), outlier_rejection=False,
        )
        if not fit.get("ok") or fit.get("jacobian_rank") != 3 or fit.get("jacobian_condition", math.inf) > 1000:
            raise ValueError(f"{branch} offset remains unobservable: {fit.get('reason', fit)}")
        if float(np.max(np.abs(fit["signed_residuals"]))) > 0.5:
            raise ValueError(f"{branch} geometry residual exceeds 0.5 mm")
        offsets[branch] = np.asarray(fit["estimate"]).tolist()
        diagnostics[branch] = {"count": counts[branch], "rank": int(fit["jacobian_rank"]),
                               "condition": float(fit["jacobian_condition"]),
                               "rms_mm": float(np.sqrt(np.mean(np.asarray(fit["signed_residuals"]) ** 2)))}
    result = {
        "schema_version": 1, "method": "relative_five_branch_fixed_z_gauge",
        "warning": "Relative offsets only; the reference z offset is held at its nominal gauge value.",
        "sphere_center_mm": center.tolist(), "contact_center_distance_mm": float(contact_distance_mm),
        **({"fixed_sphere_center_z_mm": float(fixed_sphere_center_z_mm)}
           if fixed_sphere_center_z_mm is not None else {}),
        "local_offsets_mm": offsets, "diagnostics": diagnostics,
        "source_contacts": str(contacts_path), "source_sha256": sha256(contacts_path),
        "geometry_sha256": sha256(geometry_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def fit_anchor_center(
    contacts_path: Path, local_z_offset: list[float], contact_distance_mm: float, output_path: Path,
    *, fixed_sphere_center_z_mm: float | None = None,
) -> dict:
    """Register a relocated sphere from A contacts only, never E contacts."""
    rows = read_csv(contacts_path)
    if len(rows) != 8 or any(row.get("status") != "HIT" or row.get("branch") != "z" for row in rows):
        raise ValueError("anchor registration requires exactly eight successful z-branch A contacts")
    points = np.vstack([
        ball_center_from_feedback(
            [row[f"trigger_flange_{name}"] for name in POSE_NAMES], local_z_offset,
        ) for row in rows
    ])
    center, result = _fit_sphere_points(points, contact_distance_mm, fixed_z=fixed_sphere_center_z_mm)
    vectors = points - center
    norms = np.linalg.norm(vectors, axis=1)
    normals = vectors / norms[:, None]
    if fixed_sphere_center_z_mm is not None:
        normals = normals[:, :2]
    singular = np.linalg.svd(normals, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else math.inf
    residual = norms - contact_distance_mm
    rms = float(np.sqrt(np.mean(residual**2)))
    center_noise_scale = rms * math.sqrt(len(rows)) / singular[-1] if singular[-1] > 1e-12 else math.inf
    if (not result.success or condition > 50 or float(np.max(np.abs(residual))) > 0.25
            or center_noise_scale > 0.25):
        raise ValueError(f"anchor sphere centre is not stable: condition={condition:.1f}, max={np.max(np.abs(residual)):.3f} mm")
    report = {
        "schema_version": 1, "method": "anchor_only_fixed_radius_z_branch",
        "warning": "Robot-internal sphere registration, not an independent world-coordinate reference.",
        "sphere_center_mm": center.tolist(), "contact_center_distance_mm": float(contact_distance_mm),
        **({"fixed_sphere_center_z_mm": float(fixed_sphere_center_z_mm)}
           if fixed_sphere_center_z_mm is not None else {}),
        "local_ball_offset_mm": list(local_z_offset), "contact_count": len(rows),
        "rms_radial_residual_mm": rms,
        "max_abs_radial_residual_mm": float(np.max(np.abs(residual))),
        "normal_condition": condition,
        "center_noise_scale_mm": center_noise_scale,
        "source_contacts": str(contacts_path), "source_sha256": sha256(contacts_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
