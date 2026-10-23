#!/usr/bin/env python3
"""Read-only workcell preview and non-executing path preflight."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from sphere_experiment_core import ROOT, fit_usable_for_motion, generate_placement_plans, read_csv, sha256, write_csv
from sphere_contact_safety import SAFETY_VERSION, validate_safety_receipt
from workbench_profile import (
    base_point_to_table, derived_fixture_base, frozen_profile_path, profile_sha256,
    save_profile, table_point_to_base, table_to_base_transform, validate_profile,
)


def scene_from_inputs(
    fit: dict, *, table_z: float, table_xy: list[float], table_size: list[float],
    table_thickness: float, base_z: float, base_size: list[float],
) -> dict:
    center = [float(value) for value in fit["sphere_center_mm"]]
    values = [*center, table_z, *table_xy, *table_size, table_thickness, base_z, *base_size]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("场景参数必须是有限数值")
    if any(value <= 0 for value in [*table_size, table_thickness, *base_size, fit["sphere_radius_mm"]]):
        raise ValueError("台面、磁座尺寸及标准球半径必须为正数")
    return {
        "sphere_center": center,
        "sphere_radius": float(fit["sphere_radius_mm"]),
        "table_z": float(table_z),
        "table_center_xy": [float(value) for value in table_xy],
        "table_size": [float(value) for value in table_size],
        "table_thickness": float(table_thickness),
        "base_center": [center[0], center[1], float(base_z)],
        "base_size": [float(value) for value in base_size],
    }


def scene_from_profile(profile: dict, fit: dict | None = None, flange_pose: list[float] | None = None) -> dict:
    profile = validate_profile(profile)
    workcell = profile["workcell"]
    table = workcell["table"]
    base = workcell["magnetic_base"]
    sphere = workcell["standard_sphere"]
    fixture = derived_fixture_base(profile)
    rotation, _ = table_to_base_transform(profile)
    table_origin_base = table_point_to_base(profile, [0.0, 0.0, 0.0])
    table_axis_z = [rotation[index][2] for index in range(3)]
    table_thickness = float(table["collision_thickness_mm"])
    table_center = [table_origin_base[i] - table_axis_z[i] * table_thickness / 2.0 for i in range(3)]
    expected_sphere_base = list(fixture["sphere_center"])
    center = list(expected_sphere_base if fit is None else map(float, fit["sphere_center_mm"]))
    delta = [center[i] - expected_sphere_base[i] for i in range(3)]
    base_center = [fixture["base_center"][i] + delta[i] for i in range(3)]
    stem_center = [fixture["stem_center"][i] + delta[i] for i in range(3)]
    active_sphere_table = base_point_to_table(profile, center)
    active_base_table = base_point_to_table(profile, base_center)
    active_stem_table = base_point_to_table(profile, stem_center)
    result = {
        "frame_id": profile["robot"]["frame_id"],
        "sphere_center": center,
        "sphere_center_table": active_sphere_table,
        "sphere_radius": sphere["diameter_mm"] / 2.0,
        "input_sphere_center": expected_sphere_base,
        "input_sphere_center_table": base_point_to_table(profile, expected_sphere_base),
        "table_z": table_origin_base[2],
        "table_center_xy": table_origin_base[:2],
        "table_top_center": table_origin_base,
        "table_center": table_center,
        "table_rotation": rotation,
        "table_size": list(table["size_xy_mm"]),
        "table_thickness": table_thickness,
        "base_center": base_center,
        "base_center_table": active_base_table,
        "base_size": list(base["size_xyz_mm"]),
        "base_rotation": rotation,
        "sphere_stem": {
            **dict(workcell["sphere_stem"]), "center_mm": stem_center,
            "center_table_mm": active_stem_table, "rotation": rotation,
        },
        "tool": dict(profile["tool"]),
        "flange_pose": list(flange_pose) if flange_pose is not None else None,
        "robot_base_in_table_pose": list(profile["robot"]["base_in_table_pose_mm_deg"]),
        "profile_sha256": profile_sha256(profile),
    }
    return result


def scene_checks(scene: dict) -> list[str]:
    table_relative = "sphere_center_table" in scene
    cx, cy, cz = scene.get("sphere_center_table", scene["sphere_center"])
    radius = scene["sphere_radius"]
    sx, sy = scene["table_size"]
    bx, by, bz = scene.get("base_center_table", scene["base_center"])
    if not table_relative:
        tx, ty = scene["table_center_xy"]
        cx, cy, cz = cx - tx, cy - ty, cz - scene["table_z"]
        bx, by, bz = bx - tx, by - ty, bz - scene["table_z"]
    bsx, bsy, bsz = scene["base_size"]
    findings = []
    if abs(cx - bx) > 1e-9 or abs(cy - by) > 1e-9:
        findings.append("错误：标准球与磁座XY不一致")
    if abs(bx) + bsx / 2 > sx / 2 or abs(by) + bsy / 2 > sy / 2:
        findings.append("错误：磁座外形超出台面XY边界")
    if abs(bz - bsz / 2) > 2.0:
        findings.append("注意：磁座底面与台面顶面的高度相差超过2 mm，请核对Z与尺寸")
    if cz - radius <= 0:
        findings.append("错误：标准球与台面相交")
    if cz - radius < bz + bsz / 2:
        findings.append("注意：标准球与磁座碰撞盒相交，请核对球心Z或磁座高度")
    stem = scene.get("sphere_stem")
    if stem:
        sx0, sy0, sz0 = [float(item) for item in stem.get("center_table_mm", stem["center_mm"])]
        stem_bottom = sz0 - float(stem["length_mm"]) / 2
        stem_top = sz0 + float(stem["length_mm"]) / 2
        if math.hypot(sx0 - cx, sy0 - cy) > 0.5:
            findings.append("错误：标准球杆与标准球XY不同轴")
        if abs(stem_bottom - (bz + bsz / 2)) > 2.0:
            findings.append("注意：球杆底面与磁座顶面的高度相差超过2 mm")
        if abs(stem_top - (cz - radius)) > 2.0:
            findings.append("注意：球杆顶面与标准球底面的高度相差超过2 mm")
    if not findings:
        findings.append("几何自检通过；仍需核对现场实物和MoveIt结果")
    pose = scene.get("robot_base_in_table_pose", [0, 0, 0, 0, 0, 0])
    if abs(float(pose[3])) > 0.1 or abs(float(pose[4])) > 0.1:
        findings.append("错误：当前自动路径算法要求台面与base_link的Z轴平行；请核对基座RX/RY")
    return findings


def write_scene_json(path: Path, scene: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n")
    return path


def apply_scene_to_moveit(scene: dict, timeout_sec: float = 30.0) -> str:
    """Apply the exact preview scene to MoveIt; never execute robot motion."""
    with tempfile.TemporaryDirectory(prefix="cr5_rviz_scene_") as directory:
        scene_path = write_scene_json(Path(directory) / "scene.json", scene)
        command = [sys.executable, str(ROOT / "scripts/apply_calibration_moveit_scene.py"),
                   "--scene-json", str(scene_path), "--timeout-sec", str(timeout_sec)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_sec + 10)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-1500:])
        marker_command = [
            sys.executable, str(ROOT / "scripts/publish_workcell_scene_markers.py"),
            "--scene-json", str(scene_path),
        ]
        marker_result = subprocess.run(marker_command, cwd=ROOT, text=True, capture_output=True, timeout=15)
        if marker_result.returncode:
            raise RuntimeError("场景已进入MoveIt，但RViz可视标记发布失败：" +
                               (marker_result.stderr or marker_result.stdout)[-1500:])
        return "；".join(part for part in (result.stdout.strip(), marker_result.stdout.strip()) if part)


def publish_plan_to_rviz(row: dict) -> str:
    """Publish the selected preview path to RViz without executing it."""
    with tempfile.TemporaryDirectory(prefix="cr5_rviz_path_") as directory:
        path = Path(directory) / "selected.csv"
        write_csv(path, [row])
        command = [sys.executable, str(ROOT / "scripts/publish_five_branch_plan_markers.py"),
                   "--plan", str(path)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-1500:])
        return result.stdout.strip()


def plan_preview(
    placement: str, manual: list[dict], safe_poses: dict, fit: dict,
    geometry_path: Path, table_z: float, branch_offsets: dict | None,
    stages: tuple[str, ...] | None = None,
    dataset_split: str | None = None,
) -> dict[str, list[dict]]:
    usable, reason = fit_usable_for_motion(fit)
    if not usable:
        raise ValueError("粗球心尚未达到候选路径门槛：" + reason)
    stages = stages or (("smoke", "geometry") if placement == "P0" else ("smoke", "anchor"))
    with tempfile.TemporaryDirectory(prefix="cr5_sphere_preview_") as directory:
        paths = generate_placement_plans(
            Path(directory), placement, manual, safe_poses, fit, geometry_path,
            table_z, branch_offsets, stages, dataset_split,
        )
        return {stage: read_csv(path) for stage, path in paths.items()}


def automatic_plan_preview(scene, fit, placement, stages, manual, dataset_split):
    """Search live-model poses; requires no branch anchors or taught safe poses."""
    with tempfile.TemporaryDirectory(prefix="cr5_auto_sphere_") as directory:
        folder = Path(directory)
        scene_path = write_scene_json(folder / "scene.json", scene)
        applied = subprocess.run([sys.executable, str(ROOT / "scripts/apply_calibration_moveit_scene.py"),
            "--scene-json", str(scene_path)], cwd=ROOT, text=True, capture_output=True, timeout=35)
        if applied.returncode:
            raise RuntimeError("自动规划场景加载失败：" + (applied.stderr or applied.stdout)[-1500:])
        source, target = folder / "input.json", folder / "plans.json"
        source.write_text(json.dumps({"fit": fit, "placement": placement, "stages": list(stages),
            "manual": manual, "dataset_split": dataset_split}, ensure_ascii=False))
        result = subprocess.run([sys.executable, str(ROOT / "scripts/automatic_sphere_planner.py"),
            "--input", str(source), "--output", str(target)], cwd=ROOT, text=True,
            capture_output=True, timeout=1800)
        if result.returncode or not target.exists():
            raise RuntimeError("自动五针姿态搜索失败：" + (result.stderr + result.stdout)[-2000:])
        return json.loads(target.read_text())


def freeze_preview_plans(
    session_dir: Path, placement: str, plans: dict[str, list[dict]], fit: dict,
    geometry_path: Path, table_z: float, profile: dict, checked_statuses: dict[str, str],
) -> dict[str, Path]:
    """Promote the exact previewed poses to immutable executable plan files."""
    if not plans:
        raise ValueError("没有可冻结的预览路径")
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    stages = tuple(plans)
    splits = {row.get("dataset_split") for rows in plans.values() for row in rows}
    if len(splits) != 1 or None in splits:
        raise ValueError("预览路径缺少唯一的预注册数据用途")
    dataset_split = splits.pop()
    ids = [row.get("plan_id") for rows in plans.values() for row in rows]
    if (not ids or len(set(ids)) != len(ids) or set(checked_statuses) != set(ids)
            or any(checked_statuses[plan_id] != "OK" for plan_id in ids)):
        raise ValueError("整批候选路径必须全部通过MoveIt检查才能冻结")
    for rows in plans.values():
        for row in rows:
            validate_safety_receipt(row)
    for key in ("sphere_safety_model_sha256", "sphere_safety_world_sha256"):
        if len({row[key] for rows in plans.values() for row in rows}) != 1:
            raise ValueError("整批路径使用了不同模型或场景；请重新整批预检")
    seed_label = (
        "coarse" if placement == "P0" and any(stage in ("smoke", "geometry") for stage in stages)
        else "evaluation" if stages == ("evaluation",) else "refined"
    )
    fit_path = session_dir / f"{placement}_{seed_label}_fit.json"
    fit_text = json.dumps(fit, ensure_ascii=False, indent=2) + "\n"
    if fit_path.exists() and fit_path.read_text() != fit_text:
        raise ValueError("拟合结果已变化，不能覆盖已有冻结路径")
    target_paths = {stage: session_dir / f"{placement}_{stage}_plan.csv" for stage in stages}
    if any(path.exists() for path in target_paths.values()):
        raise ValueError("当前阶段已有正式计划；冻结计划不能覆盖")
    manifest_path = session_dir / f"{placement}_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 2, "placement_id": placement, "plans": {}, "motion_authorized": False,
        "warning": "Previewed poses frozen verbatim; live MoveIt, scene, frame, release and entry checks remain required.",
    }
    if manifest.get("geometry_sha256") not in (None, sha256(geometry_path)):
        raise ValueError("五分支几何文件与已有计划不一致")
    if manifest.get("dataset_split") not in (None, dataset_split):
        raise ValueError("球位预注册用途与已有计划不一致")
    fit_path.write_text(fit_text)
    saved_profile = save_profile(frozen_profile_path(session_dir, placement), profile)
    paths = {}
    for stage, rows in plans.items():
        stage_profile = save_profile(session_dir / f"{placement}_{stage}_workbench_profile.json", profile)
        promoted = []
        for row in rows:
            item = dict(row)
            item["reference_fit_json"] = str(fit_path)
            item["coarse_fit_sha256"] = sha256(fit_path)
            item["workbench_profile_sha256"] = profile_sha256(profile)
            promoted.append(item)
        write_csv(target_paths[stage], promoted)
        paths[stage] = target_paths[stage]
        manifest["plans"][stage] = {
            "path": str(target_paths[stage]), "sha256": sha256(target_paths[stage]),
            "rows": len(rows), "source": "approved_preview",
            "collision_safety_version": SAFETY_VERSION,
            "workbench_profile": str(stage_profile),
            "workbench_profile_sha256": profile_sha256(profile),
            "preview_moveit_checked_ids": sorted(row["plan_id"] for row in rows),
        }
    manifest.update({
        "coarse_fit": str(fit_path), "geometry": str(geometry_path),
        "geometry_sha256": sha256(geometry_path), "table_z_mm": float(table_z),
        "workbench_profile": str(saved_profile), "workbench_profile_sha256": profile_sha256(profile),
        "dataset_split": dataset_split,
        "preview_moveit_checked_ids": sorted(ids),
        "collision_safety_version": SAFETY_VERSION,
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return paths


def moveit_preflight_one(scene: dict, fit: dict, row: dict, seed_joints: list[float] | None = None) -> dict:
    """Check one path through MoveIt services; never call trajectory execution."""
    if seed_joints is not None and (len(seed_joints) != 6 or not all(math.isfinite(float(value)) for value in seed_joints)):
        raise ValueError("MoveIt预检需要一组有效的六关节角")
    with tempfile.TemporaryDirectory(prefix="cr5_sphere_moveit_preview_") as directory:
        folder = Path(directory)
        fit_path, plan_path, checked_path = (folder / name for name in ("fit.json", "plan.csv", "checked.csv"))
        fit_path.write_text(json.dumps(fit, ensure_ascii=False))
        write_csv(plan_path, [row])
        scene_path = write_scene_json(folder / "scene.json", scene)
        scene_command = [
            sys.executable, str(ROOT / "scripts/apply_calibration_moveit_scene.py"),
            "--scene-json", str(scene_path),
        ]
        applied = subprocess.run(scene_command, cwd=ROOT, text=True, capture_output=True, timeout=25)
        if applied.returncode:
            raise RuntimeError("MoveIt场景未加载：" + (applied.stderr or applied.stdout)[-1500:])
        check_command = [
            sys.executable, str(ROOT / "scripts/check_five_branch_plan_moveit.py"),
            "--plan", str(plan_path), "--checked-output", str(checked_path),
            *(["--seed-joints-deg", *map(str, seed_joints)] if seed_joints is not None else []),
            "--sphere-contact-safety", "--safe-transition-mode", "cartesian", "--publish-display-trajectory",
        ]
        checked = subprocess.run(check_command, cwd=ROOT, text=True, capture_output=True, timeout=120)
        if not checked_path.exists():
            raise RuntimeError("MoveIt预检未返回结果：" + (checked.stderr or checked.stdout)[-1500:])
        result = read_csv(checked_path)[0]
        if result.get("moveit_status") == "OK":
            validate_safety_receipt(result)
            from sphere_contact_safety import path_fingerprint
            if path_fingerprint(row, bind_execution=False) != path_fingerprint(result, bind_execution=False):
                raise RuntimeError("预检返回的路径几何与当前预览不一致")
            row.update(result)
        result["preflight_log"] = (checked.stdout + checked.stderr)[-1500:]
        return result


def moveit_preflight_batch(scene: dict, plans: dict[str, list[dict]], seed_joints: list[float] | None = None) -> dict[str, str]:
    """Check every preview row against one loaded scene before freezing."""
    if seed_joints is not None and (len(seed_joints) != 6 or not all(math.isfinite(float(value)) for value in seed_joints)):
        raise ValueError("整批MoveIt预检需要有效的六关节角")
    rows = [row for stage_rows in plans.values() for row in stage_rows]
    ids = [row.get("plan_id") for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(not plan_id for plan_id in ids):
        raise ValueError("候选路径为空或计划ID重复")
    with tempfile.TemporaryDirectory(prefix="cr5_sphere_moveit_batch_") as directory:
        root = Path(directory)
        scene_path = write_scene_json(root / "scene.json", scene)
        plan_path, checked_path = root / "plans.csv", root / "checked.csv"
        write_csv(plan_path, rows)
        applied = subprocess.run(
            [sys.executable, str(ROOT / "scripts/apply_calibration_moveit_scene.py"),
             "--scene-json", str(scene_path)], cwd=ROOT, text=True, capture_output=True, timeout=35,
        )
        if applied.returncode:
            raise RuntimeError("MoveIt场景加载失败：" + (applied.stderr or applied.stdout)[-1500:])
        checked = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_five_branch_plan_moveit.py"),
             "--plan", str(plan_path), "--checked-output", str(checked_path),
             *(["--seed-joints-deg", *map(str, seed_joints)] if seed_joints is not None else []), "--sphere-contact-safety", "--safe-transition-mode", "cartesian"],
            cwd=ROOT, text=True, capture_output=True, timeout=max(180, 12 * len(rows)),
        )
        if checked.returncode or not checked_path.exists():
            raise RuntimeError("整批MoveIt预检失败：" + (checked.stderr or checked.stdout)[-1500:])
        results = read_csv(checked_path)
        statuses = {row.get("plan_id"): row.get("moveit_status", "FAIL") for row in results}
        if len(results) != len(rows) or set(statuses) != set(ids):
            raise RuntimeError("整批MoveIt预检返回的计划ID与候选路径不一致")
        if any(status != "OK" for status in statuses.values()):
            failures = [plan_id for plan_id, status in statuses.items() if status != "OK"]
            raise RuntimeError("整批MoveIt预检未通过：" + ", ".join(failures[:10]))
        by_id = {item["plan_id"]: item for item in results}
        for row in rows:
            result = by_id[row["plan_id"]]
            validate_safety_receipt(result)
            # A result may not replace the path that the user actually previewed.
            from sphere_contact_safety import path_fingerprint
            if path_fingerprint(row, bind_execution=False) != path_fingerprint(result, bind_execution=False):
                raise RuntimeError("预检返回的路径几何与当前预览不一致")
        for row in rows:
            row.update(by_id[row["plan_id"]])
        return statuses
