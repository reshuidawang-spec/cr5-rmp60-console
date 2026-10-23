#!/usr/bin/env python3
"""Check imported contacts against the deployed model; no robot connection."""
import argparse
import json
import math
import sys
from pathlib import Path

from import_sphere_gui_contacts import BRANCHES, digest, read_contacts


def audit(csv_path, runtime_root):
    runtime_root = Path(runtime_root).resolve()
    sys.path.insert(0, str(runtime_root / "scripts"))
    from cross_probe_model import branch_local_offset_mm, euler_to_matrix, load_geometry, mat_vec
    contacts, source = read_contacts(csv_path)
    geometry_path = runtime_root / "config/cross_probe_geometry.yaml"
    profile_path = runtime_root / "config/five_branch_workbench_profile.json"
    policy_path = runtime_root / "config/sphere_motion_policy.json"
    profile = json.loads(profile_path.read_text())
    geometry = load_geometry(geometry_path)
    distance = (profile["workcell"]["standard_sphere"]["diameter_mm"] +
                profile["tool"]["ruby_ball_diameter_mm"]) / 2
    points = []
    for row in contacts:
        pose = row["pose"]
        rotated = mat_vec(euler_to_matrix("xyz", pose[3:]), branch_local_offset_mm(geometry, row["branch"]))
        points.append([pose[i] + rotated[i] for i in range(3)])
    diameter = max(math.dist(a, b) for a in points for b in points)
    lower_bound = max(0.0, diameter / 2 - distance)
    missing = {branch: max(0, 2-source["branch_counts"].get(branch, 0)) for branch in BRANCHES}
    policy = json.loads(policy_path.read_text())
    blockers = []
    if any(missing.values()):
        blockers.append("现有五分支粗定位覆盖不足：" + str({b: n for b, n in missing.items() if n}))
    if lower_bound > 0.25:
        blockers.append("当前名义偏置/分支对应下不存在满足既有0.25mm最大残差门槛的球心；先核对标签与模型变换")
    if not policy.get("commissioning", {}).get("verified", False):
        blockers.append("已部署sphere_motion_policy.json中commissioning.verified=false；当前执行器会拒绝自动运动")
    return {**source, "runtime_root": str(runtime_root),
            "geometry_sha256": digest(geometry_path), "profile_sha256": digest(profile_path),
            "policy_sha256": digest(policy_path), "nominal_ball_centres_diameter_mm": diameter,
            "maximum_exact_shell_diameter_mm": 2*distance,
            "lower_bound_max_radial_residual_mm": lower_bound,
            "missing_to_existing_two_per_branch_gate": missing,
            "sphere_center_mm": None, "blockers": blockers,
            "motion_readiness": "blocked" if blockers else "not_verified",
            "note": "这是名义模型一致性诊断；不是CSV无效、控制器设置变化或测头损坏的结论。通过本检查仍需拟合及完整实机路径预检。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.csv, args.runtime_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
