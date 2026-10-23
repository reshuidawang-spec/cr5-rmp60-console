#!/usr/bin/env python3
"""Import legacy GUI trigger CSVs into the sphere workbench, without motion.

Branch names are preserved. Jog axes are never treated as physical tip IDs.
One source session is selected explicitly when a CSV contains several.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

BRANCHES = ("z", "x_pos", "x_neg", "y_pos", "y_neg")
POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_contacts(path, *, source_session=None):
    path = Path(path).resolve()
    before = digest(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if digest(path) != before:
        raise ValueError("源CSV在读取期间发生变化，请完成本次写入后重新导入")
    contacts, excluded = [], []
    sessions = sorted({r.get("session_id", "") for r in rows
                       if r.get("trigger_di1") == "1" and r.get("command") != "capture_pose"})
    if source_session is None:
        if len(sessions) != 1:
            raise ValueError(f"必须用 --source-session 明确选择一个球位/源会话：{sessions}")
        source_session = sessions[0]
    if source_session not in sessions:
        raise ValueError(f"源会话不存在：{source_session!r}；可用：{sessions}")
    for index, row in enumerate(rows, 1):
        if row.get("trigger_di1") != "1" or row.get("command") == "capture_pose":
            excluded.append({"source_row": index, "reason": "not_a_trigger"})
            continue
        if row.get("session_id", "") != source_session:
            excluded.append({"source_row": index, "reason": "other_source_session"})
            continue
        branch = row.get("branch", "")
        if branch not in BRANCHES:
            raise ValueError(f"第{index}条分支无效：{branch!r}")
        pose = [float(row[f"trigger_flange_{name}"]) for name in POSE_NAMES]
        joints = [float(row[f"trigger_joint_{i}"]) for i in range(1, 7)]
        if not all(math.isfinite(x) for x in pose + joints):
            raise ValueError(f"第{index}条包含非有限位姿/关节角")
        contacts.append({
            "kind": "manual_trigger", "branch": branch, "pose": pose, "joints": joints,
            "di1": 1, "feed_sequence": row.get("trigger_feed_sequence", ""),
            "digital_input_bits": row.get("trigger_digital_input_bits", ""),
            "wall_time": row.get("trigger_feed_wall_time", row.get("timestamp", "")),
            "pose_source": "legacy GUI trigger CSV; coordinate processing unchanged",
            "source_csv": str(path), "source_sha256": before, "source_row": index,
            "source_session_id": source_session, "source_sample_index": row.get("sample_index", ""),
            "source_command": row.get("command", ""),
            "source_physical_ball_id": row.get("physical_ball_id", ""),
            "operator_note": row.get("operator_note", ""),
            "import_key": f"{before}:{index}",
        })
    if not contacts:
        raise ValueError("所选源会话没有DI1触发记录")
    # The Oct22 selector was left at y_neg. Do not silently assign those rows
    # to tips from jog direction: tool rotation changes that relationship.
    axes = {r["source_command"].replace("held ", "").strip() for r in contacts}
    if len({r["branch"] for r in contacts}) == 1 and len(axes) > 1:
        raise ValueError("同一branch标签包含多个点动方向；请先按真实测针标记核对映射，禁止按运动轴自动改针号")
    return contacts, {"source_csv": str(path), "source_sha256": before,
                      "source_session_id": source_session, "raw_rows": len(rows),
                      "trigger_rows": len(contacts), "excluded": excluded,
                      "branch_counts": dict(Counter(r["branch"] for r in contacts))}


def import_contacts(path, session_dir, *, installation_id, standard_sphere_id,
                    placement_id="P0", robot_ip="192.168.5.1", source_session=None):
    if not all(str(x).strip() for x in (installation_id, standard_sphere_id, placement_id, robot_ip)):
        raise ValueError("安装、标准球、球位及机器人IP不能为空")
    if not placement_id.isascii() or not placement_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("球位编号只能包含ASCII字母、数字、下划线和短横线")
    session_dir = Path(session_dir).resolve()
    if (session_dir / f"{placement_id}_manifest.json").exists():
        raise ValueError("该球位已有冻结计划；请选择新的实验目录，不能改写冻结数据")
    contacts, report = read_contacts(path, source_session=source_session)
    identity = dict(installation_id=installation_id, standard_sphere_id=standard_sphere_id,
                    placement_id=placement_id, robot_ip=robot_ip)
    target = session_dir / f"{placement_id}_manual.jsonl"
    existing = [json.loads(line) for line in target.read_text().splitlines() if line.strip()] if target.exists() else []
    if any(any(r.get(k) != v for k, v in identity.items()) for r in existing):
        raise ValueError("目标手动数据与当前安装/标准球/球位/IP不一致，请使用新的实验目录")
    # Row identity stays stable if the operator appends to the source CSV.
    keys = {(r.get("source_csv"), r.get("source_row")): r for r in existing if r.get("source_csv")}
    additions = []
    for contact in contacts:
        previous = keys.get((contact["source_csv"], contact["source_row"]))
        if previous:
            for field in ("pose", "joints", "branch", "source_session_id", "source_command"):
                if previous.get(field) != contact.get(field):
                    raise ValueError("已导入的源记录发生变化，请使用新目录核查，不能悄悄覆盖")
            continue
        additions.append({**contact, **identity})
    session_dir.mkdir(parents=True, exist_ok=True)
    if additions:
        with target.open("a", encoding="utf-8") as handle:
            for record in additions:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = Counter(r["branch"] for r in existing + additions)
    report.update(target_jsonl=str(target), added_rows=len(additions), duplicate_rows=len(contacts)-len(additions),
                  total_branch_counts={b: counts[b] for b in BRANCHES},
                  missing_to_existing_two_per_branch_gate={b: max(0, 2-counts[b]) for b in BRANCHES},
                  sphere_center_mm=None, motion_authorized=False,
                  note="导入仅复用数据；球心须另经现有拟合、模型对应核验和路径预检，不授权运动")
    (session_dir / f"{placement_id}_csv_import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--installation-id", required=True)
    parser.add_argument("--standard-sphere-id", default="standard_sphere_S1")
    parser.add_argument("--placement-id", default="P0")
    parser.add_argument("--robot-ip", default="192.168.5.1")
    parser.add_argument("--source-session")
    args = parser.parse_args()
    report = import_contacts(args.csv, args.session_dir, installation_id=args.installation_id,
                             standard_sphere_id=args.standard_sphere_id, placement_id=args.placement_id,
                             robot_ip=args.robot_ip, source_session=args.source_session)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
