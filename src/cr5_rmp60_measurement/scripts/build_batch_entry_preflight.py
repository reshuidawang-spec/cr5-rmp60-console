#!/usr/bin/env python3
"""Build a MoveIt-only current-state to first-safe entry check for one batch."""

import argparse
import csv
import json
from pathlib import Path


POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")


def build_entry_row(first_row, current_pose, current_joints_deg, source):
    return build_entry_rows(first_row, current_pose, current_joints_deg, source, [])[0]


def build_entry_rows(first_row, current_pose, current_joints_deg, source, waypoints):
    targets = [list(values) for values in waypoints]
    targets.append([float(first_row[f"safe_flange_{name}"]) for name in POSE_NAMES])
    starts = [list(current_pose)] + targets[:-1]
    rows = []
    segment_count = len(targets)
    for segment_index, (start, target) in enumerate(zip(starts, targets), start=1):
        row = _build_entry_segment(
            first_row,
            start,
            target,
            current_joints_deg,
            source,
            segment_index,
            segment_count,
        )
        rows.append(row)
    return rows


def _build_entry_segment(
    first_row,
    start_pose,
    target_pose,
    current_joints_deg,
    source,
    segment_index,
    segment_count,
):
    row = dict(first_row)
    original_plan_id = first_row.get("plan_id", "")
    batch_id = first_row.get("schedule_batch_id", "batch")
    row["plan_id"] = f"entry_{batch_id}_segment_{segment_index:02d}"
    row["preflight_mode"] = "transition-only"
    row["entry_target_plan_id"] = original_plan_id if segment_index == segment_count else ""
    row["entry_state_source"] = source
    row["entry_segment_index"] = str(segment_index)
    row["entry_segment_count"] = str(segment_count)
    for name, value in zip(POSE_NAMES, start_pose):
        row[f"safe_flange_{name}"] = f"{float(value):.6f}"
    for name, value in zip(POSE_NAMES, target_pose):
        row[f"transition_flange_{name}"] = f"{float(value):.6f}"
    for index, value in enumerate(current_joints_deg, start=1):
        row[f"entry_joint{index}_deg"] = f"{float(value):.6f}"
    row["batch_entry_preflight"] = "PENDING_MOVEIT"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--current-pose", nargs=6, type=float, required=True)
    parser.add_argument("--current-joints-deg", nargs=6, type=float, required=True)
    parser.add_argument(
        "--waypoint",
        action="append",
        nargs=6,
        type=float,
        default=[],
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    parser.add_argument("--state-source", default="dobot GetPose/GetAngle")
    args = parser.parse_args()

    with Path(args.batch).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise SystemExit("batch is empty")
    first = min(rows, key=lambda row: int(row.get("schedule_index", "0")))
    if first.get("schedule_index") != "1":
        raise SystemExit("batch does not contain schedule_index=1")
    entries = build_entry_rows(
        first,
        args.current_pose,
        args.current_joints_deg,
        args.state_source,
        args.waypoint,
    )
    output_fields = list(fields)
    for field in entries[0]:
        if field not in output_fields:
            output_fields.append(field)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(entries)

    metadata = {
        "schema_version": 1,
        "status": "PENDING_MOVEIT",
        "batch": str(Path(args.batch)),
        "entry_plan": str(output),
        "target_plan_id": first.get("plan_id", ""),
        "waypoints_mm_deg": list(args.waypoint),
        "segments": len(entries),
        "current_pose_mm_deg": list(args.current_pose),
        "current_joints_deg": list(args.current_joints_deg),
        "state_source": args.state_source,
        "motion_sent": False,
    }
    metadata_output = Path(args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
