#!/usr/bin/env python3
"""Fail-closed live executor for one workbench stage, one contact at a time."""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from sphere_contact_safety import SAFETY_VERSION, validate_safety_receipt
from sphere_motion_contract import load_policy, require_commissioned, validate_bundle
from build_batch_entry_preflight import build_entry_rows
from direct_tcp_execute_five_branch_plan import Dashboard, FeedbackReader, parse_reply_values
from sphere_experiment_core import ROOT, fit_usable_for_motion, read_csv, sha256, validate_fit_dimensions, validate_fixture_fit_height, validate_short_plan, write_csv
from sphere_simulation import scene_checks, scene_from_inputs, scene_from_profile, write_scene_json
from workbench_profile import profile_sha256, require_supported_contact_geometry, validate_profile


def hit_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row.get("plan_id", "") for row in read_csv(path) if row.get("status") == "HIT"}


def require_complete_stage(plan_path: Path, contacts_path: Path, label: str) -> None:
    expected = {row["plan_id"] for row in validate_short_plan(plan_path)}
    contacts = read_csv(contacts_path) if contacts_path.exists() else []
    actual = {row.get("plan_id") for row in contacts if row.get("status") == "HIT"}
    if (len(contacts) != len(expected) or len(actual) != len(expected) or actual != expected
            or any(row.get("status") != "HIT" for row in contacts)):
        raise ValueError(f"{label}阶段的HIT记录必须与冻结计划ID逐条一致")


def validate_stage_prerequisites(session_dir: Path, placement: str, stage: str, fit: dict) -> None:
    usable, reason = fit_usable_for_motion(fit)
    if not usable:
        raise ValueError(reason)
    if stage not in ("smoke", "geometry", "anchor", "evaluation"):
        raise ValueError("unknown acquisition stage")
    if stage == "geometry" and placement != "P0":
        raise ValueError("new-installation geometry is collected only at P0")
    if stage != "smoke":
        require_complete_stage(session_dir / f"{placement}_smoke_plan.csv",
                               session_dir / f"{placement}_smoke_contacts.csv", "smoke")
    if stage in ("anchor", "evaluation") or placement != "P0":
        geometry_fit_path = session_dir / "P0_geometry_fit.json"
        if not geometry_fit_path.exists():
            raise ValueError("fit and freeze the current-installation P0 geometry first")
        geometry_fit = json.loads(geometry_fit_path.read_text())
        geometry_contacts = session_dir / "P0_geometry_contacts.csv"
        require_complete_stage(session_dir / "P0_geometry_plan.csv", geometry_contacts, "P0 geometry")
        if geometry_fit.get("source_sha256") != sha256(geometry_contacts):
            raise ValueError("P0几何拟合与当前完整触点文件不一致")
    if stage in ("anchor", "evaluation") and placement == "P0":
        require_complete_stage(session_dir / "P0_geometry_plan.csv",
                               session_dir / "P0_geometry_contacts.csv", "P0 geometry")
    if stage == "evaluation":
        anchor_plan = session_dir / f"{placement}_anchor_plan.csv"
        anchor_contacts = session_dir / f"{placement}_anchor_contacts.csv"
        require_complete_stage(anchor_plan, anchor_contacts, "anchor")
        anchor_fit = session_dir / f"{placement}_anchor_fit.json"
        if not anchor_fit.exists() or json.loads(anchor_fit.read_text()).get("source_sha256") != sha256(anchor_contacts):
            raise ValueError("fit and freeze the independent A-only sphere centre before E evaluation")


def _angle_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def same_state(a: dict, b: dict, pos_tolerance_mm=0.15, joint_tolerance_deg=0.2) -> bool:
    return (
        math.dist(a["pose"][:3], b["pose"][:3]) <= pos_tolerance_mm
        and max(abs(x - y) for x, y in zip(a["joints"], b["joints"])) <= joint_tolerance_deg
    )


class StageRunner:
    def __init__(self, *, session_dir: Path, placement: str, stage: str, robot_ip: str,
                 installation_id: str, geometry_path: Path, table_z_mm: float,
                 magnetic_base_center: list[float], magnetic_base_size: list[float],
                 frame_confirmed: bool, table_size_xy: list[float] = (1200.0, 900.0),
                 table_thickness_mm: float = 20.0, standard_sphere_id: str = "", log=lambda message: None,
                 table_center_xy: list[float] | None = None, workbench_profile: dict | None = None):
        self.session_dir = Path(session_dir)
        self.placement = placement
        self.stage = stage
        self.robot_ip = robot_ip
        self.installation_id = installation_id
        self.geometry_path = Path(geometry_path)
        self.table_z_mm = float(table_z_mm)
        self.table_size_xy = [float(x) for x in table_size_xy]
        self.table_center_xy = None if table_center_xy is None else [float(x) for x in table_center_xy]
        self.table_thickness_mm = float(table_thickness_mm)
        self.magnetic_base_center = [float(x) for x in magnetic_base_center]
        self.magnetic_base_size = [float(x) for x in magnetic_base_size]
        self.frame_confirmed = bool(frame_confirmed)
        self.standard_sphere_id = standard_sphere_id
        self.workbench_profile = None if workbench_profile is None else validate_profile(workbench_profile)
        self.log = log
        self._stop = threading.Event()
        self._process: subprocess.Popen | None = None

    def request_stop(self) -> None:
        self._stop.set()
        proc = self._process
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

        def stop_robot():
            dashboard = Dashboard(self.robot_ip)
            try:
                dashboard.stop()
            except Exception as exc:
                self.log(f"Stop command failed: {exc}")
            finally:
                dashboard.close()

        threading.Thread(target=stop_robot, daemon=True).start()

    def _command(self, command: list[str], *, timeout_sec: float = 180.0) -> None:
        if self._stop.is_set():
            raise RuntimeError("operator stopped the batch")
        self.log("运行: " + " ".join(command[:3]) + " …")
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        self._process = proc
        try:
            output, _ = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self.request_stop()
            output, _ = proc.communicate(timeout=10)
            raise TimeoutError(f"command timed out: {' '.join(command)}\n{output[-3000:]}")
        finally:
            self._process = None
        if output:
            self.log(output[-3000:].strip())
        if self._stop.is_set():
            raise RuntimeError("operator stopped the batch")
        if proc.returncode:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}\n{output[-3000:]}")

    def _check_live_state(self, feedback: FeedbackReader, dashboard: Dashboard) -> dict:
        snapshot = feedback.snapshot(0.5)
        if snapshot["di1"]:
            raise RuntimeError("DI1=1: the probe is not released")
        mode = parse_reply_values(dashboard.command("RobotMode()", timeout=2.0))
        if not mode or int(mode[0]) != 5:
            raise RuntimeError(f"robot is not enabled in expected mode 5: {mode}")
        di = parse_reply_values(dashboard.command("DI(1)", timeout=2.0))
        if not di or int(float(di[0])) != 0:
            raise RuntimeError(f"dashboard DI(1) is not released: {di}")
        flange = parse_reply_values(dashboard.command("GetPose(0,0)", timeout=2.0))
        if len(flange) != 6:
            raise RuntimeError(f"GetPose(0,0) did not return six flange values: {flange}")
        flange = [float(value) for value in flange]
        if math.dist(flange[:3], snapshot["pose"][:3]) > 0.2 or max(
            _angle_error(a, b) for a, b in zip(flange[3:], snapshot["pose"][3:])
        ) > 0.2:
            raise RuntimeError("30004 ToolVectorActual does not agree with GetPose(0,0); frame is not verified")
        return snapshot

    def run(self) -> Path:
        if not self.installation_id or not self.standard_sphere_id or not self.frame_confirmed:
            raise ValueError("current installation, standard-sphere ID and User/Tool=0 flange-frame confirmation are required")
        if len(self.magnetic_base_center) != 3 or len(self.magnetic_base_size) != 3 or any(
            value <= 0 for value in self.magnetic_base_size
        ):
            raise ValueError("enter the actual magnetic base centre and positive dimensions")
        if len(self.table_size_xy) != 2 or any(value <= 0 for value in self.table_size_xy) or self.table_thickness_mm <= 0:
            raise ValueError("enter actual positive table dimensions")
        if self.table_center_xy is not None and (len(self.table_center_xy) != 2 or
                not all(math.isfinite(value) for value in self.table_center_xy)):
            raise ValueError("enter the actual table XY centre")
        plan_path = self.session_dir / f"{self.placement}_{self.stage}_plan.csv"
        rows = validate_short_plan(plan_path)
        policy = load_policy()
        for row in rows:
            validate_safety_receipt(row)
            validate_bundle(row, policy)
        require_commissioned(policy, model_hash=rows[0]["sphere_safety_model_sha256"],
                             robot_ip=self.robot_ip, installation_id=self.installation_id)
        seed_label = (
            "coarse" if self.placement == "P0" and self.stage in ("smoke", "geometry") else
            "evaluation" if self.stage == "evaluation" else "refined"
        )
        fit_path = self.session_dir / f"{self.placement}_{seed_label}_fit.json"
        if not fit_path.exists():
            raise RuntimeError(f"missing frozen {seed_label} planning fit: {fit_path}")
        fit = json.loads(fit_path.read_text())
        if self.workbench_profile is not None:
            require_supported_contact_geometry(self.workbench_profile)
            validate_fit_dimensions(fit, self.workbench_profile)
            validate_fixture_fit_height(fit, self.workbench_profile)
        if (fit.get("installation_id") != self.installation_id or fit.get("placement_id") != self.placement
                or fit.get("standard_sphere_id") != self.standard_sphere_id):
            raise RuntimeError("coarse fit installation/sphere/placement does not match this run")
        # The sphere is fixed above the magnetic base; both share XY for this workcell.
        magnetic_base_center = [*map(float, fit["sphere_center_mm"][:2]), self.magnetic_base_center[2]]
        table_center_xy = self.table_center_xy or fit["sphere_center_mm"][:2]
        if self.workbench_profile is not None:
            scene = scene_from_profile(self.workbench_profile, fit)
        else:
            scene = scene_from_inputs(
                fit, table_z=self.table_z_mm, table_xy=table_center_xy,
                table_size=self.table_size_xy, table_thickness=self.table_thickness_mm,
                base_z=magnetic_base_center[2], base_size=self.magnetic_base_size,
            )
        scene_errors = [finding for finding in scene_checks(scene) if finding.startswith("错误")]
        if scene_errors:
            raise ValueError("现场碰撞场景参数不一致：" + "；".join(scene_errors))
        manual_path = self.session_dir / f"{self.placement}_manual.jsonl"
        if fit.get("method") != "configured_fixture_center" and fit.get("source_manual_sha256") != sha256(manual_path):
            raise RuntimeError("manual trigger data changed after the plan was frozen")
        validate_stage_prerequisites(self.session_dir, self.placement, self.stage, fit)
        if self.stage == "evaluation":
            anchor_fit = json.loads((self.session_dir / f"{self.placement}_anchor_fit.json").read_text())
            if (anchor_fit.get("installation_id") != self.installation_id or
                    anchor_fit.get("standard_sphere_id") != self.standard_sphere_id or
                    anchor_fit.get("placement_id") != self.placement):
                raise RuntimeError("A-only anchor fit belongs to another installation/placement")
        manifest = json.loads((self.session_dir / f"{self.placement}_manifest.json").read_text())
        stage_manifest = manifest["plans"][self.stage]
        if stage_manifest.get("collision_safety_version") != SAFETY_VERSION:
            raise RuntimeError("旧版冻结路径需重新进行全测针碰撞检查并冻结")
        if (stage_manifest.get("source") != "approved_preview" or
                set(stage_manifest.get("preview_moveit_checked_ids", [])) !=
                {row["plan_id"] for row in rows}):
            raise RuntimeError("当前阶段不是经过整批MoveIt检查的正式冻结计划")
        if (not manifest.get("dataset_split") or
                any(row.get("dataset_split") != manifest["dataset_split"] for row in rows)):
            raise RuntimeError("冻结计划的数据用途与manifest不一致")
        if manifest.get("geometry_sha256") != sha256(self.geometry_path):
            raise RuntimeError("five-branch geometry file changed after plan generation")
        expected_profile_hash = stage_manifest.get("workbench_profile_sha256")
        if (not expected_profile_hash or self.workbench_profile is None or
                profile_sha256(self.workbench_profile) != expected_profile_hash):
            raise RuntimeError("workbench model/configuration changed after preview plan freezing")
        frozen_geometry = self.session_dir / "P0_geometry_fit.json"
        if frozen_geometry.exists():
            frozen = json.loads(frozen_geometry.read_text())
            if frozen.get("installation_id") != self.installation_id or frozen.get("standard_sphere_id") != self.standard_sphere_id:
                raise RuntimeError("P0 frozen geometry belongs to a different installation/sphere")
        if stage_manifest["sha256"] != sha256(plan_path):
            raise RuntimeError("the plan changed after generation")
        output = self.session_dir / f"{self.placement}_{self.stage}_contacts.csv"
        if output.exists():
            prior = read_csv(output)
            expected_ids = {row["plan_id"] for row in rows}
            if any(row.get("plan_id") not in expected_ids for row in prior):
                raise RuntimeError("contact CSV contains plan IDs outside the frozen stage")
            if len({row.get("plan_id") for row in prior}) != len(prior):
                raise RuntimeError("contact CSV contains duplicate plan IDs; review before resuming")
            if any(row.get("status") != "HIT" for row in prior):
                raise RuntimeError("previous batch contains a non-HIT row; investigate and start a new run")
            if any(row.get("session_id") != self.installation_id + "_" + self.placement for row in prior):
                raise RuntimeError("contact CSV belongs to another installation/placement")
            if any(row.get("standard_sphere_id") != self.standard_sphere_id for row in prior):
                raise RuntimeError("contact CSV belongs to another standard sphere")
        finished = hit_ids(output)
        if finished:
            self.log(f"跳过已完成的 {len(finished)} 个 HIT，继续未完成条目")

        feedback = FeedbackReader(self.robot_ip)
        dashboard = Dashboard(self.robot_ip)
        feedback.start()
        try:
            feedback.wait_fresh(5.0)
            first_state = self._check_live_state(feedback, dashboard)
            # The physical scene is recreated for this placement; no historical obstacle coordinates are reused.
            active_scene = write_scene_json(
                self.session_dir / f"{self.placement}_{self.stage}_active_scene.json", scene
            )
            scene_command = [
                sys.executable, str(ROOT / "scripts/apply_calibration_moveit_scene.py"),
                "--scene-json", str(active_scene),
            ]
            self._command(scene_command)
            checked = self.session_dir / f"{self.placement}_{self.stage}_moveit_checked.csv"
            self._command([
                sys.executable, str(ROOT / "scripts/check_five_branch_plan_moveit.py"),
                "--plan", str(plan_path), "--checked-output", str(checked),
                "--seed-joints-deg", *map(str, first_state["joints"]),
                "--sphere-contact-safety", "--safe-transition-mode", "cartesian",
            ], timeout_sec=max(180.0, 10.0 * len(rows)))
            checked_rows = read_csv(checked)
            if (len(checked_rows) != len(rows) or
                    {row.get("plan_id") for row in checked_rows} != {row["plan_id"] for row in rows} or
                    any(row.get("moveit_status") != "OK" for row in checked_rows)):
                raise RuntimeError("not every selected row passed MoveIt")
            for row in checked_rows:
                validate_safety_receipt(row)
            for index, row in enumerate(checked_rows, start=1):
                if row["plan_id"] in finished:
                    continue
                if self._stop.is_set():
                    raise RuntimeError("operator stopped the batch")
                state = self._check_live_state(feedback, dashboard)
                entry_rows = build_entry_rows({**row, "schedule_index": "1"}, state["pose"], state["joints"],
                                               "30004 current feedback", [])
                entry = self.session_dir / f"entry_{index:03d}.csv"
                checked_entry = self.session_dir / f"entry_{index:03d}_checked.csv"
                write_csv(entry, entry_rows)
                self._command([
                    sys.executable, str(ROOT / "scripts/check_five_branch_plan_moveit.py"),
                    "--plan", str(entry), "--checked-output", str(checked_entry),
                    "--seed-joints-deg", *map(str, state["joints"]),
                    "--sphere-contact-safety", "--safe-transition-mode", "cartesian",
                ])
                entry_result = read_csv(checked_entry)
                if len(entry_result) != 1 or entry_result[0].get("moveit_status") != "OK":
                    raise RuntimeError(f"current-to-safe entry failed for {row['plan_id']}")
                for key in ("sphere_safety_model_sha256", "sphere_safety_world_sha256"):
                    if entry_result[0].get(key) != row.get(key):
                        raise RuntimeError("模型或障碍物在整批检查后发生变化；必须重新预检")
                if not same_state(state, self._check_live_state(feedback, dashboard)):
                    raise RuntimeError("robot state moved after entry preflight; recheck required")
                single = self.session_dir / f"execution_{index:03d}.csv"
                write_csv(single, [row])
                self.log(f"触测 {index}/{len(rows)}: {row['plan_id']}")
                dashboard.close()  # the existing executor owns Dashboard port 29999 during this contact
                feedback.close()   # and owns feedback port 30004 during this contact
                bundle = validate_bundle(row, policy)
                duration = sum(sum(max(abs(b-a) for a,b in zip(q0,q1)) * 180. / math.pi
                    for q0,q1 in zip(phase["joints_rad"], phase["joints_rad"][1:]))
                    for phase in bundle["phases"]) / policy["max_joint_speed_deg_sec"]
                self._command([
                    sys.executable, str(ROOT / "scripts/verified_sphere_executor.py"),
                    "--plan", str(single), "--output", str(output),
                    "--reference-fit-json", str(fit_path), "--ip", self.robot_ip,
                    "--installation-id", self.installation_id,
                    "--standard-sphere-id", self.standard_sphere_id, "--execute",
                ], timeout_sec=max(180., duration + 120.))
                feedback = FeedbackReader(self.robot_ip)
                feedback.start()
                feedback.wait_fresh(5.0)
                new_hits = hit_ids(output)
                if row["plan_id"] not in new_hits:
                    raise RuntimeError(f"{row['plan_id']} did not finish with HIT; stopped for review")
                finished = new_hits
            self.log(f"批次完成：{len(finished)}/{len(rows)} HIT，输出 {output}")
            return output
        except BaseException:
            try:
                dashboard.stop()
            except Exception:
                pass
            raise
        finally:
            dashboard.close()
            feedback.close()
