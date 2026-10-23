import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cross_probe_model import branch_local_offset_mm, load_geometry
from direct_tcp_execute_five_branch_plan import check_initial_robot_state
from sphere_experiment_core import (
    BRANCHES, append_jsonl, export_manual_contacts_csv, fit_anchor_center, fit_coarse_center,
    fit_manual_relative_seed, fit_new_installation_geometry, fit_relocated_center_z, fit_usable_for_motion,
    dataset_split_for_placement, validate_fit_dimensions, validate_fixture_fit_height,
    generate_placement_plans, read_csv, validate_short_plan, write_csv,
)
from sphere_experiment_runner import StageRunner, require_complete_stage, same_state
from sphere_simulation import (
    freeze_preview_plans, moveit_preflight_batch, moveit_preflight_one,
    scene_checks, scene_from_inputs, scene_from_profile,
)
from workbench_profile import (
    base_point_to_table, derived_fixture_base, derived_fixture_table, fixture_sphere_height_base_mm, load_profile,
    magnetic_base_center_from_sphere, profile_sha256, register_fixture_to_sphere,
    require_supported_contact_geometry, save_profile, table_point_to_base, validate_profile,
)
from five_branch_sphere_workbench import sanitize_moveit_runtime_environment, validate_placement_id


def synthetic_manual(center, perturb=None):
    geometry = load_geometry(ROOT / "config/cross_probe_geometry.yaml")
    normal_sets = [
        np.array([1.0, 0.0, 0.1]), np.array([-1.0, 0.0, 0.1]),
        np.array([0.0, 1.0, 0.1]), np.array([0.0, -1.0, 0.1]),
        np.array([0.2, 0.1, 1.0]), np.array([-0.2, 0.1, -1.0]),
    ]
    rows = []
    for branch in BRANCHES:
        offset = np.asarray(branch_local_offset_mm(geometry, branch))
        delta = np.asarray((perturb or {}).get(branch, [0, 0, 0]))
        for normal in normal_sets:
            normal = normal / np.linalg.norm(normal)
            pose = np.r_[np.asarray(center) + 11.0 * normal - offset - delta, [0.0, 0.0, 0.0]]
            rows.append({"kind": "manual_trigger", "branch": branch, "pose": pose.tolist()})
    return geometry, rows


def test_placement_id_is_unbounded_by_dropdown_but_safe_for_filenames():
    assert validate_placement_id("P27") == "P27"
    assert validate_placement_id("train_A01") == "train_A01"
    assert validate_placement_id(" locked-test-12 ") == "locked-test-12"
    with pytest.raises(ValueError, match="只能包含"):
        validate_placement_id("../P7")
    with pytest.raises(ValueError, match="不能为空"):
        validate_placement_id("  ")


def test_workbench_removes_only_stale_moveit_overlay(monkeypatch):
    workspace = "/tmp/custom-cr5-workspace"
    monkeypatch.setenv("CR5_WORKSPACE", workspace)
    stale = f"{workspace}/install/moveit_planners_ompl"
    monkeypatch.setenv("AMENT_PREFIX_PATH", f"{workspace}/install/cr5_moveit:{stale}:/opt/ros/humble")
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{stale}/lib:/opt/ros/humble/lib")
    sanitize_moveit_runtime_environment()
    assert stale not in __import__("os").environ["AMENT_PREFIX_PATH"].split(":")
    assert f"{stale}/lib" not in __import__("os").environ["LD_LIBRARY_PATH"].split(":")
    assert f"{workspace}/install/cr5_moveit" in __import__("os").environ["AMENT_PREFIX_PATH"]


def test_manual_save_exports_current_placement_without_changing_raw_data(tmp_path):
    source = tmp_path / "P0_manual.jsonl"
    contact = {
        "kind": "manual_trigger", "installation_id": "install-1",
        "standard_sphere_id": "sphere-1", "placement_id": "P0",
        "robot_ip": "192.168.5.1", "branch": "z", "pose": [1, 2, 3, 4, 5, 6],
        "joints": [10, 20, 30, 40, 50, 60], "di1": 1,
        "feed_sequence": 42, "wall_time": "2026-09-21T12:00:00",
    }
    append_jsonl(source, contact)
    original = source.read_bytes()
    output, count = export_manual_contacts_csv(tmp_path, "P0", "install-1", "sphere-1", "192.168.5.1")
    rows = read_csv(output)
    assert count == 1
    assert rows[0]["branch"] == "z"
    assert rows[0]["pose_x"] == "1.0"
    assert rows[0]["joint_6_deg"] == "60.0"
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="不一致"):
        export_manual_contacts_csv(tmp_path, "P0", "install-2", "sphere-1", "192.168.5.1")
    assert output.read_text()  # A rejected save must not erase the last good export.


def test_simulation_scene_derives_base_xy_and_detects_wrong_table_location():
    fit = {"sphere_center_mm": [-400.0, 120.0, 100.0], "sphere_radius_mm": 10.0}
    scene = scene_from_inputs(
        fit, table_z=0.0, table_xy=[-400.0, 120.0], table_size=[1200.0, 900.0],
        table_thickness=20.0, base_z=35.0, base_size=[60.0, 60.0, 70.0],
    )
    assert scene["base_center"] == [-400.0, 120.0, 35.0]
    assert scene_checks(scene)[0].startswith("几何自检通过")
    scene["table_center_xy"] = [500.0, 120.0]
    assert any("超出台面" in finding for finding in scene_checks(scene))


def test_profile_builds_complete_coaxial_scene_and_round_trips(tmp_path):
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    saved = save_profile(tmp_path / "profile.json", profile)
    assert profile_sha256(load_profile(saved)) == profile_sha256(profile)
    scene = scene_from_profile(profile, flange_pose=[1, 2, 3, 4, 5, 6])
    assert scene["sphere_center"][:2] == scene["base_center"][:2]
    assert scene["sphere_stem"]["center_mm"][:2] == scene["sphere_center"][:2]
    assert scene["tool"]["connector"]["length_mm"] == 49.4
    assert scene_checks(scene)[0].startswith("几何自检通过")


def test_base_relative_prefill_reconstructs_historical_fixture_coordinates():
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    fixture_base = derived_fixture_base(profile)
    fixture_table = derived_fixture_table(profile)
    assert fixture_base["base_center"] == pytest.approx([-401.9, 126.5, 35.0])
    assert fixture_base["sphere_center"] == pytest.approx([-401.9, 126.5, 89.487])
    assert fixture_table["sphere_center"] == pytest.approx([-1.9, 6.5, 89.487])
    assert table_point_to_base(profile, fixture_table["sphere_center"]) == pytest.approx(fixture_base["sphere_center"])


def test_fixture_height_comes_from_table_and_dimensions_not_fitted_base_z():
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    profile["workcell"]["magnetic_base"]["center_base_mm"][2] = -114.643
    assert fixture_sphere_height_base_mm(profile) == pytest.approx(89.487)
    fit = {"sphere_center_mm": [-738, 155, 89.487], "fixed_sphere_center_z_mm": 89.487}
    validate_fixture_fit_height(fit, profile)
    profile["robot"]["base_in_table_pose_mm_deg"][2] = 20.0
    assert fixture_sphere_height_base_mm(profile) == pytest.approx(69.487)
    with pytest.raises(ValueError, match="结构高度改变"):
        validate_fixture_fit_height(fit, profile)
    with pytest.raises(ValueError, match="球心必须固定"):
        validate_fixture_fit_height({"sphere_center_mm": [-738, 155, 69.487]}, profile)


def test_legacy_absolute_profile_is_migrated_to_base_relative_schema():
    legacy = {
        "schema_version": 1,
        "robot": {"ip": "192.168.5.1", "frame_id": "base_link", "euler_sequence": "xyz"},
        "workcell": {
            "table": {"center_xy_mm": [-400, 120], "top_z_mm": 0, "size_xyz_mm": [1200, 900, 20]},
            "magnetic_base": {"center_mm": [-401.9, 126.5, 35], "size_xyz_mm": [60, 60, 70]},
            "sphere_stem": {"center_mm": [-401.9, 126.5, 74.7435], "diameter_mm": 8, "length_mm": 9.487},
            "standard_sphere": {"center_mm": [-401.9, 126.5, 89.487], "diameter_mm": 20},
        },
        "tool": {
            "mount_xyz_mm": [0, 0, 0], "mount_rxyz_deg": [0, 0, 0],
            "connector": {"diameter_mm": 60, "length_mm": 49.4},
            "probe_body": {"diameter_mm": 63, "length_mm": 76},
            "center_stem": {"diameter_mm": 7, "length_mm": 49}, "ruby_ball_diameter_mm": 2,
        },
        "jog": {"linear_step_mm": .2, "angular_step_deg": .1, "speed": 1, "auto_retract_mm": 1,
                "timeout_sec": 5, "position_tolerance_mm": .08, "orientation_tolerance_deg": .08},
    }
    migrated = validate_profile(legacy)
    assert migrated["schema_version"] == 3
    assert migrated["robot"]["base_in_table_pose_mm_deg"] == pytest.approx([400, -120, 0, 0, 0, 0])
    assert migrated["workcell"]["magnetic_base"]["center_base_mm"] == pytest.approx([-401.9, 126.5, 35])


def test_schema_v2_table_relative_base_is_migrated_without_moving_fixture():
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    legacy_v2 = json.loads(json.dumps(profile))
    legacy_v2["schema_version"] = 2
    magnetic = legacy_v2["workcell"]["magnetic_base"]
    center_base = magnetic.pop("center_base_mm")
    magnetic["center_table_mm"] = base_point_to_table(profile, center_base)
    migrated = validate_profile(legacy_v2)
    assert migrated["schema_version"] == 3
    assert migrated["workcell"]["magnetic_base"]["center_base_mm"] == pytest.approx(center_base)


def test_fitted_sphere_registers_base_with_current_measured_dimensions():
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    profile["workcell"]["magnetic_base"]["size_xyz_mm"] = [80.0, 75.0, 50.0]
    profile["workcell"]["sphere_stem"]["length_mm"] = 15.0
    profile["workcell"]["standard_sphere"]["diameter_mm"] = 24.0
    fitted = [-360.0, 180.0, 140.0]
    center = magnetic_base_center_from_sphere(profile, fitted)
    assert center == pytest.approx([-360.0, 180.0, 88.0])
    registered = register_fixture_to_sphere(profile, fitted)
    assert registered["workcell"]["magnetic_base"]["size_xyz_mm"] == [80.0, 75.0, 50.0]
    assert derived_fixture_base(registered)["sphere_center"] == pytest.approx(fitted)


from test_sphere_contact_safety import stamped_row


def test_freeze_promotes_the_exact_preview_and_binds_profile(tmp_path):
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    fit = {
        "sphere_center_mm": [-401.9, 126.5, 89.5], "sphere_radius_mm": 10.0,
        "installation_id": "install-new", "standard_sphere_id": "sphere-1", "placement_id": "P0",
    }
    preview_row = {"plan_id": "P0-smoke-z-1", "branch": "z", "target_flange_x": "1.25",
                   "dataset_split": "train"}
    preview_row = stamped_row(**preview_row)
    checked = {preview_row["plan_id"]: "OK"}
    with pytest.raises(ValueError, match="MoveIt"):
        freeze_preview_plans(tmp_path, "P0", {"smoke": [preview_row]}, fit,
                             ROOT / "config/cross_probe_geometry.yaml", 0.0, profile, {})
    paths = freeze_preview_plans(
        tmp_path, "P0", {"smoke": [preview_row]}, fit,
        ROOT / "config/cross_probe_geometry.yaml", 0.0, profile, checked,
    )
    frozen = read_csv(paths["smoke"])[0]
    assert frozen["plan_id"] == preview_row["plan_id"]
    assert frozen["target_flange_x"] == preview_row["target_flange_x"]
    assert frozen["workbench_profile_sha256"] == profile_sha256(profile)
    manifest = json.loads((tmp_path / "P0_manifest.json").read_text())
    assert manifest["plans"]["smoke"]["source"] == "approved_preview"
    assert manifest["workbench_profile_sha256"] == profile_sha256(profile)
    assert manifest["dataset_split"] == "train"
    assert manifest["preview_moveit_checked_ids"] == [preview_row["plan_id"]]
    assert manifest["plans"]["smoke"]["workbench_profile_sha256"] == profile_sha256(profile)
    with pytest.raises(ValueError, match="不能覆盖"):
        freeze_preview_plans(
            tmp_path, "P0", {"smoke": [preview_row]}, fit,
            ROOT / "config/cross_probe_geometry.yaml", 0.0, profile, checked,
        )


def test_later_stage_freezes_new_registered_profile_without_changing_earlier_snapshot(tmp_path):
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    fit = {"sphere_center_mm": [-401.9, 126.5, 89.5], "installation_id": "new",
           "standard_sphere_id": "sphere", "placement_id": "P0"}
    geometry_path = ROOT / "config/cross_probe_geometry.yaml"
    smoke = stamped_row(plan_id="P0_smoke_z_01", dataset_split="train")
    freeze_preview_plans(tmp_path, "P0", {"smoke": [smoke]}, fit, geometry_path,
                         0.0, profile, {smoke["plan_id"]: "OK"})
    updated = json.loads(json.dumps(profile))
    updated["workcell"]["magnetic_base"]["center_base_mm"][0] += 0.1
    refined = {**fit, "sphere_center_mm": [-401.8, 126.5, 89.5]}
    anchor = stamped_row(plan_id="P0_anchor_z_01", dataset_split="train")
    freeze_preview_plans(tmp_path, "P0", {"anchor": [anchor]}, refined, geometry_path,
                         0.0, updated, {anchor["plan_id"]: "OK"})
    manifest = json.loads((tmp_path / "P0_manifest.json").read_text())
    assert manifest["plans"]["smoke"]["workbench_profile_sha256"] == profile_sha256(profile)
    assert manifest["plans"]["anchor"]["workbench_profile_sha256"] == profile_sha256(updated)
    assert profile_sha256(load_profile(tmp_path / "P0_smoke_workbench_profile.json")) == profile_sha256(profile)


def test_workbench_rejects_stale_radii_and_unsupported_mount():
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    fit = {"sphere_radius_mm": 10.0, "probe_radius_mm": 1.0,
           "contact_center_distance_mm": 11.0}
    validate_fit_dimensions(fit, profile)
    profile["workcell"]["standard_sphere"]["diameter_mm"] = 24.0
    with pytest.raises(ValueError, match="重新拟合"):
        validate_fit_dimensions(fit, profile)
    profile["workcell"]["standard_sphere"]["diameter_mm"] = 20.0
    profile["tool"]["mount_xyz_mm"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="禁止粗拟合"):
        require_supported_contact_geometry(profile)
    profile["tool"]["mount_xyz_mm"] = [0.0, 0.0, 0.0]
    profile["tool"]["connector"]["length_mm"] += 1.0
    with pytest.raises(ValueError, match="法兰连接件长度"):
        require_supported_contact_geometry(profile)


def test_nonfinite_profile_or_coarse_fit_cannot_enter_motion():
    profile = load_profile(ROOT / "config/five_branch_workbench_profile.json")
    profile["tool"]["connector"]["length_mm"] = math.nan
    with pytest.raises(ValueError, match="法兰连接件"):
        validate_profile(profile)
    fit = {"sphere_center_mm": [0, 0, 0], "manual_contact_count": 10,
           "branch_counts": {branch: 2 for branch in BRANCHES},
           "normal_condition": math.nan, "max_abs_radial_residual_mm": 0.0}
    assert not fit_usable_for_motion(fit)[0]


def test_relocated_fixture_three_z_faces_fit_xy_with_existing_height():
    center = np.array([-710.0, 120.0, 89.487])
    offset = [0.0, 0.0, 209.27089775384175]
    normals = [np.array([0.8, 0.0, 0.6]), np.array([-0.4, 0.75, 0.527]),
               np.array([-0.4, -0.75, 0.527])]
    rows = []
    for normal in normals:
        normal = normal / np.linalg.norm(normal)
        ball = center + 11.0 * normal
        rows.append({"kind": "manual_trigger", "branch": "z",
                     "pose": [*(ball - offset), 0.0, 0.0, 0.0]})
    fit = fit_relocated_center_z(rows, offset, center[2])
    assert fit["sphere_center_mm"] == pytest.approx(center, abs=1e-6)
    assert fit_usable_for_motion(fit)[0]
    side_anchor = {"kind": "manual_trigger", "branch": "x_pos", "pose": [0.0] * 6}
    assert fit_relocated_center_z(rows + [side_anchor], offset, center[2])["sphere_center_mm"] == pytest.approx(center)
    with pytest.raises(ValueError, match="不同球面位置|触点过于接近"):
        fit_relocated_center_z([rows[0]] * 3, offset, center[2])
    bad = json.loads(json.dumps(rows))
    bad[2]["pose"][2] += 4.0
    with pytest.raises(ValueError, match="固定球高|触点覆盖"):
        fit_relocated_center_z(bad, offset, center[2])


def test_new_placement_requires_explicit_dataset_role():
    assert dataset_split_for_placement("P4") == "validation"
    assert dataset_split_for_placement("train_A01", "train") == "train"
    with pytest.raises(ValueError, match="预先指定"):
        dataset_split_for_placement("P27")
    with pytest.raises(ValueError, match="P4"):
        dataset_split_for_placement("P4", "train")


def test_stage_completion_requires_exact_plan_ids(tmp_path):
    plan = tmp_path / "P0_smoke_plan.csv"
    row = {"plan_id": "expected", "probe_collision_status": "OK", "standoff_mm": "0.5",
           "overtravel_mm": "0.3"}
    for prefix in ("safe_flange", "transition_flange", "start_flange", "target_flange"):
        row.update({f"{prefix}_{name}": "0" for name in ("x", "y", "z", "rx", "ry", "rz")})
    write_csv(plan, [row])
    contacts = tmp_path / "P0_smoke_contacts.csv"
    write_csv(contacts, [{"plan_id": "foreign", "status": "HIT"}])
    with pytest.raises(ValueError, match="计划ID"):
        require_complete_stage(plan, contacts, "smoke")
    write_csv(contacts, [{"plan_id": "expected", "status": "HIT"}])
    require_complete_stage(plan, contacts, "smoke")


def test_moveit_preview_only_calls_scene_and_planning_services(monkeypatch):
    fit = {"sphere_center_mm": [-400.0, 120.0, 100.0], "sphere_radius_mm": 10.0}
    scene = scene_from_inputs(
        fit, table_z=0.0, table_xy=[-400.0, 120.0], table_size=[1200.0, 900.0],
        table_thickness=20.0, base_z=35.0, base_size=[60.0, 60.0, 70.0],
    )
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if "check_five_branch_plan_moveit.py" in command[1]:
            checked = Path(command[command.index("--checked-output") + 1])
            write_csv(checked, [stamped_row(plan_id="preview-1")])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sphere_simulation.subprocess.run", fake_run)
    result = moveit_preflight_one(scene, fit, stamped_row(plan_id="preview-1"))
    assert result["moveit_status"] == "OK"
    assert len(commands) == 2
    assert "apply_calibration_moveit_scene.py" in commands[0][1]
    assert "check_five_branch_plan_moveit.py" in commands[1][1]
    assert "--publish-display-trajectory" in commands[1]
    assert "--sphere-contact-safety" in commands[1]
    assert "--seed-joints-deg" not in commands[1]
    assert "direct_tcp_execute" not in str(commands)


def test_batch_moveit_gate_rejects_one_failed_row(monkeypatch):
    scene = scene_from_inputs(
        {"sphere_center_mm": [0, 0, 100], "sphere_radius_mm": 10},
        table_z=0, table_xy=[0, 0], table_size=[200, 200],
        table_thickness=20, base_z=35, base_size=[60, 60, 70],
    )
    plans = {"smoke": [{"plan_id": "first"}, {"plan_id": "second"}]}

    def fake_run(command, **_kwargs):
        if "check_five_branch_plan_moveit.py" in command[1]:
            checked = Path(command[command.index("--checked-output") + 1])
            write_csv(checked, [{"plan_id": "first", "moveit_status": "OK"},
                                {"plan_id": "second", "moveit_status": "FAIL"}])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sphere_simulation.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="second"):
        moveit_preflight_batch(scene, plans, [0.0] * 6)


def test_coarse_center_and_motion_gate():
    center = np.array([-400.0, 120.0, 100.0])
    geometry, rows = synthetic_manual(center)
    fit = fit_coarse_center(rows, geometry)
    assert np.linalg.norm(np.asarray(fit["sphere_center_mm"]) - center) < 1e-5
    assert fit["rms_radial_residual_mm"] < 1e-5
    assert fit_usable_for_motion(fit)[0]
    with pytest.raises(ValueError, match="each of the five"):
        fit_coarse_center(rows[:-6], geometry)


def test_p0_coarse_fit_keeps_known_height_and_exposes_bad_vertical_offset():
    center = np.array([-738.0, 155.0, 89.487])
    geometry, rows = synthetic_manual(center)
    fit = fit_coarse_center(rows, geometry, fixed_sphere_center_z_mm=center[2])
    assert fit["sphere_center_mm"] == pytest.approx(center)
    assert fit["fixed_sphere_center_z_mm"] == center[2]
    assert fit_usable_for_motion(fit)[0]
    wrong_frame = json.loads(json.dumps(rows))
    for row in wrong_frame:
        row["pose"][2] -= 150.0
    bad = fit_coarse_center(wrong_frame, geometry, fixed_sphere_center_z_mm=center[2])
    assert bad["sphere_center_mm"][2] == center[2]
    assert not fit_usable_for_motion(bad)[0]
    assert bad["max_abs_radial_residual_mm"] > 100


def test_geometry_fit_uses_independent_current_contacts(tmp_path):
    center = np.array([-400.0, 120.0, 100.0])
    perturb = {"x_pos": [0.1, -0.1, 0.05], "x_neg": [-0.08, 0.03, 0.02],
               "y_pos": [0.02, 0.09, -0.04], "y_neg": [-0.05, -0.06, 0.01]}
    geometry, manual = synthetic_manual(center, perturb)
    # At least 12 reference and 9 side trigger rows, with diverse known normals.
    contacts = []
    for branch in BRANCHES:
        source = [row for row in manual if row["branch"] == branch]
        for index in range(12 if branch == "z" else 9):
            item = source[index % len(source)]
            pose = item["pose"]
            contacts.append({"status": "HIT", "branch": branch,
                             **{f"trigger_flange_{name}": pose[i] for i, name in enumerate(("x", "y", "z", "rx", "ry", "rz"))}})
    path = tmp_path / "contacts.csv"
    write_csv(path, contacts)
    result = fit_new_installation_geometry(path, ROOT / "config/cross_probe_geometry.yaml", 11.0,
                                           tmp_path / "fit.json")
    assert np.linalg.norm(np.asarray(result["sphere_center_mm"]) - center) < 1e-3
    for branch, delta in perturb.items():
        expected = np.asarray(branch_local_offset_mm(geometry, branch)) + delta
        assert np.linalg.norm(np.asarray(result["local_offsets_mm"][branch]) - expected) < 1e-3
    assert json.loads((tmp_path / "fit.json").read_text())["method"] == "relative_five_branch_fixed_z_gauge"
    fixed = fit_new_installation_geometry(path, ROOT / "config/cross_probe_geometry.yaml", 11.0,
                                          tmp_path / "fixed_fit.json", fixed_sphere_center_z_mm=center[2])
    assert fixed["sphere_center_mm"] == pytest.approx(center, abs=1e-3)
    assert fixed["fixed_sphere_center_z_mm"] == center[2]


def test_manual_joint_seed_recovers_reinstalled_side_offsets():
    center = np.array([-400.0, 120.0, 100.0])
    perturb = {"x_pos": [0.15, -0.08, 0.05], "x_neg": [-0.08, 0.06, 0.02],
               "y_pos": [0.02, 0.1, -0.04], "y_neg": [-0.06, -0.05, 0.03]}
    geometry, rows = synthetic_manual(center, perturb)
    seed = fit_manual_relative_seed(rows, geometry, 11.0)
    assert seed["rank"] == 15
    assert np.linalg.norm(np.asarray(seed["sphere_center_mm"]) - center) < 1e-3
    for branch, delta in perturb.items():
        target = np.asarray(branch_local_offset_mm(geometry, branch)) + delta
        assert np.linalg.norm(np.asarray(seed["local_offsets_mm"][branch]) - target) < 1e-3


def test_anchor_sphere_fit_uses_only_a_rows(tmp_path):
    center = np.array([-400.0, 120.0, 100.0])
    geometry, manual = synthetic_manual(center)
    z_rows = [row for row in manual if row["branch"] == "z"]
    contacts = []
    for index in range(8):
        pose = z_rows[index % len(z_rows)]["pose"]
        contacts.append({"status": "HIT", "branch": "z", **{
            f"trigger_flange_{name}": pose[i]
            for i, name in enumerate(("x", "y", "z", "rx", "ry", "rz"))
        }})
    source = tmp_path / "anchor.csv"
    write_csv(source, contacts)
    result = fit_anchor_center(source, branch_local_offset_mm(geometry, "z"), 11.0,
                               tmp_path / "anchor_fit.json")
    assert np.linalg.norm(np.asarray(result["sphere_center_mm"]) - center) < 1e-4
    assert result["source_contacts"] == str(source)
    fixed = fit_anchor_center(source, branch_local_offset_mm(geometry, "z"), 11.0,
                              tmp_path / "fixed_anchor_fit.json", fixed_sphere_center_z_mm=center[2])
    assert fixed["sphere_center_mm"] == pytest.approx(center, abs=1e-4)
    assert fixed["fixed_sphere_center_z_mm"] == center[2]


def test_plan_and_state_gates_fail_closed(tmp_path):
    row = {"plan_id": "x_pos_1", "probe_collision_status": "OK", "standoff_mm": "0.5",
           "overtravel_mm": "0.3"}
    for prefix in ("safe_flange", "transition_flange", "start_flange", "target_flange"):
        row.update({f"{prefix}_{name}": str(index) for index, name in enumerate(("x", "y", "z", "rx", "ry", "rz"))})
    plan = tmp_path / "plan.csv"
    write_csv(plan, [row])
    assert len(validate_short_plan(plan)) == 1
    row["overtravel_mm"] = "0.6"
    write_csv(plan, [row])
    with pytest.raises(ValueError, match="exceeds limit"):
        validate_short_plan(plan)
    first = {"pose": [0.0] * 6, "joints": [0.0] * 6}
    moved = {"pose": [0.2, 0, 0, 0, 0, 0], "joints": [0.0] * 6}
    assert not same_state(first, moved)


def test_live_gate_rejects_trigger_and_frame_mismatch(tmp_path):
    runner = StageRunner(
        session_dir=tmp_path, placement="P0", stage="smoke", robot_ip="127.0.0.1",
        installation_id="current", geometry_path=ROOT / "config/cross_probe_geometry.yaml",
        table_z_mm=0, magnetic_base_center=[0, 0, 0], magnetic_base_size=[1, 1, 1],
        frame_confirmed=True,
    )

    class Feedback:
        state = {"di1": True, "pose": [0.0] * 6, "joints": [0.0] * 6}
        def snapshot(self, age):
            return self.state

    class DashboardFake:
        def command(self, text, timeout):
            return {"RobotMode()": "0,{5};", "DI(1)": "0,{0};",
                    "GetPose(0,0)": "0,{1,0,0,0,0,0};"}[text]

    feedback = Feedback()
    with pytest.raises(RuntimeError, match="DI1=1"):
        runner._check_live_state(feedback, DashboardFake())
    feedback.state = {**feedback.state, "di1": False}
    with pytest.raises(RuntimeError, match="does not agree"):
        runner._check_live_state(feedback, DashboardFake())


def test_executor_rejects_state_changed_after_entry_preflight():
    class Feedback:
        def wait_fresh(self, timeout):
            return {"di1": False, "pose": [0.3, 0, 0, 0, 0, 0], "joints": [0] * 6}

    args = SimpleNamespace(
        feedback_start_timeout_sec=1.0, expected_entry_pose=[0] * 6,
        expected_entry_joints=[0] * 6, max_entry_pos_mm=0.15,
        max_entry_orientation_deg=0.2, max_entry_joint_deg=0.2,
    )
    with pytest.raises(RuntimeError, match="moved after current-state entry preflight"):
        check_initial_robot_state(None, Feedback(), args)


def test_existing_candidate_planner_generates_disjoint_batches(tmp_path):
    geometry = load_geometry(ROOT / "config/cross_probe_geometry.yaml")
    center = np.array([500.0, 0.0, 300.0])
    contacts = []
    safe = {}
    for branch in BRANCHES:
        branch_config = next(item for item in geometry["branches"] if item["name"] == branch)
        direction = np.asarray(branch_config["direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        offset = np.asarray(branch_local_offset_mm(geometry, branch))
        for index in range(2):
            normal = direction if index == 0 else (direction + np.array([0.02, -0.01, 0.02]))
            normal /= np.linalg.norm(normal)
            position = center - 11.0 * normal - offset
            contacts.append({"kind": "manual_trigger", "branch": branch,
                             "pose": [*position.tolist(), 0, 0, 0]})
        safe[branch] = [*(position - 20.0 * normal).tolist(), 0, 0, 0]
    fit = fit_coarse_center(contacts, geometry)
    assert fit_usable_for_motion(fit)[0]
    offsets = {branch: branch_local_offset_mm(geometry, branch) for branch in BRANCHES}
    paths = generate_placement_plans(tmp_path, "P0", contacts, safe, fit,
                                     ROOT / "config/cross_probe_geometry.yaml", 0.0, offsets,
                                     stages_to_write=("smoke", "geometry", "anchor", "evaluation"))
    assert {stage: len(validate_short_plan(path)) for stage, path in paths.items()} == {
        "smoke": 5, "geometry": 48, "anchor": 8, "evaluation": 90,
    }
    plans = {stage: list(__import__("csv").DictReader(path.open())) for stage, path in paths.items()}
    source_ids = {stage: {row["source_plan_id"] for row in rows} for stage, rows in plans.items()}
    assert not source_ids["anchor"] & source_ids["evaluation"]
    assert not source_ids["geometry"] & source_ids["evaluation"]
