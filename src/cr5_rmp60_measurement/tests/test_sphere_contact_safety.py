"""Regression tests for the missed middle-of-search collision and contact policy."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xacro
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sphere_contact_safety import (BALL_LINKS, BRANCHES, PROOF_FLAGS, SAFETY_VERSION,
    Box, Capsule, ToolModel, certify_sweep, certify_tool_motion, clearance,
    path_fingerprint, validate_probe, validate_safety_receipt)


def stamped_row(**updates):
    """Synthetic service receipt for persistence/transport unit tests only."""
    row = {'plan_id': 'synthetic', 'branch': 'z', 'contact_center_distance_mm': '11', 'dataset_split': 'train'}
    for prefix in ('safe', 'transition', 'start', 'target'):
        row.update({f'{prefix}_flange_{axis}': '0' for axis in ('x','y','z','rx','ry','rz')})
    for prefix in ('initial_offset', 'approach', 'sphere_center'):
        row.update({f'{prefix}_{axis}': '0' for axis in 'xyz'})
    row.update(updates)
    row.update({key:'1' for key in PROOF_FLAGS})
    row.update(sphere_safety_version=SAFETY_VERSION, moveit_status='OK',
               sphere_safety_model_sha256='a'*64, sphere_safety_world_sha256='b'*64)
    from sphere_motion_contract import make_bundle, load_policy, digest
    import hashlib, json
    policy = load_policy()
    row['sphere_motion_policy_sha256'] = digest(policy)
    raw = json.dumps(make_bundle(row, [{'name': name, 'joints_rad': [[0.] * 6], 'active_ball': None}
        for name in ('entry', 'transition', 'approach', 'probe', 'retract', 'return')],
        row['sphere_safety_model_sha256'], row['sphere_safety_world_sha256'], policy), separators=(',', ':'))
    row['verified_trajectory_json'] = raw
    row['verified_trajectory_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
    row['sphere_safety_path_sha256'] = path_fingerprint(row)
    return row


@pytest.fixture
def tool():
    source = f'''<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
      <link name="Link6"/>
      <xacro:include filename="{ROOT / 'urdf/rmp60_probe_module.xacro'}"/>
      <xacro:rmp60_probe_module parent="Link6"/>
    </robot>'''
    document = xacro.parse(source)
    xacro.process_doc(document)
    return ToolModel(document.toxml())


def test_all_five_balls_and_rods_exist_as_separate_collision_parts(tool):
    assert set(tool.balls) == set(BRANCHES)
    assert len(tool.primitives) == 13  # adapter, body, two axial cylinders, four rods, five balls
    for branch in BRANCHES:
        assert tool.balls[branch].name == BALL_LINKS[branch]
        assert tool.balls[branch].radius == pytest.approx(1.)
    with pytest.raises(ValueError, match='separate sphere-only'):
        ToolModel('<robot><link name="Link6"/></robot>')


@pytest.mark.parametrize('part_index', range(13))
def test_each_tool_part_is_included_in_continuous_motion(tool, part_index):
    part = tool.primitives[part_index]
    middle = (part.a + part.b)/2
    obstacle = Capsule('temporary_obstacle', middle, middle, .1)
    with pytest.raises(ValueError, match='continuous sweep'):
        certify_tool_motion(tool, np.zeros(6), np.array([0,0,.8,0,0,0]), [obstacle])


def test_grazing_collision_missed_by_both_endpoints_is_rejected():
    ruby = Capsule('inactive_ruby', np.zeros(3), np.zeros(3), 1.)
    obstacle = Capsule('grazing_obstacle', np.array([1.19,0,.4]), np.array([1.19,0,.4]), .2)
    assert clearance(ruby, obstacle) > .05
    assert clearance(ruby.moved(np.eye(4), np.array([0,0,.8])), obstacle) > .05
    with pytest.raises(ValueError, match='inactive_ruby / grazing_obstacle'):
        certify_sweep(ruby, np.eye(4), np.array([0,0,.8]), [obstacle])


def test_rotating_tool_sweep_checks_middle_of_arc():
    part = Capsule('side_ball', np.array([10.,0,0]), np.array([10.,0,0]), 1.)
    center = np.array([0.,10.,0])
    obstacle = Capsule('obstacle', center, center, 1.)
    end = Rotation.from_euler('z',180,degrees=True).as_matrix()
    with pytest.raises(ValueError, match='side_ball / obstacle'):
        certify_sweep(part, np.eye(4), np.zeros(3), [obstacle], end_rotation=end)


@pytest.mark.parametrize('obstacle', [
    Box('base',np.array([1.15,0,.4]),np.eye(3),np.array([.2,.2,.1])),
    Capsule('stem',np.array([1.15,0,.35]),np.array([1.15,0,.45]),.2)])
def test_box_and_cylinder_sweeps(obstacle):
    part = Capsule('rod', np.zeros(3),np.zeros(3),1.)
    with pytest.raises(ValueError, match='continuous sweep'):
        certify_sweep(part,np.eye(4),np.array([0,0,.8]),[obstacle])


def probe_case():
    balls = {b: Capsule(BALL_LINKS[b], np.array([i*30.,0,0]),np.array([i*30.,0,0]),1.)
             for i,b in enumerate(BRANCHES)}
    model=SimpleNamespace(balls=balls,primitives=list(balls.values()))
    row=stamped_row()
    row.update(start_flange_z='11.5', target_flange_z='10.7', approach_z='-1')
    return row,model,[Capsule('calibration_sphere', np.zeros(3),np.zeros(3),10.)]


def test_correct_first_contact_and_wrong_branch_are_distinguished():
    row,model,world=probe_case()
    assert validate_probe(row,model,world,np.zeros(3),10.)['predicted_first_contact_branch']=='z'
    row['branch']='x_pos'
    row['initial_offset_x']='30'
    with pytest.raises(ValueError,match='first contact is z'):
        validate_probe(row,model,world,np.zeros(3),10.)


def test_active_rod_cannot_use_ruby_contact_exception():
    row,model,world=probe_case()
    rod=Capsule('rmp60_stylus', np.array([0,0,-.9]),np.array([0,0,1.]),.8)
    model.primitives.append(rod)
    with pytest.raises(ValueError,match='rmp60_stylus / calibration_sphere'):
        validate_probe(row,model,world,np.zeros(3),10.)


def test_insufficient_first_contact_margin_rejected():
    row,model,world=probe_case()
    other=Capsule(BALL_LINKS['x_pos'],np.array([0.,0,.1]),np.array([0.,0,.1]),1.)
    model.balls['x_pos']=other
    with pytest.raises(ValueError,match='margin'):
        validate_probe(row,model,world,np.zeros(3),10.)


@pytest.mark.parametrize('field,value', [('initial_offset_x','nan'),('approach_x','nan'),
    ('sphere_center_x','nan'),('contact_center_distance_mm','nan'),('initial_offset_x','1'),
    ('sphere_center_x','1'),('target_flange_rz','1')])
def test_inconsistent_or_nonfinite_plan_rejected(field,value):
    row,model,world=probe_case(); row[field]=value
    with pytest.raises(ValueError):
        validate_probe(row,model,world,np.zeros(3),10.)


def test_receipt_invalidated_by_path_edit_and_old_receipt():
    row=stamped_row(plan_id='one')
    validate_safety_receipt(row)
    row['target_flange_x']='.1'
    with pytest.raises(ValueError,match='路径不一致'):
        validate_safety_receipt(row)
    row=stamped_row(plan_id='one'); del row['moveit_return_path_checked']
    with pytest.raises(ValueError,match='重新预检'):
        validate_safety_receipt(row)


def test_strict_acm_clears_broad_allowance_and_exempts_only_active_ruby():
    from moveit_msgs.msg import AllowedCollisionMatrix,AllowedCollisionEntry
    from moveit_sphere_contact_safety import strict_acm
    links=['rmp60_ball_z','rmp60_stylus','rmp60_ball_x_pos','Link1']
    world={'calibration_sphere','calibration_table'}
    original=AllowedCollisionMatrix(default_entry_names=['calibration_sphere'],default_entry_values=[True])
    names=[*links,*world]
    original.entry_names=names
    original.entry_values=[AllowedCollisionEntry(enabled=[True]*len(names)) for _ in names]
    policy=strict_acm(original,links,world,'rmp60_ball_z')
    for a in links:
        for b in world:
            assert policy.entry_values[policy.entry_names.index(a)].enabled[policy.entry_names.index(b)] == (a=='rmp60_ball_z' and b=='calibration_sphere')
    assert all(all(entry.enabled) for entry in original.entry_values)


def test_world_fingerprint_ignores_timestamps_but_detects_geometry_change():
    from moveit_msgs.msg import PlanningSceneWorld,CollisionObject
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    from moveit_sphere_contact_safety import SphereContactPreflight
    world=PlanningSceneWorld()
    obj=CollisionObject(id='calibration_sphere')
    obj.header.frame_id='base_link'
    obj.primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE,dimensions=[.01])]
    obj.primitive_poses=[Pose()]
    world.collision_objects=[obj]
    before=SphereContactPreflight.fingerprint(world)
    world.collision_objects[0].header.stamp.sec=123
    world.octomap.header.stamp.sec=456
    assert SphereContactPreflight.fingerprint(world)==before
    world.collision_objects[0].primitive_poses[0].position.x+=.0001
    assert SphereContactPreflight.fingerprint(world)!=before


def test_legacy_executor_cannot_execute_verified_joint_sphere_paths():
    from direct_tcp_execute_five_branch_plan import validate_sphere_motion_mode
    args=SimpleNamespace(execute=True)
    items=[{'row':stamped_row(sphere_safety_required='1')}]
    with pytest.raises(ValueError, match='verified_sphere_executor'):
        validate_sphere_motion_mode(items,args)


def test_failed_and_successful_preflight_rows_can_both_be_written(tmp_path):
    from check_five_branch_plan_moveit import write_csv
    import csv
    path=tmp_path/'mixed.csv'
    write_csv(path,[{'plan_id':'ok','moveit_status':'OK','sphere_safety_version':SAFETY_VERSION},
                    {'plan_id':'bad','moveit_status':'FAIL','moveit_error':'grazing collision'}])
    with path.open() as handle:rows=list(csv.DictReader(handle))
    assert rows[1]['moveit_error']=='grazing collision'


def test_freeze_rejects_stale_path_even_when_status_map_says_ok(tmp_path):
    from sphere_simulation import freeze_preview_plans
    from workbench_profile import load_profile
    row=stamped_row(plan_id='one')
    row['target_flange_x']='1'
    with pytest.raises(ValueError,match='路径不一致'):
        freeze_preview_plans(tmp_path,'P0',{'smoke':[row]}, {},
            ROOT/'config/cross_probe_geometry.yaml',0.,
            load_profile(ROOT/'config/five_branch_workbench_profile.json'),{'one':'OK'})
    assert not (tmp_path/'P0_smoke_plan.csv').exists()
