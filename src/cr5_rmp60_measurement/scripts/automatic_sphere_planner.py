#!/usr/bin/env python3
"""Generate five-branch cone sampling from a sphere position, with no taught poses.

Requires a running planning-only MoveIt instance. Uses live tool geometry,
geometry checks and collision-free IK to search branch orientations/safe poses.
No robot command, action-execution or Dashboard client is constructed.
"""
from __future__ import annotations
import argparse,copy,json,math,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from check_five_branch_plan_moveit import PlanMoveItChecker,make_seed_state
from moveit_sphere_contact_safety import SphereContactPreflight
from model_based_sphere_point_planner import cone_normals,rotation_for_normal
from sphere_contact_safety import BRANCHES,pose_transform,validate_probe
from sphere_experiment_core import select_spread,write_csv,dataset_split_for_placement

AXES=('x','y','z','rx','ry','rz')


def contact_row(center,radius,branch,offset,ball_radius,rotation,approach,*,safe_retreat_mm=40.):
    center=np.asarray(center,dtype=float);offset=np.asarray(offset,dtype=float)
    approach=np.asarray(approach,dtype=float);approach=approach/np.linalg.norm(approach)
    angles=Rotation.from_matrix(rotation).as_euler('xyz',degrees=True)
    distance=radius+ball_radius
    poses={name:np.r_[center-approach*(distance+gap)-rotation@offset,angles]
           for name,gap in [('safe',6.+safe_retreat_mm),('transition',6.),('start',.5),('contact',0.),('target',-.3)]}
    row={'branch':branch,'contact_center_distance_mm':f'{distance:.12f}',
         'standoff_mm':'.5','overtravel_mm':'.3','transition_standoff_mm':'6',
         'probe_collision_status':'OK','sphere_safety_required':'1','euler_sequence':'xyz',
         'candidate_source':'automatic_fixed_orientation_cone','safe_exit_mode':'radial_outward'}
    for prefix,pose in poses.items():row.update({f'{prefix}_flange_{axis}':f'{v:.9f}' for axis,v in zip(AXES,pose)})
    for prefix,values in [('sphere_center',center),('initial_offset',offset),('approach',approach)]:
        row.update({f'{prefix}_{axis}':f'{v:.12f}' for axis,v in zip('xyz',values)})
    return row


def branch_orientations(branch,model,current_rotation,manual):
    hubs=[part.a for part in model.primitives if part.name.startswith("rmp60_branch_")]
    if len(hubs)!=4 or max(np.linalg.norm(point-hubs[0]) for point in hubs)>.01:
        raise ValueError("loaded five-branch model has inconsistent hub origins")
    hub=np.mean(hubs,axis=0)
    direction=model.balls[branch].a-hub
    if np.linalg.norm(direction)<1e-6:raise ValueError('invalid branch direction')
    direction/=np.linalg.norm(direction)
    seen=set()
    candidates=[]
    for row in reversed(manual):
        if row.get('branch')==branch and len(row.get('pose',[]))==6:
            candidates.append(pose_transform(row['pose'])[:3,:3])
    candidates.append(current_rotation)
    normals=[np.array([0.,0.,-1.])]
    for angle in range(0,360,45):
        normals.append(np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle)),-.25]))
    for normal in normals:
        normal=normal/np.linalg.norm(normal)
        for roll in (0,90,180,270,45,135,225,315):
            candidates.append(np.array(rotation_for_normal(direction.tolist(),normal.tolist(),roll_deg=roll)))
    for rotation in candidates:
        key=tuple(np.round(rotation.ravel(),7))
        if key in seen:continue
        seen.add(key)
        approach=rotation@direction
        if approach[2]>.1:continue # No approach through the magnetic fixture.
        yield rotation,approach


def group_candidates(chosen,placement,fit,stages,dataset_split):
    plans={stage:[] for stage in stages}
    for branch,rows in chosen.items():
        assignments=[('smoke',rows[:1],1),('geometry',rows[1:13] if branch=='z' else rows[1:10],1),
                     ('anchor',rows[13:21] if branch=='z' else [],1),
                     ('evaluation',rows[21:27] if branch=='z' else rows[10:16],3)]
        for stage,sources,repeats in assignments:
            if stage not in plans:continue
            for index,source in enumerate(sources,1):
                for repeat in range(1,repeats+1):
                    row=dict(source)
                    row.update(plan_id=f'{placement}_{stage}_{branch}_{index:02d}_r{repeat}',
                        standard_placement_id=placement,standard_sphere_id=fit.get('standard_sphere_id',''),
                        physical_ball_id=f"{fit.get('installation_id','unknown')}:{branch}",
                        dataset_group=stage,dataset_branch_point_index=str(index),dataset_repeat_index=str(repeat),
                        dataset_split=dataset_split,planning_mode='automatic_pose_search')
                    plans[stage].append(row)
    return plans


def generate(checker,fit,placement,stages,manual=(),dataset_split=None,search_timeout_sec=300.):
    if np.linalg.norm(np.asarray(fit['sphere_center_mm'])-checker.sphere_center)>.001:
        raise ValueError('输入球位与MoveIt当前夹具场景不一致')
    split=dataset_split_for_placement(placement,dataset_split)
    if "probe_radius_mm" in fit and any(abs(float(fit["probe_radius_mm"])-part.radius)>1e-6 for part in checker.model.balls.values()):
        raise ValueError("输入测球半径与当前MoveIt模型不一致")
    if abs(float(fit['sphere_radius_mm'])-checker.sphere_radius)>1e-6:
        raise ValueError('输入球径与当前场景不一致')
    if not math.isfinite(search_timeout_sec) or search_timeout_sec<=0:raise ValueError("invalid automatic search time budget")
    deadline=time.monotonic()+search_timeout_sec
    seed=make_seed_state(checker.node.joint_state)
    rotation=pose_transform(checker.flange_pose(seed))[:3,:3]
    chosen={};failures=[]
    for branch in BRANCHES:
        needed=27 if branch=='z' else 16
        offset=checker.model.balls[branch].a
        for orientation,axis in branch_orientations(branch,checker.model,rotation,manual):
            if time.monotonic()>deadline:raise ValueError(f"{branch}: 自动搜索达到时间预算，尚未证明完整采集路线")
            valid=[];local_seed=copy.deepcopy(seed)
            normals=cone_normals(axis.tolist(),12.,4,12)
            for index,normal in enumerate(normals):
                row=contact_row(checker.sphere_center,checker.sphere_radius,branch,offset,
                    checker.model.balls[branch].radius,orientation,normal)
                row['plan_id']=f'candidate_{branch}_{index}'
                try:
                    validate_probe(row,checker.model,checker.obstacles,checker.sphere_center,checker.sphere_radius)
                    start=[float(row[f'start_flange_{k}']) for k in AXES]
                    ok,code,state=checker.node.solve_ik(start,local_seed,checker.args,True)
                    if not ok:raise ValueError(f'start IK {code}')
                    q=[state.joint_state.position[state.joint_state.name.index(j)] for j in checker.whole.joint_names]
                    checker.whole.certify([q])
                    for pose_name in ('safe', 'transition'):
                        target=[float(row[f'{pose_name}_flange_{k}']) for k in AXES]
                        ok,code,goal_state=checker.node.solve_ik(target,state,checker.args,True)
                        if not ok:raise ValueError(f'{pose_name} IK {code}')
                        goal_q=[goal_state.joint_state.position[goal_state.joint_state.name.index(j)] for j in checker.whole.joint_names]
                        checker.whole.certify([goal_q])
                        row[f'{pose_name}_joint_seed_json']=json.dumps(goal_q,separators=(',',':'))
                    if index==0:
                        # A reachable pose alone is insufficient: prove its full
                        # entry/probe/retraction route before sampling this cone.
                        checker.check_row(row,seed)
                    local_seed=state;valid.append(row)
                except (ValueError,RuntimeError) as exc:
                    if len(failures)<25:failures.append(f'{branch}: {exc}')
                    if index==0:break # Search another orientation before sampling the cone.
            if len(valid)>=needed:
                chosen[branch]=select_spread(valid,axis,needed)
                print(f'{branch}: automatic orientation found; {len(valid)} eligible cone points',flush=True)
                break
        if branch not in chosen:
            raise ValueError(f'{branch}: 当前球位未找到完整可采集姿态；最后检查：'+ '; '.join(failures[-5:]))
    checker.assert_scene()
    return group_candidates(chosen,placement,fit,stages,split)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--seed-joints-deg',nargs=6,type=float)
    args=parser.parse_args();inputs=json.loads(Path(args.input).read_text())
    rclpy.init();node=PlanMoveItChecker('/joint_states_robot',args.seed_joints_deg)
    checker=None
    try:
        node.wait_ready(15.)
        options=SimpleNamespace(frame_id='base_link',ik_link='Link6',group='cr5_group',euler_sequence='xyz',
            safe_transition_mode='cartesian',service_timeout_sec=10.,max_step_m=.001,min_fraction=1.,
            ik_timeout=.3,jump_threshold=0.,max_ik_joint_step_deg=15.,publish_display_trajectory=False,
            motion_policy=inputs.get('motion_policy'))
        checker=SphereContactPreflight(node,options)
        plans=generate(checker,inputs['fit'],inputs['placement'],inputs['stages'],inputs.get('manual',[]),inputs.get('dataset_split'),inputs.get('search_timeout_sec',300.))
        Path(args.output).write_text(json.dumps(plans,ensure_ascii=False))
    finally:
        if checker:checker.close()
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
