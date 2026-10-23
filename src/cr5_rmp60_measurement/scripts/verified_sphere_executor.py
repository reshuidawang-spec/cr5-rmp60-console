#!/usr/bin/env python3
"""Execute freshly certified joint trajectories with guarded ServoJ, never endpoint fallback.

Real execution requires measured tracking/stopping bounds for this installation.
The generic FollowJointTrajectory server is deliberately not used because its
error path can replace the trajectory with a final-point MovJ.
"""
from __future__ import annotations
import argparse,copy,json,math,socket,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rclpy
from check_five_branch_plan_moveit import PlanMoveItChecker,make_seed_state,trajectory_max_joint_step_deg
from direct_tcp_execute_five_branch_plan import FeedbackReader,Dashboard,contact_row,append_stable_row,parse_reply_code,parse_reply_values
from moveit_sphere_contact_safety import SphereContactPreflight
from sphere_contact_safety import BALL_LINKS,validate_safety_receipt,certify_tool_motion
from sphere_motion_contract import load_policy,require_commissioned,servo_samples,validate_bundle
from sphere_experiment_core import read_csv


class ServoTransport:
    """One command transmission; an ambiguous reply is a failure, never a retry."""
    def __init__(self,ip,timeout=.2):self.ip=ip;self.timeout=timeout;self.sock=None
    def command(self,text):
        if self.sock is None:self.sock=socket.create_connection((self.ip,29999),timeout=self.timeout)
        self.sock.settimeout(self.timeout);self.sock.sendall((text+'\n').encode('ascii'))
        reply=b''
        while b';' not in reply:
            chunk=self.sock.recv(4096)
            if not chunk:raise RuntimeError('servo command connection closed; outcome unknown')
            reply+=chunk
            if len(reply)>65536:raise RuntimeError('invalid servo reply')
        value=reply.decode('ascii',errors='strict')
        if parse_reply_code(value)!=0:raise RuntimeError('servo command rejected: '+value)
        return value
    def stop(self):
        try:return self.command('Stop()')
        except Exception:
            self.close()
            # A fresh Stop is idempotent; a failed movement is never resent.
            return self.command('Stop()')
    def close(self):
        if self.sock is not None:self.sock.close();self.sock=None


class JointStreamer:
    def __init__(self,transport,feedback,policy):
        self.transport,self.feedback,self.policy=transport,feedback,policy
        c=policy['commissioning']
        self.tracking=float(c['tracking_joint_error_deg'])
        self.stopping=float(c['stop_joint_error_deg'])
        self.period=float(policy['servo_period_sec'])
        self.last_segment=None

    def snapshot(self):
        state=self.feedback.snapshot(self.policy['max_feedback_age_sec'])
        if state.get('robot_mode') not in (5,7):raise RuntimeError('robot left the commissioned enabled/running modes')
        values=np.asarray(state['joints'],dtype=float)
        if values.shape!=(6,) or not np.isfinite(values).all():raise RuntimeError('invalid live joint feedback')
        return state

    def guard(self,state,start,end,extra=0.):
        q=np.radians(state['joints']);delta=end-start
        t=float(np.clip((q-start)@delta/(delta@delta),0,1)) if delta@delta>1e-20 else 0.
        error=float(np.max(np.abs(np.degrees(q-(start+delta*t)))))
        if error>self.tracking+extra:raise RuntimeError(f'joint trajectory tube exceeded: {error:.6f} deg')

    def stop_and_settle(self):
        self.transport.stop();deadline=time.monotonic()+2.;last=self.snapshot();stable=0
        while time.monotonic()<deadline:
            state=self.snapshot()
            if self.last_segment is not None:self.guard(state,*self.last_segment,extra=self.stopping)
            if state.get('sequence')!=last.get('sequence'):
                stable=stable+1 if np.max(np.abs(np.array(state['joints'])-np.array(last['joints'])))<.002 else 0
                last=state
                if stable>=3:return state
            time.sleep(.003)
        raise RuntimeError('robot stop could not be confirmed')

    def run(self,knots,phase):
        try:
            return self._stream(knots,phase)
        except BaseException:
            self.transport.stop()
            raise

    def _stream(self,knots,phase):
        knots=np.asarray(knots,dtype=float);start=knots[0]
        state=self.snapshot();self.guard(state,start,start)
        if phase!='retract' and state['di1']:raise RuntimeError('probe already triggered before '+phase)
        armed_sequence = state.get('sequence', -1)
        try:
            for end in servo_samples(knots,self.period,self.policy['max_joint_speed_deg_sec']):
                self.last_segment=(start,end)
                state=self.snapshot();self.guard(state,start,end)
                event = self.feedback.trigger_since(armed_sequence)
                if (event is not None or state['di1']) and phase!='retract':
                    stop=self.stop_and_settle()
                    if phase=='probe':return event or state,stop
                    raise RuntimeError('unexpected DI1 before contact during '+phase)
                angles=','.join(f'{v:.9f}' for v in np.degrees(end))
                self.transport.command(f'ServoJ({angles},t={self.period:.6f})')
                deadline=time.monotonic()+self.period
                while time.monotonic()<deadline:
                    state=self.snapshot();self.guard(state,start,end)
                    event = self.feedback.trigger_since(armed_sequence)
                    if (event is not None or state['di1']) and phase!='retract':
                        stop=self.stop_and_settle()
                        if phase=='probe':return event or state,stop
                        raise RuntimeError('unexpected DI1 during '+phase)
                    time.sleep(min(.003,max(0.,deadline-time.monotonic())))
                state=self.snapshot()
                if np.max(np.abs(np.asarray(state['joints'])-np.degrees(end)))>self.tracking:
                    raise RuntimeError('servo did not reach the certified next sample')
                start=end
            return None,self.snapshot()
        except BaseException:
            self.transport.stop()
            raise


def state_from_snapshot(snapshot):
    from sensor_msgs.msg import JointState
    js=JointState(name=[f'joint{i}' for i in range(1,7)],position=np.radians(snapshot['joints']).tolist())
    return make_seed_state(js)


def certified_recovery(checker,snapshot,row,phase):
    checker.assert_scene()
    active=BALL_LINKS[row['branch']] if phase=='retract' else None
    checker.set_active(active)
    seed=state_from_snapshot(snapshot)
    target=[float(row[f'{"transition" if phase=="retract" else "safe"}_flange_{axis}']) for axis in ('x','y','z','rx','ry','rz')]
    options=copy.copy(checker.args);options.max_step_m=.0001 if phase=='retract' else .001
    if phase=='retract':certify_tool_motion(checker.model,checker.flange_pose(seed),target,checker.obstacles,active_ball=active)
    ok,code,fraction,trajectory,step=checker.node.compute_path(seed,[target],options,True)
    if not ok or fraction<1.-1e-10:
        if phase=='retract':raise RuntimeError('cannot certify contact recovery; robot remains stopped')
        trajectory=checker.plan_certified_transfer(seed,target)
        step=trajectory_max_joint_step_deg(seed,trajectory)
    if step>checker.args.max_ik_joint_step_deg:raise RuntimeError('recovery has excessive joint jump')
    names=list(trajectory.joint_trajectory.joint_names)
    knots=[np.radians(snapshot['joints']).tolist(),*[[p.positions[names.index(j)] for j in checker.whole.joint_names] for p in trajectory.joint_trajectory.points]]
    certificate=checker.whole.certify(knots,active_ball=active)
    return knots,certificate


def execute_contact(checker,row,feedback,transport,args):
    policy=checker.policy
    require_commissioned(policy,model_hash=checker.model_hash,robot_ip=args.ip,installation_id=args.installation_id)
    validate_safety_receipt(row);validate_bundle(row,policy)
    if row["sphere_safety_model_sha256"] != checker.model_hash or row["sphere_safety_world_sha256"] != checker.world_hash:
        raise ValueError("当前整机模型或场景与冻结路径不一致；禁止执行")
    start=feedback.wait_fresh(5.)
    if start['di1']:raise RuntimeError('DI1 is not released')
    proof,_=checker.check_row(row,state_from_snapshot(start))
    actual={**row,**proof};bundle=validate_bundle(actual,policy)
    current=feedback.snapshot(policy['max_feedback_age_sec'])
    if np.max(np.abs(np.asarray(current['joints'])-np.asarray(start['joints'])))>.01:
        raise RuntimeError('robot moved while its actual entry trajectory was being checked')
    fk=checker.flange_pose(state_from_snapshot(current))
    c=policy['commissioning']
    if np.linalg.norm(fk[:3]-np.asarray(current['pose'][:3]))>float(c['geometry_error_mm']):
        raise RuntimeError('live TCP/flange pose disagrees with the commissioned joint FK')
    streamer=JointStreamer(transport,feedback,policy)
    audit={'plan_id':row['plan_id'],'model_sha256':checker.model_hash,'world_sha256':checker.world_hash,
           'verified_trajectory_sha256':actual['verified_trajectory_sha256'],'phases':[]}
    item={'row':actual,'plan_id':row['plan_id'],'branch':row['branch'],
          'start_pose':[float(row[f'start_flange_{k}']) for k in ('x','y','z','rx','ry','rz')],
          'target_pose':[float(row[f'target_flange_{k}']) for k in ('x','y','z','rx','ry','rz')]}
    trigger=stop=None
    written=False
    try:
        for phase in bundle['phases'][:4]:
            checker.assert_scene()
            record={'name':phase['name'],'joints_rad':phase['joints_rad'],'status':'STARTED'}
            audit['phases'].append(record)
            trigger,stop=streamer.run(phase['joints_rad'],phase['name'])
            record['status']='TRIGGERED' if trigger is not None else 'COMPLETED'
        if trigger is None:
            transport.stop();status='MISS'
        else:
            status='HIT'
            with Path(args.output).with_suffix('.triggers.jsonl').open('a') as handle:
                handle.write(json.dumps({'plan_id':row['plan_id'], 'trigger':trigger, 'stop':stop})+'\n')
                handle.flush()
            for phase in ('retract','return'):
                snapshot=feedback.snapshot(policy['max_feedback_age_sec'])
                knots,certificate=certified_recovery(checker,snapshot,row,phase)
                audit['phases'].append({'name':phase,'joints_rad':knots,'certificate':certificate})
                streamer.run(knots,phase)
                if phase=='retract' and feedback.snapshot(policy['max_feedback_age_sec'])['di1']:
                    raise RuntimeError('DI1 remains triggered after certified recovery')
        append_stable_row(args.output,contact_row(args,item,status,trigger,stop))
        written=True
        if status!='HIT':raise RuntimeError('no DI1 contact inside certified probing stroke; robot stopped')
    except BaseException:
        transport.stop()
        if not written:
            append_stable_row(args.output,contact_row(args,item,'MOTION_FAIL',trigger,stop))
        raise
    finally:
        audit_path=Path(args.output).with_suffix('.motion.jsonl')
        with audit_path.open('a') as handle:handle.write(json.dumps(audit,allow_nan=False)+'\n')
        checker.set_active(None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-fit-json',type=Path,required=True)
    parser.add_argument('--ip',required=True);parser.add_argument('--installation-id',required=True)
    parser.add_argument('--standard-sphere-id',required=True);parser.add_argument('--motion-policy')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();rows=read_csv(args.plan)
    if len(rows)!=1:raise ValueError('guarded sphere executor accepts one contact at a time')
    row=rows[0];validate_safety_receipt(row)
    policy=load_policy(args.motion_policy) if args.motion_policy else load_policy()
    if not args.execute:
        validate_bundle(row,policy);print('verified joint trajectory dry-run OK; no robot connection');return
    # Block before any robot connection if physical commissioning is missing.
    require_commissioned(policy,model_hash=row['sphere_safety_model_sha256'],robot_ip=args.ip,installation_id=args.installation_id)
    args.session_id=args.installation_id+'_'+row['standard_placement_id']
    args.workpiece_id=args.artifact_id=args.standard_sphere_id
    args.physical_ball_id=args.installation_id;args.artifact_type='sphere'
    args.operator_note='verified joint sphere acquisition';args.euler_sequence='xyz'
    feedback=FeedbackReader(args.ip);transport=ServoTransport(args.ip);node=checker=None
    rclpy.init()
    try:
        feedback.start();snapshot=feedback.wait_fresh(5.)
        node=PlanMoveItChecker('/joint_states_robot',snapshot['joints']);node.wait_ready(10.)
        options=SimpleNamespace(frame_id='base_link',ik_link='Link6',group='cr5_group',euler_sequence='xyz',
            safe_transition_mode='cartesian',service_timeout_sec=10.,max_step_m=.001,min_fraction=1.,ik_timeout=.3,
            jump_threshold=0.,max_ik_joint_step_deg=15.,publish_display_trajectory=False,motion_policy=args.motion_policy)
        checker=SphereContactPreflight(node,options)
        execute_contact(checker,row,feedback,transport,args)
    except BaseException:
        if transport.sock is not None:transport.stop()
        raise
    finally:
        transport.close();feedback.close()
        if checker:checker.close()
        if node:node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':main()
