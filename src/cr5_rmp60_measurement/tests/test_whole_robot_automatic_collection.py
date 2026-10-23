import sys,math,copy,json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from whole_robot_collision import WholeRobotCollision
from sphere_contact_safety import Capsule
from sphere_motion_contract import load_policy,require_commissioned,servo_samples,validate_bundle
from automatic_sphere_planner import contact_row,group_candidates
from test_sphere_contact_safety import stamped_row


def revolute_xml():
    return '''<robot><link name="base_link"/><link name="Link1">
      <collision><origin xyz="0.002 0 0"/><geometry><sphere radius="0.0001"/></geometry></collision></link>
      <joint name="joint1" type="revolute"><parent link="base_link"/><child link="Link1"/>
      <axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint></robot>'''


def test_whole_arm_mid_interval_collision_is_rejected():
    point=np.array([2.,0,0]);world=[Capsule('obstacle',point,point,.1)]
    robot=WholeRobotCollision(revolute_xml(),world,margin_mm=.01)
    q0=np.array([-.4,0,0,0,0,0]);q1=-q0
    try:
        robot.certify([q0]);robot.certify([q1])
        with pytest.raises(ValueError,match='Link1 / obstacle'):robot.certify([q0,q1])
    finally:robot.close()


def test_joint_tracking_and_stopping_tube_changes_clearance():
    point=np.array([2.,.5,0]);world=[Capsule('obstacle',point,point,.1)]
    robot=WholeRobotCollision(revolute_xml(),world,margin_mm=.01,tracking_joint_error_deg=10.)
    try:
        with pytest.raises(ValueError,match='required='):robot.certify([np.zeros(6)])
    finally:robot.close()


def test_unlisted_self_collision_cannot_be_hidden_by_srdf_never():
    xml='''<robot><link name="base_link"/><link name="A"><collision><geometry><sphere radius=".001"/></geometry></collision></link>
    <link name="B"><collision><geometry><sphere radius=".001"/></geometry></collision></link>
    <joint name="a" type="fixed"><parent link="base_link"/><child link="A"/></joint>
    <joint name="b" type="fixed"><parent link="base_link"/><child link="B"/></joint></robot>'''
    robot=WholeRobotCollision(xml,[])
    try:
        with pytest.raises(ValueError,match='B / A'):robot.certify([np.zeros(6)])
    finally:robot.close()


def test_continuous_joint_limit_check_uses_unwrapped_angles():
    robot=WholeRobotCollision(revolute_xml(),[])
    try:
        with pytest.raises(ValueError,match='joint limit'):robot.certify([[3.1,0,0,0,0,0]])
    finally:robot.close()


def test_exact_native_primitive_distance_and_convex_containment():
    from native_collision import NativeCollision
    from itertools import product
    native=NativeCollision()
    try:
        small=native.shape(1,[.1]);hull=native.convex(list(product((-1.,1.),repeat=3)))
        assert native.distance(small,np.eye(4),hull,np.eye(4))<0
        pose=np.eye(4);pose[0,3]=2
        assert native.distance(small,pose,hull,np.eye(4))==pytest.approx(.9,abs=1e-5)
    finally:native.close()


def test_servo_samples_follow_the_certified_joint_segments_without_shortcut():
    knots=np.array([[0]*6,[.1,0,0,0,0,0],[.1,.1,0,0,0,0]])
    samples=list(servo_samples(knots,.03,1.))
    assert np.allclose(samples[-1],knots[-1])
    assert any(np.allclose(q,knots[1]) for q in samples)
    assert max(np.max(np.abs(np.degrees(b-a))) for a,b in zip([knots[0],*samples],samples))<=.030001
    assert all(q[1]==0 or q[0]==.1 for q in samples)


def test_uncommissioned_policy_blocks_real_motion_and_changed_bundle_rejected():
    policy=load_policy()
    with pytest.raises(ValueError,match='仅允许仿真'):
        require_commissioned(policy,model_hash='a'*64,robot_ip='192.168.5.1',installation_id='new')
    row=stamped_row(plan_id='one');validate_bundle(row,policy)
    bundle=json.loads(row['verified_trajectory_json']);bundle['phases'][0]['joints_rad'][0][0]=1
    row['verified_trajectory_json']=json.dumps(bundle)
    with pytest.raises(ValueError,match='已改变'):validate_bundle(row)


def test_sphere_translation_moves_all_pose_positions_without_changing_approach():
    arguments=(10.,'z',[0,0,209.27],1.,np.eye(3),[0,0,-1])
    first=contact_row([-400,120,89.487],*arguments)
    second=contact_row([-600,-150,89.487],*arguments)
    for prefix in ('safe','transition','start','target'):
        a=np.array([float(first[f'{prefix}_flange_{k}']) for k in 'xyz'])
        b=np.array([float(second[f'{prefix}_flange_{k}']) for k in 'xyz'])
        assert np.allclose(b-a,[-200,-270,0])
    assert first['approach_z']==second['approach_z']


def test_auto_grouping_preserves_original_sample_counts_and_repeats():
    from sphere_contact_safety import BRANCHES
    chosen={b:[{'approach_x':str(i)} for i in range(27 if b=='z' else 16)] for b in BRANCHES}
    stages=('smoke','geometry','anchor','evaluation')
    plans=group_candidates(chosen,'P0',{'installation_id':'new','standard_sphere_id':'ball'},stages,'train')
    assert {stage:len(rows) for stage,rows in plans.items()}=={'smoke':5,'geometry':48,'anchor':8,'evaluation':90}
    assert len({r['plan_id'] for rows in plans.values() for r in rows})==151


class SimulatedFeedback:
    def __init__(self):self.q=np.zeros(6);self.sequence=0;self.event=None;self.di=False
    def snapshot(self,age):
        self.sequence+=1
        return {'joints':self.q.tolist(),'sequence':self.sequence,'di1':self.di,'robot_mode':5,
                'pose':[0.]*6,'monotonic_time':0.}
    def trigger_since(self,sequence):return self.event if self.event is not None and self.event['sequence']>sequence else None


class SimulatedTransport:
    def __init__(self,feedback,fail=False,trigger=False):self.feedback=feedback;self.commands=[];self.fail=fail;self.trigger=trigger
    def command(self,text):
        self.commands.append(text)
        if text.startswith('ServoJ'):
            if self.fail:raise RuntimeError('simulated command failure')
            self.feedback.q=np.array([float(x) for x in text[7:].split(',')[:6]])
            if self.trigger:
                self.feedback.sequence+=1
                self.feedback.event={'sequence':self.feedback.sequence,'joints':self.feedback.q.tolist(),'di1':True}
        return '0,{},ServoJ();'
    def stop(self):self.commands.append('Stop()')


def simulation_stream_policy():
    policy=load_policy();policy['commissioning'].update(tracking_joint_error_deg=.1,stop_joint_error_deg=.1)
    return policy


def test_stream_failure_stops_without_endpoint_fallback():
    from verified_sphere_executor import JointStreamer
    feedback=SimulatedFeedback();transport=SimulatedTransport(feedback,fail=True)
    streamer=JointStreamer(transport,feedback,simulation_stream_policy())
    with pytest.raises(RuntimeError,match='simulated command failure'):
        streamer.run([[0.]*6,[.0001,0,0,0,0,0]],'entry')
    assert sum(command.startswith('ServoJ') for command in transport.commands)==1
    assert 'Stop()' in transport.commands
    assert not any(command.startswith(('MovJ','MovL')) for command in transport.commands)


def test_brief_di_pulse_is_latched_and_stops_the_probe():
    from verified_sphere_executor import JointStreamer
    feedback=SimulatedFeedback();transport=SimulatedTransport(feedback,trigger=True)
    streamer=JointStreamer(transport,feedback,simulation_stream_policy())
    trigger,stopped=streamer.run([[0.]*6,[.0001,0,0,0,0,0]],'probe')
    assert trigger['di1'] is True and feedback.di is False
    assert 'Stop()' in transport.commands


def test_trigger_during_transfer_is_rejected_and_stopped():
    from verified_sphere_executor import JointStreamer
    feedback=SimulatedFeedback();transport=SimulatedTransport(feedback,trigger=True)
    streamer=JointStreamer(transport,feedback,simulation_stream_policy())
    with pytest.raises(RuntimeError,match='unexpected DI1'):
        streamer.run([[0.]*6,[.0001,0,0,0,0,0]],'transition')
    assert 'Stop()' in transport.commands


def test_tracking_departure_stops_before_sending_the_next_point():
    from verified_sphere_executor import JointStreamer
    feedback=SimulatedFeedback();feedback.q[0]=2.
    transport=SimulatedTransport(feedback)
    streamer=JointStreamer(transport,feedback,simulation_stream_policy())
    with pytest.raises(RuntimeError,match='tube exceeded'):
        streamer.run([[0.]*6,[.0001,0,0,0,0,0]],'entry')
    assert not any(command.startswith('ServoJ') for command in transport.commands)
    assert 'Stop()' in transport.commands


def test_active_ruby_contact_does_not_allow_excessive_penetration():
    xml='''<robot><link name="base_link"/><link name="rmp60_ball_z"><collision><geometry>
    <sphere radius=".001"/></geometry></collision></link><joint name="ruby" type="fixed">
    <parent link="base_link"/><child link="rmp60_ball_z"/><origin xyz="0 0 .0102"/></joint></robot>'''
    robot=WholeRobotCollision(xml,[Capsule('calibration_sphere',np.zeros(3),np.zeros(3),10.)])
    try:
        with pytest.raises(ValueError,match='overtravel'):
            robot.certify([np.zeros(6)],active_ball='rmp60_ball_z')
    finally:robot.close()


def test_static_scene_obstacles_are_checked_for_unintended_overlap():
    point=np.array([10.,0,0])
    with pytest.raises(ValueError,match='static scene bodies collide'):
        robot=WholeRobotCollision(revolute_xml(),[Capsule('one',point,point,1.),Capsule('two',point,point,1.)])


def test_joint_transfer_search_finds_a_certified_detour_around_midpath_obstacle():
    links=['<link name="base_link"/>']
    for i in range(1,7):
        collision='<collision><origin xyz=".002 0 0"/><geometry><sphere radius=".0001"/></geometry></collision>' if i==2 else ''
        links.append(f'<link name="L{i}">{collision}</link>')
        parent='base_link' if i==1 else f'L{i-1}'
        position='.002 0 0' if i==2 else '0 0 0'
        links.append(f'<joint name="joint{i}" type="revolute"><parent link="{parent}"/><child link="L{i}"/>'
                     f'<origin xyz="{position}"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>')
    point=np.array([4.,0,0])
    robot=WholeRobotCollision('<robot>'+''.join(links)+'</robot>',[Capsule('obstacle',point,point,.2)],margin_mm=.01)
    start=np.array([-.5,0,0,0,0,0]);goal=-start
    try:
        with pytest.raises(ValueError):robot.certify([start,goal])
        route=robot.plan_joint_transfer(start,goal,timeout_sec=3.)
        assert np.allclose(route[0],start) and np.allclose(route[-1],goal)
        robot.certify(route)
    finally:robot.close()


def test_real_execution_is_blocked_before_any_robot_connection(tmp_path,monkeypatch):
    import verified_sphere_executor as executor
    from sphere_experiment_core import write_csv
    plan=tmp_path/'plan.csv';write_csv(plan,[stamped_row(plan_id='one')])
    def connection_forbidden(*args,**kwargs):raise AssertionError('robot connection attempted before commissioning')
    monkeypatch.setattr(executor,'FeedbackReader',connection_forbidden)
    monkeypatch.setattr(executor,'ServoTransport',connection_forbidden)
    monkeypatch.setattr(sys,'argv',['verified_sphere_executor.py','--plan',str(plan),'--output',str(tmp_path/'hits.csv'),
        '--reference-fit-json',str(tmp_path/'fit.json'),'--ip','192.168.5.1','--installation-id','new',
        '--standard-sphere-id','ball','--execute'])
    with pytest.raises(ValueError,match='仅允许仿真'):executor.main()
    assert not (tmp_path/'hits.csv').exists()


def test_automatic_pose_search_retries_orientation_when_full_entry_route_fails(monkeypatch):
    import automatic_sphere_planner as planner
    from types import SimpleNamespace
    from sensor_msgs.msg import JointState
    from scipy.spatial.transform import Rotation
    monkeypatch.setattr(planner,'BRANCHES',('z',))
    monkeypatch.setattr(planner,'validate_probe',lambda *args,**kwargs:{})
    first=np.eye(3);second=Rotation.from_euler('z',90,degrees=True).as_matrix()
    monkeypatch.setattr(planner,'branch_orientations',lambda *args:iter([(first,np.array([0,0,-1.])),(second,np.array([0,0,-1.]))]))
    names=[f'joint{i}' for i in range(1,7)]
    js=JointState(name=names,position=[0.]*6)
    checks=[]
    def check_row(row,seed):
        checks.append(row)
        if len(checks)==1:raise ValueError('full entry route is blocked')
        return {},seed
    checker=SimpleNamespace(sphere_center=np.array([0.,0,89.487]),sphere_radius=10.,obstacles=[],args=SimpleNamespace(),
        model=SimpleNamespace(balls={'z':SimpleNamespace(a=np.array([0.,0,209.27]),radius=1.)}),
        node=SimpleNamespace(joint_state=js,solve_ik=lambda pose,seed,args,avoid:(True,1,seed)),
        whole=SimpleNamespace(joint_names=names,certify=lambda knots:{}),
        flange_pose=lambda seed:[0,0,300,180,0,0],check_row=check_row,assert_scene=lambda:None)
    plans=planner.generate(checker,{'sphere_center_mm':[0,0,89.487],'sphere_radius_mm':10.,'probe_radius_mm':1.},
        'P1',('smoke',),dataset_split='train')
    assert len(checks)==2
    assert float(plans['smoke'][0]['start_flange_rz'])==pytest.approx(90.)


def test_rviz_srdf_no_longer_hides_nonadjacent_robot_or_wrist_tool_collisions():
    import xml.etree.ElementTree as ET
    root=ET.parse(ROOT/'config/moveit/cr5_rmp60.srdf').getroot()
    for pair in root.findall('disable_collisions'):
        assert pair.attrib['reason'] not in ('Never','Mounted tool near wrist')
        assert not (pair.attrib['link1']=='rmp60_probe_body' and pair.attrib['link2'].startswith('rmp60_branch_'))
