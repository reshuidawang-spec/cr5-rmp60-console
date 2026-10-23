"""Versioned motion contract: certify and execute the same joint knots, never MovJ fallback."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np

DEFAULT_POLICY=Path(__file__).resolve().parents[1]/'config/sphere_motion_policy.json'
EXECUTION_MODE='verified_joint_servo_v1'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_policy(path=DEFAULT_POLICY):
    policy=json.loads(Path(path).read_text())
    for key in ('model_clearance_mm','servo_period_sec','max_joint_speed_deg_sec','max_feedback_age_sec','max_probe_overtravel_mm'):
        value=policy.get(key)
        if isinstance(value,bool) or value is None or not math.isfinite(float(value)) or float(value)<=0:raise ValueError(f'invalid motion policy {key}')
    if not .02<=policy['servo_period_sec']<=.1 or policy['max_joint_speed_deg_sec']>1.:
        raise ValueError('sphere servo policy exceeds validated software limits')
    return policy


def require_commissioned(policy,*,model_hash,robot_ip,installation_id):
    c=policy.get('commissioning',{})
    if c.get('verified') is not True:
        raise ValueError('自动关节轨迹采集尚缺本次安装的控制器跟踪/停止距离验证；当前仅允许仿真')
    if c.get('robot_model_sha256')!=model_hash or c.get('robot_ip')!=robot_ip or c.get('installation_id')!=installation_id or not c.get('controller_firmware'):
        raise ValueError('运动验证记录与当前机器人、安装或模型不一致')
    for name in ('tracking_joint_error_deg','stop_joint_error_deg','geometry_error_mm'):
        value=c.get(name)
        if value is None or isinstance(value,bool) or not math.isfinite(float(value)) or float(value)<=0:
            raise ValueError(f'缺少实测误差上界：{name}')
    evidence=c.get('evidence')
    if not evidence:raise ValueError('缺少控制器轨迹及停止验证原始证据')
    for item in evidence:
        path=Path(item['path'])
        if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest()!=item.get('sha256'):
            raise ValueError('控制器验证证据缺失或内容已改变')
    return c


def make_bundle(row,phases,model_hash,world_hash,policy):
    return {'schema_version':1,'execution_mode':EXECUTION_MODE,'plan_id':row['plan_id'],
            'model_sha256':model_hash,'world_sha256':world_hash,'policy_sha256':digest(policy),
            'phases':phases}


def validate_bundle(row,policy=None):
    raw=row.get('verified_trajectory_json','')
    if hashlib.sha256(raw.encode()).hexdigest()!=row.get('verified_trajectory_sha256'):
        raise ValueError('已验证关节轨迹缺失或已改变')
    bundle=json.loads(raw)
    if bundle.get('execution_mode')!=EXECUTION_MODE or bundle.get('plan_id')!=row.get('plan_id'):
        raise ValueError('关节轨迹不属于当前采点计划')
    if bundle.get('model_sha256')!=row.get('sphere_safety_model_sha256') or bundle.get('world_sha256')!=row.get('sphere_safety_world_sha256'):
        raise ValueError('关节轨迹与碰撞模型/场景不一致')
    if policy is not None and bundle.get('policy_sha256')!=digest(policy):raise ValueError('误差余量或执行参数已变化；须重新检查轨迹')
    phases=bundle.get('phases',[])
    expected=['entry','transition','approach','probe','retract','return']
    if [p.get('name') for p in phases]!=expected:raise ValueError('关节轨迹缺少完整采集阶段')
    previous=None
    for phase in phases:
        q=np.asarray(phase['joints_rad'],dtype=float)
        if q.ndim!=2 or q.shape[1]!=6 or len(q)<1 or not np.isfinite(q).all():raise ValueError('无效的已验证关节轨迹')
        if previous is not None and np.max(np.abs(q[0]-previous))>1e-8:raise ValueError('关节轨迹阶段之间不连续')
        previous=q[-1]
    return bundle


def servo_samples(knots,period_sec,max_speed_deg_sec):
    knots=np.asarray(knots,dtype=float)
    if knots.ndim!=2 or knots.shape[1]!=6 or not np.isfinite(knots).all():raise ValueError('invalid servo knots')
    if not math.isfinite(period_sec*max_speed_deg_sec) or min(period_sec,max_speed_deg_sec)<=0:raise ValueError('invalid servo timing')
    for start,end in zip(knots,knots[1:]):
        count=max(1,int(math.ceil(np.max(np.abs(np.degrees(end-start)))/(max_speed_deg_sec*period_sec))))
        for index in range(1,count+1):yield start+(end-start)*(index/count)
