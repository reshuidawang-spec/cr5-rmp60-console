"""Continuous clearance certificates for piecewise linear joint trajectories.

Every URDF collision mesh is enclosed by its convex hull. At each interval
midpoint, a joint-displacement bound certifies all instants; otherwise the
interval is subdivided. Failure to prove clearance rejects the trajectory.
Only declared physical assembly contacts are excluded, never SRDF 'Never'.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
from native_collision import NativeCollision
from sphere_contact_safety import Box,Capsule,Plane,transform


@dataclass
class Part:
    name:str
    geometry:int
    origin:np.ndarray
    radius:float
    bounds:np.ndarray
    ancestors:set


def origin(element):
    return transform([float(v)*1000 for v in element.attrib.get('xyz','0 0 0').split()],
                     [float(v) for v in element.attrib.get('rpy','0 0 0').split()]) if element is not None else np.eye(4)


def resolve_mesh(uri):
    if uri.startswith('package://'):
        package,relative=uri[len('package://'):].split('/',1)
        return Path(get_package_share_directory(package))/relative
    if uri.startswith('file://'):return Path(uri[7:])
    path=Path(uri)
    if not path.is_absolute():raise ValueError('collision mesh path must be absolute or package://')
    return path


class WholeRobotCollision:
    def __init__(self,xml,obstacles,*,margin_mm=.05,tracking_joint_error_deg=0.,stop_joint_error_deg=0.,geometry_error_mm=0.,max_probe_overtravel_mm=.5):
        self.native=NativeCollision();self.parts=[];self.world=[];self.margin=float(margin_mm)
        self.joint_error=np.full(6,math.radians(tracking_joint_error_deg+stop_joint_error_deg))
        self.geometry_error=float(geometry_error_mm)
        self.max_probe_overtravel=float(max_probe_overtravel_mm)
        if not math.isfinite(self.max_probe_overtravel) or self.max_probe_overtravel<=0:
            raise ValueError("contact overtravel bound must be finite and positive")
        self.base_vertices=None
        self.world_spheres={}
        if not np.isfinite([margin_mm,tracking_joint_error_deg,stop_joint_error_deg,geometry_error_mm]).all() or min(margin_mm,tracking_joint_error_deg,stop_joint_error_deg,geometry_error_mm)<0:
            raise ValueError('collision uncertainty bounds must be finite and nonnegative')
        root=ET.fromstring(xml);self.xml=xml;self.joints=[];self.limits={};self.mesh_hashes={}
        self.parents={};self.children={};self.joint_names=[f'joint{i}' for i in range(1,7)]
        for joint in root.findall('joint'):
            name=joint.attrib['name'];kind=joint.attrib['type']
            parent=joint.find('parent').attrib['link'];child=joint.find('child').attrib['link']
            if kind not in ('fixed','revolute','continuous'):raise ValueError(f'unsupported joint {name}')
            if kind!='fixed' and name not in self.joint_names:raise ValueError(f'unsupported movable joint {name}')
            axis=joint.find('axis')
            axis=np.array([float(v) for v in axis.attrib.get('xyz','1 0 0').split()]) if axis is not None else np.array([1.,0,0])
            if not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-10:raise ValueError('invalid joint axis')
            axis=axis/np.linalg.norm(axis)
            item=(name,kind,parent,child,origin(joint.find('origin')),axis)
            self.joints.append(item);self.parents[child]=item;self.children.setdefault(parent,[]).append(item)
            if kind!='fixed':
                limit=joint.find('limit')
                if kind=='continuous':self.limits[name]=(-2*math.pi,2*math.pi)
                elif limit is None:raise ValueError('missing joint limits')
                else:self.limits[name]=(float(limit.attrib['lower']),float(limit.attrib['upper']))
        self.root=next(link.attrib['name'] for link in root.findall('link') if link.attrib['name'] not in self.parents)
        self.assembly_pairs=set()
        # Adjacent movable robot housings physically share a joint.
        # A fixed frame parent is not evidence of physical contact: side rods
        # are parented to the probe body but are mounted on the lower hub.
        for _,kind,parent,child,_,_ in self.joints:
            if kind != 'fixed':self.assembly_pairs.add(frozenset((parent,child)))
        tool_contacts=[('Link6','rmp60_adapter'),('rmp60_adapter','rmp60_probe_body'),
            ('rmp60_probe_body','rmp60_stylus'),('rmp60_stylus','rmp60_ball_z')]
        tool_contacts += [(f'rmp60_branch_{b}',f'rmp60_ball_{b}') for b in ('x_pos','x_neg','y_pos','y_neg')]
        for pair in tool_contacts:self.assembly_pairs.add(frozenset(pair))
        hub=['rmp60_stylus',*[f'rmp60_branch_{b}' for b in ('x_pos','x_neg','y_pos','y_neg')]]
        for i,a in enumerate(hub):
            for b in hub[i+1:]:self.assembly_pairs.add(frozenset((a,b)))
        # The robot base is bolted to the table. All moving links remain checked.
        self.fixture_pairs=set()
        for link in root.findall('link'):
            name=link.attrib['name'];chain=self.chain(name)
            for collision in link.findall('collision'):
                local=origin(collision.find('origin'));geometry=collision.find('geometry')
                sphere,cylinder,box,mesh=(geometry.find(tag) for tag in ('sphere','cylinder','box','mesh'))
                if sphere is not None:
                    radius=float(sphere.attrib['radius'])*1000;handle=self.native.shape(1,[radius])
                elif cylinder is not None:
                    r=float(cylinder.attrib['radius'])*1000;length=float(cylinder.attrib['length'])*1000
                    handle=self.native.shape(2,[r,length]);radius=math.hypot(r,length/2)
                elif box is not None:
                    size=np.array([float(v)*1000 for v in box.attrib['size'].split()]);radius=np.linalg.norm(size)/2
                    handle=self.native.shape(3,size)
                elif mesh is not None:
                    path=resolve_mesh(mesh.attrib['filename']);self.mesh_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    vertices=self.native.vertices(path)*np.array([float(v) for v in mesh.attrib.get('scale','1 1 1').split()])*1000
                    if name=="base_link":self.base_vertices=(local[:3,:3]@vertices.T).T+local[:3,3]
                    handle=self.native.convex(vertices);radius=float(np.linalg.norm(vertices,axis=1).max())
                else:raise ValueError(f'unsupported collision shape: {name}')
                radius+=np.linalg.norm(local[:3,3])
                bounds=np.zeros(6);ancestors=set()
                for index,joint in enumerate(chain):
                    if joint[1]=='fixed':continue
                    j=self.joint_names.index(joint[0]);ancestors.add(j)
                    bounds[j]=radius+sum(np.linalg.norm(other[4][:3,3]) for other in chain[index+1:])
                self.parts.append(Part(name,handle,local,float(radius),bounds,ancestors))
        for obstacle in obstacles:
            placement=np.eye(4)
            if isinstance(obstacle,Box):
                placement[:3,:3]=obstacle.rotation;placement[:3,3]=obstacle.center
                handle=self.native.shape(3,obstacle.half_size*2)
            elif isinstance(obstacle,Capsule):
                delta=obstacle.b-obstacle.a;length=np.linalg.norm(delta)
                placement[:3,3]=(obstacle.a+obstacle.b)/2
                if length<1e-10:
                    handle=self.native.shape(1,[obstacle.radius])
                    self.world_spheres[obstacle.name]=(obstacle.a,obstacle.radius)
                else:
                    direction=delta/length;z=np.array([0.,0,1.]);cross=np.cross(z,direction)
                    if np.linalg.norm(cross)>1e-12:placement[:3,:3]=Rotation.from_rotvec(cross/np.linalg.norm(cross)*math.acos(np.clip(z@direction,-1,1))).as_matrix()
                    elif direction[2]<0:placement[:3,:3]=Rotation.from_euler('x',math.pi).as_matrix()
                    handle=self.native.shape(4,[obstacle.radius,length])
            else:raise ValueError('whole-robot checker requires bounded obstacle geometry')
            self.world.append((obstacle.name,handle,placement))
        # Permit only verified mounting contact of the fixed robot base and table.
        for obstacle in obstacles:
            if obstacle.name=="calibration_table" and isinstance(obstacle,Box) and self.base_vertices is not None:
                in_table=(obstacle.rotation.T@(self.base_vertices-obstacle.center).T).T
                height=float(np.min(in_table[:,2])-obstacle.half_size[2])
                if height < -.1:
                    raise ValueError("机器人固定底座被台面模型埋入；请修正台面/基座坐标")
                if abs(height)<=.1 and np.all(np.max(np.abs(in_table[:,:2]),axis=0)<=obstacle.half_size[:2]+.1):
                    self.fixture_pairs.add(frozenset(("base_link","calibration_table")))
        # Check static bodies too, except the three intended fixture joints.
        fixture_assembly={frozenset(p) for p in [
            ("calibration_table","calibration_magnetic_base"),
            ("calibration_magnetic_base","calibration_sphere_stem"),
            ("calibration_sphere_stem","calibration_sphere")]}
        for i,(a,ga,ta) in enumerate(self.world):
            for b,gb,tb in self.world[:i]:
                if a==b or frozenset((a,b)) in fixture_assembly:continue
                if self.native.distance(ga,ta,gb,tb)<=self.margin+2*self.geometry_error:
                    raise ValueError(f"static scene bodies collide: {a} / {b}")
        algorithm_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ("whole_robot_collision.py","native_collision.py","sphere_contact_safety.py","sphere_motion_contract.py")}
        self.identity=hashlib.sha256(json.dumps({'xml':xml,'meshes':self.mesh_hashes,'native':self.native.source_hash,'algorithms':algorithm_hashes,
            'assembly':sorted(sorted(p) for p in self.assembly_pairs),'fixture':sorted(sorted(p) for p in self.fixture_pairs)},sort_keys=True).encode()).hexdigest()

    def chain(self,link):
        chain=[]
        while link in self.parents:
            item=self.parents[link];chain.append(item);link=item[2]
        return list(reversed(chain))

    def fk(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('invalid joint vector')
        for i,name in enumerate(self.joint_names):
            if name not in self.limits:continue
            low,high=self.limits[name]
            if not low-1e-9<=q[i]<=high+1e-9:raise ValueError(f'joint limit exceeded: {name}')
        placements={self.root:np.eye(4)}
        stack=list(self.children.get(self.root,[]))
        while stack:
            name,kind,parent,child,fixed,axis=stack.pop()
            rotation=np.eye(4)
            if kind!='fixed':rotation[:3,:3]=Rotation.from_rotvec(axis*q[self.joint_names.index(name)]).as_matrix()
            placements[child]=placements[parent]@fixed@rotation;stack.extend(self.children.get(child,[]))
        base=np.linalg.inv(placements['base_link'])
        return {name:base@pose for name,pose in placements.items()}

    def pairs(self,active_ball=None):
        pairs=[]
        for i,part in enumerate(self.parts):
            for name,handle,pose in self.world:
                if (part.name==active_ball and name=='calibration_sphere') or frozenset((part.name,name)) in self.fixture_pairs:continue
                pairs.append((i,None,name,handle,pose,part.bounds))
            for j in range(i):
                other=self.parts[j]
                if part.name==other.name or frozenset((part.name,other.name)) in self.assembly_pairs:continue
                bounds=part.bounds+other.bounds
                # Common upstream joints move both bodies rigidly and preserve distance.
                bounds=bounds.copy()
                for k in part.ancestors & other.ancestors:bounds[k]=0.
                pairs.append((i,j,other.name,other.geometry,None,bounds))
        return pairs

    def certify(self,knots,*,active_ball=None,max_depth=22):
        knots=np.asarray(knots,dtype=float)
        if knots.ndim!=2 or knots.shape[1]!=6 or len(knots)<1:raise ValueError('invalid joint trajectory')
        if len(knots)==1:knots=np.vstack([knots,knots])
        pairs=self.pairs(active_ball);count=0;minimum=math.inf
        for q0,q1 in zip(knots,knots[1:]):
            self.fk(q0);self.fk(q1);delta=np.abs(q1-q0)
            # Bounds apply to the actual, unwrapped bounded-joint trajectory.
            stack=[(0.,1.,0)]
            while stack:
                left,right,depth=stack.pop();mid=(left+right)/2
                placements=self.fk(q0+(q1-q0)*mid)
                poses=[placements[p.name]@p.origin for p in self.parts]
                unresolved=False
                if active_ball:
                    if "calibration_sphere" not in self.world_spheres:
                        raise ValueError("active contact requires a spherical world object")
                    center,sphere_radius=self.world_spheres["calibration_sphere"]
                    active=[i for i,p in enumerate(self.parts) if p.name==active_ball]
                    if len(active)!=1:raise ValueError("active ruby collision body is ambiguous")
                    index=active[0];part=self.parts[index]
                    gap=float(np.linalg.norm(poses[index][:3,3]-center))-part.radius-sphere_radius
                    uncertainty=2*self.geometry_error+float(part.bounds@self.joint_error)
                    if gap-uncertainty < -self.max_probe_overtravel:
                        raise ValueError("active ruby exceeds the permitted contact overtravel")
                    if gap-uncertainty-float(part.bounds@delta)*(right-left)/2 < -self.max_probe_overtravel:
                        unresolved=True
                for i,j,name,handle,pose,bounds in pairs:
                    d=self.native.distance(self.parts[i].geometry,poses[i],handle,poses[j] if j is not None else pose);count+=1
                    threshold=self.margin+2*self.geometry_error+float(bounds@self.joint_error)
                    if d<=threshold:
                        raise ValueError(f'whole robot collision/clearance: {self.parts[i].name} / {name}, gap={d:.6f} mm, required={threshold:.6f} mm')
                    lower=d-float(bounds@delta)*(right-left)/2-threshold
                    if lower<=0:unresolved=True
                    else:minimum=min(minimum,lower)
                if not unresolved:continue
                if depth>=max_depth:raise ValueError('whole robot sweep cannot be certified')
                stack.extend(((left,mid,depth+1),(mid,right,depth+1)))
        return {'whole_robot_distance_evaluations':count,'minimum_extra_clearance_mm':minimum}

    def close(self):self.native.close()

    def plan_joint_transfer(self,start,goal,*,max_nodes=600,timeout_sec=8.):
        """Bidirectional RRT-Connect; every accepted edge has the same certificate."""
        import time
        start=np.asarray(start,dtype=float);goal=np.asarray(goal,dtype=float)
        self.certify([start]);self.certify([goal])
        try:
            self.certify([start,goal])
            return [start.tolist(),goal.tolist()]
        except ValueError:pass
        limits=np.array([self.limits[name] for name in self.joint_names])
        seed=int(hashlib.sha256(np.r_[start,goal].tobytes()).hexdigest()[:8],16)
        random=np.random.default_rng(seed);deadline=time.monotonic()+timeout_sec
        trees=[([start],[-1]),([goal],[-1])]
        pair_bounds=[pair[-1] for pair in self.pairs()]
        active=np.max(pair_bounds,axis=0)>1e-12 if pair_bounds else np.zeros(6,dtype=bool)
        step=.18
        def extend(tree,target):
            points,parents=tree
            index=int(np.argmin([np.linalg.norm(q-target) for q in points]))
            delta=target-points[index];length=np.linalg.norm(delta)
            if length<1e-9:return index,True
            new=points[index]+delta*min(1.,step/length)
            try:self.certify([points[index],new])
            except ValueError:return None,False
            points.append(new);parents.append(index)
            return len(points)-1,length<=step
        def trace(tree,index):
            path=[]
            while index!=-1:path.append(tree[0][index]);index=tree[1][index]
            return list(reversed(path))
        for iteration in range(max_nodes):
            if time.monotonic()>deadline:break
            first,second=iteration%2,1-iteration%2
            if iteration%5==0:
                sample=(start+goal)/2
                sample[active]=random.uniform(limits[active,0],limits[active,1])
            elif iteration%4==0:sample=trees[second][0][0]
            else:
                low=np.maximum(limits[:,0],np.minimum(start,goal)-1.2)
                high=np.minimum(limits[:,1],np.maximum(start,goal)+1.2)
                sample=(start+goal)/2
                sample[active]=random.uniform(low[active],high[active])
            i,_=extend(trees[first],sample)
            if i is None:continue
            target=trees[first][0][i]
            while time.monotonic()<deadline:
                j,reached=extend(trees[second],target)
                if j is None:break
                if reached:
                    a,b=trace(trees[first],i),trace(trees[second],j)
                    path=a+list(reversed(b))[1:] if first==0 else b+list(reversed(a))[1:]
                    # Certify and simplify without introducing unchecked shortcuts.
                    simplified=[path[0]];index=0
                    while index<len(path)-1:
                        end=len(path)-1
                        while end>index+1:
                            try:self.certify([path[index],path[end]]);break
                            except ValueError:end-=1
                        simplified.append(path[end]);index=end
                    # Keep exported knot steps below the common continuity gate.
                    result=[simplified[0].tolist()]
                    for a,b in zip(simplified,simplified[1:]):
                        count=max(1,int(math.ceil(np.max(np.abs(b-a))/.18)))
                        result.extend((a+(b-a)*t/count).tolist() for t in range(1,count+1))
                    self.certify(result)
                    return result
        raise ValueError('no whole-robot certified joint transfer route found within search budget')
