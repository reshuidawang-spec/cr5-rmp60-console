"""Shared, fail-closed MoveIt preflight for sphere workbench paths.

Mutates only the planning-scene ACM while holding a per-domain process lock.
No execution service is used. Tool/world motion has a continuous clearance
certificate; arm/self collision checking uses MoveIt's sampled trajectories.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AllowedCollisionEntry, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPositionFK, GetStateValidity
from whole_robot_collision import WholeRobotCollision
from sphere_motion_contract import load_policy, make_bundle, digest
from rcl_interfaces.srv import GetParameters
from rosidl_runtime_py.convert import message_to_ordereddict
from shape_msgs.msg import SolidPrimitive

from sphere_contact_safety import (
    BALL_LINKS, Box, Capsule, Plane, PROOF_FLAGS, SAFETY_VERSION, ToolModel,
    certify_tool_motion, path_fingerprint, validate_probe,
)
from ros_wait_utils import wait_for_future


def message_transform(pose):
    value = np.eye(4)
    q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    value[:3, :3] = np.eye(3) if np.linalg.norm(q) < 1e-12 else Rotation.from_quat(q).as_matrix()
    value[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z]) * 1000
    if not np.isfinite(value).all():
        raise ValueError("nonfinite scene pose")
    return value


def world_geometry(world, frame, fixed_frames=None):
    result = []
    if world.octomap.octomap.data:
        raise ValueError("octomap cannot be certified by the continuous tool checker")
    for obj in world.collision_objects:
        object_frame = obj.header.frame_id.lstrip("/")
        frames = fixed_frames or {frame.lstrip("/"): np.eye(4)}
        if object_frame not in frames:
            raise ValueError(f"unsupported scene frame for {obj.id}: {obj.header.frame_id}")
        base = frames[object_frame] @ message_transform(getattr(obj, "pose", Pose()))
        if len(obj.primitives) != len(obj.primitive_poses) or len(obj.meshes) != len(obj.mesh_poses) or len(obj.planes) != len(obj.plane_poses):
            raise ValueError(f"incomplete scene geometry: {obj.id}")
        for primitive, pose in zip(obj.primitives, obj.primitive_poses):
            placement = base @ message_transform(pose)
            center, rotation = placement[:3, 3], placement[:3, :3]
            dimensions = np.array(primitive.dimensions) * 1000
            if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                raise ValueError(f"invalid obstacle dimensions: {obj.id}")
            if primitive.type == SolidPrimitive.SPHERE and len(dimensions) == 1:
                result.append(Capsule(obj.id, center, center, dimensions[0]))
            elif primitive.type == SolidPrimitive.CYLINDER and len(dimensions) == 2:
                axis = rotation[:, 2] * dimensions[0] / 2
                result.append(Capsule(obj.id, center - axis, center + axis, dimensions[1]))
            elif primitive.type == SolidPrimitive.BOX and len(dimensions) == 3:
                result.append(Box(obj.id, center, rotation, dimensions / 2))
            else:
                raise ValueError(f"unsupported obstacle primitive: {obj.id}")
        for mesh, pose in zip(obj.meshes, obj.mesh_poses):
            if not mesh.vertices:
                raise ValueError(f"empty collision mesh: {obj.id}")
            vertices = np.array([[p.x, p.y, p.z] for p in mesh.vertices]) * 1000
            if not np.isfinite(vertices).all():
                raise ValueError(f"nonfinite collision mesh: {obj.id}")
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            placement = base @ message_transform(pose)
            result.append(Box(obj.id, placement[:3, :3] @ ((low + high) / 2) + placement[:3, 3],
                              placement[:3, :3], (high - low) / 2))
        for plane, pose in zip(obj.planes, obj.plane_poses):
            normal = np.array(plane.coef[:3], dtype=float)
            norm = np.linalg.norm(normal)
            if not math.isfinite(norm) or norm < 1e-12:
                raise ValueError(f"invalid plane: {obj.id}")
            placement = base @ message_transform(pose)
            normal = placement[:3, :3] @ normal / norm
            result.append(Plane(obj.id, normal, float(plane.coef[3]) * 1000 / norm - normal @ placement[:3, 3]))
        if not (obj.primitives or obj.meshes or obj.planes):
            raise ValueError(f"empty obstacle: {obj.id}")
    spheres = [p for p in result if p.name == "calibration_sphere"]
    if len(spheres) != 1 or not isinstance(spheres[0], Capsule) or not np.array_equal(spheres[0].a, spheres[0].b):
        raise ValueError("scene requires exactly one spherical calibration_sphere")
    required = {"calibration_table", "calibration_magnetic_base", "calibration_sphere_stem"}
    if not required.issubset({p.name for p in result}):
        raise ValueError("scene is missing table, magnetic base or sphere stem")
    return result, spheres[0]


def strict_acm(original, robot_links, world_ids, active_ball=None, *, assembly_pairs=None, fixture_pairs=()):
    """Override even inherited/default allowances for every robot/world pair."""
    matrix = copy.deepcopy(original)
    names = list(matrix.entry_names)
    old = {(a, b): bool(matrix.entry_values[i].enabled[j])
           for i, a in enumerate(names) for j, b in enumerate(names)}
    names = list(dict.fromkeys([*names, *robot_links, *world_ids]))
    defaults = dict(zip(matrix.default_entry_names, matrix.default_entry_values))
    matrix.entry_names = names
    matrix.entry_values = []
    for a in names:
        values = []
        for b in names:
            allowed = old.get((a, b), bool(defaults.get(a, False) or defaults.get(b, False)))
            if assembly_pairs is not None and a in robot_links and b in robot_links:
                allowed = a == b or frozenset((a, b)) in assembly_pairs
            if (a in world_ids and b in robot_links) or (b in world_ids and a in robot_links):
                allowed = (bool(active_ball) and {a, b} == {active_ball, "calibration_sphere"}) or frozenset((a, b)) in fixture_pairs
            values.append(allowed)
        matrix.entry_values.append(AllowedCollisionEntry(enabled=values))
    return matrix


class SphereContactPreflight:
    def __init__(self, node, args):
        self.node, self.args = node, copy.copy(args)
        if args.frame_id != "base_link" or args.ik_link != "Link6" or args.euler_sequence != "xyz" or args.safe_transition_mode != "cartesian":
            raise ValueError("sphere safety requires base_link / Link6 / xyz / Cartesian staging")
        self.args.min_fraction = 1.
        self.args.max_step_m = min(args.max_step_m, .001)
        self.lock = open(f"/tmp/cr5_sphere_acm_{os.environ.get('ROS_DOMAIN_ID', '0')}.lock", "a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("another sphere collision preflight is using this MoveIt domain")
        self.original_acm = None
        try:
            self.get_scene = self.client(GetPlanningScene, "/get_planning_scene")
            self.apply_scene = self.client(ApplyPlanningScene, "/apply_planning_scene")
            self.fk = self.client(GetPositionFK, "/compute_fk")
            self.validity = self.client(GetStateValidity, "/check_state_validity")
            parameters = self.client(GetParameters, "/move_group/get_parameters")
            response = self.call(parameters, GetParameters.Request(names=["robot_description"]))
            xml = response.values[0].string_value
            self.model = ToolModel(xml)
            self.xml = xml
            self.policy = load_policy(getattr(args, "motion_policy", None) or __import__("sphere_motion_contract").DEFAULT_POLICY)
            scene = self.scene()
            self.original_acm = copy.deepcopy(scene.allowed_collision_matrix)
            self.obstacles, sphere = world_geometry(scene.world, args.frame_id, self.model.base_frames)
            self.sphere_center, self.sphere_radius = sphere.a, sphere.radius
            c = self.policy.get("commissioning", {})
            uncertainty = {key: float(c[key]) if c.get("verified") is True and c.get(key) is not None else 0.
                           for key in ("tracking_joint_error_deg", "stop_joint_error_deg", "geometry_error_mm")}
            self.whole = WholeRobotCollision(xml, self.obstacles, margin_mm=self.policy["model_clearance_mm"],
                max_probe_overtravel_mm=self.policy["max_probe_overtravel_mm"], **uncertainty)
            self.model_hash = self.whole.identity
            self.world_hash = self.fingerprint(scene.world)
            self.world_ids = {obj.id for obj in scene.world.collision_objects}
            self.set_active(None)
        except BaseException:
            self.close()
            raise

    def client(self, service, name):
        client = self.node.create_client(service, name)
        if not client.wait_for_service(timeout_sec=self.args.service_timeout_sec):
            raise RuntimeError(f"{name} is not available")
        return client

    def call(self, client, request):
        return wait_for_future(self.node, client.call_async(request), 15., client.srv_name)

    def scene(self):
        request = GetPlanningScene.Request()
        request.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
            PlanningSceneComponents.OCTOMAP | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        scene = self.call(self.get_scene, request).scene
        if scene.robot_state.attached_collision_objects:
            raise ValueError("attached collision objects need a supported continuous geometry model")
        return scene

    @staticmethod
    def fingerprint(world):
        # Timestamps are transport metadata, not geometry. Sort object order too.
        world = copy.deepcopy(world)
        world.collision_objects.sort(key=lambda obj: obj.id)
        world.octomap.header.stamp.sec = world.octomap.header.stamp.nanosec = 0
        for obj in world.collision_objects:
            obj.header.stamp.sec = obj.header.stamp.nanosec = 0
        # CDR byte streams can contain nondeterministic padding. Hash field
        # values rather than transport bytes, which are not a geometry identity.
        return hashlib.sha256(json.dumps(message_to_ordereddict(world), sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()

    def assert_scene(self):
        if self.fingerprint(self.scene().world) != self.world_hash:
            raise ValueError("MoveIt world changed during preflight; restart the whole check")

    def set_active(self, ball):
        self.apply_matrix(strict_acm(self.original_acm, self.model.robot_collision_links, self.world_ids, ball,
                                     assembly_pairs=self.whole.assembly_pairs, fixture_pairs=self.whole.fixture_pairs))

    def apply_matrix(self, matrix):
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.allowed_collision_matrix = matrix
        if not self.call(self.apply_scene, ApplyPlanningScene.Request(scene=scene)).success:
            raise RuntimeError("MoveIt refused collision policy update")

    def close(self):
        try:
            if self.original_acm is not None:
                self.apply_matrix(self.original_acm)
                self.original_acm = None
        finally:
            if hasattr(self, "whole"):
                self.whole.close()
            if not self.lock.closed:
                self.lock.close()

    def flange_pose(self, state):
        req = GetPositionFK.Request()
        req.header.frame_id = self.args.frame_id
        req.fk_link_names = [self.args.ik_link]
        req.robot_state = state
        response = self.call(self.fk, req)
        if response.error_code.val != 1 or len(response.pose_stamped) != 1:
            raise RuntimeError("could not compute current flange FK")
        value = message_transform(response.pose_stamped[0].pose)
        return np.r_[value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_euler("xyz", degrees=True)]

    def plan_certified_transfer(self, state, target, goal_hint=None):
        """Search alternate IK branches and connect using the conservative model."""
        from moveit_msgs.msg import RobotTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        start = [state.joint_state.position[state.joint_state.name.index(j)] for j in self.whole.joint_names]
        failures = []
        if goal_hint is not None:
            try:
                hint = np.asarray(goal_hint, dtype=float)
                if hint.shape != (6,) or not np.isfinite(hint).all():
                    raise ValueError("invalid preferred joint goal")
                pose = self.whole.fk(hint)[self.args.ik_link]
                expected = __import__("sphere_contact_safety").pose_transform(target)
                if np.linalg.norm(pose[:3, 3]-expected[:3, 3]) > .01 or Rotation.from_matrix(pose[:3, :3].T @ expected[:3, :3]).magnitude() > .001:
                    raise ValueError("preferred joint goal disagrees with flange pose")
                route = self.whole.plan_joint_transfer(start, hint)
                trajectory = RobotTrajectory()
                trajectory.joint_trajectory.joint_names = self.whole.joint_names
                for index, q in enumerate(route):
                    point = JointTrajectoryPoint(positions=q)
                    point.time_from_start.sec = index
                    trajectory.joint_trajectory.points.append(point)
                return trajectory
            except ValueError as exc:
                failures.append(str(exc))
        for delta in (None, (1, .6), (1, -.6), (2, .6), (2, -.6), (4, .6), (4, -.6)):
            seed = copy.deepcopy(state)
            if delta:
                index = seed.joint_state.name.index(self.whole.joint_names[delta[0]])
                seed.joint_state.position[index] += delta[1]
            ok, code, solution = self.node.solve_ik(target, seed, self.args, True)
            if not ok:
                continue
            goal = [solution.joint_state.position[solution.joint_state.name.index(j)] for j in self.whole.joint_names]
            try:
                knots = self.whole.plan_joint_transfer(start, goal)
            except ValueError as exc:
                failures.append(str(exc))
                continue
            trajectory = RobotTrajectory()
            trajectory.joint_trajectory.joint_names = self.whole.joint_names
            for i, q in enumerate(knots):
                point = JointTrajectoryPoint(positions=q)
                point.time_from_start.sec = i
                trajectory.joint_trajectory.points.append(point)
            return trajectory
        raise ValueError("no certified staging route: " + "; ".join(failures[-2:]))

    def check_row(self, row, seed):
        from check_five_branch_plan_moveit import pose_from_row
        self.assert_scene()
        entry = row.get("preflight_mode") == "transition-only"
        proof = {} if entry else validate_probe(row, self.model, self.obstacles, self.sphere_center, self.sphere_radius)
        state, current = copy.deepcopy(seed), self.flange_pose(seed)
        trajectories, maximum, evaluations = [], 0., 0
        joint_phases, whole_evaluations = [], 0
        self.set_active(None)
        valid = self.call(self.validity, GetStateValidity.Request(robot_state=state, group_name=self.args.group))
        if not valid.valid:
            raise ValueError("current robot state is in collision")
        phases = [("entry", pose_from_row(row, "safe_flange"), None, False),
                  ("transition", pose_from_row(row, "transition_flange"), None, False)]
        if not entry:
            ball = BALL_LINKS[row["branch"]]
            phases += [("approach", pose_from_row(row, "start_flange"), None, False),
                       ("probe", pose_from_row(row, "target_flange"), ball, True),
                       ("retract", pose_from_row(row, "transition_flange"), ball, True),
                       ("return", pose_from_row(row, "safe_flange"), None, False)]
        try:
            for name, target, ball, fine in phases:
                hint_text = row.get(f"{'safe' if name in ('entry', 'return') else 'transition'}_joint_seed_json", "")
                hint = json.loads(hint_text) if hint_text and name in ("entry", "transition", "return") else None
                self.set_active(ball)
                # The returned joint trajectory is the execution contract. Cartesian
                # tool sweep additionally checks the intended straight probe line.
                if name in ("approach", "probe", "retract"):
                    evaluations += certify_tool_motion(self.model, current, target, self.obstacles, active_ball=ball)
                options = copy.copy(self.args)
                if fine:
                    options.max_step_m = min(options.max_step_m, .0001)
                ok, code, fraction, trajectory, step = self.node.compute_path(state, [target], options, True)
                if (not ok or fraction < 1. - 1e-10) and name in ("entry", "transition", "return"):
                    trajectory = self.plan_certified_transfer(state, target, hint)
                    from check_five_branch_plan_moveit import trajectory_max_joint_step_deg
                    step = trajectory_max_joint_step_deg(state, trajectory)
                    ok, fraction, code = True, 1., 1
                if not ok or fraction < 1. - 1e-10 or step > self.args.max_ik_joint_step_deg or not trajectory.joint_trajectory.points:
                    raise ValueError(f"{name} failed: code={code}, fraction={fraction:.9f}, joint_step={step:.6f}")
                names = list(trajectory.joint_trajectory.joint_names)
                if set(names) != set(self.whole.joint_names):
                    raise ValueError("planned trajectory has incomplete joints")
                start_q = [state.joint_state.position[state.joint_state.name.index(j)] for j in self.whole.joint_names]
                knots = [start_q, *[[point.positions[names.index(j)] for j in self.whole.joint_names]
                                   for point in trajectory.joint_trajectory.points]]
                try:
                    certificate = self.whole.certify(knots, active_ball=ball)
                except ValueError as exc:
                    if name not in ("entry", "transition", "return"):
                        raise ValueError(f"{name}: {exc}") from exc
                    trajectory = self.plan_certified_transfer(state, target, hint)
                    names = list(trajectory.joint_trajectory.joint_names)
                    knots = [start_q, *[[point.positions[names.index(j)] for j in self.whole.joint_names]
                                       for point in trajectory.joint_trajectory.points]]
                    certificate = self.whole.certify(knots)
                    from check_five_branch_plan_moveit import trajectory_max_joint_step_deg
                    step = trajectory_max_joint_step_deg(state, trajectory)
                if step > self.args.max_ik_joint_step_deg:
                    raise ValueError(f"{name}: certified route exceeds the joint step limit")
                whole_evaluations += certificate["whole_robot_distance_evaluations"]
                joint_phases.append({"name": name, "active_ball": ball, "joints_rad": knots})
                maximum = max(maximum, step)
                trajectories.append(trajectory)
                state.joint_state.name = list(trajectory.joint_trajectory.joint_names)
                state.joint_state.position = list(trajectory.joint_trajectory.points[-1].positions)
                state.joint_state.velocity = []
                state.joint_state.effort = []
                current = np.asarray(target)
            self.assert_scene()
        finally:
            self.set_active(None)
        proof.update({"sphere_safety_version": SAFETY_VERSION, "sphere_safety_model_sha256": self.model_hash,
                      "sphere_safety_world_sha256": self.world_hash, "moveit_status": "OK",
                      "moveit_full_model_collision_checked": "1", "moveit_state_start_to_safe_checked": "1",
                      "moveit_probe_collision_checked": "1", "moveit_joint_continuity_ok": "1",
                      "moveit_max_trajectory_joint_step_deg": f"{maximum:.9f}",
                      "tool_sweep_clearance_evaluations": str(evaluations),
                      "whole_robot_distance_evaluations": str(whole_evaluations),
                      "moveit_whole_robot_continuous_ok": "1",
                      "moveit_self_collision_policy_audited": "1",
                      "sphere_motion_policy_sha256": digest(self.policy)})
        if entry:
            # An entry-only check must never inherit a full acquisition receipt.
            proof.update({"moveit_first_contact_ok": "0", "moveit_continuous_probe_sweep_ok": "0",
                          "moveit_return_path_checked": "0", "sphere_safety_path_sha256": ""})
        else:
            proof.update({key: "1" for key in PROOF_FLAGS})
            bundle = make_bundle(row, joint_phases, self.model_hash, self.world_hash, self.policy)
            raw = json.dumps(bundle, separators=(",", ":"), allow_nan=False)
            proof["verified_trajectory_json"] = raw
            proof["verified_trajectory_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            proof["sphere_safety_path_sha256"] = path_fingerprint({**row, **proof})
        if self.args.publish_display_trajectory:
            self.node.publish_display_trajectory(seed, trajectories)
        return proof, state
