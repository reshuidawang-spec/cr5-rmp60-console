"""Pure geometry for fixed-orientation sphere probing and continuous tool sweeps.

All distances are millimetres. Cylinders are bounded by capsules, making the
clearance proof conservative. Unresolved intervals are rejected, never accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

BRANCHES = ("z", "x_pos", "x_neg", "y_pos", "y_neg")
BALL_LINKS = {branch: f"rmp60_ball_{branch}" for branch in BRANCHES}
SAFETY_VERSION = "sphere_contact_v3"
PROOF_FLAGS = ("moveit_full_model_collision_checked", "moveit_first_contact_ok",
               "moveit_continuous_probe_sweep_ok", "moveit_return_path_checked",
               "moveit_state_start_to_safe_checked", "moveit_whole_robot_continuous_ok",
               "moveit_self_collision_policy_audited")


def requires_sphere_safety(row):
    return row.get("sphere_safety_required") == "1" or row.get("dataset_group") in ("smoke", "geometry", "anchor", "evaluation")


def path_fingerprint(row, *, bind_execution=True):
    keys = [*(f"{prefix}_flange_{axis}" for prefix in ("safe", "transition", "start", "target")
              for axis in ("x", "y", "z", "rx", "ry", "rz")),
            *(f"{prefix}_{axis}" for prefix in ("initial_offset", "approach", "sphere_center") for axis in "xyz"),
            "contact_center_distance_mm"]
    geometry = {key: float(row[key]) for key in keys}
    if not all(math.isfinite(value) for value in geometry.values()):
        raise ValueError("nonfinite plan geometry")
    geometry.update({key: row.get(key, "") for key in ("branch", "physical_ball_id", "standard_sphere_id", "plan_id", "verified_trajectory_sha256", "sphere_motion_policy_sha256", "safe_joint_seed_json", "transition_joint_seed_json")})
    if not bind_execution:
        geometry.pop("verified_trajectory_sha256", None)
        geometry.pop("sphere_motion_policy_sha256", None)
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def validate_safety_receipt(row):
    if row.get("sphere_safety_version") != SAFETY_VERSION or any(row.get(key) != "1" for key in PROOF_FLAGS):
        raise ValueError("路径缺少新版全测针碰撞/首次接触/返回检查；请重新预检并冻结")
    if row.get("moveit_status") != "OK" or row.get("sphere_safety_path_sha256") != path_fingerprint(row):
        raise ValueError("碰撞检查结果与当前路径不一致；请重新预检")
    from sphere_motion_contract import validate_bundle
    validate_bundle(row)
    for key in ("sphere_safety_model_sha256", "sphere_safety_world_sha256"):
        value = row.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("碰撞检查缺少模型/场景标识")


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def pose_transform(pose):
    return transform(pose[:3], np.radians(pose[3:]))


def point_segment_distance(point, a, b):
    delta = b - a
    t = float((point - a) @ delta) / float(delta @ delta) if delta @ delta > 1e-20 else 0.
    return float(np.linalg.norm(point - (a + np.clip(t, 0, 1) * delta)))


def segment_segment_distance(a, b, c, d):
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = float(u @ u), float(u @ v), float(v @ v), float(u @ w), float(v @ w)
    if aa < 1e-20:
        return point_segment_distance(a, c, d)
    if cc < 1e-20:
        return point_segment_distance(c, a, b)
    determinant = aa * cc - bb * bb
    s = float(np.clip((bb * ee - cc * dd) / determinant, 0, 1)) if determinant > 1e-20 else 0.
    t = (bb * s + ee) / cc
    if t < 0:
        t, s = 0., float(np.clip(-dd / aa, 0, 1))
    elif t > 1:
        t, s = 1., float(np.clip((bb - dd) / aa, 0, 1))
    return float(np.linalg.norm(w + s * u - t * v))


def segment_box_distance(a, b, center, rotation, half_size):
    """Exact segment/OBB distance from its piecewise quadratic objective."""
    a = rotation.T @ (a - center)
    b = rotation.T @ (b - center)
    delta = b - a
    cuts = [0., 1.]
    for axis in range(3):
        if abs(delta[axis]) > 1e-15:
            cuts.extend(float(t) for bound in (-half_size[axis], half_size[axis])
                        if 0 < (t := (bound - a[axis]) / delta[axis]) < 1)
    cuts = sorted(set(cuts))

    def squared(t):
        gap = np.maximum(np.abs(a + delta * t) - half_size, 0.)
        return float(gap @ gap)

    best = min(squared(t) for t in cuts)
    for left, right in zip(cuts, cuts[1:]):
        middle = a + delta * ((left + right) / 2)
        active = np.abs(middle) > half_size
        if not active.any():
            return 0.
        coefficient = delta[active]
        constant = a[active] - np.sign(middle[active]) * half_size[active]
        denominator = float(coefficient @ coefficient)
        if denominator > 1e-20:
            t = float(np.clip(-float(coefficient @ constant) / denominator, left, right))
            best = min(best, squared(t))
    return math.sqrt(max(0., best))


@dataclass
class Capsule:
    name: str
    a: np.ndarray
    b: np.ndarray
    radius: float

    def moved(self, pose, translation=None):
        rotation, origin = pose[:3, :3], pose[:3, 3]
        if translation is not None:
            origin = origin + translation
        return Capsule(self.name, rotation @ self.a + origin, rotation @ self.b + origin, self.radius)


@dataclass
class Box:
    name: str
    center: np.ndarray
    rotation: np.ndarray
    half_size: np.ndarray


@dataclass
class Plane:
    name: str
    normal: np.ndarray
    offset: float


def clearance(tool, obstacle):
    if isinstance(obstacle, Capsule):
        return segment_segment_distance(tool.a, tool.b, obstacle.a, obstacle.b) - tool.radius - obstacle.radius
    if isinstance(obstacle, Box):
        return segment_box_distance(tool.a, tool.b, obstacle.center, obstacle.rotation, obstacle.half_size) - tool.radius
    if isinstance(obstacle, Plane):
        # Both sides of an arbitrary plane are obstacles; crossing it is unsafe.
        da, db = float(tool.a @ obstacle.normal + obstacle.offset), float(tool.b @ obstacle.normal + obstacle.offset)
        return (0. if da * db <= 0 else min(abs(da), abs(db))) - tool.radius
    raise ValueError("unsupported obstacle geometry")


def certify_sweep(tool, start_pose, translation, obstacles, *, exempt_pair=None, margin_mm=.01, end_rotation=None):
    """Certify every instant by a Lipschitz bound, recursively refining near hits."""
    relative_rotation = (np.zeros(3) if end_rotation is None else
                         Rotation.from_matrix(start_pose[:3, :3].T @ end_rotation).as_rotvec())
    radius = max(float(np.linalg.norm(tool.a)), float(np.linalg.norm(tool.b))) + tool.radius
    travel = float(np.linalg.norm(translation)) + radius * float(np.linalg.norm(relative_rotation))
    checked = 0
    minimum_bound = math.inf
    for obstacle in obstacles:
        if exempt_pair == (tool.name, obstacle.name):
            continue
        stack = [(0., 1., 0)]
        while stack:
            left, right, depth = stack.pop()
            middle = (left + right) / 2
            placement = start_pose.copy()
            placement[:3, :3] = start_pose[:3, :3] @ Rotation.from_rotvec(relative_rotation * middle).as_matrix()
            gap = clearance(tool.moved(placement, translation * middle), obstacle)
            checked += 1
            if not math.isfinite(gap):
                raise ValueError("nonfinite collision clearance")
            if gap <= margin_mm:
                raise ValueError(f"continuous sweep collision/clearance: {tool.name} / {obstacle.name}, gap={gap:.6f} mm")
            lower_bound = gap - travel * (right - left) / 2
            if lower_bound > margin_mm:
                minimum_bound = min(minimum_bound, lower_bound)
                continue
            if depth >= 24 or travel * (right - left) < 1e-6:
                raise ValueError(f"continuous sweep cannot be certified: {tool.name} / {obstacle.name}")
            stack.extend(((left, middle, depth + 1), (middle, right, depth + 1)))
    return {"clearance_evaluations": checked, "certified_min_clearance_mm": minimum_bound}


def certify_tool_motion(model, start, target, obstacles, *, active_ball=None):
    placement, end = pose_transform(start), pose_transform(target)
    evaluations = 0
    for part in model.primitives:
        proof = certify_sweep(part, placement, end[:3, 3] - placement[:3, 3], obstacles,
                              exempt_pair=(active_ball, "calibration_sphere"), end_rotation=end[:3, :3])
        evaluations += proof["clearance_evaluations"]
    return evaluations


class ToolModel:
    def __init__(self, xml):
        root = ET.fromstring(xml)
        links = {link.attrib["name"]: link for link in root.findall("link")}
        self.robot_collision_links = [name for name, link in links.items() if link.findall("collision")]
        # MoveIt returns world objects in its root frame (dummy_link on CR5).
        # Resolve only frames rigidly connected to base_link from the live URDF.
        self.base_frames = {"base_link": np.eye(4)}
        fixed = [joint for joint in root.findall("joint") if joint.attrib["type"] == "fixed"]
        for _ in range(len(fixed) + 1):
            changed = False
            for joint in fixed:
                parent, child = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
                origin = joint.find("origin")
                xyz = [float(v) * 1000 for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
                rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
                relative = transform(xyz, rpy)
                if parent in self.base_frames and child not in self.base_frames:
                    self.base_frames[child] = self.base_frames[parent] @ relative
                    changed = True
                elif child in self.base_frames and parent not in self.base_frames:
                    self.base_frames[parent] = self.base_frames[child] @ np.linalg.inv(relative)
                    changed = True
            if not changed:
                break
        transforms = {"Link6": np.eye(4)}
        pending = list(root.findall("joint"))
        while pending:
            remaining = []
            for joint in pending:
                parent = joint.find("parent").attrib["link"]
                if parent not in transforms:
                    remaining.append(joint)
                    continue
                if joint.attrib["type"] != "fixed":
                    raise ValueError("tool descendants of Link6 must be rigid")
                child = joint.find("child").attrib["link"]
                origin = joint.find("origin")
                xyz = np.array([float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]) * 1000 if origin is not None else [0, 0, 0]
                rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
                transforms[child] = transforms[parent] @ transform(xyz, rpy)
            if len(remaining) == len(pending):
                break
            pending = remaining
        self.primitives, self.balls = [], {}
        for name, placement in transforms.items():
            if name == "Link6":
                continue
            for collision in links[name].findall("collision"):
                origin = collision.find("origin")
                xyz = [float(v) * 1000 for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
                rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
                local = placement @ transform(xyz, rpy)
                geometry = collision.find("geometry")
                sphere, cylinder, box = geometry.find("sphere"), geometry.find("cylinder"), geometry.find("box")
                center = local[:3, 3]
                if sphere is not None:
                    primitive = Capsule(name, center, center, float(sphere.attrib["radius"]) * 1000)
                elif cylinder is not None:
                    half = float(cylinder.attrib["length"]) * 500
                    axis = local[:3, 2] * half
                    primitive = Capsule(name, center - axis, center + axis, float(cylinder.attrib["radius"]) * 1000)
                elif box is not None:
                    size = np.array([float(v) * 1000 for v in box.attrib["size"].split()])
                    primitive = Capsule(name, center, center, float(np.linalg.norm(size) / 2))
                else:
                    raise ValueError(f"unsupported tool collision primitive: {name}")
                if (not np.isfinite(np.r_[primitive.a, primitive.b, primitive.radius]).all()
                        or primitive.radius <= 0
                        or cylinder is not None and float(cylinder.attrib["length"]) <= 0
                        or box is not None and np.any(size <= 0)):
                    raise ValueError(f"invalid tool collision dimensions: {name}")
                self.primitives.append(primitive)
        for branch, name in BALL_LINKS.items():
            found = [part for part in self.primitives if part.name == name]
            if (len(found) != 1 or not np.allclose(found[0].a, found[0].b)
                    or links[name].find("collision/geometry/sphere") is None):
                raise ValueError(f"robot model requires separate sphere-only link {name}; restart MoveIt with updated URDF")
            self.balls[branch] = found[0]
        if not all(any(p.name == name for p in self.primitives) for name in
                   ["rmp60_adapter", "rmp60_probe_body", "rmp60_stylus", *[f"rmp60_branch_{b}" for b in BRANCHES[1:]]]):
            raise ValueError("tool collision model is incomplete")


def first_sphere_entry(point, direction, center, radius):
    relative = point - center
    if float(relative @ relative) <= radius ** 2:
        return 0.
    projection = float(relative @ direction)
    discriminant = projection ** 2 - float(relative @ relative) + radius ** 2
    if discriminant < 0:
        return math.inf
    return min((v for v in (-projection - math.sqrt(discriminant), -projection + math.sqrt(discriminant)) if v >= 0), default=math.inf)


def validate_probe(row, model, obstacles, sphere_center, sphere_radius, *, minimum_first_margin_mm=1.):
    start = np.array([float(row[f"start_flange_{a}"]) for a in ("x", "y", "z", "rx", "ry", "rz")])
    target = np.array([float(row[f"target_flange_{a}"]) for a in ("x", "y", "z", "rx", "ry", "rz")])
    if not np.isfinite(np.r_[start, target, sphere_center]).all():
        raise ValueError("nonfinite probing geometry")
    placement = pose_transform(start)
    if not np.allclose(placement[:3, :3], pose_transform(target)[:3, :3], atol=1e-8):
        raise ValueError("continuous sphere probing requires a fixed flange orientation")
    delta = target[:3] - start[:3]
    travel = float(np.linalg.norm(delta))
    if not 0 < travel <= 1.0001:
        raise ValueError("sphere probing travel must be within 1 mm")
    branch = row.get("branch")
    if branch not in model.balls:
        raise ValueError("unknown physical probe branch")
    expected_distance = sphere_radius + model.balls[branch].radius
    distance = float(row.get("contact_center_distance_mm", math.nan))
    if not math.isfinite(distance) or abs(distance - expected_distance) > 1e-5:
        raise ValueError("planned radii differ from the loaded collision model")
    embedded_center = np.array([float(row[f"sphere_center_{a}"]) for a in "xyz"])
    if not np.isfinite(embedded_center).all() or np.linalg.norm(embedded_center - sphere_center) > .001:
        raise ValueError("planned sphere centre differs from the loaded scene")
    offset = np.array([float(row[f"initial_offset_{a}"]) for a in "xyz"])
    if not np.isfinite(offset).all() or np.linalg.norm(offset - model.balls[branch].a) > .01:
        raise ValueError("planned stylus offset differs from the loaded MoveIt model; restart with current branch offsets")
    direction = delta / travel
    approach = np.array([float(row[f"approach_{a}"]) for a in "xyz"])
    if not np.isfinite(approach).all() or np.linalg.norm(approach - direction) > 3e-4:
        raise ValueError("planned approach disagrees with the search line")
    entries = {b: first_sphere_entry(part.moved(placement).a, direction, sphere_center, sphere_radius + part.radius)
               for b, part in model.balls.items()}
    order = sorted(entries, key=entries.get)
    first, second = entries[order[0]], entries[order[1]]
    if order[0] != branch or not 0 < first < travel:
        raise ValueError(f"expected {branch}, but first contact is {order[0]} at {first:.6f} mm")
    if second - first < minimum_first_margin_mm:
        raise ValueError("first/second stylus contact margin is insufficient")
    evaluations = 0
    for part in model.primitives:
        proof = certify_sweep(part, placement, delta, obstacles,
                              exempt_pair=(BALL_LINKS[branch], "calibration_sphere"))
        evaluations += proof["clearance_evaluations"]
    return {"sphere_safety_version": SAFETY_VERSION, "moveit_first_contact_ok": "1",
            "predicted_first_contact_branch": branch, "predicted_first_contact_distance_mm": f"{first:.9f}",
            "first_to_second_contact_margin_mm": f"{second-first:.9f}" if math.isfinite(second) else "inf",
            "moveit_continuous_probe_sweep_ok": "1", "probe_sweep_clearance_evaluations": str(evaluations)}
