#!/usr/bin/env python3
"""Robust fixed-center sphere fitting for a stylus ball offset."""

from __future__ import annotations

import math

import numpy as np


def _evaluate(
    positions: np.ndarray,
    rotations: np.ndarray,
    sphere_center: np.ndarray,
    contact_distance: float,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ball_centers = positions + np.einsum("nij,j->ni", rotations, offset)
    sphere_to_ball = ball_centers - sphere_center[None, :]
    distances = np.linalg.norm(sphere_to_ball, axis=1)
    if np.any(distances <= 1e-12):
        raise ValueError("a fitted stylus ball center coincides with the standard sphere center")
    outward_normals = sphere_to_ball / distances[:, None]
    signed_residuals = distances - float(contact_distance)
    jacobian = np.einsum("ni,nij->nj", outward_normals, rotations)
    return signed_residuals, outward_normals, jacobian, ball_centers


def _rank_and_condition(jacobian: np.ndarray) -> tuple[int, float]:
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = float(np.linalg.cond(jacobian)) if rank == 3 else math.inf
    return rank, condition


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    weights = np.ones_like(residuals, dtype=float)
    large = np.abs(residuals) > float(delta)
    weights[large] = float(delta) / np.abs(residuals[large])
    return weights


def _irls_gauss_newton(
    positions: np.ndarray,
    rotations: np.ndarray,
    sphere_center: np.ndarray,
    contact_distance: float,
    initial_offset: np.ndarray,
    active_mask: np.ndarray,
    *,
    huber_delta_mm: float,
    max_iterations: int,
    step_tolerance_mm: float,
) -> dict[str, object]:
    estimate = np.asarray(initial_offset, dtype=float).copy()
    active_indices = np.flatnonzero(active_mask)
    if len(active_indices) < 3:
        return {"ok": False, "reason": "fewer than three active sphere contacts"}

    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        residuals, _, jacobian, _ = _evaluate(
            positions,
            rotations,
            sphere_center,
            contact_distance,
            estimate,
        )
        active_jacobian = jacobian[active_indices]
        rank, _ = _rank_and_condition(active_jacobian)
        if rank < 3:
            return {
                "ok": False,
                "reason": f"sphere offset Jacobian rank {rank} < 3",
                "iterations": iterations,
                "jacobian_rank": rank,
                "jacobian_condition": math.inf,
            }
        active_residuals = residuals[active_indices]
        weights = _huber_weights(active_residuals, huber_delta_mm)
        sqrt_weights = np.sqrt(weights)
        weighted_jacobian = active_jacobian * sqrt_weights[:, None]
        weighted_residuals = active_residuals * sqrt_weights
        step, _, _, _ = np.linalg.lstsq(weighted_jacobian, -weighted_residuals, rcond=None)
        estimate += step
        if float(np.linalg.norm(step)) <= float(step_tolerance_mm):
            converged = True
            break

    residuals, normals, jacobian, ball_centers = _evaluate(
        positions,
        rotations,
        sphere_center,
        contact_distance,
        estimate,
    )
    rank, condition = _rank_and_condition(jacobian[active_indices])
    return {
        "ok": True,
        "estimate": estimate,
        "signed_residuals": residuals,
        "outward_normals": normals,
        "jacobian": jacobian,
        "ball_centers": ball_centers,
        "iterations": iterations,
        "converged": converged,
        "jacobian_rank": rank,
        "jacobian_condition": condition,
    }


def solve_fixed_center_sphere_offset(
    positions,
    rotations,
    sphere_center,
    contact_distance,
    initial_offset,
    *,
    huber_delta_mm=0.2,
    max_iterations=50,
    step_tolerance_mm=1e-9,
    max_condition=1e6,
    outlier_rejection=True,
    outlier_threshold_mm=0.6,
    outlier_min_samples=4,
    outlier_min_inliers=3,
):
    """Fit ``||P + R o - S|| = d`` and report observability diagnostics."""
    positions = np.asarray(positions, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    sphere_center = np.asarray(sphere_center, dtype=float)
    initial_offset = np.asarray(initial_offset, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if rotations.shape != (len(positions), 3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    if sphere_center.shape != (3,) or initial_offset.shape != (3,):
        raise ValueError("sphere_center and initial_offset must have shape (3,)")
    if len(positions) < 3:
        return {"ok": False, "reason": "sphere offset fit requires at least three contacts"}
    if contact_distance <= 0 or huber_delta_mm <= 0:
        raise ValueError("contact_distance and huber_delta_mm must be positive")

    all_mask = np.ones(len(positions), dtype=bool)
    first = _irls_gauss_newton(
        positions,
        rotations,
        sphere_center,
        contact_distance,
        initial_offset,
        all_mask,
        huber_delta_mm=huber_delta_mm,
        max_iterations=max_iterations,
        step_tolerance_mm=step_tolerance_mm,
    )
    if not first.get("ok"):
        return first

    inlier_mask = all_mask
    if outlier_rejection and len(positions) >= int(outlier_min_samples):
        candidate_mask = np.abs(first["signed_residuals"]) <= float(outlier_threshold_mm)
        if int(np.count_nonzero(candidate_mask)) >= int(outlier_min_inliers):
            inlier_mask = candidate_mask

    final = first
    if not np.all(inlier_mask):
        final = _irls_gauss_newton(
            positions,
            rotations,
            sphere_center,
            contact_distance,
            first["estimate"],
            inlier_mask,
            huber_delta_mm=huber_delta_mm,
            max_iterations=max_iterations,
            step_tolerance_mm=step_tolerance_mm,
        )
        if not final.get("ok"):
            return final

    final["inlier_mask"] = inlier_mask
    final["rejected_rows"] = int(len(positions) - np.count_nonzero(inlier_mask))
    if not final["converged"]:
        final["ok"] = False
        final["reason"] = f"sphere offset fit did not converge in {max_iterations} iterations"
    elif final["jacobian_condition"] > float(max_condition):
        final["ok"] = False
        final["reason"] = (
            f"sphere offset Jacobian condition {final['jacobian_condition']:.6g} "
            f"> {float(max_condition):.6g}"
        )
    return final


def evaluate_fixed_center_sphere_offset(
    positions, rotations, sphere_center, contact_distance, offset
):
    """Evaluate a frozen offset without fitting or changing any observations."""
    positions = np.asarray(positions, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    sphere_center = np.asarray(sphere_center, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if rotations.shape != (len(positions), 3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    residuals, normals, jacobian, ball_centers = _evaluate(
        positions, rotations, sphere_center, contact_distance, offset
    )
    rank, condition = _rank_and_condition(jacobian)
    return {
        "signed_residuals": residuals,
        "outward_normals": normals,
        "jacobian": jacobian,
        "ball_centers": ball_centers,
        "jacobian_rank": rank,
        "jacobian_condition": condition,
    }
