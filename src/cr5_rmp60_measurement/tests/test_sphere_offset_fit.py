#!/usr/bin/env python3
"""Test fixed-sphere offset fitting and residual calculations."""

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from sphere_offset_fit import evaluate_fixed_center_sphere_offset, solve_fixed_center_sphere_offset


def sphere_normals():
    normals = []
    for polar_deg in (20.0, 45.0, 70.0):
        polar = np.radians(polar_deg)
        for azimuth_deg in range(0, 360, 45):
            azimuth = np.radians(azimuth_deg)
            normals.append(
                [
                    np.sin(polar) * np.cos(azimuth),
                    np.sin(polar) * np.sin(azimuth),
                    np.cos(polar),
                ]
            )
    return np.asarray(normals, dtype=float)


class FixedCenterSphereOffsetTest(unittest.TestCase):
    def test_frozen_evaluation_does_not_refit_offset(self):
        wrong_offset = self.true_offset + np.asarray([0.25, -0.1, 0.05])
        result = evaluate_fixed_center_sphere_offset(
            self.positions, self.rotations, self.sphere_center, self.contact_distance, wrong_offset
        )

        self.assertEqual(result["jacobian_rank"], 3)
        self.assertGreater(np.max(np.abs(result["signed_residuals"])), 0.01)

    def setUp(self):
        self.sphere_center = np.asarray([125.0, -42.0, 87.0], dtype=float)
        self.contact_distance = 11.0
        self.true_offset = np.asarray([4.2, -3.7, 178.5], dtype=float)
        self.normals = sphere_normals()
        self.rotations = np.repeat(np.eye(3)[None, :, :], len(self.normals), axis=0)
        ball_centers = self.sphere_center + self.contact_distance * self.normals
        self.positions = ball_centers - self.true_offset

    def solve(self, positions=None, normals=None, **overrides):
        positions = self.positions if positions is None else positions
        count = len(positions)
        options = {
            "huber_delta_mm": 0.2,
            "max_iterations": 80,
            "step_tolerance_mm": 1e-11,
            "max_condition": 1e5,
            "outlier_rejection": True,
            "outlier_threshold_mm": 0.3,
            "outlier_min_samples": 4,
            "outlier_min_inliers": 3,
        }
        options.update(overrides)
        initial = self.true_offset + np.asarray([0.8, -0.6, 0.5])
        return solve_fixed_center_sphere_offset(
            positions,
            self.rotations[:count],
            self.sphere_center,
            self.contact_distance,
            initial,
            **options,
        )

    def test_recovers_offset_without_assuming_commanded_approach_is_surface_normal(self):
        result = self.solve()

        self.assertTrue(result["ok"], result.get("reason"))
        np.testing.assert_allclose(result["estimate"], self.true_offset, atol=1e-8)
        self.assertEqual(result["jacobian_rank"], 3)
        self.assertLess(result["jacobian_condition"], 10.0)
        self.assertLess(np.max(np.abs(result["signed_residuals"])), 1e-9)

    def test_rejects_gross_radial_trigger_outlier(self):
        positions = self.positions.copy()
        positions[5] += 2.0 * self.normals[5]

        result = self.solve(positions=positions)

        self.assertTrue(result["ok"], result.get("reason"))
        self.assertEqual(result["rejected_rows"], 1)
        self.assertFalse(bool(result["inlier_mask"][5]))
        np.testing.assert_allclose(result["estimate"], self.true_offset, atol=1e-7)

    def test_rejects_rank_deficient_contact_set(self):
        normal = np.asarray([0.0, 0.0, 1.0])
        positions = np.repeat(
            (self.sphere_center + self.contact_distance * normal - self.true_offset)[None, :],
            6,
            axis=0,
        )

        result = self.solve(positions=positions)

        self.assertFalse(result["ok"])
        self.assertIn("rank", result["reason"])


if __name__ == "__main__":
    unittest.main()
