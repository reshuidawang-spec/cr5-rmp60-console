#!/usr/bin/env python3
"""Test collision-aware safe-exit pose selection for sphere measurements."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from model_based_sphere_point_planner import safe_pose_for_candidate


class SafeExitLogicTest(unittest.TestCase):
    def test_radial_exit_retreats_opposite_approach_without_rotating(self):
        args = SimpleNamespace(radial_exit_mm=15.0, x_positive_exit_mm=0.0, safe_pose=None)
        transition = [-448.0, 126.0, 274.0, -179.8, -0.7, 110.5]
        approach = [0.6, -0.8, 0.0]

        safe, mode = safe_pose_for_candidate(args, transition, approach)

        self.assertEqual(mode, "radial_outward")
        self.assertEqual(safe[3:], transition[3:])
        self.assertAlmostEqual(safe[0], transition[0] - 9.0)
        self.assertAlmostEqual(safe[1], transition[1] + 12.0)
        self.assertAlmostEqual(safe[2], transition[2])

    def test_fixed_x_and_radial_exit_cannot_be_combined(self):
        args = SimpleNamespace(radial_exit_mm=15.0, x_positive_exit_mm=15.0, safe_pose=None)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            safe_pose_for_candidate(args, [0.0] * 6, [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
