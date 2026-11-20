"""
Unit tests for control/kinematics.py (Student 1).
Run from the project folder:  python3 -m unittest discover tests
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control"))
import kinematics as kin


class TestKinematics(unittest.TestCase):
    def test_ik_then_fk_returns_same_point(self):
        """IK -> FK must give back the target (100 random reachable points)."""
        rng = random.Random(0)
        n = 0
        while n < 100:
            x, y, z = rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55), rng.uniform(0.0, 0.12)
            sols = kin.ik_all(x, y, z)
            for q in sols:
                self.assertLess(math.dist((x, y, z), kin.fk(*q)), 1e-9)
            n += bool(sols)

    def test_two_elbow_solutions(self):
        sols = kin.ik_all(0.40, 0.10, 0.05)
        self.assertEqual(len(sols), 2)
        self.assertNotAlmostEqual(sols[0][1], sols[1][1])

    def test_unreachable_point_raises(self):
        with self.assertRaises(ValueError):
            kin.ik(0.60, 0.0, 0.05)          # farther than L1 + L2 = 0.55 m
        with self.assertRaises(ValueError):
            kin.ik(0.02, 0.0, 0.05)          # closer than |L1 - L2| = 0.05 m

    def test_closest_solution_is_chosen(self):
        up, down = sorted(kin.ik_all(0.40, 0.10, 0.05), key=lambda q: q[1])
        self.assertEqual(kin.ik(0.40, 0.10, 0.05, current=up), up)
        self.assertEqual(kin.ik(0.40, 0.10, 0.05, current=down), down)


if __name__ == "__main__":
    unittest.main()
