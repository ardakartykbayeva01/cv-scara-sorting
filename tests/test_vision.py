"""
Unit tests for vision/detect.py and vision/calibration.py (Student 2).
They use synthetic images, so CoppeliaSim is not needed.
Run from the project folder:  python3 -m unittest discover tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision"))
import cv2
import numpy as np

from detect import detect_objects
from calibration import fit_homography, apply_homography

CUBES = {"red": ((100, 120), (0, 0, 220)), "green": ((300, 200), (0, 200, 0)),
         "blue": ((200, 400), (220, 0, 0))}          # centre (u, v), colour (B, G, R)


def synthetic_frame():
    img = np.full((512, 512, 3), 235, np.uint8)       # white-grey background
    for (u, v), bgr in CUBES.values():
        cv2.rectangle(img, (u - 13, v - 13), (u + 13, v + 13), bgr, -1)
    return img


class TestDetect(unittest.TestCase):
    def test_finds_three_cubes_with_correct_centres(self):
        objs = detect_objects(synthetic_frame())
        self.assertEqual(sorted(o["color"] for o in objs), ["blue", "green", "red"])
        for o in objs:
            u, v = CUBES[o["color"]][0]
            self.assertAlmostEqual(o["center_px"][0], u, delta=1.0)
            self.assertAlmostEqual(o["center_px"][1], v, delta=1.0)

    def test_empty_scene_gives_empty_list(self):
        self.assertEqual(detect_objects(np.full((512, 512, 3), 235, np.uint8)), [])

    def test_small_blob_is_ignored(self):
        img = np.full((512, 512, 3), 235, np.uint8)
        cv2.rectangle(img, (50, 50), (53, 53), (0, 0, 220), -1)     # 4x4 px < MIN_AREA_PX
        self.assertEqual(detect_objects(img), [])

    def test_blur_helps_under_strong_noise(self):
        rng = np.random.default_rng(0)
        noisy = np.clip(synthetic_frame() + rng.normal(0, 100, (512, 512, 3)), 0, 255).astype(np.uint8)
        self.assertEqual(len(detect_objects(noisy, blur=5)), 3)


class TestCalibration(unittest.TestCase):
    def test_homography_recovers_known_mapping(self):
        H_true = np.array([[-0.0015, 0.0, 0.74], [0.0, 0.0015, -0.39], [0.0, 0.0, 1.0]])
        pix = [(u, v) for u in range(100, 450, 70) for v in range(100, 450, 70)]
        pts = [apply_homography(H_true, u, v) for u, v in pix]
        H = fit_homography(pix, pts)
        x, y = apply_homography(H, 256, 256)
        xt, yt = apply_homography(H_true, 256, 256)
        self.assertLess(np.hypot(x - xt, y - yt), 1e-6)


if __name__ == "__main__":
    unittest.main()
