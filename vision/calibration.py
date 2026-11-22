"""
calibration.py  (Student 2)
Pixel -> robot coordinates with a homography.

The cubes always lie on the flat work mat, so their top faces are in one plane
(z = 0.05 m). A plane seen by a camera is mapped to the image by a 3x3
homography H. We estimate H from pairs (pixel, known robot position) and then use
                [x]       [u]
                [y] ~ H * [v]
                [1]       [1]
"""

import json
import os

import cv2
import numpy as np

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
_H = None


def fit_homography(pixels, points_xy):
    """Least-squares homography from >= 4 pairs (u, v) -> (x, y)."""
    H, _ = cv2.findHomography(np.array(pixels, dtype=np.float64),
                              np.array(points_xy, dtype=np.float64), 0)
    return H


def apply_homography(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def save_calibration(H, info):
    with open(CALIB_FILE, "w") as f:
        json.dump({"H": H.tolist(), **info}, f, indent=2)


def pixel_to_robot(u, v):
    """Pixel centre of a cube -> (x, y) in the robot base frame (metres)."""
    global _H
    if _H is None:
        if not os.path.exists(CALIB_FILE):
            raise FileNotFoundError("calibration.json not found - run calibrate.py first")
        with open(CALIB_FILE) as f:
            _H = np.array(json.load(f)["H"])
    return apply_homography(_H, u, v)
