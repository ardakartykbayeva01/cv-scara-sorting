"""
calibrate.py  (Student 2, step 5)
Automatic camera calibration in CoppeliaSim.

  1. The robot goes to HOME (outside the camera view); green and blue cubes are moved away.
  2. The red cube is put on a 5 x 5 grid over the mat (known positions).
     At each point we take a frame and detect the cube -> pairs (u, v) <-> (x, y).
  3. A homography H is fitted (calibration.json).
  4. Validation: 20 NEW random positions (not used for fitting) -> error in mm.

How to run (CoppeliaSim open with cv_scara_scene.ttt, simulation STOPPED):
  cd vision
  python3 calibrate.py
"""

import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control"))

import numpy as np

from robot_control import Robot
from camera import grab_frame
from detect import detect_objects
from calibration import fit_homography, apply_homography, save_calibration

CUBE_Z = 0.03                         # cube centre height on the mat
AWAY = {"green": [-0.60, 0.50, CUBE_Z], "blue": [-0.60, -0.50, CUBE_Z]}
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

robot = Robot()
sim = robot.sim
cam = sim.getObject('/Camera')
red = robot.cubes["red"]
start_pos = {c: sim.getObjectPosition(h, -1) for c, h in robot.cubes.items()}


def measure(x, y):
    """Put the red cube at (x, y), take a frame and return its pixel centre (or None)."""
    sim.setObjectPosition(red, -1, [x, y, CUBE_Z])
    robot._step(); robot._step()               # let the camera render the new scene
    found = [o for o in detect_objects(grab_frame(sim, cam)) if o["color"] == "red"]
    return found[0]["center_px"] if len(found) == 1 else None


robot.start()
try:
    robot.home()
    for c, p in AWAY.items():
        sim.setObjectPosition(robot.cubes[c], -1, p)

    # ---- 1. Calibration grid (5 x 5) ----
    pixels, points = [], []
    for x in np.linspace(0.20, 0.50, 5):
        for y in np.linspace(-0.25, 0.25, 5):
            uv = measure(x, y)
            if uv is None:
                print(f"  ({x:.3f}, {y:.3f}): cube not detected, skipped")
                continue
            pixels.append(uv)
            points.append([x, y])
    print(f"Calibration points: {len(pixels)}")
    H = fit_homography(pixels, points)
    fit_err = [1000 * math.dist(apply_homography(H, *uv), p) for uv, p in zip(pixels, points)]

    # ---- 2. Validation on new random positions ----
    random.seed(1)
    rows, val_err = [], []
    for _ in range(20):
        x, y = random.uniform(0.20, 0.50), random.uniform(-0.25, 0.25)
        uv = measure(x, y)
        if uv is None:
            continue
        ex, ey = apply_homography(H, *uv)
        e = 1000 * math.hypot(ex - x, ey - y)
        val_err.append(e)
        rows.append([f"{x:.4f}", f"{y:.4f}", f"{uv[0]:.2f}", f"{uv[1]:.2f}",
                     f"{ex:.4f}", f"{ey:.4f}", f"{e:.2f}"])
finally:
    robot.stop()
    for c, h in robot.cubes.items():          # put the scene back as it was
        sim.setObjectPosition(h, -1, start_pos[c])

info = {"n_points": len(pixels),
        "fit_mean_mm": float(np.mean(fit_err)), "fit_max_mm": float(np.max(fit_err)),
        "val_n": len(val_err),
        "val_mean_mm": float(np.mean(val_err)), "val_max_mm": float(np.max(val_err))}
save_calibration(H, info)
with open(os.path.join(RESULTS, "calibration_validation.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["true_x", "true_y", "u", "v", "est_x", "est_y", "error_mm"])
    w.writerows(rows)

print("H =\n", np.array2string(H, precision=5))
print(f"Fit error:        mean {info['fit_mean_mm']:.2f} mm, max {info['fit_max_mm']:.2f} mm")
print(f"Validation error: mean {info['val_mean_mm']:.2f} mm, max {info['val_max_mm']:.2f} mm "
      f"({info['val_n']} new positions)")
print("Saved: vision/calibration.json, results/calibration_validation.csv")
