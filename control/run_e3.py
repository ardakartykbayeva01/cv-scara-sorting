"""
run_e3.py  (experiment E3: can Gaussian blur fix the noise problem?)

E2 showed that recall drops when sigma >= 50, while precision stays high:
noise lowers the saturation of single pixels, the cube mask breaks into small
pieces and the area filter removes them. Idea: smooth the image with a Gaussian
filter BEFORE the HSV threshold, so each pixel becomes an average of its neighbours.

Same 30 layouts as E1/E2 (same random seed). For every layout and every sigma the
SAME noisy frame is tested with each blur kernel (paired comparison).

How to run (simulation STOPPED, calibration done):
  python3 run_e3.py
About 1-2 minutes. Results go to ../results/.
"""

import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision"))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_control import Robot, GRASP_TOL
import kinematics as kin
from camera import grab_frame
from detect import detect_objects, draw_detections
from calibration import pixel_to_robot

N_TRIALS = 30
SIGMAS = [0, 40, 60, 80, 100, 120, 150]
KERNELS = [0, 3, 5, 7]          # 0 = no blur (the E2 detector)
MATCH_MM = 20
MIN_GAP = 0.09
CUBE_Z = 0.03
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RES, exist_ok=True)


def random_layout(rng):
    """The same layout generator as in run_experiments.py."""
    while True:
        pts = [(rng.uniform(0.20, 0.50), rng.uniform(-0.25, 0.25)) for _ in range(3)]
        far = all(math.dist(a, b) >= MIN_GAP for i, a in enumerate(pts) for b in pts[i + 1:])
        reach = all(kin.ik_all(x, y, 0.05) for x, y in pts)
        if far and reach:
            return dict(zip(["red", "green", "blue"], pts))


def evaluate(objects, truth):
    est, fp = {}, 0
    for o in objects:
        x, y = pixel_to_robot(*o["center_px"])
        tx, ty = truth[o["color"]]
        err = 1000 * math.hypot(x - tx, y - ty)
        if err <= MATCH_MM and o["color"] not in est:
            est[o["color"]] = err
        else:
            fp += 1
    return len(est), fp, len(truth) - len(est), list(est.values())


robot = Robot()
sim = robot.sim
cam = sim.getObject('/Camera')
start_pos = {c: sim.getObjectPosition(h, -1) for c, h in robot.cubes.items()}

rng = random.Random(42)                      # same seed -> same 30 layouts as E1/E2
layouts = [random_layout(rng) for _ in range(N_TRIALS)]
rng_np = np.random.default_rng(7)
rows = []
example = None

robot.start()
try:
    robot.home()
    for k, layout in enumerate(layouts, 1):
        for c, h in robot.cubes.items():
            sim.setObjectParent(h, -1, True)
            sim.setObjectPosition(h, -1, [*layout[c], CUBE_Z])
        robot._step(); robot._step()
        frame = grab_frame(sim, cam)
        for s in SIGMAS:
            noisy = frame if s == 0 else np.clip(
                frame.astype(np.float32) + rng_np.normal(0, s, frame.shape), 0, 255).astype(np.uint8)
            if k == 1 and s == 100:
                example = noisy
            for kern in KERNELS:
                tp, fp, fn, errs = evaluate(detect_objects(noisy, blur=kern), layout)
                graspable = sum(e <= 1000 * GRASP_TOL for e in errs)
                rows.append([k, s, kern, tp, fp, fn,
                             f"{np.mean(errs):.2f}" if errs else "", graspable])
        print(f"  layout {k:2d}/{N_TRIALS} done")
finally:
    robot.stop()
    for c, h in robot.cubes.items():
        sim.setObjectPosition(h, -1, start_pos[c])

with open(os.path.join(RES, "e3_blur.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "sigma", "blur_kernel", "TP", "FP", "FN",
                "mean_localization_error_mm", "graspable"])
    w.writerows(rows)

# ---------------- Summary table ----------------
summ = {}
lines = ["E3: Gaussian blur vs noise",
         "sigma | kernel | precision % | recall % | loc. error mm | graspable %"]
for s in SIGMAS:
    for kern in KERNELS:
        r = [x for x in rows if x[1] == s and x[2] == kern]
        TP, FP, FN = (sum(x[i] for x in r) for i in (3, 4, 5))
        errs = [float(x[6]) for x in r if x[6] != ""]
        prec = f"{100 * TP / (TP + FP):6.1f}" if TP + FP else "   n/a"
        rec = 100 * TP / (TP + FN)
        err = np.mean(errs) if errs else float("nan")
        grasp = 100 * sum(x[7] for x in r) / (3 * len(r))
        summ[(s, kern)] = (rec, err)
        lines.append(f"  {s:3d} |   {kern}x{kern} | {prec} | {rec:6.1f} | {err:6.2f} | {grasp:6.1f}"
                     .replace("0x0", "off"))
text = "\n".join(lines)
with open(os.path.join(RES, "e3_summary.txt"), "w") as f:
    f.write(text + "\n")
print(text)

# ---------------- Plots ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for kern, m in zip(KERNELS, "osD^"):
    lab = "no blur" if kern == 0 else f"blur {kern}x{kern}"
    ax[0].plot(SIGMAS, [summ[(s, kern)][0] for s in SIGMAS], m + "-", label=lab)
    ax[1].plot(SIGMAS, [summ[(s, kern)][1] for s in SIGMAS], m + "-", label=lab)
ax[0].set_xlabel("noise sigma (grey levels)"); ax[0].set_ylabel("recall (%)"); ax[0].set_ylim(0, 105)
ax[0].set_title("E3: recall with and without blur"); ax[0].grid(alpha=0.3); ax[0].legend()
ax[1].set_xlabel("noise sigma (grey levels)"); ax[1].set_ylabel("mean localization error (mm)")
ax[1].set_title("E3: localization error (detected cubes)"); ax[1].grid(alpha=0.3); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(RES, "e3_blur.png"), dpi=150)

# Side-by-side example at sigma = 100: no blur vs blur 5x5
a = draw_detections(example, detect_objects(example, blur=0))
b = draw_detections(example, detect_objects(example, blur=5))
cv2.putText(a, "sigma=100, no blur", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(b, "sigma=100, blur 5x5", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.imwrite(os.path.join(RES, "e3_example_sigma100.png"), np.hstack([a, b]))
print("Saved in results/: e3_blur.csv, e3_summary.txt, e3_blur.png, e3_example_sigma100.png")
