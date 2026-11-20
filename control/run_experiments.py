"""
run_experiments.py  (step 10: experiments for the report)

E1  30 trials, random cube positions, full vision-based sorting:
    camera -> detect_objects -> pixel_to_robot -> IK -> pick and place.
    Metrics: detection precision/recall, localization error (mm),
             grasp success, sorting success, placement error, cycle time.
E2  Noise robustness: the same 30 layouts, Gaussian image noise
    sigma = 0, 20, 40, 50, 60, 70, 80, 100 (grey levels; 0-255 scale) is added to the camera frame.
    Metrics: precision, recall, localization error, and the share of cubes whose
    estimated position is inside the suction tolerance (15 mm), i.e. graspable.

How to run (CoppeliaSim open with cv_scara_scene.ttt, simulation STOPPED,
vision/calibrate.py already done):
  python3 run_experiments.py
It takes several minutes. Results go to ../results/.
"""

import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_control import Robot, GRASP_TOL
import kinematics as kin
from camera import grab_frame
from detect import detect_objects
from calibration import pixel_to_robot

N_TRIALS = 30
SIGMAS = [0, 20, 40, 50, 60, 70, 80, 100]
MATCH_MM = 20          # a detection is correct if its colour is right and it is within 20 mm
MIN_GAP = 0.09         # minimum distance between cube centres (m)
CUBE_Z = 0.03
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RES, exist_ok=True)


def random_layout(rng):
    """Three non-overlapping, reachable cube positions on the mat, inside the camera view."""
    while True:
        pts = [(rng.uniform(0.20, 0.50), rng.uniform(-0.25, 0.25)) for _ in range(3)]
        far = all(math.dist(a, b) >= MIN_GAP for i, a in enumerate(pts) for b in pts[i + 1:])
        reach = all(kin.ik_all(x, y, 0.05) for x, y in pts)
        if far and reach:
            return dict(zip(["red", "green", "blue"], pts))


def add_noise(frame, sigma, rng_np):
    if sigma == 0:
        return frame
    noisy = frame.astype(np.float32) + rng_np.normal(0, sigma, frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def evaluate(objects, truth):
    """Match detections to the true cubes. Returns TP, FP, FN and per-cube estimates."""
    est, fp = {}, 0
    for o in objects:
        x, y = pixel_to_robot(*o["center_px"])
        tx, ty = truth[o["color"]]
        err = 1000 * math.hypot(x - tx, y - ty)
        if err <= MATCH_MM and o["color"] not in est:
            est[o["color"]] = (x, y, err)
        else:
            fp += 1
    return len(est), fp, len(truth) - len(est), est


robot = Robot()
sim = robot.sim
cam = sim.getObject('/Camera')
start_pos = {c: sim.getObjectPosition(h, -1) for c, h in robot.cubes.items()}


def place_cubes(layout):
    for c, h in robot.cubes.items():
        sim.setObjectParent(h, -1, True)
        sim.setObjectPosition(h, -1, [*layout[c], CUBE_Z])
    robot.held = None
    robot._step(); robot._step()            # let the camera render the new scene


rng = random.Random(42)
rng_np = np.random.default_rng(42)
layouts = [random_layout(rng) for _ in range(N_TRIALS)]
e1_rows, e2_rows = [], []

robot.start()
try:
    robot.home()
    # ---------------- E2: noise (detection only, fast) ----------------
    print("E2: noise robustness ...")
    for k, layout in enumerate(layouts, 1):
        place_cubes(layout)
        frame = grab_frame(sim, cam)
        for s in SIGMAS:
            tp, fp, fn, est = evaluate(detect_objects(add_noise(frame, s, rng_np)), layout)
            errs = [e[2] for e in est.values()]
            graspable = sum(e <= 1000 * GRASP_TOL for e in errs)
            e2_rows.append([k, s, tp, fp, fn, f"{np.mean(errs):.2f}" if errs else "", graspable])

    # ---------------- E1: 30 full sorting trials ----------------
    print("E1: 30 sorting trials (this takes a while) ...")
    for k, layout in enumerate(layouts, 1):
        place_cubes(layout)
        robot.log.clear()
        tp, fp, fn, est = evaluate(detect_objects(grab_frame(sim, cam)), layout)
        ok_count = 0
        for color in ["red", "green", "blue"]:
            if color not in est:                     # not detected -> cannot be sorted
                e1_rows.append([k, color, *layout[color], 0, "", 0, 0, "", ""])
                continue
            x, y, loc = est[color]
            try:
                grasped, placed, cycle = robot.pick_and_place((x, y), color)
            except ValueError:                       # unreachable -> safety stop
                grasped, placed, cycle = None, None, float("nan")
            zx, zy = robot.zone_xy(color)
            perr = 1000 * math.hypot(placed[0] - zx, placed[1] - zy) if placed else None
            sorted_ok = grasped == color and perr is not None and perr < 50
            ok_count += sorted_ok
            e1_rows.append([k, color, *layout[color], 1, f"{loc:.2f}", int(grasped == color),
                            int(sorted_ok), f"{perr:.2f}" if perr is not None else "",
                            f"{cycle:.2f}"])
        robot.home()
        print(f"  trial {k:2d}/{N_TRIALS}: detected {tp}/3, false {fp}, sorted {ok_count}/3")
finally:
    robot.stop()
    for c, h in robot.cubes.items():
        sim.setObjectPosition(h, -1, start_pos[c])

# ---------------- Save CSV files ----------------
with open(os.path.join(RES, "e1_sorting_trials.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "color", "true_x", "true_y", "detected", "localization_error_mm",
                "grasped", "sorted", "placement_error_mm", "cycle_time_s"])
    w.writerows(e1_rows)
with open(os.path.join(RES, "e2_noise.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "sigma", "TP", "FP", "FN", "mean_localization_error_mm", "graspable"])
    w.writerows(e2_rows)

# ---------------- E1 summary ----------------
n = len(e1_rows)
det = sum(r[4] for r in e1_rows)
loc = [float(r[5]) for r in e1_rows if r[5] != ""]
cyc = [float(r[9]) for r in e1_rows if r[9] not in ("", "nan")]
pe = [float(r[8]) for r in e1_rows if r[8] != ""]
summary = [
    ["Cubes in total", n],
    ["Detection recall (%)", f"{100 * det / n:.1f}"],
    ["Localization error mean / max (mm)", f"{np.mean(loc):.2f} / {np.max(loc):.2f}"],
    ["Grasp success (%)", f"{100 * sum(r[6] for r in e1_rows) / n:.1f}"],
    ["Sorting success (%)", f"{100 * sum(r[7] for r in e1_rows) / n:.1f}"],
    ["Placement error mean (mm)", f"{np.mean(pe):.2f}" if pe else ""],
    ["Cycle time mean +- SD (s)", f"{np.mean(cyc):.2f} +- {np.std(cyc, ddof=1):.2f}"],
]

# ---------------- E2 summary ----------------
e2_sum = []
for s in SIGMAS:
    rows = [r for r in e2_rows if r[1] == s]
    TP, FP, FN = (sum(r[i] for r in rows) for i in (2, 3, 4))
    errs = [float(r[5]) for r in rows if r[5] != ""]
    prec = 100 * TP / (TP + FP) if TP + FP else 0
    rec = 100 * TP / (TP + FN)
    grasp = 100 * sum(r[6] for r in rows) / (3 * len(rows))
    e2_sum.append([s, prec, rec, np.mean(errs) if errs else float("nan"), grasp])

with open(os.path.join(RES, "summary.txt"), "w") as f:
    f.write("E1: 30 trials, vision-based sorting\n")
    for a, b in summary:
        f.write(f"  {a}: {b}\n")
    f.write("\nE2: noise robustness (sigma | precision % | recall % | loc. error mm | graspable %)\n")
    for s, p, r, e, g in e2_sum:
        f.write(f"  {s:3d} | {p:6.1f} | {r:6.1f} | {e:6.2f} | {g:6.1f}\n")
print(open(os.path.join(RES, "summary.txt")).read())

# ---------------- Plots for the report ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(loc, bins=12, color="#4C72B0", edgecolor="white")
ax[0].set_xlabel("localization error (mm)"); ax[0].set_ylabel("number of cubes")
ax[0].set_title("E1: localization error (90 cubes)")
ax[1].hist(cyc, bins=12, color="#55A868", edgecolor="white")
ax[1].set_xlabel("cycle time per cube (s)"); ax[1].set_ylabel("number of cubes")
ax[1].set_title("E1: cycle time")
fig.tight_layout(); fig.savefig(os.path.join(RES, "e1_results.png"), dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sg = [r[0] for r in e2_sum]
ax[0].plot(sg, [r[1] for r in e2_sum], "o-", label="precision")
ax[0].plot(sg, [r[2] for r in e2_sum], "s-", label="recall")
ax[0].plot(sg, [r[4] for r in e2_sum], "^-", label="graspable (error <= 15 mm)")
ax[0].set_xlabel("noise sigma (grey levels)"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 105)
ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title("E2: detection under noise")
ax[1].plot(sg, [r[3] for r in e2_sum], "o-", color="#C44E52")
ax[1].set_xlabel("noise sigma (grey levels)"); ax[1].set_ylabel("mean localization error (mm)")
ax[1].grid(alpha=0.3); ax[1].set_title("E2: localization error under noise")
fig.tight_layout(); fig.savefig(os.path.join(RES, "e2_noise.png"), dpi=150)

# Example noisy frame for the report (sigma = 60, near the breaking point)
import cv2
from detect import draw_detections
place_frame = add_noise(frame, 60, rng_np)
cv2.imwrite(os.path.join(RES, "noise_sigma60_example.png"),
            draw_detections(place_frame, detect_objects(place_frame)))
print("Saved in results/: e1_sorting_trials.csv, e2_noise.csv, summary.txt, "
      "e1_results.png, e2_noise.png, noise_sigma60_example.png")
