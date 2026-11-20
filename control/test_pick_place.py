"""
test_pick_place.py  (Student 1, steps 4, 5 and 8)
Sorts the 3 cubes using their KNOWN positions from the simulator (no vision yet).
This checks the control part on its own: IK, smooth trajectories, suction, safety stop.

How to run (CoppeliaSim open with cv_scara_scene.ttt, simulation STOPPED):
  python3 test_pick_place.py
Press Ctrl+C at any time for an emergency stop.
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_control import Robot

os.makedirs("../results", exist_ok=True)

robot = Robot()
sim = robot.sim
robot.start()
rows = []

try:
    robot.home()
    for color in ["red", "green", "blue"]:
        cp = sim.getObjectPosition(robot.cubes[color], -1)      # known (true) cube position
        try:
            grasped, placed, cycle = robot.pick_and_place((cp[0], cp[1]), color)
        except ValueError as e:                                  # unreachable -> safety stop
            print(f"{color}: SAFETY STOP - {e}")
            rows.append([color, cp[0], cp[1], "", "", "", "unreachable", ""])
            continue
        if grasped is None:
            print(f"{color}: grasp FAILED")
            rows.append([color, cp[0], cp[1], "", "", "", "grasp failed", f"{cycle:.2f}"])
            continue
        zx, zy = robot.zone_xy(color)
        err = 1000 * math.hypot(placed[0] - zx, placed[1] - zy)
        ok = grasped == color and err < 50                       # inside the 10 cm zone
        print(f"{color}: picked {grasped}, placement error = {err:.2f} mm, "
              f"cycle time = {cycle:.2f} s -> {'SUCCESS' if ok else 'FAIL'}")
        rows.append([color, cp[0], cp[1], placed[0], placed[1], f"{err:.2f}",
                     "success" if ok else "fail", f"{cycle:.2f}"])
    robot.home()
except KeyboardInterrupt:
    print("EMERGENCY STOP (Ctrl+C): robot stopped.")
finally:
    robot.save_log("../results/joint_trajectory.csv")
    robot.stop()

with open("../results/pick_place_known_positions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["color", "cube_x", "cube_y", "placed_x", "placed_y",
                "placement_error_mm", "result", "cycle_time_s"])
    w.writerows(rows)

# Plot joint positions and velocities (for the report)
t = [r[0] for r in robot.log]
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t, [math.degrees(r[1]) for r in robot.log], label="q1 (deg)")
ax[0].plot(t, [math.degrees(r[2]) for r in robot.log], label="q2 (deg)")
ax[0].plot(t, [1000 * r[3] for r in robot.log], label="d3 (mm)")
ax[0].set_ylabel("joint position")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(t, [math.degrees(r[4]) for r in robot.log], label="q1 velocity (deg/s)")
ax[1].plot(t, [math.degrees(r[5]) for r in robot.log], label="q2 velocity (deg/s)")
ax[1].plot(t, [1000 * r[6] for r in robot.log], label="d3 velocity (mm/s)")
ax[1].set_xlabel("simulation time (s)"); ax[1].set_ylabel("joint velocity")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Quintic joint trajectories during sorting (known positions)")
fig.tight_layout()
fig.savefig("../results/joint_trajectory.png", dpi=150)
print("Saved: results/pick_place_known_positions.csv, joint_trajectory.csv, joint_trajectory.png")
