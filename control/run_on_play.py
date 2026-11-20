"""
run_on_play.py
Start this once, then control everything from CoppeliaSim:
  - press PLAY  -> the robot sorts the cubes
  - press STOP  -> the robot stops, the cubes go back to their start positions
  - press PLAY again -> a new run starts
Press Ctrl+C in the Terminal to close the program.

The cube positions come from the CAMERA (detect_objects + pixel_to_robot),
so run vision/calibrate.py once before this script.

How to run (CoppeliaSim open with cv_scara_scene.ttt):
  python3 run_on_play.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision"))

from robot_control import Robot, SimulationStopped
from camera import grab_frame
from detect import detect_objects
from calibration import pixel_to_robot

robot = Robot(realtime=True)
sim = robot.sim
cam = sim.getObject('/Camera')
pixel_to_robot(256, 256)          # stops here at once if calibration.json is missing

# Wait until the simulation is stopped, then remember the start positions of the cubes
if robot.is_running():
    print("Please press STOP in CoppeliaSim first...")
    while robot.is_running():
        time.sleep(0.2)
start_pos = {c: sim.getObjectPosition(h, -1) for c, h in robot.cubes.items()}


def reset_cubes():
    """Put every cube back to its start position (also after an interrupted run)."""
    for c, h in robot.cubes.items():
        sim.setObjectParent(h, -1, True)
        sim.setObjectPosition(h, -1, start_pos[c])
    robot.held = None


def get_targets():
    """List of cubes to sort: (color, x, y), found by the CAMERA (no simulator positions).
    frame -> detect_objects() -> pixel_to_robot().
    The true positions are read only to print the localization error (for the report)."""
    targets = []
    for o in detect_objects(grab_frame(sim, cam)):
        x, y = pixel_to_robot(*o["center_px"])
        true = sim.getObjectPosition(robot.cubes[o["color"]], -1)
        err = 1000 * math.hypot(x - true[0], y - true[1])
        print(f"  camera: {o['color']} at pixel ({o['center_px'][0]:.1f}, {o['center_px'][1]:.1f})"
              f" -> robot ({x:.3f}, {y:.3f}) m, localization error {err:.1f} mm")
        targets.append((o["color"], x, y))
    if not targets:
        print("  camera: no cubes found")
    return targets


print("READY. Press PLAY in CoppeliaSim to start sorting (Ctrl+C here to quit).")
run = 0
try:
    while True:
        while not robot.is_running():            # wait for PLAY
            time.sleep(0.1)
        run += 1
        robot.start()
        reset_cubes()
        print(f"\nRun {run}: PLAY pressed, sorting started.")
        try:
            robot.home()                         # robot leaves the camera view first
            for color, x, y in get_targets():
                grasped, placed, cycle = robot.pick_and_place((x, y), color)
                if grasped is None:
                    print(f"  {color}: grasp failed")
                    continue
                zx, zy = robot.zone_xy(color)
                err = 1000 * math.hypot(placed[0] - zx, placed[1] - zy)
                print(f"  {color}: placed, error = {err:.2f} mm, cycle time = {cycle:.2f} s")
            robot.home()
            print(f"Run {run} finished. Press STOP, then PLAY to run again.")
        except SimulationStopped:
            print(f"Run {run}: STOP pressed, robot stopped.")
            robot.held = None
        except ValueError as e:
            print(f"Run {run}: SAFETY STOP - {e}")
        while robot.is_running():                # wait for STOP
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nProgram closed.")
