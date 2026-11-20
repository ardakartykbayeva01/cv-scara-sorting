"""
test_robot.py
Checks the robot and the camera:
  1. moves J1, J2, J3 to a test pose
  2. compares the real tool position with our forward kinematics (FK)
  3. saves one camera frame (camera_frame.png) for the vision part

How to run (CoppeliaSim open with cv_scara_scene.ttt, simulation STOPPED):
  python3 test_robot.py
"""

import math
import time

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
try:
    sim = client.require('sim')
except AttributeError:
    sim = client.getObject('sim')

# Robot dimensions (the same as in build_scene.py)
L1, L2 = 0.30, 0.25
TOOL_Z0 = 0.08          # tool height when d3 = 0

j1 = sim.getObject('/J1')
j2 = sim.getObject('/J2')
j3 = sim.getObject('/J3')
tool = sim.getObject('/Tool')
cam = sim.getObject('/Camera')
base = sim.getObject('/SCARA_base')


def fk(q1, q2, d3):
    """Our forward kinematics: joint values -> tool position (x, y, z)."""
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
    z = TOOL_Z0 - d3
    return x, y, z


sim.startSimulation()
time.sleep(0.5)

# ---------------- 1-2. Move to test poses and compare with FK ----------------
tests = [(0, 0, 0.0), (30, 45, 0.03), (-40, 60, 0.06), (90, -90, 0.0)]
print("Pose (q1, q2, d3)        | CoppeliaSim tool (x, y, z) | Our FK (x, y, z)        | error (mm)")
for q1d, q2d, d3 in tests:
    q1, q2 = math.radians(q1d), math.radians(q2d)
    sim.setJointPosition(j1, q1)
    sim.setJointPosition(j2, q2)
    sim.setJointPosition(j3, d3)
    time.sleep(0.3)
    p = sim.getObjectPosition(tool, -1)              # real position in the world
    f = fk(q1, q2, d3)
    err = 1000 * math.dist(p, f)
    print(f"({q1d:4d}, {q2d:4d}, {d3:.2f})       | "
          f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})      | "
          f"({f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f})   | {err:.2f}")

# back to home pose
sim.setJointPosition(j1, math.radians(90))
sim.setJointPosition(j2, math.radians(-90))
sim.setJointPosition(j3, 0.0)
time.sleep(0.5)

# ---------------- 3. Save one camera frame ----------------
img, res = sim.getVisionSensorImg(cam)
w, h = res
frame = np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)
frame = cv2.flip(frame, 0)                           # CoppeliaSim images are upside down
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)       # OpenCV uses BGR
cv2.imwrite("camera_frame.png", frame)
print(f"Camera frame saved: camera_frame.png ({w} x {h})")

sim.stopSimulation()
