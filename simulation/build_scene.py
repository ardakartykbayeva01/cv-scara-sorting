"""
build_scene.py
Builds the whole project scene in CoppeliaSim automatically and saves it as a .ttt file:
  - SCARA robot: base, revolute joints J1 and J2, prismatic joint J3, suction tool
  - white work mat with 3 coloured cubes (red, green, blue)
  - 3 coloured sorting zones
  - overhead vision sensor (camera)

How to run:
  1. Open CoppeliaSim with a NEW EMPTY scene. The simulation must be STOPPED.
  2. In the Terminal:  python3 build_scene.py
  3. The scene appears in CoppeliaSim and is saved as cv_scara_scene.ttt in this folder.
"""

import math
import os

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ---------------- Connect to CoppeliaSim ----------------
client = RemoteAPIClient()
try:
    sim = client.require('sim')          # newer client versions
except AttributeError:
    sim = client.getObject('sim')        # older client versions

WORLD = -1                               # positions are given in world coordinates

# ---------------- Robot dimensions (metres) ----------------
L1, L2 = 0.30, 0.25                      # link lengths (used later in the kinematics)
BASE_H = 0.30                            # height of the base column
J1_Z = 0.30                              # height of joint 1
J2_Z = 0.33                              # height of joint 2
J3_Z = 0.34                              # height of the prismatic joint (end of link 2)
ROD_LEN = 0.25                           # length of the vertical rod

# Joint mode: kinematic (CoppeliaSim 4.6+), passive in older versions
try:
    KINEMATIC = sim.jointmode_kinematic
except Exception:
    KINEMATIC = sim.jointmode_passive


# ---------------- Helper functions ----------------
def make_static(h):
    """No physics: the object only moves when we move it (kinematic control)."""
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)


def box(name, size, pos, color, parent=None):
    """Cuboid with a name, size (x, y, z), world position and colour."""
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
    sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, WORLD, pos)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    make_static(h)
    if parent is not None:
        sim.setObjectParent(h, parent, True)
    return h


def cylinder(name, size, pos, color, parent=None):
    h = sim.createPrimitiveShape(sim.primitiveshape_cylinder, size, 0)
    sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, WORLD, pos)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    make_static(h)
    if parent is not None:
        sim.setObjectParent(h, parent, True)
    return h


def joint(name, jtype, pos, parent, interval, cyclic=False, flip=False):
    """Revolute or prismatic joint in kinematic mode."""
    h = sim.createJoint(jtype, KINEMATIC, 0)
    sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, WORLD, pos)
    if flip:                                   # turn the joint axis to point down (-z)
        sim.setObjectOrientation(h, WORLD, [math.pi, 0, 0])
    sim.setJointInterval(h, cyclic, interval)  # [minimum, range]
    sim.setObjectParent(h, parent, True)
    return h


# ---------------- Safety check: do not build twice ----------------
try:
    sim.getObject('/SCARA_base')
    raise SystemExit("The scene already has a SCARA robot. Open a NEW empty scene and run again.")
except SystemExit:
    raise
except Exception:
    pass                                       # not found -> we can build

GREY = [0.75, 0.75, 0.75]
DARK = [0.35, 0.35, 0.35]
LIGHT = [0.85, 0.85, 0.85]

# ---------------- SCARA robot ----------------
base = cylinder('SCARA_base', [0.10, 0.10, BASE_H], [0, 0, BASE_H / 2], DARK)

j1 = joint('J1', sim.joint_revolute, [0, 0, J1_Z], base,
           [-math.radians(170), math.radians(340)])
link1 = box('Link1', [L1, 0.05, 0.03], [L1 / 2, 0, 0.315], LIGHT, parent=j1)

j2 = joint('J2', sim.joint_revolute, [L1, 0, J2_Z], link1,
           [-math.radians(150), math.radians(300)])
link2 = box('Link2', [L2, 0.04, 0.02], [L1 + L2 / 2, 0, 0.335], LIGHT, parent=j2)

# Prismatic joint: axis flipped down, so a positive d3 moves the tool DOWN
j3 = joint('J3', sim.joint_prismatic, [L1 + L2, 0, J3_Z], link2,
           [0.0, 0.10], flip=True)
rod = cylinder('Rod', [0.02, 0.02, ROD_LEN], [L1 + L2, 0, J3_Z - ROD_LEN / 2], GREY, parent=j3)
suction = cylinder('Suction', [0.03, 0.03, 0.01],
                   [L1 + L2, 0, J3_Z - ROD_LEN - 0.005], DARK, parent=rod)
tool = sim.createDummy(0.01)                   # tool tip: the point the IK controls
sim.setObjectAlias(tool, 'Tool')
sim.setObjectPosition(tool, WORLD, [L1 + L2, 0, J3_Z - ROD_LEN - 0.01])
sim.setObjectParent(tool, suction, True)

# ---------------- Work mat and cubes ----------------
box('WorkMat', [0.40, 0.60, 0.01], [0.35, 0.0, 0.005], [1.0, 1.0, 1.0])

CUBE = 0.04
cubes = {
    'Cube_red':   ([0.30, -0.15], [0.90, 0.10, 0.10]),
    'Cube_green': ([0.42,  0.05], [0.10, 0.75, 0.10]),
    'Cube_blue':  ([0.32,  0.18], [0.10, 0.25, 0.90]),
}
for name, (xy, color) in cubes.items():
    box(name, [CUBE, CUBE, CUBE], [xy[0], xy[1], 0.01 + CUBE / 2], color)

# ---------------- Sorting zones (outside the camera view) ----------------
zones = {
    'Zone_red':   ([-0.30,  0.30], [0.90, 0.10, 0.10]),
    'Zone_green': ([-0.40,  0.00], [0.10, 0.75, 0.10]),
    'Zone_blue':  ([-0.30, -0.30], [0.10, 0.25, 0.90]),
}
for name, (xy, color) in zones.items():
    box(name, [0.10, 0.10, 0.005], [xy[0], xy[1], 0.0025], color)

# ---------------- Overhead camera ----------------
CAM_POS = [0.35, 0.0, 1.0]
try:
    # options = 2 -> perspective camera; resolution 512 x 512; view angle 45 degrees
    cam = sim.createVisionSensor(2, [512, 512, 0, 0],
                                 [0.01, 2.0, math.radians(45), 0.05, 0.05, 0.05,
                                  0, 0, 0, 0, 0])
    sim.setObjectAlias(cam, 'Camera')
    sim.setObjectPosition(cam, WORLD, CAM_POS)
    sim.setObjectOrientation(cam, WORLD, [math.pi, 0, 0])   # look straight down
    print("Camera created.")
except Exception as e:
    print("Could not create the camera automatically:", e)
    print("Add it by hand: Add -> Vision sensor -> Perspective type, name it 'Camera',")
    print("position (0.35, 0, 1.0), orientation alpha = 180 deg, resolution 512 x 512, angle 45 deg.")

# ---------------- Save the scene ----------------
path = os.path.abspath('cv_scara_scene.ttt')
sim.saveScene(path)
print("Scene built and saved to:", path)
