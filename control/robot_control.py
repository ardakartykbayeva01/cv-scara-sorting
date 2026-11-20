"""
robot_control.py
Control interface for the SCARA robot in CoppeliaSim:
  home(), move_to(x, y, z), gripper_close(), gripper_open(), pick_and_place(...)

All motions use our own inverse kinematics (kinematics.py) and smooth
quintic joint trajectories (trajectory.py). The simulation runs in stepping
mode, so every motion is deterministic and every step is logged.
"""

import csv
import math
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import kinematics as kin
from trajectory import duration, quintic

# ---------------- Task heights (m) ----------------
CUBE = 0.04                      # cube size
MAT_TOP = 0.01                   # top of the white work mat
ZONE_TOP = 0.005                 # top of a sorting zone
PICK_Z = MAT_TOP + CUBE          # tool touches the top of a cube on the mat (0.05)
PLACE_Z = ZONE_TOP + CUBE        # tool height when the cube stands on a zone (0.045)
TRAVEL_Z = 0.12                  # safe height for moving above other cubes
GRASP_TOL = 0.015                # max distance tool <-> cube top for a successful grasp (m)

HOME_Q = (math.radians(150), math.radians(-60), kin.TOOL_Z0 - TRAVEL_Z)   # outside the camera view

ZONES = {'red': '/Zone_red', 'green': '/Zone_green', 'blue': '/Zone_blue'}
CUBES = {'red': '/Cube_red', 'green': '/Cube_green', 'blue': '/Cube_blue'}


class SimulationStopped(Exception):
    """Raised when the user presses Stop in CoppeliaSim during a motion."""


class Robot:
    def __init__(self, realtime=False):
        """realtime=False: Python starts the simulation and steps it (tests, experiments).
        realtime=True: the user presses Play/Stop in CoppeliaSim, Python follows the clock."""
        self.realtime = realtime
        self.client = RemoteAPIClient()
        try:
            self.sim = self.client.require('sim')
        except AttributeError:
            self.sim = self.client.getObject('sim')
        sim = self.sim

        self.j = [sim.getObject('/J1'), sim.getObject('/J2'), sim.getObject('/J3')]
        self.tool = sim.getObject('/Tool')
        self.cubes = {c: sim.getObject(p) for c, p in CUBES.items()}
        self.zones = {c: sim.getObject(p) for c, p in ZONES.items()}

        # Allow the prismatic joint to lift the tool above its zero height (safe travel height)
        sim.setJointInterval(self.j[2], False, [kin.D3_LIM[0], kin.D3_LIM[1] - kin.D3_LIM[0]])

        self.held = None             # name of the cube held by the suction tool
        self.log = []                # rows: t, q1, q2, d3, v1, v2, v3
        self.t = 0.0

    # ---------------- Simulation control ----------------
    def start(self):
        if self.realtime:                         # Play is pressed by the user
            self.dt = self.sim.getSimulationTimeStep()
            return
        try:
            self.sim.setStepping(True)            # newer CoppeliaSim / client
            self._step = self.sim.step
        except Exception:
            self.client.setStepping(True)         # older versions
            self._step = self.client.step
        self.sim.startSimulation()
        self.dt = self.sim.getSimulationTimeStep()
        self._step()

    def stop(self):
        self.sim.stopSimulation()

    def sim_time(self):
        return self.sim.getSimulationTime()

    # ---------------- Joint level ----------------
    def joints(self):
        return [self.sim.getJointPosition(h) for h in self.j]

    def is_running(self):
        return self.sim.getSimulationState() != self.sim.simulation_stopped

    def move_joints(self, qf):
        """Smooth quintic motion from the current joint values to qf."""
        q0 = self.joints()
        T = duration(q0, qf)
        if self.realtime:
            t0 = self.sim_time()
            while True:
                if not self.is_running():         # user pressed Stop
                    raise SimulationStopped()
                t = self.sim_time() - t0
                pos, vel = quintic(q0, qf, T, t)
                for h, p in zip(self.j, pos):
                    self.sim.setJointPosition(h, p)
                self.log.append([self.sim_time()] + pos + vel)
                if t >= T:
                    return
                time.sleep(0.01)
        steps = max(1, math.ceil(T / self.dt))
        for k in range(1, steps + 1):
            pos, vel = quintic(q0, qf, T, k * self.dt)
            for h, p in zip(self.j, pos):
                self.sim.setJointPosition(h, p)
            self._step()
            self.log.append([self.sim_time()] + pos + vel)

    # ---------------- Task level ----------------
    def home(self):
        self.move_joints(list(HOME_Q))

    def move_to(self, x, y, z):
        """Move the tool to (x, y, z) in the base frame using our IK.
        Raises ValueError (safety stop) if the point is unreachable."""
        q = kin.ik(x, y, z, current=self.joints())
        self.move_joints(list(q))

    def tool_position(self):
        return self.sim.getObjectPosition(self.tool, -1)

    def gripper_close(self):
        """Suction ON: attach the cube right under the tool, if it is close enough."""
        tp = self.tool_position()
        for color, h in self.cubes.items():
            cp = self.sim.getObjectPosition(h, -1)
            top = [cp[0], cp[1], cp[2] + CUBE / 2]
            if math.dist(tp, top) < GRASP_TOL:
                self.sim.setObjectParent(h, self.tool, True)
                self.held = color
                return color
        return None                              # nothing to grasp

    def gripper_open(self):
        """Suction OFF: release the held cube where it is."""
        if self.held is None:
            return None
        h = self.cubes[self.held]
        self.sim.setObjectParent(h, -1, True)
        color, self.held = self.held, None
        return color

    def zone_xy(self, color):
        p = self.sim.getObjectPosition(self.zones[color], -1)
        return p[0], p[1]

    def pick_and_place(self, pick_xy, color):
        """Full cycle: approach -> lower -> suction on -> lift -> move -> lower -> release -> lift.
        Returns (grasped_color, placed_cube_position, cycle_time)."""
        t0 = self.sim_time()
        px, py = pick_xy
        zx, zy = self.zone_xy(color)

        self.move_to(px, py, TRAVEL_Z)           # approach
        self.move_to(px, py, PICK_Z)             # lower onto the cube
        grasped = self.gripper_close()
        if grasped is None:
            self.move_to(px, py, TRAVEL_Z)
            return None, None, self.sim_time() - t0
        self.move_to(px, py, TRAVEL_Z)           # lift
        self.move_to(zx, zy, TRAVEL_Z)           # transfer
        self.move_to(zx, zy, PLACE_Z)            # lower into the zone
        self.gripper_open()                      # release
        self.move_to(zx, zy, TRAVEL_Z)           # retreat
        placed = self.sim.getObjectPosition(self.cubes[grasped], -1)
        return grasped, placed, self.sim_time() - t0

    # ---------------- Logging ----------------
    def save_log(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "q1", "q2", "d3", "v1", "v2", "v3"])
            w.writerows(self.log)
