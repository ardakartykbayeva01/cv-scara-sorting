"""
trajectory.py
Quintic (5th-order) polynomial trajectory in joint space.

For each joint, from q0 to qf in time T, with s = t / T:
  q(t)   = q0 + (qf - q0) * (10 s^3 - 15 s^4 + 6 s^5)
  q'(t)  = (qf - q0) / T * (30 s^2 - 60 s^3 + 30 s^4)
Velocity and acceleration are zero at the start and at the end,
so the robot starts and stops smoothly (no jumps).
"""

V_MAX_REV = 1.0      # max speed of the revolute joints (rad/s)
V_MAX_PRI = 0.10     # max speed of the prismatic joint (m/s)
T_MIN = 0.5          # shortest allowed motion time (s)


def duration(q0, qf):
    """Motion time so that no joint goes faster than its maximum speed.
    The peak speed of a quintic profile is 1.875 * (qf - q0) / T."""
    t_rev = max(abs(qf[0] - q0[0]), abs(qf[1] - q0[1])) * 1.875 / V_MAX_REV
    t_pri = abs(qf[2] - q0[2]) * 1.875 / V_MAX_PRI
    return max(T_MIN, t_rev, t_pri)


def quintic(q0, qf, T, t):
    """Joint positions and velocities at time t (0 <= t <= T)."""
    s = min(max(t / T, 0.0), 1.0)
    shape = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    dshape = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    pos = [a + (b - a) * shape for a, b in zip(q0, qf)]
    vel = [(b - a) * dshape for a, b in zip(q0, qf)]
    return pos, vel
