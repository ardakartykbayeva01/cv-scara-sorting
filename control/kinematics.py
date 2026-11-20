"""
kinematics.py
Forward and inverse kinematics of our SCARA robot (2 revolute joints + 1 prismatic joint).

  J1: q1, rotation about the vertical z-axis at the base
  J2: q2, rotation about the vertical z-axis at the end of link 1
  J3: d3, prismatic joint, positive d3 moves the tool DOWN

Forward kinematics:
  x = L1*cos(q1) + L2*cos(q1 + q2)
  y = L1*sin(q1) + L2*sin(q1 + q2)
  z = TOOL_Z0 - d3
"""

import math

# ---------------- Robot parameters (the same as in build_scene.py) ----------------
L1 = 0.30                       # link 1 length (m)
L2 = 0.25                       # link 2 length (m)
TOOL_Z0 = 0.08                  # tool height when d3 = 0 (m)

# Joint limits
Q1_LIM = (math.radians(-170), math.radians(170))
Q2_LIM = (math.radians(-150), math.radians(150))
D3_LIM = (-0.05, 0.10)          # negative d3 lifts the tool above its zero height


def wrap(a):
    """Wrap an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def fk(q1, q2, d3):
    """Joint values -> tool position (x, y, z) in the base frame."""
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
    z = TOOL_Z0 - d3
    return x, y, z


def within_limits(q1, q2, d3):
    return (Q1_LIM[0] <= q1 <= Q1_LIM[1] and
            Q2_LIM[0] <= q2 <= Q2_LIM[1] and
            D3_LIM[0] <= d3 <= D3_LIM[1])


def ik_all(x, y, z):
    """
    All inverse kinematics solutions (elbow-down and elbow-up) for a tool position.
    Returns a list of (q1, q2, d3). The list is empty if the point is unreachable.
    """
    r2 = x * x + y * y

    # Step 1: law of cosines -> cos(q2)
    c2 = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(c2) > 1:                         # reachability: |L1 - L2| <= r <= L1 + L2
        return []

    # Step 2: the prismatic joint follows directly from the height
    d3 = TOOL_Z0 - z

    solutions = []
    for elbow in (+1, -1):                  # +1: elbow-down (q2 > 0), -1: elbow-up (q2 < 0)
        s2 = elbow * math.sqrt(1 - c2 ** 2)
        q2 = math.atan2(s2, c2)
        # Step 3: q1 = angle to the target - inner angle of the triangle
        q1 = wrap(math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2))
        if within_limits(q1, q2, d3):
            solutions.append((q1, q2, d3))
    return solutions


def ik(x, y, z, current=None):
    """
    Best inverse kinematics solution for (x, y, z).
    If the current joint values are given, the solution closest to them is chosen
    (smallest joint motion). Raises ValueError if the point cannot be reached.
    """
    sols = ik_all(x, y, z)
    if not sols:
        raise ValueError(f"Target ({x:.3f}, {y:.3f}, {z:.3f}) is unreachable or outside joint limits")
    if current is None:
        return sols[0]
    return min(sols, key=lambda s: abs(s[0] - current[0]) + abs(s[1] - current[1]))


if __name__ == "__main__":
    # Self-test: IK -> FK must return the same point
    for p in [(0.30, -0.15, 0.05), (0.42, 0.05, 0.05), (-0.30, 0.30, 0.045),
              (-0.40, 0.0, 0.045), (-0.30, -0.30, 0.045), (0.60, 0.0, 0.05)]:
        try:
            q = ik(*p)
            back = fk(*q)
            err = 1000 * math.dist(p, back)
            print(f"{p} -> q1={math.degrees(q[0]):7.2f} deg, q2={math.degrees(q[1]):7.2f} deg, "
                  f"d3={q[2]:.3f} m | FK error = {err:.4f} mm")
        except ValueError as e:
            print(e)
