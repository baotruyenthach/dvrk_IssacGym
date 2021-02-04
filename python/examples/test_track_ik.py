#!/usr/bin/env python
from trac_ik_python.trac_ik import IK

ik_solver = IK("world", "psm_tool_yaw_link")

seed_state = [0.0] * ik_solver.number_of_joints

IK_solution = ik_solver.get_ik(seed_state, 0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0) # X, Y, Z, QX, QY, QZ, QW
print(IK_solution)