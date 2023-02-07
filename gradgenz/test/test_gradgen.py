import os
import unittest
import casadi.casadi as cs
from gradgenz.cost_gradient import *
import logging
import numpy as np
import subprocess


logger = logging.getLogger(__name__)


class GradgenTestCase(unittest.TestCase):

    @classmethod
    def create_example(cls):
        nx, nu = 4, 3
        m, M, ell, F, g, ts = 1, 3, 0.0005, 4, 9.81, 0.01

        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)
        # (q0, q1, q2, q3, wx, wy, wz, nx, ny, nz) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])


        # System dynamics, f
        f = cs.vertcat(
            x[0] + ts * x[1],
            x[1] + ts * (4 * m * ell * x[3] ** 2 * cs.sin(x[2]) + 4 * F - 3 * m * g * cs.sin(x[2]) * cs.cos(x[2]))
            /(4 * (M + m) - 3 * m * cs.cos(x[2]) ** 2),
            x[2] + ts * x[3],
            x[3] + ts * (-3) * ((m * ell * x[3] ** 2 * cs.sin(x[2]) * cs.cos(x[2]) + F * cs.cos(x[2]) - (M + m) * g * cs.sin(x[2]))
                  / ((4 * (M + m) - 3 * m * cs.cos(x[2]) ** 2) * ell))
            )

        # Stage cost function, ell
        ell = 5*x[0]**2 + 0.01*x[1]**2 + 0.01*x[2]**2+ 0.05*x[3]**2 + 2.2*u**2

        # terminal cost function, vf
        vf = 0.5 * (x[0]**2 + 50 * x[1]**2 + 100 * x[2]**2)
        return x, u, f, ell, vf

    def test_generate_code_and_build(self):
        x, u, f, ell, vf = GradgenTestCase.create_example()
        N = 15
        gradObj = CostGradient(x, u, f, ell, vf, N).with_name(
            "inverted_pendulum_test").with_target_path("codegenz")
        gradObj.build(no_rust_build=True)


if __name__ == '__main__':
    logger.setLevel(logging.ERROR)
    unittest.main()
