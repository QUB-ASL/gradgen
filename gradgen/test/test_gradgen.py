import os
import unittest
import casadi.casadi as cs
from ..cost_gradient import *
import logging


import numpy as np

import subprocess


logger = logging.getLogger(__name__)


class GradgenTestCase(unittest.TestCase):

    @classmethod
    def create_example(cls):
        nx, nu = 4, 1
        m, I, g, ts= 1, 0.0005,9.81,0.01

        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)

        # System dynamics, f
        f = cs.vertcat(
            x[0] + ts * x[1],
            x[1] + ts * ((5/7)*x[0]*x[3]**2-g * cs.sin(x[2])),
            x[2] + ts * x[3],
            x[3] + ts * ((u[0] - m * g * x[0] * cs.cos(x[2]) - 2 * m * x[0] * x[1] * x[2] ) / (m*x[0]**2+I)))

        # Stage cost function, ell
        ell = cs.dot(x, x) + cs.dot(u, u)

        # terminal cost function, vf
        vf = 10 * cs.dot(x, x)
        return x, u, f, ell, vf

    def test_generate_code_and_build(self):
        x, u, f, ell, vf = GradgenTestCase.create_example()
        N = 15
        gradObj = CostGradient(x, u, f, ell, vf, N).with_name(
            "ball_test").with_target_path("codegenz")
        gradObj.build(no_rust_build=True)


if __name__ == '__main__':
    logger.setLevel(logging.ERROR)
    unittest.main()
