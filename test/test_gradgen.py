import os
import unittest
import casadi.casadi as cs
import gradgen as gg
import logging
import numpy as np
# import subprocess


logger = logging.getLogger(__name__)


class GradgenTestCase(unittest.TestCase):

    @classmethod
    def create_example(cls):
        nx, nu = 3, 2
        ts, L = 0.1, 0.9

        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)

        # System dynamics, f
        theta_dot = (u[1] * cs.cos(x[2]) - u[0] * cs.sin(x[2])) / L
        f = cs.vertcat(
            x[0] + ts * (u[0] + L * cs.sin(x[2]) * theta_dot),
            x[1] + ts * (u[1] - L * cs.cos(x[2]) * theta_dot),
            x[2] + ts * theta_dot)

        # Stage cost function, ell
        q = np.array([1, 2, 3]).reshape((nx, 1))
        ell = 0.5 * cs.dot(x, x) + q.T @ x + 5 * \
            u[0]**2 + 6 * u[1]**2 + 7 * u[0] + 8 * u[1]

        # terminal cost function, vf
        vf = 0.5 * (x[0]**2 + 50 * x[1]**2 + 100 * x[2]**2)
        return x, u, f, ell, vf

    def test_generate_code_and_build(self):
        x, u, f, ell, vf = GradgenTestCase.create_example()
        N = 6
        gradObj = gg.CostGradient(x, u, f, ell, vf, N).with_name(
            "alice_test").with_target_path(".")
        gradObj.build()


if __name__ == '__main__':
    logger.setLevel(logging.ERROR)
    unittest.main()
