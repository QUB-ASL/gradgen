import os
import unittest
import casadi.casadi as cs
from gradgen.cost_gradient import *
import logging
import numpy as np
import subprocess

logger = logging.getLogger(__name__)


class GradgenTestCase(unittest.TestCase):

    @classmethod
    def create_example(cls):
        nx, nu = 10, 3
        k1, k2, Ts = 252, 20, 1 / 125
        Gamma_u = np.diag([0, 0, 124.659])
        Gamma_n = np.diag([2.287469, 1.35699886, -0.42388942])
        I = np.diag([0.01788, 0.03014, 0.04614])
        Iinv = np.linalg.inv(I)
        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)
        (q0, q1, q2, q3, wx, wy, wz, nx, ny, nz) = (
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
        Q_bar_q = cs.vertcat(cs.vertcat(0, -x[5], -x[6], -x[7]).T,
                             cs.vertcat(x[5], 0, x[7], -x[6]).T,
                             cs.vertcat(x[6], -x[7], 0, x[5]).T,
                             cs.vertcat(x[7], x[6], -x[5], 0).T)
        tau = cs.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
        expQbar = cs.vertcat(
            cs.vertcat(cs.cosh(tau), -wx * tau, -wy * tau, -wz * tau).T,
            cs.vertcat(wx * tau, cs.cosh(tau), wz * tau, wy * tau).T,
            cs.vertcat(wy * tau, -wz * tau, cs.cosh(tau), wx * tau).T,
            cs.vertcat(wz * tau, wy * tau, -wx * tau, cs.cosh(tau)).T)
        q = cs.vertcat(q0, q1, q2, q3)
        w = cs.vertcat(wx, wy, wz)
        n = cs.vertcat(nx, ny, nz)
        f = cs.vertcat((Ts / 2) * expQbar @ q,
                       w + Ts * (Gamma_n @ n + Gamma_u @ u -
                                 Iinv @ cs.cross(w, I @ w)),
                       n + Ts * (k1 * k2 * u - k2 * n)
                       )

        # Stage cost function, ell
        ell = 5*x[0]**2 + 0.01*x[1]**2 + 0.01*x[2]**2

        # terminal cost function, vf
        vf = 0.5 * (x[0]**2 + 50 * x[1]**2 + 100 * x[2]**2)
        return x, u, f, ell, vf

    def test_generate_code_and_build(self):
        x, u, f, ell, vf = GradgenTestCase.create_example()
        N = 15
        gradObj = CostGradient(x, u, f, ell, vf, N).with_name(
            "quadcopter_test").with_target_path(".")
        gradObj.build(no_rust_build=True)


if __name__ == '__main__':
    logger.setLevel(logging.ERROR)
    unittest.main()
