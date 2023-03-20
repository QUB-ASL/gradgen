import casadi.casadi as cs
import numpy as np


def _check_state_input_dims(x, u, nx, nu):
    assert x.shape[0] == nx, f"x must have dim={nx}"
    assert u.shape[0] == nu, f"u must have dim={nu}"


class ModelFactory:

    @staticmethod
    def create_random_linear_model(x, u):
        nx = x.shape[0]
        nu = u.shape[0]
        A = 0.9 * np.eye(nx) + 0.05 * np.random.rand(nx, nx)
        B = np.random.rand(nx, nu)
        f = A @ x + B @ u
        return f

    @ staticmethod
    def create_bicycle_model():
        raise NotImplementedError()

    @ staticmethod
    def create_inv_pend_model(x, u, m=1, ell=1, M=1, g=9.81, t_sampling=0.01):
        _check_state_input_dims(x, u, 2, 1)
        th, w = x[0], x[1]
        Mtot = M + m
        psi = -3 * (m*ell*cs.sin(2*th) + u * cs.cos(th) - Mtot*g*cs.sin(th))
        psi /= ell * (4*Mtot - 3*m*cs.cos(th)**2)
        f = cs.vertcat(w, psi)
        return cs.simplify(x + t_sampling * f)

    @ staticmethod
    def create_quadcopter_model(x, u):
        _check_state_input_dims(x, u, 10, 3)
        raise NotImplementedError()

    @ staticmethod
    def create_ball_and_beam_model(x, u, m=1, g=9.81, moi=5e-4, t_sampling=0.01):
        _check_state_input_dims(x, u, 4, 1)
        f = cs.vertcat(x[1],
                       (5/7) * (x[0] * x[3]**2 - g * cs.sin(x[2])),
                       x[3],
                       (u - m*g * x[0] * cs.cos(x[1]) - 2*m *
                        x[0] * x[1] * x[2]) / (m * x[0]**2 + moi)
                       )
        return x + t_sampling * f
