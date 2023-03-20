import casadi.casadi as cs


class ModelFactory:

    @staticmethod
    def create_random_linear_model(nx, nu):
        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)
        f = 0
        return x, u, f

    @staticmethod
    def create_bicycle_model():
        x = cs.SX.sym('x', 3)
        u = cs.SX.sym('u', 2)
        f = 0
        return x, u, f

    @staticmethod
    def create_inv_pend_model():
        pass

    @staticmethod
    def create_quadcopter_model():
        pass

    @staticmethod
    def create_ball_and_beam_model():
        pass
