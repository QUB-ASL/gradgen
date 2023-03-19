import numpy as np
import casadi as cs
import gradgen

p = np.array([[0.5, 0.4, 0.1],
              [0.5, 0.3, 0.2],
              [0.6, 0.2, 0.2]])
v = np.array([0.1, 0.8, 0.1])
(N, tau) = (2, 2)
markov_tree = gradgen.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

# markov_tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')
print(markov_tree)

nx, nu = 10, 3
k1, k2, Ts = 252, 20, 1 / 125
Gamma_u = np.diag([0, 0, 124.659])
Gamma_n = np.diag([2.287469, 1.35699886, -0.42388942])
I = np.diag([0.01788, 0.03014, 0.04614])
Iinv = np.linalg.inv(I)
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
(q0, q1, q2, q3, omegax, omegay, omegaz, nx, ny, nz) = (
    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
Q_bar_q = cs.vertcat(cs.vertcat(0, -x[5], -x[6], -x[7]).T,
                     cs.vertcat(x[5], 0, x[7], -x[6]).T,
                     cs.vertcat(x[6], -x[7], 0, x[5]).T,
                     cs.vertcat(x[7], x[6], -x[5], 0).T)
tau = cs.sqrt(omegax ** 2 + omegay ** 2 + omegaz ** 2)
expQbar = cs.vertcat(
    cs.vertcat(cs.cosh(tau), -omegax * tau, -omegay * tau, -omegaz * tau).T,
    cs.vertcat(omegax * tau, cs.cosh(tau), omegaz * tau, omegay * tau).T,
    cs.vertcat(omegay * tau, -omegaz * tau, cs.cosh(tau), omegax * tau).T,
    cs.vertcat(omegaz * tau, omegay * tau, -omegax * tau, cs.cosh(tau)).T)
q = cs.vertcat(q0, q1, q2, q3)
omega = cs.vertcat(omegax, omegay, omegaz)
n = cs.vertcat(nx, ny, nz)
f = cs.vertcat((Ts / 2) * expQbar @ q,
               omega + Ts * (Gamma_n @ n + Gamma_u @ u -
                             Iinv @ cs.cross(omega, I @ omega)),
               n + Ts * (k1 * k2 * u - k2 * n)
               )

# stage cost function, ell
ell = 5 * x[0] ** 2 + 0.01 * x[1] ** 2 + 0.01 * x[2] ** 2

# terminal cost function, vf
vf = 0.5 * (x[0] ** 2 + 50 * x[1] ** 2 + 100 * x[2] ** 2)

# dynamics functions and cost functions for each event
f_list = [f * 2, f * 3, f * 4]
ell_list = [ell * 2, ell * 3, ell * 4]

# gradiator
uncertain_gradiator = gradgen.CostGradientStochastic(markov_tree, x, u, f_list, ell_list, vf) \
    .with_name("stochastic_quadcopter_test") \
    .with_target_path("../codegenz")
uncertain_gradiator.build(no_rust_build=True)
