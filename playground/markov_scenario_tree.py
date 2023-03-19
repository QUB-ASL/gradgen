import numpy as np
import casadi as cs
import gradgen

p = np.array([[0.5, 0.4, 0.1],
              [0.5, 0.3, 0.2],
              [0.6, 0.2, 0.2]])
v = np.array([0.1, 0.8, 0.1])
(N, tau) = (3, 2)
markov_tree = gradgen.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

# markov_tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')
print(markov_tree)


nx, nu = 3, 2
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)

f = cs.vertcat(1.1 * x[0] + 2 * u[0],
               x[1] + 0.5 * x[0] + 4 * u[1],
               x[2] + 0.9 * x[1])
f_list = [f, 0 * f, 10 * f]

# stage cost function, ell
ell0 = 5 * cs.dot(x, x) + 0.1 * x[0] + cs.dot(u, u)
ell1 = 3 * cs.dot(x, x) + cs.dot(u, u)
ell2 = 10 * cs.dot(x, x) - 0.1 * x[0] + 3 * cs.dot(u, u)
ell_list = [ell0, ell1, ell2]

# terminal cost function, vf
vf = 0.5 * cs.dot(x, x)


# gradiator
uncertain_gradiator = gradgen.CostGradientStochastic(markov_tree, x, u, f_list, ell_list, vf) \
    .with_name("stochastic_quadcopter") \
    .with_target_path("codegenz")
uncertain_gradiator.build(no_rust_build=True)
