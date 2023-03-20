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

A = np.array([[0.9, -0.5, 0.0],
              [0.0, 0.1, 0.5],
              [-0.6, 0.7, 0.8]])
B1 = np.array([[2, 0],
              [0, 1],
              [0, 0]])
B2 = np.array([[1, 0],
              [0.5, 2],
              [0, 0]])
B3 = np.array([[0, 0],
              [0, 0],
              [0, 0]])

f_list = [A @ x + B1 @ u,
          (3 * A) @ x + B2 @ u,
          0.5 * A @ x + B3 @ u]

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
uncertain_gradiator.build(no_rust_build=False)
