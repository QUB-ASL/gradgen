import numpy as np
import casadi as cs
import gradgen
import matplotlib.pyplot as plt


p = np.array([[0.2, 0.3, 0.5],
              [0.1, 0.25, 0.65],
              [0.05, 0.8, 0.15]])
v_tree = np.array([0.6, 0.1, 0.3])
N = 30
tau = 3
markov_tree = gradgen.MarkovChainScenarioTreeFactory(
    p, v_tree, N, tau).create()
print(markov_tree)

nx, nu = 2, 1
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
f_list = [gradgen.ModelFactory.create_inv_pend_model(x, u, t_sampling=0.008),
          gradgen.ModelFactory.create_inv_pend_model(x, u, t_sampling=0.01),
          gradgen.ModelFactory.create_inv_pend_model(x, u, t_sampling=0.05)]

# stage cost function, ell
ell0 = 5 * cs.dot(x, x) + cs.dot(u, u)
ell1 = 3 * cs.dot(x, x) + cs.dot(u, u)
ell2 = 10 * cs.dot(x, x) + 3 * cs.dot(u, u)
ell_list = [ell0, ell1, ell2]

# terminal cost function, vf
vf = 0.5 * cs.dot(x, x)

# gradgen
grad = gradgen.CostGradientStochastic(markov_tree, x, u, f_list, ell_list, vf) \
    .with_name("stoc_ip") \
    .with_target_path("basic_optimizer")
grad.build(no_rust_build=False)
