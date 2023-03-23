import numpy as np
import casadi as cs
import gradgen
import matplotlib.pyplot as plt


p = np.array([[1]])
v = np.array([1])
N = 40
tau = 3
markov_tree = gradgen.MarkovChainScenarioTreeFactory(p, v, N, tau).create()

print(markov_tree)


nx, nu = 2, 1
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
f_list = [gradgen.ModelFactory.create_inv_pend_model(x, u, t_sampling=0.008),
          gradgen.ModelFactory.create_inv_pend_model(x, u, t_sampling=0.01),
          gradgen.ModelFactory.create_inv_pend_model(x, u, t_sampling=0.05)]

# nx, nu = 10, 3
# x = cs.SX.sym('x', nx)
# u = cs.SX.sym('u', nu)
# f_list = [gradgen.ModelFactory.create_quadcopter_model(x, u, t_sampling=0.008)]

# print(cs.jacobian(f_list[0], x))
# stage cost function, ell
ell0 = 5 * cs.dot(x, x) + 0.1 * x[0] + cs.dot(u, u)
ell1 = 3 * cs.dot(x, x) + cs.dot(u, u)
ell2 = 10 * cs.dot(x, x) - 0.1 * x[0] + 3 * cs.dot(u, u)
ell_list = [ell0, ell1, ell2]

# terminal cost function, vf
vf = 0.5 * cs.dot(x, x)


# gradiator
uncertain_gradiator = gradgen.CostGradientStochastic(markov_tree, x, u, f_list, ell_list, vf) \
    .with_name("stochastic_ip") \
    .with_target_path("codegenz")
uncertain_gradiator.build(no_rust_build=False)
