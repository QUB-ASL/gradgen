import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import gradgen

# System parameters
M = 3
m = 1
L = 0.35
g = 9.81
Mtot = M + m

# Cost parameters
q_theta = 1
q_theta_dot = 0.2
r = 1

# Tree
p = np.array([[0.2, 0.3, 0.5],
              [0.1, 0.25, 0.65],
              [0.05, 0.8, 0.15]])
v_tree = np.array([0.6, 0.1, 0.3])
N = 40
tau = 3
tree = gradgen.MarkovChainScenarioTreeFactory(p, v_tree, N, tau).create()
print(tree)


def f(x, u, t_sampling):
    th, w = x[0], x[1]
    psi = -3 * (m*L*cs.sin(2*th) + u * cs.cos(th) - Mtot*g*cs.sin(th))
    psi /= L * (4*Mtot - 3*m*cs.cos(th)**2)
    f = cs.vertcat(w, psi)
    return cs.simplify(x + t_sampling * f)


def ell(x, u):
    return q_theta * x[0]**2 + q_theta_dot * x[1]**2 + r * u


nx = 2
nu = 1
u = cs.SX.sym('u', nu*tree.num_nonleaf_nodes)
z0 = cs.SX.sym('z0', nx)


cost = ell(z0, u[0])

z_sequence = [None]*tree.num_nonleaf_nodes
z_sequence[0] = z0

c = 0
for i in range(1, tree.num_nonleaf_nodes):  # Looping through all non-leaf nodes
    idx_anc = tree.ancestor_of(i)
    prob_i = tree.probability_of_node(i)

    x_anc = z_sequence[idx_anc]
    u_anc = u[idx_anc*nu:(idx_anc+1)*nu]
    u_current = u[i*nu:(i+1)*nu]

    t_s_current = 0.01 * (1 + 0.5 * tree.event_at_node(i))

    x_current = f(x_anc, u_anc, t_s_current)

    cost += ell(x_current, u_current)

    z_sequence[i] = cs.vertcat(x_current)

bounds = og.constraints.NoConstraints()
problem = og.builder.Problem(u, z0, cost)\
    .with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("basic_optimizer")\
    .with_build_mode("debug")
meta = og.config.OptimizerMeta().with_optimizer_name("big_grad")
solver_config = og.config.SolverConfiguration()

builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config).with_generate_not_build_flag(True)
builder.build()
