import casadi.casadi as cs
import numpy as np
import gradgen
import opengen as og

x = cs.SX.sym('x', 2)
u = cs.SX.sym('x', 1)
f = gradgen.ModelFactory.create_inv_pend_model(x, u)
ell = cs.dot(x, x) + u**2
vf = 10 * cs.dot(x, x)
N = 30

# With Gradgen
gradiator = gradgen.CostGradient(x, u, f, ell, vf, N)
gradiator.with_target_path("codegenz").with_name("inv_pend_benchmark").build()


def f(x, u, m=1, L=1, M=1, g=9.81, t_sampling=0.01):
    th, w = x[0], x[1]
    Mtot = M + m
    psi = -3 * (m*L*cs.sin(2*th) + u * cs.cos(th) - Mtot*g*cs.sin(th))
    psi /= L * (4*Mtot - 3*m*cs.cos(th)**2)
    f = cs.vertcat(w, psi)
    return cs.simplify(x + t_sampling * f)


# With Casadi
cost = 0
x0 = cs.SX.sym('x0', 2)
u_seq = cs.SX.sym('u_seq', N)
x = x0
for t in range(N):
    u = u_seq[t]
    cost += cs.dot(x, x) + u**2
    x = f(x, u)

cost += 10 * cs.dot(x, x)

problem = og.builder.Problem(u_seq, x0, cost)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("basic_optimizer")
meta = og.config.OptimizerMeta().with_optimizer_name("inv_pend_benchmark_casadi")
solver_config = og.config.SolverConfiguration()

builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config).with_generate_not_build_flag(True)
builder.build()
