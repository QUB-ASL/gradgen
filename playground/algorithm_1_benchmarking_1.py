import casadi.casadi as cs
import numpy as np
import gradgen
import opengen as og

N = 5

# x = cs.SX.sym('x', 2)
# u = cs.SX.sym('x', 1)
# f = gradgen.ModelFactory.create_inv_pend_model(x, u)
# ell = cs.dot(x, x) + u**2
# vf = 10 * cs.dot(x, x)


# Generate code with Gradgen
# The generated code is at codegenz/inv_pend_benchmark
# gradiator = gradgen.CostGradient(x, u, f, ell, vf, N)
# gradiator.with_target_path("codegenz").with_name("inv_pend_benchmark").build()

# This is the system dynamics:


# Generate code with Casadi (via OpEn)
# First we need to define the total cost function:
cost = 0
x0 = cs.SX.sym('x0', 2)
u_seq = cs.SX.sym('u_seq', N)
x = x0
for t in range(N):
    psi = -3 * (cs.sin(2*x[0]) + u_seq[t] * cs.cos(x[0]) -
                10*cs.sin(x[0])) / (8 - 3*cs.cos(x[0])**2)
    x = x + 0.5 * cs.vertcat(x[1], psi)
    cost = cost + cs.dot(x, x) + cs.dot(u_seq[t], u_seq[t])

cost = cost + 10 * cs.dot(x, x)


# Generate CasADi code via OpEn
# This will generate C code at:
#   basic_optimizer/inv_pend_benchmark_casadi/icasadi_inv_pend_benchmark_casadi/extern/auto_casadi_grad.c
#
# and a Rust wrapper at
# basic_optimizer/inv_pend_benchmark_casadi/icasadi_inv_pend_benchmark_casadi/
#
# Have a look at
# basic_optimizer/inv_pend_benchmark_casadi/icasadi_inv_pend_benchmark_casadi/src/lib.rs
#
# There you will find the function:
# pub fn grad(u: &[f64], xi: &[f64], static_params: &[f64], cost_jacobian: &mut [f64]) -> i32
#
# Have a look at the test function tst_call_grad (line 303) to see how to use it
bounds = og.constraints.BallInf(None, 100.0)
problem = og.builder.Problem(u_seq, x0, cost).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("optimizers").with_tcp_interface_config()
meta = og.config.OptimizerMeta().with_optimizer_name("ipb")
solver_config = og.config.SolverConfiguration()

builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()

mng = og.tcp.OptimizerTcpManager('optimizers/ipb')
mng.start()
response = mng.call([1.0, 50.0])
print(response["solution"])
mng.kill()