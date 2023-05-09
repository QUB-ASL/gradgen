import casadi.casadi as cs
import numpy as np
import gradgen
import opengen as og

x = cs.SX.sym('x', 10)
u = cs.SX.sym('u', 3)
f = gradgen.ModelFactory.create_quadcopter_model(x, u)
ell = cs.dot(x, x) + u**2
vf = 10 * cs.dot(x, x)
N = 30

# # With Gradgen
gradiator = gradgen.CostGradient(x, u, f, ell, vf, N)
gradiator.with_target_path("codegenz").with_name(
    "quadcopter_benchmark").build()


def f(x, u,
      t_sampling=0.01,
      gamma_u_z=125.,
      gamma_n=[-2.3, -1.4, 0.4],
      moi=[0.02, 0.02, 0.05],
      k1=252, k2=20):
    q = x[0:4]
    w = x[4:7]
    n = x[7:]
    norm_w = cs.norm_2(w)
    tau = cs.sin(norm_w)/norm_w
    wx_tau = w[0] * tau
    wy_tau = w[1] * tau
    wz_tau = w[2] * tau
    csw = cs.cos(norm_w)
    expQw = cs.vertcat(
        cs.horzcat(csw, -wx_tau, -wy_tau, -wz_tau),
        cs.horzcat(wx_tau, csw, wz_tau, -wx_tau),
        cs.horzcat(wy_tau, -wz_tau, csw, wx_tau),
        cs.horzcat(wz_tau, wy_tau, -wx_tau, csw)
    )
    fq = 0.5 * t_sampling * (expQw @ q)
    gamma_u = np.diagflat([0, 0, gamma_u_z])
    gamma_n = np.diagflat(gamma_n)
    moi = np.diagflat(moi)
    moi_inv = np.linalg.inv(moi)
    fw = w + t_sampling * (gamma_u @ u + gamma_n @
                           n - moi_inv @ cs.cross(w, moi @ w))
    fn = n + t_sampling * (k1*k2*u - k2*n)
    return cs.vertcat(fq, fw, fn)


# # With Casadi
cost = 0
x0 = cs.SX.sym('x0', 10)
u_seq = cs.SX.sym('u_seq', N*3)
x = x0
for t in range(N):
    u = u_seq[3*t: 3*(t+1)]
    cost += cs.dot(x, x) + cs.dot(u, u)
    x = f(x, u)

cost += 10 * cs.dot(x, x)


problem = og.builder.Problem(u_seq, x0, cost)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("basic_optimizer")
meta = og.config.OptimizerMeta().with_optimizer_name("quadcopter_benchmark_casadi")
solver_config = og.config.SolverConfiguration()

builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config).with_generate_not_build_flag(True)
builder.build()
