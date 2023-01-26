import casadi.casadi as cs
import numpy as np

# In this scirpt we test the correctness of the backward method
# (using a for loop)

nx = 3
nu = 3
N = 6

x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
d = cs.SX.sym('d', nx)
f = cs.sin(x) + 2*u
ell = 0.5 * (cs.dot(x, x) + cs.dot(u, u))
vf = 20 * cs.dot(x, x)

# Make Jacobians
jfx = cs.jacobian(f, x) @ d
jfu = cs.jacobian(f, u) @ d
ellx = cs.jacobian(ell, x).T
ellu = cs.jacobian(ell, u).T
vfx = cs.jacobian(vf, x).T


f_fun = cs.Function('f', [x, u], [f])
jfx_fun = cs.Function('jfx', [x, u, d], [jfx])
jfu_fun = cs.Function('jfu', [x, u, d], [jfu])
ell_fun = cs.Function('ell', [x, u], [ell])
ellx_fun = cs.Function('ellx', [x, u], [ellx])
ellu_fun = cs.Function('ellu', [x, u], [ellu])
vf_fun = cs.Function('vf', [x], [vf])
vfx_fun = cs.Function('vfx', [x], [vfx])


def f_py(x_, u_):
    return f_fun(x_, u_).full()


def jfx_py(x_, u_, d_):
    return jfx_fun(x_, u_, d_).full()


def jfu_py(x_, u_, d_):
    return jfu_fun(x_, u_, d_).full()


def ellx_py(x_, u_):
    return ellx_fun(x_, u_).full()


def ellu_py(x_, u_):
    return ellu_fun(x_, u_).full()


def vfx_py(x_):
    return vfx_fun(x_).full()


us = np.random.uniform(size=(nu, N))  # just a random sequence of inputs
xs = np.zeros((nx, N+1))  # sequence of states
xs[:, 0] = [1, 0, -1]  # some arbitrary initial state


# ----------------------------------------------------
# BACKWARD METHOD
# ----------------------------------------------------
# Simulate the system
for i in range(N):
    xs[:, i+1] = f_py(xs[:, i], us[:, i]).reshape(nx)

grads_matrix = np.zeros((nu, N))
w = vfx_py(xs[:, N])
for j in range(1, N+1):
    ellu_N_j = ellu_py(xs[:, N-j], us[:, N-j])
    ellx_N_j = ellx_py(xs[:, N-j], us[:, N-j])
    fu_N_j_w = jfu_py(xs[:, N-j], us[:, N-j], w)
    fx_N_j_w = jfx_py(xs[:, N-j], us[:, N-j], w)
    grad_N_j_VN = ellu_N_j + fu_N_j_w
    w = ellx_N_j + fx_N_j_w
    grads_matrix[:, N-j] = grad_N_j_VN.T

# ----------------------------------------------------


# Testing (comparison with CasADi's result)...
VN = 0
x = xs[:, 0]
u_seq = cs.SX.sym('u_seq', nu, N)

for i in range(N):
    VN = VN + ell_fun(x, u_seq[:, i])
    x = f_fun(x, u_seq[:, i])

VN = VN + vf_fun(x)
nabla_VN_u_N = cs.jacobian(VN, u_seq)
nabla_VN_u_N_fun = cs.Function(
    'nabla_VN_u_N', [u_seq], [nabla_VN_u_N])
correct_nabla_VN_u = nabla_VN_u_N_fun(us)

err = np.linalg.norm(
    grads_matrix - correct_nabla_VN_u.reshape((nu, N)), np.inf)
print(f"Error = {err}")
assert (err < 1e-6)
