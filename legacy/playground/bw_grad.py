import casadi.casadi as cs
import numpy as np

# This is not so interesting; check out playground/bw_grad_for_loop.py


nx = 3
nu = 3

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


# --- The following are Python functions; later we will generate
# --- C code


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


N = 4
us = np.random.uniform(size=(nu, N))  # just a random sequence of inputs
xs = np.zeros((nx, N+1))  # sequence of states
xs[:, 0] = [1, 0, -1]  # some arbitrary initial state


# BACKWARD METHOD
# Simulate the system
for i in range(N):
    xs[:, i+1] = f_py(xs[:, i], us[:, i]).reshape(nx)

# Then determine the gradient of VN with respect to u(N-1)
nabla_Vf_N_1 = vfx_py(xs[:, N])
z_N_1 = nabla_Vf_N_1
nabla_VN_u_N_1 = ellu_py(xs[:, N-1], us[:, N-1]) + \
    jfu_py(xs[:, N-1], us[:, N-1], z_N_1)

# Gradient of VN with respect o u(N-2)
z_N_2 = jfx_py(xs[:, N-1], us[:, N-1], z_N_1)
ell_u_N_2 = ellu_py(xs[:, N-2], us[:, N-2])
ell_x_N_1 = ellx_py(xs[:, N-1], us[:, N-1])
nabla_Vf_N_2 = ell_u_N_2 + jfu_py(xs[:, N-2], us[:, N-2], ell_x_N_1 + z_N_2)

# Gradient of VN with respect o u(N-3)
z_N_3 = jfx_py(xs[:, N-2], us[:, N-2], z_N_2)
ell_u_N_3 = ellu_py(xs[:, N-3], us[:, N-3])
ell_x_N_2 = ellx_py(xs[:, N-2], us[:, N-2])
nabla_Vf_N_3 = ell_u_N_3 + \
    jfu_py(xs[:, N-3], us[:, N-3], ell_x_N_2 +
           jfx_py(xs[:, N-2], us[:, N-2], ell_x_N_1 + z_N_2))


# But is this correct?
# Let us define VN...
VN = 0
x = xs[:, 0]
u_seq = cs.SX.sym('u_seq', nu, N)

for i in range(N):
    VN = VN + ell_fun(x, u_seq[:, i])
    x = f_fun(x, u_seq[:, i])

VN = VN + vf_fun(x)

# and now let us use CasADi to determine the Jacobian of VN (wrt big-u) directly...
# -- wrt N-2
nabla_VN_u_N_1_sym = cs.jacobian(VN, u_seq[:, N-1])
nabla_VN_u_N_1_fun = cs.Function(
    'nabla_VN_u_N_1', [u_seq], [nabla_VN_u_N_1_sym])
correct_nabla_VN_u_N_1 = nabla_VN_u_N_1_fun(us)

# -- wrt N-2
nabla_VN_u_N_2_sym = cs.jacobian(VN, u_seq[:, N-2])
nabla_VN_u_N_2_fun = cs.Function(
    'nabla_VN_u_N_2', [u_seq], [nabla_VN_u_N_2_sym])
correct_nabla_VN_u_N_2 = nabla_VN_u_N_2_fun(us)

# -- wrt N-3
nabla_VN_u_N_3_sym = cs.jacobian(VN, u_seq[:, N-3])
nabla_VN_u_N_3_fun = cs.Function(
    'nabla_VN_u_N_3', [u_seq], [nabla_VN_u_N_3_sym])
correct_nabla_VN_u_N_3 = nabla_VN_u_N_3_fun(us)

# --- ERRORS ---
err = np.linalg.norm(nabla_VN_u_N_1 - correct_nabla_VN_u_N_1.T)
print(f"Error(N-1) = {err}")
assert err < 1e-6

err = np.linalg.norm(nabla_Vf_N_2-correct_nabla_VN_u_N_2.T)
print(f"Error(N-2) = {err}")
assert err < 1e-6

err = np.linalg.norm(nabla_Vf_N_3 - correct_nabla_VN_u_N_3.T)
print(f"Error(N-3) = {err}")
assert err < 1e-6
