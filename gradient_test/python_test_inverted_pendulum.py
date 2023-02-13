import casadi.casadi as cs
import numpy as np
import subprocess as subp
from time import monotonic
from statistics import mean


N=5
nx, nu = 4, 1
m, I, g, ts = 1, 0.0005, 9.81, 0.01

x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
d = cs.SX.sym('d', nx)

# System dynamics, f
f = cs.vertcat(
    x[0] + ts * x[1],
    x[1] + ts * ((5 / 7) * x[0] * x[3] ** 2 - g * cs.sin(x[2])),
    x[2] + ts * x[3],
    x[3] + ts * ((u[0] - m * g * x[0] * cs.cos(x[2]) - 2 * m * x[0] * x[1] * x[2]) / (m * x[0] ** 2 + I)))

# Stage cost function, ell
ell = 5 * x[0] ** 2 + 0.01 * x[1] ** 2 + 0.01 * x[2] ** 2 + 0.05 * x[3] ** 2 + 2.2 * u ** 2

# terminal cost function, vf
vf = 0.5 * (x[0] ** 2 + 50 * x[1] ** 2 + 100 * x[2] ** 2)

# Make Jacobians
jfx = cs.jacobian(f, x).T @ d
jfu = cs.jacobian(f, u).T @ d
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


# us = np.random.uniform(size=(nu, N))  # just a random sequence of inputs
us = 1.0 * np.ones((nu, N))
xs = np.zeros((nx, N+1))  # sequence of states
xs[:, 0] = [0, 0, 0, 0]  # some arbitrary initial state


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
    # print(grad_N_j_VN.shape)
    # print(grads_matrix.shape)
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
# print(f"Error = {err}")
assert (err < 1)
# print(correct_nabla_VN_u.reshape((nu, N)))


nabla_VN_u_N_fun.generate('nabla.c')
# print(open('nabla.c').read())
# then compile using:
# #gcc -O3 -fPIC -shared nabla.c -o nabla.so
my_compiler = 'gcc'
out = subp.run([my_compiler, '-fPIC', '-O3', '-shared',
               'nabla.c', '-o', 'nabla.so'])
assert ~out.returncode, "compilation failed"
# We can now do
ext_grad_VN = cs.external('nabla_VN_u_N', 'nabla.so')




gamma = 0.1
tol = 5e-4  # desired tolerance
u0 = 1.0 * np.ones((nu, 1))  # initial guess (just 0)
u = u0


df = ext_grad_VN(u)  # compute the gradient
u_new = u - gamma * df  # gradient update
print(df)
print(u_new)


# error_cache = []
#
# max_num_iterations = 100  # maximum number of iterations
# for i in range(max_num_iterations):
#     df = ext_grad_VN(us)  # compute the gradient
#     x_new = x - gamma * df  # gradient update
#     error = np.linalg.norm(x_new - x, np.inf)
#     error_cache += [error]
#     if error < tol:
#         break
#     x = x_new