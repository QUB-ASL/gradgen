from casadi import *
import casadi.casadi as cs

nx = 3
nu = 3
N = 3

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


def ell_py(x_, u_):
    return ell_fun(x_, u_).full()


def ellx_py(x_, u_):
    return ellx_fun(x_, u_).full()


def ellu_py(x_, u_):
    return ellu_fun(x_, u_).full()


def vfx_py(x_):
    return vfx_fun(x_).full()


def et(t):
    e = np.zeros((nu, N*nu))
    e[:, nu*t:nu*(t+1)] = np.eye(nu)
    return e

us = np.random.uniform(size=(nu, N))  # just a random sequence of inputs
xs = np.zeros((nx, N + 1))  # sequence of states              
xs[:, 0] = [1, 0, -1]  # some arbitrary initial state

JF_this_iteration = np.zeros((nx, nu * N))
JF_cache = []
JF0 = np.zeros((nx, nu * N))
JF_cache = [JF0]
g_ell_sum = 0

# FORWARD METHOD
# Simulate the system

for t in range(0, N):
    jac_fx_t = jfx_py(xs[:, t], us[:, t], JF_this_iteration)
    jac_fu_t = jfu_py(xs[:, t], us[:, t], et(t))
    JF_t_plus_1 = jac_fx_t + jac_fu_t
    g_ellx_t = ellx_py(xs[:, t], us[:, t])
    g_ellu_t = ellu_py(xs[:, t], us[:, t])
    g_ell = JF_this_iteration.T @ g_ellx_t + et(t).T @ g_ellu_t
    g_ell_sum += g_ell
    JF_this_iteration = JF_t_plus_1
    xs[:, t + 1] = f_py(xs[:, t], us[:, t]).reshape(nx)

# print(g_ell_sum, g_ell_sum.shape)
g_Vf = g_ell_sum + JF_t_plus_1.T @ vfx_py(xs[:, N])
print(g_Vf, g_Vf.shape)


# Let us define VN
VN = 0
x = xs[:, 0]
u_seq = cs.SX.sym('u_seq', nu, N)

for i in range(N):
    VN = VN + ell_fun(x, u_seq[:, i])
    x = f_fun(x, u_seq[:, i])       # update xt

VN = VN + vf_fun(x)    # here x is xN

# and now let us use CasADi to determine the Jacobian of VN (wrt big-u) directly
nabla_VN_u_sym = cs.jacobian(VN, u_seq)
nabla_VN_u_fun = cs.Function('nabla_VN_u', [u_seq], [nabla_VN_u_sym])
correct_nabla_VN_u = nabla_VN_u_fun(us)
print(correct_nabla_VN_u)
err = np.linalg.norm(g_Vf.T - correct_nabla_VN_u)
print(f"Error = {err}")
assert err < 1e-6
