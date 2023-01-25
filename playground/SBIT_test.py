from casadi import *
import casadi.casadi as cs
nx = 3
nu = 3
N = 3

x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
d = cs.SX.sym('u', nx)
f = cs.sin(x) + 2*u
ell = 0.5 * (cs.dot(x, x) + cs.dot(u, u))
vf = 20 * cs.dot(x, x)

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


us = np.random.uniform(size=(nu, N))  # just a random sequence of inputs
xs = np.zeros((nx, N+1))  # sequence of states
xs[:, 0] = [1, 0, -1]  # some arbitrary initial state



# BACKWARD METHOD
# Simulate the system

for t in range(N):
    xs[:, t+1] = f_py(xs[:, t], us[:, t]).reshape(nx)


g_VN_t = []
g_VN_t_cache = []
z = vfx_py(xs[:, N])
Lambda = np.zeros([nx, N-1])
Lmbda_store = np.zeros((1, nx))


for t in reversed(range(0, N)):
    g_VN_t = 0
    sum = np.zeros((1, nx))
    Lambda_store = np.zeros((1, nx))
    g_VN_t = ellu_py(xs[:, t], us[:, t])
    if t != N-1:
        z = jfx_py(xs[:, t + 1], us[:, t + 1], z)
    sum += z.reshape((nx,))
    for b in reversed(range(0, N - 1 - t)):
        if b != 0:
            Lambda[:, b] = jfx_py(xs[:, t], us[:, t], Lambda[:, b-1]).reshape((nx,))
        elif b == 0:
            Lambda[:, b] = ellx_py(xs[:, t+1], us[:, t+1]).reshape((nx,))
        Lambda_store += Lambda[:, b]
    sum += Lambda_store.reshape((nx,))
    g_VN_t += jfu_py(xs[:, t], us[:, t], sum)
    g_VN_t_cache.extend(g_VN_t)


print(g_VN_t_cache)

# Let us define VN
VN = 0
x = xs[:, 0]
u_seq = cs.SX.sym('u_seq', nu, N)
# print(x)
# print(u_seq )
for i in range(N):
    VN = VN + ell_fun(x, u_seq[:, i])
    x = f_fun(x, u_seq[:, i])       # update xt

VN = VN + vf_fun(x)    # here x is xN

# and now let us use CasADi to determine the Jacobian of VN (wrt big-u) directly
nabla_VN_u_N_1_sym = cs.jacobian(VN,u_seq[:,N-1])    #all function
nabla_VN_u_N_1_fun = cs.Function('nabla_VN_u_N_1', [u_seq], [nabla_VN_u_N_1_sym])
correct_nabla_VN_u_N_1 = nabla_VN_u_N_1_fun(us)
print('nabla_VN_u_N_1',correct_nabla_VN_u_N_1)

nabla_VN_u_N_2_sym = cs.jacobian(VN,u_seq[:,N-2])    #all function
nabla_VN_u_N_2_fun = cs.Function('nabla_VN_u_N_2', [u_seq], [nabla_VN_u_N_2_sym])
correct_nabla_VN_u_N_2 = nabla_VN_u_N_2_fun(us)
print('nabla_VN_u_N_2',correct_nabla_VN_u_N_2)

nabla_VN_u_N_3_sym = cs.jacobian(VN,u_seq[:,N-3])    #all function
nabla_VN_u_N_3_fun = cs.Function('nabla_VN_u_N_3', [u_seq], [nabla_VN_u_N_3_sym])
correct_nabla_VN_u_N_3 = nabla_VN_u_N_3_fun(us)
print('nabla_VN_u_N_3',correct_nabla_VN_u_N_3)

nabla_VN_u_N_sym = cs.jacobian(VN,u_seq)    #all function
nabla_VN_u_N_fun = cs.Function('nabla_VN_u_N', [u_seq], [nabla_VN_u_N_sym])
correct_nabla_VN_u_N = nabla_VN_u_N_fun(us)
print('nabla_VN_u_N',correct_nabla_VN_u_N)

# err = np.linalg.norm(g_VN_t_cache.T - correct_nabla_VN_u_N_1)
# print(f"Error = {err}")
# assert err < 1e-6
