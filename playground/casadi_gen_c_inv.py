import casadi.casadi as cs
import numpy as np
import subprocess as subp
from time import monotonic
from statistics import mean
import random


N=5


def create_inv_pend_model(x, u, m=1, ell=1, M=1, g=9.81, t_sampling=0.01):
    # Check input dimensions
    def _check_state_input_dims(x, u, nx, nu):
        assert x.shape == (nx, 1), "Invalid state dimensions"
        assert u.shape == (nu, 1), "Invalid input dimensions"

    _check_state_input_dims(x, u, 2, 1)

    # Extract state variables
    th, w = x[0], x[1]

    # Compute total mass and psi
    Mtot = M + m
    psi = -3 * (m * ell * cs.sin(2 * th) + u * cs.cos(th) - Mtot * g * cs.sin(th))
    psi /= ell * (4 * Mtot - 3 * m * cs.cos(th) ** 2)

    # Construct the system dynamics
    f = cs.vertcat(w, psi)

    # Discretize the system dynamics
    x_next = x + t_sampling * f
    x_next = cs.simplify(x_next)

    return x_next


nx, nu = 2, 1

# Create symbolic variables
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
d = cs.SX.sym('d', nx)

# Create the model function
f = create_inv_pend_model(x, u)
# Stage cost function, ell
ell = cs.dot(x, x) + cs.dot(u, u)

# terminal cost function, vf
vf = 10 * cs.dot(x, x)


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


# sequence of inputs
us = 0.1 * np.ones((nu, N))
# sequence of states
xs = np.zeros((nx, N+1))
# ramdom initial state
x1 = random.uniform(-0.524,0.524)
x2 = random.uniform(-1.0,1.0)
xs[:, 0] = [x1, x2]

# generate gradient
VN = 0
xt = xs[:, 0]
u_seq = cs.SX.sym('u_seq', nu, N)

for i in range(N):
    VN = VN + ell_fun(xt, u_seq[i*nu:(i+1)*nu])
    xt = f_fun(xt, u_seq[i*nu:(i+1)*nu])

VN = VN + vf_fun(xt)
nabla_VN_u_N = cs.jacobian(VN, u_seq)
nabla_VN_u_N_fun = cs.Function('nabla_VN_u_N', [u_seq], [nabla_VN_u_N])
correct_nabla_VN_u = nabla_VN_u_N_fun(us)


# ----------------------------------------------------
# err = np.linalg.norm(
#     grads_matrix - correct_nabla_VN_u.reshape((nu, N)), np.inf)
# # print(f"Error = {err}")
# assert (err < 1)
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





# ---------------------


runtime = 50000
a = []

for t in range(0, runtime):
    start_time = monotonic()
    result_from_so = ext_grad_VN(us)
    end_time = (monotonic() - start_time) * 1e6
    a.append(end_time)
average = mean(a)
stda = np.std(a)
print(N, average, stda)


