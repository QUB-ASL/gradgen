import casadi.casadi as cs
import numpy as np
import subprocess as subp
from time import monotonic
from statistics import mean
import random
# np.set_printoptions(precision=16)
# cs.SX.set_precision(64)



N=30
n_pred_max = 6
nx, nu = 10, 3
k1, k2, Ts = 252, 20, 1 / 125
Gamma_u = np.diag([0, 0, 124.659])
Gamma_n = np.diag([2.287469, 1.35699886, -0.42388942])
I = np.diag([0.01788, 0.03014, 0.04614])
Iinv = np.linalg.inv(I)

x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
d = cs.SX.sym('d', nx)

(q0, q1, q2, q3, wx, wy, wz, n_x, ny, nz) = (
    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
Q_bar_q = cs.vertcat(cs.vertcat(0, -x[5], -x[6], -x[7]).T,
                     cs.vertcat(x[5], 0, x[7], -x[6]).T,
                     cs.vertcat(x[6], -x[7], 0, x[5]).T,
                     cs.vertcat(x[7], x[6], -x[5], 0).T)
tau = cs.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
expQbar = cs.vertcat(
    cs.vertcat(cs.cosh(tau), -wx * tau, -wy * tau, -wz * tau).T,
    cs.vertcat(wx * tau, cs.cosh(tau), wz * tau, wy * tau).T,
    cs.vertcat(wy * tau, -wz * tau, cs.cosh(tau), wx * tau).T,
    cs.vertcat(wz * tau, wy * tau, -wx * tau, cs.cosh(tau)).T)
q = cs.vertcat(q0, q1, q2, q3)
w = cs.vertcat(wx, wy, wz)
n = cs.vertcat(n_x, ny, nz)
f = cs.vertcat((Ts / 2) * expQbar @ q,
               w + Ts * (Gamma_n @ n + Gamma_u @ u - Iinv @ cs.cross(w, I @ w)),
               n + Ts * (k1 * k2 * u - k2 * n)
               )

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
# random initial state
# Generate some random initial angles between -5 and 5 degrees
angles_deg = np.random.uniform(low=-5, high=5, size=3)
angles_rad = np.deg2rad(angles_deg)

# Convert the angles to a quaternion
yaw, pitch, roll = angles_rad
cphi = np.cos(roll / 2)
ctheta = np.cos(pitch / 2)
cpsi = np.cos(yaw / 2)
sphi = np.sin(roll / 2)
stheta = np.sin(pitch / 2)
spsi = np.sin(yaw / 2)
q = [cphi * ctheta * cpsi + sphi * stheta * spsi,
     sphi * ctheta * cpsi - cphi * stheta * spsi,
     cphi * stheta * cpsi + sphi * ctheta * spsi,
     cphi * ctheta * spsi - sphi * stheta * cpsi]

w1 = random.uniform(-0.1, 0.1)
w2 = random.uniform(-0.1, 0.1)
w3 = random.uniform(-0.1, 0.1)

n1 = random.uniform(-0.2, 0.2)
n2 = random.uniform(-0.2, 0.2)
n3 = random.uniform(-0.2, 0.2)

xs[:, 0] = [q[0], q[1], q[2], q[3], w1, w2, w3, n1, n2, n3]


# generate gradient
VN = 0
xt = xs[:, 0]
u_seq = cs.SX.sym('u_seq', nu, N)

for i in range(N):
    VN = VN + ell_fun(xt, u_seq[i*nu:(i+1)*nu])
    # print(u_seq[i*nu:(i+1)*nu])
    xt = f_fun(xt, u_seq[i*nu:(i+1)*nu])

VN = VN + vf_fun(xt)
nabla_VN_u_N = cs.jacobian(VN, u_seq)
# nabla_VN_u_N = cs.gradient(VN, u_seq).T
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

runtime = 5000
a = []

print()

for t in range(0, runtime):
    start_time = monotonic()
    result_from_so = ext_grad_VN(us)
    end_time = (monotonic() - start_time) * 1e6
    a.append(end_time)
average = mean(a)
stda = np.std(a)
print(N, average, stda)

