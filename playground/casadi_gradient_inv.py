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


# us = np.random.uniform(size=(nu, N))  # just a random sequence of inputs
us = 0.1 * np.ones((nu, N))
xs = np.zeros((nx, N+1))  # sequence of states
# xs[:, 0] = 0.1 * np.ones(nx)  # some arbitrary initial state
x3 = random.uniform(-0.524,0.524)
x4 = random.uniform(-1.0,1.0)
# xs[:, 0] = [x3, x4]
xs[:, 0] = [0.1, 0.1]

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
# generate f

VN = 0
x_seq = cs.SX.sym('z_seq', nx, N+1)
u_seq = cs.SX.sym('u_seq', nu, N)
x_t = x_seq[:, 0]

for i in range(N):
    VN = VN + ell_fun(x_t, u_seq[i * nu:(i + 1) * nu])
    x_t = f_fun(x_t, u_seq[i * nu:(i + 1) * nu])

VN = VN + vf_fun(x_t)
    # return VN

VN_fun = cs.Function('VN', [x_seq, u_seq], [VN])

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


VN_fun.generate('VN.c')
my_compiler = 'gcc'
out = subp.run([my_compiler, '-fPIC', '-O3', '-shared',
               'VN.c', '-o', 'VN.so'])
assert ~out.returncode, "compilation failed"
# We can now do
ext_VN = cs.external('VN', 'VN.so')




# ---------------------



n_runs =50
max_num_iterations = 1000


# for N in range(5, n_pred_max+1):
store = []   # storage of computation time
box = []  # storage of iteration
k = 1
j = 1
sign = True

for _j in range(1, n_runs):
    j = _j - k
    sign = True

    tol = 1e-5
    re_tol = 1e-4
    vn = 0.0
    f_new = 0.0

    # initial states
    x_seq = np.zeros((nx, N + 1))
    x3 = random.uniform(-0.524, 0.524)
    x4 = random.uniform(-1.0, 1.0)
    # x_seq[:, 0] = [0.1, 0.1]
    x_seq[:, 0] = [x3, x4]


    # initial input guess
    u_seq = 0.1 * np.ones((nu, N))
    # print("u_seq,shape.u_seq", u_seq, u_seq.shape)


    # ** value for Armijo condition
    c1 = 0.5
    update=0.5
    alpha = 1

    # p = 0.05

    break_outer_loop = False

    start_time = monotonic()
    for v in range(1, max_num_iterations):
        print("u_seq", u_seq)
        # print("x_seq[:, 0]", x_seq[:, 0])
        df = ext_grad_VN(u_seq.reshape((nu, N)))  # compute the gradient
        f = ext_VN(x_seq , u_seq)
        nrm = np.dot(df, df.T)
        print("df", df)
        print("f", f)
        print("nrm", nrm)


        if v == 1:
            epsilon = 1e-9*np.ones((1, nu*N))
            epsilon_n = 1e-9
            r = 1*np.ones((1, nu*N))

            grad_f_x0 = df
            u_seq_new = u_seq.reshape((1,nu* N)) + epsilon_n * r

            grad_f_x1 = ext_grad_VN(u_seq_new.reshape((nu, N)))
            numerator = np.linalg.norm(grad_f_x1-grad_f_x0)
            denominator = np.linalg.norm(r * epsilon)
            L =numerator / denominator
            alpha =0.99 / L
            print("alpha_in",alpha)


        if v >= 2:
            norm_grad_prev = np.dot(grad_prev, grad_prev.T)
            norm_grad = np.dot(df, df.T)
            alpha = alpha * norm_grad_prev / norm_grad
            print("alpha_in", alpha)



        u_seq_new = u_seq.reshape((1, nu*N)) - alpha * df # gradient update
        # u_seq_new = np.maximum(p, u_seq_new.reshape((1, nu*N)))
        # print(df)
        # print(dff)
        # print("u_seq", u_seq)
        # print("u_seq_new",u_seq_new)

        f_new = ext_VN(x_seq,u_seq_new.reshape((nu, N)))
        while f_new > f - c1 * alpha * nrm:
            alpha *= update
            print("alpha", alpha)
            if alpha < 1e-8:
                print("alpha too small")
                v=v+0.01
                sign = False
                # break_outer_loop
                break_outer_loop = True
                break
            u_seq_new = u_seq.reshape((1, nu * N)) - alpha * df
            # u_seq_new = np.maximum(p, u_seq_new.reshape((1, nu*N)))
            f_new = ext_VN(x_seq, u_seq_new.reshape((nu,N)))
        print("u_seq_new", u_seq_new)
        print("u_seq", u_seq)
        error = np.linalg.norm(u_seq_new.reshape((1, nu*N)) - u_seq.reshape((1, nu*N)), np.inf)
        print("v", v,"error", error,"alpha",alpha)


        if v == 1:
            re_tol = tol * error
            tol = tol + re_tol

        if error < tol:
            box.append(v)
            break
        if v == max_num_iterations:
            sign = False
            break
        u_seq = u_seq_new.reshape((nu, N))
        print("---------------------")
        grad_prev= df
        if break_outer_loop:
            break

    # print("u_seq_new", u_seq_new)
    if sign:
        end_time = (monotonic() - start_time) * 1e6
        store.append(end_time)
        # print("store", store)
        # print("j", j)
    else:
        end_time = (monotonic() - start_time) * 1e6
        k += 1
        # print("_j", _j)

# print("j", j)
average = mean(store)
stda = np.std(store)
print(N, average, stda)
# if the iteration has .1 means in this run alpha is too small
print("box", box)



