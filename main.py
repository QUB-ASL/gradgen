import gradgen as grad
import casadi.casadi as cs


nx = 3
nu = 2
N = 6
x = cs.SX.sym('x', nx)
u = cs.SX.sym('u', nu)
f = cs.sin(x) + 2*(u[0]+u[1]**3)*x
ell = 0.5 * (cs.dot(x, x) + cs.dot(u, u))
vf = 20 * cs.dot(x, x)

gradObj = grad.CostGradient(x, u, f, ell, vf, N)
gradObj.build()
