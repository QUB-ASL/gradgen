import casadi.casadi as cs
import subprocess as subp

# Define state and input vectors
n, m = 3, 2
x = cs.MX.sym('x', n)
u = cs.MX.sym('u', m)

# Define function f (system dynamics)
f = cs.vertcat(cs.cos(x[0] + cs.asinh(x[1]**2)) + u[0],
               -cs.sin(x[1]) + u[0] * x[2] + u[1],
               x[2] - x[1] + u[0] - u[1]**3)


# Determine the Jacobian matrices:
jfx = cs.jacobian(f, x)
jfu = cs.jacobian(f, u)

# But...
# We don't really want the Jacobians of f...
# The above Jacobians return *matrices*. Instead, we want
# a function that takes x, u, and d in R^n, and return the
# matrix-vector product Jf(x, u)'*d
# >> Less memory, faster computations <<
d = cs.MX.sym('d', n)
jfx_d = jfx.T @ d
jfu_d = jfu.T @ d


# Construct a CasADi function from the Jacobian
# This is a function that takes x and u and returns
# the Jacobian matrix of f with respect to x
jfx_fun = cs.Function('jfx', [x, u, d], [jfx_d], ['x', 'u', 'd'], ['jfx_d'])
jfu_fun = cs.Function('jfu', [x, u, d], [jfu_d], ['x', 'u', 'd'], ['jfu_d'])

# Now let us generate code - we will put both functions in the
# same C file
codegen = cs.CodeGenerator('jacobians.c')
codegen.add(jfx_fun)
codegen.add(jfu_fun)
codegen.generate()


# Next you need to compute the generated code; on Linux you
# can run the following code:
#
# $ gcc -fPIC -shared jacobians.c -o jacobians.so
#
# This will create the file jacobians.so
# The following line of code will compile the C file
my_compiler = 'gcc'  # choose your compiler (clang, gcc or other)
out = subp.run([my_compiler, '-fPIC', '-O3', '-shared',
               'jacobians.c', '-o', 'jacobians.so'])
assert ~out.returncode, "compilation failed"

# We can now do
jfx_d_c = cs.external('jfx', 'jacobians.so')
val2 = jfx_d_c([1.45, 9, 28], [15, -1], [1, 1, 1])  # this is a DM object
print(f"J_x f(x, u)' * d = \n{val2.full()}")
