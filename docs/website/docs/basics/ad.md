---
sidebar_position: 3
---

# Automatic Differentiation

The current AD layer supports both forward-mode and reverse-mode symbolic differentiation.

Forward-mode entry points:

- `derivative(expr, wrt)` for scalar derivatives
- `jvp(expr, wrt, tangent)` for Jacobian-vector products

Reverse-mode entry points:

- `gradient(expr, wrt)` for scalar-output reverse gradients
- `vjp(expr, wrt, cotangent)` for vector-Jacobian products

Higher-order helpers:

- `jacobian(expr, wrt)`
- `hessian(expr, wrt)` for scalar-output expressions

Smooth elementary functions such as `atan2`, `hypot`, `asinh`, `acosh`,
`atanh`, `cbrt`, `erf`, and `erfc` participate in AD. Nonsmooth functions such
as `min`, `floor`, `ceil`, `round`, `trunc`, `fract`, `signum`, `norm1`, and
`norm_inf` currently raise if differentiation is requested. `norm_p(...)` and
`norm_p_to_p(...)` support AD when `p` is a constant greater than `1`.

### Forward mode

### Scalar derivative

```python
from gradgen import Function, SX, derivative

x = SX.sym("x")
expr = x.sin() + x * x
dexpr = derivative(expr, x)

df = Function("df", [x], [dexpr])
print(df(2.0))
```

### Vector directional derivative

```python
from gradgen import Function, SXVector, jvp

x = SXVector.sym("x", 2)
expr = x.dot(x)
directional = jvp(expr, x, [1.0, 0.0])

df = Function("df", [x], [directional])
print(df([3.0, 4.0]))  # 6.0
```

### Differentiating a `Function`

You can also build a new function representing a forward-mode directional derivative of an existing `Function`:

```python
from gradgen import Function, SX

x = SX.sym("x")
y = SX.sym("y")
f = Function("f", [x, y], [x * y + x.sin()])

df_dx = f.jvp(1.0, 0.0)
df_dy = f.jvp(0.0, 1.0)
```

The returned object is another `Function` with the same primal inputs and differentiated outputs.

### Reverse mode

Reverse mode is especially useful for scalar-output expressions with many inputs.

### Scalar gradient

```python
from gradgen import Function, SX, gradient

x = SX.sym("x")
y = SX.sym("y")
expr = x * y + x.sin()

grad_x = gradient(expr, x)
grad_y = gradient(expr, y)

g = Function("g", [x, y], [grad_x, grad_y])
print(g(2.0, 5.0))
```

### Vector-Jacobian product

```python
from gradgen import Function, SX, SXVector, vjp

x = SX.sym("x")
outputs = SXVector((x.sin(), x * x))
sensitivity = vjp(outputs, x, [2.0, 3.0])

g = Function("g", [x], [sensitivity])
print(g(2.0))
```

### Reverse-mode differentiation of a `Function`

You can also build a reverse-mode differentiated function from cotangent seeds on the outputs:

```python
from gradgen import Function, SX

x = SX.sym("x")
y = SX.sym("y")
f = Function("f", [x, y], [x * y + x.sin()])

reverse = f.vjp(1.0)
print(reverse(2.0, 5.0))
```

`Function.vjp(...)` returns a new `Function` with:

- the same primal inputs as the original function
- outputs ordered like the original inputs
- values equal to the vector-Jacobian product for the supplied cotangent direction

If you want a runtime-seeded reverse-mode function instead, provide `wrt_index`:

```python
from gradgen import Function, SXVector

x = SXVector.sym("x", 2)
G = Function(
    "G",
    [x],
    [SXVector((x[0] + x[1], x[0] * x[1], x[1].sin()))],
    input_names=["x"],
    output_names=["y"],
)

reverse_x = G.vjp(wrt_index=0)
print(reverse_x([3.0, 4.0], [2.0, -1.0, 5.0]))
```

This returns a `Function` with:

- the original primal inputs
- one appended cotangent input per declared output
- a single output block equal to $J_G(x)^\top v$ for the selected input block

For scalar-output functions, there is also a high-level `Function.gradient(...)` helper:

```python
grad_f = some_scalar_function.gradient(0)
```

This returns a new `Function` whose outputs represent the gradient with respect to the selected input block.

### Jacobians

The library also supports symbolic Jacobian construction.

```python
from gradgen import Function, SXVector, jacobian

x = SXVector.sym("x", 2)
expr = x.dot(x)

jac = jacobian(expr, x)
f = Function("jac", [x], [jac])
print(f([3.0, 4.0]))  # (6.0, 8.0)
```

For vector-output by vector-input cases, Jacobians are currently represented as
flat row-major vectors because full matrix types are not implemented yet.

For example:

```python
from gradgen import Function, SXVector

x = SXVector.sym("x", 2)
G = Function(
    "G",
    [x],
    [SXVector((x[0] + x[1], x[0] * x[1], x[1].sin()))],
    input_names=["x"],
    output_names=["y"],
)

JG = G.jacobian(0)
print(JG([3.0, 4.0]))  # (1.0, 1.0, 4.0, 3.0, 0.0, cos(4.0))
```

This corresponds to the $3 \times 2$ matrix

$$\begin{bmatrix}
1 & 1 \\
4 & 3 \\
0 & \cos(4)
\end{bmatrix}$$

stored row by row.

You can also derive a Jacobian block from a function:

```python
jac_f = some_function.jacobian(0)
```

If you want multiple input blocks at once, use:

```python
blocks = some_function.jacobian_blocks()
```

For Rust code generation workflows that want one kernel to compute several
artifacts together, you can also build a combined symbolic function:

```python
joint = some_function.joint(("f", "jf"), 0, simplify_joint="high")
```

This low-level `Function.joint(...)` API operates on one selected `wrt` block at
a time. Supported component names are:

- `"f"` for the primal outputs
- `"grad"` for the gradient block
- `"jf"` for the Jacobian block
- `"hessian"` for the Hessian block
- `"hvp"` for the Hessian-vector product

The output order follows the requested component tuple. If `"hvp"` is included,
the returned function appends a tangent input block.

### Hessians

Hessians are supported for scalar-output expressions.

```python
from gradgen import Function, SXVector, hessian

x = SXVector.sym("x", 2)
expr = x[0] * x[0] + x[0] * x[1] + x[1] * x[1]

hes = hessian(expr, x)
f = Function("hes", [x], list(hes))
print(f([3.0, 4.0]))  # ((2.0, 1.0), (1.0, 2.0))
```

At the function level:

```python
hes_f = some_scalar_function.hessian(0)
```

For multiple input blocks:

```python
blocks = some_scalar_function.hessian_blocks()
```

### Hessian-vector products

For scalar-output functions, you can also build Hessian-vector product kernels.
The resulting function keeps the original inputs and appends one extra tangent
input block:

```python
hvp_f = some_scalar_function.hvp(0)
```

To build several blocks at once:

```python
blocks = some_scalar_function.hvp_blocks()
```

## Simplification

The library includes a bounded symbolic simplifier for expressions and functions.

```python
from gradgen import SX, derivative, simplify

x = SX.sym("x")
expr = derivative(x * x, x)

print(expr)                         # unsimplified symbolic derivative
print(simplify(expr, "medium"))     # cleaner symbolic expression
```

Supported simplification controls:

- integer pass counts like `max_effort=5`
- named effort presets:
  - `"none"`
  - `"basic"`
  - `"medium"`
  - `"high"`
  - `"max"`

Functions can also be simplified directly:

```python
f_simplified = f.simplify(max_effort="high")
```

This works for ordinary functions and derived ones such as Jacobians and Hessians.

## Common Subexpression Elimination

The library can extract reusable intermediate expressions into a computation plan, which is especially useful as a precursor to Rust code generation.

```python
from gradgen import SX, cse

x = SX.sym("x")
z = x * x + 1
expr = z + z * z

plan = cse([expr])
for assignment in plan.assignments:
    print(assignment.name, assignment.expr, assignment.use_count)
```

You can also build a plan directly from a function:

```python
plan = f.cse(prefix="w", min_uses=2)
```

`CSEPlan` currently contains:

- `assignments`: named reusable intermediates in topological order
- `outputs`: flattened scalar outputs
- `use_counts`: DAG reference counts for all visited nodes

This is intended to become the bridge between symbolic graphs and workspace-based Rust code generation.