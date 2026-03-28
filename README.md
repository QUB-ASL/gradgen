# Gradgen

`gradgen` is a Python library for symbolic differentiation and Rust code generation.

The project is being built incrementally. The current implementation focuses on:

- `SX`-style symbolic expressions
- vector-first semantics
- `Function` as a core abstraction
- forward-mode automatic differentiation

Rust code generation, reverse-mode AD, matrix operations, `MX`, and solver-related features are still to come.

## Current Status

The library already supports:

- scalar symbolic expressions with hash-consed DAG nodes
- symbolic vectors built from scalar `SX` expressions
- basic scalar operations such as `+`, `-`, `*`, `/`, `**`, `sin`, `cos`, `exp`, `log`, and `sqrt`
- vector operations including elementwise `+`, `-`, `/`, scalar-vector products, and dot products
- symbolic and numeric `Function` calls
- forward-mode AD through scalar derivatives and Jacobian-vector products

Some intentional limitations at this stage:

- `SX` symbols are formal symbols representing real-valued scalar quantities
- elementwise vector-vector multiplication with `x * y` is not supported yet
- expression simplification is still minimal, so derivatives may be structurally correct without being algebraically simplified
- only forward-mode AD is implemented so far

## Example

```python
from gradgen import Function, SX, SXVector

# Scalar input and numeric evaluation.
x = SX.sym("x")
f = Function("f", [x], [x * x + 1])
print(f(3.0))  # 10.0

# Scalar input and symbolic substitution.
z = SX.sym("z")
print(f(z))    # symbolic expression equivalent to z * z + 1

# Vector input with multiple outputs.
v = SXVector.sym("v", 2)
g = Function("g", [v], [v, v.dot(v)])
print(g([2.0, 3.0]))  # ((2.0, 3.0), 13.0)
```

## Core Concepts

### `SX`

`SX` is the scalar symbolic type. An `SX` value is not a concrete number. It is a symbolic expression node in a DAG whose intended meaning is a real-valued scalar computation.

```python
from gradgen import SX

x = SX.sym("x")
y = SX.sym("y")
expr = x.sin() + x * y
```

Repeated identical expressions reuse the same underlying node internally, which keeps graphs compact and helps later AD and code generation.

### `SXVector`

`SXVector` is a lightweight vector container built from scalar `SX` expressions.

```python
from gradgen import SXVector

x = SXVector.sym("x", 3)
y = SXVector.sym("y", 3)

z = x + y
d = x.dot(y)
```

For now:

- `x + y`, `x - y`, and `x / y` are elementwise
- `2 * x` and `x * 2` are supported
- `x * y` is intentionally not supported yet

### `Function`

`Function` defines a symbolic boundary around inputs and outputs.

```python
from gradgen import Function, SX

x = SX.sym("x")
y = SX.sym("y")
f = Function("f", [x, y], [x + y, x * y])
```

You can call a function numerically:

```python
print(f(2.0, 3.0))  # (5.0, 6.0)
```

Or symbolically:

```python
z = SX.sym("z")
w = SX.sym("w")
print(f(z, w))  # symbolic outputs
```

Vector inputs accept either `SXVector` values or Python sequences:

```python
from gradgen import Function, SXVector

v = SXVector.sym("v", 2)
g = Function("g", [v], [v.dot(v)])

print(g([2.0, 3.0]))  # 13.0
```

## Forward-Mode AD

The current AD layer is forward-mode only.

Two main entry points are available:

- `derivative(expr, wrt)` for scalar derivatives
- `jvp(expr, wrt, tangent)` for Jacobian-vector products

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

## Development

Run the test suite locally with:

```bash
PYTHONPATH=src pytest
```
