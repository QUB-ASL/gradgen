# Gradgen

`gradgen` is a Python library for symbolic differentiation and Rust code generation.

The project is being built incrementally. The current implementation focuses on:

- `SX`-style symbolic expressions
- vector-first semantics
- `Function` as a core abstraction
- symbolic automatic differentiation
- simplification and common subexpression extraction
- Rust code generation for primal and derivative kernels

Matrix operations, `MX`, complex-domain semantics, and solver-related features are still to come.

## Current Status

The library already supports:

- scalar symbolic expressions with hash-consed DAG nodes
- symbolic vectors built from scalar `SX` expressions
- basic scalar operations such as `+`, `-`, `*`, `/`, `**`, `sin`, `cos`, `exp`, `log`, and `sqrt`
- vector operations including elementwise `+`, `-`, `/`, scalar-vector products, and dot products
- symbolic and numeric `Function` calls
- forward-mode AD through scalar derivatives and Jacobian-vector products
- reverse-mode AD through gradients and vector-Jacobian products
- Jacobian and Hessian construction
- `Function`-level gradient, Jacobian-block, and Hessian-block helpers
- bounded symbolic simplification
- common subexpression elimination plans for later code generation
- Rust crate generation for primal functions and derivative bundles

Some intentional limitations at this stage:

- `SX` symbols are formal symbols representing real-valued scalar quantities
- elementwise vector-vector multiplication with `x * y` is not supported yet
- simplification is still rule-based and bounded, not a full computer algebra system
- full matrix types are not implemented yet, so Jacobians and Hessians use vector-first row-wise representations where needed

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

## Automatic Differentiation

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

For vector-output by vector-input cases, Jacobians are currently represented row by row because full matrix types are not implemented yet.

You can also derive a Jacobian block from a function:

```python
jac_f = some_function.jacobian(0)
```

If you want multiple input blocks at once, use:

```python
blocks = some_function.jacobian_blocks()
```

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

## Rust Code Generation

The library can now generate primal Rust code for a `Function`.

The generated Rust currently uses:

- deterministic naming
- a slice-based ABI
- explicit workspace variables stored in a mutable `work` slice
- configurable scalar types (`f64` or `f32`)
- `std` and `no_std` backend modes

### Generate Rust source in memory

```python
from gradgen import Function, SX

x = SX.sym("x")
f = Function("square_plus_one", [x], [x * x + 1], input_names=["x"], output_names=["y"])

result = f.generate_rust()
print(result.source)
print(result.workspace_size)
```

The generated function looks like a plain Rust function with this style of signature:

```rust
pub fn square_plus_one(x: &[f64], y: &mut [f64], work: &mut [f64]) {
    // ...
}
```

For multiple inputs and outputs, each declared input and output becomes its own slice argument.

You can also configure the backend explicitly:

```python
from gradgen import RustBackendConfig

config = (
    RustBackendConfig()
    .with_backend_mode("no_std")
    .with_scalar_type("f32")
    .with_math_lib("libm")
    .with_function_name("eval_kernel")
)

result = f.generate_rust(config=config)
```

### Create a Rust project on disk

You can also create a minimal Cargo project at a user-specified path:

```python
from gradgen import Function, SX

x = SX.sym("x")
f = Function("square_plus_one", [x], [x * x + 1], input_names=["x"], output_names=["y"])

project = f.create_rust_project("./square_plus_one")
print(project.project_dir)
```

This writes:

- `Cargo.toml`
- `README.md`
- `src/lib.rs`

The generated `README.md` contains simple generic instructions such as:

```bash
cargo build
```

### Custom crate and function names

```python
project = f.create_rust_project(
    "/tmp/my_kernel",
    crate_name="my_kernel",
    function_name="eval_kernel",
)
```

There is also a module-level helper if you prefer:

```python
from gradgen import create_rust_project

project = create_rust_project(f, "/tmp/my_kernel")
```

### Create a derivative Rust bundle

You can generate a directory containing Rust crates for the primal function, Jacobians, and Hessians:

```python
bundle = f.create_rust_derivative_bundle(
    "/tmp/my_bundle",
    simplify_derivatives="high",
)
```

This creates a bundle directory with entries such as:

- `primal/`
- `f_jacobian_<input_name>/`
- `f_hessian_<input_name>/` for scalar-output functions

There is also a module-level helper:

```python
from gradgen import create_rust_derivative_bundle

bundle = create_rust_derivative_bundle(f, "/tmp/my_bundle")
```

### Current ABI conventions

At the moment the generated Rust code assumes:

- scalar inputs are length-1 slices
- vector inputs are dense `&[T]` slices where `T` is `f64` or `f32`
- scalar outputs are length-1 `&mut [T]` slices
- vector outputs are dense `&mut [T]` slices
- intermediate values live in `work: &mut [T]`

The `no_std` backend uses a configurable math namespace and defaults to `libm`.

## Development

Run the test suite locally with:

```bash
PYTHONPATH=src pytest
```
