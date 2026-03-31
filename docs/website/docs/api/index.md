---
sidebar_position: 3
---

# API Reference

## Core Classes

### SX

`SX` represents a scalar symbolic expression.

#### Methods

- `SX.sym(name: str, metadata: dict = {})` - Create a symbolic variable
- `SX.const(value: float)` - Create a constant
- `op` - Operation name (e.g., "add", "mul", "sin")
- `name` - Variable name (for symbols)
- `value` - Numeric value (for constants)
- `args` - Argument nodes for operations

#### Operations

- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Trigonometric: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`
- Hyperbolic: `sinh()`, `cosh()`, `tanh()`, `asinh()`, `acosh()`, `atanh()`
- Exponential/Logarithmic: `exp()`, `expm1()`, `log()`, `log1p()`
- Other: `sqrt()`, `cbrt()`, `abs()`, `floor()`, `ceil()`, `round()`

### SXVector

`SXVector` is a collection of scalar expressions.

#### Methods

- `SXVector.sym(name: str, size: int)` - Create a symbolic vector
- `len(v)` - Vector size
- `v[i]` - Get scalar at index
- `v[a:b]` - Get slice as a new vector
- `dot(other)` - Dot product
- `norm1()` - L1 norm
- `norm2()` - L2 norm
- `norm2sq()` - Squared L2 norm
- `norm_inf()` - Infinity norm
- `norm_p(p)` - Lp norm

#### Operations

- Elementwise: `v + w`, `v - w`, `v / w`, `c * v`
- Norms: `norm1()`, `norm2()`, `norm2sq()`, `norm_inf()`, `norm_p(p)`

### Function

A `Function` wraps symbolic expressions with named inputs/outputs.

#### Constructor

```python
Function(name, inputs, outputs, input_names=None, output_names=None)
```

#### Methods

- `__call__(*args)` - Evaluate numerically or symbolically
- `jacobian(wrt_index=0)` - Get Jacobian function
- `gradient(wrt_index=0)` - Get gradient function (scalar output)
- `hessian(wrt_index=0)` - Get Hessian function (scalar output)
- `hvp(wrt_index=0)` - Get Hessian-vector product function
- `jvp(*tangent_inputs)` - Get JVP function
- `vjp(*cotangent_outputs)` - Get VJP function
- `simplify(max_effort="basic")` - Simplify expression
- `cse(prefix="w", min_uses=2)` - Common subexpression elimination
- `generate_rust(config=None, ...)` - Generate Rust code
- `create_rust_project(path, ...)` - Create Rust project

## Automatic Differentiation

### Forward Mode

- `jvp(expr, wrt, tangent)` - Jacobian-vector product
- `Function.jvp(*tangent_inputs)` - Function JVP

### Reverse Mode

- `gradient(expr, wrt)` - Scalar gradient
- `vjp(expr, wrt, cotangent)` - Vector-Jacobian product
- `Function.vjp(...)` - Function VJP

## Batching Operations

### map_function

Apply a function repeatedly over a batch:

```python
from gradgen import map_function

mapped = map_function(f, count, input_name=None, name=None)
```

### zip_function

Zip multiple input sequences through a function:

```python
from gradgen import zip_function

zipped = zip_function(f, count, input_names=None, name=None)
```

## Rust Code Generation

### RustBackendConfig

Configure Rust code generation:

```python
config = (
    RustBackendConfig()
    .with_backend_mode("std")  # or "no_std"
    .with_scalar_type("f64")   # or "f32"
)
```

### Methods

- `function.generate_rust(config=None, ...)` - Generate source
- `function.create_rust_project(path, ...)` - Create project

## See Also

- [Getting Started](../guide/getting-started)
- [Examples](../examples/basic-examples)
