# Gradgen 

The project is being built incrementally. The current implementation focuses on:

- `SX`-style symbolic expressions
- vector-first semantics
- `Function` as a core abstraction
- symbolic automatic differentiation
- simplification and common subexpression extraction
- Rust code generation for primal and derivative kernels

Matrix operations, `MX`, complex-domain semantics, and solver-related features are still to come.

The library already supports:

- scalar symbolic expressions with hash-consed DAG nodes
- symbolic vectors built from scalar `SX` expressions
- scalar operations such as `+`, `-`, `*`, `/`, `**`, `min`, `max`, and a growing set of elementary functions
- vector operations including elementwise `+`, `-`, `/`, scalar-vector products, dot products, slicing, and vector norms
- symbolic and numeric `Function` calls
- forward-mode AD through scalar derivatives and Jacobian-vector products
- reverse-mode AD through gradients and vector-Jacobian products
- Jacobian and Hessian construction
- `Function`-level gradient, Jacobian-block, Hessian-block, HVP, and joint primal-Jacobian helpers
- bounded symbolic simplification
- common subexpression elimination plans for later code generation
- Rust crate generation for primal functions, derivative bundles, and multi-kernel crates

Some intentional limitations at this stage:

- `SX` symbols are formal symbols representing real-valued scalar quantities
- elementwise vector-vector multiplication with `x * y` is not supported yet
- simplification is still rule-based and bounded, not a full computer algebra system
- full matrix types are not implemented yet, so Jacobians and Hessians use flat row-major vector representations where needed

Currently implemented elementary functions include:

- `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`, `atan2`
- `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- `exp`, `expm1`, `log`, `log1p`
- `sqrt`, `cbrt`, `hypot`
- `erf`, `erfc`
- `abs`, `floor`, `ceil`, `round`, `trunc`, `fract`, `signum`

Current vector norm helpers include:

- `norm2()`
- `norm2sq()`
- `norm1()`
- `norm_inf()`
- `norm_p(p)`
- `norm_p_to_p(p)`

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
- `x[i]` returns an `SX`, while `x[a:b]` returns an `SXVector` view

Example with slicing:

```python
from gradgen import Function, SXVector

z = SXVector.sym("z", 4)
x = z[0:3]
u = z[3:4]

f = Function("f", [z], [x.norm2() * u + x.norm2sq()])
print(f([1.0, 2.0, 3.0, 4.0]))
```





### Staged Composition

For long compositions of state-transform stages, `ComposedFunction` keeps the
stage structure explicit instead of expanding everything into one large symbolic
expression graph. This is especially useful when you want generated Rust to use
separate stage helpers and loop-based repeated application.

Each state-transform stage must have the signature:

- `G(state, p) -> next_state`

and the terminal function must have the signature:

- `h(state, pf) -> scalar`

You can build compositions with:

- `.then(G, p=...)` for one stage
- `.chain([...])` for several possibly different stages
- `.repeat(G, params=[...])` for repeated application of the same stage
- `.finish(h, p=...)` for the terminal scalar output

Numeric stage parameters are embedded as constants. Symbolic stage parameters
are packed into a single extra runtime input slice named `parameters`, ordered
in forward stage order with the terminal parameter block last.

```python
from gradgen import ComposedFunction, Function, SXVector

x = SXVector.sym("x", 2)
state = SXVector.sym("state", 2)
p = SXVector.sym("p", 2)
pf = SXVector.sym("pf", 1)

G = Function(
    "g",
    [state, p],
    [SXVector((state[0] + p[0], state[1] * p[1]))],
    input_names=["state", "p"],
    output_names=["next_state"],
)
h = Function(
    "h",
    [state, pf],
    [state[0] + state[1] + pf[0]],
    input_names=["state", "pf"],
    output_names=["y"],
)

composed = (
    ComposedFunction("f_chain", x)
    .then(G, p=p)
    .repeat(G, params=[p, p])
    .finish(h, p=pf)
)

expanded = composed.to_function()
gradient = composed.gradient()
```

The expanded symbolic function has the inputs:

- `x`
- `parameters`

where `parameters` contains the packed stage parameters for the composition.

### Single Shooting Optimal Control



### Complete end-to-end example

The example below:

- defines two scalar functions of the same 3D input
- evaluates them in Python
- generates one Rust crate containing kernels for both

```python
from gradgen import (
    CodeGenerationBuilder,
    Function,
    FunctionBundle,
    RustBackendConfig,
    SXVector,
)

x = SXVector.sym("x", 3)

f = Function(
    "f",
    [x],
    [x[0] * x[0] + x[0] * x[1] + x[2].sin() + x[1] * x[2]],
    input_names=["x"],
    output_names=["y"],
)

g = Function(
    "g",
    [x],
    [x.norm2() + x[0] * x[2]],
    input_names=["x"],
    output_names=["z"],
)

# Try the symbolic functions in Python first.
jf = f.jacobian(0)
hvp = f.hvp(0)

print("f([1,2,3]) =", f([1.0, 2.0, 3.0]))
print("jf([1,2,3]) =", jf([1.0, 2.0, 3.0]))
print("hvp([1,2,3], [1,0,-1]) =", hvp([1.0, 2.0, 3.0], [1.0, 0.0, -1.0]))
print("g([1,2,3]) =", g([1.0, 2.0, 3.0]))

config = (
    RustBackendConfig()
    .with_crate_name("my_kernel")
    .with_backend_mode("std")
    .with_scalar_type("f64")
)

builder = (
    CodeGenerationBuilder()
    .with_backend_config(config)
    .for_function(f)
        .add_primal()
        .add_jacobian()
        .add_hvp()
        .add_joint(
            FunctionBundle()
            .add_f()
            .add_jf(wrt=0)
            .add_hvp(wrt=0)
        )
        .with_simplification("medium")
        .done()
    .for_function(g)
        .add_primal()
        .add_gradient()
        .with_simplification("medium")
        .done()
)

project = builder.build("./my_kernel")

print("Generated crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)
```

Then build the generated crate with:

```bash
cd my_kernel
cargo build
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

For test coverage:

```bash
# First run the following (-e is important!)
pip install -e .

# Run test coverage
coverage run --source=src/gradgen -m pytest

# Generate and open HTML report
coverage html && open htmlcov/index.html

# Report in terminal (not recommended)
coverage report -m
```