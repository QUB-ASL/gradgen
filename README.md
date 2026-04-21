<div align="center">    
 <img alt="cgapp logo" src="https://i.postimg.cc/G3M2szz5/Logo-Makr-4z-HKa0.png" width="224px"/><br/>    
    
    
<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dm/gradgen?color=blue&style=flat-square)  -->
[![CI](https://github.com/QUB-ASL/gradgen/actions/workflows/python-tests.yml/badge.svg)](https://github.com/QUB-ASL/gradgen/actions/workflows/python-tests.yml)  [![Docs site](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://qub-asl.github.io/gradgen/) [![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)
 
</div>    


# Gradgen: what it does

<p align="center">
  <img src="https://raw.githubusercontent.com/QUB-ASL/gradgen/refs/heads/main/docs/img/gradgen-what-it-does.png" alt="Gradgen overview" width="80%" />
</p>

**Gradgen** is a Python library for symbolic differentiation and (embedded) Rust code generation.

## Code generation example

[See documentation](https://qub-asl.github.io/gradgen/docs/basics/codegen)

Here is an example where we will define the function 

$$f(x, u) = \Vert x \Vert_2^2 + u  \sin(x_1) + x_2  x_3,$$

for a three-dimensional input $x$ and scalar $u$.

The goal is to generate Rust code for the functions $f$, $Jf$ (the Jacobian matrix
of $f$). 

Furthermore, we want to generate a Rust function that computes simultaneous $f$ 
and $\nabla_x f$. This often is computationally more efficient compared to computing
$f(x, u)$ and $\nabla_x f(x, u)$ in separate functions (look for `FunctionBundle` below).

```python
from gradgen import CodeGenerationBuilder, Function, RustBackendConfig, SXVector, sin

# Define the symbolic inputs.
x = SXVector.sym("x", 3)
u = SXVector.sym("u", 1)

# Build a simple scalar-valued function of x and u
# f(x, u) = ||x||_2^2 + u_1 * sin(x_1) + x_2 * x_3
f_expr = x.norm2sq() + u[0] * sin(x[0]) + x[1] * x[2]

# Define a Function object
f = Function(
    "energy",
    [x, u],
    [f_expr],
    input_names=["x", "u"],
    output_names=["energy"],
)

# (Optional) Evaluate f in Python
x_value = [1.0, 2.0, -0.5]
u_value = [3.0]
print("f(x, u) =", f(x_value, u_value))

# Generate code
project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("my_kernel")
        .with_backend_mode("no_std")
        .with_scalar_type("f64")
    )
    .for_function(f)
        .add_primal()
        .add_jacobian()
        .add_joint(
            FunctionBundle()
            .add_f()
            .add_jf(wrt=0)
        )
        .with_simplification("medium")
        .done()
    .build(Path(__file__).resolve().parent / "codegen_kernel")
)
```

See the [demos](./demos) and this more complete [tutorial](https://qub-asl.github.io/gradgen/docs/basics/codegen).

## Special case: optimal control

[See tutorial](https://qub-asl.github.io/gradgen/docs/basics/ocp)

In applications such as optimal control, the generated code
can become too large very easily. 
However, the problem structure can be exploited to generate
code with complexity that doesn't increase with the prediction 
horizon.

Instead of completely unrolled code, **Gradgen** exploits
the problem structure to create high-performance, 
human-readable embeddable Rust code.

See this complete [tutorial](https://qub-asl.github.io/gradgen/docs/basics/ocp) for details.

## Unique features

- Truly embdedable safe Rust code with optional [`#[no_std]`](https://docs.rust-embedded.org/book/intro/no-std.html), no dynamic memory allocation, no `panic!`s
- Specialised code generation tools for optimal control problems ([docs](https://qub-asl.github.io/gradgen/docs/basics/ocp))
- Very efficient code generation thanks to modular code generation using [`map`](./demos/map_zip/), [`zip`](./demos/zip_3/), [`repeat`](./demos/composed_function/), and [`chain`](./demos/composed_chain/) high-order functions.
- Supports both single (`f32`) and double (`f64`) precision arithmetic

## Where to go next?

See the [demos](./demos) and this more complete [documentation](https://qub-asl.github.io/gradgen/docs) for details.

## Show us some love!

If you find **Gradgen** useful, give us a star on GitHub!
