---
sidebar_position: 1
---


# Symbolic framework

We can define scalar symbolic variables as follows

```python
from gradgen import SX

x = SX.sym("x")
```

for vectors, use `SXVector`. For example, to create a vector of dimensions 2 do

```python 
from gradgen import SXVector

x = SXVector.sym("x", 2)
```

If you print `x` you will see

```
SXVector(elements=(SX.sym('x_0'), SX.sym('x_1')))
```

<details>

Although `SXVector`s of length 1 behave like scalar, you can "cast" 
an `SXVector` of length 1 as an `SX` using 

```python
x = SXVector.sym("x", 2)
x = x[0]
```

</details>

## Symbolic expressions

Using scalar and vector symbols we can construct symbolic expressions. 
For example, to define the function $f(x) = ux/\Vert x \Vert_1$ we can do  

```python
f = u * x / x.norm1()
```

Several operations are supported. Indicatively, we support the operators `**`;
to define the expression $f = 1 + 0.1  z - 4 z^3$ we can write

```python
z = SX.sym("u")

f = 1 + 0.1 * z - 4 * z**3
```

The operator `**`, when applied to vectors (`SXVector`), operates element-wise.

All **trigonometric** (`cos`, `sin`, `tan`), 
**inverse trigonometric** (`acos`, `asin`, `atan`) and **hyperbolic** operations 
(`cosh`, `sinh`, `tanh`) and their inverses (`acosh`, `asinh`, `atanh`) are 
supported. 

For vectors, the following scalar-valued operations are available:

- `norm2()`: Euclidean norm
- `norm2sq()`: squared Euclidean norm
- `norm1()`: norm-1
- `norm_inf()`: Infinity norm
- `norm_p(p)`: $p$-norm
- `norm_p_to_p(p)`: $p$-norm to the power $p$ 

Moreover, 



## Example

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


Add **Markdown or React** files to `src/pages` to create a **standalone page**:

- `src/pages/index.js` → `localhost:3000/`
- `src/pages/foo.md` → `localhost:3000/foo`
- `src/pages/foo/bar.js` → `localhost:3000/foo/bar`

## Create your first React Page

Create a file at `src/pages/my-react-page.js`:

```jsx title="src/pages/my-react-page.js"
import React from 'react';
import Layout from '@theme/Layout';

export default function MyReactPage() {
  return (
    <Layout>
      <h1>My React page</h1>
      <p>This is a React page</p>
    </Layout>
  );
}
```

A new page is now available at [http://localhost:3000/my-react-page](http://localhost:3000/my-react-page).

## Create your first Markdown Page

Create a file at `src/pages/my-markdown-page.md`:

```mdx title="src/pages/my-markdown-page.md"
# My Markdown page

This is a Markdown page
```

A new page is now available at [http://localhost:3000/my-markdown-page](http://localhost:3000/my-markdown-page).
