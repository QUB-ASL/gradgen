---
sidebar_position: 7
---

# Custom function

Suppose you want to use a scalar-valued function, 
$f:\mathbb{R}^n\times \mathbb{R}^p \to \mathbb{R}$, that is not currently implemented in gradgen. You can easily register it.

## Registering custom functions

Your function must have the signature

```python
def my_function(x, w):
    return 0.0
```

where `x` and `w` are tuples (or lists) and the function returns a 
float. 

Think of `x` as the *main* argument of your function and `w` as a 
parameter vector.

The first step is to define a Python function (or `lambda`) for your 
function. Your function can include *anything* since gradgen will 
treat it as a black box.

Optionally, you can also provide any of the following:
(i) the Jacobian with respect
to $x$, that is, $J_x f(x, w)$, aka the gradient, 
(ii) the Hessian matrix,
(iii) a function that computes Hessian-vector products, which 
is often more efficient than computing the full Hessian.

Let's look at a concrete example. Consider the function

$$f(x, w) = 2^{w_1} x_1^2 + w_2 x_2^2 + \sin(x_1 x_2),$$

with $x\in \mathbb{R}^2$ and $w\in \mathbb{R}^2$.

Let us create a Python function that evaluates $f$ using `numpy`

```python
import numpy as np

def eval_f(x, w):
    return np.exp2(w[0]) * x[0] * x[0] + w[1] * x[1] * x[1] \
        + np.sin(x[0] * x[1])
```

We can also determine the Jacobian of $f$, which is

$$J_x f(x, w) = 
\begin{bmatrix}
2 \cdot 2^{w_1} x_1 + x_2 \cos(x_1 x_2) \\\\ 
2 w_2 x_2 + x_1 \cos(x_1 x_2)
\end{bmatrix}.$$

We can create a Python function for $J_x f$:

```python
def eval_jf(x, w):
    return [
        2. * np.exp2(w[0]) * x[0] + x[1] * np.cos(x[0] * x[1]),
        2. * w[1] * x[1] + x[0] * np.cos(x[0] * x[1]),
    ]
```

We will omit the Hessian for now. Let us register this fancy function 
to gradgen

```python
f = register_elementary_function(
    name="f",
    input_dimension=2, # dimension of x
    parameter_dimension=2, # dimension of w
    eval_python=eval_f, 
    jacobian=eval_jf
)
```

:::note

Note that every registered function needs to have a unique name.

:::

<span id="create-function">

The object `f` is an instance of `RegisteredElementaryFunction`
and can be used to create a `Function` object as follows 


```python
x = SXVector.sym("x", 2)
w = SXVector.sym("w", 2)

f_fun = Function(
    "energy",
    [x, w],
    [f(x, w=w)],
    input_names=["x", "w"],
    output_names=["y"],
)
```

We can now evaluate this function

```python
a = f_fun([1, 2], [3, 4])
```

Since we have specified the Jacobian, we can also determine 
$Jf(x, w)$

```python
jf_fun = f_fun.jacobian(wrt_index=0)
print(jf_fun([1, 2], [3, 4]))
```

More importantly, we can use `f` in other expressions and 
compute derivatives with automatic differentiation. Here is a 
simple example:


```python
g = Function(
    "g",
    [x, w],
    [f(x.sin() * w[1], 2 * w.asinh())],
    input_names=["x", "w"],
    output_names=["z"],
)
jg = g.jacobian(wrt_index=0)
print(jg([1, 2], [3, 4]))
```

We will come back to Hessians later. Let us see how we can generated 
Rust code for custom functions, or for functions that involve custom 
functions.

## Code generation

Custom functions can be used in code generation. To this end, the user
needs to provide a Rust implementation of $f$ (and, optionally, its gradient
and Hessian or Hessian-vector products).

**See first:** [Rust code generation](./codegen)

### Custom Rust implementation 

When the writing your custom Rust implementation, please note:

- instead of `f32` or `f64` data types, it is best to use `{{ scalar_type }}`; 
  this will be replaced by the correct data type during code generation
- instead of explicitly using `libm` (or other math libraries), it is best to 
  use `{{ math_library }}`.

Here is a Rust implementation of the function

```python
# Rust implementation of the f(x, w)...
RUST_F = """
fn f(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    {{ math_library }}::exp2(w[0]) * x[0] * x[0]
        + w[1] * x[1] * x[1]
        + {{ math_library }}::sin(x[0] * x[1])
}
"""
```

:::warning Important note!

It is important for the name of the function, `f`, to be the same as in `register_elementary_function`.

:::


:::note Additional libraries

Currently, it is not possible to import additional libraries to use in your custom implementation. This will be supported in a future version.

:::

The gradient of $f$ can be defined similarly as shown below

```python
RUST_JACOBIAN = """
fn f_jacobian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let xy = x[0] * x[1];
    out[0] = 2.0_{{ scalar_type }} * {{ math_library }}::exp2(w[0]) * x[0]
        + x[1] * {{ math_library }}::cos(xy);
    out[1] = 2.0_{{ scalar_type }} * w[1] * x[1]
        + x[0] * {{ math_library }}::cos(xy);
}
"""
```

We can now register this function and specify both its Python implementations and the above Rust implementations:

```python
f = register_elementary_function(
    name="f",
    input_dimension=2,           # dimension of x
    parameter_dimension=2,       # dimension of w
    eval_python=eval_f,          # Python callback for f
    jacobian=eval_jf,            # Python callback for the gradient of f
    rust_primal=RUST_F,          # Rust code for f
    rust_jacobian=RUST_JACOBIAN  # Rust code for grad f
)
```

### A code generation example

To generate Rust code for our function (and its gradient) we first 
need to construct a `Function` object (see [above](#create-function)).

We can then generate a Rust crate as follows:

```python
project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("custom")
        .with_backend_mode("no_std")
        .with_scalar_type("f64")
    )
    # Specify what needs to be generated
    .for_function(f_fun)
        .add_joint(      
            FunctionBundle()
                .add_f()
                .add_jf(wrt=0)
        )
        .done()
    .build("./my_crates/custom")
)
```

Here we generate Rust code for both $f$ and $\nabla f$,
which are computed in the same function. This is what the 
`add_joint` does.

The generated Rust function looks like this...

```rust
pub fn custom_energy_f_jf_x(
    x: &[f64],
    w: &[f64],
    y: &mut [f64],
    jacobian_y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    // implementation ...
}
```

The name is the function is the name of the crate, followed by 
the name of the function we defined [earlier](#create-function), 
followed `_f`, which means that the function itself it being 
computed, follwed by `_jf_x` meaning that its gradient is being
computed too.



## Hessian-vector products

For a function $f:\mathbb{R}^n\times \mathbb{R}^p \to \mathbb{R}$ we
may want to calculate Hessian-vector products, i.e., the mapping 

$$(x, w, v) \mapsto \nabla_x^2 f(x, w) v.$$

For the above function,

$$\nabla_x^2 f(x, w) =
\begin{bmatrix}
2 \cdot 2^{w_1} - x_2^2 \sin(x_1 x_2) &
\cos(x_1 x_2) - x_1 x_2 \sin(x_1 x_2) \\\\
\cos(x_1 x_2) - x_1 x_2 \sin(x_1 x_2) &
2 w_2 - x_1^2 \sin(x_1 x_2)
\end{bmatrix}.$$


## Hessians