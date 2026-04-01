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

The object `f` is an instance of `RegisteredElementaryFunction`
and can be used to create a `Function` object as follows

```python
x = SXVector.sym("x", 2)
w = SXVector.sym("w", 2)

f = Function(
    "f",
    [x, w],
    [f(x, w=w)],
    input_names=["x", "w"],
    output_names=["y"],
)
```

We can now evaluate this function

```python
a = f([1, 2], [3, 4])
```

Since we have specified the Jacobian, we can also determine 
$Jf(x, w)$

```python
jf = f.jacobian(wrt_index=0)
print(jf([1, 2], [3, 4]))
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

```python
# Rust implementation of the f(x, w)...
RUST_F = """
fn custom_energy_demo(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    {{ math_library }}::exp2(w[0]) * x[0] * x[0]
        + w[1] * x[1] * x[1]
        + {{ math_library }}::sin(x[0] * x[1])
}
"""
```