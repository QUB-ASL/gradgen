---
sidebar_position: 2
---

# Function

## Scalar input arguments

We can define function as follows

```python
from gradgen import Function, SX

x = SX.sym("x")
y = SX.sym("y")
f = Function("f", [x, y], [x + y, x * y])
```

where for `f` we have specified that it is a function of the input 
variables `x` and `y` and returns two outputs. Mathematically, it is 
the function $f:\mathbb{R} \times \mathbb{R} \to \mathbb{R}$ with

$$f(x, y) = 
\begin{bmatrix}
x + y \\\\ xy
\end{bmatrix}.$$

The function `f` can be called with numerical arguments

```python
print(f(2.0, 3.0))  # (5.0, 6.0)
```

and the result is a tuple of `float`.

Functions can also be called with symbolic arguments


```python
z = SX.sym("z")
w = SX.sym("w")
print(f(z + 1, w.sin()))  # symbolic outputs
```

and the result is a tuple of `SX`.

## Vector inputs

We can also define functions that take vector inputs. For example,
we can define the function $g:\mathbb{R}^2 \to \mathbb{R}$ given by 
$g(x) = \Vert x \Vert_2$ as follows

```python
from gradgen import Function, SXVector

x = SXVector.sym("x", 2)
g = Function("g", [x], [x.norm2()])

print(g([3.0, 4.0]))  # returns 5.
```

Or, we can define the function 
$h:\mathbb{R}^3 \times \mathbb{R} \to \mathbb{R}$
given by $h(x, a) = a\Vert x \Vert_{1}$ as follows

```python
from gradgen import Function, SXVector, SX

x = SXVector.sym("x", 3)
a = SX.sym("a")
h = Function("h", [x, a], [a * x.norm1()])

x0 = [3.0, 4.0, 1.0]
a0 = -1.
h0 = h(x0, a0)

print(h0) 
```

## Composition of functions

Functions can be composed in a very natural manner. 
Suppose we have a function $f:\mathbb{R}^2 \times \mathbb{R} \to \mathbb{R}$
given by 

$$f(x, a) = \Vert (a^2 + 1) x \Vert_{\infty},$$

and we have the function $g: \mathbb{R}^2 \to \mathbb{R}^2$ given by 

$$g(x) = \frac{x}{\Vert x \Vert_2^2}.$$

We can then define the function 
$h:\mathbb{R}^2 \times \mathbb{R} \to \mathbb{R}$ given by $h(x, a) = f(g(x), a)$. 
This can be defined as follows

```python 
from gradgen import Function, SXVector, SX

x = SXVector.sym("x", 3)
a = SX.sym("a")

f = Function("f", [x, a], [(a**2 + 1) * x.norm_inf()])
g = Function("g", [x], [x / x.norm2sq()])

h = Function("h", [x, a], [f(g(x), a)]) # h(x, a) = f(g(x), a)
```


## Named input arguments

We can call a function with named input arguments. For example,
suppose we have the function $f(x, y) = x\Vert y \Vert$,
where $x\in\mathbb{R}$ and $y\in\mathbb{R}^2$ defined as follows:

```python
x = SX.sym("x")
y = SXVector.sym("y", 3)

f = Function(
    "f",
    [x, y],
    [x * y.norm2()],
    input_names=["x", "y"],
    output_names=["z"],
)
```

We can call `f` by passing named arguments, that is,

```python
z = f(x=1, y=[2, 3])
```
