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
x = SXVector.sym("x", 1)
x = x[0]
```

</details>

Lastly, note that `x[a:b]` returns an `SXVector` view.

```
n = 10
x = SXVector.sym("x", n)
x_slice = x[1:4] # <-- this is (x[1], x[2], x[3])
```

## Symbolic expressions

Using scalar and vector symbols we can construct symbolic expressions. 
For example, to define the function $f(x) = ux/\Vert x \Vert_1$ we can do  

```python
x = SXVector.sym("x", 5)
u = SX.sym("u")

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


## Constants

Scalar constants are first-class symbolic expressions. You can create them
explicitly with `SX.const(...)` or the top-level `const(...)` helper:

```python
from gradgen import SX

c1 = SX.const(3)
```

Constants participate in the same symbolic expressions as variables, so they
work naturally inside function definitions and code generation.

## Vectors from iterables

If you already have an iterable of scalar-like values, use `vector(...)` to
coerce it into an `SXVector`:

```python
from gradgen import SX, SXVector, vector

a = SX.sym("a")
b = SX.sym("b")
v = vector([a, b, 1.0])
```

This is convenient when building vectors from Python lists or tuples. If you
want a flat packed vector made from existing vectors, unpack them first:

```python
w = SXVector.sym("w", 5)
u = SXVector((a, b))
p = SXVector((*w, *u))
```


## Matvec, bilinear, and quadratic functions

In optimal control applications we frequently encounter terms of the form 
$x^\intercal P x$, where $P$ is a symmetric matrix. Such expressions can be
constructed with `quadform` as follows

```python
import gradgen as gg

x = SXVector.sym("x", 2)
P = [[1, 4], 
     [4, 5]]
f = gg.quadform(P, x, is_symmetric=True)
```

If `is_symmetric=True` is used, then a simpler expression is generated,
however, an exception is raised if the provided matrix `P` is not truly 
symmetric.

One limitation of the current framework is that $P$ needs to be a constant 
matrix. Quadratic forms with a symbolic matrix will be supported in a future 
version.

The lower-level helpers `matvec(...)` and `bilinear_form(...)` are also
available when you want to express a constant matrix-vector product or a
bilinear form directly:

```python
import gradgen as gg

x = SXVector.sym("x", 2)
y = SXVector.sym("y", 2)
P = [[1, 4],
     [2, 5]]

mx = gg.matvec(P, x)
b = gg.bilinear_form(x, P, y)
q = gg.quadform(P, x)
```

`matvec(...)` returns a vector, `bilinear_form(...)` returns a scalar, and
`quadform(...)` is the quadratic-form special case.

Likewise, we often need to compute dot products. This can be done with `dot`:

```python
n = 3
x = SXVector.sym("x", n)
q = SXVector.sym("q", n)
f = x.dot(q)
```

To get the length of a vector, do 

```python
x = SXVector.sym("x", 10)
l = len(x)  # l = 10
```

## Equality of symbols

Symbols are uniquely identified by their *name*. For example, the following 
symbols are equal

```python
x1 = SX.sym("x")
x2 = SX.sym("x")
assert x1 == x2
```

Because symbols compare by name, they are also suitable as dictionary keys or
set members when you want name-based aliasing.


## Concatenation

Two or more symbols can be packed into a vector using the constructor of `SXVector`. For example,

```python
a = SX.sym("a")
b = SX.sym("b")
c = SX.sym("c")
x = SXVector((a, b, c))

assert x[0] == a
assert len(x) == 3
```

Vectors can also be concatenated as follows:

```python
x = SXVector.sym("x", 3)
y = SXVector.sym("y", 5)
a = SX.sym("a")
z = SXVector((*x, *y, a))

assert len(z) == 9
assert z[1] == x[1]
assert z[3] == y[0]
assert z[8] == a
```

Note that we unpack `*x` and `*y` to pass them to the constructor.

If you omit the unpacking, the constructor stores nested vectors instead of
flattening them. For packed state and parameter vectors, the unpacked form is
usually what you want.
