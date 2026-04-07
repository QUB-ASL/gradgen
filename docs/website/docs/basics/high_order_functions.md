---
sidebar_position: 7
---

# Higher-order functions

Gradgen supports high-order functions such as map, zip, and reduce.

## Map

For an integer $N$, the **map** function consumes one packed sequence

$$\mathbf{x} = (x^{(1)}, x^{(2)}, \dots, x^{(N)}),$$

and returns the outputs

$$\mathbf{y} = (u(x^{(1)}), u(x^{(2)}), \dots, u(x^{(N)})),$$

i.e., it applies $u$ to each of the vectors of the sequence.
In other words, it is the mapping $\mathrm{map}_{u, N}: \mathbf{x} \mapsto \mathbf{y}$. This operation is illustrated in the figure below.


<div align="center">
<img src="/gradgen/img/map.png" width="50%" alt="map operation"/>
</div>

The user need to specify the function $u$ and the integer $N$. For example, 
consider the function $u(x) = \sin(x) + x^2$

```python
# Stage map kernel:
#   u(x) = sin(x) + x^2
map_kernel = Function(
    "map_kernel",
    [x],
    [x.sin() + x * x],
    input_names=["x"],
    output_names=["m"],
)
```

We can use `map_function` to define 

```python
N = 5
mapped = map_function(map_kernel, N, input_name="x_seq", name="mapped_seq")
```

This is an object of type `gradgen.map_zip.ZippedFunction`; this can be cast as a [`Function`](./functions) as follows

```python
mapped_function = mapped.to_function()
x_seq = [1, 2, 3, 4, 5]
print(mapped_function(x_seq))
```

## Zip

Similarly, for an integer $M$, the **zip** function consumes two packed sequences

$$\mathbf{a} = (a^{(1)}, \dots, a^{(M)}), \qquad
\mathbf{c} = (c^{(1)}, \dots, c^{(M)}),$$

and returns

$$\mathbf{z} = (g(a^{(1)}, c^{(1)}), \dots, g(a^{(M)}, c^{(M)})).$$

In other words, it is the mapping $\mathrm{zip}_{b, M}: (\mathbf{a}, \mathbf{c}) \mapsto \mathbf{z}$. 
The zip operation is illustrated in the figure below.

<div align="center">
<img src="/gradgen/img/zip.png" width="50%" alt="zip operation illustration"/>
</div>


As an example, consider the function $h:\mathbb{R}^2 \times \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ given by

$$h(a, b, c) = a_1 b + a_2 + \sin(c),$$

with stage input types $a \in \mathbb{R}^2$, $b \in \mathbb{R}$, and $c \in \mathbb{R}$. This is defined as follows:

```python
# Define symbols a, b, and c
a = SXVector.sym("a", 2)
b = SX.sym("b")
c = SX.sym("c")
# Define the function h(a, b, c)
h = Function(
    "h",
    [a, b, c],
    [a[0] * b + a[1] + sin(c) ],
    input_names=["a", "b", "c"],
    output_names=["y"],
)
```

For an integer $N$, the zipped kernel computes

$$
((a_1,\dots,a_N),\ (b_1,\dots,b_N),\ (c_1,\dots,c_N))
\mapsto
\bigl(h(a_1,b_1,c_1),\dots,h(a_N, b_N, c_N)\bigr).
$$

This can be constructed as follows:

```python
N = 5
zipped = zip_function(h, N,
                      input_names=("a_seq", "b_seq", "c_seq"),
                      name="zip3")
```

This is an object of type `gradgen.map_zip.ZippedFunction`.
We can again use `.to_function()` to cast `zipped` as function.

## Reduce

Reduce is a higher-order function where given 

- a sequence of elements (scalars or vectors) $\mathbf{x} = (x_1, \dots, x_N)$, with $x_i \in \mathbb{R}^n$
- a binary operation $\otimes: \mathbb{R}^m\times \mathbb{R}^n\to \mathbb{R}^m$ 
- an initial value $a_0 \in \mathbb{R}^m$

procudes $(((a_0 \otimes x_1) \otimes x_2)\otimes \ldots \otimes x_{n-1}) \otimes x_n$ as shown below

<div align="center">
<img src="/gradgen/img/reduce.png" width="50%" alt="reduce operation"/>
</div>

This can be described by the following pseudocode

```python
# Pseudocode: reduce
# Here `*` denotes the binary operator
z = a0
for i = 0, ..., N:
    z = z * x[i]
```

:::note 

If $x \otimes y = x + y$ and $a_0 = 0$, then **reduce** can be used to compute the sum of a sequence of 
symbols. Likewise, if $x \otimes y = xy$ and $a_0 = 1$, **reduce** produces the product of (scalar) symbols.

:::

Let us look at an example. Suppose $\otimes: \mathbb{R} \times \mathbb{R}^3 \to \mathbb{R}$ defined as 

$$a \otimes x = \sin(a + \Vert x \Vert_2^2).$$

This is 

```python
a = SX.sym("a")
x = SXVector.sym("x", 3)
r = Function(
    "h",
    [a, x],
    [sin(a + x.norm2sq())],
    input_names=["a", "x"],
    output_names=["z"],
)
```

Let us define $\mathbf{x} = (x_1, \ldots, x_N)$.
The function `gradgen.reduce_function` maps 

$$\mathrm{reduce}_{\otimes, N}: (a_0, \mathbf{x}) \mapsto a_0 \otimes x_1 \ldots \otimes x_n.$$


```python
N = 5
reduced = gg.reduce_function(
    r, N,
    accumulator_input_name="acc",
    input_name="x_seq",
    output_name="acc_final",
    name="reduced_val",
)
```

This is an object of type `ReducedFunction` and can be cast as a [`Function`](./functions) using `.to_function()`. 
Here is an example:


```python
reduced_fun = reduced.to_function()
result = reduced_fun(acc=0, x_seq=[0.1, 0.2, 0.3]*N)
```



## Composition

<div align="center">
<img src="/gradgen/img/composer.png" width="30%" alt="reduce operation"/>
</div>


## Repeat and chain

Repeat and chain are variants of the reduce function.


## Code generation

