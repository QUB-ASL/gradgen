---
sidebar_position: 2
---

# Function

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

