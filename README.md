# Gradgen

`gradgen` is a Python library for symbolic differentiation and Rust code generation.

The project is being built incrementally, starting with:

- `SX`-style symbolic expressions
- vector-first semantics
- `Function` as a core abstraction
- automatic differentiation
- Rust code generation

Matrix operations, `MX`, and solver-related features are intentionally out of scope for the first phase.

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
