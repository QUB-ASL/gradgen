---
sidebar_position: 2
---

# Basic Examples

Here are some basic examples to get you started with Gradgen.

## Example 1: Simple Function

Create and evaluate a simple function:

```python
from gradgen import Function, SX

x = SX.sym("x")
f = Function("quadratic", [x], [x * x + 2 * x + 1])

# Evaluate numerically
print(f(3.0))  # (3 + 1)^2 = 16.0
```

## Example 2: Multi-Variable Function

Work with multiple inputs:

```python
from gradgen import Function, SX

x = SX.sym("x")
y = SX.sym("y")
f = Function("sum_of_squares", [x, y], [x * x + y * y])

print(f(3.0, 4.0))  # 25.0
```

## Example 3: Vector Operations

Use vector expressions:

```python
from gradgen import Function, SXVector

v = SXVector.sym("v", 3)
f = Function("norm_squared", [v], [v.norm2sq()])

print(f([1.0, 2.0, 2.0]))  # 1 + 4 + 4 = 9.0
```

## Example 4: Jacobian

Compute a Jacobian matrix:

```python
from gradgen import Function, SXVector

x = SXVector.sym("x", 2)
f = Function(
    "f",
    [x],
    [SXVector((x[0] + x[1], x[0] * x[1]))],
    output_names=["sum", "product"]
)

# Get Jacobian
jf = f.jacobian(0)
print(jf([2.0, 3.0]))  # Jacobian at point (2, 3)
```

## Example 5: Rust Code Generation

Generate efficient Rust code:

```python
from gradgen import Function, SX

x = SX.sym("x")
f = Function("square", [x], [x * x])

# Generate Rust source
result = f.generate_rust()
print(result.source)

# Create a Rust project
project = f.create_rust_project("./square_kernel")
print(f"Created project at: {project.project_dir}")
```

## Example 6: Batching Operations

Use `map_function` for batch operations:

```python
from gradgen import Function, SX, map_function

x = SX.sym("x")
f = Function("square", [x], [x * x])

# Map this function over a batch of inputs
mapped = map_function(f, 3, input_name="x_seq")
expanded = mapped.to_function()

print(expanded([1.0, 2.0, 3.0]))  # (1.0, 4.0, 9.0)
```

## Next Steps

- Explore [Getting Started](../guide/getting-started)
