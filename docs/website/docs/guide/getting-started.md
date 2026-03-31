---
sidebar_position: 1
---

# Getting Started

This guide will help you get started with Gradgen.

## Installation

Install Gradgen using pip:

```bash
pip install gradgen
```

## Basic Concepts

### Symbolic Expressions (SX)

An `SX` represents a scalar symbolic expression. You create symbols and build expressions:

```python
from gradgen import SX

x = SX.sym("x")
y = SX.sym("y")

# Build expressions
expr = x * x + y * y
```

### Vector Expressions (SXVector)

An `SXVector` is a collection of scalar expressions:

```python
from gradgen import SXVector

v = SXVector.sym("v", 3)  # Create a 3-dimensional symbolic vector
```

### Functions

A `Function` wraps symbolic expressions with named inputs and outputs:

```python
from gradgen import Function, SX

x = SX.sym("x")
f = Function("f", [x], [x * x + 1], input_names=["x"], output_names=["y"])

# Numeric evaluation
print(f(2.0))  # 5.0

# Symbolic substitution
z = SX.sym("z")
print(f(z))    # symbolic: z * z + 1
```

## Your First AD Computation

### Forward Mode: Jacobian-Vector Product (JVP)

Compute directional derivatives:

```python
from gradgen import Function, SX, jvp

x = SX.sym("x")
y = SX.sym("y")
expr = x * x + x * y + y * y

# Directional derivative
directional = jvp(expr, x, 1.0)

f = Function("f", [x, y], [directional])
print(f(1.0, 2.0))  # 1 + 2 = 3
```

### Reverse Mode: Gradient

Compute gradients for scalar outputs:

```python
from gradgen import Function, SX, gradient

x = SX.sym("x")
y = SX.sym("y")
expr = x * x + x * y + y * y

grad_x = gradient(expr, x)
grad_y = gradient(expr, y)

f = Function("grad", [x, y], [grad_x, grad_y])
print(f(1.0, 2.0))  # (2*1 + 2, 1 + 2*2) = (4, 5)
```

## Rust Code Generation

Generate optimized Rust code for your functions:

```python
from gradgen import Function, SX

x = SX.sym("x")
f = Function("square_plus_one", [x], [x * x + 1])

result = f.generate_rust()
print(result.source)

# Or create a full Rust project
project = f.create_rust_project("./my_kernel")
```

## Next Steps

- Explore [Examples](../examples/basic-examples)
