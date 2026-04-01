---
sidebar_position: 1
---

# Welcome to Gradgen

**Gradgen** is a Python library for symbolic automatic differentiation and Rust code generation. It helps you build efficient computational kernels with automatic differentiation capabilities.

<img src="https://camo.githubusercontent.com/0a0a39c7417ae1cc88302a16176929a06aa01de5680f7d3f41bb06e1de7e6284/68747470733a2f2f692e706f7374696d672e63632f47334d32737a7a352f4c6f676f2d4d616b722d347a2d484b61302e706e67" style="margin: auto; display: block; width: 16%">

## What is Gradgen?

Gradgen provides:

- **Symbolic Expression Trees**: Build symbolic expressions using `SX` (scalar expressions) and `SXVector` (vector expressions)
- **Automatic Differentiation**: Both forward-mode and reverse-mode AD
- **Rust Code Generation**: Generate optimized Rust code for primal functions and their derivatives
- **Loop-Structured Batching**: Efficiently handle batched operations with `map_function` and `zip_function`
- **Single Shooting Optimal Control**: Build and solve optimal control problems symbolically

## Quick Start

Install Gradgen:

```bash
pip install gradgen
```

Create a simple function:

```python
from gradgen import Function, SX

x = SX.sym("x")
f = Function("f", [x], [x * x + 1])
print(f(3.0))  # Output: 10.0
```

Generate a Jacobian:

```python
jacobian_f = f.jacobian(0)
print(jacobian_f(3.0))  # Output: 6.0
```

Generate Rust code:

```python
result = f.generate_rust()
print(result.source)
```

## Key Features

- **Vector-first semantics**: Built-in support for vector operations
- **Hash-consed DAG nodes**: Efficient expression reuse
- **Symbolic simplification**: Automatic expression simplification
- **Common Subexpression Elimination**: Optimize generated code
- **Multiple AD modes**: Forward-mode JVP and reverse-mode VJP

