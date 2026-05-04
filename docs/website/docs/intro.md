---
sidebar_position: 1
description: "Gradgen is a Python library for symbolic automatic differentiation and Rust code generation, with support for embeddable no_std kernels and optimal control workflows."
keywords:
  - Gradgen
  - Rust code generation
  - automatic differentiation
  - symbolic expressions
  - no_std
  - optimal control
last_update:
  date: 2026-04-30
  author: Pantelis Sopasakis
---

# Welcome to Gradgen

**Gradgen** is a Python library for symbolic automatic differentiation and Rust code generation. It helps you build efficient computational kernels with automatic differentiation capabilities.

<img src="/gradgen/img/gradgen.png" style="margin: auto; display: block; width: 16%" alt="gradgen logo">

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
