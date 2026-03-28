# Gradgen

`gradgen` is a Python library for symbolic differentiation and Rust code generation.

The project is being built incrementally, starting with:

- `SX`-style symbolic expressions
- vector-first semantics
- `Function` as a core abstraction
- automatic differentiation
- Rust code generation

Matrix operations, `MX`, and solver-related features are intentionally out of scope for the first phase.
