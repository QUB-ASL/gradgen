# Demos

This directory contains runnable end-to-end examples showing how to use
`gradgen` from Python and how to consume the generated Rust crates.

## Available demos

- [codegen](./codegen/README.md): basic symbolic functions and Rust code generation
- [custom_function](./custom_function/README.md): opaque custom elementary functions with user-provided derivatives
- [multi_function](./multi_function/README.md): generating one Rust crate from multiple source functions
- [vjp](./vjp/README.md): Jacobian generation and runtime-seeded vector-Jacobian products
- [map_zip](./map_zip/README.md): staged loop-preserving packed map/zip kernels
- [composed_function](./composed_function/README.md): staged compositions with `ComposedFunction` and repeated stages
- [single_shooting](./single_shooting/README.md): loop-based deterministic single-shooting OCP code generation

## Running all demos

There is also a [Makefile](./Makefile) in this directory.

From the repository root, after activating your virtual environment, you can run:

```bash
make -C demos PYTHON=../venv/bin/python
```

This will execute all demo `main.py` files and regenerate their Rust crates.
