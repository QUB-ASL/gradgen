# Python Interface Wrapper: composed_chain_kernel_python

This crate is the PyO3 wrapper for the generated Rust crate `composed_chain_kernel`.
It exists so the low-level kernel crate can stay `no_std`-friendly and be used
independently of Python.

Python module name: `composed_chain_kernel`

## Layout

- `../composed_chain_kernel`: low-level Rust crate with the generated kernels
- `src/lib.rs`: PyO3 wrapper that calls into the low-level crate
- `pyproject.toml`: maturin configuration for building the Python module

## Build

```bash
cargo build
```

## Install

From this directory, after activating your Python virtual environment:

```bash
python -m pip install -e .
```

## Python API

The generated module exposes:

- `__version__`: the wrapper version from `pyproject.toml`
- `__all__`: the public Python API exported by the module

- `workspace_for_function("composed_chain_kernel_chain_demo_f")`
- `call("composed_chain_kernel_chain_demo_f", ...)`
- `workspace_for_function("composed_chain_kernel_chain_demo_grad_x")`
- `call("composed_chain_kernel_chain_demo_grad_x", ...)`

Single-output functions return a scalar or list. Multi-output functions return a
dictionary keyed by output name.