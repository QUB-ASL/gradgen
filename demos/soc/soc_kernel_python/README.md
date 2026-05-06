# Python Interface Wrapper: soc_kernel_python

This crate is the PyO3 wrapper for the generated Rust crate `soc_kernel`.
It exists so the low-level kernel crate can stay `no_std`-friendly and be used
independently of Python.

Python module name: `soc_kernel`

## Layout

- `../soc_kernel`: low-level Rust crate with the generated kernels
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

- `workspace_for_function("soc_kernel_energy")`
- `call("soc_kernel_energy", ...)`
- `workspace_for_function("soc_kernel_energy_grad_x")`
- `call("soc_kernel_energy_grad_x", ...)`

Single-output functions return a scalar or list. Multi-output functions return a
dictionary keyed by output name.