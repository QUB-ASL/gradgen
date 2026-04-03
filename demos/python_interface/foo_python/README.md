# Python Interface Wrapper: foo_python

This crate is the PyO3 wrapper for the generated Rust crate `foo`.
It exists so the low-level kernel crate can stay `no_std`-friendly and be used
independently of Python.

Python module name: `foo`

## Layout

- `../foo`: low-level Rust crate with the generated kernels
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

- `workspace_for_function("energy")`
- `call("energy", ...)`

Single-output functions return a scalar or list. Multi-output functions return a
dictionary keyed by output name.