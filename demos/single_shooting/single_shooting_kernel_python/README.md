# Python Interface Wrapper: single_shooting_kernel_python

This crate is the PyO3 wrapper for the generated Rust crate `single_shooting_kernel`.
It exists so the low-level kernel crate can stay `no_std`-friendly and be used
independently of Python.

Python module name: `single_shooting_kernel`

## Layout

- `../single_shooting_kernel`: low-level Rust crate with the generated kernels
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

- `workspace_for_function("single_shooting_kernel_mpc_cost_f_states")`
- `call("single_shooting_kernel_mpc_cost_f_states", ...)`
- `workspace_for_function("single_shooting_kernel_mpc_cost_grad_states_u_seq")`
- `call("single_shooting_kernel_mpc_cost_grad_states_u_seq", ...)`
- `workspace_for_function("single_shooting_kernel_mpc_cost_hvp_states_u_seq")`
- `call("single_shooting_kernel_mpc_cost_hvp_states_u_seq", ...)`
- `workspace_for_function("single_shooting_kernel_mpc_cost_f_grad_states_u_seq")`
- `call("single_shooting_kernel_mpc_cost_f_grad_states_u_seq", ...)`

Single-output functions return a scalar or list. Multi-output functions return a
dictionary keyed by output name.