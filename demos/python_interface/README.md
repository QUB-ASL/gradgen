# Python Interface Demo

This demo shows how to generate a Rust crate plus a separate PyO3 wrapper that
can be imported from Python.

The generated crates:

- `foo/`: the low-level Rust crate with the generated kernels
- `foo_python/`: the PyO3 wrapper crate that exposes the Python module `foo`

The Python wrapper:

- provides `all_functions()` for discovery
- provides `function_info("energy")` for metadata
- provides `workspace_for_function("energy")`
- provides `call("energy", x, w, workspace)`
- exposes direct callables like `foo.energy(...)`
- returns a dictionary when the generated function has multiple outputs

## Files

- [`main.py`](./main.py): generates the low-level Rust crate and Python wrapper
- `foo/`: generated low-level Rust crate
- `foo_python/`: generated Python wrapper crate
- [`runner/main.py`](./runner/main.py): creates an isolated Python virtual
  environment, installs the wrapper crate, and calls the exported module

## Running the demo

From the repository root, after activating your virtual environment:

```bash
python demos/python_interface/main.py
python demos/python_interface/runner/main.py
```

The first command generates the low-level crate in `demos/python_interface/foo`
and the Python wrapper in `demos/python_interface/foo_python`.
The second command installs the wrapper into an isolated temporary virtual
environment and prints the result of calling `foo.call(...)` from Python.
