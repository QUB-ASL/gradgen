# SOC Demo

This demo shows how to generate Rust code for the half-squared distance to a
second-order cone and expose the generated kernels through a PyO3 Python
wrapper.

The cone is

$$
C_\alpha = \{x = (y, t) : \lVert y \rVert_2 \leq \alpha t\},
$$

and the demo uses `SquaredDistanceToSet.second_order_cone(...)` to generate
the half-squared distance and its gradient. The resulting Rust project is
wrapped as an importable Python module so the generated functions can be
called directly from Python.

## Files

- [`main.py`](./main.py): builds the SOC distance demo and prints the wrapper
  outputs
- `soc_kernel/`: generated low-level Rust crate
- `soc_kernel_python/`: generated PyO3 wrapper crate

## Running the demo

From the repository root, after activating your virtual environment:

```bash
python demos/soc/main.py
```

This will generate the Rust crate and the wrapper crate, install the wrapper
into the active Python environment, and print the results of calling the
generated functions.
