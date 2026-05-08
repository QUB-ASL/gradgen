# Single Shooting Benchmark

This benchmark compares the code size of Gradgen and CasADi for the same
single-shooting optimal-control problem across horizons `N = 10, 20, ..., 100`.

The model is a bicycle-style kinematic system with:

- 4 state variables
- 2 control inputs
- quadratic tracking cost
- quadratic control regularization

For each horizon, the script generates:

- Gradgen Rust code for the total cost and its gradient with respect to the
  packed control sequence
- CasADi C code for the same two functions

The benchmark prints the number of non-empty lines in each generated source
file so you can compare code size directly.

## Run

From the repository root, after activating your virtual environment:

```bash
python demos/benchmarks/single_shooting/main.py
```
