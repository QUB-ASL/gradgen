# Single Shooting Benchmark

This benchmark compares Gradgen and CasADi for the same single-shooting
optimal-control problem across horizons `N = 10, 20, ..., 100`.

You can choose the Gradgen scalar type with `--gradgen-scalar-type`
(`f64` by default, or `f32` for a smaller floating-point build).

You can also control the number of repeated benchmark calls with
`--num-runs` (`1000` by default).

Pass `--flatten true` to benchmark the flattened Gradgen path built from
`to_function()` instead of the staged `SingleShootingProblem` lowering.

The model is a bicycle-style kinematic system with:

- 4 state variables
- 2 control inputs
- quadratic tracking cost
- quadratic control regularization

For each horizon, the script generates and runs native code for:

- a Gradgen Rust crate that evaluates the total cost and its gradient in
  one joint call with respect to the packed control sequence
- a small C runner that calls the generated CasADi joint function directly

The benchmark prints the number of non-empty lines in each generated source
file, and it reports the average runtime of one joint cost-plus-gradient call
after dividing the total time for all repeated runs by `--num-runs`.

The generated runners live in `runners/`.

## Run

From the repository root, after activating your virtual environment:

```bash
python demos/benchmarks/single_shooting/main.py
```
